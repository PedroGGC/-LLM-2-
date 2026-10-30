"""Optional, bounded representation and gradient diagnostics outside training and benchmark timing.

Collects per-scale, per-iteration activation magnitudes (RMS) and optional eval-mode LM gradients
without mutating weights, parameter `.grad` buffers, module training modes, or RNG states.
"""

import torch

from ..objectives import CausalLanguageModeling
from ..runtime import attention_context, autocast_context, capture_rng, restore_rng


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt())


def _l2_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().float()))


def _safe_ratio(numerator: float | None, denominator: float | None, eps: float = 1e-12) -> float | None:
    if numerator is None or denominator is None:
        return None
    return float(numerator / (denominator + eps))


def linear_cka(x, y):
    """Centered linear CKA; undefined for a constant representation."""
    if x.shape[0] != y.shape[0]:
        return None  # Unequal native resolutions have no aligned sample correspondence.
    x, y = x.double() - x.double().mean(0), y.double() - y.double().mean(0)
    a, b = x @ x.T, y @ y.T
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((a * b).sum() / denominator) if denominator > 0 else None


def representation_stats(value):
    x = value.float()
    centered = x - x.mean(0)
    eigenvalues = torch.linalg.svdvals(centered.double()).square()
    total = eigenvalues.sum()
    if total > 0:
        probabilities = eigenvalues / total
        positive = probabilities[probabilities > 0]
        effective_rank = float((-(positive * positive.log()).sum()).exp())
        participation = float(total.square() / eigenvalues.square().sum())
    else:
        effective_rank = participation = 0.0
    return {
        "rms": float(x.square().mean().sqrt()),
        "mean_feature_std": float(x.std(0, unbiased=False).mean()),
        "effective_rank_entropy": effective_rank,
        "participation_rank": participation,
        "samples": x.shape[0],
        "dimension": x.shape[1],
    }


def _default_targets(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.shape[1] > 1:
        return torch.cat((input_ids[:, 1:], input_ids[:, :1]), dim=1)
    return input_ids.clone()


def _lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    token_losses = torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        reduction="none",
        ignore_index=-100,
    ).view_as(targets)
    mask = targets != -100
    count = int(mask.sum())
    return token_losses.sum() / count if count else logits.sum() * 0.0


def _grad_summary(
    tensor: torch.Tensor | None,
    *,
    valid_slice=None,
    null_reason: str | None = None,
    scope: str = "per_call",
) -> dict:
    if tensor is None:
        return {
            "grad_rms": None,
            "grad_l2_norm": None,
            "scope": None,
            "null_reason": null_reason or "tensor not recorded",
        }
    grad = getattr(tensor, "grad", None)
    if grad is None:
        return {
            "grad_rms": None,
            "grad_l2_norm": None,
            "scope": None,
            "null_reason": null_reason or "tensor does not receive gradient in this pass",
        }
    g = grad[valid_slice] if valid_slice is not None else grad
    return {
        "grad_rms": _rms(g),
        "grad_l2_norm": _l2_norm(g),
        "scope": scope,
        "null_reason": None,
    }


def _collect_fractal_telemetry(model, input_ids, targets, *, precision: str, deterministic: bool, include_gradients: bool):
    """Run an instrumented eval-mode pass on FractalLatentModel to collect per-scale/iteration RMS and gradients."""
    research = model.research
    scale_exec = research.scale_execution
    steps = model.active_iterations()
    B, T = input_ids.shape
    s1_len = (T + 3) // 4 if scale_exec.startswith("compressed") else T
    s2_len = (T + 15) // 16 if scale_exec.startswith("compressed") else T
    scale_names = [f"s{i}" for i in range(len(research.scales))]
    scale_lengths = [T, s1_len, s2_len] if scale_exec.startswith("compressed") else [T] * len(research.scales)

    # Track calls inside SharedFractalCore during each FractalRecursiveBlock iteration.
    current_iter = {"index": 0}
    in_block = {"active": False}
    preloop_cached_projections = []
    iter_records = {}

    def _ensure_iter(idx: int):
        if idx not in iter_records:
            iter_records[idx] = {
                "core_calls": [],
                "states_in": None,
                "post_states": None,
            }
        return iter_records[idx]

    hooks = []

    def block_pre_hook(module, args):
        in_block["active"] = True
        step_idx = args[2] if len(args) > 2 and isinstance(args[2], int) else current_iter["index"]
        current_iter["index"] = step_idx
        rec = _ensure_iter(step_idx)
        states_in = args[0]
        if include_gradients:
            for st in states_in:
                if isinstance(st, torch.Tensor) and st.requires_grad:
                    st.retain_grad()
        rec["states_in"] = states_in

    def block_post_hook(module, args, output):
        step_idx = current_iter["index"]
        rec = _ensure_iter(step_idx)
        if include_gradients:
            for st in output:
                if isinstance(st, torch.Tensor) and st.requires_grad:
                    st.retain_grad()
        rec["post_states"] = output
        current_iter["index"] = step_idx + 1
        in_block["active"] = False

    hooks.append(model.block.register_forward_pre_hook(block_pre_hook))
    hooks.append(model.block.register_forward_hook(block_post_hook))

    for core_idx, core in enumerate(model.block.cores):
        def make_core_hooks(c_idx, core_mod):
            call_state = {}

            def core_pre(mod, a, kwargs):
                call_state.clear()
                call_state["context_is_projected"] = bool(kwargs.get("context_is_projected", False))

            def sn_hook(mod, a, out):
                if include_gradients and isinstance(out, torch.Tensor) and out.requires_grad:
                    out.retain_grad()
                call_state["state_norm_out"] = out

            def cp_hook(mod, a, out):
                if include_gradients and isinstance(out, torch.Tensor) and out.requires_grad:
                    out.retain_grad()
                if in_block["active"]:
                    call_state["ctx_proj_out"] = out
                else:
                    preloop_cached_projections.append(out)

            def b0_pre(mod, a):
                tin = a[0]
                if include_gradients and isinstance(tin, torch.Tensor) and tin.requires_grad:
                    tin.retain_grad()
                call_state["core_input_sum"] = tin

            def attn_norm_h(mod, a, out):
                call_state["attn_norm_out"] = out

            def ffn_norm_h(mod, a, out):
                call_state["ffn_norm_out"] = out

            def b_last_post(mod, a, out):
                if include_gradients and isinstance(out, torch.Tensor) and out.requires_grad:
                    out.retain_grad()
                call_state["core_branch_out"] = out

            def core_post(mod, a, out):
                state_arg, ctx_arg = a[0], a[1]
                is_proj = bool(call_state.get("context_is_projected", False))
                ctx_proj_used = ctx_arg if is_proj else call_state.get("ctx_proj_out")
                if include_gradients:
                    for t_ref in (state_arg, ctx_arg, ctx_proj_used, out):
                        if isinstance(t_ref, torch.Tensor) and t_ref.requires_grad:
                            t_ref.retain_grad()
                rec = _ensure_iter(current_iter["index"])
                rec["core_calls"].append({
                    "core_index": c_idx,
                    "context_is_projected": is_proj,
                    "state_in": state_arg,
                    "ctx_arg": ctx_arg,
                    "state_norm_out": call_state.get("state_norm_out"),
                    "ctx_proj_out": ctx_proj_used,
                    "core_input_sum": call_state.get("core_input_sum"),
                    "attn_norm_out": call_state.get("attn_norm_out"),
                    "ffn_norm_out": call_state.get("ffn_norm_out"),
                    "core_branch_out": call_state.get("core_branch_out"),
                    "core_output": out,
                })
                call_state.clear()

            return [
                core_mod.register_forward_pre_hook(core_pre, with_kwargs=True),
                core_mod.state_norm.register_forward_hook(sn_hook),
                core_mod.context_projection.register_forward_hook(cp_hook),
                core_mod.blocks[0].register_forward_pre_hook(b0_pre),
                core_mod.blocks[0].attn_norm.register_forward_hook(attn_norm_h),
                core_mod.blocks[0].ffn_norm.register_forward_hook(ffn_norm_h),
                core_mod.blocks[-1].register_forward_hook(b_last_post),
                core_mod.register_forward_hook(core_post),
            ]

        hooks.extend(make_core_hooks(core_idx, core))

    try:
        with torch.no_grad():
            if scale_exec.startswith("compressed"):
                raw_c0 = model.embedding(input_ids)
                raw_c1 = model.block.comp4.compress(raw_c0)
                raw_c2 = model.block.comp16.compress(raw_c0)
                raw_contexts = (raw_c0, raw_c1, raw_c2)
                if model.block.conditioning is not None:
                    if research.fused_film:
                        cond_contexts = model.block.conditioning.forward_scales(raw_c0, raw_c1, raw_c2)
                    else:
                        cond_contexts = tuple(model.block.conditioning(c, i) for i, c in enumerate(raw_contexts))
                else:
                    cond_contexts = raw_contexts
                proj_contexts_static = tuple(model.block.cores[0].context_projection(c) for c in cond_contexts)
            else:
                raw_contexts = model.encoder(model.embedding(input_ids))
                if model.block.conditioning is not None:
                    cond_contexts = tuple(model.block.conditioning(c, i) for i, c in enumerate(raw_contexts))
                else:
                    cond_contexts = raw_contexts
                proj_contexts_static = tuple(
                    model.block.cores[0 if research.share_scales else i].context_projection(c)
                    for i, c in enumerate(cond_contexts)
                )

        grad_ctx = torch.enable_grad() if include_gradients else torch.no_grad()
        with grad_ctx, attention_context(deterministic), autocast_context(input_ids.device, precision):
            preloop_cached_projections.clear()
            if include_gradients:
                for p in model.parameters():
                    p.grad = None
            output = model(input_ids, return_logits=True)
            loss_val = None
            if include_gradients:
                loss_tensor = _lm_loss(output.logits, targets)
                loss_val = float(loss_tensor.detach().cpu())
                loss_tensor.backward()

        # Organize per-iteration, per-scale telemetry.
        telemetry_iterations = []
        for step_idx in range(steps):
            rec = iter_records.get(step_idx, {"core_calls": [], "states_in": (), "post_states": ()})
            calls = rec["core_calls"]
            states_in = rec["states_in"] or ()
            post_states = rec["post_states"] or ()
            skip_fine = (research.v_cycle and step_idx > 0) or (research.inv_vcycle and step_idx == 0)

            scale_entries = {}
            padding_telemetry = {}

            for s_idx, s_name in enumerate(scale_names):
                valid_len = scale_lengths[s_idx]
                s_in = states_in[s_idx] if s_idx < len(states_in) else None
                s_post = post_states[s_idx] if s_idx < len(post_states) else None
                raw_c = raw_contexts[s_idx]
                cond_c = cond_contexts[s_idx]

                # Locate corresponding core call and slice for this scale.
                call = None
                batch_slice = slice(None)
                time_slice = slice(0, valid_len)
                has_padding = False
                pad_slice = None

                if scale_exec == "compressed_grouped":
                    if s_idx == 0:
                        if not skip_fine and len(calls) >= 1:
                            call = calls[0]
                    else:
                        macro_call_idx = 0 if skip_fine else 1
                        if len(calls) > macro_call_idx:
                            call = calls[macro_call_idx]
                            batch_slice = slice(0, B) if s_idx == 1 else slice(B, 2 * B)
                            if s_idx == 2 and s1_len > s2_len:
                                has_padding = True
                                pad_slice = (slice(B, 2 * B), slice(s2_len, s1_len), slice(None))
                elif scale_exec in ("compressed", "compressed_native"):
                    if s_idx == 0:
                        if not skip_fine and len(calls) >= 1:
                            call = calls[0]
                    else:
                        offset = 0 if skip_fine else 1
                        target_call = offset + (s_idx - 1)
                        if len(calls) > target_call:
                            call = calls[target_call]
                elif scale_exec == "batched":
                    if len(calls) >= 1:
                        call = calls[0]
                        batch_slice = slice(s_idx * B, (s_idx + 1) * B)
                else:
                    if len(calls) > s_idx:
                        call = calls[s_idx]

                valid_sel = (batch_slice, time_slice, slice(None))

                if call is not None:
                    sn_out = call["state_norm_out"][valid_sel]
                    cp_out_tensor = call["ctx_proj_out"] if call["ctx_proj_out"] is not None else proj_contexts_static[s_idx]
                    cp_out = cp_out_tensor[valid_sel] if cp_out_tensor.shape[0] > B else cp_out_tensor[:, time_slice, :]
                    c_sum = call["core_input_sum"][valid_sel]
                    an_out = call["attn_norm_out"][valid_sel]
                    fn_out = call["ffn_norm_out"][valid_sel]
                    c_out = call["core_output"][valid_sel]
                    branch_delta = (c_out.detach().float() - s_in.detach().float())

                    state_rms = _rms(s_in)
                    sn_rms = _rms(sn_out)
                    raw_c_rms = _rms(raw_c)
                    cond_c_rms = _rms(cond_c)
                    proj_c_rms = _rms(cp_out)
                    sum_rms = _rms(c_sum)
                    an_rms = _rms(an_out)
                    fn_rms = _rms(fn_out)
                    branch_rms = _rms(branch_delta)
                    core_out_rms = _rms(c_out)
                    post_rms = _rms(s_post)

                    if has_padding and pad_slice is not None:
                        padding_telemetry[s_name] = {
                            "valid_positions": valid_len,
                            "padded_positions": s1_len - valid_len,
                            "padded_fraction": float((s1_len - valid_len) / s1_len),
                            "padded_state_norm_rms": _rms(call["state_norm_out"][pad_slice]),
                            "padded_core_input_sum_rms": _rms(call["core_input_sum"][pad_slice]),
                            "padded_core_output_rms": _rms(call["core_output"][pad_slice]),
                        }
                else:
                    # Fine scale skipped in V-Cycle / Inv-V-Cycle.
                    state_rms = _rms(s_in) if s_in is not None else 0.0
                    sn_rms = None
                    raw_c_rms = _rms(raw_c)
                    cond_c_rms = _rms(cond_c)
                    proj_c_rms = _rms(proj_contexts_static[s_idx])
                    sum_rms = None
                    an_rms = None
                    fn_rms = None
                    branch_rms = 0.0
                    core_out_rms = state_rms
                    post_rms = _rms(s_post) if s_post is not None else 0.0

                ratios = {
                    "state_norm_over_projected_context": _safe_ratio(sn_rms, proj_c_rms),
                    "projected_context_over_core_input_sum": _safe_ratio(proj_c_rms, sum_rms),
                    "state_norm_over_core_input_sum": _safe_ratio(sn_rms, sum_rms),
                    "core_branch_delta_over_state_in": _safe_ratio(branch_rms, state_rms) if step_idx > 0 else None,
                    "core_branch_delta_over_core_output": _safe_ratio(branch_rms, core_out_rms),
                    "post_interaction_delta_over_core_output": (
                        _safe_ratio(_rms(s_post.detach().float() - c_out.detach().float()), core_out_rms)
                        if (call is not None and s_post is not None)
                        else None
                    ),
                }

                grads = None
                if include_gradients:
                    is_init_zero = step_idx == 0
                    is_unused_final_macro = (step_idx == steps - 1) and (s_idx > 0)
                    init_reason = (
                        "initial state at iteration 1 is a constant zero tensor (requires_grad=False)"
                        if is_init_zero
                        else None
                    )
                    final_macro_reason = (
                        "macro scale output at final iteration is not consumed by lm_head"
                        if is_unused_final_macro
                        else None
                    )
                    if call is not None:
                        cp_ref = call["ctx_proj_out"] if call["ctx_proj_out"] is not None else call["ctx_arg"]
                        cp_slice = valid_sel if cp_ref.shape[0] > B else (slice(None), time_slice, slice(None))
                        is_cached_mode = bool(getattr(research, "cache_context_projection", False))
                        is_reused_across_iters = steps > 1 and (
                            is_cached_mode or any(cp_ref is cached_t for cached_t in preloop_cached_projections)
                        )
                        cp_scope = "accumulated_across_iterations" if is_reused_across_iters else "per_call"
                        call_input_grad = (
                            _grad_summary(call["core_input_sum"], valid_slice=valid_sel, scope="per_call")
                            if is_reused_across_iters
                            else _grad_summary(cp_ref, valid_slice=cp_slice, scope="per_call")
                        )
                        if is_cached_mode and len(preloop_cached_projections) > s_idx:
                            cached_accum_grad = _grad_summary(
                                preloop_cached_projections[s_idx],
                                valid_slice=(slice(None), time_slice, slice(None)),
                                scope="accumulated_across_iterations" if steps > 1 else "per_call",
                            )
                            proj_ctx_grad = cached_accum_grad
                        else:
                            cached_accum_grad = _grad_summary(
                                None, null_reason="cache_context_projection is disabled"
                            )
                            proj_ctx_grad = _grad_summary(
                                cp_ref,
                                valid_slice=cp_slice,
                                scope=cp_scope,
                            )
                        grads = {
                            "state_in": _grad_summary(s_in, null_reason=init_reason, scope="per_call"),
                            "state_norm_out": _grad_summary(
                                call["state_norm_out"], valid_slice=valid_sel, null_reason=init_reason, scope="per_call"
                            ),
                            "projected_context": proj_ctx_grad,
                            "projected_context_call_input": call_input_grad,
                            "cached_projected_context_accumulated": cached_accum_grad,
                            "core_input_sum": _grad_summary(
                                call["core_input_sum"], valid_slice=valid_sel, scope="per_call"
                            ),
                            "core_output": _grad_summary(
                                call["core_output"], valid_slice=valid_sel, scope="per_call"
                            ),
                            "post_interaction_state": _grad_summary(
                                s_post, null_reason=final_macro_reason, scope="per_call"
                            ),
                        }
                    else:
                        grads = {
                            "state_in": _grad_summary(s_in, null_reason=init_reason or "core skipped on this iteration"),
                            "state_norm_out": _grad_summary(None, null_reason="core skipped on this iteration"),
                            "projected_context": _grad_summary(None, null_reason="core skipped on this iteration"),
                            "projected_context_call_input": _grad_summary(
                                None, null_reason="core skipped on this iteration"
                            ),
                            "cached_projected_context_accumulated": _grad_summary(
                                None, null_reason="core skipped on this iteration"
                            ),
                            "core_input_sum": _grad_summary(None, null_reason="core skipped on this iteration"),
                            "core_output": _grad_summary(None, null_reason="core skipped on this iteration"),
                            "post_interaction_state": _grad_summary(s_post, null_reason=final_macro_reason),
                        }

                scale_entries[s_name] = {
                    "valid_length": valid_len,
                    "activations_rms": {
                        "state_in": state_rms,
                        "state_norm": sn_rms,
                        "raw_context": raw_c_rms,
                        "conditioned_context": cond_c_rms,
                        "projected_context": proj_c_rms,
                        "core_input_sum": sum_rms,
                        "attn_norm_out": an_rms,
                        "ffn_norm_out": fn_rms,
                        "core_branch_delta": branch_rms,
                        "core_output": core_out_rms,
                        "post_interaction_state": post_rms,
                    },
                    "ratios": ratios,
                    "gradients": grads,
                }

            telemetry_iterations.append({
                "iteration": step_idx + 1,
                "scales": scale_entries,
                "padding_telemetry": padding_telemetry if padding_telemetry else None,
            })

        parameter_gradients = None
        if include_gradients:
            core0 = model.block.cores[0]
            param_scope = "accumulated_across_all_scales_and_iterations_in_single_eval_pass"
            parameter_gradients = {
                "scope": param_scope,
                "eval_lm_loss": loss_val,
                "embedding_weight": _grad_summary(model.embedding.weight, scope=param_scope),
                "core_state_norm_weight": _grad_summary(core0.state_norm.weight, scope=param_scope),
                "core_context_projection_weight": _grad_summary(core0.context_projection.weight, scope=param_scope),
                "core_gate": _grad_summary(core0.gate, scope=param_scope),
                "core_attn_qkv_weight": _grad_summary(core0.blocks[0].attn.qkv.weight, scope=param_scope),
                "core_attn_proj_weight": _grad_summary(core0.blocks[0].attn.proj.weight, scope=param_scope),
                "core_ffn_down_weight": _grad_summary(core0.blocks[0].ffn.down.weight, scope=param_scope),
                "interaction_mixing": _grad_summary(model.block.interaction.mixing, scope=param_scope),
                "interaction_projection_weight": _grad_summary(
                    model.block.interaction.projection.weight, scope=param_scope
                ),
                "interaction_gate": _grad_summary(model.block.interaction.gate, scope=param_scope),
                "final_norm_weight": _grad_summary(model.norm.weight, scope=param_scope),
            }
            if model.block.conditioning is not None:
                parameter_gradients["conditioning_film"] = _grad_summary(
                    model.block.conditioning.film, scope=param_scope
                )
            if model.block.ncm is not None:
                parameter_gradients["ncm_memory"] = _grad_summary(model.block.ncm.memory, scope=param_scope)

        return output.logits.detach().clone(), {
            "ratio_definitions": {
                "state_norm_over_projected_context": "RMS(state_norm(state_in)) / (RMS(W_ctx(context)) + 1e-12)",
                "projected_context_over_core_input_sum": "RMS(W_ctx(context)) / (RMS(state_norm(state_in) + W_ctx(context)) + 1e-12)",
                "state_norm_over_core_input_sum": "RMS(state_norm(state_in)) / (RMS(state_norm(state_in) + W_ctx(context)) + 1e-12)",
                "core_branch_delta_over_state_in": "RMS(sigmoid(gate) * blocks(core_input_sum)) / (RMS(state_in) + 1e-12) for iteration >= 2 (null at iteration 1 where state_in == 0)",
                "core_branch_delta_over_core_output": "RMS(sigmoid(gate) * blocks(core_input_sum)) / (RMS(core_output) + 1e-12)",
                "post_interaction_delta_over_core_output": "RMS(post_interaction_state - core_output) / (RMS(core_output) + 1e-12)",
            },
            "iterations": telemetry_iterations,
            "parameter_gradients": parameter_gradients,
        }
    finally:
        for h in hooks:
            h.remove()


def _collect_lrm_telemetry(model, input_ids, targets, *, precision: str, deterministic: bool, include_gradients: bool):
    """Run an instrumented eval-mode pass on LatentRecursiveModel to collect per-iteration RMS and gradients."""
    research = model.research
    steps = model.active_iterations()
    current_iter = {"index": 0}
    in_core = {"active": False}
    preloop_cached_projections = []
    iter_records = []
    hooks = []

    cores = [model.core] if model.shared else list(model.cores)
    for c_idx, core_mod in enumerate(cores):
        def make_hooks(core_index, mod):
            call_state = {}

            def core_pre(m, a, kwargs):
                call_state.clear()
                in_core["active"] = True
                call_state["context_is_projected"] = bool(kwargs.get("context_is_projected", False))

            def sn_hook(m, a, out):
                if include_gradients and isinstance(out, torch.Tensor) and out.requires_grad:
                    out.retain_grad()
                call_state["state_norm_out"] = out

            def cp_hook(m, a, out):
                if include_gradients and isinstance(out, torch.Tensor) and out.requires_grad:
                    out.retain_grad()
                if in_core["active"]:
                    call_state["ctx_proj_out"] = out
                else:
                    preloop_cached_projections.append(out)

            def b0_pre(m, a):
                tin = a[0]
                if include_gradients and isinstance(tin, torch.Tensor) and tin.requires_grad:
                    tin.retain_grad()
                call_state["core_input_sum"] = tin

            def attn_norm_h(m, a, out):
                call_state["attn_norm_out"] = out

            def ffn_norm_h(m, a, out):
                call_state["ffn_norm_out"] = out

            def b_last_post(m, a, out):
                if include_gradients and isinstance(out, torch.Tensor) and out.requires_grad:
                    out.retain_grad()
                call_state["core_branch_out"] = out

            def core_post(m, a, out):
                s_in, c_in = a[0], a[1]
                is_proj = bool(call_state.get("context_is_projected", False))
                ctx_proj_used = c_in if is_proj else call_state.get("ctx_proj_out")
                if include_gradients:
                    for t_ref in (s_in, c_in, ctx_proj_used, out):
                        if isinstance(t_ref, torch.Tensor) and t_ref.requires_grad:
                            t_ref.retain_grad()
                iter_records.append({
                    "iteration": current_iter["index"] + 1,
                    "context_is_projected": is_proj,
                    "state_in": s_in,
                    "ctx_arg": c_in,
                    "state_norm_out": call_state.get("state_norm_out"),
                    "ctx_proj_out": ctx_proj_used,
                    "core_input_sum": call_state.get("core_input_sum"),
                    "attn_norm_out": call_state.get("attn_norm_out"),
                    "ffn_norm_out": call_state.get("ffn_norm_out"),
                    "core_branch_out": call_state.get("core_branch_out"),
                    "core_output": out,
                })
                current_iter["index"] += 1
                in_core["active"] = False
                call_state.clear()

            return [
                mod.register_forward_pre_hook(core_pre, with_kwargs=True),
                mod.state_norm.register_forward_hook(sn_hook),
                mod.context_projection.register_forward_hook(cp_hook),
                mod.blocks[0].register_forward_pre_hook(b0_pre),
                mod.blocks[0].attn_norm.register_forward_hook(attn_norm_h),
                mod.blocks[0].ffn_norm.register_forward_hook(ffn_norm_h),
                mod.blocks[-1].register_forward_hook(b_last_post),
                mod.register_forward_hook(core_post),
            ]

        hooks.extend(make_hooks(c_idx, core_mod))

    try:
        with torch.no_grad():
            raw_c = model.embedding(input_ids)

        grad_ctx = torch.enable_grad() if include_gradients else torch.no_grad()
        with grad_ctx, attention_context(deterministic), autocast_context(input_ids.device, precision):
            preloop_cached_projections.clear()
            if include_gradients:
                for p in model.parameters():
                    p.grad = None
            output = model(input_ids, return_logits=True)
            loss_val = None
            if include_gradients:
                loss_tensor = _lm_loss(output.logits, targets)
                loss_val = float(loss_tensor.detach().cpu())
                loss_tensor.backward()

        telemetry_iterations = []
        for idx, rec in enumerate(iter_records):
            s_in = rec["state_in"]
            sn_out = rec["state_norm_out"]
            cp_out = rec["ctx_proj_out"]
            c_sum = rec["core_input_sum"]
            an_out = rec["attn_norm_out"]
            fn_out = rec["ffn_norm_out"]
            c_out = rec["core_output"]
            branch_delta = c_out.detach().float() - s_in.detach().float()

            state_rms = _rms(s_in)
            sn_rms = _rms(sn_out)
            raw_c_rms = _rms(raw_c)
            proj_c_rms = _rms(cp_out)
            sum_rms = _rms(c_sum)
            an_rms = _rms(an_out)
            fn_rms = _rms(fn_out)
            branch_rms = _rms(branch_delta)
            core_out_rms = _rms(c_out)

            init_reason = (
                "initial state at iteration 1 is a constant zero tensor (requires_grad=False)"
                if idx == 0
                else None
            )
            grads = None
            if include_gradients:
                is_reused_across_iters = (
                    steps > 1 and any(cp_out is cached_t for cached_t in preloop_cached_projections)
                )
                cp_scope = "accumulated_across_iterations" if is_reused_across_iters else "per_call"
                call_input_grad = (
                    _grad_summary(c_sum, scope="per_call")
                    if is_reused_across_iters
                    else _grad_summary(cp_out, scope="per_call")
                )
                if getattr(research, "cache_context_projection", False) and preloop_cached_projections:
                    cached_accum_grad = _grad_summary(
                        preloop_cached_projections[0],
                        scope="accumulated_across_iterations" if steps > 1 else "per_call",
                    )
                else:
                    cached_accum_grad = _grad_summary(
                        None, null_reason="cache_context_projection is disabled"
                    )
                grads = {
                    "state_in": _grad_summary(s_in, null_reason=init_reason, scope="per_call"),
                    "state_norm_out": _grad_summary(sn_out, null_reason=init_reason, scope="per_call"),
                    "projected_context": _grad_summary(cp_out, scope=cp_scope),
                    "projected_context_call_input": call_input_grad,
                    "cached_projected_context_accumulated": cached_accum_grad,
                    "core_input_sum": _grad_summary(c_sum, scope="per_call"),
                    "core_output": _grad_summary(c_out, scope="per_call"),
                    "post_interaction_state": _grad_summary(c_out, scope="per_call"),
                }

            telemetry_iterations.append({
                "iteration": idx + 1,
                "scales": {
                    "s0": {
                        "valid_length": int(input_ids.shape[1]),
                        "activations_rms": {
                            "state_in": state_rms,
                            "state_norm": sn_rms,
                            "raw_context": raw_c_rms,
                            "conditioned_context": raw_c_rms,
                            "projected_context": proj_c_rms,
                            "core_input_sum": sum_rms,
                            "attn_norm_out": an_rms,
                            "ffn_norm_out": fn_rms,
                            "core_branch_delta": branch_rms,
                            "core_output": core_out_rms,
                            "post_interaction_state": core_out_rms,
                        },
                        "ratios": {
                            "state_norm_over_projected_context": _safe_ratio(sn_rms, proj_c_rms),
                            "projected_context_over_core_input_sum": _safe_ratio(proj_c_rms, sum_rms),
                            "state_norm_over_core_input_sum": _safe_ratio(sn_rms, sum_rms),
                            "core_branch_delta_over_state_in": _safe_ratio(branch_rms, state_rms) if idx > 0 else None,
                            "core_branch_delta_over_core_output": _safe_ratio(branch_rms, core_out_rms),
                            "post_interaction_delta_over_core_output": 0.0,
                        },
                        "gradients": grads,
                    }
                },
                "padding_telemetry": None,
            })

        parameter_gradients = None
        if include_gradients:
            core0 = cores[0]
            param_scope = "accumulated_across_all_iterations_in_single_eval_pass"
            parameter_gradients = {
                "scope": param_scope,
                "eval_lm_loss": loss_val,
                "embedding_weight": _grad_summary(model.embedding.weight, scope=param_scope),
                "core_state_norm_weight": _grad_summary(core0.state_norm.weight, scope=param_scope),
                "core_context_projection_weight": _grad_summary(core0.context_projection.weight, scope=param_scope),
                "core_gate": _grad_summary(core0.gate, scope=param_scope),
                "final_norm_weight": _grad_summary(model.norm.weight, scope=param_scope),
            }

        return output.logits.detach().clone(), {
            "ratio_definitions": {
                "state_norm_over_projected_context": "RMS(state_norm(state_in)) / (RMS(W_ctx(context)) + 1e-12)",
                "projected_context_over_core_input_sum": "RMS(W_ctx(context)) / (RMS(state_norm(state_in) + W_ctx(context)) + 1e-12)",
                "state_norm_over_core_input_sum": "RMS(state_norm(state_in)) / (RMS(state_norm(state_in) + W_ctx(context)) + 1e-12)",
                "core_branch_delta_over_state_in": "RMS(sigmoid(gate) * blocks(core_input_sum)) / (RMS(state_in) + 1e-12) for iteration >= 2 (null at iteration 1 where state_in == 0)",
                "core_branch_delta_over_core_output": "RMS(sigmoid(gate) * blocks(core_input_sum)) / (RMS(core_output) + 1e-12)",
                "post_interaction_delta_over_core_output": "0.0 for single-scale LRM",
            },
            "iterations": telemetry_iterations,
            "parameter_gradients": parameter_gradients,
        }
    finally:
        for h in hooks:
            h.remove()


def _collect_selected_telemetry(model, input_ids, targets, *, precision, deterministic, include_gradients):
    records, tensors, handles = [], [], []
    count = 1 + len(model.block.factors)
    def capture(module, args, output):
        index = len(records)
        records.append({"iteration": index // count + 1,
                        "scale": (1, *model.block.factors)[index % count],
                        "output_rms": _rms(output), "activation_gradient_l2": None})
        tensors.append(output)
    handles.extend(core.register_forward_hook(capture) for core in model.block.cores)
    try:
        with torch.enable_grad(), attention_context(deterministic), autocast_context(input_ids.device, precision):
            logits = model(input_ids).logits
            parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            gradients = (torch.autograd.grad(_lm_loss(logits, targets),
                [*tensors, *(p for _, p in parameters)], allow_unused=True) if include_gradients else ())
        if include_gradients:
            for record, grad in zip(records, gradients[:len(tensors)]):
                record["activation_gradient_l2"] = None if grad is None else _l2_norm(grad)
        return logits.detach(), {"kind": "selected_native_scales", "core_calls": records,
            "parameter_gradients": ({name: None if grad is None else _l2_norm(grad)
                for (name, _), grad in zip(parameters, gradients[len(tensors):])} if include_gradients else None)}
    finally:
        for handle in handles:
            handle.remove()


def diagnose_model(
    model,
    input_ids,
    *,
    targets=None,
    batch_source="unspecified",
    include_gradients=True,
    max_samples=128,
    precision="fp32",
    deterministic=True,
):
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("diagnostics need at least two samples")
    modes = [(module, module.training) for module in model.modules()]
    saved_grads = {
        name: (None if param.grad is None else param.grad.detach().clone())
        for name, param in model.named_parameters()
    }
    rng = capture_rng()
    snapshots = []
    hook = None

    def capture(module, args, output):
        states = output if isinstance(output, tuple) else (output,)
        snapshots.append(
            tuple(state.detach().float().flatten(0, 1)[:max_samples].cpu().clone() for state in states)
        )

    # A fractal block returns all scales; a shared LRM core returns one state.
    core = getattr(model, "block", getattr(model, "core", None))
    if core is not None:
        hook = core.register_forward_hook(capture)
    try:
        model.eval()
        with torch.no_grad(), attention_context(deterministic), autocast_context(input_ids.device, precision):
            ref_output = model(input_ids, return_logits=True)
            ref_logits = ref_output.logits.detach().clone()
        if not snapshots:
            snapshots.append((ref_output.hidden_states.detach().float().flatten(0, 1)[:max_samples].cpu(),))
        final = snapshots[-1]
        compressed = getattr(getattr(model, "research", None), "scale_execution", "").startswith("compressed")
        if any(not torch.isfinite(value).all() for value in final):
            raise FloatingPointError("non-finite diagnostic states")
        history = []
        previous = None
        for iteration, values in enumerate(snapshots, 1):
            history.append({
                "iteration": iteration,
                "rms_by_scale": [float(x.square().mean().sqrt()) for x in values],
                "relative_update_by_scale": (
                    [
                        float(torch.linalg.vector_norm(x - old) / (torch.linalg.vector_norm(old) + 1e-8))
                        for x, old in zip(values, previous)
                    ]
                    if previous is not None
                    else None
                ),
            })
            previous = values

        # Collect Etapa 1 internal magnitude and gradient telemetry without mutating model state.
        hook.remove() if hook is not None else None
        hook = None
        diag_targets = targets if targets is not None else _default_targets(input_ids)
        target_source = batch_source if targets is not None else f"{batch_source}_shifted_input_fallback"

        stage1_telemetry = None
        if hasattr(model, "block"):
            collector = _collect_selected_telemetry if getattr(model.block, "reduced", False) else _collect_fractal_telemetry
            grad_logits, stage1_telemetry = collector(
                model,
                input_ids,
                diag_targets,
                precision=precision,
                deterministic=deterministic,
                include_gradients=include_gradients,
            )
            if not torch.allclose(ref_logits, grad_logits, rtol=1e-5, atol=1e-6):
                raise RuntimeError("diagnostic telemetry pass altered forward logits")
        elif hasattr(model, "core") or hasattr(model, "cores"):
            grad_logits, stage1_telemetry = _collect_lrm_telemetry(
                model,
                input_ids,
                diag_targets,
                precision=precision,
                deterministic=deterministic,
                include_gradients=include_gradients,
            )
            if not torch.allclose(ref_logits, grad_logits, rtol=1e-5, atol=1e-6):
                raise RuntimeError("diagnostic telemetry pass altered forward logits")

        result = {
            "batch_metadata": {
                "batch_source": target_source,
                "batch_size": int(input_ids.shape[0]),
                "seq_len": int(input_ids.shape[1]),
                "evaluated_tokens": int((diag_targets != -100).sum().item()),
                "eval_mode_dropout_disabled": True,
                "include_gradients": bool(include_gradients),
            },
            "scales": [representation_stats(value) for value in final],
            "linear_cka": [
                [linear_cka(x, y) if i == j or not compressed else None for j, y in enumerate(final)]
                for i, x in enumerate(final)
            ],
            "iterations": history,
            "stage1_telemetry": stage1_telemetry,
            "gates": {
                name: parameter.sigmoid().detach().cpu().tolist()
                for name, parameter in model.named_parameters()
                if name.endswith("gate")
            },
            "max_samples": max_samples,
            "device": str(input_ids.device),
            "precision": precision,
            "limitations": (
                "Bounded states and optional eval-mode LM gradients from one fixed validation batch; "
                "raw recurrent states for LRM/FLRM, final hidden for Transformer. "
                "Cross-scale CKA is null for compressed modes until temporal anchors are aligned; "
                "raw ranks with different sample counts are not directly comparable. "
                "Shared parameter gradients are accumulated across all scales and iterations in the single eval pass; "
                "per-scale/per-iteration attribution uses activation gradients. "
                "Initial state at iteration 1 is a constant zero tensor (requires_grad=False), so state_in and state_norm_out gradients at iteration 1 are null by design."
            ),
        }
        if hasattr(model, "block"):
            result["cross_scale_mixing"] = model.block.interaction.mixing.softmax(-1).detach().cpu().tolist()
            if model.block.conditioning is not None:
                film = model.block.conditioning.film.detach().float()
                result["film"] = {
                    "gamma_gain_min": float((1 + 0.1 * film[:, 0].tanh()).min()),
                    "gamma_gain_max": float((1 + 0.1 * film[:, 0].tanh()).max()),
                    "beta_rms": float((0.1 * film[:, 1]).square().mean().sqrt()),
                }
        if getattr(model, "paired_init_provenance", None) is not None:
            result["paired_init_provenance"] = model.paired_init_provenance
        return result
    finally:
        if hook is not None:
            hook.remove()
        for name, param in model.named_parameters():
            param.grad = saved_grads[name]
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
