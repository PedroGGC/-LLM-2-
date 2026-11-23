"""Rota 1 Architectural Distillation & Conversion (RADLADS / LoopUS style) from open CausalLM teachers into FractalDelta V2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable, Iterable

from safetensors.torch import save_model
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DistillConfig:
    kl_weight: float = 1.0
    ce_weight: float = 0.5
    hidden_mse_weight: float = 0.25
    exit_gate_weight: float = 0.10
    mtp_weight: float = 0.25
    temperature: float = 2.0
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 20
    max_steps: int = 200
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    precision: str = "fp32"  # fp32 | bf16
    surgery_transplant: bool = False
    freeze_ffn: bool = False
    freeze_embeddings: bool = False
    layer_mse_weight: float = 0.0

    def __post_init__(self):
        if self.temperature <= 0 or self.lr <= 0 or self.max_steps <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("temperature, lr, max_steps, and grad_accum_steps must be positive")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32|bf16")


def _get_teacher_embedding_weight(teacher: nn.Module) -> torch.Tensor | None:
    if hasattr(teacher, "get_input_embeddings"):
        emb = teacher.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
    if hasattr(teacher, "embedding") and hasattr(teacher.embedding, "weight"):
        return teacher.embedding.weight
    return None


@torch.no_grad()
def warmstart_student_embeddings(teacher: nn.Module, student: nn.Module) -> dict:
    """Warm-start student tied embedding/lm_head from teacher via direct copy or truncated SVD projection."""
    t_w = _get_teacher_embedding_weight(teacher)
    if t_w is None or not hasattr(student, "embedding"):
        return {"warmstarted": False, "reason": "missing_embedding"}
    s_w = student.embedding.weight
    t_vocab, t_dim = t_w.shape
    s_vocab, s_dim = s_w.shape
    if t_vocab != s_vocab:
        return {"warmstarted": False, "reason": f"vocab_mismatch:{t_vocab}!={s_vocab}"}
    src = t_w.detach().float()
    if t_dim == s_dim:
        s_w.copy_(src.to(device=s_w.device, dtype=s_w.dtype))
        return {"warmstarted": True, "mode": "exact_copy", "vocab": s_vocab, "d_model": s_dim}
    # Project teacher embeddings into student d_model preserving top singular directions + matched RMS norm.
    proj_src = src.to(device="cpu")
    if t_dim > s_dim:
        u, s_vals, _ = torch.svd_lowrank(proj_src, q=s_dim, niter=2)
        reduced = u * s_vals.unsqueeze(0)
    else:
        reduced = F.pad(proj_src, (0, s_dim - t_dim))
    target_rms = proj_src.square().mean().sqrt().clamp_min(1e-4)
    cur_rms = reduced.square().mean().sqrt().clamp_min(1e-6)
    reduced = reduced * (target_rms / cur_rms)
    s_w.copy_(reduced.to(device=s_w.device, dtype=s_w.dtype))
    return {"warmstarted": True, "mode": "svd_lowrank", "vocab": s_vocab, "teacher_dim": t_dim, "student_dim": s_dim}


def _extract_teacher_layers_and_norm(teacher: nn.Module) -> tuple[list[nn.Module], nn.Module | None]:
    if hasattr(teacher, "model") and hasattr(teacher.model, "layers"):
        final_norm = getattr(teacher.model, "norm", None) or getattr(teacher.model, "final_layernorm", None)
        return list(teacher.model.layers), final_norm
    if hasattr(teacher, "layers"):
        return list(teacher.layers), getattr(teacher, "norm", None)
    return [], None


def _copy_matching_or_slice(dst: torch.Tensor, src: torch.Tensor) -> None:
    s = src.detach().float()
    if dst.shape == s.shape:
        dst.copy_(s.to(device=dst.device, dtype=dst.dtype))
        return
    # Safe top-left slice / zero-pad fallback when dimensions differ
    out = torch.zeros_like(dst, dtype=torch.float32, device="cpu")
    s_cpu = s.to(device="cpu")
    if dst.ndim == 1:
        m = min(dst.shape[0], s_cpu.shape[0])
        out[:m] = s_cpu[:m]
    else:
        m0 = min(dst.shape[0], s_cpu.shape[0])
        m1 = min(dst.shape[1], s_cpu.shape[1])
        out[:m0, :m1] = s_cpu[:m0, :m1]
    dst.copy_(out.to(device=dst.device, dtype=dst.dtype))


def _expand_gqa_kv(weight: torch.Tensor, target_out: int) -> torch.Tensor:
    w = weight.detach().float()
    if w.shape[0] == target_out:
        return w
    if w.shape[0] > 0 and target_out % w.shape[0] == 0:
        rep = target_out // w.shape[0]
        return w.repeat_interleave(rep, dim=0)
    return w


@torch.no_grad()
def transplant_teacher_weights(
    teacher: nn.Module,
    student: nn.Module,
    *,
    quiet_init_gates: bool = True,
) -> dict:
    """RADLADS / Jet-Nemotron Attention-Surgery: transplant pretrained Embeddings, RMSNorms, SwiGLU FFNs, and QKV/Out projections."""
    emb_info = warmstart_student_embeddings(teacher, student)
    t_layers, t_final_norm = _extract_teacher_layers_and_norm(teacher)
    if not t_layers or not hasattr(student, "cells"):
        return {"transplanted": False, "embedding": emb_info, "reason": "no_compatible_layers"}

    if t_final_norm is not None and hasattr(t_final_norm, "weight") and hasattr(student, "norm"):
        _copy_matching_or_slice(student.norm.weight, t_final_norm.weight)

    n_t = len(t_layers)
    n_c = len(student.cells)
    d_s = student.config.d_model
    ffn_copied = 0
    qkv_copied = 0
    norms_copied = 0

    for c_idx, cell in enumerate(student.cells):
        t_idx = 0 if n_c == 1 else round(c_idx * (n_t - 1) / max(n_c - 1, 1))
        if n_c == 1:
            t_idx = n_t // 2
        t_layer = t_layers[t_idx]

        # 1. Pre-attention and pre-FFN RMSNorm weights
        anorm = getattr(t_layer, "input_layernorm", None) or getattr(t_layer, "attn_norm", None)
        fnorm = getattr(t_layer, "post_attention_layernorm", None) or getattr(t_layer, "ffn_norm", None)
        if anorm is not None and hasattr(anorm, "weight"):
            _copy_matching_or_slice(cell.attn_norm.weight, anorm.weight)
            norms_copied += 1
        if fnorm is not None and hasattr(fnorm, "weight"):
            _copy_matching_or_slice(cell.ffn_norm.weight, fnorm.weight)
            norms_copied += 1

        # 2. SwiGLU FFN weights (gate, up, down)
        mlp = getattr(t_layer, "mlp", None) or getattr(t_layer, "ffn", None)
        if mlp is not None:
            g_proj = getattr(mlp, "gate_proj", None) or getattr(mlp, "gate", None)
            u_proj = getattr(mlp, "up_proj", None) or getattr(mlp, "up", None)
            d_proj = getattr(mlp, "down_proj", None) or getattr(mlp, "down", None)
            if g_proj is not None and u_proj is not None and d_proj is not None:
                _copy_matching_or_slice(cell.ffn.gate.weight, g_proj.weight)
                _copy_matching_or_slice(cell.ffn.up.weight, u_proj.weight)
                _copy_matching_or_slice(cell.ffn.down.weight, d_proj.weight)
                ffn_copied += 1

        # 3. QKV & Out projection initialization from teacher self-attention (HF q/k/v/o_proj or native llm2 attn.qkv/proj)
        attn = getattr(t_layer, "self_attn", None) or getattr(t_layer, "attn", None) or getattr(t_layer, "attention", None)
        if attn is not None and hasattr(cell.qkv, "weight") and hasattr(cell.out_proj, "weight"):
            if hasattr(attn, "qkv") and hasattr(attn, "proj"):
                _copy_matching_or_slice(cell.qkv.weight, attn.qkv.weight)
                _copy_matching_or_slice(cell.out_proj.weight, attn.proj.weight)
                qkv_copied += 1
            else:
                q_p = getattr(attn, "q_proj", None) or getattr(attn, "q", None)
                k_p = getattr(attn, "k_proj", None) or getattr(attn, "k", None)
                v_p = getattr(attn, "v_proj", None) or getattr(attn, "v", None)
                o_p = getattr(attn, "o_proj", None) or getattr(attn, "out", None)
                if q_p is not None and k_p is not None and v_p is not None and o_p is not None:
                    qw = q_p.weight.detach().float()
                    kw = _expand_gqa_kv(k_p.weight, qw.shape[0])
                    vw = _expand_gqa_kv(v_p.weight, qw.shape[0])
                    if qw.shape[0] == d_s and qw.shape[1] == d_s and kw.shape == qw.shape and vw.shape == qw.shape:
                        qkv_cat = torch.cat((qw, kw, vw), dim=0)
                        _copy_matching_or_slice(cell.qkv.weight, qkv_cat)
                    _copy_matching_or_slice(cell.out_proj.weight, o_p.weight)
                    qkv_copied += 1

        if quiet_init_gates:
            # Start depthwise conv as near-identity residual so transplanted QKV sees clean normed states at Step 0.
            nn.init.normal_(cell.conv.weight, std=1e-4)
            nn.init.zeros_(cell.conv.bias)

    if quiet_init_gates and hasattr(student, "assoc_readout"):
        nn.init.normal_(student.assoc_readout.weight, std=1e-4)
    if quiet_init_gates and hasattr(student, "adapters"):
        for ad in student.adapters:
            if getattr(ad, "anchor_gate", None) is not None:
                ad.anchor_gate.fill_(-8.0)
    if quiet_init_gates and hasattr(student, "hybrid_windows"):
        for hw in student.hybrid_windows:
            nn.init.normal_(hw.out_proj.weight, std=1e-4)

    return {
        "transplanted": True,
        "embedding": emb_info,
        "teacher_layers": n_t,
        "student_cells": n_c,
        "ffn_cells_copied": ffn_copied,
        "qkv_cells_copied": qkv_copied,
        "norms_copied": norms_copied,
    }


def apply_surgery_freeze(
    student: nn.Module,
    *,
    freeze_ffn: bool = True,
    freeze_embeddings: bool = True,
) -> dict[str, int]:
    """Freeze transplanted SwiGLU FFNs and tied Embeddings so distillation focuses 100% on Gated DeltaNet-2 state conversion."""
    student.requires_grad_(True)
    if freeze_embeddings and hasattr(student, "embedding"):
        student.embedding.requires_grad_(False)
        if hasattr(student, "lm_head"):
            student.lm_head.requires_grad_(False)
    if freeze_ffn and hasattr(student, "cells"):
        for cell in student.cells:
            cell.ffn.requires_grad_(False)
            cell.ffn_norm.requires_grad_(False)
    trainable = sum(p.numel() for p in {p.data_ptr(): p for p in student.parameters() if p.requires_grad}.values())
    frozen = sum(p.numel() for p in {p.data_ptr(): p for p in student.parameters() if not p.requires_grad}.values())
    return {"trainable_params": trainable, "frozen_params": frozen}


def _forward_teacher(
    teacher: nn.Module,
    input_ids: torch.Tensor,
    *,
    need_layer_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...]]:
    """Unified teacher forward supporting HuggingFace CausalLMs and native llm2 Backbones."""
    try:
        out = teacher(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    except TypeError:
        try:
            out = teacher(input_ids, return_layer_states=need_layer_states)
        except TypeError:
            out = teacher(input_ids)
    logits = out.logits
    hidden = None
    layers_h: tuple[torch.Tensor, ...] = ()
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        if isinstance(out.hidden_states, (tuple, list)):
            hidden = out.hidden_states[-1]
            # HF hidden_states = (embed_out, layer_0, ..., layer_{L-1})
            if len(out.hidden_states) > 1:
                layers_h = tuple(out.hidden_states[1:])
        else:
            hidden = out.hidden_states
    if not layers_h and getattr(out, "layer_states", ()):
        layers_h = tuple(out.layer_states)
    return logits, hidden, layers_h


def compute_distill_loss(
    teacher: nn.Module,
    student: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg: DistillConfig,
    *,
    hidden_adapter: nn.Module | None = None,
    mtp_heads: nn.ModuleList | None = None,
    use_amp: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute FP32 RADLADS/LoopUS distillation loss (KL + CE + per-layer/final hidden alignment + Exit Gate + MTP)."""
    device_type = input_ids.device.type
    need_layers = bool(cfg.layer_mse_weight > 0.0)
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            t_logits, t_hidden, t_layers = _forward_teacher(teacher, input_ids, need_layer_states=need_layers)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        try:
            s_out = student(input_ids, return_layer_states=need_layers)
        except TypeError:
            s_out = student(input_ids)

    valid_mask = targets != -100
    s_logits_f = s_out.logits.float()
    t_logits_f = t_logits.detach().float()
    v_common = min(s_logits_f.shape[-1], t_logits_f.shape[-1])

    # 1. Hard-target CE in FP32
    ce_loss = F.cross_entropy(s_logits_f.flatten(0, 1), targets.flatten(), ignore_index=-100)
    total = cfg.ce_weight * ce_loss
    metrics: dict[str, float] = {"ce_loss": float(ce_loss.detach().item())}

    # 2. Temperature-scaled KL distillation in FP32 over valid tokens
    if cfg.kl_weight > 0.0 and valid_mask.any():
        inv_t = 1.0 / float(cfg.temperature)
        s_sel = s_logits_f[..., :v_common][valid_mask] * inv_t
        t_sel = t_logits_f[..., :v_common][valid_mask] * inv_t
        log_p_s = F.log_softmax(s_sel, dim=-1)
        log_p_t = F.log_softmax(t_sel, dim=-1)
        kl_loss = F.kl_div(log_p_s, log_p_t, reduction="batchmean", log_target=True) * (cfg.temperature ** 2)
        total = total + cfg.kl_weight * kl_loss
        metrics["kl_loss"] = float(kl_loss.detach().item())

    # 3. Hidden-state representation alignment (normalized MSE)
    if cfg.hidden_mse_weight > 0.0 and t_hidden is not None and s_out.hidden_states is not None:
        s_h = s_out.hidden_states.float()
        t_h = t_hidden.detach().float()
        if hidden_adapter is not None:
            t_h = hidden_adapter(t_h.to(dtype=next(hidden_adapter.parameters()).dtype)).float()
        if t_h.shape == s_h.shape:
            s_norm = F.normalize(s_h, dim=-1, eps=1e-6)
            t_norm = F.normalize(t_h, dim=-1, eps=1e-6)
            hid_loss = F.mse_loss(s_norm, t_norm)
            total = total + cfg.hidden_mse_weight * hid_loss
            metrics["hidden_mse"] = float(hid_loss.detach().item())

    # 3b. Per-layer RADLADS attention-replacement alignment (prevents depth snowball drift across transplanted FFNs)
    s_layers = getattr(s_out, "layer_states", ())
    if cfg.layer_mse_weight > 0.0 and t_layers and s_layers:
        n_t = len(t_layers)
        n_s = len(s_layers)
        layer_losses = []
        for s_idx, sh in enumerate(s_layers):
            t_idx = 0 if n_s == 1 else round(s_idx * (n_t - 1) / max(n_s - 1, 1))
            th_raw = t_layers[t_idx]
            th = th_raw.detach().float()
            use_normed_s = (
                t_hidden is not None
                and s_out.hidden_states is not None
                and th_raw.data_ptr() == t_hidden.data_ptr()
            )
            sh_f = s_out.hidden_states.float() if use_normed_s else sh.float()
            if hidden_adapter is not None and th.shape[-1] != sh_f.shape[-1]:
                th = hidden_adapter(th.to(dtype=next(hidden_adapter.parameters()).dtype)).float()
            if th.shape == sh_f.shape:
                layer_losses.append(
                    F.mse_loss(F.normalize(sh_f, dim=-1, eps=1e-6), F.normalize(th, dim=-1, eps=1e-6))
                )
        if layer_losses:
            l_mse = torch.stack(layer_losses).mean()
            total = total + cfg.layer_mse_weight * l_mse
            metrics["layer_mse"] = float(l_mse.detach().item())

    # 4. Self-distilled Exit Gate supervision
    if cfg.exit_gate_weight > 0.0 and getattr(s_out, "exit_logits", ()) and getattr(s_out, "intermediate_states", ()) and valid_mask.any():
        with torch.no_grad():
            final_pred = s_logits_f.detach().argmax(dim=-1)
        eg_losses = []
        for e_logit, inter_h in zip(s_out.exit_logits, s_out.intermediate_states):
            with torch.no_grad():
                inter_pred = student.lm_head(inter_h.detach()).argmax(dim=-1)
                y_halt = (inter_pred == final_pred).float()
            eg_losses.append(F.binary_cross_entropy_with_logits(e_logit.squeeze(-1).float()[valid_mask], y_halt[valid_mask]))
        eg_loss = torch.stack(eg_losses).mean()
        total = total + cfg.exit_gate_weight * eg_loss
        metrics["exit_gate_loss"] = float(eg_loss.detach().item())

    # 5. Optional MTP heads supervision
    if cfg.mtp_weight > 0.0 and mtp_heads is not None and len(mtp_heads) > 0 and s_out.hidden_states is not None:
        mtp_losses = []
        for offset, head in enumerate(mtp_heads, start=1):
            if input_ids.shape[1] > offset:
                m_logits = head(s_out.hidden_states[:, :-offset]).float()
                mtp_losses.append(F.cross_entropy(m_logits.flatten(0, 1), targets[:, offset:].flatten(), ignore_index=-100))
        if mtp_losses:
            mtp_loss = torch.stack(mtp_losses).mean()
            total = total + cfg.mtp_weight * mtp_loss
            metrics["mtp_loss"] = float(mtp_loss.detach().item())

    metrics["total_loss"] = float(total.detach().item())
    return total, metrics


def run_distill_conversion(
    teacher: nn.Module,
    student: nn.Module,
    batch_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    cfg: DistillConfig,
    device: torch.device,
    *,
    mtp_heads: nn.ModuleList | None = None,
    warmstart_embed: bool = True,
    eval_fn: Callable[[nn.Module], dict] | None = None,
    save_path: str | Path | None = None,
    log_interval: int = 50,
) -> dict:
    """Run Rota 1 distillation loop converting open CausalLM knowledge into a looped FractalDelta V2 student."""
    if cfg.precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 distillation requires CUDA")

    teacher.eval()
    teacher.requires_grad_(False)
    student.train()

    if cfg.surgery_transplant:
        warm_info = transplant_teacher_weights(teacher, student)
    elif warmstart_embed:
        warm_info = warmstart_student_embeddings(teacher, student)
    else:
        warm_info = {"warmstarted": False}

    freeze_info = (
        apply_surgery_freeze(student, freeze_ffn=cfg.freeze_ffn, freeze_embeddings=cfg.freeze_embeddings)
        if (cfg.freeze_ffn or cfg.freeze_embeddings)
        else None
    )

    # Build linear hidden adapter automatically when teacher & student widths differ.
    hidden_adapter = None
    t_dim = getattr(getattr(teacher, "config", None), "hidden_size", None) or getattr(getattr(teacher, "config", None), "d_model", None)
    s_dim = student.config.d_model
    if cfg.hidden_mse_weight > 0.0 and t_dim is not None and int(t_dim) != int(s_dim):
        hidden_adapter = nn.Linear(int(t_dim), int(s_dim), bias=False, device=device)
        nn.init.normal_(hidden_adapter.weight, std=0.02)

    params = list(student.parameters())
    if hidden_adapter is not None:
        params += list(hidden_adapter.parameters())
    if mtp_heads is not None:
        params += list(mtp_heads.parameters())
    # Deduplicate shared embedding / lm_head parameters.
    unique_params = list({p.data_ptr(): p for p in params if p.requires_grad}.values())
    optimizer = torch.optim.AdamW(unique_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _lr_scale(step_idx: int) -> float:
        if step_idx < cfg.warmup_steps:
            return float(step_idx + 1) / float(max(cfg.warmup_steps, 1))
        progress = float(step_idx - cfg.warmup_steps) / float(max(cfg.max_steps - cfg.warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return cfg.min_lr_ratio + 0.5 * (1.0 - cfg.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale)
    use_amp = cfg.precision == "bf16" and device.type == "cuda"

    it = iter(batch_iter)
    history: list[dict] = []
    tokens_total = 0
    t0 = time.perf_counter()

    for step in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_metrics: dict[str, float] = {}
        for _ in range(cfg.grad_accum_steps):
            bx, by = next(it)
            bx = bx.to(device=device)
            by = by.to(device=device)
            loss, m = compute_distill_loss(
                teacher,
                student,
                bx,
                by,
                cfg,
                hidden_adapter=hidden_adapter,
                mtp_heads=mtp_heads,
                use_amp=use_amp,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite distillation loss at step {step}")
            (loss / float(cfg.grad_accum_steps)).backward()
            tokens_total += int((by != -100).sum().item())
            for k, v in m.items():
                step_metrics[k] = step_metrics.get(k, 0.0) + v / float(cfg.grad_accum_steps)
        torch.nn.utils.clip_grad_norm_(unique_params, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        history.append(step_metrics)
        if log_interval > 0 and ((step + 1) % log_interval == 0 or step == cfg.max_steps - 1):
            el = time.perf_counter() - t0
            print(
                f"[Distill step={step+1}/{cfg.max_steps}] total={step_metrics['total_loss']:.4f} "
                f"ce={step_metrics['ce_loss']:.4f} kl={step_metrics.get('kl_loss', 0.0):.4f} "
                f"hid={step_metrics.get('hidden_mse', 0.0):.4f} lyr={step_metrics.get('layer_mse', 0.0):.4f} elapsed={el:.1f}s",
                flush=True,
            )

    elapsed_s = time.perf_counter() - t0
    eval_res = eval_fn(student) if eval_fn is not None else None
    if save_path is not None:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Invariant (AGENTS.md): use save_model to preserve shared embedding/lm_head tying in safetensors.
        save_model(student, str(out_p))

    return {
        "warmstart": warm_info,
        "surgery_freeze": freeze_info,
        "steps_completed": len(history),
        "tokens_processed": tokens_total,
        "elapsed_seconds": round(elapsed_s, 2),
        "tok_per_s": round(tokens_total / max(elapsed_s, 1e-9), 1),
        "initial_metrics": history[0] if history else {},
        "final_metrics": history[-1] if history else {},
        "eval": eval_res,
        "saved_checkpoint": None if save_path is None else str(save_path),
    }
