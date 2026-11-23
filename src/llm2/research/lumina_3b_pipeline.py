"""Dual-GPU (2x T4) & Single-Device Conversion Pipeline for Lumina-V1-3B (True 3:1 Gated DeltaNet-2 + NoPE SWA Hybrid)."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Iterable

from safetensors.torch import load_model, save_model
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..architectures.fractal_delta import FractalDeltaCell, FractalDeltaMemory, SlidingWindowGQACell, SWAState
from ..config import ModelConfig, ResearchConfig
from .distill_convert import (
    _attn_delta_loss,
    _extract_teacher_layers_and_norm,
    _repr_alignment_loss,
    apply_surgery_freeze,
    attach_ffn_swa_lora,
    merge_and_remove_lora,
    transplant_teacher_weights,
)


def build_lumina_3b_config_from_hf(
    hf_cfg,
    *,
    window: int = 1024,
    sinks: int = 4,
    hybrid_stride: int = 4,
    max_seq_len: int = 4096,
    gradient_checkpointing: bool = False,
    swa_rope: bool | None = None,
    deltanet_rope: bool = True,
) -> tuple[ModelConfig, ResearchConfig]:
    """Build exact Lumina-V1 True 3:1 Hybrid (ModelConfig, ResearchConfig) from a HuggingFace CausalLM config."""
    d_model = int(getattr(hf_cfg, "hidden_size"))
    n_layers = int(getattr(hf_cfg, "num_hidden_layers"))
    n_heads = int(getattr(hf_cfg, "num_attention_heads"))
    n_kv = int(getattr(hf_cfg, "num_key_value_heads", n_heads))
    hidden_dim = int(getattr(hf_cfg, "intermediate_size"))
    vocab_size = int(getattr(hf_cfg, "vocab_size"))
    rope_theta = float(getattr(hf_cfg, "rope_theta", 10000.0))
    norm_eps = float(getattr(hf_cfg, "rms_norm_eps", 1e-6))

    # Auto-detect SmolLM3 NoPE on every 4th layer (no_rope_layer_interval == hybrid_stride -> SWA layers are NoPE).
    if swa_rope is None:
        no_rope_int = getattr(hf_cfg, "no_rope_layer_interval", None)
        use_swa_rope = not (no_rope_int is not None and int(no_rope_int) == int(hybrid_stride))
    else:
        use_swa_rope = bool(swa_rope)

    m_cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        norm_eps=norm_eps,
        dropout=0.0,
    )
    r_cfg = ResearchConfig(
        architecture="fractal_delta",
        scale_conditioning=False,
        share_scales=False,
        state_execution="chunked",
        state_chunk_size=32,
        gradient_checkpointing=gradient_checkpointing,
        fractal_beta_scale=2.0,
        fractal_window=int(window),
        fractal_hybrid_stride=int(hybrid_stride),
        fractal_hybrid_swa=True,
        fractal_swa_kv_heads=n_kv,
        fractal_swa_sinks=int(sinks),
        fractal_swa_rope=use_swa_rope,
        fractal_deltanet_rope=bool(deltanet_rope),
        fractal_golden_phase=not bool(deltanet_rope),
        fractal_decoupled_gates=True,
        fractal_assoc_delta=True,
        fractal_exit_gate=False,
    )
    return m_cfg, r_cfg


@torch.no_grad()
def build_student_for_surgery(
    m_cfg: ModelConfig,
    r_cfg: ResearchConfig,
    *,
    dtype: torch.dtype = torch.float16,
) -> FractalDeltaMemory:
    """Instantiate FractalDeltaMemory on meta -> to_empty(cpu) skipping redundant 3B FFN CPU RNG (~175s saved)."""
    from ..architectures.fractal_delta import _init_decay_bias, _GOLDEN_PHI

    with torch.device("meta"):
        student = FractalDeltaMemory(m_cfg, r_cfg)
    student.to_empty(device="cpu")
    student.to(dtype=dtype)
    student.lm_head.weight = student.embedding.weight

    # Re-initialize buffers & lightweight non-transplanted heads/gates cleanly (FFN/SWA/QKV/Embed will be overwritten by transplant).
    for mod in student.modules():
        if hasattr(mod, "weight") and mod.weight is not None and mod.weight.ndim == 1:
            nn.init.ones_(mod.weight)
    nn.init.normal_(student.local_conv.weight, std=0.02)
    nn.init.normal_(student.assoc_address.weight, std=0.02)
    nn.init.normal_(student.assoc_value.weight, std=0.02)
    nn.init.normal_(student.assoc_write_gate.weight, std=0.02)
    nn.init.constant_(student.assoc_write_gate.bias, -2.0)
    nn.init.normal_(student.assoc_readout.weight, std=1e-4)
    student.assoc_scale.fill_(1.0)
    if student.assoc_lag_mix is not None:
        student.assoc_lag_mix.zero_()
    for lvl_i, ad in enumerate(student.adapters):
        ad.attn_scale.fill_(1.0)
        ad.attn_shift.zero_()
        ad.ffn_scale.fill_(1.0)
        ad.ffn_shift.zero_()
        span = (lvl_i - 0.5 * (student.depth - 1)) * 0.5
        ad.decay_bias.fill_(span)
        ad.beta_bias.fill_(-max(0.0, span) if student.leaky_budget else 0.0)
        if ad.beta_write_bias is not None:
            ad.beta_write_bias.fill_(-max(0.0, span) if student.leaky_budget else 0.0)
        ad.euler_attn.fill_(1.0)
        ad.euler_ffn.fill_(1.0)
        if ad.anchor_gate is not None:
            ad.anchor_gate.fill_(-15.0 if lvl_i in student.swa_levels else -8.0)
        if ad.val_mix is not None:
            ad.val_mix.fill_(-1.0)
        if ad.phase_scale is not None:
            ad.phase_scale.fill_(1.0)

    proj_std = 0.02 / math.sqrt(2 * student.depth)
    for cell in student.cells:
        if isinstance(cell, SlidingWindowGQACell):
            if cell.use_rope:
                inv_freq = 1.0 / (
                    float(m_cfg.rope_theta) ** (torch.arange(0, cell.head_dim, 2, dtype=torch.float64) / float(cell.head_dim))
                )
                cell.register_buffer("rope_inv_freq", inv_freq, persistent=False)
            continue
        nn.init.normal_(cell.conv.weight, std=1e-4)
        nn.init.zeros_(cell.conv.bias)
        nn.init.normal_(cell.decay.weight, std=0.02)
        if cell.kda:
            nn.init.normal_(cell.decay_down.weight, std=0.02)
            cell.decay.bias.copy_(_init_decay_bias(cell.heads).repeat_interleave(cell.head_dim).to(dtype=cell.decay.bias.dtype))
        else:
            cell.decay.bias.copy_(_init_decay_bias(cell.heads).to(dtype=cell.decay.bias.dtype))
        nn.init.normal_(cell.beta.weight, std=0.02)
        nn.init.constant_(cell.beta.bias, -1.0)
        if cell.beta_write is not None:
            nn.init.normal_(cell.beta_write.weight, std=0.02)
            nn.init.constant_(cell.beta_write.bias, -1.0)
        nn.init.normal_(cell.out_gate.weight, std=1e-4)
        cell.out_gate_bias = torch.tensor(1.784376, dtype=dtype)
        cell.head_gamma.fill_(0.15)
        nn.init.normal_(cell.out_proj.weight, std=proj_std)
        if cell.golden_phase:
            rot_pairs = max(1, cell.head_dim // 2)
            exponents = 1.0 + 12.0 * torch.arange(rot_pairs, dtype=torch.float64) / max(1, rot_pairs)
            cell.register_buffer("golden_inv_freq", math.pi / (_GOLDEN_PHI ** exponents), persistent=False)
        elif cell.deltanet_rope:
            inv_freq = 1.0 / (
                float(m_cfg.rope_theta) ** (torch.arange(0, cell.head_dim, 2, dtype=torch.float64) / float(cell.head_dim))
            )
            cell.register_buffer("rope_inv_freq", inv_freq, persistent=False)
    return student


class CompactAdamW(torch.optim.Optimizer):
    """Memory-compact AdamW storing exp_avg and sqrt(exp_avg_sq) in FP16 (4 B/param) while stepping in FP32."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float16)
                    state["exp_avg_sq_sqrt"] = torch.zeros_like(p, dtype=torch.float16)
                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"].float()
                v = state["exp_avg_sq_sqrt"].float().square()
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_c1 = 1.0 - beta1 ** step
                bias_c2 = 1.0 - beta2 ** step
                denom = (v.sqrt() / math.sqrt(bias_c2)).add_(eps)
                p_f = p.float()
                if wd != 0.0:
                    p_f.mul_(1.0 - lr * wd)
                p_f.addcdiv_(m / bias_c1, denom, value=-lr)
                p.copy_(p_f.to(dtype=p.dtype))
                state["exp_avg"].copy_(m.clamp(-65500.0, 65500.0).to(dtype=torch.float16))
                state["exp_avg_sq_sqrt"].copy_(v.sqrt().clamp(0.0, 65500.0).to(dtype=torch.float16))
        return loss


def _cosine_lr_scale(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(warmup_steps, 1))
    progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _teacher_forward_hidden_and_deltas(
    teacher: nn.Module,
    input_ids: torch.Tensor,
    *,
    delta_levels: set[int] | None = None,
    need_attn_deltas: bool = False,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], dict[int, torch.Tensor]]:
    """Run teacher base model (skipping full vocab lm_head) returning (final_hidden, layer_inputs, layer_outputs, attn_deltas)."""
    t_layers, _ = _extract_teacher_layers_and_norm(teacher)
    u_captured: dict[int, torch.Tensor] = {}
    hooks = []
    if need_attn_deltas:
        for idx, tl in enumerate(t_layers):
            if delta_levels is not None and idx not in delta_levels:
                continue
            fnorm = getattr(tl, "post_attention_layernorm", None) or getattr(tl, "ffn_norm", None)
            if fnorm is not None:
                hooks.append(
                    fnorm.register_forward_pre_hook(
                        lambda _m, args, _i=idx: u_captured.__setitem__(_i, args[0].detach())
                    )
                )
    try:
        with torch.autocast(device_type=input_ids.device.type, dtype=amp_dtype, enabled=use_amp):
            if hasattr(teacher, "model") and callable(teacher.model):
                out = teacher.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                final_h = getattr(out, "last_hidden_state", hs[-1])
                layer_inputs = list(hs[:-1])
                layer_outputs = list(hs[1:])
            else:
                try:
                    out = teacher(input_ids, return_logits=False, return_layer_states=True)
                except TypeError:
                    out = teacher(input_ids)
                final_h = getattr(out, "hidden_states", None)
                layer_outputs = list(getattr(out, "layer_states", ()))
                if hasattr(teacher, "embedding") and layer_outputs:
                    h0 = teacher.embedding(input_ids)
                    layer_inputs = [h0, *layer_outputs[:-1]]
                else:
                    layer_inputs = []
    finally:
        for h in hooks:
            h.remove()

    attn_deltas: dict[int, torch.Tensor] = {}
    if need_attn_deltas and layer_inputs:
        for idx, u in u_captured.items():
            if idx < len(layer_inputs):
                attn_deltas[idx] = (u.float() - layer_inputs[idx].float()).to(dtype=u.dtype)
    return final_h, layer_inputs, layer_outputs, attn_deltas


def _xfer_tensor(t: torch.Tensor, target_dev: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Transfer detached tensors safely across GPUs via CPU staging when needed (avoids broken PCIe P2P hangs)."""
    dt = t.dtype if dtype is None else dtype
    if t.device == target_dev:
        return t if t.dtype == dt else t.to(dtype=dt)
    if t.device.type == "cuda" and target_dev.type == "cuda":
        return t.to(device="cpu", dtype=dt).to(device=target_dev)
    return t.to(device=target_dev, dtype=dt)
def _get_teacher_lm_head(teacher: nn.Module) -> nn.Module:
    if hasattr(teacher, "lm_head") and teacher.lm_head is not None:
        return teacher.lm_head
    if hasattr(teacher, "get_output_embeddings") and teacher.get_output_embeddings() is not None:
        return teacher.get_output_embeddings()
    raise ValueError("teacher model has no lm_head / output embeddings")


def _chunked_hidden_ce_kl_loss(
    s_hidden_flat: torch.Tensor,
    s_lm_head: nn.Module,
    t_hidden_flat: torch.Tensor,
    t_lm_head: nn.Module,
    targets_flat: torch.Tensor,
    *,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    chunk_tokens: int = 256,
) -> tuple[torch.Tensor, float, float]:
    """Exact FP32 CE + KL from teacher & student final hidden states without materializing full (N, V) logits."""
    n_tok = int(s_hidden_flat.shape[0])
    total_valid = int((targets_flat != -100).sum().item())
    if total_valid <= 0:
        return s_hidden_flat.sum() * 0.0, 0.0, 0.0
    inv_t = 1.0 / float(temperature)
    t_sq = float(temperature) ** 2
    v_common = min(int(s_lm_head.weight.shape[0]), int(t_lm_head.weight.shape[0]))
    ce_acc = s_hidden_flat.new_zeros((), dtype=torch.float32)
    kl_acc = s_hidden_flat.new_zeros((), dtype=torch.float32)
    s_dev = s_hidden_flat.device
    t_dev = t_lm_head.weight.device

    def _chunk_fn(hs_c: torch.Tensor, ht_c: torch.Tensor, y_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_log = s_lm_head(hs_c.to(dtype=s_lm_head.weight.dtype)).float()
        ce_c = F.cross_entropy(s_log, y_c, ignore_index=-100, reduction="sum")
        vm = y_c != -100
        if kl_weight > 0.0 and vm.any():
            with torch.no_grad():
                ht_in = ht_c.to(device=t_dev, dtype=t_lm_head.weight.dtype)
                t_log = t_lm_head(ht_in).float()
                vm_t = vm.to(device=t_dev) if t_dev != vm.device else vm
                log_pt = F.log_softmax(t_log[:, :v_common][vm_t] * inv_t, dim=-1)
                if log_pt.device != s_dev:
                    log_pt = log_pt.to(device="cpu").to(device=s_dev)
            log_ps = F.log_softmax(s_log[:, :v_common][vm] * inv_t, dim=-1)
            kl_c = F.kl_div(log_ps, log_pt, reduction="sum", log_target=True)
        else:
            kl_c = s_log.new_zeros(())
        return ce_c, kl_c

    use_ckpt = torch.is_grad_enabled() and s_hidden_flat.requires_grad
    for start in range(0, n_tok, chunk_tokens):
        end = min(start + chunk_tokens, n_tok)
        hs_c = s_hidden_flat[start:end]
        ht_c = t_hidden_flat[start:end].detach()
        y_c = targets_flat[start:end]
        if use_ckpt:
            ce_c, kl_c = grad_checkpoint(_chunk_fn, hs_c, ht_c, y_c, use_reentrant=False)
        else:
            ce_c, kl_c = _chunk_fn(hs_c, ht_c, y_c)
        ce_acc = ce_acc + ce_c
        kl_acc = kl_acc + kl_c

    ce_mean = ce_acc / float(total_valid)
    kl_mean = (kl_acc / float(total_valid)) * t_sq
    total = ce_weight * ce_mean + (kl_weight * kl_mean if kl_weight > 0.0 else 0.0)
    return total, float(ce_mean.detach().item()), float(kl_mean.detach().item())


def _move_cell_attn_only(cell: FractalDeltaCell, device: torch.device, dtype: torch.dtype) -> None:
    """Move only the DeltaNet attention submodules of a cell to device (keeping frozen SwiGLU FFN on CPU in Subfase 1)."""
    for name in ("attn_norm", "conv", "qkv", "decay_down", "decay", "beta", "beta_write", "out_gate", "out_proj", "fe_norm", "fe_down", "fe_up"):
        mod = getattr(cell, name, None)
        if mod is not None:
            mod.to(device=device, dtype=dtype)
    cell.head_gamma.data = cell.head_gamma.data.to(device=device, dtype=dtype)
    cell.out_gate_bias = cell.out_gate_bias.to(device=device, dtype=dtype)
    for buf_name in ("golden_inv_freq", "rope_inv_freq"):
        buf = getattr(cell, buf_name, None)
        if buf is not None:
            setattr(cell, buf_name, buf.to(device=device))


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _cuda_dev_ctx(dev: torch.device):
    return torch.cuda.device(dev) if dev.type == "cuda" else _NullCtx()


def _place_student_split(
    student: FractalDeltaMemory,
    dev0: torch.device,
    dev1: torch.device,
    split_layer: int,
    *,
    frozen_dtype: torch.dtype,
    trainable_dtype: torch.dtype = torch.float32,
) -> None:
    """Place student across (dev0, dev1) with frozen weights in frozen_dtype (FP16 on T4) and trainable weights in FP32."""
    def _cast_module_by_grad(mod: nn.Module, target_dev: torch.device) -> None:
        mod.to(device=target_dev)
        for p in mod.parameters():
            dt = trainable_dtype if p.requires_grad else frozen_dtype
            if p.dtype != dt:
                p.data = p.data.to(dtype=dt)
        for buf_name, buf in list(mod.named_buffers(recurse=True)):
            if buf.is_floating_point() and "inv_freq" not in buf_name:
                buf.data = buf.data.to(device=target_dev, dtype=trainable_dtype)
            else:
                buf.data = buf.data.to(device=target_dev)

    # Keep tied embedding / lm_head, assoc channel, and final norm on head_dev (= dev1 if split_layer==0 else dev0).
    head_dev = dev1 if split_layer <= 0 else dev0
    for name in ("embedding", "input_norm", "local_conv", "local_norm", "assoc_address", "assoc_value", "assoc_write_gate", "assoc_readout", "norm", "window_attn", "exit_gate"):
        mod = getattr(student, name, None)
        if mod is not None:
            _cast_module_by_grad(mod, head_dev)
    student.assoc_scale.data = student.assoc_scale.data.to(
        device=head_dev, dtype=trainable_dtype if student.assoc_scale.requires_grad else frozen_dtype
    )
    if student.assoc_lag_mix is not None:
        student.assoc_lag_mix.data = student.assoc_lag_mix.data.to(
            device=head_dev, dtype=trainable_dtype if student.assoc_lag_mix.requires_grad else frozen_dtype
        )
    student.lm_head.weight = student.embedding.weight

    for lvl in range(student.depth):
        t_dev = dev0 if lvl < split_layer else dev1
        _cast_module_by_grad(student.cells[lvl], t_dev)
        if lvl < len(student.adapters):
            _cast_module_by_grad(student.adapters[lvl], t_dev)


def _forward_student_split(
    student: FractalDeltaMemory,
    input_ids: torch.Tensor,
    dev0: torch.device,
    dev1: torch.device,
    split_layer: int,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    return_layer_states: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run full 36-layer student forward split cleanly across (dev0, dev1) or single-GPU when split_layer<=0."""
    use_ckpt = bool(student.training and torch.is_grad_enabled() and getattr(student.research, "gradient_checkpointing", False))
    head_dev = dev1 if split_layer <= 0 else dev0
    with torch.autocast(device_type=head_dev.type, dtype=amp_dtype, enabled=use_amp):
        raw = student.embedding(input_ids)
        assoc_out, _, _ = student._run_assoc(raw, None, execution="chunked")
        x = raw
        anchor0 = x
        cur_phase0 = raw.new_zeros((raw.shape[0],), dtype=torch.int64) if student.use_phase_pos else None

    layer_states: list[torch.Tensor] = []
    on_dev1 = bool(split_layer <= 0)
    anchor = anchor0
    cur_phase = cur_phase0

    for lvl in range(student.depth):
        if lvl == split_layer and dev1 != dev0 and not on_dev1:
            x = x.to(dev1)
            anchor = anchor0.to(dev1)
            cur_phase = None if cur_phase0 is None else cur_phase0.to(dev1)
            on_dev1 = True
        cur_dev = dev1 if on_dev1 else dev0
        adapter = student.adapters[lvl] if student.depth > 1 else None
        cell = student.cells[lvl]
        with torch.autocast(device_type=cur_dev.type, dtype=amp_dtype, enabled=use_amp):
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid().to(dtype=x.dtype) * anchor
            if use_ckpt:
                if isinstance(cell, SlidingWindowGQACell):
                    def _run_swa(x_in, _c=cell, _a=adapter):
                        xo, _ = _c(x_in, None, adapter=_a)
                        return xo
                    x = grad_checkpoint(_run_swa, x, use_reentrant=False)
                else:
                    def _run_delta(x_in, anc_in, pos_in, _c=cell, _a=adapter):
                        xo, _ = _c(x_in, None, adapter=_a, phase_pos=pos_in, anchor=anc_in)
                        return xo
                    x = grad_checkpoint(_run_delta, x, anchor, cur_phase, use_reentrant=False)
            else:
                x, _ = cell(x, None, adapter=adapter, phase_pos=cur_phase, anchor=anchor)
        if return_layer_states:
            layer_states.append(x)

    if on_dev1 and dev1 != head_dev:
        x = x.to(head_dev)
    with torch.autocast(device_type=head_dev.type, dtype=amp_dtype, enabled=use_amp):
        hidden = student.norm(x + assoc_out)
    return hidden, layer_states


def save_lumina_checkpoint(
    student: FractalDeltaMemory,
    out_dir: str | Path,
    *,
    meta: dict | None = None,
    save_dtype: torch.dtype | None = None,
) -> Path:
    """Save pure FractalDeltaMemory safetensors (preserving tied embedding/lm_head) with zero duplicate model RAM."""
    import gc

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sd = {
        k: v.detach().to(
            device="cpu",
            dtype=save_dtype if (save_dtype is not None and v.is_floating_point()) else v.dtype,
        )
        for k, v in student.state_dict().items()
    }
    if "lm_head.weight" in sd and "embedding.weight" in sd:
        sd["lm_head.weight"] = sd["embedding.weight"]
    with torch.device("meta"):
        student_meta = FractalDeltaMemory(student.config, student.research)
    student_meta.load_state_dict(sd, strict=True, assign=True)
    student_meta.lm_head.weight = student_meta.embedding.weight
    weights_path = out_path / "model.safetensors"
    save_model(student_meta, str(weights_path))
    cfg_payload = {
        "model": asdict(student.config),
        "research": asdict(student.research),
        "meta": meta or {},
    }
    (out_path / "config.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")
    del student_meta, sd
    gc.collect()
    return weights_path


def load_lumina_checkpoint(ckpt_dir: str | Path, *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> tuple[FractalDeltaMemory, dict]:
    """Load saved Lumina-V1 checkpoint preserving shared embedding/lm_head tying with minimal RAM & time."""
    from safetensors.torch import load_file

    ckpt_path = Path(ckpt_dir)
    payload = json.loads((ckpt_path / "config.json").read_text(encoding="utf-8"))
    m_cfg = ModelConfig(**payload["model"])
    r_cfg = ResearchConfig(**payload["research"])
    with torch.device("meta"):
        model = FractalDeltaMemory(m_cfg, r_cfg)
    dev_str = str(device)
    sd = load_file(str(ckpt_path / "model.safetensors"), device=dev_str)
    if "lm_head.weight" not in sd and "embedding.weight" in sd:
        sd["lm_head.weight"] = sd["embedding.weight"]
    if dtype is not None:
        for k, v in list(sd.items()):
            if v.is_floating_point() and v.dtype != dtype:
                sd[k] = v.to(dtype=dtype)
    res_load = model.load_state_dict(sd, strict=False, assign=True)
    unexpected_missing = [k for k in res_load.missing_keys if not k.endswith("_inv_freq")]
    if unexpected_missing or res_load.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={unexpected_missing} unexpected={res_load.unexpected_keys}")
    # Reconstruct non-persistent positional/frequency buffers cleanly on target device.
    from ..architectures.fractal_delta import SlidingWindowGQACell, _GOLDEN_PHI

    for cell in model.cells:
        if isinstance(cell, SlidingWindowGQACell):
            if cell.use_rope:
                inv_freq = 1.0 / (
                    float(m_cfg.rope_theta) ** (torch.arange(0, cell.head_dim, 2, dtype=torch.float64, device=dev_str) / float(cell.head_dim))
                )
                cell.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        else:
            if cell.golden_phase:
                rot_pairs = max(1, cell.head_dim // 2)
                exponents = 1.0 + 12.0 * torch.arange(rot_pairs, dtype=torch.float64, device=dev_str) / max(1, rot_pairs)
                cell.register_buffer("golden_inv_freq", math.pi / (_GOLDEN_PHI ** exponents), persistent=False)
            elif cell.deltanet_rope:
                inv_freq = 1.0 / (
                    float(m_cfg.rope_theta) ** (torch.arange(0, cell.head_dim, 2, dtype=torch.float64, device=dev_str) / float(cell.head_dim))
                )
                cell.register_buffer("rope_inv_freq", inv_freq, persistent=False)
    model.lm_head.weight = model.embedding.weight
    return model, payload.get("meta", {})


def run_subfase1_dual_gpu(
    teacher: nn.Module,
    student: FractalDeltaMemory,
    batch_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    dev0: torch.device,
    dev1: torch.device,
    max_steps: int,
    grad_accum_steps: int = 1,
    lr: float = 1.5e-3,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    precision: str = "fp16",
    log_interval: int = 20,
    ckpt_dir: str | Path | None = None,
    ckpt_interval: int = 0,
) -> dict:
    """Subfase 1: Layerwise parallel AttnDelta conversion of DeltaNet layers only (FFNs stay on CPU, SWA layers untouched)."""
    use_amp = precision in ("fp16", "bf16") and dev1.type == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    teacher.eval()
    teacher.requires_grad_(False)
    apply_surgery_freeze(student, freeze_ffn=True, freeze_embeddings=True, freeze_swa=True)

    delta_levels = [lvl for lvl in range(student.depth) if lvl not in student.swa_levels]
    delta_set = set(delta_levels)
    if not delta_levels:
        return {"steps_completed": 0, "tokens_processed": 0, "delta_levels": 0}

    # Move ONLY the DeltaNet attention submodules + adapters to dev1 (frozen FFNs stay on CPU saving ~5 GB VRAM!).
    student.train()
    with _cuda_dev_ctx(dev1):
        trainable_params: list[nn.Parameter] = []
        for lvl in delta_levels:
            cell = student.cells[lvl]
            assert isinstance(cell, FractalDeltaCell)
            _move_cell_attn_only(cell, dev1, torch.float32)
            for p in cell.parameters():
                if p.requires_grad:
                    trainable_params.append(p)
            if lvl < len(student.adapters):
                student.adapters[lvl].to(device=dev1, dtype=torch.float32)
                for p in student.adapters[lvl].parameters():
                    if p.requires_grad:
                        trainable_params.append(p)
        unique_params = list({p.data_ptr(): p for p in trainable_params}.values())
        optimizer = torch.optim.AdamW(unique_params, lr=lr, weight_decay=weight_decay, foreach=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: _cosine_lr_scale(s, warmup_steps, max_steps, min_lr_ratio)
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=(precision == "fp16" and dev1.type == "cuda"))

        it = iter(batch_iter)
        calibrated = False
        history: list[dict[str, float]] = []
        tokens_total = 0
        t0 = time.perf_counter()

        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_acc = 0.0
            cos_acc = 0.0
            for _ in range(grad_accum_steps):
                bx, by = next(it)
                with _cuda_dev_ctx(dev0):
                    bx0 = bx.to(device=dev0)
                    _, t_inputs, _, t_deltas = _teacher_forward_hidden_and_deltas(
                        teacher, bx0, delta_levels=delta_set, need_attn_deltas=True, use_amp=(precision in ("fp16", "bf16") and dev0.type == "cuda"), amp_dtype=amp_dtype
                    )
                    if dev0.type == "cuda":
                        torch.cuda.synchronize(dev0)
                anc1 = _xfer_tensor(t_inputs[0], dev1, torch.float32)

                if not calibrated:
                    with torch.no_grad():
                        for lvl in delta_levels:
                            hin1 = _xfer_tensor(t_inputs[lvl], dev1, torch.float32)
                            td1 = _xfer_tensor(t_deltas[lvl], dev1, torch.float32)
                            with torch.autocast(device_type=dev1.type, dtype=amp_dtype, enabled=use_amp):
                                sd1 = student.forward_level_attn_delta(lvl, hin1, anc1).float()
                            ratio = (td1.norm(dim=-1).mean() / sd1.norm(dim=-1).mean().clamp_min(1e-6)).item()
                            ca = F.cosine_similarity(sd1, td1, dim=-1, eps=1e-6).mean().item()
                            scale = max(0.02, min(10.0, ratio * max(0.25, ca)))
                            student.cells[lvl].head_gamma.data.mul_(scale)
                    calibrated = True

                # Compute & backprop 1 DeltaNet layer at a time: mathematically exact sum, 27x lower peak activation VRAM!
                denom = float(len(delta_levels) * grad_accum_steps)
                for lvl in delta_levels:
                    hin1 = _xfer_tensor(t_inputs[lvl], dev1, torch.float32)
                    td1 = _xfer_tensor(t_deltas[lvl], dev1, torch.float32)
                    with torch.autocast(device_type=dev1.type, dtype=amp_dtype, enabled=use_amp):
                        sd1 = student.forward_level_attn_delta(lvl, hin1, anc1).float()
                        loss_l = _attn_delta_loss(sd1, td1)
                    if not torch.isfinite(loss_l):
                        raise RuntimeError(f"Non-finite Subfase-1 loss at step {step} level {lvl}")
                    scaler.scale(loss_l / denom).backward()
                    loss_acc += float(loss_l.detach().item()) / denom
                    with torch.no_grad():
                        cos_acc += float(F.cosine_similarity(sd1.detach(), td1, dim=-1, eps=1e-6).mean().item()) / denom
                    del hin1, td1, sd1, loss_l

                tokens_total += int((by != -100).sum().item())
                del t_inputs, t_deltas, anc1

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unique_params, grad_clip, foreach=False)
            prev_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_enabled() or scaler.get_scale() >= prev_scale:
                scheduler.step()
            history.append({"attn_delta_loss": loss_acc, "mean_attn_cos": cos_acc, "lr": optimizer.param_groups[0]["lr"]})

            if log_interval > 0 and ((step + 1) % log_interval == 0 or step == max_steps - 1):
                el = time.perf_counter() - t0
                print(
                    f"[Subfase 1 step={step+1}/{max_steps}] attn_loss={loss_acc:.4f} mean_cos={cos_acc:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} tok/s={tokens_total / max(el, 1e-6):.0f} elapsed={el:.1f}s",
                    flush=True,
                )
            if ckpt_dir is not None and ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
                if dev1.type == "cuda":
                    torch.cuda.synchronize(dev1)
                save_lumina_checkpoint(
                    student,
                    ckpt_dir,
                    meta={"stage": "subfase1", "step": step + 1, "tokens": tokens_total, "mean_cos": cos_acc},
                    save_dtype=torch.float16 if precision == "fp16" else None,
                )

        if dev1.type == "cuda":
            torch.cuda.synchronize(dev1)
        # Cleanly release optimizer state before Subfase 2.
        del optimizer, scheduler, scaler
        if dev1.type == "cuda":
            torch.cuda.empty_cache()
    elapsed_s = time.perf_counter() - t0
    return {
        "steps_completed": len(history),
        "tokens_processed": tokens_total,
        "elapsed_seconds": round(elapsed_s, 2),
        "tok_per_s": round(tokens_total / max(elapsed_s, 1e-9), 1),
        "initial_metrics": history[0] if history else {},
        "final_metrics": history[-1] if history else {},
    }


def run_subfase2_dual_gpu(
    teacher: nn.Module,
    student: FractalDeltaMemory,
    batch_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    dev0: torch.device,
    dev1: torch.device,
    split_layer: int | None = None,
    max_steps: int,
    grad_accum_steps: int = 1,
    lr: float = 3e-4,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 30,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    temperature: float = 2.0,
    kl_weight: float = 1.0,
    ce_weight: float = 0.5,
    layer_mse_weight: float = 0.25,
    logit_chunk_tokens: int = 256,
    precision: str = "fp16",
    lora_rank: int = 0,
    lora_alpha: float = 32.0,
    compact_optimizer: bool | None = None,
    log_interval: int = 10,
    ckpt_dir: str | Path | None = None,
    ckpt_interval: int = 0,
) -> dict:
    """Subfase 2a (DeltaNet E2E KL+CE+layer_mse) or Subfase 2b (with temporary FFN+SWA LoRA merged back cleanly)."""
    use_amp = precision in ("fp16", "bf16") and dev0.type == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    frozen_dtype = amp_dtype if use_amp else torch.float32
    split = 0 if (split_layer is None and dev1 != dev0) else (student.depth if dev0 == dev1 else int(split_layer))
    head_dev = dev1 if split <= 0 else dev0
    use_compact = bool((split <= 0 and precision == "fp16") if compact_optimizer is None else compact_optimizer)

    teacher.eval()
    teacher.requires_grad_(False)
    t_lm_head = _get_teacher_lm_head(teacher)

    apply_surgery_freeze(student, freeze_ffn=True, freeze_embeddings=True, freeze_swa=True)
    lora_info = None
    if lora_rank > 0:
        lora_info = attach_ffn_swa_lora(student, rank=lora_rank, alpha=lora_alpha, include_ffn=True, include_swa=True)

    _place_student_split(student, dev0, dev1, split, frozen_dtype=frozen_dtype, trainable_dtype=torch.float32)
    student.train()

    with _cuda_dev_ctx(head_dev):
        unique_params = list({p.data_ptr(): p for p in student.parameters() if p.requires_grad}.values())
        optimizer = (
            CompactAdamW(unique_params, lr=lr, weight_decay=weight_decay)
            if use_compact
            else torch.optim.AdamW(unique_params, lr=lr, weight_decay=weight_decay, foreach=False)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: _cosine_lr_scale(s, warmup_steps, max_steps, min_lr_ratio)
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=(precision == "fp16" and head_dev.type == "cuda"))

        it = iter(batch_iter)
        history: list[dict[str, float]] = []
        tokens_total = 0
        t0 = time.perf_counter()

        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            step_m = {"total_loss": 0.0, "ce_loss": 0.0, "kl_loss": 0.0, "layer_mse": 0.0}
            for _ in range(grad_accum_steps):
                bx, by = next(it)
                with _cuda_dev_ctx(dev0):
                    bx_t = bx.to(device=dev0)
                    t_final_h, _, t_layers, _ = _teacher_forward_hidden_and_deltas(
                        teacher, bx_t, need_attn_deltas=False, use_amp=use_amp, amp_dtype=amp_dtype
                    )
                    if dev0.type == "cuda":
                        torch.cuda.synchronize(dev0)
                bx_s = bx_t if head_dev == dev0 else bx.to(device=head_dev)
                by_s = by.to(device=head_dev)
                s_hidden, s_layers = _forward_student_split(
                    student, bx_s, dev0, dev1, split, use_amp=use_amp, amp_dtype=amp_dtype, return_layer_states=(layer_mse_weight > 0.0)
                )

                loss, ce_v, kl_v = _chunked_hidden_ce_kl_loss(
                    s_hidden.flatten(0, 1),
                    student.lm_head,
                    t_final_h.detach().flatten(0, 1),
                    t_lm_head,
                    by_s.flatten(),
                    temperature=temperature,
                    ce_weight=ce_weight,
                    kl_weight=kl_weight,
                    chunk_tokens=logit_chunk_tokens,
                )

                l_mse_v = 0.0
                if layer_mse_weight > 0.0 and t_layers and s_layers:
                    l_losses = []
                    for idx, (sh, th) in enumerate(zip(s_layers, t_layers)):
                        use_normed = t_final_h is not None and th.data_ptr() == t_final_h.data_ptr()
                        sh_use = s_hidden if use_normed else sh
                        th_d = _xfer_tensor(th.detach(), sh_use.device, torch.float32)
                        l_losses.append(_repr_alignment_loss(sh_use.float(), th_d).to(head_dev))
                    l_mse = torch.stack(l_losses).mean()
                    loss = loss + layer_mse_weight * l_mse
                    l_mse_v = float(l_mse.detach().item())

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite Subfase-2 loss at step {step}")
                scaler.scale(loss / float(grad_accum_steps)).backward()
                tokens_total += int((by_s != -100).sum().item())
                step_m["total_loss"] += float(loss.detach().item()) / float(grad_accum_steps)
                step_m["ce_loss"] += ce_v / float(grad_accum_steps)
                step_m["kl_loss"] += kl_v / float(grad_accum_steps)
                step_m["layer_mse"] += l_mse_v / float(grad_accum_steps)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unique_params, grad_clip, foreach=False)
            prev_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_enabled() or scaler.get_scale() >= prev_scale:
                scheduler.step()
            history.append(step_m)

            if log_interval > 0 and ((step + 1) % log_interval == 0 or step == max_steps - 1):
                el = time.perf_counter() - t0
                print(
                    f"[Subfase 2{'b-LoRA' if lora_rank > 0 else 'a'} step={step+1}/{max_steps}] "
                    f"total={step_m['total_loss']:.4f} ce={step_m['ce_loss']:.4f} kl={step_m['kl_loss']:.4f} "
                    f"lyr={step_m['layer_mse']:.4f} tok/s={tokens_total / max(el, 1e-6):.0f} elapsed={el:.1f}s",
                    flush=True,
                )
            if ckpt_dir is not None and ckpt_interval > 0 and (step + 1) % ckpt_interval == 0 and lora_rank == 0:
                if head_dev.type == "cuda":
                    torch.cuda.synchronize(head_dev)
                save_lumina_checkpoint(
                    student,
                    ckpt_dir,
                    meta={"stage": "subfase2a", "step": step + 1, "tokens": tokens_total, **step_m},
                    save_dtype=torch.float16 if precision == "fp16" else None,
                )

        if head_dev.type == "cuda":
            torch.cuda.synchronize(head_dev)
        del optimizer, scheduler, scaler
        merged_lora = 0
        if lora_rank > 0:
            merged_lora = merge_and_remove_lora(student)
        if head_dev.type == "cuda":
            torch.cuda.synchronize(head_dev)

    for d in ({dev0, dev1, head_dev}):
        if d.type == "cuda":
            with torch.cuda.device(d):
                torch.cuda.synchronize(d)
                torch.cuda.empty_cache()

    if ckpt_dir is not None:
        save_lumina_checkpoint(
            student,
            ckpt_dir,
            meta={
                "stage": "subfase2b_merged" if lora_rank > 0 else "subfase2a_done",
                "steps": len(history),
                "tokens": tokens_total,
                "merged_lora_layers": merged_lora,
                **(history[-1] if history else {}),
            },
            save_dtype=torch.float16 if precision == "fp16" else None,
        )
    elapsed_s = time.perf_counter() - t0
    return {
        "lora": lora_info,
        "merged_lora_layers": merged_lora,
        "steps_completed": len(history),
        "tokens_processed": tokens_total,
        "elapsed_seconds": round(elapsed_s, 2),
        "tok_per_s": round(tokens_total / max(elapsed_s, 1e-9), 1),
        "initial_metrics": history[0] if history else {},
        "final_metrics": history[-1] if history else {},
    }
