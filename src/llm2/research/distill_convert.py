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
from torch.utils.checkpoint import checkpoint as grad_checkpoint


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
    precision: str = "fp32"  # fp32 | bf16 | fp16
    surgery_transplant: bool = False
    freeze_ffn: bool = False
    freeze_embeddings: bool = False
    freeze_swa: bool = True
    layer_mse_weight: float = 0.0
    teacher_forced_ratio: float = 0.5
    logit_chunk_tokens: int = 0

    def __post_init__(self):
        if self.temperature <= 0 or self.lr <= 0 or self.max_steps <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("temperature, lr, max_steps, and grad_accum_steps must be positive")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32|bf16|fp16")
        if self.logit_chunk_tokens < 0:
            raise ValueError("logit_chunk_tokens must be nonnegative")


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


def _expand_gqa_kv(weight: torch.Tensor, target_out: int, head_dim: int | None = None) -> torch.Tensor:
    w = weight.detach().float()
    if w.shape[0] == target_out:
        return w
    if w.shape[0] > 0 and target_out % w.shape[0] == 0:
        rep = target_out // w.shape[0]
        if head_dim is None:
            for candidate_hd in (128, 96, 80, 64, 48, 32, 16, 8):
                if w.shape[0] % candidate_hd == 0 and target_out % candidate_hd == 0:
                    head_dim = candidate_hd
                    break
        if head_dim is not None and head_dim > 0 and w.shape[0] % head_dim == 0:
            n_kv = w.shape[0] // head_dim
            return w.reshape(n_kv, head_dim, -1).repeat_interleave(rep, dim=0).reshape(target_out, -1)
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
    swa_copied = 0
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

        # 3. Attention projections: verbatim GQA transplant for SlidingWindowGQACell or QKV+Out for FractalDeltaCell
        attn = getattr(t_layer, "self_attn", None) or getattr(t_layer, "attn", None) or getattr(t_layer, "attention", None)
        if attn is not None and hasattr(cell, "q_proj") and hasattr(cell, "k_proj") and hasattr(cell, "v_proj"):
            if hasattr(attn, "qkv") and hasattr(attn, "proj") and attn.qkv.weight.shape[0] == 3 * d_s:
                q_w, k_w, v_w = attn.qkv.weight.detach().float().split(d_s, dim=0)
                _copy_matching_or_slice(cell.q_proj.weight, q_w)
                _copy_matching_or_slice(cell.k_proj.weight, k_w)
                _copy_matching_or_slice(cell.v_proj.weight, v_w)
                _copy_matching_or_slice(cell.out_proj.weight, attn.proj.weight)
                swa_copied += 1
            else:
                q_p = getattr(attn, "q_proj", None) or getattr(attn, "q", None)
                k_p = getattr(attn, "k_proj", None) or getattr(attn, "k", None)
                v_p = getattr(attn, "v_proj", None) or getattr(attn, "v", None)
                o_p = getattr(attn, "o_proj", None) or getattr(attn, "out", None)
                if q_p is not None and k_p is not None and v_p is not None and o_p is not None:
                    _copy_matching_or_slice(cell.q_proj.weight, q_p.weight)
                    if cell.k_proj.weight.shape[0] != k_p.weight.shape[0]:
                        _copy_matching_or_slice(
                            cell.k_proj.weight,
                            _expand_gqa_kv(k_p.weight, cell.k_proj.weight.shape[0], head_dim=getattr(cell, "head_dim", None)),
                        )
                        _copy_matching_or_slice(
                            cell.v_proj.weight,
                            _expand_gqa_kv(v_p.weight, cell.v_proj.weight.shape[0], head_dim=getattr(cell, "head_dim", None)),
                        )
                    else:
                        _copy_matching_or_slice(cell.k_proj.weight, k_p.weight)
                        _copy_matching_or_slice(cell.v_proj.weight, v_p.weight)
                    _copy_matching_or_slice(cell.out_proj.weight, o_p.weight)
                    swa_copied += 1
            continue

        cell_hd = getattr(cell, "head_dim", None)
        vw_for_scale: torch.Tensor | None = None
        if attn is not None and hasattr(cell, "qkv") and hasattr(cell.qkv, "weight") and hasattr(cell.out_proj, "weight"):
            if hasattr(attn, "qkv") and hasattr(attn, "proj"):
                _copy_matching_or_slice(cell.qkv.weight, attn.qkv.weight)
                _copy_matching_or_slice(cell.out_proj.weight, attn.proj.weight)
                if attn.qkv.weight.shape[0] == 3 * d_s:
                    vw_for_scale = attn.qkv.weight.detach().float()[2 * d_s : 3 * d_s]
                qkv_copied += 1
            else:
                q_p = getattr(attn, "q_proj", None) or getattr(attn, "q", None)
                k_p = getattr(attn, "k_proj", None) or getattr(attn, "k", None)
                v_p = getattr(attn, "v_proj", None) or getattr(attn, "v", None)
                o_p = getattr(attn, "o_proj", None) or getattr(attn, "out", None)
                if q_p is not None and k_p is not None and v_p is not None and o_p is not None:
                    qw = q_p.weight.detach().float()
                    kw = _expand_gqa_kv(k_p.weight, qw.shape[0], head_dim=cell_hd)
                    vw = _expand_gqa_kv(v_p.weight, qw.shape[0], head_dim=cell_hd)
                    if qw.shape[0] == d_s and qw.shape[1] == d_s and kw.shape == qw.shape and vw.shape == qw.shape:
                        qkv_cat = torch.cat((qw, kw, vw), dim=0)
                        _copy_matching_or_slice(cell.qkv.weight, qkv_cat)
                        vw_for_scale = vw
                    _copy_matching_or_slice(cell.out_proj.weight, o_p.weight)
                    qkv_copied += 1

        if quiet_init_gates and hasattr(cell, "conv"):
            # Start depthwise conv as near-identity residual and out_gate around SiLU(1.7844)=1.0 so transplanted W_O is not scrambled.
            nn.init.normal_(cell.conv.weight, std=1e-4)
            nn.init.zeros_(cell.conv.bias)
            if hasattr(cell, "out_gate") and hasattr(cell.out_gate, "weight"):
                nn.init.normal_(cell.out_gate.weight, std=1e-4)
            if hasattr(cell, "out_gate_bias"):
                cell.out_gate_bias.fill_(1.784376)
            if hasattr(cell, "head_gamma"):
                if vw_for_scale is not None and vw_for_scale.shape[0] == cell.head_gamma.shape[0]:
                    v_rms = (0.35 * vw_for_scale.norm(dim=1)).clamp(min=0.04, max=0.50)
                    cell.head_gamma.copy_(v_rms.to(device=cell.head_gamma.device, dtype=cell.head_gamma.dtype))
                else:
                    nn.init.constant_(cell.head_gamma, 0.15)

    swa_lvls = getattr(student, "swa_levels", frozenset())
    if quiet_init_gates and hasattr(student, "assoc_readout"):
        nn.init.normal_(student.assoc_readout.weight, std=1e-4)
    if quiet_init_gates and hasattr(student, "adapters"):
        for lvl_i, ad in enumerate(student.adapters):
            if getattr(ad, "anchor_gate", None) is not None:
                ad.anchor_gate.fill_(-15.0 if lvl_i in swa_lvls else -8.0)
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
        "swa_cells_copied": swa_copied,
        "norms_copied": norms_copied,
    }


def apply_surgery_freeze(
    student: nn.Module,
    *,
    freeze_ffn: bool = True,
    freeze_embeddings: bool = True,
    freeze_swa: bool = True,
) -> dict[str, int]:
    """Freeze transplanted SwiGLU FFNs, SWA layers, and tied Embeddings so distillation focuses on Gated DeltaNet-2 conversion."""
    student.requires_grad_(True)
    if freeze_embeddings and hasattr(student, "embedding"):
        student.embedding.requires_grad_(False)
        if hasattr(student, "lm_head"):
            student.lm_head.requires_grad_(False)
    swa_lvls = getattr(student, "swa_levels", frozenset())
    if hasattr(student, "cells"):
        for lvl_i, cell in enumerate(student.cells):
            if freeze_swa and lvl_i in swa_lvls:
                cell.requires_grad_(False)
                if hasattr(student, "adapters") and lvl_i < len(student.adapters):
                    student.adapters[lvl_i].requires_grad_(False)
            elif freeze_ffn:
                cell.ffn.requires_grad_(False)
                cell.ffn_norm.requires_grad_(False)
    trainable = sum(p.numel() for p in {p.data_ptr(): p for p in student.parameters() if p.requires_grad}.values())
    frozen = sum(p.numel() for p in {p.data_ptr(): p for p in student.parameters() if not p.requires_grad}.values())
    return {"trainable_params": trainable, "frozen_params": frozen}


class LoRALinear(nn.Module):
    """Minimal zero-init LoRA wrapper for Stage 2b FFN/SWA co-adaptation, mergeable back into pure nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(self.rank, self.in_features, device=dev, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.to(dtype=self.base.weight.dtype)
        out = self.base(xb)
        delta = F.linear(F.linear(xb, self.lora_a.to(dtype=xb.dtype)), self.lora_b.to(dtype=xb.dtype))
        return (out + delta * self.scale).to(dtype=x.dtype)

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        dev = self.base.weight.device
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                delta_w = (self.lora_b.float() @ self.lora_a.float()) * self.scale
                self.base.weight.add_(delta_w.to(device=dev, dtype=self.base.weight.dtype))
        else:
            delta_w = (self.lora_b.float() @ self.lora_a.float()) * self.scale
            self.base.weight.add_(delta_w.to(device=dev, dtype=self.base.weight.dtype))
        return self.base


def attach_ffn_swa_lora(
    student: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    include_ffn: bool = True,
    include_swa: bool = True,
) -> dict[str, int]:
    """Attach zero-init LoRA adapters to frozen SwiGLU FFNs and/or SlidingWindowGQACell projections."""
    wrapped = 0
    lora_params = 0
    swa_lvls = getattr(student, "swa_levels", frozenset())
    for lvl_i, cell in enumerate(getattr(student, "cells", ())):
        if include_ffn and hasattr(cell, "ffn"):
            for name in ("gate", "up", "down"):
                layer = getattr(cell.ffn, name, None)
                if isinstance(layer, nn.Linear):
                    w = LoRALinear(layer, rank=rank, alpha=alpha)
                    setattr(cell.ffn, name, w)
                    wrapped += 1
                    lora_params += w.lora_a.numel() + w.lora_b.numel()
        if include_swa and lvl_i in swa_lvls:
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                layer = getattr(cell, name, None)
                if isinstance(layer, nn.Linear):
                    w = LoRALinear(layer, rank=rank, alpha=alpha)
                    setattr(cell, name, w)
                    wrapped += 1
                    lora_params += w.lora_a.numel() + w.lora_b.numel()
    return {"wrapped_layers": wrapped, "lora_params": lora_params}


@torch.no_grad()
def merge_and_remove_lora(student: nn.Module) -> int:
    """Fold all LoRALinear adapters back into their underlying nn.Linear weights with zero inference overhead."""
    merged = 0

    def _walk(parent: nn.Module) -> None:
        nonlocal merged
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge_into_base())
                merged += 1
            else:
                _walk(child)

    _walk(student)
    return merged


def _forward_teacher(
    teacher: nn.Module,
    input_ids: torch.Tensor,
    *,
    need_layer_states: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    """Unified teacher forward returning (logits, final_hidden, layer_outputs, layer_inputs, attn_deltas)."""
    u_captured: list[torch.Tensor] = []
    hooks = []
    if need_layer_states:
        t_layers_list, _ = _extract_teacher_layers_and_norm(teacher)
        for tl in t_layers_list:
            fnorm = getattr(tl, "post_attention_layernorm", None) or getattr(tl, "ffn_norm", None)
            if fnorm is not None:
                hooks.append(fnorm.register_forward_pre_hook(lambda _m, args: u_captured.append(args[0].detach())))
    try:
        try:
            out = teacher(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        except TypeError:
            try:
                out = teacher(input_ids, return_layer_states=need_layer_states)
            except TypeError:
                out = teacher(input_ids)
    finally:
        for h in hooks:
            h.remove()
    logits = out.logits
    hidden = None
    layers_h: tuple[torch.Tensor, ...] = ()
    layer_inputs: tuple[torch.Tensor, ...] = ()
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        if isinstance(out.hidden_states, (tuple, list)):
            hidden = out.hidden_states[-1]
            # HF hidden_states = (embed_out, layer_0, ..., layer_{L-1})
            if len(out.hidden_states) > 1:
                layers_h = tuple(out.hidden_states[1:])
                layer_inputs = tuple(out.hidden_states[:-1])
        else:
            hidden = out.hidden_states
    if not layers_h and getattr(out, "layer_states", ()):
        layers_h = tuple(out.layer_states)
        if hasattr(teacher, "embedding"):
            h0 = teacher.embedding(input_ids)
            layer_inputs = (h0, *layers_h[:-1])
    attn_deltas: tuple[torch.Tensor, ...] = ()
    if u_captured and len(u_captured) == len(layer_inputs):
        attn_deltas = tuple(u.float() - hin.float() for u, hin in zip(u_captured, layer_inputs))
    return logits, hidden, layers_h, layer_inputs, attn_deltas


def _repr_alignment_loss(s_h: torch.Tensor, t_h: torch.Tensor) -> torch.Tensor:
    """Dimension-invariant cosine distance + Smooth-L1 log-norm loss (immune to attention-sink outlier explosion)."""
    cos_dist = (1.0 - F.cosine_similarity(s_h, t_h, dim=-1, eps=1e-6)).mean()
    s_log_n = s_h.norm(dim=-1).clamp_min(1e-4).log()
    t_log_n = t_h.norm(dim=-1).clamp_min(1e-4).log()
    log_norm_loss = F.smooth_l1_loss(s_log_n, t_log_n)
    return cos_dist + 0.25 * log_norm_loss


def _attn_delta_loss(s_delta: torch.Tensor, t_delta: torch.Tensor) -> torch.Tensor:
    """Direct RADLADS attention-delta alignment loss matching both cosine direction and relative L2 magnitude."""
    cos_l = (1.0 - F.cosine_similarity(s_delta, t_delta, dim=-1, eps=1e-6)).mean()
    mse_l = ((s_delta - t_delta).square().mean(dim=-1) / t_delta.square().mean(dim=-1).clamp_min(1e-4)).mean()
    return 0.5 * (cos_l + mse_l)


@torch.no_grad()
def calibrate_surgery_attention_scales(
    teacher: nn.Module,
    student: nn.Module,
    calib_input_ids: torch.Tensor,
) -> None:
    """Calibrate per-layer head_gamma on 1 batch so Step-0 student attention update norms match teacher attention norms."""
    if not hasattr(student, "forward_level_attn_delta") or not hasattr(student, "cells"):
        return
    _, _, _, t_inputs, t_attn_deltas = _forward_teacher(teacher, calib_input_ids, need_layer_states=True)
    if not t_inputs or not t_attn_deltas:
        return
    n_t = len(t_inputs)
    n_s = getattr(student, "depth", len(student.cells))
    swa_lvls = getattr(student, "swa_levels", frozenset())
    anc = t_inputs[0].float()
    for lvl in range(n_s):
        if lvl in swa_lvls:
            continue
        t_idx = 0 if n_s == 1 else round(lvl * (n_t - 1) / max(n_s - 1, 1))
        hin = t_inputs[t_idx].float()
        t_attn = t_attn_deltas[t_idx].float()
        s_attn = student.forward_level_attn_delta(lvl, hin, anc).float()
        ratio = (t_attn.norm(dim=-1).mean() / s_attn.norm(dim=-1).mean().clamp_min(1e-6)).item()
        cos_a = F.cosine_similarity(s_attn, t_attn, dim=-1, eps=1e-6).mean().item()
        scale = max(0.02, min(10.0, ratio * max(0.25, cos_a)))
        cell = student.cells[student.cell_index(lvl)] if hasattr(student, "cell_index") else student.cells[lvl]
        if hasattr(cell, "head_gamma"):
            cell.head_gamma.mul_(scale)


def _chunked_ce_kl_loss(
    s_hidden_flat: torch.Tensor,
    s_lm_head: nn.Module,
    t_logits_flat: torch.Tensor,
    targets_flat: torch.Tensor,
    *,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    chunk_tokens: int,
) -> tuple[torch.Tensor, float, float]:
    """Exact FP32 CE + KL loss computed in token chunks with checkpointing to cap peak large-vocab VRAM."""
    n_tok = int(s_hidden_flat.shape[0])
    total_valid = int((targets_flat != -100).sum().item())
    if total_valid <= 0:
        z = s_hidden_flat.sum() * 0.0
        return z, 0.0, 0.0
    inv_t = 1.0 / float(temperature)
    t_sq = float(temperature) ** 2
    v_common = min(int(s_lm_head.weight.shape[0]), int(t_logits_flat.shape[-1]))
    ce_acc = s_hidden_flat.new_zeros((), dtype=torch.float32)
    kl_acc = s_hidden_flat.new_zeros((), dtype=torch.float32)

    def _chunk_fn(h_c: torch.Tensor, t_c: torch.Tensor, y_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_log = s_lm_head(h_c).float()
        ce_c = F.cross_entropy(s_log, y_c, ignore_index=-100, reduction="sum")
        vm = y_c != -100
        if kl_weight > 0.0 and vm.any():
            log_ps = F.log_softmax(s_log[:, :v_common][vm] * inv_t, dim=-1)
            log_pt = F.log_softmax(t_c[:, :v_common][vm].float() * inv_t, dim=-1)
            kl_c = F.kl_div(log_ps, log_pt, reduction="sum", log_target=True)
        else:
            kl_c = s_log.new_zeros(())
        return ce_c, kl_c

    use_ckpt = torch.is_grad_enabled() and s_hidden_flat.requires_grad
    for start in range(0, n_tok, chunk_tokens):
        end = min(start + chunk_tokens, n_tok)
        h_c = s_hidden_flat[start:end]
        t_c = t_logits_flat[start:end].detach()
        y_c = targets_flat[start:end]
        if use_ckpt:
            ce_c, kl_c = grad_checkpoint(_chunk_fn, h_c, t_c, y_c, use_reentrant=False)
        else:
            ce_c, kl_c = _chunk_fn(h_c, t_c, y_c)
        ce_acc = ce_acc + ce_c
        kl_acc = kl_acc + kl_c

    ce_mean = ce_acc / float(total_valid)
    kl_mean = (kl_acc / float(total_valid)) * t_sq
    total = ce_weight * ce_mean + (kl_weight * kl_mean if kl_weight > 0.0 else 0.0)
    return total, float(ce_mean.detach().item()), float(kl_mean.detach().item())


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
    amp_dtype: torch.dtype = torch.bfloat16,
    teacher_forced: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute FP32 RADLADS/LoopUS distillation loss (Subfase 1 attention-delta or Subfase 2 E2E KL+CE+layer_mse)."""
    device_type = input_ids.device.type
    need_layers = bool(cfg.layer_mse_weight > 0.0 or teacher_forced)
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            t_logits, t_hidden, t_layers, t_inputs, t_attn_deltas = _forward_teacher(
                teacher, input_ids, need_layer_states=need_layers
            )

    # Subfase 1 (RADLADS parallel attention-delta alignment on DeltaNet levels only, skipping frozen FFNs & exact SWA layers)
    if (
        teacher_forced
        and t_inputs
        and t_attn_deltas
        and hasattr(student, "forward_level_attn_delta")
        and int(getattr(student.config, "d_model", -1)) == int(t_inputs[0].shape[-1])
    ):
        n_t = len(t_inputs)
        n_s = getattr(student, "depth", len(getattr(student, "cells", ())))
        swa_lvls = getattr(student, "swa_levels", frozenset())
        anc = t_inputs[0].detach().float()
        delta_losses = []
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            for lvl in range(n_s):
                if lvl in swa_lvls:
                    continue
                t_idx = 0 if n_s == 1 else round(lvl * (n_t - 1) / max(n_s - 1, 1))
                hin = t_inputs[t_idx].detach().float()
                t_delta = t_attn_deltas[t_idx].detach().float()
                s_delta = student.forward_level_attn_delta(lvl, hin, anc).float()
                delta_losses.append(_attn_delta_loss(s_delta, t_delta))
        d_loss = torch.stack(delta_losses).mean()
        val_d = float(d_loss.detach().item())
        return d_loss, {
            "ce_loss": val_d,
            "kl_loss": val_d,
            "hidden_mse": val_d,
            "layer_mse": val_d,
            "total_loss": val_d,
        }

    use_chunked_logits = bool(
        cfg.logit_chunk_tokens > 0
        and cfg.exit_gate_weight == 0.0
        and hasattr(student, "lm_head")
        and getattr(student, "supports_hidden_only", False)
    )
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        try:
            s_out = student(input_ids, return_logits=not use_chunked_logits, return_layer_states=need_layers)
        except TypeError:
            use_chunked_logits = False
            try:
                s_out = student(input_ids, return_layer_states=need_layers)
            except TypeError:
                s_out = student(input_ids)

    valid_mask = targets != -100
    metrics: dict[str, float] = {}
    if use_chunked_logits and s_out.hidden_states is not None:
        total, ce_val, kl_val = _chunked_ce_kl_loss(
            s_out.hidden_states.flatten(0, 1),
            student.lm_head,
            t_logits.detach().flatten(0, 1),
            targets.flatten(),
            temperature=cfg.temperature,
            ce_weight=cfg.ce_weight,
            kl_weight=cfg.kl_weight,
            chunk_tokens=cfg.logit_chunk_tokens,
        )
        metrics["ce_loss"] = ce_val
        if cfg.kl_weight > 0.0:
            metrics["kl_loss"] = kl_val
        s_logits_f = None
    else:
        s_logits_f = s_out.logits.float()
        t_logits_f = t_logits.detach().float()
        v_common = min(s_logits_f.shape[-1], t_logits_f.shape[-1])

        # 1. Hard-target CE in FP32
        ce_loss = F.cross_entropy(s_logits_f.flatten(0, 1), targets.flatten(), ignore_index=-100)
        total = cfg.ce_weight * ce_loss
        metrics["ce_loss"] = float(ce_loss.detach().item())

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

    # 3. Hidden-state representation alignment (dimension-invariant cosine + scale-matched L2)
    if cfg.hidden_mse_weight > 0.0 and t_hidden is not None and s_out.hidden_states is not None:
        s_h = s_out.hidden_states.float()
        t_h = t_hidden.detach().float()
        if hidden_adapter is not None:
            t_h = hidden_adapter(t_h.to(dtype=next(hidden_adapter.parameters()).dtype)).float()
        if t_h.shape == s_h.shape:
            hid_loss = _repr_alignment_loss(s_h, t_h)
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
                layer_losses.append(_repr_alignment_loss(sh_f, th))
        if layer_losses:
            l_mse = torch.stack(layer_losses).mean()
            total = total + cfg.layer_mse_weight * l_mse
            metrics["layer_mse"] = float(l_mse.detach().item())

    # 4. Self-distilled Exit Gate supervision
    if (
        cfg.exit_gate_weight > 0.0
        and s_logits_f is not None
        and getattr(s_out, "exit_logits", ())
        and getattr(s_out, "intermediate_states", ())
        and valid_mask.any()
    ):
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
        apply_surgery_freeze(
            student,
            freeze_ffn=cfg.freeze_ffn,
            freeze_embeddings=cfg.freeze_embeddings,
            freeze_swa=cfg.freeze_swa,
        )
        if (cfg.freeze_ffn or cfg.freeze_embeddings or cfg.freeze_swa)
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
    use_amp = cfg.precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.precision == "fp16" and device.type == "cuda"))

    it = iter(batch_iter)
    history: list[dict] = []
    tokens_total = 0
    t0 = time.perf_counter()

    tf_steps = int(cfg.max_steps * max(0.0, min(1.0, float(cfg.teacher_forced_ratio)))) if cfg.surgery_transplant else 0
    calibrated_scales = False

    for step in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_metrics: dict[str, float] = {}
        use_tf_step = bool(step < tf_steps)
        for _ in range(cfg.grad_accum_steps):
            bx, by = next(it)
            bx = bx.to(device=device)
            by = by.to(device=device)
            if cfg.surgery_transplant and not calibrated_scales:
                calibrate_surgery_attention_scales(teacher, student, bx)
                calibrated_scales = True
            loss, m = compute_distill_loss(
                teacher,
                student,
                bx,
                by,
                cfg,
                hidden_adapter=hidden_adapter,
                mtp_heads=mtp_heads,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                teacher_forced=use_tf_step,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite distillation loss at step {step}")
            scaler.scale(loss / float(cfg.grad_accum_steps)).backward()
            tokens_total += int((by != -100).sum().item())
            for k, v in m.items():
                step_metrics[k] = step_metrics.get(k, 0.0) + v / float(cfg.grad_accum_steps)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unique_params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
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
