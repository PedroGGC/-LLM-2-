"""Fractal recursive Gated Delta memory with multi-timescale causal state,
lag-1 associative induction, optional value residual, and optional local window attention.
"""

from copy import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, SwiGLU
from .quarot import HadamardLinearINT4, pack_int4_signed, unpack_int4_signed
from .state_kernels import dplr_chunked, dplr_sequential


_GOLDEN_PHI = (1.0 + math.sqrt(5.0)) * 0.5


class TernaryLinear(nn.Module):
    """BitNet b1.58 ternary weight quantization {-1, 0, +1} with absmean scaling + STE."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def reset_weight(self, std: float) -> None:
        nn.init.normal_(self.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        scale = w.abs().mean().clamp_min(1e-5)
        w_q = (w / scale).round().clamp(-1.0, 1.0) * scale
        return F.linear(x, w + (w_q - w).detach(), self.bias)


class QuaternionLinear(nn.Module):
    """Hamilton quaternion linear transform sharing 4 component matrices (4x param compression)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if in_features % 4 != 0 or out_features % 4 != 0:
            raise ValueError("quaternion weight mode requires dimensions divisible by 4")
        self.in_features = in_features
        self.out_features = out_features
        self.components = nn.Parameter(torch.empty(4, out_features // 4, in_features // 4))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.components, std=0.02)

    def reset_weight(self, std: float) -> None:
        nn.init.normal_(self.components, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wr, wi, wj, wk = self.components.unbind(0)
        row0 = torch.cat((wr, -wi, -wj, -wk), dim=1)
        row1 = torch.cat((wi, wr, -wk, wj), dim=1)
        row2 = torch.cat((wj, wk, wr, -wi), dim=1)
        row3 = torch.cat((wk, -wj, wi, wr), dim=1)
        weight = torch.cat((row0, row1, row2, row3), dim=0)
        return F.linear(x, weight, self.bias)


def _make_linear(in_features: int, out_features: int, mode: str, *, bias: bool = False) -> nn.Module:
    if mode == "ternary":
        return TernaryLinear(in_features, out_features, bias=bias)
    if mode == "quaternion":
        return QuaternionLinear(in_features, out_features, bias=bias)
    if mode == "quarot_int4":
        return HadamardLinearINT4(in_features, out_features, bias=bias, qat=True)
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(layer.weight, std=0.02)
    return layer


def _apply_golden_phase(tensor: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply norm-preserving 2D phase rotation on key/query pairs."""
    pairs = angles.shape[-1]
    rot_dim = pairs * 2
    rot = tensor[..., :rot_dim]
    x1 = rot[..., 0::2]
    x2 = rot[..., 1::2]
    cos = angles.cos()
    sin = angles.sin()
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    if rot_dim == tensor.shape[-1]:
        return rotated
    return torch.cat((rotated, tensor[..., rot_dim:]), dim=-1)


def _apply_split_half_rope(tensor: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply standard HuggingFace/Llama split-half FP32 RoPE (norm-preserving)."""
    half = tensor.shape[-1] // 2
    x1 = tensor[..., :half]
    x2 = tensor[..., half:]
    cos = angles.cos()
    sin = angles.sin()
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _init_decay_bias(heads, min_tau=4.0, max_tau=4096.0):
    if heads == 1:
        taus = torch.tensor([max_tau], dtype=torch.float32)
    else:
        taus = torch.logspace(math.log10(min_tau), math.log10(max_tau), heads, dtype=torch.float32)
    rates = (-1.0 / taus).exp().clamp(1e-4, 1.0 - 1.0e-4)
    return torch.log(rates / (1.0 - rates))


def _assoc_sequential(q, k, v, prev_mem=None, *, return_all_states=False):
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = q.float(), k.float(), v.float()
        b, t, key_dim = q.shape
        val_dim = v.shape[-1]
        mem = q.new_zeros(b, key_dim, val_dim) if prev_mem is None else prev_mem.float()
        outputs = []
        states = [] if return_all_states else None
        for idx in range(t):
            mem = mem + k[:, idx].unsqueeze(2) * v[:, idx].unsqueeze(1)
            outputs.append(torch.bmm(q[:, idx].unsqueeze(1), mem).squeeze(1))
            if states is not None:
                states.append(mem)
        out = torch.stack(outputs, dim=1)
        if states is not None:
            return out, mem.contiguous(), torch.stack(states, dim=0).contiguous()
        return out, mem.contiguous()


def _assoc_chunked(q, k, v, prev_mem=None, chunk_size=32):
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = q.float(), k.float(), v.float()
        b, t, key_dim = q.shape
        val_dim = v.shape[-1]
        if t <= 2 or chunk_size <= 1:
            return _assoc_sequential(q, k, v, prev_mem)

        c = min(chunk_size, t)
        pad = (c - (t % c)) % c
        qf, kf, vf = q, k, v
        if pad:
            qf = F.pad(qf, (0, 0, 0, pad))
            kf = F.pad(kf, (0, 0, 0, pad))
            vf = F.pad(vf, (0, 0, 0, pad))

        n_chunks = qf.shape[1] // c
        qc = qf.reshape(b, n_chunks, c, key_dim)
        kc = kf.reshape(b, n_chunks, c, key_dim)
        vc = vf.reshape(b, n_chunks, c, val_dim)

        upper = torch.triu(torch.ones(c, c, device=q.device, dtype=torch.bool), diagonal=1)
        intra_scores = torch.matmul(qc, kc.transpose(-1, -2)).masked_fill(upper, 0.0)
        intra_out = torch.matmul(intra_scores, vc)
        chunk_deltas = torch.matmul(kc.transpose(-1, -2), vc)

        mem = q.new_zeros(b, key_dim, val_dim) if prev_mem is None else prev_mem.float()
        outputs = []
        for idx in range(n_chunks):
            outputs.append(torch.matmul(qc[:, idx], mem) + intra_out[:, idx])
            mem = mem + chunk_deltas[:, idx]

        out = torch.cat(outputs, dim=1)[:, :t].contiguous()
        return out, mem.contiguous()


@dataclass
class FractalLevelState:
    memory: torch.Tensor
    tail: torch.Tensor


@dataclass
class LocalWindowState:
    k_tail: torch.Tensor
    v_tail: torch.Tensor
    valid_tail: torch.Tensor


@dataclass
class SWAState:
    k_sinks: torch.Tensor
    v_sinks: torch.Tensor
    pos_sinks: torch.Tensor
    k_tail: torch.Tensor
    v_tail: torch.Tensor
    pos_tail: torch.Tensor
    seen: torch.Tensor
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None
    is_static: bool = False


def _quantize_kv_cache(t: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Symmetric per-token-head KV cache compression ('none'|'fp8'|'int4') for global/SWA layers."""
    if mode == "none" or t.shape[2] == 0:
        return t, None
    f = t.float()
    if mode == "fp8":
        scale = (f.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-6)
        q = (f / scale).round().clamp(-127.0, 127.0).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    if mode == "int4":
        scale = (f.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-6)
        q = (f / scale).round().clamp(-7.0, 7.0).to(torch.int8)
        return pack_int4_signed(q), scale.to(torch.float16).contiguous()
    raise ValueError(f"unsupported kv_quant mode: {mode}")


def _dequantize_kv_cache(q: torch.Tensor, scale: torch.Tensor | None, mode: str, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct KV cache back to target compute dtype before SDPA (exact int8/int4 -> compute dtype)."""
    if mode == "none" or scale is None or q.shape[2] == 0:
        return q if q.dtype == dtype else q.to(dtype)
    if mode == "fp8":
        return q.to(dtype) * scale.to(dtype)
    if mode == "int4":
        return unpack_int4_signed(q).to(dtype) * scale.to(dtype)
    raise ValueError(f"unsupported kv_quant mode: {mode}")


@dataclass
class FractalDeltaState:
    levels: tuple[FractalLevelState | SWAState, ...]
    assoc_memory: torch.Tensor
    local_tail: torch.Tensor
    window: LocalWindowState | None = None
    phase_pos: torch.Tensor | None = None
    hybrid_windows: tuple[LocalWindowState, ...] | None = None


class CausalLocalWindow(nn.Module):
    """Fixed-horizon causal sliding-window softmax attention with O(W) fixed state."""

    def __init__(self, config, window: int):
        super().__init__()
        d = config.d_model
        self.window = window
        self.heads = 2 if d < 64 else 4
        self.head_dim = max(4, (d // 2) // self.heads)
        self.win_dim = self.heads * self.head_dim
        self.dropout = config.dropout

        self.norm = RMSNorm(d, eps=config.norm_eps)
        self.qkv = nn.Linear(d, 3 * self.win_dim, bias=False)
        self.out_proj = nn.Linear(self.win_dim, d, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(self.heads, window))

        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    def validate_state(self, state: LocalWindowState, b: int, device: torch.device) -> None:
        if not isinstance(state, LocalWindowState):
            raise ValueError("invalid fractal delta state shape/dtype")
        w_tail = self.window - 1
        checks = (
            (state.k_tail, (b, self.heads, w_tail, self.head_dim)),
            (state.v_tail, (b, self.heads, w_tail, self.head_dim)),
            (state.valid_tail, (b, w_tail)),
        )
        for tensor, expected in checks:
            if tensor.shape != expected or tensor.dtype != torch.float32 or tensor.device != device:
                raise ValueError("invalid fractal delta state shape/dtype")

    def empty_state(self, b: int, device: torch.device) -> LocalWindowState:
        w_tail = self.window - 1
        return LocalWindowState(
            torch.zeros(b, self.heads, w_tail, self.head_dim, device=device, dtype=torch.float32),
            torch.zeros(b, self.heads, w_tail, self.head_dim, device=device, dtype=torch.float32),
            torch.zeros(b, w_tail, device=device, dtype=torch.float32),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: LocalWindowState | None = None,
        *,
        return_all_states: bool = False,
    ) -> tuple[torch.Tensor, LocalWindowState]:
        b, t, _ = x.shape
        w = self.window
        w_tail = w - 1
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = (tensor.transpose(1, 2).float() for tensor in (q, k, v))

            if state is None:
                k_tail = q.new_zeros(b, self.heads, w_tail, self.head_dim)
                v_tail = q.new_zeros(b, self.heads, w_tail, self.head_dim)
                valid_tail = q.new_zeros(b, w_tail)
            else:
                k_tail, v_tail, valid_tail = state.k_tail, state.v_tail, state.valid_tail

            bias = self.rel_bias.float()
            drop = self.dropout if self.training else 0.0

            if t == 1:
                k_full = torch.cat((k_tail, k), dim=2)
                v_full = torch.cat((v_tail, v), dim=2)
                valid_full = torch.cat((valid_tail, q.new_ones(b, 1)), dim=1)
                mask = torch.where(valid_full > 0.5, 0.0, float("-inf"))[:, None, None, :] + bias[None, :, None, :]
                attn = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, dropout_p=drop)
                nk = k_full[:, :, 1:].clone(memory_format=torch.contiguous_format)
                nv = v_full[:, :, 1:].clone(memory_format=torch.contiguous_format)
                nval = valid_full[:, 1:].clone(memory_format=torch.contiguous_format)
                if return_all_states:
                    next_state = LocalWindowState(nk.unsqueeze(0), nv.unsqueeze(0), nval.unsqueeze(0))
                else:
                    next_state = LocalWindowState(nk, nv, nval)
                attn_out = attn.transpose(1, 2).reshape(b, 1, self.win_dim).to(x.dtype)
                return self.out_proj(attn_out).to(x.dtype), next_state

            # Single vectorized O(T * W) banded causal sliding window (zero Python chunk loop).
            k_ctx = torch.cat((k_tail, k), dim=2)
            v_ctx = torch.cat((v_tail, v), dim=2)
            valid_ctx = torch.cat((valid_tail, q.new_ones(b, t)), dim=1)

            k_win = k_ctx.unfold(2, w, 1)  # (B, H, T, Dh, W)
            v_win = v_ctx.unfold(2, w, 1).transpose(-1, -2)  # (B, H, T, W, Dh)
            valid_win = valid_ctx.unfold(1, w, 1)  # (B, T, W)

            scale = self.head_dim ** -0.5
            scores = torch.matmul(q.unsqueeze(-2), k_win).squeeze(-2) * scale + bias[None, :, None, :]
            scores = scores.masked_fill(valid_win[:, None, :, :] <= 0.5, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            if drop > 0.0:
                probs = F.dropout(probs, p=drop, training=True)
            attn = torch.matmul(probs.unsqueeze(-2), v_win).squeeze(-2)  # (B, H, T, Dh)

            out = attn.transpose(1, 2).contiguous().reshape(b, t, self.win_dim).to(x.dtype)
            if return_all_states:
                k_steps = torch.stack([k_ctx[:, :, idx + 1 : idx + 1 + w_tail] for idx in range(t)], dim=0).contiguous()
                v_steps = torch.stack([v_ctx[:, :, idx + 1 : idx + 1 + w_tail] for idx in range(t)], dim=0).contiguous()
                val_steps = torch.stack([valid_ctx[:, idx + 1 : idx + 1 + w_tail] for idx in range(t)], dim=0).contiguous()
                next_state = LocalWindowState(k_steps, v_steps, val_steps)
            else:
                next_state = LocalWindowState(
                    k_ctx[:, :, -w_tail:].clone(memory_format=torch.contiguous_format),
                    v_ctx[:, :, -w_tail:].clone(memory_format=torch.contiguous_format),
                    valid_ctx[:, -w_tail:].clone(memory_format=torch.contiguous_format),
                )
        return self.out_proj(out).to(x.dtype), next_state

    def parameter_count(self, d: int) -> int:
        return d + 3 * d * self.win_dim + self.win_dim * d + self.heads * self.window

    def state_floats_per_seq(self) -> int:
        w_tail = self.window - 1
        return 2 * self.heads * w_tail * self.head_dim + w_tail


class SlidingWindowGQACell(nn.Module):
    """Full-width GQA Transformer block with bounded O(1) Sliding-Window+Sinks OR Global Full Attention (NoPE/RoPE, optional QK-Norm & KV INT4/FP8)."""

    def __init__(self, config, research):
        super().__init__()
        self.d_model = config.d_model
        self.heads = config.n_heads
        self.head_dim = research.fractal_head_dim if research.fractal_head_dim > 0 else (self.d_model // self.heads)
        self.attn_dim = self.heads * self.head_dim
        kv_heads = research.fractal_swa_kv_heads if research.fractal_swa_kv_heads > 0 else self.heads
        if self.heads % kv_heads != 0:
            raise ValueError(f"n_heads ({self.heads}) must be divisible by fractal_swa_kv_heads ({kv_heads})")
        self.kv_heads = kv_heads
        self.kv_groups = self.heads // self.kv_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.hidden_dim = config.hidden_dim
        self.global_attn = bool(research.fractal_swa_global)
        self.window = 0 if self.global_attn else research.fractal_window
        self.sink_tokens = 0 if self.global_attn else research.fractal_swa_sinks
        self.use_rope = research.fractal_swa_rope
        self.qk_norm = bool(research.fractal_qk_norm)
        self.kv_quant = str(research.fractal_kv_quant)
        self.dropout = config.dropout
        self.weight_mode = research.fractal_weight_mode

        self.attn_norm = RMSNorm(self.d_model, eps=config.norm_eps)
        self.q_proj = _make_linear(self.d_model, self.attn_dim, self.weight_mode)
        self.k_proj = _make_linear(self.d_model, self.kv_dim, self.weight_mode)
        self.v_proj = _make_linear(self.d_model, self.kv_dim, self.weight_mode)
        self.out_proj = _make_linear(self.attn_dim, self.d_model, self.weight_mode)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if self.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if self.qk_norm else None
        self.ffn_norm = RMSNorm(self.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(self.d_model, self.hidden_dim)
        if self.weight_mode != "dense":
            self.ffn.gate = _make_linear(self.d_model, self.hidden_dim, self.weight_mode)
            self.ffn.up = _make_linear(self.d_model, self.hidden_dim, self.weight_mode)
            self.ffn.down = _make_linear(self.hidden_dim, self.d_model, self.weight_mode)

        if self.use_rope:
            inv_freq = 1.0 / (
                float(config.rope_theta) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / float(self.head_dim))
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        proj_std = 0.02 / math.sqrt(2 * config.n_layers)
        _init_linear_weight(self.q_proj, 0.02)
        _init_linear_weight(self.k_proj, 0.02)
        _init_linear_weight(self.v_proj, 0.02)
        _init_linear_weight(self.out_proj, proj_std)
        _init_linear_weight(self.ffn.gate, 0.02)
        _init_linear_weight(self.ffn.up, 0.02)
        _init_linear_weight(self.ffn.down, proj_std)

    def _stored_dim(self) -> int:
        return self.head_dim // 2 if self.kv_quant == "int4" else self.head_dim

    def _stored_dtype(self, compute_dtype: torch.dtype) -> torch.dtype:
        if self.kv_quant == "int4":
            return torch.uint8
        if self.kv_quant == "fp8":
            return torch.int8
        return compute_dtype

    def empty_state(self, b: int, device: torch.device, dtype: torch.dtype) -> SWAState:
        w_tail = 0 if self.global_attn else (self.window - 1)
        s = self.sink_tokens
        sd = self._stored_dim()
        sdt = self._stored_dtype(dtype)
        k_sc = torch.ones(b, self.kv_heads, w_tail, 1, device=device, dtype=torch.float16) if self.kv_quant != "none" else None
        v_sc = torch.ones(b, self.kv_heads, w_tail, 1, device=device, dtype=torch.float16) if self.kv_quant != "none" else None
        return SWAState(
            torch.zeros(b, self.kv_heads, s, self.head_dim, device=device, dtype=dtype),
            torch.zeros(b, self.kv_heads, s, self.head_dim, device=device, dtype=dtype),
            torch.full((b, s), -1, device=device, dtype=torch.int64),
            torch.zeros(b, self.kv_heads, w_tail, sd, device=device, dtype=sdt),
            torch.zeros(b, self.kv_heads, w_tail, sd, device=device, dtype=sdt),
            torch.full((b, w_tail), -1, device=device, dtype=torch.int64),
            torch.zeros((b,), device=device, dtype=torch.int64),
            k_sc,
            v_sc,
        )

    def validate_state(self, state: SWAState, b: int, device: torch.device) -> None:
        if not isinstance(state, SWAState):
            raise ValueError("invalid fractal delta state shape/dtype")
        s = self.sink_tokens
        sd = self._stored_dim()
        tail_len = state.k_tail.shape[2] if self.global_attn else (self.window - 1)
        checks = (
            (state.k_sinks, (b, self.kv_heads, s, self.head_dim)),
            (state.v_sinks, (b, self.kv_heads, s, self.head_dim)),
            (state.k_tail, (b, self.kv_heads, tail_len, sd)),
            (state.v_tail, (b, self.kv_heads, tail_len, sd)),
        )
        for tensor, expected in checks:
            if tensor.shape != expected or tensor.device != device:
                raise ValueError("invalid fractal delta state shape/dtype")
        if self.kv_quant == "none" and (not state.k_tail.is_floating_point() or not state.v_tail.is_floating_point()):
            raise ValueError("invalid fractal delta state shape/dtype")
        int_checks = (
            (state.pos_sinks, (b, s)),
            (state.pos_tail, (b, tail_len)),
            (state.seen, (b,)),
        )
        for tensor, expected in int_checks:
            if tensor.shape != expected or tensor.dtype != torch.int64 or tensor.device != device:
                raise ValueError("invalid fractal delta state shape/dtype")

    def _update_state_single(self, st: SWAState, k: torch.Tensor, v: torch.Tensor, new_pos: torch.Tensor) -> SWAState:
        """Sync-free 1-token SWA+Sink (static CUDAGraph-safe) or Global KV append."""
        kq, ks = _quantize_kv_cache(k, self.kv_quant)
        vq, vs = _quantize_kv_cache(v, self.kv_quant)
        if self.global_attn:
            nk_tail = torch.cat((st.k_tail, kq), dim=2).contiguous()
            nv_tail = torch.cat((st.v_tail, vq), dim=2).contiguous()
            np_tail = torch.cat((st.pos_tail, new_pos), dim=1).contiguous()
            nks = None if self.kv_quant == "none" else torch.cat((st.k_scale, ks), dim=2).contiguous()
            nvs = None if self.kv_quant == "none" else torch.cat((st.v_scale, vs), dim=2).contiguous()
            return SWAState(st.k_sinks, st.v_sinks, st.pos_sinks, nk_tail, nv_tail, np_tail, st.seen + 1, nks, nvs)
        s = self.sink_tokens
        p = new_pos[:, 0]
        if s > 0:
            slot = p.clamp(0, s - 1)
            is_sink = p < s
            sink_mask = (torch.arange(s, device=k.device)[None, :] == slot[:, None]) & is_sink[:, None]
            m4 = sink_mask[:, None, :, None]
            nk_sinks = torch.where(m4, k, st.k_sinks)
            nv_sinks = torch.where(m4, v, st.v_sinks)
            np_sinks = torch.where(sink_mask, p[:, None], st.pos_sinks)
            is_tail = p >= s
        else:
            nk_sinks, nv_sinks, np_sinks = st.k_sinks, st.v_sinks, st.pos_sinks
            is_tail = torch.ones_like(p, dtype=torch.bool)
        sh_k = torch.cat((st.k_tail[:, :, 1:, :], kq), dim=2)
        sh_v = torch.cat((st.v_tail[:, :, 1:, :], vq), dim=2)
        sh_p = torch.cat((st.pos_tail[:, 1:], new_pos), dim=1)
        t4 = is_tail[:, None, None, None]
        nk_tail = torch.where(t4, sh_k, st.k_tail)
        nv_tail = torch.where(t4, sh_v, st.v_tail)
        np_tail = torch.where(is_tail[:, None], sh_p, st.pos_tail)
        if self.kv_quant != "none":
            nks = torch.where(t4, torch.cat((st.k_scale[:, :, 1:, :], ks), dim=2), st.k_scale)
            nvs = torch.where(t4, torch.cat((st.v_scale[:, :, 1:, :], vs), dim=2), st.v_scale)
        else:
            nks, nvs = None, None
        return SWAState(nk_sinks, nv_sinks, np_sinks, nk_tail, nv_tail, np_tail, st.seen + 1, nks, nvs)

    def _update_state_block(
        self, st: SWAState, k: torch.Tensor, v: torch.Tensor, new_pos: torch.Tensor, *, return_all_states: bool = False
    ) -> SWAState:
        b, _, t, _ = k.shape
        if t == 1:
            nxt = self._update_state_single(st, k, v, new_pos)
            if not return_all_states:
                return nxt
            return SWAState(
                nxt.k_sinks.unsqueeze(0),
                nxt.v_sinks.unsqueeze(0),
                nxt.pos_sinks.unsqueeze(0),
                nxt.k_tail.unsqueeze(0),
                nxt.v_tail.unsqueeze(0),
                nxt.pos_tail.unsqueeze(0),
                nxt.seen.unsqueeze(0),
                None if nxt.k_scale is None else nxt.k_scale.unsqueeze(0),
                None if nxt.v_scale is None else nxt.v_scale.unsqueeze(0),
            )
        if return_all_states:
            cur = st
            steps = []
            for idx in range(t):
                cur = self._update_state_single(cur, k[:, :, idx : idx + 1, :], v[:, :, idx : idx + 1, :], new_pos[:, idx : idx + 1])
                steps.append(cur)
            if self.global_attn:
                # Pad growing global states to final length for uniform (T, ...) stack indexing if needed.
                max_len = steps[-1].k_tail.shape[2]
                pad_k, pad_v, pad_p, pad_ks, pad_vs = [], [], [], [], []
                for s_i in steps:
                    d_len = max_len - s_i.k_tail.shape[2]
                    if d_len > 0:
                        pad_k.append(F.pad(s_i.k_tail, (0, 0, 0, d_len)))
                        pad_v.append(F.pad(s_i.v_tail, (0, 0, 0, d_len)))
                        pad_p.append(F.pad(s_i.pos_tail, (0, d_len), value=-1))
                        if s_i.k_scale is not None:
                            pad_ks.append(F.pad(s_i.k_scale, (0, 0, 0, d_len), value=1.0))
                            pad_vs.append(F.pad(s_i.v_scale, (0, 0, 0, d_len), value=1.0))
                    else:
                        pad_k.append(s_i.k_tail)
                        pad_v.append(s_i.v_tail)
                        pad_p.append(s_i.pos_tail)
                        if s_i.k_scale is not None:
                            pad_ks.append(s_i.k_scale)
                            pad_vs.append(s_i.v_scale)
                return SWAState(
                    torch.stack([s_i.k_sinks for s_i in steps], dim=0).contiguous(),
                    torch.stack([s_i.v_sinks for s_i in steps], dim=0).contiguous(),
                    torch.stack([s_i.pos_sinks for s_i in steps], dim=0).contiguous(),
                    torch.stack(pad_k, dim=0).contiguous(),
                    torch.stack(pad_v, dim=0).contiguous(),
                    torch.stack(pad_p, dim=0).contiguous(),
                    torch.stack([s_i.seen for s_i in steps], dim=0).contiguous(),
                    torch.stack(pad_ks, dim=0).contiguous() if pad_ks else None,
                    torch.stack(pad_vs, dim=0).contiguous() if pad_vs else None,
                )
            return SWAState(
                torch.stack([s_i.k_sinks for s_i in steps], dim=0).contiguous(),
                torch.stack([s_i.v_sinks for s_i in steps], dim=0).contiguous(),
                torch.stack([s_i.pos_sinks for s_i in steps], dim=0).contiguous(),
                torch.stack([s_i.k_tail for s_i in steps], dim=0).contiguous(),
                torch.stack([s_i.v_tail for s_i in steps], dim=0).contiguous(),
                torch.stack([s_i.pos_tail for s_i in steps], dim=0).contiguous(),
                torch.stack([s_i.seen for s_i in steps], dim=0).contiguous(),
                torch.stack([s_i.k_scale for s_i in steps], dim=0).contiguous() if steps[0].k_scale is not None else None,
                torch.stack([s_i.v_scale for s_i in steps], dim=0).contiguous() if steps[0].v_scale is not None else None,
            )
        kq, ks = _quantize_kv_cache(k, self.kv_quant)
        vq, vs = _quantize_kv_cache(v, self.kv_quant)
        if self.global_attn:
            nk_tail = torch.cat((st.k_tail, kq), dim=2).contiguous()
            nv_tail = torch.cat((st.v_tail, vq), dim=2).contiguous()
            np_tail = torch.cat((st.pos_tail, new_pos), dim=1).contiguous()
            nks = None if self.kv_quant == "none" else torch.cat((st.k_scale, ks), dim=2).contiguous()
            nvs = None if self.kv_quant == "none" else torch.cat((st.v_scale, vs), dim=2).contiguous()
            return SWAState(st.k_sinks, st.v_sinks, st.pos_sinks, nk_tail, nv_tail, np_tail, (st.seen + t).contiguous(), nks, nvs)
        s = self.sink_tokens
        w_tail = self.window - 1
        if s > 0:
            s_arange = torch.arange(s, device=k.device, dtype=torch.int64)[None, :, None]
            eq = new_pos[:, None, :] == s_arange  # (B, S, T)
            has_sink = eq.any(dim=-1)  # (B, S)
            tok_idx = eq.long().argmax(dim=-1)  # (B, S)
            gather_idx = tok_idx[:, None, :, None].expand(b, self.kv_heads, s, self.head_dim)
            k_g = torch.gather(k, dim=2, index=gather_idx)
            v_g = torch.gather(v, dim=2, index=gather_idx)
            p_g = torch.gather(new_pos, dim=1, index=tok_idx)
            m4 = has_sink[:, None, :, None]
            nk_sinks = torch.where(m4, k_g, st.k_sinks).contiguous()
            nv_sinks = torch.where(m4, v_g, st.v_sinks).contiguous()
            np_sinks = torch.where(has_sink, p_g, st.pos_sinks).contiguous()
        else:
            nk_sinks, nv_sinks, np_sinks = st.k_sinks, st.v_sinks, st.pos_sinks
        masked_new_pos = torch.where(new_pos >= s, new_pos, torch.full_like(new_pos, -1))
        nk_tail = torch.cat((st.k_tail, kq), dim=2)[:, :, -w_tail:, :].contiguous()
        nv_tail = torch.cat((st.v_tail, vq), dim=2)[:, :, -w_tail:, :].contiguous()
        np_tail = torch.cat((st.pos_tail, masked_new_pos), dim=1)[:, -w_tail:].contiguous()
        nks = None if self.kv_quant == "none" else torch.cat((st.k_scale, ks), dim=2)[:, :, -w_tail:, :].contiguous()
        nvs = None if self.kv_quant == "none" else torch.cat((st.v_scale, vs), dim=2)[:, :, -w_tail:, :].contiguous()
        return SWAState(nk_sinks, nv_sinks, np_sinks, nk_tail, nv_tail, np_tail, (st.seen + t).contiguous(), nks, nvs)

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kv_groups == 1:
            return k, v
        return k.repeat_interleave(self.kv_groups, dim=1), v.repeat_interleave(self.kv_groups, dim=1)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, attn_mask=None, is_causal: bool = False, dropout_p: float = 0.0) -> torch.Tensor:
        if self.kv_groups == 1:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p)
        try:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p, enable_gqa=True
            )
        except TypeError:
            k_exp, v_exp = self._expand_kv(k, v)
            return F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p)

    def pad_state_static(self, st: SWAState, capacity: int) -> SWAState:
        """Preallocate fixed-capacity static KV buffers for Global attention (CUDAGraph-safe decode)."""
        if not self.global_attn:
            return st
        cur_len = st.k_tail.shape[2]
        cap = max(int(capacity), cur_len)
        if st.is_static and cur_len == cap:
            return st
        b = st.k_tail.shape[0]
        device = st.k_tail.device
        sd = self._stored_dim()
        sdt = st.k_tail.dtype
        k_buf = torch.zeros(b, self.kv_heads, cap, sd, device=device, dtype=sdt)
        v_buf = torch.zeros(b, self.kv_heads, cap, sd, device=device, dtype=sdt)
        pos_buf = torch.full((b, cap), -1, device=device, dtype=torch.int64)
        k_sc = torch.ones(b, self.kv_heads, cap, 1, device=device, dtype=torch.float16) if self.kv_quant != "none" else None
        v_sc = torch.ones(b, self.kv_heads, cap, 1, device=device, dtype=torch.float16) if self.kv_quant != "none" else None
        if cur_len > 0:
            k_buf[:, :, :cur_len, :].copy_(st.k_tail)
            v_buf[:, :, :cur_len, :].copy_(st.v_tail)
            pos_buf[:, :cur_len].copy_(st.pos_tail)
            if k_sc is not None and st.k_scale is not None:
                k_sc[:, :, :cur_len, :].copy_(st.k_scale)
                v_sc[:, :, :cur_len, :].copy_(st.v_scale)
        return SWAState(
            st.k_sinks,
            st.v_sinks,
            st.pos_sinks,
            k_buf,
            v_buf,
            pos_buf,
            st.seen.clone(),
            k_sc,
            v_sc,
            is_static=True,
        )

    def forward_attn_delta(
        self,
        x: torch.Tensor,
        state: SWAState | None = None,
        adapter=None,
        *,
        return_all_states: bool = False,
    ) -> tuple[torch.Tensor, SWAState]:
        b, t, _ = x.shape
        normed = self.attn_norm(x)
        if adapter is not None:
            normed = normed * adapter.attn_scale + adapter.attn_shift
        q = self.q_proj(normed).reshape(b, t, self.heads, self.head_dim)
        k = self.k_proj(normed).reshape(b, t, self.kv_heads, self.head_dim)
        v = self.v_proj(normed).reshape(b, t, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        st = self.empty_state(b, x.device, k.dtype) if state is None else state
        new_pos = st.seen[:, None] if t == 1 else (st.seen[:, None] + torch.arange(t, device=x.device, dtype=torch.int64)[None, :])
        if self.use_rope:
            with torch.autocast(device_type=x.device.type, enabled=False):
                angles = (new_pos[:, None, :, None].double() * self.rope_inv_freq.double().view(1, 1, 1, -1)).float()
                q = _apply_split_half_rope(q.float(), angles).to(k.dtype)
                k = _apply_split_half_rope(k.float(), angles).to(k.dtype)

        drop_p = self.dropout if self.training else 0.0
        if st.is_static and self.global_attn and not return_all_states:
            cap = st.k_tail.shape[2]
            kq, ks = _quantize_kv_cache(k, self.kv_quant)
            vq, vs = _quantize_kv_cache(v, self.kv_quant)
            if t == 1:
                # Sync-free, zero-allocation in-place static Global KV update (CUDAGraph-safe).
                slot = st.seen.clamp(0, cap - 1)
                idx_kv = slot.view(b, 1, 1, 1).expand(b, self.kv_heads, 1, self._stored_dim())
                st.k_tail.scatter_(2, idx_kv, kq)
                st.v_tail.scatter_(2, idx_kv, vq)
                st.pos_tail.scatter_(1, slot[:, None], new_pos)
                if self.kv_quant != "none":
                    idx_sc = slot.view(b, 1, 1, 1).expand(b, self.kv_heads, 1, 1)
                    st.k_scale.scatter_(2, idx_sc, ks)
                    st.v_scale.scatter_(2, idx_sc, vs)
                k_all = _dequantize_kv_cache(st.k_tail, st.k_scale, self.kv_quant, k.dtype)
                v_all = _dequantize_kv_cache(st.v_tail, st.v_scale, self.kv_quant, v.dtype)
                mask = (st.pos_tail >= 0)[:, None, None, :]
                attn = self._sdpa(q, k_all, v_all, attn_mask=mask, dropout_p=drop_p)
                st.seen.add_(1)
                next_state = st
            else:
                # Chunked prefill into preallocated static Global KV buffer (no O(L) cat reallocation).
                s0 = int(st.seen[0].item())
                if s0 == 0 and (b == 1 or bool((st.seen == s0).all().item())):
                    attn = self._sdpa(q, k, v, is_causal=True, dropout_p=drop_p)
                    end = min(s0 + t, cap)
                    w_len = end - s0
                    st.k_tail[:, :, s0:end, :].copy_(kq[:, :, :w_len, :])
                    st.v_tail[:, :, s0:end, :].copy_(vq[:, :, :w_len, :])
                    st.pos_tail[:, s0:end].copy_(new_pos[:, :w_len])
                    if self.kv_quant != "none":
                        st.k_scale[:, :, s0:end, :].copy_(ks[:, :, :w_len, :])
                        st.v_scale[:, :, s0:end, :].copy_(vs[:, :, :w_len, :])
                    st.seen.add_(t)
                    next_state = st
                elif b == 1 or bool((st.seen == s0).all().item()):
                    k_prev = _dequantize_kv_cache(
                        st.k_tail[:, :, :s0, :],
                        None if st.k_scale is None else st.k_scale[:, :, :s0, :],
                        self.kv_quant,
                        k.dtype,
                    )
                    v_prev = _dequantize_kv_cache(
                        st.v_tail[:, :, :s0, :],
                        None if st.v_scale is None else st.v_scale[:, :, :s0, :],
                        self.kv_quant,
                        v.dtype,
                    )
                    k_all = torch.cat((k_prev, k), dim=2)
                    v_all = torch.cat((v_prev, v), dim=2)
                    pos_all = torch.cat((st.pos_tail[:, :s0], new_pos), dim=1)
                    pq = new_pos[:, :, None]
                    pk = pos_all[:, None, :]
                    mask = ((pk >= 0) & (pk <= pq)).unsqueeze(1)
                    attn = self._sdpa(q, k_all, v_all, attn_mask=mask, dropout_p=drop_p)
                    end = min(s0 + t, cap)
                    w_len = max(0, end - s0)
                    if w_len > 0:
                        st.k_tail[:, :, s0:end, :].copy_(kq[:, :, :w_len, :])
                        st.v_tail[:, :, s0:end, :].copy_(vq[:, :, :w_len, :])
                        st.pos_tail[:, s0:end].copy_(new_pos[:, :w_len])
                        if self.kv_quant != "none":
                            st.k_scale[:, :, s0:end, :].copy_(ks[:, :, :w_len, :])
                            st.v_scale[:, :, s0:end, :].copy_(vs[:, :, :w_len, :])
                    st.seen.add_(t)
                    next_state = st
                else:
                    raise ValueError("static Global KV multi-token prefill requires uniform batch seen offsets")
        elif state is None and (self.global_attn or t <= self.window) and not return_all_states:
            attn = self._sdpa(q, k, v, is_causal=True, dropout_p=drop_p)
            next_state = self._update_state_block(st, k, v, new_pos, return_all_states=return_all_states)
        else:
            k_prev = _dequantize_kv_cache(st.k_tail, st.k_scale, self.kv_quant, k.dtype)
            v_prev = _dequantize_kv_cache(st.v_tail, st.v_scale, self.kv_quant, v.dtype)
            if self.sink_tokens > 0:
                k_all = torch.cat((st.k_sinks.to(k.dtype), k_prev, k), dim=2)
                v_all = torch.cat((st.v_sinks.to(v.dtype), v_prev, v), dim=2)
                pos_all = torch.cat((st.pos_sinks, st.pos_tail, new_pos), dim=1)
            else:
                k_all = torch.cat((k_prev, k), dim=2)
                v_all = torch.cat((v_prev, v), dim=2)
                pos_all = torch.cat((st.pos_tail, new_pos), dim=1)
            pq = new_pos[:, :, None]
            pk = pos_all[:, None, :]
            if self.global_attn:
                mask = ((pk >= 0) & (pk <= pq)).unsqueeze(1)
            else:
                mask = ((pk >= 0) & (pk <= pq) & ((pk < self.sink_tokens) | ((pq - pk) < self.window))).unsqueeze(1)
            attn = self._sdpa(q, k_all, v_all, attn_mask=mask, dropout_p=drop_p)
            next_state = self._update_state_block(st, k, v, new_pos, return_all_states=return_all_states)

        out = attn.transpose(1, 2).contiguous().reshape(b, t, self.attn_dim).to(x.dtype)
        attn_update = F.dropout(self.out_proj(out).to(x.dtype), p=self.dropout, training=self.training)
        if adapter is not None:
            attn_update = attn_update * adapter.euler_attn
        return attn_update, next_state

    def forward(
        self,
        x: torch.Tensor,
        state: SWAState | None = None,
        adapter=None,
        *,
        v0=None,
        return_v0: bool = False,
        phase_pos=None,
        anchor=None,
        return_all_states: bool = False,
    ):
        attn_update, next_state = self.forward_attn_delta(
            x, state=state, adapter=adapter, return_all_states=return_all_states
        )
        x = x + attn_update
        ffn_in = self.ffn_norm(x)
        if adapter is not None:
            ffn_in = ffn_in * adapter.ffn_scale + adapter.ffn_shift
        ffn_update = F.dropout(self.ffn(ffn_in).to(x.dtype), p=self.dropout, training=self.training)
        if adapter is not None:
            ffn_update = ffn_update * adapter.euler_ffn
        next_x = x + ffn_update
        if return_v0:
            return next_x, next_state, None
        return next_x, next_state

    def parameter_count(self) -> int:
        d = self.d_model
        ad = self.attn_dim
        ff = self.hidden_dim
        linear_scale = 4 if self.weight_mode == "quaternion" else 1
        attn_proj = (d * ad + 2 * d * self.kv_dim + ad * d) // linear_scale
        ffn_proj = (3 * d * ff) // linear_scale
        qkn = (2 * self.head_dim) if self.qk_norm else 0
        return 2 * d + qkn + attn_proj + ffn_proj

    def state_bytes_per_seq(self, kv_dtype_bytes: int = 4, seq_len: int = 0) -> int:
        tail_slots = max(int(seq_len), 0) if self.global_attn else (self.window - 1)
        sink_bytes = 2 * self.kv_heads * self.sink_tokens * self.head_dim * kv_dtype_bytes
        if self.kv_quant == "int4":
            tail_bytes = 2 * self.kv_heads * tail_slots * (self.head_dim // 2 + 2)
        elif self.kv_quant == "fp8":
            tail_bytes = 2 * self.kv_heads * tail_slots * (self.head_dim + 2)
        else:
            tail_bytes = 2 * self.kv_heads * tail_slots * self.head_dim * kv_dtype_bytes
        pos_bytes = (self.sink_tokens + tail_slots + 1) * 8
        return sink_bytes + tail_bytes + pos_bytes


class FractalLevelAdapter(nn.Module):
    """Lightweight per-level conditioning and Euler residual scaling (all 1D, no weight decay)."""

    def __init__(
        self,
        d_model,
        heads,
        level,
        depth,
        *,
        value_residual: bool = False,
        golden_phase: bool = False,
        free_energy: bool = False,
        leaky_budget: bool = False,
        decoupled_gates: bool = False,
    ):
        super().__init__()
        self.attn_scale = nn.Parameter(torch.ones(d_model))
        self.attn_shift = nn.Parameter(torch.zeros(d_model))
        self.ffn_scale = nn.Parameter(torch.ones(d_model))
        self.ffn_shift = nn.Parameter(torch.zeros(d_model))
        span = (level - 0.5 * (depth - 1)) * 0.5
        self.decay_bias = nn.Parameter(torch.full((heads,), span))
        beta_init = -max(0.0, span) if leaky_budget else 0.0
        self.beta_bias = nn.Parameter(torch.full((heads,), beta_init))
        self.beta_write_bias = nn.Parameter(torch.full((heads,), beta_init)) if decoupled_gates else None
        self.euler_attn = nn.Parameter(torch.ones(d_model))
        self.euler_ffn = nn.Parameter(torch.ones(d_model))
        self.anchor_gate = nn.Parameter(torch.full((d_model,), -2.0)) if level > 0 else None
        self.val_mix = nn.Parameter(torch.full((heads,), -1.0)) if (value_residual and level > 0) else None
        self.phase_scale = nn.Parameter(torch.ones(heads)) if golden_phase else None
        self.fe_alpha = nn.Parameter(torch.ones(1)) if (free_energy and level > 0) else None
        self.fe_bias = nn.Parameter(torch.ones(1)) if (free_energy and level > 0) else None
        self.fe_err_mix = nn.Parameter(torch.full((d_model,), -2.0)) if (free_energy and level > 0) else None


def _init_linear_weight(layer: nn.Module, std: float) -> None:
    if hasattr(layer, "reset_weight"):
        layer.reset_weight(std)
    else:
        nn.init.normal_(layer.weight, std=std)


class FractalDeltaCell(nn.Module):
    def __init__(self, config, research):
        super().__init__()
        self.d_model = config.d_model
        self.heads = config.n_heads
        self.head_dim = research.fractal_head_dim if research.fractal_head_dim > 0 else (self.d_model // self.heads)
        self.attn_dim = self.heads * self.head_dim
        req_kv = int(research.fractal_swa_kv_heads) if research.fractal_delta_gqa else 0
        if req_kv > 0 and req_kv < self.heads and self.heads % req_kv == 0:
            self.kv_heads = req_kv
        else:
            self.kv_heads = self.heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.kv_groups = self.heads // self.kv_heads
        self.hidden_dim = config.hidden_dim
        self.conv_kernel = 4
        self.dropout = config.dropout
        self.execution = research.state_execution
        self.chunk_size = research.state_chunk_size
        self.beta_scale = research.fractal_beta_scale
        self.weight_mode = research.fractal_weight_mode
        self.golden_phase = research.fractal_golden_phase
        self.deltanet_rope = research.fractal_deltanet_rope
        self.qk_norm = bool(research.fractal_qk_norm)
        self.out_gate_rank = int(research.fractal_out_gate_rank)
        self.free_energy = research.fractal_free_energy
        self.kda = research.fractal_kda
        self.decoupled_gates = research.fractal_decoupled_gates

        self.attn_norm = RMSNorm(self.d_model, eps=config.norm_eps)
        self.conv = nn.Conv1d(self.d_model, self.d_model, self.conv_kernel, groups=self.d_model)
        self.qkv = _make_linear(self.d_model, self.attn_dim + 2 * self.kv_dim, self.weight_mode)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if self.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if self.qk_norm else None
        if self.kda:
            self.decay_rank = max(16, self.kv_heads)
            self.decay_down = nn.Linear(self.d_model, self.decay_rank, bias=False)
            self.decay = nn.Linear(self.decay_rank, self.kv_dim, bias=True)
        else:
            self.decay_rank = 0
            self.decay_down = None
            self.decay = nn.Linear(self.d_model, self.kv_heads, bias=True)
        self.beta = nn.Linear(self.d_model, self.kv_heads, bias=True)
        self.beta_write = nn.Linear(self.d_model, self.kv_heads, bias=True) if self.decoupled_gates else None
        if self.out_gate_rank > 0:
            self.out_gate_down = nn.Linear(self.d_model, self.out_gate_rank, bias=False)
            self.out_gate = nn.Linear(self.out_gate_rank, self.attn_dim, bias=False)
        else:
            self.out_gate_down = None
            self.out_gate = _make_linear(self.d_model, self.attn_dim, self.weight_mode)
        self.register_buffer("out_gate_bias", torch.zeros(()))
        self.head_gamma = nn.Parameter(torch.ones(self.attn_dim))
        self.out_proj = _make_linear(self.attn_dim, self.d_model, self.weight_mode)
        self.ffn_norm = RMSNorm(self.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(self.d_model, self.hidden_dim)
        if self.weight_mode != "dense":
            self.ffn.gate = _make_linear(self.d_model, self.hidden_dim, self.weight_mode)
            self.ffn.up = _make_linear(self.d_model, self.hidden_dim, self.weight_mode)
            self.ffn.down = _make_linear(self.hidden_dim, self.d_model, self.weight_mode)

        if self.golden_phase:
            rot_pairs = max(1, self.head_dim // 2)
            exponents = 1.0 + 12.0 * torch.arange(rot_pairs, dtype=torch.float64) / max(1, rot_pairs)
            inv_freq = math.pi / (_GOLDEN_PHI ** exponents)
            self.register_buffer("golden_inv_freq", inv_freq, persistent=False)
        elif self.deltanet_rope:
            inv_freq = 1.0 / (
                float(config.rope_theta) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / float(self.head_dim))
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        if self.free_energy:
            self.fe_rank = min(16, max(4, self.d_model // 6))
            self.fe_norm = RMSNorm(self.d_model, eps=config.norm_eps)
            self.fe_down = nn.Linear(self.d_model, self.fe_rank, bias=False)
            self.fe_up = nn.Linear(self.fe_rank, self.d_model, bias=False)

        self._reset_parameters(config.n_layers)

    def _reset_parameters(self, depth):
        proj_std = 0.02 / math.sqrt(2 * depth)
        nn.init.normal_(self.conv.weight, std=0.02)
        nn.init.zeros_(self.conv.bias)
        _init_linear_weight(self.qkv, 0.02)
        if self.kda:
            nn.init.normal_(self.decay_down.weight, std=0.02)
            nn.init.normal_(self.decay.weight, std=0.02)
            with torch.no_grad():
                self.decay.bias.copy_(_init_decay_bias(self.kv_heads).repeat_interleave(self.head_dim))
        else:
            nn.init.normal_(self.decay.weight, std=0.02)
            with torch.no_grad():
                self.decay.bias.copy_(_init_decay_bias(self.kv_heads))
        nn.init.normal_(self.beta.weight, std=0.02)
        nn.init.constant_(self.beta.bias, -1.0)
        if self.beta_write is not None:
            nn.init.normal_(self.beta_write.weight, std=0.02)
            nn.init.constant_(self.beta_write.bias, -1.0)
        if self.out_gate_down is not None:
            nn.init.normal_(self.out_gate_down.weight, std=0.02)
        _init_linear_weight(self.out_gate, 0.02)
        _init_linear_weight(self.out_proj, proj_std)
        _init_linear_weight(self.ffn.gate, 0.02)
        _init_linear_weight(self.ffn.up, 0.02)
        _init_linear_weight(self.ffn.down, proj_std)
        if self.free_energy:
            nn.init.normal_(self.fe_down.weight, std=0.02)
            nn.init.normal_(self.fe_up.weight, std=proj_std)

    def forward(
        self,
        x,
        state=None,
        adapter=None,
        *,
        v0=None,
        return_v0: bool = False,
        phase_pos=None,
        anchor=None,
        return_all_states: bool = False,
    ):
        b, t, d = x.shape
        normed = self.attn_norm(x)
        if adapter is not None:
            normed = normed * adapter.attn_scale + adapter.attn_shift

        surprise = None
        if self.free_energy and anchor is not None and adapter is not None and adapter.fe_alpha is not None:
            pred = self.fe_up(F.silu(self.fe_down(normed)))
            err = self.fe_norm(anchor) - pred
            err_rms = err.float().square().mean(-1, keepdim=True).sqrt()
            surprise = torch.sigmoid(adapter.fe_alpha.float() * (err_rms - 1.0) + adapter.fe_bias.float()).to(x.dtype)
            normed = normed + adapter.fe_err_mix.sigmoid() * err

        tail_width = self.conv_kernel - 1
        with torch.autocast(device_type=x.device.type, enabled=False):
            norm_f = normed.float().transpose(1, 2)
            prefix = norm_f.new_zeros(b, d, tail_width) if state is None else state.tail
            padded = torch.cat((prefix, norm_f), dim=2)
            new_tail = (
                torch.stack([padded[:, :, idx + 1 : idx + 1 + tail_width] for idx in range(t)], dim=0).clone()
                if return_all_states
                else padded[:, :, -tail_width:].clone()
            )
            conv_out = F.conv1d(padded, self.conv.weight.float(), self.conv.bias.float(), groups=d).transpose(1, 2)
            local = (normed.float() + conv_out).to(normed.dtype)

        q_flat, k_flat, v_flat = self.qkv(local).split((self.attn_dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q_flat.reshape(b, t, self.heads, self.head_dim)
        k = k_flat.reshape(b, t, self.kv_heads, self.head_dim)
        v = v_flat.reshape(b, t, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if self.kda:
            decay_raw = self.decay(F.silu(self.decay_down(local))).reshape(b, t, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            decay_raw = self.decay(local).transpose(1, 2)
        beta_raw = self.beta(local).transpose(1, 2)

        with torch.autocast(device_type=x.device.type, enabled=False):
            key = F.normalize(k.float(), dim=-1, eps=1e-6)
            query = F.normalize(q.float(), dim=-1, eps=1e-6)
            if self.golden_phase and self.head_dim >= 2:
                base_pos = x.new_zeros((b,), dtype=torch.int64) if phase_pos is None else phase_pos
                pos_steps = (base_pos[:, None] if t == 1 else (base_pos[:, None] + torch.arange(t, device=x.device, dtype=torch.int64)[None, :])).double()
                freq = self.golden_inv_freq.double().view(1, 1, 1, -1)
                if adapter is not None and adapter.phase_scale is not None:
                    ps = adapter.phase_scale.double()
                    freq_k = freq * ps.view(1, self.kv_heads, 1, 1)
                    freq_q = freq * (ps.repeat_interleave(self.kv_groups) if self.kv_groups > 1 else ps).view(1, self.heads, 1, 1)
                    query = _apply_golden_phase(query, (pos_steps[:, None, :, None] * freq_q).remainder(2.0 * math.pi).float())
                    key = _apply_golden_phase(key, (pos_steps[:, None, :, None] * freq_k).remainder(2.0 * math.pi).float())
                else:
                    angles = (pos_steps[:, None, :, None] * freq).remainder(2.0 * math.pi).float()
                    query = _apply_golden_phase(query, angles)
                    key = _apply_golden_phase(key, angles)
            elif self.deltanet_rope and self.head_dim >= 2:
                base_pos = x.new_zeros((b,), dtype=torch.int64) if phase_pos is None else phase_pos
                pos_steps = (base_pos[:, None] if t == 1 else (base_pos[:, None] + torch.arange(t, device=x.device, dtype=torch.int64)[None, :])).double()
                angles = (pos_steps[:, None, :, None] * self.rope_inv_freq.double().view(1, 1, 1, -1)).float()
                query = _apply_split_half_rope(query, angles)
                key = _apply_split_half_rope(key, angles)
            val_f = v.float()
            if adapter is not None and adapter.val_mix is not None and v0 is not None:
                mix = adapter.val_mix.float().sigmoid().view(1, self.kv_heads, 1, 1)
                val_f = torch.lerp(val_f, v0, mix)

            decay_logits = decay_raw.float()
            beta_logits = beta_raw.float()
            if adapter is not None:
                decay_bias = adapter.decay_bias.float().view(1, self.kv_heads, 1, 1) if self.kda else adapter.decay_bias.float().view(1, self.kv_heads, 1)
                decay_logits = decay_logits + decay_bias
                beta_logits = beta_logits + adapter.beta_bias.float().view(1, self.kv_heads, 1)

            log_w = F.logsigmoid(decay_logits) if self.kda else F.logsigmoid(decay_logits).unsqueeze(-1)
            beta_erase = (self.beta_scale * beta_logits.sigmoid()).unsqueeze(-1)
            if self.beta_write is not None:
                bw_logits = self.beta_write(local).transpose(1, 2).float()
                if adapter is not None and adapter.beta_write_bias is not None:
                    bw_logits = bw_logits + adapter.beta_write_bias.float().view(1, self.kv_heads, 1)
                beta_write = (self.beta_scale * bw_logits.sigmoid()).unsqueeze(-1)
            else:
                beta_write = beta_erase
            w = log_w.exp()
            a = -w * beta_erase * key
            values = beta_write * val_f

            prev_mem = None if state is None else state.memory
            if return_all_states:
                y, _, mem_steps = dplr_sequential(
                    query, key, values, log_w, a, key, prev_mem, return_all_states=True
                )
                new_mem = mem_steps.float().contiguous()
            elif self.execution == "sequential" or t == 1:
                y, new_mem = dplr_sequential(query, key, values, log_w, a, key, prev_mem)
                new_mem = new_mem.float().contiguous()
            else:
                y, new_mem = dplr_chunked(query, key, values, log_w, a, key, prev_mem, chunk_size=self.chunk_size)
                new_mem = new_mem.float().contiguous()

            gamma = self.head_gamma.float().view(1, self.heads, 1, self.head_dim)
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * gamma
            y = y.transpose(1, 2).contiguous().reshape(b, t, self.attn_dim).to(x.dtype)

        gate_base = self.out_gate_down(normed) if self.out_gate_down is not None else normed
        gate_in = self.out_gate(gate_base) + self.out_gate_bias.to(dtype=normed.dtype)
        gated = y * F.silu(gate_in).to(y.dtype)
        attn_update = F.dropout(self.out_proj(gated).to(x.dtype), p=self.dropout, training=self.training)
        if adapter is not None:
            attn_update = attn_update * adapter.euler_attn
        if surprise is not None:
            attn_update = attn_update * surprise
        x = x + attn_update

        ffn_in = self.ffn_norm(x)
        if adapter is not None:
            ffn_in = ffn_in * adapter.ffn_scale + adapter.ffn_shift
        ffn_update = F.dropout(self.ffn(ffn_in).to(x.dtype), p=self.dropout, training=self.training)
        if adapter is not None:
            ffn_update = ffn_update * adapter.euler_ffn
        if surprise is not None:
            ffn_update = ffn_update * surprise
        next_x = x + ffn_update
        level_state = FractalLevelState(new_mem, new_tail)
        if return_v0:
            return next_x, level_state, val_f
        return next_x, level_state


class FractalDeltaMemory(Backbone):
    supports_hidden_only = True

    def __init__(self, config, research):
        super().__init__()
        self.config = config
        self.research = research
        self.depth = config.n_layers
        self.shared = research.share_scales
        self.heads = config.n_heads
        self.head_dim = research.fractal_head_dim if research.fractal_head_dim > 0 else (config.d_model // self.heads)
        self.attn_dim = self.heads * self.head_dim
        req_kv = int(research.fractal_swa_kv_heads) if research.fractal_delta_gqa else 0
        self.delta_kv_heads = req_kv if (0 < req_kv < self.heads and self.heads % req_kv == 0) else self.heads
        self.delta_kv_dim = self.delta_kv_heads * self.head_dim
        self.assoc_dim = research.memory_dim
        self.use_assoc_channel = bool(research.fractal_assoc_channel)
        self.chunk_size = research.state_chunk_size
        self.value_residual = research.fractal_value_residual
        self.golden_phase = research.fractal_golden_phase
        self.deltanet_rope = research.fractal_deltanet_rope
        self.use_phase_pos = bool(self.golden_phase or self.deltanet_rope)
        self.free_energy = research.fractal_free_energy
        self.weight_mode = research.fractal_weight_mode
        self.leaky_budget = research.fractal_leaky_budget
        self.assoc_delta = research.fractal_assoc_delta
        self.prelude_coda = research.fractal_prelude_coda
        self.assoc_multilag = research.fractal_assoc_multilag
        self.decoupled_gates = research.fractal_decoupled_gates
        self.hybrid_stride = research.fractal_hybrid_stride
        self.hybrid_swa = research.fractal_hybrid_swa
        self.use_exit_gate = research.fractal_exit_gate

        d = config.d_model
        self.assoc_val_dim = min(d, research.fractal_assoc_val_dim) if research.fractal_assoc_val_dim > 0 else min(d, 8 if d < 32 else 24)
        self.embedding = nn.Embedding(config.vocab_size, d)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Causal lag-1 shared-address associative induction channel (optional for surgery hybrids).
        if self.use_assoc_channel:
            self.input_norm = RMSNorm(d, eps=config.norm_eps)
            self.local_conv = nn.Conv1d(d, d, 3, groups=d, bias=False)
            nn.init.normal_(self.local_conv.weight, std=0.02)
            self.local_norm = RMSNorm(d, eps=config.norm_eps)
            self.assoc_address = nn.Linear(d, self.assoc_dim, bias=False)
            self.assoc_value = nn.Linear(d, self.assoc_val_dim, bias=False)
            self.assoc_write_gate = nn.Linear(d, 1, bias=True)
            self.assoc_readout = nn.Linear(self.assoc_val_dim, d, bias=False)
            for layer in (self.assoc_address, self.assoc_value, self.assoc_write_gate, self.assoc_readout):
                nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(self.assoc_write_gate.bias, -2.0)
            self.assoc_scale = nn.Parameter(torch.ones(d))
            self.assoc_lag_mix = nn.Parameter(torch.zeros(d)) if self.assoc_multilag else None
        else:
            self.input_norm = None
            self.local_conv = None
            self.local_norm = None
            self.assoc_address = None
            self.assoc_value = None
            self.assoc_write_gate = None
            self.assoc_readout = None
            self.assoc_scale = None
            self.assoc_lag_mix = None

        if self.hybrid_swa:
            if research.fractal_swa_levels is not None and len(research.fractal_swa_levels) > 0:
                self.swa_levels = frozenset(int(i) for i in research.fractal_swa_levels if 0 <= int(i) < self.depth)
            else:
                self.swa_levels = frozenset(lvl for lvl in range(self.depth) if (lvl + 1) % self.hybrid_stride == 0)
            self.hybrid_window_levels = ()
            self._hybrid_level_to_idx = {}
            self.hybrid_windows = nn.ModuleList()
            self.window_attn = None
            self.cells = nn.ModuleList(
                [
                    SlidingWindowGQACell(config, research) if lvl in self.swa_levels else FractalDeltaCell(config, research)
                    for lvl in range(self.depth)
                ]
            )
        else:
            self.swa_levels = frozenset()
            if self.hybrid_stride > 0:
                lvls = tuple(lvl for lvl in range(self.depth) if (lvl + 1) % self.hybrid_stride == 0)
                self.hybrid_window_levels = lvls if lvls else (self.depth - 1,)
                self._hybrid_level_to_idx = {lvl: idx for idx, lvl in enumerate(self.hybrid_window_levels)}
                self.hybrid_windows = nn.ModuleList(
                    [CausalLocalWindow(config, research.fractal_window) for _ in self.hybrid_window_levels]
                )
                self.window_attn = None
            else:
                self.hybrid_window_levels = ()
                self._hybrid_level_to_idx = {}
                self.hybrid_windows = nn.ModuleList()
                self.window_attn = CausalLocalWindow(config, research.fractal_window) if research.fractal_window > 0 else None

            cell_count = 3 if self.prelude_coda else (1 if self.shared else self.depth)
            self.cells = nn.ModuleList([FractalDeltaCell(config, research) for _ in range(cell_count)])
        self.adapters = (
            nn.ModuleList(
                [
                    FractalLevelAdapter(
                        d,
                        self.delta_kv_heads,
                        lvl,
                        self.depth,
                        value_residual=self.value_residual,
                        golden_phase=self.golden_phase,
                        free_energy=self.free_energy,
                        leaky_budget=self.leaky_budget,
                        decoupled_gates=self.decoupled_gates,
                    )
                    for lvl in range(self.depth)
                ]
            )
            if self.depth > 1
            else nn.ModuleList()
        )
        if self.use_exit_gate:
            self.exit_gate = nn.Linear(d, 1, bias=True)
            nn.init.normal_(self.exit_gate.weight, std=0.02)
            nn.init.constant_(self.exit_gate.bias, -2.0)
        else:
            self.exit_gate = None
        self.norm = RMSNorm(d, eps=config.norm_eps)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _validate_state(self, state, x):
        if not isinstance(state, FractalDeltaState) or len(state.levels) != self.depth:
            raise ValueError("invalid fractal delta state shape/dtype")
        b, _, d = x.shape
        mem_shape = (b, self.delta_kv_heads, self.head_dim, self.head_dim)
        tail_shape = (b, d, 3)
        if self.use_assoc_channel:
            assoc_checks = (
                (state.assoc_memory, (b, self.assoc_dim, self.assoc_val_dim)),
                (state.local_tail, (b, d, 2)),
            )
        else:
            assoc_checks = (
                (state.assoc_memory, (b, 0, 0)),
                (state.local_tail, (b, 0, 0)),
            )
        for tensor, expected in assoc_checks:
            if tensor.shape != expected or tensor.dtype != torch.float32 or tensor.device != x.device:
                raise ValueError("invalid fractal delta state shape/dtype")
        for lvl_idx, level_state in enumerate(state.levels):
            if lvl_idx in self.swa_levels:
                self.cells[lvl_idx].validate_state(level_state, b, x.device)
                continue
            if not isinstance(level_state, FractalLevelState):
                raise ValueError("invalid fractal delta state shape/dtype")
            for tensor, expected in ((level_state.memory, mem_shape), (level_state.tail, tail_shape)):
                if tensor.shape != expected or tensor.dtype != torch.float32 or tensor.device != x.device:
                    raise ValueError("invalid fractal delta state shape/dtype")
        if self.window_attn is None:
            if state.window is not None:
                raise ValueError("invalid fractal delta state shape/dtype")
        else:
            self.window_attn.validate_state(state.window, b, x.device)
        if not self.hybrid_windows:
            if state.hybrid_windows is not None:
                raise ValueError("invalid fractal delta state shape/dtype")
        else:
            if not isinstance(state.hybrid_windows, tuple) or len(state.hybrid_windows) != len(self.hybrid_windows):
                raise ValueError("invalid fractal delta state shape/dtype")
            for win, w_state in zip(self.hybrid_windows, state.hybrid_windows):
                win.validate_state(w_state, b, x.device)
        if self.use_phase_pos:
            if (
                state.phase_pos is None
                or state.phase_pos.shape != (b,)
                or state.phase_pos.dtype != torch.int64
                or state.phase_pos.device != x.device
            ):
                raise ValueError("invalid fractal delta state shape/dtype")
        elif state.phase_pos is not None:
            raise ValueError("invalid fractal delta state shape/dtype")

    def _run_assoc(self, raw_embed, state=None, *, execution="chunked", return_all_states: bool = False):
        b, t, d = raw_embed.shape
        if not self.use_assoc_channel:
            empty_shape = (t, b, 0, 0) if return_all_states else (b, 0, 0)
            zero_mem = raw_embed.new_zeros(empty_shape, dtype=torch.float32)
            return raw_embed.new_zeros((b, t, d)), zero_mem, zero_mem
        with torch.autocast(device_type=raw_embed.device.type, enabled=False):
            x_norm = self.input_norm(raw_embed).float()
            x_norm_t = x_norm.transpose(1, 2)
            if state is None:
                local_tail = x_norm_t.new_zeros(b, d, 2)
                prev_mem = None
            else:
                local_tail = state.local_tail
                prev_mem = state.assoc_memory

            padded = torch.cat((local_tail, x_norm_t), dim=2)
            next_local_tail = (
                torch.stack([padded[:, :, idx + 1 : idx + 3] for idx in range(t)], dim=0).clone()
                if return_all_states
                else padded[:, :, -2:].clone()
            )
            conv_out = F.conv1d(padded, self.local_conv.weight.float(), None, groups=d).transpose(1, 2)
            local = self.local_norm((x_norm + conv_out).to(raw_embed.dtype)).float()

            shifted = padded[:, :, 1:-1].transpose(1, 2)
            if self.assoc_lag_mix is not None:
                lag2 = padded[:, :, :-2].transpose(1, 2)
                shifted = torch.lerp(shifted, lag2, self.assoc_lag_mix.float().sigmoid())

            q = F.normalize(F.linear(x_norm, self.assoc_address.weight.float()), dim=-1, eps=1e-6)
            k_base = F.normalize(F.linear(shifted, self.assoc_address.weight.float()), dim=-1, eps=1e-6)
            gate = torch.sigmoid(F.linear(local, self.assoc_write_gate.weight.float(), self.assoc_write_gate.bias.float()))
            v = F.linear(local, self.assoc_value.weight.float())

            if self.assoc_delta:
                # Kohonen (1984) / Widrow-Hoff Delta projection rule + Hinton & Plaut (1987) elastic retention.
                q_u = q.unsqueeze(1)
                k_u = k_base.unsqueeze(1)
                g_u = gate.unsqueeze(1)
                v_u = (gate * v).unsqueeze(1)
                log_w = torch.log1p(-0.01 * g_u)
                a_u = -(log_w.exp() * g_u) * k_u
                dplr_prev = None if prev_mem is None else prev_mem.transpose(-1, -2).unsqueeze(1).contiguous()
                if return_all_states:
                    y_u, _, dplr_steps = dplr_sequential(
                        q_u, k_u, v_u, log_w, a_u, k_u, dplr_prev, return_all_states=True
                    )
                    next_mem = dplr_steps.squeeze(2).transpose(-1, -2).contiguous()
                elif execution == "sequential" or t == 1:
                    y_u, next_dplr = dplr_sequential(q_u, k_u, v_u, log_w, a_u, k_u, dplr_prev)
                    next_mem = next_dplr.squeeze(1).transpose(-1, -2).contiguous()
                else:
                    y_u, next_dplr = dplr_chunked(q_u, k_u, v_u, log_w, a_u, k_u, dplr_prev, chunk_size=self.chunk_size)
                    next_mem = next_dplr.squeeze(1).transpose(-1, -2).contiguous()
                read = y_u.squeeze(1)
            else:
                k = k_base * gate
                if return_all_states:
                    read, _, next_mem = _assoc_sequential(q, k, v, prev_mem, return_all_states=True)
                elif execution == "sequential" or t == 1:
                    read, next_mem = _assoc_sequential(q, k, v, prev_mem)
                else:
                    read, next_mem = _assoc_chunked(q, k, v, prev_mem, chunk_size=self.chunk_size)

        assoc_out = self.assoc_readout(read.to(raw_embed.dtype)).to(raw_embed.dtype) * self.assoc_scale
        return assoc_out, next_mem.float().contiguous(), next_local_tail

    def _skip_level_state(self, prev: FractalLevelState | SWAState | None, x: torch.Tensor, level: int = 0):
        b, _, d = x.shape
        if level in self.swa_levels:
            return self.cells[level].empty_state(b, x.device, x.dtype) if prev is None else prev
        tail_width = 3
        with torch.autocast(device_type=x.device.type, enabled=False):
            norm_f = x.detach().float().transpose(1, 2)
            if prev is None:
                mem = norm_f.new_zeros(b, self.heads, self.head_dim, self.head_dim)
                prefix = norm_f.new_zeros(b, d, tail_width)
            else:
                mem = prev.memory
                prefix = prev.tail
            padded = torch.cat((prefix, norm_f), dim=2)
            new_tail = padded[:, :, -tail_width:].clone()
            return FractalLevelState(mem, new_tail)

    def _resolve_raw_embeddings(self, input_ids=None, inputs_embeds=None) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            return self.embedding(input_ids)
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.d_model:
            raise ValueError(f"inputs_embeds must have shape (batch, seq_len, {self.config.d_model})")
        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        state=None,
        return_logits=True,
        max_levels: int | None = None,
        return_layer_states: bool = False,
    ):
        raw = self._resolve_raw_embeddings(input_ids, inputs_embeds)
        if state is not None:
            self._validate_state(state, raw)
        active_depth = self.depth if max_levels is None else max(1, min(int(max_levels), self.depth))
        exec_mode = next((c.execution for c in self.cells if hasattr(c, "execution")), "chunked")
        assoc_out, next_assoc_mem, next_local_tail = self._run_assoc(raw, state, execution=exec_mode)
        x = F.dropout(raw, p=self.config.dropout, training=self.training)
        next_window = None
        if self.window_attn is not None:
            win_out, next_window = self.window_attn(x, None if state is None else state.window)
            x = x + F.dropout(win_out, p=self.config.dropout, training=self.training)
        cur_phase = None
        next_phase_pos = None
        if self.use_phase_pos:
            cur_phase = (
                raw.new_zeros((raw.shape[0],), dtype=torch.int64)
                if state is None
                else state.phase_pos
            )
            next_phase_pos = (cur_phase + int(raw.shape[1])).contiguous()
        anchor = x
        next_levels = []
        next_hybrid_windows = [None] * len(self.hybrid_windows) if self.hybrid_windows else None
        intermediate_states = []
        exit_logits = []
        layer_states = []
        collect_inter = bool(
            (getattr(self.research, "intermediate_weight", 0.0) > 0 or getattr(self.research, "exit_gate_weight", 0.0) > 0)
            and self.depth > 1
        )
        collect_exit = bool(self.exit_gate is not None and self.depth > 1)
        use_ckpt = bool(getattr(self.research, "gradient_checkpointing", False) and self.training and torch.is_grad_enabled())
        assoc_drop = F.dropout(assoc_out, p=self.config.dropout, training=self.training)
        v0 = None
        for level in range(self.depth):
            prev = None if state is None else state.levels[level]
            if level >= active_depth:
                next_levels.append(self._skip_level_state(prev, x, level=level))
                if level in self._hybrid_level_to_idx:
                    w_idx = self._hybrid_level_to_idx[level]
                    prev_hw = None if state is None else state.hybrid_windows[w_idx]
                    next_hybrid_windows[w_idx] = (
                        self.hybrid_windows[w_idx].empty_state(raw.shape[0], raw.device)
                        if prev_hw is None
                        else prev_hw
                    )
                continue
            adapter = self.adapters[level] if self.depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.cells[self.cell_index(level)]
            if use_ckpt:
                if isinstance(cell, SlidingWindowGQACell):
                    p_ks = None if prev is None else prev.k_sinks
                    p_vs = None if prev is None else prev.v_sinks
                    p_ps = None if prev is None else prev.pos_sinks
                    p_kt = None if prev is None else prev.k_tail
                    p_vt = None if prev is None else prev.v_tail
                    p_pt = None if prev is None else prev.pos_tail
                    p_seen = None if prev is None else prev.seen
                    p_ksc = None if prev is None else prev.k_scale
                    p_vsc = None if prev is None else prev.v_scale

                    def _run_swa(x_in, ks, vs, ps, kt, vt, pt, sn, ksc, vsc, _cell=cell, _ad=adapter):
                        st_in = None if ks is None else SWAState(ks, vs, ps, kt, vt, pt, sn, ksc, vsc)
                        xo, cur = _cell(x_in, st_in, adapter=_ad)
                        return xo, cur.k_sinks, cur.v_sinks, cur.pos_sinks, cur.k_tail, cur.v_tail, cur.pos_tail, cur.seen, cur.k_scale, cur.v_scale

                    x, nks, nvs, nps, nkt, nvt, npt, nsn, nksc, nvsc = grad_checkpoint(
                        _run_swa, x, p_ks, p_vs, p_ps, p_kt, p_vt, p_pt, p_seen, p_ksc, p_vsc, use_reentrant=False
                    )
                    current = SWAState(nks, nvs, nps, nkt, nvt, npt, nsn, nksc, nvsc)
                else:
                    prev_mem = None if prev is None else prev.memory
                    prev_tail = None if prev is None else prev.tail
                    if self.value_residual and level == 0:
                        def _run_lvl0(x_in, anc_in, p_mem, p_tail, p_pos, _cell=cell, _ad=adapter):
                            st_in = None if p_mem is None else FractalLevelState(p_mem, p_tail)
                            xo, cur, v0_o = _cell(x_in, st_in, adapter=_ad, return_v0=True, phase_pos=p_pos, anchor=anc_in)
                            return xo, cur.memory, cur.tail, v0_o

                        x, cur_mem, cur_tail, v0 = grad_checkpoint(
                            _run_lvl0, x, anchor, prev_mem, prev_tail, cur_phase, use_reentrant=False
                        )
                        current = FractalLevelState(cur_mem, cur_tail)
                    else:
                        def _run_lvl(x_in, anc_in, v0_in, p_mem, p_tail, p_pos, _cell=cell, _ad=adapter):
                            st_in = None if p_mem is None else FractalLevelState(p_mem, p_tail)
                            xo, cur = _cell(x_in, st_in, adapter=_ad, v0=v0_in, phase_pos=p_pos, anchor=anc_in)
                            return xo, cur.memory, cur.tail

                        x, cur_mem, cur_tail = grad_checkpoint(
                            _run_lvl, x, anchor, v0, prev_mem, prev_tail, cur_phase, use_reentrant=False
                        )
                        current = FractalLevelState(cur_mem, cur_tail)
            elif self.value_residual and level == 0:
                x, current, v0 = cell(
                    x, prev, adapter=adapter, return_v0=True, phase_pos=cur_phase, anchor=anchor
                )
            else:
                x, current = cell(
                    x, prev, adapter=adapter, v0=v0, phase_pos=cur_phase, anchor=anchor
                )
            next_levels.append(current)
            if level in self._hybrid_level_to_idx:
                w_idx = self._hybrid_level_to_idx[level]
                hw_mod = self.hybrid_windows[w_idx]
                prev_hw = None if state is None else state.hybrid_windows[w_idx]
                if use_ckpt:
                    pk = None if prev_hw is None else prev_hw.k_tail
                    pv = None if prev_hw is None else prev_hw.v_tail
                    pval = None if prev_hw is None else prev_hw.valid_tail

                    def _run_hw(x_in, k_in, v_in, val_in, _hw=hw_mod):
                        hw_st = None if k_in is None else LocalWindowState(k_in, v_in, val_in)
                        ho, nst = _hw(x_in, hw_st)
                        return ho, nst.k_tail, nst.v_tail, nst.valid_tail

                    hw_out, nk, nv, nval = grad_checkpoint(_run_hw, x, pk, pv, pval, use_reentrant=False)
                    next_hw = LocalWindowState(nk, nv, nval)
                else:
                    hw_out, next_hw = hw_mod(x, prev_hw)
                x = x + F.dropout(hw_out, p=self.config.dropout, training=self.training)
                next_hybrid_windows[w_idx] = next_hw
            if return_layer_states:
                layer_states.append(x)
            if (collect_inter or collect_exit) and level < active_depth - 1:
                inter_h = self.norm(x + assoc_drop)
                if collect_inter:
                    intermediate_states.append(inter_h)
                if collect_exit:
                    exit_logits.append(self.exit_gate(inter_h))
        x = x + assoc_drop
        hidden = self.norm(x)
        logits = self.lm_head(hidden) if return_logits else None
        next_state = FractalDeltaState(
            tuple(next_levels),
            next_assoc_mem,
            next_local_tail,
            next_window,
            next_phase_pos,
            None if next_hybrid_windows is None else tuple(next_hybrid_windows),
        )
        return ModelOutput(
            logits=logits,
            hidden_states=hidden,
            state=next_state,
            intermediate_states=tuple(intermediate_states),
            exit_logits=tuple(exit_logits),
            layer_states=tuple(layer_states),
        )

    def forward_teacher_forced_layers(
        self,
        teacher_inputs: tuple[torch.Tensor, ...],
        *,
        norm_last: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """RADLADS Stage-1 parallel blockwise forward: feed clean teacher input h_l into each student level l."""
        if not teacher_inputs:
            return ()
        n_in = len(teacher_inputs)
        anchor = teacher_inputs[0].detach().to(dtype=self.embedding.weight.dtype)
        cur_phase = (
            anchor.new_zeros((anchor.shape[0],), dtype=torch.int64)
            if self.use_phase_pos
            else None
        )
        outputs = []
        for level in range(self.depth):
            t_idx = 0 if self.depth == 1 else round(level * (n_in - 1) / max(self.depth - 1, 1))
            x = teacher_inputs[t_idx].detach().to(dtype=self.embedding.weight.dtype)
            adapter = self.adapters[level] if self.depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.cells[self.cell_index(level)]
            x, _ = cell(x, None, adapter=adapter, v0=None, phase_pos=cur_phase, anchor=anchor)
            if level in self._hybrid_level_to_idx:
                w_idx = self._hybrid_level_to_idx[level]
                hw_out, _ = self.hybrid_windows[w_idx](x, None)
                x = x + F.dropout(hw_out, p=self.config.dropout, training=self.training)
            if norm_last and level == self.depth - 1:
                x = self.norm(x)
            outputs.append(x)
        return tuple(outputs)

    def forward_level_attn_delta(
        self,
        level: int,
        h_in: torch.Tensor,
        anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute only the attention + hybrid-window delta of level l on clean teacher input h_in (skipping frozen FFN)."""
        cell = self.cells[self.cell_index(level)]
        target_dtype = cell.attn_norm.weight.dtype
        x = h_in.to(dtype=target_dtype)
        anc = x if anchor is None else anchor.to(dtype=target_dtype)
        adapter = self.adapters[level] if self.depth > 1 else None
        if adapter is not None and adapter.anchor_gate is not None:
            x = x + adapter.anchor_gate.sigmoid().to(dtype=target_dtype) * anc
        if isinstance(cell, SlidingWindowGQACell):
            s_delta, _ = cell.forward_attn_delta(x, state=None, adapter=adapter)
            return s_delta
        b, t, d = x.shape
        normed = cell.attn_norm(x)
        if adapter is not None:
            normed = normed * adapter.attn_scale + adapter.attn_shift
        tail_width = cell.conv_kernel - 1
        with torch.autocast(device_type=x.device.type, enabled=False):
            norm_f = normed.float().transpose(1, 2)
            prefix = norm_f.new_zeros(b, d, tail_width)
            padded = torch.cat((prefix, norm_f), dim=2)
            conv_out = F.conv1d(padded, cell.conv.weight.float(), cell.conv.bias.float(), groups=d).transpose(1, 2)
            local = (normed.float() + conv_out).to(normed.dtype)
        q_flat, k_flat, v_flat = cell.qkv(local).split((cell.attn_dim, cell.kv_dim, cell.kv_dim), dim=-1)
        q = q_flat.reshape(b, t, cell.heads, cell.head_dim)
        k = k_flat.reshape(b, t, cell.kv_heads, cell.head_dim)
        v = v_flat.reshape(b, t, cell.kv_heads, cell.head_dim)
        if cell.q_norm is not None:
            q = cell.q_norm(q)
            k = cell.k_norm(k)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if cell.kda:
            decay_raw = cell.decay(F.silu(cell.decay_down(local))).reshape(b, t, cell.kv_heads, cell.head_dim).permute(0, 2, 1, 3)
        else:
            decay_raw = cell.decay(local).transpose(1, 2)
        beta_raw = cell.beta(local).transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            key = F.normalize(k.float(), dim=-1, eps=1e-6)
            query = F.normalize(q.float(), dim=-1, eps=1e-6)
            if cell.golden_phase and cell.head_dim >= 2:
                pos_steps = torch.arange(t, device=x.device, dtype=torch.int64)[None, :].double()
                freq = cell.golden_inv_freq.double().view(1, 1, 1, -1)
                if adapter is not None and adapter.phase_scale is not None:
                    ps = adapter.phase_scale.double()
                    freq_k = freq * ps.view(1, cell.kv_heads, 1, 1)
                    freq_q = freq * (ps.repeat_interleave(cell.kv_groups) if cell.kv_groups > 1 else ps).view(1, cell.heads, 1, 1)
                    query = _apply_golden_phase(query, (pos_steps[:, None, :, None] * freq_q).remainder(2.0 * math.pi).float())
                    key = _apply_golden_phase(key, (pos_steps[:, None, :, None] * freq_k).remainder(2.0 * math.pi).float())
                else:
                    angles = (pos_steps[:, None, :, None] * freq).remainder(2.0 * math.pi).float()
                    query = _apply_golden_phase(query, angles)
                    key = _apply_golden_phase(key, angles)
            elif cell.deltanet_rope and cell.head_dim >= 2:
                pos_steps = torch.arange(t, device=x.device, dtype=torch.int64)[None, :].double()
                angles = (pos_steps[:, None, :, None] * cell.rope_inv_freq.double().view(1, 1, 1, -1)).float()
                query = _apply_split_half_rope(query, angles)
                key = _apply_split_half_rope(key, angles)
            val_f = v.float()
            decay_logits = decay_raw.float()
            beta_logits = beta_raw.float()
            if adapter is not None:
                decay_bias = adapter.decay_bias.float().view(1, cell.kv_heads, 1, 1) if cell.kda else adapter.decay_bias.float().view(1, cell.kv_heads, 1)
                decay_logits = decay_logits + decay_bias
                beta_logits = beta_logits + adapter.beta_bias.float().view(1, cell.kv_heads, 1)
            log_w = F.logsigmoid(decay_logits) if cell.kda else F.logsigmoid(decay_logits).unsqueeze(-1)
            beta_erase = (cell.beta_scale * beta_logits.sigmoid()).unsqueeze(-1)
            if cell.beta_write is not None:
                bw_logits = cell.beta_write(local).transpose(1, 2).float()
                if adapter is not None and adapter.beta_write_bias is not None:
                    bw_logits = bw_logits + adapter.beta_write_bias.float().view(1, cell.kv_heads, 1)
                beta_write = (cell.beta_scale * bw_logits.sigmoid()).unsqueeze(-1)
            else:
                beta_write = beta_erase
            w = log_w.exp()
            a = -w * beta_erase * key
            values = beta_write * val_f
            if cell.execution == "sequential" or t == 1:
                y, _ = dplr_sequential(query, key, values, log_w, a, key, None)
            else:
                y, _ = dplr_chunked(query, key, values, log_w, a, key, None, chunk_size=cell.chunk_size)
            gamma = cell.head_gamma.float().view(1, cell.heads, 1, cell.head_dim)
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * gamma
            y = y.transpose(1, 2).contiguous().reshape(b, t, cell.attn_dim).to(x.dtype)
        gate_base = cell.out_gate_down(normed) if cell.out_gate_down is not None else normed
        gate_in = cell.out_gate(gate_base) + cell.out_gate_bias.to(dtype=normed.dtype)
        gated = y * F.silu(gate_in).to(y.dtype)
        s_delta = F.dropout(cell.out_proj(gated).to(x.dtype), p=cell.dropout, training=self.training)
        if adapter is not None:
            s_delta = s_delta * adapter.euler_attn
        if level in self._hybrid_level_to_idx:
            w_idx = self._hybrid_level_to_idx[level]
            hw_out, _ = self.hybrid_windows[w_idx](h_in.to(dtype=x.dtype) + s_delta, None)
            s_delta = s_delta + F.dropout(hw_out, p=self.config.dropout, training=self.training)
        return s_delta

    def state_bytes_per_sequence(self, seq_len: int = 0):
        d = self.config.d_model
        kv_dtype_bytes = 2 if self.embedding.weight.dtype in (torch.float16, torch.bfloat16) else 4
        delta_level_bytes = (self.delta_kv_heads * self.head_dim * self.head_dim + d * 3) * 4
        levels_bytes = 0
        for lvl in range(self.depth):
            if lvl in self.swa_levels:
                levels_bytes += self.cells[lvl].state_bytes_per_seq(kv_dtype_bytes=kv_dtype_bytes, seq_len=seq_len)
            else:
                levels_bytes += delta_level_bytes
        assoc_floats = (self.assoc_dim * self.assoc_val_dim + 2 * d) if self.use_assoc_channel else 0
        win_floats = self.window_attn.state_floats_per_seq() if self.window_attn is not None else 0
        win_floats += sum(w.state_floats_per_seq() for w in self.hybrid_windows)
        phase_bytes = 8 if self.use_phase_pos else 0
        return levels_bytes + (assoc_floats + win_floats) * 4 + phase_bytes

    def cell_index(self, level: int) -> int:
        if self.prelude_coda:
            if level == 0:
                return 0
            if level == self.depth - 1:
                return 2
            return 1
        return 0 if self.shared else level

    def analytical_cost(self, seq_len, *, include_logits=True):
        d = self.config.d_model
        h = self.heads
        dh = self.head_dim
        ff = self.config.hidden_dim
        delta_cell0 = next((c for c in self.cells if isinstance(c, FractalDeltaCell)), None)
        if delta_cell0 is not None:
            decay_flops = (2 * d * delta_cell0.decay_rank + 2 * delta_cell0.decay_rank * d) if delta_cell0.kda else 2 * d * h
            beta_flops = (4 * d * h) if self.decoupled_gates else (2 * d * h)
            delta_level_flops = seq_len * (10 * d * d + beta_flops + decay_flops + 6 * d * ff + 6 * d * dh)
        else:
            delta_level_flops = 0
        levels_flops = 0
        for lvl in range(self.depth):
            cell = self.cells[self.cell_index(lvl)]
            if isinstance(cell, SlidingWindowGQACell):
                eff_w = min(seq_len, cell.window + cell.sink_tokens)
                levels_flops += seq_len * (4 * d * d + 4 * d * cell.kv_dim + 4 * eff_w * d + 6 * d * ff)
            else:
                levels_flops += delta_level_flops
        assoc_flops = seq_len * (
            4 * d * self.assoc_dim
            + 4 * d * self.assoc_val_dim
            + 6 * d
            + 4 * self.assoc_dim * self.assoc_val_dim
        )
        win_flops = 0
        if self.window_attn is not None:
            wd = self.window_attn.win_dim
            win_flops += seq_len * (8 * d * wd + 4 * min(seq_len, self.window_attn.window) * wd)
        for hw in self.hybrid_windows:
            wd = hw.win_dim
            win_flops += seq_len * (8 * d * wd + 4 * min(seq_len, hw.window) * wd)
        exit_flops = (2 * seq_len * d * max(self.depth - 1, 0)) if self.use_exit_gate else 0
        forward = (
            levels_flops
            + assoc_flops
            + win_flops
            + exit_flops
            + (2 * seq_len * d * self.config.vocab_size if include_logits else 0)
        )
        return forward, 0, self.depth

    def parameter_formula(self):
        c = self.config
        d, h, kv_h, ad, kv_d, ff, k = c.d_model, self.heads, self.delta_kv_heads, self.attn_dim, self.delta_kv_dim, c.hidden_dim, 4
        embed = c.vocab_size * d
        assoc_params = (
            d + 3 * d + d + d * self.assoc_dim + d * self.assoc_val_dim + (d + 1) + self.assoc_val_dim * d + d + (d if self.assoc_multilag else 0)
            if self.use_assoc_channel
            else 0
        )
        win_params = self.window_attn.parameter_count(d) if self.window_attn is not None else 0
        win_params += sum(hw.parameter_count(d) for hw in self.hybrid_windows)
        linear_scale = 4 if self.weight_mode == "quaternion" else 1
        delta_cell0 = next((cell_i for cell_i in self.cells if isinstance(cell_i, FractalDeltaCell)), None)
        if delta_cell0 is not None:
            gate_params = (
                (d * delta_cell0.out_gate_rank + delta_cell0.out_gate_rank * ad)
                if delta_cell0.out_gate_rank > 0
                else (d * ad) // linear_scale
            )
            dense_proj = (d * (ad + 2 * kv_d) + ad * d + 3 * d * ff) // linear_scale + gate_params
            decay_cell = (d * delta_cell0.decay_rank + (delta_cell0.decay_rank + 1) * kv_d) if delta_cell0.kda else (d + 1) * kv_h
            fe_cell = (d + 2 * d * delta_cell0.fe_rank) if self.free_energy else 0
            beta_cell = 2 * (d + 1) * kv_h if self.decoupled_gates else (d + 1) * kv_h
            qkn_cell = (2 * delta_cell0.head_dim) if delta_cell0.qk_norm else 0
            delta_cell_params = (
                d  # attn_norm
                + d * (k + 1)  # depthwise conv weight + bias
                + dense_proj  # qkv, out_gate, out_proj, SwiGLU
                + qkn_cell  # optional q_norm + k_norm
                + decay_cell  # decay
                + beta_cell  # beta (and beta_write when decoupled)
                + ad  # head_gamma
                + d  # ffn_norm
                + fe_cell
            )
        else:
            delta_cell_params = 0
        cells_params = sum(
            cell_i.parameter_count() if isinstance(cell_i, SlidingWindowGQACell) else delta_cell_params
            for cell_i in self.cells
        )
        adapters = self.depth * (6 * d + 2 * kv_h) + max(self.depth - 1, 0) * d if self.depth > 1 else 0
        if self.decoupled_gates and self.depth > 1:
            adapters += self.depth * kv_h
        if self.value_residual and self.depth > 1:
            adapters += (self.depth - 1) * kv_h
        if self.golden_phase and self.depth > 1:
            adapters += self.depth * kv_h
        if self.free_energy and self.depth > 1:
            adapters += (self.depth - 1) * (d + 2)
        exit_params = (d + 1) if self.use_exit_gate else 0
        final_norm = d
        return embed + assoc_params + win_params + cells_params + adapters + exit_params + final_norm


class FractalDecode:
    """Incremental decode adapter for FractalDeltaMemory supporting 1-token and multi-token block steps."""

    is_recurrent = True

    def __init__(self, model):
        if type(model) is not FractalDeltaMemory or model.training:
            raise ValueError("fractal decode adapter requires evaluation FractalDeltaMemory")
        self.model = model
        self.step_cells = tuple(copy(cell) for cell in model.cells)
        for cell in self.step_cells:
            if hasattr(cell, "execution"):
                cell.execution = "sequential"

    @staticmethod
    def _slice_level_state(lvl, index: int):
        if isinstance(lvl, SWAState):
            return SWAState(
                lvl.k_sinks[index].contiguous(),
                lvl.v_sinks[index].contiguous(),
                lvl.pos_sinks[index].contiguous(),
                lvl.k_tail[index].contiguous(),
                lvl.v_tail[index].contiguous(),
                lvl.pos_tail[index].contiguous(),
                lvl.seen[index].contiguous(),
                None if lvl.k_scale is None else lvl.k_scale[index].contiguous(),
                None if lvl.v_scale is None else lvl.v_scale[index].contiguous(),
            )
        return FractalLevelState(lvl.memory[index].contiguous(), lvl.tail[index].contiguous())

    @staticmethod
    def _slice_window_state(win_state: LocalWindowState | None, index: int) -> LocalWindowState | None:
        if win_state is None:
            return None
        if win_state.k_tail.ndim == 5:
            return LocalWindowState(
                win_state.k_tail[index].contiguous(),
                win_state.v_tail[index].contiguous(),
                win_state.valid_tail[index].contiguous(),
            )
        return win_state

    @staticmethod
    def slice_step_state(stacked_state: FractalDeltaState, index: int) -> FractalDeltaState:
        """Extract a single step state from stacked (T, ...) verify states."""
        return FractalDeltaState(
            tuple(FractalDecode._slice_level_state(lvl, index) for lvl in stacked_state.levels),
            stacked_state.assoc_memory[index].contiguous(),
            stacked_state.local_tail[index].contiguous(),
            FractalDecode._slice_window_state(stacked_state.window, index),
            None if stacked_state.phase_pos is None else stacked_state.phase_pos[index].contiguous(),
            None
            if stacked_state.hybrid_windows is None
            else tuple(FractalDecode._slice_window_state(hw, index) for hw in stacked_state.hybrid_windows),
        )

    def prefill_hidden(self, ids=None, capacity: int | None = None, *, inputs_embeds=None):
        seq_len = ids.shape[1] if ids is not None else (0 if inputs_embeds is None else inputs_embeds.shape[1])
        cap = seq_len if capacity is None else int(capacity)
        if not 0 < seq_len <= cap <= self.model.config.max_seq_len:
            raise ValueError("invalid prefill capacity")
        out = self.model(ids, inputs_embeds=inputs_embeds, return_logits=False)
        last_hidden = out.hidden_states[:, -1:]
        return self.model.lm_head(last_hidden), last_hidden, out.state

    def prefill(self, ids=None, capacity: int | None = None, *, inputs_embeds=None):
        logits, _, state = self.prefill_hidden(ids, capacity, inputs_embeds=inputs_embeds)
        return logits, state

    def pad_global_state_static(self, state: FractalDeltaState, capacity: int) -> FractalDeltaState:
        """Convert dynamic Global SWAState levels into fixed-capacity static buffers for CUDAGraph decode."""
        levels = []
        for level, lvl_state in enumerate(state.levels):
            cell = self.step_cells[self.model.cell_index(level)]
            if isinstance(cell, SlidingWindowGQACell) and isinstance(lvl_state, SWAState) and cell.global_attn:
                levels.append(cell.pad_state_static(lvl_state, capacity))
            else:
                levels.append(lvl_state)
        return FractalDeltaState(
            tuple(levels),
            state.assoc_memory,
            state.local_tail,
            state.window,
            state.phase_pos,
            state.hybrid_windows,
        )

    def prefill_static(
        self,
        ids=None,
        capacity: int | None = None,
        *,
        inputs_embeds=None,
        chunk_size: int = 1024,
    ):
        """Prefill into static-capacity Global KV buffers, chunking long prompts (16K-64K+) via fast chunked DPLR."""
        seq_len = ids.shape[1] if ids is not None else (0 if inputs_embeds is None else inputs_embeds.shape[1])
        cap = seq_len if capacity is None else int(capacity)
        if not 0 < seq_len <= cap <= self.model.config.max_seq_len:
            raise ValueError("invalid prefill capacity")
        first_end = min(seq_len, max(1, int(chunk_size)))
        out = self.model(
            None if ids is None else ids[:, :first_end],
            inputs_embeds=None if inputs_embeds is None else inputs_embeds[:, :first_end],
            return_logits=False,
        )
        state = self.pad_global_state_static(out.state, cap)
        for start in range(first_end, seq_len, max(1, int(chunk_size))):
            end = min(start + max(1, int(chunk_size)), seq_len)
            out = self.model(
                None if ids is None else ids[:, start:end],
                inputs_embeds=None if inputs_embeds is None else inputs_embeds[:, start:end],
                state=state,
                return_logits=False,
            )
            state = out.state
        last_hidden = out.hidden_states[:, -1:]
        return self.model.lm_head(last_hidden), state

    def prefill_embeds(self, inputs_embeds: torch.Tensor, capacity: int | None = None):
        return self.prefill_hidden(None, capacity, inputs_embeds=inputs_embeds)

    def _skip_decode_level(self, prev: FractalLevelState | SWAState, x: torch.Tensor, *, return_all_states: bool = False):
        t = x.shape[1]
        if isinstance(prev, SWAState):
            if not return_all_states:
                return prev
            return SWAState(
                prev.k_sinks.unsqueeze(0).expand(t, *prev.k_sinks.shape).clone(),
                prev.v_sinks.unsqueeze(0).expand(t, *prev.v_sinks.shape).clone(),
                prev.pos_sinks.unsqueeze(0).expand(t, *prev.pos_sinks.shape).clone(),
                prev.k_tail.unsqueeze(0).expand(t, *prev.k_tail.shape).clone(),
                prev.v_tail.unsqueeze(0).expand(t, *prev.v_tail.shape).clone(),
                prev.pos_tail.unsqueeze(0).expand(t, *prev.pos_tail.shape).clone(),
                prev.seen.unsqueeze(0).expand(t, *prev.seen.shape).clone(),
                None if prev.k_scale is None else prev.k_scale.unsqueeze(0).expand(t, *prev.k_scale.shape).clone(),
                None if prev.v_scale is None else prev.v_scale.unsqueeze(0).expand(t, *prev.v_scale.shape).clone(),
            )
        tail_width = 3
        with torch.autocast(device_type=x.device.type, enabled=False):
            norm_f = x.detach().float().transpose(1, 2)
            padded = torch.cat((prev.tail, norm_f), dim=2)
            if return_all_states:
                new_tail = torch.stack([padded[:, :, idx + 1 : idx + 1 + tail_width] for idx in range(t)], dim=0).clone()
                new_mem = prev.memory.unsqueeze(0).expand(t, *prev.memory.shape).clone()
            else:
                new_tail = padded[:, :, -tail_width:].clone()
                new_mem = prev.memory
            return FractalLevelState(new_mem, new_tail)

    @staticmethod
    def _skip_window_state(prev: LocalWindowState, t: int, *, return_all_states: bool = False) -> LocalWindowState:
        if not return_all_states:
            return prev
        return LocalWindowState(
            prev.k_tail.unsqueeze(0).expand(t, *prev.k_tail.shape).clone(),
            prev.v_tail.unsqueeze(0).expand(t, *prev.v_tail.shape).clone(),
            prev.valid_tail.unsqueeze(0).expand(t, *prev.valid_tail.shape).clone(),
        )

    def step_hidden(
        self,
        ids=None,
        state=None,
        *,
        inputs_embeds=None,
        return_all_states: bool = False,
        max_levels: int | None = None,
    ):
        if (ids is None) == (inputs_embeds is None) or state is None:
            raise ValueError("fractal recurrent step requires state and exactly one of ids or inputs_embeds")
        batch_size = state.assoc_memory.shape[0]
        if ids is not None:
            if ids.ndim != 2 or ids.shape[0] != batch_size or ids.shape[1] < 1:
                raise ValueError("fractal recurrent step requires shape (batch, >=1)")
            t = ids.shape[1]
            raw = self.model.embedding(ids)
        else:
            if (
                inputs_embeds.ndim != 3
                or inputs_embeds.shape[0] != batch_size
                or inputs_embeds.shape[1] < 1
                or inputs_embeds.shape[2] != self.model.config.d_model
            ):
                raise ValueError("fractal recurrent step requires inputs_embeds shape (batch, >=1, d_model)")
            t = inputs_embeds.shape[1]
            raw = inputs_embeds
        active_depth = self.model.depth if max_levels is None else max(1, min(int(max_levels), self.model.depth))
        assoc_out, next_assoc_mem, next_local_tail = self.model._run_assoc(
            raw, state, execution="sequential", return_all_states=return_all_states
        )
        x = raw
        next_window = None
        if self.model.window_attn is not None:
            win_out, next_window = self.model.window_attn(x, state.window, return_all_states=return_all_states)
            x = x + win_out
        cur_phase = None
        next_phase_pos = None
        if self.model.use_phase_pos:
            cur_phase = state.phase_pos
            next_phase_pos = (
                torch.stack([cur_phase + (idx + 1) for idx in range(t)], dim=0).contiguous()
                if return_all_states
                else (cur_phase + t).contiguous()
            )
        anchor = x
        next_levels = []
        next_hybrid_windows = [None] * len(self.model.hybrid_windows) if self.model.hybrid_windows else None
        v0 = None
        for level, prev in enumerate(state.levels):
            if level >= active_depth:
                next_levels.append(self._skip_decode_level(prev, x, return_all_states=return_all_states))
                if level in self.model._hybrid_level_to_idx:
                    w_idx = self.model._hybrid_level_to_idx[level]
                    next_hybrid_windows[w_idx] = self._skip_window_state(
                        state.hybrid_windows[w_idx], t, return_all_states=return_all_states
                    )
                continue
            adapter = self.model.adapters[level] if self.model.depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.step_cells[self.model.cell_index(level)]
            if self.model.value_residual and level == 0:
                x, current, v0 = cell(
                    x,
                    prev,
                    adapter=adapter,
                    return_v0=True,
                    phase_pos=cur_phase,
                    anchor=anchor,
                    return_all_states=return_all_states,
                )
            else:
                x, current = cell(
                    x,
                    prev,
                    adapter=adapter,
                    v0=v0,
                    phase_pos=cur_phase,
                    anchor=anchor,
                    return_all_states=return_all_states,
                )
            next_levels.append(current)
            if level in self.model._hybrid_level_to_idx:
                w_idx = self.model._hybrid_level_to_idx[level]
                hw_out, next_hw = self.model.hybrid_windows[w_idx](
                    x, state.hybrid_windows[w_idx], return_all_states=return_all_states
                )
                x = x + hw_out
                next_hybrid_windows[w_idx] = next_hw
        x = x + assoc_out
        hidden = self.model.norm(x)
        logits = self.model.lm_head(hidden)
        next_state = FractalDeltaState(
            tuple(next_levels),
            next_assoc_mem,
            next_local_tail,
            next_window,
            next_phase_pos,
            None if next_hybrid_windows is None else tuple(next_hybrid_windows),
        )
        return logits, hidden, next_state

    def step_adaptive(
        self,
        ids,
        state,
        *,
        confidence_threshold: float = 0.90,
        delta_threshold: float = 0.0,
        exit_gate_threshold: float = 0.0,
        min_layers: int = 2,
        vocab_limit: int | None = None,
    ):
        """Adaptive Early-Exit step skipping deeper Core iterations via learned Exit Gate, confidence, or delta."""
        if ids.ndim != 2 or ids.shape[0] != state.assoc_memory.shape[0] or ids.shape[1] < 1:
            raise ValueError("fractal recurrent step requires shape (batch, >=1)")
        t = ids.shape[1]
        depth = self.model.depth
        min_l = max(1, min(int(min_layers), depth))
        vlim = self.model.config.vocab_size if vocab_limit is None else int(vocab_limit)
        raw = self.model.embedding(ids)
        assoc_out, next_assoc_mem, next_local_tail = self.model._run_assoc(raw, state, execution="sequential")
        x = raw
        next_window = None
        if self.model.window_attn is not None:
            win_out, next_window = self.model.window_attn(x, state.window)
            x = x + win_out
        cur_phase = None
        next_phase_pos = None
        if self.model.use_phase_pos:
            cur_phase = state.phase_pos
            next_phase_pos = (cur_phase + t).contiguous()
        anchor = x
        next_levels = []
        next_hybrid_windows = [None] * len(self.model.hybrid_windows) if self.model.hybrid_windows else None
        v0 = None
        layers_executed = 0
        halted = False
        cached_hidden = None
        cached_logits = None
        for level, prev in enumerate(state.levels):
            if halted:
                next_levels.append(self._skip_decode_level(prev, x, return_all_states=False))
                if level in self.model._hybrid_level_to_idx:
                    w_idx = self.model._hybrid_level_to_idx[level]
                    next_hybrid_windows[w_idx] = state.hybrid_windows[w_idx]
                continue
            prev_x = x
            adapter = self.model.adapters[level] if depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.step_cells[self.model.cell_index(level)]
            if self.model.value_residual and level == 0:
                x, current, v0 = cell(
                    x, prev, adapter=adapter, return_v0=True, phase_pos=cur_phase, anchor=anchor
                )
            else:
                x, current = cell(
                    x, prev, adapter=adapter, v0=v0, phase_pos=cur_phase, anchor=anchor
                )
            next_levels.append(current)
            if level in self.model._hybrid_level_to_idx:
                w_idx = self.model._hybrid_level_to_idx[level]
                hw_out, next_hw = self.model.hybrid_windows[w_idx](x, state.hybrid_windows[w_idx])
                x = x + hw_out
                next_hybrid_windows[w_idx] = next_hw
            layers_executed += 1
            if layers_executed >= min_l and level < depth - 1:
                if exit_gate_threshold > 0.0 and self.model.exit_gate is not None:
                    cached_hidden = self.model.norm(x + assoc_out)
                    halt_p = torch.sigmoid(self.model.exit_gate(cached_hidden).float()).amin()
                    if float(halt_p.item()) >= exit_gate_threshold:
                        halted = True
                if not halted and delta_threshold > 0.0:
                    rel_delta = (x - prev_x).float().norm(dim=-1) / (x.float().norm(dim=-1).clamp_min(1e-6))
                    if float(rel_delta.max().item()) <= delta_threshold:
                        halted = True
                if not halted and confidence_threshold < 1.0:
                    cached_hidden = self.model.norm(x + assoc_out)
                    cached_logits = self.model.lm_head(cached_hidden)
                    probs = F.softmax(cached_logits[..., :vlim].float(), dim=-1)
                    if float(probs.amax(dim=-1).min().item()) >= confidence_threshold:
                        halted = True
        if cached_logits is None or not halted:
            cached_hidden = self.model.norm(x + assoc_out) if cached_hidden is None or not halted else cached_hidden
            cached_logits = self.model.lm_head(cached_hidden)
        next_state = FractalDeltaState(
            tuple(next_levels),
            next_assoc_mem,
            next_local_tail,
            next_window,
            next_phase_pos,
            None if next_hybrid_windows is None else tuple(next_hybrid_windows),
        )
        return cached_logits, cached_hidden, next_state, layers_executed

    def step_early_exit_prefix(self, ids, state, *, min_layers: int, vocab_limit: int):
        """Compiled/CUDAGraph-friendly stage 1 of Early-Exit: runs levels 0..min_layers-1 and halting signals."""
        t = ids.shape[1]
        depth = self.model.depth
        min_l = max(1, min(int(min_layers), depth))
        raw = self.model.embedding(ids)
        assoc_out, next_assoc_mem, next_local_tail = self.model._run_assoc(raw, state, execution="sequential")
        x = raw
        next_window = None
        if self.model.window_attn is not None:
            win_out, next_window = self.model.window_attn(x, state.window)
            x = x + win_out
        cur_phase = None
        next_phase_pos = None
        if self.model.use_phase_pos:
            cur_phase = state.phase_pos
            next_phase_pos = (cur_phase + t).contiguous()
        anchor = x
        prefix_levels = []
        prefix_hybrid_windows = (
            list(state.hybrid_windows) if state.hybrid_windows is not None else None
        )
        v0 = None
        prev_x = x
        for level in range(min_l):
            prev_x = x
            prev = state.levels[level]
            adapter = self.model.adapters[level] if depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.step_cells[self.model.cell_index(level)]
            if self.model.value_residual and level == 0:
                x, current, v0 = cell(
                    x, prev, adapter=adapter, return_v0=True, phase_pos=cur_phase, anchor=anchor
                )
            else:
                x, current = cell(
                    x, prev, adapter=adapter, v0=v0, phase_pos=cur_phase, anchor=anchor
                )
            prefix_levels.append(current)
            if level in self.model._hybrid_level_to_idx:
                w_idx = self.model._hybrid_level_to_idx[level]
                hw_out, next_hw = self.model.hybrid_windows[w_idx](x, state.hybrid_windows[w_idx])
                x = x + hw_out
                prefix_hybrid_windows[w_idx] = next_hw
        inter_hidden = self.model.norm(x + assoc_out)
        inter_logits = self.model.lm_head(inter_hidden)
        probs = F.softmax(inter_logits[..., :vocab_limit].float(), dim=-1)
        top_prob = probs.amax(dim=-1).amin()
        rel_delta = ((x - prev_x).float().norm(dim=-1) / x.float().norm(dim=-1).clamp_min(1e-6)).amax()
        halt_prob = (
            torch.sigmoid(self.model.exit_gate(inter_hidden).float()).amin()
            if self.model.exit_gate is not None
            else top_prob.new_zeros(())
        )
        return (
            x,
            assoc_out,
            v0,
            anchor,
            cur_phase,
            tuple(prefix_levels),
            next_assoc_mem,
            next_local_tail,
            next_window,
            next_phase_pos,
            None if prefix_hybrid_windows is None else tuple(prefix_hybrid_windows),
            inter_logits,
            top_prob,
            rel_delta,
            halt_prob,
        )

    def step_early_exit_skip(
        self,
        x,
        state,
        prefix_levels,
        next_assoc_mem,
        next_local_tail,
        next_window,
        next_phase_pos,
        prefix_hybrid_windows=None,
        *,
        min_layers: int,
    ):
        """Compiled/CUDAGraph-friendly zero-FLOP skip branch for levels min_layers..depth-1."""
        depth = self.model.depth
        min_l = max(1, min(int(min_layers), depth))
        levels = list(prefix_levels)
        for level in range(min_l, depth):
            levels.append(self._skip_decode_level(state.levels[level], x, return_all_states=False))
        return FractalDeltaState(
            tuple(levels),
            next_assoc_mem,
            next_local_tail,
            next_window,
            next_phase_pos,
            prefix_hybrid_windows,
        )

    def step_early_exit_suffix(
        self,
        x,
        assoc_out,
        v0,
        anchor,
        cur_phase,
        state,
        prefix_levels,
        next_assoc_mem,
        next_local_tail,
        next_window,
        next_phase_pos,
        prefix_hybrid_windows=None,
        *,
        min_layers: int,
    ):
        """Compiled/CUDAGraph-friendly stage 2 for hard tokens: runs levels min_layers..depth-1."""
        depth = self.model.depth
        min_l = max(1, min(int(min_layers), depth))
        levels = list(prefix_levels)
        hybrid_wins = None if prefix_hybrid_windows is None else list(prefix_hybrid_windows)
        for level in range(min_l, depth):
            prev = state.levels[level]
            adapter = self.model.adapters[level] if depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.step_cells[self.model.cell_index(level)]
            x, current = cell(
                x, prev, adapter=adapter, v0=v0, phase_pos=cur_phase, anchor=anchor
            )
            levels.append(current)
            if level in self.model._hybrid_level_to_idx:
                w_idx = self.model._hybrid_level_to_idx[level]
                hw_out, next_hw = self.model.hybrid_windows[w_idx](x, state.hybrid_windows[w_idx])
                x = x + hw_out
                hybrid_wins[w_idx] = next_hw
        hidden = self.model.norm(x + assoc_out)
        logits = self.model.lm_head(hidden)
        next_state = FractalDeltaState(
            tuple(levels),
            next_assoc_mem,
            next_local_tail,
            next_window,
            next_phase_pos,
            None if hybrid_wins is None else tuple(hybrid_wins),
        )
        return logits, next_state

    def step(self, ids, state):
        if ids.shape != (state.assoc_memory.shape[0], 1):
            raise ValueError("fractal recurrent step requires one byte")
        logits, _, next_state = self.step_hidden(ids, state)
        return logits, next_state
