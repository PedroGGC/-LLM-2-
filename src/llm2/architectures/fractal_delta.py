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
from .quarot import HadamardLinearINT4
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
class FractalDeltaState:
    levels: tuple[FractalLevelState, ...]
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
        self.head_dim = self.d_model // self.heads
        self.hidden_dim = config.hidden_dim
        self.conv_kernel = 4
        self.dropout = config.dropout
        self.execution = research.state_execution
        self.chunk_size = research.state_chunk_size
        self.beta_scale = research.fractal_beta_scale
        self.weight_mode = research.fractal_weight_mode
        self.golden_phase = research.fractal_golden_phase
        self.free_energy = research.fractal_free_energy
        self.kda = research.fractal_kda
        self.decoupled_gates = research.fractal_decoupled_gates

        self.attn_norm = RMSNorm(self.d_model, eps=config.norm_eps)
        self.conv = nn.Conv1d(self.d_model, self.d_model, self.conv_kernel, groups=self.d_model)
        self.qkv = _make_linear(self.d_model, 3 * self.d_model, self.weight_mode)
        if self.kda:
            self.decay_rank = max(16, self.heads)
            self.decay_down = nn.Linear(self.d_model, self.decay_rank, bias=False)
            self.decay = nn.Linear(self.decay_rank, self.d_model, bias=True)
        else:
            self.decay_rank = 0
            self.decay_down = None
            self.decay = nn.Linear(self.d_model, self.heads, bias=True)
        self.beta = nn.Linear(self.d_model, self.heads, bias=True)
        self.beta_write = nn.Linear(self.d_model, self.heads, bias=True) if self.decoupled_gates else None
        self.out_gate = _make_linear(self.d_model, self.d_model, self.weight_mode)
        self.head_gamma = nn.Parameter(torch.ones(self.d_model))
        self.out_proj = _make_linear(self.d_model, self.d_model, self.weight_mode)
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
                self.decay.bias.copy_(_init_decay_bias(self.heads).repeat_interleave(self.head_dim))
        else:
            nn.init.normal_(self.decay.weight, std=0.02)
            with torch.no_grad():
                self.decay.bias.copy_(_init_decay_bias(self.heads))
        nn.init.normal_(self.beta.weight, std=0.02)
        nn.init.constant_(self.beta.bias, -1.0)
        if self.beta_write is not None:
            nn.init.normal_(self.beta_write.weight, std=0.02)
            nn.init.constant_(self.beta_write.bias, -1.0)
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

        q, k, v = self.qkv(local).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if self.kda:
            decay_raw = self.decay(F.silu(self.decay_down(local))).reshape(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            decay_raw = self.decay(local).transpose(1, 2)
        beta_raw = self.beta(local).transpose(1, 2)

        with torch.autocast(device_type=x.device.type, enabled=False):
            key = F.normalize(k.float(), dim=-1, eps=1e-6)
            query = F.normalize(q.float(), dim=-1, eps=1e-6)
            if self.golden_phase and self.head_dim >= 2:
                base_pos = x.new_zeros((b,), dtype=torch.int64) if phase_pos is None else phase_pos
                pos_steps = (base_pos[:, None] + torch.arange(t, device=x.device, dtype=torch.int64)[None, :]).double()
                freq = self.golden_inv_freq.double().view(1, 1, 1, -1)
                if adapter is not None and adapter.phase_scale is not None:
                    freq = freq * adapter.phase_scale.double().view(1, self.heads, 1, 1)
                angles = (pos_steps[:, None, :, None] * freq).remainder(2.0 * math.pi).float()
                query = _apply_golden_phase(query, angles)
                key = _apply_golden_phase(key, angles)
            val_f = v.float()
            if adapter is not None and adapter.val_mix is not None and v0 is not None:
                mix = adapter.val_mix.float().sigmoid().view(1, self.heads, 1, 1)
                val_f = torch.lerp(val_f, v0, mix)

            decay_logits = decay_raw.float()
            beta_logits = beta_raw.float()
            if adapter is not None:
                decay_bias = adapter.decay_bias.float().view(1, self.heads, 1, 1) if self.kda else adapter.decay_bias.float().view(1, self.heads, 1)
                decay_logits = decay_logits + decay_bias
                beta_logits = beta_logits + adapter.beta_bias.float().view(1, self.heads, 1)

            log_w = F.logsigmoid(decay_logits) if self.kda else F.logsigmoid(decay_logits).unsqueeze(-1)
            beta_erase = (self.beta_scale * beta_logits.sigmoid()).unsqueeze(-1)
            if self.beta_write is not None:
                bw_logits = self.beta_write(local).transpose(1, 2).float()
                if adapter is not None and adapter.beta_write_bias is not None:
                    bw_logits = bw_logits + adapter.beta_write_bias.float().view(1, self.heads, 1)
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
            y = y.transpose(1, 2).contiguous().reshape(b, t, d).to(x.dtype)

        gated = y * F.silu(self.out_gate(normed)).to(y.dtype)
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
        self.head_dim = config.d_model // self.heads
        self.assoc_dim = research.memory_dim
        self.chunk_size = research.state_chunk_size
        self.value_residual = research.fractal_value_residual
        self.golden_phase = research.fractal_golden_phase
        self.free_energy = research.fractal_free_energy
        self.weight_mode = research.fractal_weight_mode
        self.leaky_budget = research.fractal_leaky_budget
        self.assoc_delta = research.fractal_assoc_delta
        self.prelude_coda = research.fractal_prelude_coda
        self.assoc_multilag = research.fractal_assoc_multilag
        self.decoupled_gates = research.fractal_decoupled_gates
        self.hybrid_stride = research.fractal_hybrid_stride
        self.use_exit_gate = research.fractal_exit_gate

        d = config.d_model
        self.assoc_val_dim = min(d, research.fractal_assoc_val_dim) if research.fractal_assoc_val_dim > 0 else min(d, 8 if d < 32 else 24)
        self.embedding = nn.Embedding(config.vocab_size, d)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Causal lag-1 shared-address associative induction channel.
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
                        self.heads,
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
        mem_shape = (b, self.heads, self.head_dim, self.head_dim)
        tail_shape = (b, d, 3)
        assoc_checks = (
            (state.assoc_memory, (b, self.assoc_dim, self.assoc_val_dim)),
            (state.local_tail, (b, d, 2)),
        )
        for tensor, expected in assoc_checks:
            if tensor.shape != expected or tensor.dtype != torch.float32 or tensor.device != x.device:
                raise ValueError("invalid fractal delta state shape/dtype")
        for level_state in state.levels:
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
        if self.golden_phase:
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

    def _skip_level_state(self, prev: FractalLevelState | None, x: torch.Tensor) -> FractalLevelState:
        b, _, d = x.shape
        tail_width = self.cells[0].conv_kernel - 1
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
        exec_mode = self.cells[0].execution
        assoc_out, next_assoc_mem, next_local_tail = self._run_assoc(raw, state, execution=exec_mode)
        x = F.dropout(raw, p=self.config.dropout, training=self.training)
        next_window = None
        if self.window_attn is not None:
            win_out, next_window = self.window_attn(x, None if state is None else state.window)
            x = x + F.dropout(win_out, p=self.config.dropout, training=self.training)
        cur_phase = None
        next_phase_pos = None
        if self.golden_phase:
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
                next_levels.append(self._skip_level_state(prev, x))
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

    def state_bytes_per_sequence(self):
        d = self.config.d_model
        levels_floats = self.depth * (self.heads * self.head_dim * self.head_dim + d * 3)
        assoc_floats = self.assoc_dim * self.assoc_val_dim + 2 * d
        win_floats = self.window_attn.state_floats_per_seq() if self.window_attn is not None else 0
        win_floats += sum(w.state_floats_per_seq() for w in self.hybrid_windows)
        phase_bytes = 8 if self.golden_phase else 0
        return (levels_floats + assoc_floats + win_floats) * 4 + phase_bytes

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
        cell0 = self.cells[0]
        decay_flops = (2 * d * cell0.decay_rank + 2 * cell0.decay_rank * d) if cell0.kda else 2 * d * h
        beta_flops = (4 * d * h) if self.decoupled_gates else (2 * d * h)
        per_level = seq_len * (10 * d * d + beta_flops + decay_flops + 6 * d * ff + 6 * d * dh)
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
            self.depth * per_level
            + assoc_flops
            + win_flops
            + exit_flops
            + (2 * seq_len * d * self.config.vocab_size if include_logits else 0)
        )
        return forward, 0, self.depth

    def parameter_formula(self):
        c = self.config
        d, h, ff, k = c.d_model, self.heads, c.hidden_dim, 4
        embed = c.vocab_size * d
        assoc_params = d + 3 * d + d + d * self.assoc_dim + d * self.assoc_val_dim + (d + 1) + self.assoc_val_dim * d + d + (d if self.assoc_multilag else 0)
        win_params = self.window_attn.parameter_count(d) if self.window_attn is not None else 0
        win_params += sum(hw.parameter_count(d) for hw in self.hybrid_windows)
        linear_scale = 4 if self.weight_mode == "quaternion" else 1
        dense_proj = (3 * d * d + 2 * d * d + 3 * d * ff) // linear_scale
        cell0 = self.cells[0]
        decay_cell = (d * cell0.decay_rank + (cell0.decay_rank + 1) * d) if cell0.kda else (d + 1) * h
        fe_cell = (d + 2 * d * cell0.fe_rank) if self.free_energy else 0
        beta_cell = 2 * (d + 1) * h if self.decoupled_gates else (d + 1) * h
        cell = (
            d  # attn_norm
            + d * (k + 1)  # depthwise conv weight + bias
            + dense_proj  # qkv, out_gate, out_proj, SwiGLU
            + decay_cell  # decay
            + beta_cell  # beta (and beta_write when decoupled)
            + d  # head_gamma
            + d  # ffn_norm
            + fe_cell
        )
        adapters = self.depth * (6 * d + 2 * h) + max(self.depth - 1, 0) * d if self.depth > 1 else 0
        if self.decoupled_gates and self.depth > 1:
            adapters += self.depth * h
        if self.value_residual and self.depth > 1:
            adapters += (self.depth - 1) * h
        if self.golden_phase and self.depth > 1:
            adapters += self.depth * h
        if self.free_energy and self.depth > 1:
            adapters += (self.depth - 1) * (d + 2)
        exit_params = (d + 1) if self.use_exit_gate else 0
        final_norm = d
        return embed + assoc_params + win_params + len(self.cells) * cell + adapters + exit_params + final_norm


class FractalDecode:
    """Incremental decode adapter for FractalDeltaMemory supporting 1-token and multi-token block steps."""

    is_recurrent = True

    def __init__(self, model):
        if type(model) is not FractalDeltaMemory or model.training:
            raise ValueError("fractal decode adapter requires evaluation FractalDeltaMemory")
        self.model = model
        self.step_cells = tuple(copy(cell) for cell in model.cells)
        for cell in self.step_cells:
            cell.execution = "sequential"

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
            tuple(
                FractalLevelState(lvl.memory[index].contiguous(), lvl.tail[index].contiguous())
                for lvl in stacked_state.levels
            ),
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

    def prefill_embeds(self, inputs_embeds: torch.Tensor, capacity: int | None = None):
        return self.prefill_hidden(None, capacity, inputs_embeds=inputs_embeds)

    def _skip_decode_level(self, prev: FractalLevelState, x: torch.Tensor, *, return_all_states: bool = False) -> FractalLevelState:
        t = x.shape[1]
        tail_width = self.step_cells[0].conv_kernel - 1
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
        if ids is not None:
            if ids.ndim != 2 or ids.shape[0] != state.levels[0].memory.shape[0] or ids.shape[1] < 1:
                raise ValueError("fractal recurrent step requires shape (batch, >=1)")
            t = ids.shape[1]
            raw = self.model.embedding(ids)
        else:
            if (
                inputs_embeds.ndim != 3
                or inputs_embeds.shape[0] != state.levels[0].memory.shape[0]
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
        if self.model.golden_phase:
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
        if ids.ndim != 2 or ids.shape[0] != state.levels[0].memory.shape[0] or ids.shape[1] < 1:
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
        if self.model.golden_phase:
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
        if self.model.golden_phase:
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
        if ids.shape != (state.levels[0].memory.shape[0], 1):
            raise ValueError("fractal recurrent step requires one byte")
        logits, _, next_state = self.step_hidden(ids, state)
        return logits, next_state
