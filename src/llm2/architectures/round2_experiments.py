"""Round 2 Lumina 2 Experimental Building Blocks:
- E2: Complex KDA (CKDA, signed alpha in [-1, 1], beta in [0, 2]) exact sequential & chunked DPLR
- E3: ChainDoRA (Tensor-Train DoRA adapter with <=20% trainable params vs LoRA r=32)
- E4: LLVQ24Linear (24D Golay G24 / Leech-inspired vector quantization, Hadamard-free, ~2.17 bpw)
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# E2: Complex KDA (CKDA) — real DPLR with signed alpha in [-1, 1], beta in [0, 2]
# ---------------------------------------------------------------------------

def ckda_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact sequential CKDA recurrence: S_t = S_{t-1} diag(alpha_t) (I - beta_t k_t k_t^T) + beta_t v_t k_t^T.

    Supports signed alpha in [-1, 1] (reflections) and beta in [0, 2] (negative Householder eigenvalues)
    generating complex-conjugate eigenvalues in a single real rank-1 layer.
    Layout: q (B, Hq, T, D), k/v (B, Hkv, T, D), alpha (B, Hkv, T, 1|D), beta (B, Hkv, T, 1).
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, alpha, beta = (x.float() for x in (q, k, v, alpha, beta))
        b, hq, t_len, d = q.shape
        hkv = k.shape[1]
        g_mult = hq // hkv
        mem = v.new_zeros(b, hkv, v.shape[-1], d) if state is None else state.float()
        outs = []
        for t in range(t_len):
            a_t = alpha[:, :, t]
            b_t = beta[:, :, t]
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            # S_t = S_{t-1} diag(a_t) - (S_{t-1} (a_t * b_t * k_t)) k_t^T + (b_t * v_t) k_t^T
            decayed = mem * a_t.unsqueeze(-2)
            proj = torch.matmul(decayed, k_t.unsqueeze(-1))  # (B, Hkv, V, 1)
            mem = decayed - proj * (b_t * k_t).unsqueeze(-2) + (b_t * v_t).unsqueeze(-1) * k_t.unsqueeze(-2)
            if g_mult == 1:
                outs.append(torch.matmul(mem, q[:, :, t].unsqueeze(-1)).squeeze(-1))
            else:
                qg_t = q[:, :, t].reshape(b, hkv, g_mult, d).transpose(-1, -2)
                outs.append(torch.matmul(mem, qg_t).transpose(-1, -2).reshape(b, hq, v.shape[-1]))
        return torch.stack(outs, dim=-2), mem


def ckda_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
    *,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact chunked CKDA supporting signed alpha in [-1, 1] via (cum_sign, cum_log_abs) prefix factorization."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, alpha, beta = (x.float() for x in (q, k, v, alpha, beta))
        b, hq, total_len, d = q.shape
        hkv = k.shape[1]
        g_mult = hq // hkv
        v_dim = v.shape[-1]
        w_dim = alpha.shape[-1]
        if w_dim != 1:
            # Fallback to exact sequential for per-channel signed alpha
            return ckda_sequential(q, k, v, alpha, beta, state)

        mem = v.new_zeros(b, hkv, v_dim, d) if state is None else state.float()
        pad = (-total_len) % chunk_size
        if pad:
            q, k, v, beta = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, beta))
            alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)
        n_chunks = q.shape[-2] // chunk_size

        # Signed scalar decay alpha_t = s_t * exp(log|alpha_t|), with eps floor away from exact 0
        sign_a = torch.where(alpha >= 0.0, 1.0, -1.0)
        mag_a = alpha.abs().clamp_min(1e-5)
        log_mag = mag_a.log()

        kc = k.reshape(b, hkv, n_chunks, chunk_size, d)
        vc = (beta * v).reshape(b, hkv, n_chunks, chunk_size, v_dim)
        ac = (-alpha * beta * k).reshape(b, hkv, n_chunks, chunk_size, d)
        sc = sign_a.reshape(b, hkv, n_chunks, chunk_size)
        lc = log_mag.reshape(b, hkv, n_chunks, chunk_size)

        cum_s = sc.cumprod(-1)
        prev_s = cum_s * sc  # since sc in {-1, +1}, cum_s / sc == cum_s * sc
        cum_l = lc.cumsum(-1)
        prev_l = cum_l - lc

        idx = torch.arange(chunk_size, device=q.device)
        strict = idx[:, None] > idx[None, :]
        causal = idx[:, None] >= idx[None, :]

        # Transition from step j to step i: (cum_s[i] * cum_s[j]) * exp(cum_l[i] - cum_l[j])
        sign_before = prev_s.unsqueeze(-1) * cum_s.unsqueeze(-2)
        decay_before = (
            sign_before
            * torch.where(strict, prev_l.unsqueeze(-1) - cum_l.unsqueeze(-2), 0.0).exp()
        ).masked_fill(~strict, 0.0)
        rank = torch.matmul(ac, kc.transpose(-1, -2)) * decay_before
        inputs = rank

        sign_after = cum_s.unsqueeze(-1) * cum_s.unsqueeze(-2)
        decay_after = (
            sign_after
            * torch.where(causal, cum_l.unsqueeze(-1) - cum_l.unsqueeze(-2), 0.0).exp()
        ).masked_fill(~causal, 0.0)

        eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
        prev_factor = (prev_s * prev_l.exp()).unsqueeze(-1)
        rhs = torch.cat((inputs @ vc, ac * prev_factor), dim=-1)
        solved = torch.linalg.solve_triangular(eye - rank, rhs, upper=False, unitriangular=True)
        u_local, u_state = solved.split((v_dim, d), dim=-1)

        tail_sign = cum_s[:, :, :, -1:] * cum_s
        tail = (tail_sign * (cum_l[:, :, :, -1:] - cum_l).exp()).unsqueeze(-1)
        kc_tail = kc * tail
        mem_decay = (cum_s[:, :, :, -1] * cum_l[:, :, :, -1].exp()).unsqueeze(-1).unsqueeze(-1)
        mem_local = vc.transpose(-1, -2) @ kc_tail + u_local.transpose(-1, -2) @ kc_tail
        state_trans = u_state.transpose(-1, -2) @ kc_tail

        qc_g = q.reshape(b, hkv, g_mult, n_chunks, chunk_size, d)
        reads = torch.matmul(qc_g, kc.unsqueeze(2).transpose(-1, -2)) * decay_after.unsqueeze(2)
        kv_local = torch.matmul(reads, vc.unsqueeze(2)) + torch.matmul(reads, u_local.unsqueeze(2))
        cum_factor = (cum_s * cum_l.exp()).unsqueeze(2).unsqueeze(-1)
        read_proj = qc_g * cum_factor + torch.matmul(reads, u_state.unsqueeze(2))

        kv_flat = kv_local.transpose(2, 3).reshape(b, hkv, n_chunks, g_mult * chunk_size, v_dim)
        rp_flat = read_proj.transpose(2, 3).reshape(b, hkv, n_chunks, g_mult * chunk_size, d)
        outs = []
        for m in range(n_chunks):
            outs.append(kv_flat[:, :, m] + torch.matmul(rp_flat[:, :, m], mem.transpose(-1, -2)))
            mem = mem * mem_decay[:, :, m] + mem_local[:, :, m] + mem @ state_trans[:, :, m]
        out = (
            torch.stack(outs, dim=2)
            .reshape(b, hkv, n_chunks, g_mult, chunk_size, v_dim)
            .transpose(2, 3)
            .reshape(b, hq, -1, v_dim)
        )
        return (out[:, :, :total_len] if pad else out), mem


# ---------------------------------------------------------------------------
# E3: ChainDoRA — Tensor-Train Factorized Weight-Decomposed Low-Rank Adaptation
# ---------------------------------------------------------------------------

def _factor_two(n: int) -> tuple[int, int]:
    root = int(math.isqrt(n))
    for f in range(root, 0, -1):
        if n % f == 0:
            return f, n // f
    return 1, n


class ChainDoRALinear(nn.Module):
    """DoRA adapter whose directional delta is factorized as a compact 2-mode Tensor-Train chain."""

    def __init__(self, base: nn.Linear, *, tt_rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.o1, self.o2 = _factor_two(out_f)
        self.i1, self.i2 = _factor_two(in_f)
        self.tt_rank = int(tt_rank)
        self.scale = float(alpha) / max(1, self.tt_rank)

        dev, dt = base.weight.device, torch.float32
        # Core 1: (o1, i1, r), Core 2: (r, o2, i2) -> contracts to (o1*o2, i1*i2)
        self.core1 = nn.Parameter(torch.empty(self.o1, self.i1, self.tt_rank, device=dev, dtype=dt))
        self.core2 = nn.Parameter(torch.zeros(self.tt_rank, self.o2, self.i2, device=dev, dtype=dt))
        nn.init.normal_(self.core1, std=1.0 / math.sqrt(self.i1 * self.tt_rank))
        # DoRA trainable per-output-row magnitude initialized from frozen base weight norm
        with torch.no_grad():
            mag = base.weight.detach().float().norm(dim=1).clamp_min(1e-6)
        self.magnitude = nn.Parameter(mag.to(device=dev, dtype=dt))

    def trainable_param_count(self) -> int:
        return self.core1.numel() + self.core2.numel() + self.magnitude.numel()

    def _delta_weight(self) -> torch.Tensor:
        # (o1, i1, r) @ (r, o2*i2) -> (o1, i1, o2, i2) -> (o1*o2, i1*i2)
        c1 = self.core1.reshape(self.o1 * self.i1, self.tt_rank)
        c2 = self.core2.reshape(self.tt_rank, self.o2 * self.i2)
        dw = (c1 @ c2).reshape(self.o1, self.i1, self.o2, self.i2).permute(0, 2, 1, 3)
        return dw.reshape(self.o1 * self.o2, self.i1 * self.i2) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w0 = self.base.weight.float()
        w_comb = w0 + self._delta_weight()
        denom = w_comb.norm(dim=1, keepdim=True).detach().clamp_min(1e-6)
        w_eff = (self.magnitude.unsqueeze(1) * (w_comb / denom)).to(x.dtype)
        b = None if self.base.bias is None else self.base.bias.to(x.dtype)
        return F.linear(x, w_eff, b)

    @torch.no_grad()
    def merge_to_linear(self) -> nn.Linear:
        w0 = self.base.weight.detach().float()
        w_comb = w0 + self._delta_weight().detach()
        denom = w_comb.norm(dim=1, keepdim=True).clamp_min(1e-6)
        w_eff = (self.magnitude.detach().unsqueeze(1) * (w_comb / denom)).to(self.base.weight.dtype)
        self.base.weight.copy_(w_eff)
        return self.base


# ---------------------------------------------------------------------------
# E4: LLVQ24Linear — 24D Extended Golay G24 Lattice Vector Quantization (Hadamard-free)
# ---------------------------------------------------------------------------

# Standard 12x12 parity matrix B for the [24, 12, 8] Extended Binary Golay Code G24
_GOLAY_B = (
    (1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1),
    (1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1),
    (0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1),
    (1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1),
    (1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1),
    (1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1),
    (0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1),
    (0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1),
    (0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1),
    (1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1),
    (0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0),
)

_GOLAY_CACHE: dict[torch.device, torch.Tensor] = {}


def get_golay24_bipolar_codebook(device: torch.device) -> torch.Tensor:
    """Return all 4096 bipolar {-1, +1}^24 codewords of the Extended Binary Golay Code G24 (4096 x 24)."""
    if device in _GOLAY_CACHE:
        return _GOLAY_CACHE[device]
    eye = torch.eye(12, dtype=torch.int64)
    b_mat = torch.tensor(_GOLAY_B, dtype=torch.int64)
    gen = torch.cat((eye, b_mat), dim=1)  # (12, 24)
    msgs = ((torch.arange(4096, dtype=torch.int64)[:, None] >> torch.arange(12)) & 1)  # (4096, 12)
    bits = (msgs @ gen) & 1  # (4096, 24) in {0, 1}
    cb = (bits.float() * 2.0 - 1.0).to(device=device)  # {-1, +1}^24, unit norm = sqrt(24)
    _GOLAY_CACHE[device] = cb
    return cb


class LLVQ24Linear(nn.Module):
    """Multi-stage Residual 24D Golay G24 Vector Quantization (Hadamard-free; 4 stages = 2.00 bpw indices).

    Each 24-weight block is approximated by:
        w ~= sum_{k=0}^{n_stages-1} s_k * c_{idx_k}
    where c_i in {-1, +1}^24 are codewords of the Extended Binary Golay Code G24 (4096 entries, 12 bits/stage).
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, n_stages: int = 4):
        super().__init__()
        if n_stages <= 0 or n_stages % 2 != 0:
            raise ValueError("n_stages must be a positive even integer")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_stages = int(n_stages)
        self.pad_in = (-self.in_features) % 24
        self.padded_in = self.in_features + self.pad_in
        self.n_blocks = self.padded_in // 24
        # Pack pairs of 12-bit Golay indices into 3 uint8 bytes (4 stages = 6 bytes / 24 weights = 2.00 bpw!)
        self.register_buffer(
            "packed_idx", torch.zeros((self.out_features, self.n_blocks, (self.n_stages // 2) * 3), dtype=torch.uint8)
        )
        self.register_buffer("scales", torch.ones((self.out_features, self.n_blocks, self.n_stages), dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    @staticmethod
    def _pack_12bit_pair(idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        b0 = (idx1 & 0xFF).to(torch.uint8)
        b1 = (((idx1 >> 8) & 0x0F) | ((idx2 & 0x0F) << 4)).to(torch.uint8)
        b2 = ((idx2 >> 4) & 0xFF).to(torch.uint8)
        return torch.stack((b0, b1, b2), dim=-1)

    @staticmethod
    def _unpack_12bit_pair(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = packed.to(torch.int64)
        idx1 = p[..., 0] | ((p[..., 1] & 0x0F) << 8)
        idx2 = ((p[..., 1] >> 4) & 0x0F) | (p[..., 2] << 4)
        return idx1, idx2

    @torch.no_grad()
    def pack_from_float_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None, *, chunk_rows: int = 256) -> None:
        dev = weight.device
        cb = get_golay24_bipolar_codebook(dev)  # (4096, 24)
        w = weight.detach().float()
        if self.pad_in:
            w = F.pad(w, (0, self.pad_in))
        w_blk = w.view(self.out_features, self.n_blocks, 24)

        idx_stages = [
            torch.empty((self.out_features, self.n_blocks), dtype=torch.int64, device=dev)
            for _ in range(self.n_stages)
        ]
        sc_stages = [
            torch.empty((self.out_features, self.n_blocks), dtype=torch.float16, device=dev)
            for _ in range(self.n_stages)
        ]

        for r0 in range(0, self.out_features, chunk_rows):
            r1 = min(r0 + chunk_rows, self.out_features)
            res = w_blk[r0:r1].reshape(-1, 24).clone()
            rows = r1 - r0
            for st in range(self.n_stages):
                best = (res @ cb.T).argmax(dim=-1)
                c = cb[best]
                s = ((res * c).sum(dim=-1) / 24.0).clamp_min(0.0)
                res = res - s.unsqueeze(-1) * c
                idx_stages[st][r0:r1] = best.view(rows, self.n_blocks)
                sc_stages[st][r0:r1] = s.view(rows, self.n_blocks).to(torch.float16)

        for p_i in range(self.n_stages // 2):
            self.packed_idx[..., p_i * 3 : (p_i + 1) * 3].copy_(
                self._pack_12bit_pair(idx_stages[p_i * 2], idx_stages[p_i * 2 + 1])
            )
        for st in range(self.n_stages):
            self.scales[..., st].copy_(sc_stages[st])
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.detach())

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, n_stages: int = 4) -> LLVQ24Linear:
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, n_stages=n_stages).to(
            device=linear.weight.device
        )
        layer.pack_from_float_weight(linear.weight, linear.bias)
        return layer

    def dequantize_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        cb = get_golay24_bipolar_codebook(self.packed_idx.device)
        acc = torch.zeros((self.out_features, self.n_blocks, 24), device=self.packed_idx.device, dtype=torch.float32)
        for p_i in range(self.n_stages // 2):
            i1, i2 = self._unpack_12bit_pair(self.packed_idx[..., p_i * 3 : (p_i + 1) * 3])
            acc = acc + cb[i1] * self.scales[..., p_i * 2 : p_i * 2 + 1].float()
            acc = acc + cb[i2] * self.scales[..., p_i * 2 + 1 : p_i * 2 + 2].float()
        w = acc.reshape(self.out_features, self.padded_in)
        if self.pad_in:
            w = w[:, : self.in_features]
        return w.to(dtype)

    def weight_memory_bytes(self) -> int:
        return int(self.packed_idx.numel() + self.scales.numel() * self.scales.element_size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)


def quantize_model_llvq24(model: nn.Module, *, inplace: bool = False) -> tuple[nn.Module, dict]:
    """Quantize all eligible nn.Linear layers to 2-stage 24D Golay LLVQ (Hadamard-free)."""
    from .sub4bit import _iter_quantizable_linears, _set_module_by_name

    target = model if inplace else copy.deepcopy(model)
    replaced = 0
    total_bytes = 0
    dense_bytes = 0
    for full_name, lin in _iter_quantizable_linears(target):
        q = LLVQ24Linear.from_linear(lin)
        _set_module_by_name(target, full_name, q)
        replaced += 1
        total_bytes += q.weight_memory_bytes()
        dense_bytes += lin.weight.numel() * 2
    return target, {
        "mode": "llvq24_golay",
        "replaced_count": replaced,
        "quant_weight_bytes": total_bytes,
        "dense_fp16_bytes": dense_bytes,
        "effective_bpw": round(8.0 * total_bytes / max(dense_bytes // 2, 1), 3),
    }
