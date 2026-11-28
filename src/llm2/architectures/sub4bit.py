"""Wave 2 Sub-4-Bit Mathematical Compression Tournament:
- 2A: E8 Lattice 2-bit Vector Quantizer (2-stage residual E8-240 root codebook + Block-Hadamard, ~2.25 bpw)
- 2B: Kintsugi GUE/Spectral-Guided Mixed-Precision Allocator (automatic 4-bit / 2-bit E8 per-layer budget)
- 2C: ParetoQ Ternary (1.58-bit) / 2-bit QAT with learnable group scale & STE + packed inference
- 2D: Bit-Plane Progressive Linear (MSB->LSB 1-bit planes allowing 2/3/4-bit reads from one representation)
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .quarot import (
    HadamardLinearINT4,
    _SKIP_LINEAR_NAMES,
    apply_hadamard_rotation,
    hadamard_matrix,
    largest_pow2_divisor,
    rademacher_signs,
)


def build_e8_root_codebook(*, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build exact 256x8 codebook: 240 unit-normalized E8 minimal roots + 16 axis/zero vectors (1 byte/8 weights)."""
    roots: list[list[float]] = []
    # 1) 112 Type-1 E8 roots: permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0) normalized to unit L2 norm (1/sqrt(2))
    inv_s2 = 1.0 / math.sqrt(2.0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-inv_s2, inv_s2):
                for sj in (-inv_s2, inv_s2):
                    v = [0.0] * 8
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    # 2) 128 Type-2 E8 roots: (+-1/2)^8 with even number of minus signs (already unit L2 norm: sqrt(8 * 1/4) = sqrt(2), so /sqrt(2) -> +-1/(2*sqrt(2)))
    half_norm = 0.5 / math.sqrt(2.0)
    for mask in range(256):
        if bin(mask).count("1") % 2 == 0:
            roots.append([-half_norm if (mask >> bit) & 1 else half_norm for bit in range(8)])
    assert len(roots) == 240
    # 3) 16 axis vectors (+-e_k) for sharp sparse residual capture (15 unit axes + 0 vector = 256 exact uint8 entries)
    roots.append([0.0] * 8)
    for k in range(8):
        v = [0.0] * 8
        v[k] = 1.0
        roots.append(v)
    for k in range(7):
        v = [0.0] * 8
        v[k] = -1.0
        roots.append(v)
    assert len(roots) == 256
    return torch.tensor(roots, device=device, dtype=dtype)


class E8Linear2Bit(nn.Module):
    """2-stage Residual E8 Lattice Vector Quantizer (2.00 bpw codebook indices + FP16 group scale = 2.25 bpw)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        group_size: int = 64,
        max_block: int = 128,
        seed: int = 1729,
        rotate_input: bool = True,
        stage2_ratio: float = 0.5,
    ):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError(f"E8Linear2Bit requires in_features divisible by 8, got {in_features}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        g = math.gcd(self.in_features, max(8, int(group_size)))
        self.group_size = g if g % 8 == 0 else 8
        self.n_groups = self.in_features // self.group_size
        self.n_blocks8 = self.in_features // 8
        self.rotate_input = bool(rotate_input)
        self.stage2_ratio = float(stage2_ratio)
        self.block_size = largest_pow2_divisor(self.in_features, max_block)

        self.register_buffer("codebook", build_e8_root_codebook(dtype=torch.float16))
        self.register_buffer("signs", rademacher_signs(self.in_features, seed=seed))
        self.register_buffer("hadamard_block", hadamard_matrix(self.block_size))
        self.register_buffer("idx1", torch.zeros((self.out_features, self.n_blocks8), dtype=torch.uint8))
        self.register_buffer("idx2", torch.zeros((self.out_features, self.n_blocks8), dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.ones((self.out_features, self.n_groups), dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    @torch.no_grad()
    def pack_from_float_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None, *, row_chunk: int = 1024) -> None:
        """Rotate float weights by Block-Hadamard (optional) and quantize each 8D block via 2-stage residual E8."""
        dev = weight.device
        w_f = weight.detach().float()
        if self.rotate_input:
            w_f = apply_hadamard_rotation(w_f, self.signs.to(dev), self.hadamard_block.to(dev))
        cb = self.codebook.to(device=dev, dtype=torch.float32)  # (256, 8)
        cb_t = cb.t().contiguous()  # (8, 256)
        cb_norm2 = cb.square().sum(dim=-1)  # (256,)
        r2 = self.stage2_ratio

        # Initialize per-group scale for E8 projection (0.80 * mean 8D norm matches E8 shell covering radius)
        grouped = w_f.view(self.out_features, self.n_groups, self.group_size)
        block_norms = grouped.view(self.out_features, self.n_groups, self.group_size // 8, 8).norm(dim=-1)
        scale = (0.80 * block_norms.mean(dim=-1)).clamp_min(1e-6)  # (out_features, n_groups)

        idx1_out = torch.empty((self.out_features, self.n_blocks8), device=dev, dtype=torch.uint8)
        idx2_out = torch.empty((self.out_features, self.n_blocks8), device=dev, dtype=torch.uint8)
        scale_opt = torch.empty_like(scale)
        blocks_per_group = self.group_size // 8

        for r0 in range(0, self.out_features, row_chunk):
            r1 = min(r0 + row_chunk, self.out_features)
            sc_exp = scale[r0:r1].repeat_interleave(blocks_per_group, dim=1).unsqueeze(-1)  # (R, B8, 1)
            v = (w_f[r0:r1].view(r1 - r0, self.n_blocks8, 8) / sc_exp).reshape(-1, 8)  # (N, 8)
            # Stage 1: minimize ||v - c||^2 = ||v||^2 + ||c||^2 - 2 v·c
            dist1 = cb_norm2.unsqueeze(0) - 2.0 * (v @ cb_t)
            i1 = dist1.argmin(dim=-1)
            res = v - cb[i1]
            # Stage 2: minimize ||res - r2 * c||^2
            dist2 = (r2 * r2) * cb_norm2.unsqueeze(0) - (2.0 * r2) * (res @ cb_t)
            i2 = dist2.argmin(dim=-1)
            # Closed-form L2-optimal per-group scale refit (zero extra bits, +1-2 dB SNR!)
            q_unit = (cb[i1] + r2 * cb[i2]).view(r1 - r0, self.n_groups, self.group_size)
            w_grp = w_f[r0:r1].view(r1 - r0, self.n_groups, self.group_size)
            num = (w_grp * q_unit).sum(dim=-1)
            den = q_unit.square().sum(dim=-1).clamp_min(1e-12)
            scale_opt[r0:r1] = (num / den).clamp_min(1e-6)
            idx1_out[r0:r1] = i1.to(torch.uint8).view(r1 - r0, self.n_blocks8)
            idx2_out[r0:r1] = i2.to(torch.uint8).view(r1 - r0, self.n_blocks8)

        self.idx1.copy_(idx1_out.to(self.idx1.device))
        self.idx2.copy_(idx2_out.to(self.idx2.device))
        self.weight_scale.copy_(scale_opt.to(device=self.weight_scale.device, dtype=torch.float16))
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.detach().to(self.bias.device))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int = 64,
        max_block: int = 128,
        seed: int = 1729,
        rotate_input: bool = True,
    ) -> E8Linear2Bit:
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            max_block=max_block,
            seed=seed,
            rotate_input=rotate_input,
        ).to(device=linear.weight.device)
        layer.pack_from_float_weight(linear.weight, linear.bias)
        return layer

    def dequantize_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        cb = self.codebook.to(dtype)
        w8 = cb[self.idx1.long()] + self.stage2_ratio * cb[self.idx2.long()]  # (out, n_blocks8, 8)
        w = w8.view(self.out_features, self.n_groups, self.group_size)
        return (w * self.weight_scale.to(dtype).unsqueeze(-1)).view(self.out_features, self.in_features)

    def enable_qad(self) -> list[nn.Parameter]:
        """Convert codebook (256x8) and weight_scale to trainable FP32 Parameters for zero-overhead AQLM-style QAD."""
        cb_p = nn.Parameter(self.codebook.detach().float())
        sc_p = nn.Parameter(self.weight_scale.detach().float())
        # Remove buffers first before assigning Parameters of the same name
        del self._buffers["codebook"]
        del self._buffers["weight_scale"]
        self.codebook = cb_p
        self.weight_scale = sc_p
        return [cb_p, sc_p]

    @torch.no_grad()
    def freeze_qad(self) -> None:
        """Cast trained QAD codebook and weight_scale back to compact FP16 buffers."""
        if isinstance(self.codebook, nn.Parameter):
            cb_buf = self.codebook.detach().to(torch.float16).contiguous()
            del self._parameters["codebook"]
            self.register_buffer("codebook", cb_buf)
        if isinstance(self.weight_scale, nn.Parameter):
            sc_buf = self.weight_scale.detach().to(torch.float16).contiguous()
            del self._parameters["weight_scale"]
            self.register_buffer("weight_scale", sc_buf)

    def weight_memory_bytes(self) -> int:
        return int(
            self.idx1.numel() * self.idx1.element_size()
            + self.idx2.numel() * self.idx2.element_size()
            + self.weight_scale.numel() * self.weight_scale.element_size()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = apply_hadamard_rotation(x, self.signs, self.hadamard_block) if self.rotate_input else x
        w_eff = self.dequantize_weight(dtype=x_in.dtype)
        b = None if self.bias is None else self.bias.to(x_in.dtype)
        return F.linear(x_in, w_eff, b)


def pack_2bit_unsigned(q_u8: torch.Tensor) -> torch.Tensor:
    """Pack four 2-bit codes in {0, 1, 2, 3} per uint8 byte along last dim (divisible by 4)."""
    if q_u8.shape[-1] % 4 != 0:
        raise ValueError("pack_2bit_unsigned requires last dimension divisible by 4")
    u = q_u8.to(torch.uint8) & 0x03
    return (u[..., 0::4] | (u[..., 1::4] << 2) | (u[..., 2::4] << 4) | (u[..., 3::4] << 6)).contiguous()


def unpack_2bit_unsigned(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 byte tensor back to 2-bit codes in {0, 1, 2, 3}."""
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    return torch.stack((b0, b1, b2, b3), dim=-1).flatten(-2)


class ParetoQLinear(nn.Module):
    """ParetoQ Ternary (1.58-bit {-1,0,+1}) or Symmetric 2-bit ({-1.5,-0.5,+0.5,+1.5}) QAT + Packed Inference."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        mode: str = "ternary",
        group_size: int = 64,
        qat: bool = True,
    ):
        super().__init__()
        if mode not in ("ternary", "2bit"):
            raise ValueError("ParetoQLinear mode must be 'ternary' or '2bit'")
        if in_features % 4 != 0:
            raise ValueError("ParetoQLinear requires in_features divisible by 4")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mode = mode
        g = math.gcd(self.in_features, max(4, int(group_size)))
        self.group_size = g if g % 4 == 0 else 4
        self.n_groups = self.in_features // self.group_size
        self.qat = bool(qat)
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        if self.qat:
            self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
            self.scale = nn.Parameter(torch.ones(self.out_features, self.n_groups))
            nn.init.normal_(self.weight, std=0.02)
            self.register_buffer("packed_weight", None)
            self.register_buffer("weight_scale", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("scale", None)
            self.register_buffer("packed_weight", torch.zeros((self.out_features, self.in_features // 4), dtype=torch.uint8))
            self.register_buffer("weight_scale", torch.ones((self.out_features, self.n_groups), dtype=torch.float16))

    @torch.no_grad()
    def init_from_linear(self, linear: nn.Linear) -> None:
        w = linear.weight.detach().float()
        grouped = w.view(self.out_features, self.n_groups, self.group_size)
        # Optimal L2-minimizing initial step size for Gaussian weights
        init_scale = (grouped.abs().mean(dim=-1) * (1.25 if self.mode == "ternary" else 1.0)).clamp_min(1e-6)
        if self.qat:
            self.weight.copy_(linear.weight.detach())
            self.scale.copy_(init_scale.to(self.scale.dtype))
        else:
            self._pack_tensor(w, init_scale)
        if linear.bias is not None and self.bias is not None:
            self.bias.copy_(linear.bias.detach())

    @torch.no_grad()
    def _pack_tensor(self, w_f: torch.Tensor, sc_f: torch.Tensor) -> None:
        grouped = w_f.view(self.out_features, self.n_groups, self.group_size)
        sc = sc_f.clamp_min(1e-6).unsqueeze(-1)
        if self.mode == "ternary":
            q = (grouped / sc).round().clamp(-1.0, 1.0)
            codes = (q + 1.0).to(torch.uint8).view(self.out_features, self.in_features)  # {0, 1, 2}
        else:
            # {-1.5, -0.5, +0.5, +1.5} -> codes {0, 1, 2, 3}
            codes = ((grouped / sc) + 1.5).round().clamp(0.0, 3.0).to(torch.uint8).view(self.out_features, self.in_features)
        packed = pack_2bit_unsigned(codes)
        if self.packed_weight is None:
            self.register_buffer("packed_weight", packed)
        else:
            self.packed_weight.copy_(packed)
        sc16 = sc_f.clamp_min(1e-6).to(torch.float16)
        if self.weight_scale is None:
            self.register_buffer("weight_scale", sc16)
        else:
            self.weight_scale.copy_(sc16)

    @torch.no_grad()
    def freeze_to_packed(self) -> None:
        """Convert QAT float weights + learnable scale into 2-bit packed inference buffers and free FP weights."""
        if not self.qat or self.weight is None:
            return
        self._pack_tensor(self.weight.detach().float(), self.scale.detach().float())
        self.register_parameter("weight", None)
        self.register_parameter("scale", None)
        self.qat = False

    def dequantize_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        codes = unpack_2bit_unsigned(self.packed_weight).to(dtype)
        vals = (codes - 1.0) if self.mode == "ternary" else (codes - 1.5)
        w = vals.view(self.out_features, self.n_groups, self.group_size) * self.weight_scale.to(dtype).unsqueeze(-1)
        return w.view(self.out_features, self.in_features)

    def weight_memory_bytes(self) -> int:
        if self.qat and self.weight is not None:
            return int(self.weight.numel() * self.weight.element_size() + self.scale.numel() * self.scale.element_size())
        return int(
            self.packed_weight.numel() * self.packed_weight.element_size()
            + self.weight_scale.numel() * self.weight_scale.element_size()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = None if self.bias is None else self.bias.to(x.dtype)
        if self.qat and self.weight is not None:
            w_f = self.weight.float().view(self.out_features, self.n_groups, self.group_size)
            sc = self.scale.float().abs().clamp_min(1e-6).unsqueeze(-1)
            if self.mode == "ternary":
                u = (w_f / sc).clamp(-1.0, 1.0)
                u_ste = u + (u.round() - u).detach()
                w_eff = (u_ste * sc).view(self.out_features, self.in_features).to(x.dtype)
            else:
                u = ((w_f / sc) + 1.5).clamp(0.0, 3.0)
                u_ste = u + (u.round() - u).detach()
                w_eff = ((u_ste - 1.5) * sc).view(self.out_features, self.in_features).to(x.dtype)
            return F.linear(x, w_eff, b)
        return F.linear(x, self.dequantize_weight(dtype=x.dtype), b)


class BitPlaneLinear(nn.Module):
    """Bit-plane progressive INT4 storage: 4 separate 1-bit planes (MSB->LSB) supporting 2/3/4-bit reads on the fly."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, group_size: int = 32, active_bits: int = 4):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError("BitPlaneLinear requires in_features divisible by 8")
        if active_bits not in (2, 3, 4):
            raise ValueError("active_bits must be 2, 3, or 4")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = math.gcd(self.in_features, max(8, int(group_size)))
        self.n_groups = self.in_features // self.group_size
        self.active_bits = int(active_bits)
        # 4 bit-planes packed 8 bits per uint8: plane 0 = MSB (bit 3), plane 3 = LSB (bit 0)
        self.register_buffer("planes", torch.zeros((4, self.out_features, self.in_features // 8), dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.ones((self.out_features, self.n_groups), dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    @torch.no_grad()
    def pack_from_float_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        w = weight.detach().float().view(self.out_features, self.n_groups, self.group_size)
        sc = (w.abs().amax(dim=-1) / 7.5).clamp_min(1e-6)
        # Symmetric offset-binary codes in [0..15] representing [-7.5, -6.5, ..., +7.5]
        u = ((w / sc.unsqueeze(-1)) + 7.5).round().clamp(0.0, 15.0).to(torch.uint8).view(self.out_features, self.in_features)
        u8 = u.view(self.out_features, self.in_features // 8, 8)
        shifts = torch.arange(8, device=u.device, dtype=torch.uint8)
        for p_idx, bit_pos in enumerate((3, 2, 1, 0)):  # MSB -> LSB
            bits = ((u8 >> bit_pos) & 1) << shifts
            self.planes[p_idx].copy_(bits.sum(dim=-1).to(torch.uint8))
        self.weight_scale.copy_(sc.to(torch.float16))
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.detach())

    def dequantize_progressive(self, bits: int | None = None, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        k = self.active_bits if bits is None else int(bits)
        shifts = torch.arange(8, device=self.planes.device, dtype=torch.uint8)
        acc = torch.zeros((self.out_features, self.in_features // 8, 8), device=self.planes.device, dtype=torch.float32)
        for p_idx, bit_pos in enumerate((3, 2, 1, 0)[:k]):
            bits_f = ((self.planes[p_idx].unsqueeze(-1) >> shifts) & 1).float()
            acc = acc + bits_f * float(1 << bit_pos)
        # Add midpoint rounding bias for unread lower planes so truncation is unbiased!
        if k < 4:
            acc = acc + float((1 << (4 - k)) - 1) * 0.5
        vals = (acc - 7.5).view(self.out_features, self.n_groups, self.group_size)
        return (vals * self.weight_scale.float().unsqueeze(-1)).view(self.out_features, self.in_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_progressive(self.active_bits, dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)


@torch.no_grad()
def compute_gue_spectral_sensitivity(linear: nn.Linear, name: str, *, halo_levels: set[int] | None = None) -> float:
    """Compute Kintsugi GUE-inspired sensitivity score (outlier kurtosis + layer/HALO structural criticality)."""
    w = linear.weight.detach().float()
    # Normalized fourth moment (kurtosis) measures non-Gaussian bulk-escape spikes that suffer most at 2-bit
    var = w.var().clamp_min(1e-12)
    kurt = float(((w - w.mean()).pow(4).mean() / (var * var)).item())
    # Structural prior: attention projections & HALO global / boundary layers carry highest KL sensitivity
    prior = 1.0
    if any(tok in name for tok in ("q_proj", "k_proj", "v_proj", "out_proj", "qkv")):
        prior *= 1.6
    if "ffn.down" in name:
        prior *= 1.25
    if halo_levels is not None:
        for part in name.split("."):
            if part.isdigit():
                lvl = int(part)
                if lvl in halo_levels:
                    prior *= 1.5
                if lvl in (0, 1):
                    prior *= 1.4
                break
    return prior * math.log1p(max(kurt, 1.0))


@torch.no_grad()
def compute_mp_spectral_sensitivity(
    linear: nn.Linear,
    name: str,
    *,
    halo_levels: set[int] | None = None,
    top_k: int = 16,
) -> float:
    """Marchenko-Pastur spectral sensitivity: fraction of singular energy escaping the MP bulk edge sigma_+."""
    w = linear.weight.detach().float()
    m, n = w.shape
    sigma = float(w.std().clamp_min(1e-8).item())
    # Marchenko-Pastur upper bulk edge for (m x n) matrix: sigma * (sqrt(m) + sqrt(n))
    mp_edge = sigma * (math.sqrt(m) + math.sqrt(n))
    k = min(int(top_k), min(m, n))
    try:
        _, s, _ = torch.svd_lowrank(w, q=k, niter=2)
    except Exception:
        return compute_gue_spectral_sensitivity(linear, name, halo_levels=halo_levels)
    frob_sq = float(w.square().sum().clamp_min(1e-12).item())
    excess_sq = float((s.square() - mp_edge * mp_edge).clamp_min(0.0).sum().item())
    spike_ratio = excess_sq / frob_sq
    edge_ratio = float(s[0].item()) / max(mp_edge, 1e-8)

    prior = 1.0
    if any(tok in name for tok in ("q_proj", "k_proj", "v_proj", "out_proj", "qkv")):
        prior *= 1.6
    if "ffn.down" in name:
        prior *= 1.25
    if halo_levels is not None:
        for part in name.split("."):
            if part.isdigit():
                lvl = int(part)
                if lvl in halo_levels:
                    prior *= 1.5
                if lvl in (0, 1):
                    prior *= 1.4
                break
    return prior * (1.0 + 4.0 * spike_ratio + math.log1p(max(edge_ratio - 1.0, 0.0)))


def _iter_quantizable_linears(model: nn.Module):
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in _SKIP_LINEAR_NAMES:
            continue
        if module.in_features % 8 != 0 or module.in_features < 64:
            continue
        yield full_name, module


def _set_module_by_name(root: nn.Module, full_name: str, new_mod: nn.Module) -> None:
    if "." in full_name:
        parent_name, attr = full_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent, attr = root, full_name
    setattr(parent, attr, new_mod)


def quantize_model_e8_2bit(
    model: nn.Module,
    *,
    group_size: int = 64,
    inplace: bool = False,
    rotate_input: bool = True,
) -> tuple[nn.Module, dict]:
    """Quantize all eligible nn.Linear layers to 2-stage Residual E8 Lattice 2-bit (~2.25 bpw)."""
    target = model if inplace else copy.deepcopy(model)
    replaced = 0
    for idx, (full_name, lin) in enumerate(_iter_quantizable_linears(target)):
        q_lin = E8Linear2Bit.from_linear(
            lin, group_size=group_size, seed=1729 + idx * 31, rotate_input=rotate_input
        )
        _set_module_by_name(target, full_name, q_lin)
        replaced += 1
    return target, {"mode": "e8_2bit", "replaced_count": replaced, "rotate_input": rotate_input}


def quantize_model_kintsugi_mixed(
    model: nn.Module,
    *,
    high_bit_fraction: float = 0.20,
    halo_levels: list[int] | None = None,
    inplace: bool = False,
    use_triton: bool = True,
    rotate_input: bool = True,
) -> tuple[nn.Module, dict]:
    """Kintsugi GUE-guided mixed-precision: top `high_bit_fraction` sensitive layers -> INT4, rest -> E8 2-bit."""
    target = model if inplace else copy.deepcopy(model)
    halo_set = set(halo_levels) if halo_levels is not None else None
    candidates = list(_iter_quantizable_linears(target))
    scored = [
        (compute_gue_spectral_sensitivity(lin, name, halo_levels=halo_set), idx, name, lin)
        for idx, (name, lin) in enumerate(candidates)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    n_high = max(1, int(round(len(scored) * float(high_bit_fraction)))) if scored else 0
    high_names = {item[2] for item in scored[:n_high]}

    n_int4 = 0
    n_e8 = 0
    for _, idx, name, lin in scored:
        if name in high_names:
            q_lin = HadamardLinearINT4.from_linear(
                lin,
                group_size=32,
                act_bits=16,
                seed=1729 + idx * 31,
                use_triton=use_triton,
                rotate_input=False,
            )
            n_int4 += 1
        else:
            q_lin = E8Linear2Bit.from_linear(
                lin,
                group_size=64,
                seed=1729 + idx * 31,
                rotate_input=rotate_input,
            )
            n_e8 += 1
        _set_module_by_name(target, name, q_lin)
    return target, {
        "mode": "kintsugi_gue_mixed",
        "high_bit_fraction": high_bit_fraction,
        "int4_layers": n_int4,
        "e8_2bit_layers": n_e8,
        "replaced_count": n_int4 + n_e8,
    }


def quantize_model_paretoq(
    model: nn.Module,
    *,
    mode: str = "ternary",
    group_size: int = 64,
    qat: bool = False,
    inplace: bool = False,
) -> tuple[nn.Module, dict]:
    """Convert all eligible nn.Linear layers to ParetoQ Ternary (1.58-bit) or 2-bit QAT / packed inference."""
    target = model if inplace else copy.deepcopy(model)
    replaced = 0
    for full_name, lin in _iter_quantizable_linears(target):
        q_lin = ParetoQLinear(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            mode=mode,
            group_size=group_size,
            qat=qat,
        ).to(device=lin.weight.device, dtype=lin.weight.dtype if qat else torch.float16)
        q_lin.init_from_linear(lin)
        _set_module_by_name(target, full_name, q_lin)
        replaced += 1
    return target, {"mode": f"paretoq_{mode}", "qat": qat, "replaced_count": replaced}
