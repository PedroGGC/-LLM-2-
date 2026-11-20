"""QuaRot-style randomized Block-Hadamard rotation with packed INT4 weights and W4A8/W4A16 inference/QAT."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def largest_pow2_divisor(n: int, max_block: int = 128) -> int:
    """Return the largest power-of-two divisor of n capped at max_block."""
    if n <= 0:
        raise ValueError("dimension must be positive")
    if max_block <= 0 or (max_block & (max_block - 1)) != 0:
        raise ValueError("max_block must be a positive power of 2")
    lowbit = n & -n
    return min(lowbit, max_block)


def hadamard_matrix(size: int, *, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Construct an orthonormal Sylvester Hadamard matrix H of shape (size, size) with H @ H.T = I."""
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {size}")
    h = torch.ones((1, 1), device=device, dtype=torch.float32)
    cur = 1
    while cur < size:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
        cur *= 2
    h = h * (1.0 / math.sqrt(float(size)))
    return h.to(dtype=dtype)


def rademacher_signs(dim: int, *, seed: int = 1729, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic {+1, -1}^dim sign vector preventing worst-case Walsh alignment."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + int(dim) * 7919)
    bits = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int64)
    signs = (bits * 2 - 1).to(dtype=dtype)
    return signs.to(device=device) if device is not None else signs


def fast_walsh_hadamard_transform(x: torch.Tensor, *, block_size: int | None = None) -> torch.Tensor:
    """Apply orthonormal block-Walsh-Hadamard transform along the last dimension (supports composite d)."""
    d = x.shape[-1]
    b_size = largest_pow2_divisor(d, 256) if block_size is None else int(block_size)
    if b_size <= 0 or (b_size & (b_size - 1)) != 0 or d % b_size != 0:
        raise ValueError(f"block_size={b_size} must be a power of 2 dividing d={d}")
    if b_size == 1:
        return x
    h_mat = hadamard_matrix(b_size, device=x.device, dtype=torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype)
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    out = (work.unflatten(-1, (-1, b_size)) @ h_mat).flatten(-2)
    return out.to(x.dtype)


def apply_hadamard_rotation(x: torch.Tensor, signs: torch.Tensor, hadamard_block: torch.Tensor) -> torch.Tensor:
    """Apply randomized orthogonal block-Hadamard transform: y = ((x * s) @ H_B)."""
    b_size = hadamard_block.shape[0]
    work = x.float() * signs.float()
    if b_size > 1:
        work = (work.unflatten(-1, (-1, b_size)) @ hadamard_block.float()).flatten(-2)
    return work.to(x.dtype)


def inverse_hadamard_rotation(y: torch.Tensor, signs: torch.Tensor, hadamard_block: torch.Tensor) -> torch.Tensor:
    """Exact inverse of apply_hadamard_rotation since H_B^T = H_B and s^2 = 1."""
    b_size = hadamard_block.shape[0]
    work = y.float()
    if b_size > 1:
        work = (work.unflatten(-1, (-1, b_size)) @ hadamard_block.float()).flatten(-2)
    work = work * signs.float()
    return work.to(y.dtype)


def pack_int4_signed(q_int8: torch.Tensor) -> torch.Tensor:
    """Pack two signed 4-bit integers in [-8, 7] per uint8 byte along an even last dimension."""
    if q_int8.shape[-1] % 2 != 0:
        raise ValueError("pack_int4_signed requires an even last dimension")
    u = (q_int8.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    low = u[..., 0::2]
    high = u[..., 1::2]
    return (low | (high << 4)).contiguous()


def unpack_int4_signed(packed_u8: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 byte tensor back to signed int8 values in [-8, 7]."""
    low = (packed_u8 & 0x0F).to(torch.int8) - 8
    high = ((packed_u8 >> 4) & 0x0F).to(torch.int8) - 8
    return torch.stack((low, high), dim=-1).flatten(-2)


def quantize_int8_dynamic(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Dynamic symmetric INT8 activation quantization per token/group with STE."""
    d = x.shape[-1]
    g = math.gcd(d, int(group_size)) or 1
    grouped = x.float().unflatten(-1, (-1, g))
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-6)
    q = (grouped / scale).round().clamp(-127.0, 127.0)
    deq = (q * scale).flatten(-2).to(x.dtype)
    return x + (deq - x).detach() if x.requires_grad else deq


class HadamardLinearINT4(nn.Module):
    """QuaRot linear layer with randomized Block-Hadamard rotation, packed uint8 INT4 weights, and W4A8/W4A16 execution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        group_size: int = 32,
        act_bits: int = 8,
        max_block: int = 128,
        seed: int = 1729,
        qat: bool = False,
    ):
        super().__init__()
        if in_features % 2 != 0:
            raise ValueError("HadamardLinearINT4 requires even in_features for INT4 byte packing")
        if act_bits not in (8, 16, 32):
            raise ValueError("act_bits must be 8 (W4A8), 16 (W4A16), or 32")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = math.gcd(self.in_features, int(group_size)) or 2
        self.n_groups = self.in_features // self.group_size
        self.act_bits = int(act_bits)
        self.block_size = largest_pow2_divisor(self.in_features, max_block)
        self.qat = bool(qat)

        self.register_buffer("signs", rademacher_signs(self.in_features, seed=seed))
        self.register_buffer("hadamard_block", hadamard_matrix(self.block_size))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        if self.qat:
            self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
            nn.init.normal_(self.weight, std=0.02)
            self.register_buffer("packed_weight", None)
            self.register_buffer("weight_scale", None)
        else:
            self.register_parameter("weight", None)
            self.register_buffer(
                "packed_weight",
                torch.zeros((self.out_features, self.in_features // 2), dtype=torch.uint8),
            )
            self.register_buffer(
                "weight_scale",
                torch.ones((self.out_features, self.n_groups), dtype=torch.float16),
            )

    def reset_weight(self, std: float) -> None:
        if self.weight is not None:
            nn.init.normal_(self.weight, std=std)

    @torch.no_grad()
    def pack_from_float_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        """Rotate float weights offline by Block-Hadamard and pack into 2xINT4 per uint8 byte."""
        w_rot = apply_hadamard_rotation(weight.detach().float(), self.signs, self.hadamard_block)
        grouped = w_rot.view(self.out_features, self.n_groups, self.group_size)
        scale = (grouped.abs().amax(dim=-1) / 7.0).clamp_min(1e-6)
        q = (grouped / scale.unsqueeze(-1)).round().clamp(-7.0, 7.0).to(torch.int8).view(self.out_features, self.in_features)
        packed = pack_int4_signed(q)
        if self.packed_weight is None:
            self.register_buffer("packed_weight", packed)
        else:
            self.packed_weight.copy_(packed)
        scale_f16 = scale.to(torch.float16)
        if self.weight_scale is None:
            self.register_buffer("weight_scale", scale_f16)
        else:
            self.weight_scale.copy_(scale_f16)
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.detach())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int = 32,
        act_bits: int = 8,
        max_block: int = 128,
        seed: int = 1729,
    ) -> HadamardLinearINT4:
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            act_bits=act_bits,
            max_block=max_block,
            seed=seed,
            qat=False,
        ).to(device=linear.weight.device)
        layer.pack_from_float_weight(linear.weight, linear.bias)
        return layer

    def dequantize_rotated_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Unpack and dequantize the rotated INT4 weight matrix."""
        q = unpack_int4_signed(self.packed_weight).to(torch.float32)
        scale = self.weight_scale.to(torch.float32).unsqueeze(-1)
        w_rot = (q.view(self.out_features, self.n_groups, self.group_size) * scale).view(self.out_features, self.in_features)
        return w_rot.to(dtype)

    def weight_memory_bytes(self) -> int:
        if self.qat and self.weight is not None:
            return self.weight.numel() * self.weight.element_size()
        packed_b = self.packed_weight.numel() * self.packed_weight.element_size()
        scale_b = self.weight_scale.numel() * self.weight_scale.element_size()
        return int(packed_b + scale_b)

    def dense_memory_bytes(self, dtype_bytes: int = 2) -> int:
        return int(self.out_features * self.in_features * dtype_bytes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rot = apply_hadamard_rotation(x, self.signs, self.hadamard_block)
        if self.act_bits == 8:
            x_rot = quantize_int8_dynamic(x_rot, group_size=self.group_size)
        if self.qat and self.weight is not None:
            w_rot = apply_hadamard_rotation(self.weight, self.signs, self.hadamard_block)
            grouped = w_rot.float().view(self.out_features, self.n_groups, self.group_size)
            scale = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-6)
            w_q = ((grouped / scale).round().clamp(-7.0, 7.0) * scale).view(self.out_features, self.in_features).to(x.dtype)
            w_eff = w_rot + (w_q - w_rot).detach()
        else:
            w_eff = self.dequantize_rotated_weight(dtype=x_rot.dtype)
        bias = None if self.bias is None else self.bias.to(x_rot.dtype)
        return F.linear(x_rot, w_eff, bias)


_SKIP_LINEAR_NAMES = frozenset({"lm_head", "assoc_write_gate", "assoc_address", "assoc_value", "decay", "beta"})


def quantize_model_quarot_int4(
    model: nn.Module,
    *,
    group_size: int = 32,
    act_bits: int = 8,
    max_block: int = 128,
    include_assoc: bool = True,
    inplace: bool = True,
    seed: int = 1729,
) -> tuple[nn.Module, dict]:
    """Convert dense Linear layers of a model into packed HadamardLinearINT4 while preserving shared embeddings."""
    target = model if inplace else copy.deepcopy(model)
    emb_weight = getattr(getattr(target, "embedding", None), "weight", None)
    replaced: list[str] = []
    dense_bf16_bytes = 0
    dense_fp32_bytes = 0
    quarot_bytes = 0

    def _convert(parent: nn.Module, prefix: str = "") -> None:
        nonlocal dense_bf16_bytes, dense_fp32_bytes, quarot_bytes
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if name in _SKIP_LINEAR_NAMES or full_name in _SKIP_LINEAR_NAMES:
                    continue
                if not include_assoc and full_name.startswith("assoc_"):
                    continue
                if emb_weight is not None and child.weight is emb_weight:
                    continue
                if child.in_features % 2 != 0 or child.out_features < 8:
                    continue
                q_layer = HadamardLinearINT4.from_linear(
                    child,
                    group_size=group_size,
                    act_bits=act_bits,
                    max_block=max_block,
                    seed=seed + len(replaced),
                )
                setattr(parent, name, q_layer)
                replaced.append(full_name)
                dense_bf16_bytes += q_layer.dense_memory_bytes(2)
                dense_fp32_bytes += q_layer.dense_memory_bytes(4)
                quarot_bytes += q_layer.weight_memory_bytes()
            else:
                _convert(child, full_name)

    _convert(target)
    shared_ok = True
    if hasattr(target, "lm_head") and hasattr(target, "embedding"):
        shared_ok = bool(target.lm_head.weight is target.embedding.weight)
    report = {
        "replaced_count": len(replaced),
        "replaced_layers": replaced,
        "group_size": group_size,
        "act_bits": act_bits,
        "dense_bf16_bytes": dense_bf16_bytes,
        "dense_fp32_bytes": dense_fp32_bytes,
        "quarot_int4_bytes": quarot_bytes,
        "compression_vs_bf16": (dense_bf16_bytes / quarot_bytes) if quarot_bytes else 1.0,
        "compression_vs_fp32": (dense_fp32_bytes / quarot_bytes) if quarot_bytes else 1.0,
        "shared_embedding_preserved": shared_ok,
    }
    return target, report
