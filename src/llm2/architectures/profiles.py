"""Perfis de quantização próprios `Lumina-Q8/Q6/Q5/Q4`: INT8 nas camadas sensíveis, INT4 (W4A16 Fast) no resto.

Seleção por sensibilidade (kurtosis + prior estrutural de `compute_gue_spectral_sensitivity`) até cobrir a fração
de PARÂMETROS pedida em INT8 (não fração de camadas), para o tamanho final bater com o alvo.
`lm_head`/embedding compartilhado e gates pequenos ficam em FP16 (mesma regra do QuaRot).
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .quarot import HadamardLinearINT4
from .sub4bit import _iter_quantizable_linears, _set_module_by_name, compute_gue_spectral_sensitivity

# fração de parâmetros em INT8; o restante vai para INT4 g32
PROFILE_INT8_FRACTION = {"q8": 1.0, "q6": 0.5, "q5": 0.25, "q4": 0.0}


class Int8GroupLinear(nn.Module):
    """Linear com pesos INT8 simétricos por grupo + escala FP16 (dequantiza para o dtype da ativação).

    shortcut: dequantização em PyTorch puro (sem kernel fundido), serve para qualidade/VRAM; fundir em Triton
    antes de medir tok/s.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, group_size: int = 128):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = math.gcd(self.in_features, int(group_size)) or 1
        self.n_groups = self.in_features // self.group_size
        self.register_buffer("qweight", torch.zeros((self.out_features, self.in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones((self.out_features, self.n_groups), dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    @torch.no_grad()
    def pack_from_float_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        w = weight.detach().float().view(self.out_features, self.n_groups, self.group_size)
        sc = (w.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
        q = (w / sc.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        self.qweight.copy_(q.view(self.out_features, self.in_features))
        self.weight_scale.copy_(sc.to(torch.float16))
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.detach())

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, group_size: int = 128) -> Int8GroupLinear:
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, group_size=group_size)
        layer = layer.to(device=linear.weight.device)
        layer.pack_from_float_weight(linear.weight, linear.bias)
        return layer

    def dequantize_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        w = self.qweight.view(self.out_features, self.n_groups, self.group_size).to(torch.float32)
        w = w * self.weight_scale.to(torch.float32).unsqueeze(-1)
        return w.view(self.out_features, self.in_features).to(dtype)

    def weight_memory_bytes(self) -> int:
        return int(self.qweight.numel() + self.weight_scale.numel() * self.weight_scale.element_size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantize_weight(x.dtype), b)


def quantize_model_profile(
    model: nn.Module,
    profile: str,
    *,
    halo_levels: list[int] | None = None,
    int8_group_size: int = 128,
    int4_group_size: int = 32,
    inplace: bool = False,
    use_triton: bool = True,
) -> tuple[nn.Module, dict]:
    """Aplica `q8|q6|q5|q4`. Retorna (modelo, info) com bytes de pesos quantizados por tipo."""
    if profile not in PROFILE_INT8_FRACTION:
        raise ValueError(f"profile deve ser um de {sorted(PROFILE_INT8_FRACTION)}")
    target = model if inplace else copy.deepcopy(model)
    halo_set = set(halo_levels) if halo_levels is not None else None
    cands = [(n, m) for n, m in _iter_quantizable_linears(target) if m.in_features % int4_group_size == 0]
    total = sum(m.weight.numel() for _, m in cands)
    budget = PROFILE_INT8_FRACTION[profile] * total
    ranked = sorted(cands, key=lambda it: compute_gue_spectral_sensitivity(it[1], it[0], halo_levels=halo_set), reverse=True)

    int8_names: set[str] = set()
    acc = 0
    for name, lin in ranked:
        if acc >= budget:
            break
        int8_names.add(name)
        acc += lin.weight.numel()

    bytes8 = bytes4 = n8 = n4 = 0
    for idx, (name, lin) in enumerate(cands):
        if name in int8_names:
            q = Int8GroupLinear.from_linear(lin, group_size=int8_group_size)
            bytes8 += q.weight_memory_bytes()
            n8 += 1
        else:
            q = HadamardLinearINT4.from_linear(
                lin, group_size=int4_group_size, act_bits=16, seed=1729 + idx, use_triton=use_triton, rotate_input=False
            )
            bytes4 += q.weight_memory_bytes()
            n4 += 1
        _set_module_by_name(target, name, q)
    return target, {
        "profile": profile,
        "int8_layers": n8,
        "int4_layers": n4,
        "int8_param_fraction": round(acc / max(total, 1), 4),
        "quant_weight_bytes": bytes8 + bytes4,
        "dense_fp16_bytes": total * 2,
        "compression_vs_fp16": round(total * 2 / max(bytes8 + bytes4, 1), 3),
    }
