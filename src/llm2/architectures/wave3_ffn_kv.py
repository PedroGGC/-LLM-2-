"""Wave 3 Structural FFN (60% of model) & Context KV (4x reduction) Tournament:
- 3A: MPOLinear (Quantum Tensor Network 2-core Matrix Product Operator factorization of FFN weights)
- 3B: ProSparse ReLU^2 SwiGLU wrapper (cortical 85-95% activation sparsity measurement & execution)
- 3C: Gemma-4 style K=V + Cross-Layer KV-Sharing evaluation & memory estimator for 64K -> 512K scaling
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def _best_factor_pair(n: int) -> tuple[int, int]:
    """Factor integer n into (a, b) with a * b == n closest to sqrt(n)."""
    root = int(math.isqrt(n))
    for a in range(root, 0, -1):
        if n % a == 0:
            return a, n // a
    return 1, n


class MPOLinear(nn.Module):
    """2-Core Matrix Product Operator (Quantum Tensor Train) factorization of a Linear(in_features, out_features).

    W of shape (m1*m2, n1*n2) is reshaped to (m1, n1, m2, n2) and SVD-factorized into:
      core1: (m1, n1, chi)
      core2: (chi, m2, n2)
    Forward contracts x (*, n1, n2) -> (*, m1, m2) via two compact batched matmuls without materializing W!
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bond_dim: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.m1, self.m2 = _best_factor_pair(self.out_features)
        self.n1, self.n2 = _best_factor_pair(self.in_features)
        max_chi = min(self.m1 * self.n1, self.m2 * self.n2)
        self.bond_dim = max(1, min(int(bond_dim), max_chi))

        self.core1 = nn.Parameter(torch.empty(self.m1, self.n1, self.bond_dim))
        self.core2 = nn.Parameter(torch.empty(self.bond_dim, self.m2, self.n2))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        nn.init.normal_(self.core1, std=0.02)
        nn.init.normal_(self.core2, std=0.02)

    @torch.no_grad()
    def fit_from_linear(self, linear: nn.Linear) -> float:
        """Exact truncated SVD initialization of MPO cores from dense weight; returns relative Frobenius error."""
        w = linear.weight.detach().float()  # (m1*m2, n1*n2)
        w4 = w.view(self.m1, self.m2, self.n1, self.n2).permute(0, 2, 1, 3).reshape(self.m1 * self.n1, self.m2 * self.n2)
        # Randomized or exact SVD
        u, s, vh = torch.linalg.svd(w4, full_matrices=False)
        chi = self.bond_dim
        s_sqrt = s[:chi].sqrt()
        c1 = (u[:, :chi] * s_sqrt.unsqueeze(0)).reshape(self.m1, self.n1, chi)
        c2 = (s_sqrt.unsqueeze(1) * vh[:chi, :]).reshape(chi, self.m2, self.n2)
        self.core1.copy_(c1.to(self.core1.dtype))
        self.core2.copy_(c2.to(self.core2.dtype))
        if linear.bias is not None and self.bias is not None:
            self.bias.copy_(linear.bias.detach().to(self.bias.dtype))
        rec = (u[:, :chi] * s[:chi].unsqueeze(0)) @ vh[:chi, :]
        rel_err = float((w4 - rec).norm().item() / w4.norm().clamp_min(1e-12).item())
        return rel_err

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, bond_dim: int = 128) -> tuple[MPOLinear, float]:
        mod = cls(
            linear.in_features,
            linear.out_features,
            bond_dim=bond_dim,
            bias=linear.bias is not None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        rel_err = mod.fit_from_linear(linear)
        return mod, rel_err

    def compression_ratio(self) -> float:
        dense = self.in_features * self.out_features
        mpo = self.core1.numel() + self.core2.numel()
        return dense / max(mpo, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x2 = x.reshape(-1, self.n1, self.n2)  # (B, n1, n2)
        # Step 1: contract n2 with core2 (chi, m2, n2) -> (B, n1, chi, m2)
        c2_t = self.core2.to(x.dtype).reshape(self.bond_dim * self.m2, self.n2).t()  # (n2, chi*m2)
        h = torch.matmul(x2, c2_t).reshape(-1, self.n1, self.bond_dim, self.m2)
        # Step 2: contract (n1, chi) with core1 (m1, n1, chi) -> (B, m2, m1) -> (B, m1, m2)
        h_flat = h.permute(0, 3, 1, 2).reshape(-1, self.m2, self.n1 * self.bond_dim)
        c1_t = self.core1.to(x.dtype).reshape(self.m1, self.n1 * self.bond_dim).t()  # (n1*chi, m1)
        out = torch.matmul(h_flat, c1_t).transpose(1, 2).reshape(*orig_shape, self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class ReLUSquaredActivation(nn.Module):
    """ProSparse / Primer ReLU^2 activation with optional running sparsity telemetry."""

    def __init__(self, *, threshold: float = 0.0, track_sparsity: bool = True):
        super().__init__()
        self.threshold = float(threshold)
        self.track_sparsity = bool(track_sparsity)
        self.last_zero_fraction: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(x - self.threshold) if self.threshold > 0.0 else F.relu(x)
        y = y * y
        if self.track_sparsity:
            with torch.no_grad():
                self.last_zero_fraction = float((y == 0).float().mean().item())
        return y


def apply_mpo_to_ffn(model: nn.Module, *, bond_dim: int = 256, inplace: bool = False) -> tuple[nn.Module, dict]:
    """Factorize all SwiGLU FFN projections (gate_proj/up_proj/down_proj or gate/up/down) using 2-core MPO."""
    target = model if inplace else copy.deepcopy(model)
    ffn_suffixes = ("gate_proj", "up_proj", "down_proj", "ffn.gate", "ffn.up", "ffn.down")
    replaced = 0
    rel_errs = []
    orig_params = 0
    mpo_params = 0
    for full_name, mod in list(target.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if not any(full_name.endswith(sfx) for sfx in ffn_suffixes):
            continue
        mpo_lin, err = MPOLinear.from_linear(mod, bond_dim=bond_dim)
        orig_params += mod.weight.numel()
        mpo_params += mpo_lin.core1.numel() + mpo_lin.core2.numel()
        rel_errs.append(err)
        if "." in full_name:
            parent_name, attr = full_name.rsplit(".", 1)
            setattr(target.get_submodule(parent_name), attr, mpo_lin)
        else:
            setattr(target, full_name, mpo_lin)
        replaced += 1
    return target, {
        "mode": "mpo_ffn",
        "bond_dim": bond_dim,
        "replaced_count": replaced,
        "ffn_compression_ratio": round(orig_params / max(mpo_params, 1), 2),
        "mean_rel_frob_err": round(sum(rel_errs) / max(len(rel_errs), 1), 4),
    }


def apply_relu2_sparsity_to_hf_mlp(model: nn.Module, *, threshold: float = 0.0, inplace: bool = False) -> tuple[nn.Module, list[ReLUSquaredActivation]]:
    """Replace `mlp.act_fn` (SiLU) with ProSparse ReLU^2 on HF Qwen3/Llama-style MLP blocks."""
    target = model if inplace else copy.deepcopy(model)
    trackers: list[ReLUSquaredActivation] = []
    for _, mod in target.named_modules():
        if hasattr(mod, "act_fn") and hasattr(mod, "gate_proj") and hasattr(mod, "up_proj"):
            act = ReLUSquaredActivation(threshold=threshold, track_sparsity=True)
            mod.act_fn = act
            trackers.append(act)
    return target, trackers


def kv_sharing_memory_table(
    *,
    n_global_layers: int = 9,
    kv_heads: int = 8,
    head_dim: int = 128,
    n_delta_layers: int = 27,
    d_model: int = 2560,
    contexts: tuple[int, ...] = (16384, 32768, 65536, 131072, 262144, 524288),
) -> dict[str, dict[str, float]]:
    """Exact state VRAM table (MiB) comparing baseline 9G KV-FP8 vs Gemma-4 K=V + 2x Cross-Layer KV-Sharing."""
    delta_bytes = n_delta_layers * (kv_heads * head_dim * head_dim + d_model * 3) * 4
    # Per token per global layer in FP8: 2 (K+V) * kv_heads * (head_dim + 2) + 8 (pos)
    fp8_tok_per_layer = 2 * kv_heads * (head_dim + 2) + 8
    # K=V eliminates V cache (only K stored): 1 * kv_heads * (head_dim + 2) + 8
    keqv_fp8_tok_per_layer = 1 * kv_heads * (head_dim + 2) + 8
    keqv_int4_tok_per_layer = 1 * kv_heads * (head_dim // 2 + 2) + 8
    # Cross-layer KV sharing on HALO_9 pairs (0,1),(6,7),(23,24),(34,35) + standalone (12) -> 5 unique KV caches instead of 9!
    unique_shared_layers = (n_global_layers + 1) // 2  # 5 for 9 layers
    out: dict[str, dict[str, float]] = {}
    for ctx in contexts:
        out[f"{ctx // 1024}K"] = {
            "baseline_9g_kv_fp8_mib": round((delta_bytes + n_global_layers * ctx * fp8_tok_per_layer) / (1024**2), 1),
            "keqv_only_9g_fp8_mib": round((delta_bytes + n_global_layers * ctx * keqv_fp8_tok_per_layer) / (1024**2), 1),
            "keqv_plus_share_5g_fp8_mib": round((delta_bytes + unique_shared_layers * ctx * keqv_fp8_tok_per_layer) / (1024**2), 1),
            "keqv_plus_share_5g_int4_mib": round((delta_bytes + unique_shared_layers * ctx * keqv_int4_tok_per_layer) / (1024**2), 1),
        }
    return out
