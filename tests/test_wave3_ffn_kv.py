"""Unit tests for Wave 3 structural FFN & KV compression (MPO-FFN, ProSparse ReLU^2, Gemma-4 K=V + KV-Sharing)."""
import torch
from torch import nn

from llm2.architectures.wave3_ffn_kv import (
    MPOLinear,
    ReLUSquaredActivation,
    apply_mpo_to_ffn,
    kv_sharing_memory_table,
)


def test_mpo_linear_exact_full_rank_and_compression():
    torch.manual_seed(42)
    lin = nn.Linear(64, 48, bias=True)
    # Full bond dim = min(m1*n1, m2*n2) = min(6*8, 8*8) = 48 -> exact reconstruction!
    mpo_exact, err_exact = MPOLinear.from_linear(lin, bond_dim=64)
    assert err_exact < 1e-4
    x = torch.randn(3, 5, 64)
    assert torch.allclose(lin(x), mpo_exact(x), atol=1e-4, rtol=1e-4)

    # Low bond dim compresses significantly while remaining finite and differentiable
    mpo_comp, _ = MPOLinear.from_linear(lin, bond_dim=8)
    assert mpo_comp.compression_ratio() > 3.0
    y = mpo_comp(x)
    assert y.shape == (3, 5, 48) and torch.isfinite(y).all()
    y.sum().backward()
    assert mpo_comp.core1.grad is not None and mpo_comp.core2.grad is not None


def test_relu_squared_sparsity_tracking():
    torch.manual_seed(7)
    act = ReLUSquaredActivation(threshold=0.0, track_sparsity=True)
    x = torch.randn(16, 1024)
    y = act(x)
    assert (y >= 0).all()
    assert 0.40 <= act.last_zero_fraction <= 0.60
    # Higher threshold increases cortical sparsity
    act_hi = ReLUSquaredActivation(threshold=0.5, track_sparsity=True)
    _ = act_hi(x)
    assert act_hi.last_zero_fraction > act.last_zero_fraction


def test_apply_mpo_to_ffn_and_kv_table():
    class _TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(64, 128, bias=False)
            self.up_proj = nn.Linear(64, 128, bias=False)
            self.down_proj = nn.Linear(128, 64, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    mlp = _TinyMLP()
    mpo_mlp, info = apply_mpo_to_ffn(mlp, bond_dim=16, inplace=False)
    assert info["replaced_count"] == 3
    assert info["ffn_compression_ratio"] > 2.5
    assert torch.isfinite(mpo_mlp(torch.randn(2, 64))).all()

    tbl = kv_sharing_memory_table()
    # 512K with K=V + 5/9 KV sharing in INT4 must fit well under 1.5 GiB!
    assert tbl["512K"]["keqv_plus_share_5g_int4_mib"] < 1500.0
    assert tbl["64K"]["keqv_plus_share_5g_fp8_mib"] < 400.0
