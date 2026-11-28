"""Unit tests for Round 2 experiments: E2 (CKDA signed DPLR), E3 (ChainDoRA), E4 (LLVQ24 Golay)."""

import torch
from torch import nn

from llm2.architectures.round2_experiments import (
    ChainDoRALinear,
    LLVQ24Linear,
    ckda_chunked,
    ckda_sequential,
    get_golay24_bipolar_codebook,
)


def test_ckda_chunked_matches_sequential_with_negative_alpha_and_beta_gt_1():
    torch.manual_seed(13)
    b, hq, hkv, t, d = 2, 8, 2, 45, 16
    q = torch.nn.functional.normalize(torch.randn(b, hq, t, d), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, hkv, t, d), dim=-1)
    v = torch.randn(b, hkv, t, d) * 0.2
    # Signed alpha in [-0.95, 0.95] (crossing zero!) and beta in [0.2, 1.9] (Householder reflections!)
    alpha = torch.tanh(torch.randn(b, hkv, t, 1) * 1.5)
    beta = 2.0 * torch.sigmoid(torch.randn(b, hkv, t, 1))
    s0 = torch.randn(b, hkv, d, d) * 0.1

    y_seq, m_seq = ckda_sequential(q, k, v, alpha, beta, s0)
    y_chk, m_chk = ckda_chunked(q, k, v, alpha, beta, s0, chunk_size=16)
    torch.testing.assert_close(y_chk, y_seq, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(m_chk, m_seq, atol=1e-4, rtol=1e-4)


def test_chain_dora_param_reduction_and_exact_merge():
    torch.manual_seed(5)
    base = nn.Linear(256, 512, bias=False)
    adapter = ChainDoRALinear(base, tt_rank=4, alpha=16.0)
    lora_r32_params = 32 * (256 + 512)
    assert adapter.trainable_param_count() < 0.20 * lora_r32_params

    # Simulate a training step updating cores and magnitude
    x = torch.randn(4, 256)
    y = adapter(x)
    y.pow(2).mean().backward()
    with torch.no_grad():
        for p in adapter.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad, alpha=-0.1)
    with torch.no_grad():
        y_before = adapter(x)
        merged = adapter.merge_to_linear()
        y_after = merged(x)
    torch.testing.assert_close(y_before, y_after, atol=1e-5, rtol=1e-5)


def test_llvq24_golay_codebook_and_quantization_snr():
    cb = get_golay24_bipolar_codebook(torch.device("cpu"))
    assert cb.shape == (4096, 24)
    # Every codeword is {-1, +1}^24 and min Hamming distance of G24 is 8 -> max non-self dot is 24 - 16 = 8
    dots = cb[:64] @ cb.T
    dots[torch.arange(64), torch.arange(64)] = -24
    assert int(dots.max().item()) == 8

    torch.manual_seed(7)
    lin = nn.Linear(256, 128, bias=True)
    nn.init.normal_(lin.weight, std=0.05)
    q = LLVQ24Linear.from_linear(lin)
    x = torch.randn(8, 256)
    with torch.no_grad():
        ref = lin(x)
        out = q(x)
    rel_err = (out - ref).norm() / ref.norm()
    # 4-stage 24D Golay (2.0 bpw indices, Hadamard-free) approaches 2-bit Gaussian rate-distortion
    assert rel_err < 0.35
