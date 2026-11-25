"""Unit tests for Wave 2 sub-4-bit mathematical compression tournament (E8 2-bit, Kintsugi GUE Mixed, ParetoQ, Bit-Plane)."""
import torch
from torch import nn

from llm2.architectures.sub4bit import (
    BitPlaneLinear,
    E8Linear2Bit,
    ParetoQLinear,
    build_e8_root_codebook,
    quantize_model_e8_2bit,
    quantize_model_kintsugi_mixed,
    quantize_model_paretoq,
)


def test_e8_codebook_geometry_and_snr():
    cb = build_e8_root_codebook(dtype=torch.float32)
    assert cb.shape == (256, 8)
    # First 240 entries are exact unit-norm E8 roots
    norms = cb[:240].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(240), atol=1e-5)

    # Verify 2-stage Residual E8 beats naive 2-bit scalar quantization on Gaussian weights
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=False)
    nn.init.normal_(lin.weight, std=0.02)
    e8 = E8Linear2Bit.from_linear(lin, group_size=64, rotate_input=True)
    x = torch.randn(4, 128)
    y_ref = lin(x)
    y_e8 = e8(x)
    pq2 = ParetoQLinear(128, 64, bias=False, mode="2bit", group_size=64, qat=False)
    pq2.init_from_linear(lin)
    y_scalar2 = pq2(x)
    snr_e8 = 10.0 * torch.log10(y_ref.square().mean() / (y_ref - y_e8).square().mean())
    snr_s2 = 10.0 * torch.log10(y_ref.square().mean() / (y_ref - y_scalar2).square().mean())
    assert snr_e8.item() > 9.5 and snr_e8.item() > snr_s2.item() + 0.5, (
        f"Expected E8 2-stage SNR > 9.5 dB and > scalar2 + 0.5 dB, got E8={snr_e8.item():.2f} vs scalar2={snr_s2.item():.2f}"
    )
    # Storage: 2 bytes / 8 weights + 2 bytes / 64 weights = 2.25 bpw vs 16 bpw FP16 (>7x compression!)
    ratio = (lin.weight.numel() * 2) / e8.weight_memory_bytes()
    assert ratio > 7.0


def test_kintsugi_gue_mixed_allocator():
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(64, 64, bias=False),
        nn.Linear(64, 128, bias=False),
        nn.Linear(128, 64, bias=False),
        nn.Linear(64, 64, bias=False),
    )
    # Inject a heavy outlier into layer 0 to verify GUE kurtosis prioritizes it for 4-bit
    with torch.no_grad():
        model[0].weight[0, 0] = 25.0
    q_model, info = quantize_model_kintsugi_mixed(model, high_bit_fraction=0.25, inplace=False, use_triton=False)
    assert info["int4_layers"] == 1
    assert info["e8_2bit_layers"] == 3
    assert type(q_model[0]).__name__ == "HadamardLinearINT4"
    x = torch.randn(2, 64)
    y = q_model(x)
    assert torch.isfinite(y).all()


def test_paretoq_ternary_qat_and_freeze_parity():
    torch.manual_seed(11)
    lin = nn.Linear(64, 32, bias=True)
    pq = ParetoQLinear(64, 32, bias=True, mode="ternary", group_size=64, qat=True)
    pq.init_from_linear(lin)

    x = torch.randn(3, 64, requires_grad=True)
    y_qat = pq(x)
    loss = y_qat.square().mean()
    loss.backward()
    assert pq.weight.grad is not None and torch.isfinite(pq.weight.grad).all()
    assert pq.scale.grad is not None and torch.isfinite(pq.scale.grad).all()

    with torch.no_grad():
        y_before = pq(x.detach())
        pq.freeze_to_packed()
        y_after = pq(x.detach())
    assert torch.allclose(y_before, y_after, atol=1e-3, rtol=1e-3)
    assert pq.weight is None and pq.packed_weight is not None


def test_bitplane_progressive_monotonic_accuracy():
    torch.manual_seed(19)
    lin = nn.Linear(64, 32, bias=False)
    bp = BitPlaneLinear(64, 32, bias=False, group_size=32, active_bits=4)
    bp.pack_from_float_weight(lin.weight)
    x = torch.randn(8, 64)
    y_ref = lin(x)

    errs = []
    for bits in (2, 3, 4):
        bp.active_bits = bits
        y_b = bp(x)
        errs.append(float((y_ref - y_b).square().mean().item()))
    # Monotonically improving error as more bit-planes are read from the exact same storage!
    assert errs[0] > errs[1] > errs[2], f"Bit-plane errors not monotonic: {errs}"


def test_model_quantizers_smoke():
    torch.manual_seed(3)
    m = nn.Sequential(nn.Linear(64, 64, bias=False), nn.ReLU(), nn.Linear(64, 64, bias=False))
    x = torch.randn(2, 64)
    q_e8, info_e8 = quantize_model_e8_2bit(m, inplace=False)
    q_pq, info_pq = quantize_model_paretoq(m, mode="ternary", qat=False, inplace=False)
    assert info_e8["replaced_count"] == 2 and torch.isfinite(q_e8(x)).all()
    assert info_pq["replaced_count"] == 2 and torch.isfinite(q_pq(x)).all()
