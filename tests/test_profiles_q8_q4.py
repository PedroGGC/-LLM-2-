"""Perfis Lumina-Q8/Q6/Q5/Q4: ordem de erro, tamanho e fração INT8 por parâmetros."""
import pytest
import torch
from torch import nn

from llm2.architectures.profiles import Int8GroupLinear, quantize_model_profile


def _stack(seed: int = 5) -> nn.Sequential:
    torch.manual_seed(seed)
    layers = []
    for _ in range(8):
        layers += [nn.Linear(128, 128, bias=False), nn.SiLU()]
    m = nn.Sequential(*layers[:-1])
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.08)
    return m.eval()


def test_int8_layer_is_near_lossless():
    torch.manual_seed(1)
    lin = nn.Linear(256, 64, bias=True)
    q = Int8GroupLinear.from_linear(lin, group_size=128)
    x = torch.randn(4, 256)
    rel = (lin(x) - q(x)).norm() / lin(x).norm()
    assert rel < 0.01


def test_profiles_error_and_size_ordering():
    m = _stack()
    x = torch.randn(16, 128)
    with torch.no_grad():
        ref = m(x)
        errs, sizes, fracs = {}, {}, {}
        for p in ("q8", "q6", "q5", "q4"):
            qm, info = quantize_model_profile(m, p, use_triton=False)
            errs[p] = ((qm(x) - ref).norm() / ref.norm()).item()
            sizes[p] = info["quant_weight_bytes"]
            fracs[p] = info["int8_param_fraction"]
    assert errs["q8"] < errs["q6"] < errs["q5"] < errs["q4"], errs
    assert errs["q8"] < 0.02
    assert sizes["q8"] > sizes["q6"] > sizes["q5"] > sizes["q4"], sizes
    assert fracs["q8"] == 1.0 and fracs["q4"] == 0.0
    assert fracs["q6"] == pytest.approx(0.5, abs=0.13) and fracs["q5"] == pytest.approx(0.25, abs=0.13)


def test_invalid_profile_rejected():
    with pytest.raises(ValueError):
        quantize_model_profile(_stack(), "q3", use_triton=False)


def test_mp_spectral_sensitivity_profile():
    m = _stack(seed=9)
    x = torch.randn(8, 128)
    with torch.no_grad():
        ref = m(x)
        q6, info = quantize_model_profile(m, "q6", use_triton=False, sensitivity_mode="mp")
        q4, _ = quantize_model_profile(m, "q4", use_triton=False, sensitivity_mode="mp")
        err_q6 = ((q6(x) - ref).norm() / ref.norm()).item()
        err_q4 = ((q4(x) - ref).norm() / ref.norm()).item()
    assert err_q6 < err_q4
    assert info["int8_layers"] > 0 and info["int4_layers"] > 0


@pytest.mark.cuda
def test_int8_triton_cuda_decode_parity():
    from llm2.architectures.triton_quarot import HAS_TRITON, int8_group_linear

    if not (torch.cuda.is_available() and HAS_TRITON):
        pytest.skip("CUDA + Triton required")
    torch.manual_seed(7)
    lin = nn.Linear(512, 256, bias=True).cuda().half()
    q = Int8GroupLinear.from_linear(lin, group_size=128)
    x = torch.randn(1, 1, 512, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        y_tri = int8_group_linear(x, q.qweight, q.weight_scale, q.bias, q.group_size, use_triton=True)
        y_ref = int8_group_linear(x, q.qweight, q.weight_scale, q.bias, q.group_size, use_triton=False)
    torch.testing.assert_close(y_tri, y_ref, atol=2e-2, rtol=2e-2)
