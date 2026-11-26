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
