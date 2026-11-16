import math
import numpy as np
import pytest
import torch
from llm2.config import ModelConfig, ResearchConfig
from llm2.architectures.fractal_delta import FractalDeltaCell, FractalDeltaMemory

def test_rejection_sampling_distribution_identity():
    np.random.seed(42)
    V = 10
    N = 100_000
    q_raw = np.array([0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.015, 0.005])
    q = q_raw / q_raw.sum()
    p_raw = np.array([0.15, 0.30, 0.10, 0.10, 0.15, 0.05, 0.08, 0.04, 0.02, 0.01])
    p = p_raw / p_raw.sum()

    residual = np.maximum(0, q - p)
    q_prime = residual / residual.sum()

    samples = []
    for _ in range(N):
        x_hat = np.random.choice(V, p=p)
        r = min(1.0, q[x_hat] / p[x_hat])
        if np.random.random() <= r:
            samples.append(x_hat)
        else:
            samples.append(np.random.choice(V, p=q_prime))

    emp_freq = np.bincount(samples, minlength=V) / N
    tvd = 0.5 * np.sum(np.abs(emp_freq - q))
    assert tvd < 0.005, f"Total variation distance too large: {tvd}"

def test_speculative_decoding_expected_tokens():
    alpha = 0.70
    K = 2
    theo_mean = (1 - alpha**(K + 1)) / (1 - alpha)
    assert abs(theo_mean - 2.19) < 1e-4

    # Monte Carlo check
    np.random.seed(42)
    trials = 50_000
    tokens = []
    for _ in range(trials):
        t = 1
        for _ in range(K):
            if np.random.random() <= alpha:
                t += 1
            else:
                break
        tokens.append(t)
    emp_mean = np.mean(tokens)
    assert abs(emp_mean - theo_mean) / theo_mean < 0.01

def test_speculative_decoding_speedup_and_energy():
    alpha = 0.70
    K = 2
    beta = 0.0375 # corresponds to gamma = 1 + beta*K = 1.075
    speedup = ((1 - alpha**(K + 1)) / (1 - alpha)) / (1 + beta * K)
    assert abs(speedup - 2.0372) < 1e-3

    p0 = 72.0
    t0 = 1395.4
    e0 = p0 / t0 # 51.60 mJ

    p1 = 73.5
    t1 = t0 * speedup
    e1 = p1 / t1 # 25.86 mJ
    red = (e0 - e1) / e0
    assert abs(red - 0.4989) < 1e-3

def test_core_parameter_count_and_l2_sizing():
    # Production arm config from Campaign 010 (d_model=384, heads=6, hidden_dim=1024)
    config = ModelConfig(vocab_size=4096, d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, dropout=0.0)
    research = ResearchConfig(
        architecture="fractal_delta",
        scale_conditioning=False,
        fractal_prelude_coda=True,
        fractal_golden_phase=True,
        fractal_free_energy=False,
        fractal_kda=False,
        state_chunk_size=32,
    )
    cell = FractalDeltaCell(config, research)
    core_params = sum(p.numel() for p in cell.parameters())
    assert core_params == 1_924_620

    bf16_bytes = core_params * 2
    fp32_bytes = core_params * 4
    assert bf16_bytes == 3_849_240
    assert fp32_bytes == 7_698_480

    # L2 occupancy in L4 (48 MB) and RTX 4090 (72 MB)
    l4_occupancy_bf16 = (bf16_bytes / 1e6) / 48.0
    rtx4090_occupancy_bf16 = (bf16_bytes / 1e6) / 72.0
    assert l4_occupancy_bf16 < 0.085 # < 8.5%
    assert rtx4090_occupancy_bf16 < 0.055 # < 5.5%

    # Compact arm (d_model=256, heads=4, hidden_dim=683)
    config256 = ModelConfig(vocab_size=4096, d_model=256, n_layers=8, n_heads=4, hidden_dim=683, dropout=0.0)
    cell256 = FractalDeltaCell(config256, research)
    core256_params = sum(p.numel() for p in cell256.parameters())
    assert core256_params == 856_328
    assert core256_params * 2 == 1_712_656

def test_thermal_and_cmos_scaling_model():
    # RTX 4090 450W -> 290W (-35.56%), clock 2760 -> 2615 MHz (-5.25%)
    gamma_4090 = math.log(290.0 / 450.0) / math.log(2615.0 / 2760.0)
    assert 7.0 < gamma_4090 < 9.0

    # L4 72W -> 50W (-30.56%), clock 2040 -> 1940 MHz (-4.90%)
    gamma_l4 = math.log(50.0 / 72.0) / math.log(1940.0 / 2040.0)
    assert 6.5 < gamma_l4 < 8.0

    # Synergy table efficiency multiplier
    eff_baseline = 1395.4 / 72.0 # 19.38 tok/J
    eff_optimized = 4825.0 / 48.5 # 99.48 tok/J
    ratio = eff_optimized / eff_baseline
    assert abs(ratio - 5.13) < 0.05
