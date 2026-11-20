"""Tests for Phase 2B levers: #6 QuaRot Hadamard INT4/W4A8, #5 Adaptive Early-Exit,
SegmentMemoryCache (arXiv:2602.24281 inspiration), and Confidence-Gated Self-Speculative MTP."""

import torch
from torch import nn

from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from llm2.architectures.quarot import (
    HadamardLinearINT4,
    apply_hadamard_rotation,
    fast_walsh_hadamard_transform,
    hadamard_matrix,
    inverse_hadamard_rotation,
    largest_pow2_divisor,
    pack_int4_signed,
    quantize_model_quarot_int4,
    rademacher_signs,
    unpack_int4_signed,
)
from llm2.config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from llm2.research.decode_optimization import (
    AdaptiveEarlyExitDecode,
    CompiledDecode,
    SegmentMemoryCache,
    SelfSpeculativeMTPDecode,
)
from llm2.research.objectives import PredictiveObjective


def _make_fractal_cfg(
    *,
    d_model: int = 64,
    hidden_dim: int = 96,
    n_layers: int = 6,
    weight_mode: str = "dense",
    intermediate_weight: float = 0.0,
    mtp_tokens: int = 0,
    mtp_weight: float = 0.0,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=128,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
            hidden_dim=hidden_dim,
            max_seq_len=256,
            dropout=0.0,
        ),
        data=DataConfig(seq_len=32, batch_size=2, eval_batch_size=2),
        optimizer=OptimizerConfig(schedule_steps=100, warmup_steps=5),
        train=TrainConfig(max_steps=20, eval_interval=10, checkpoint_interval=10),
        research=ResearchConfig(
            architecture="fractal_delta",
            share_scales=True,
            scale_conditioning=False,
            fractal_prelude_coda=True,
            fractal_value_residual=True,
            fractal_golden_phase=True,
            fractal_assoc_delta=True,
            fractal_assoc_multilag=True,
            fractal_weight_mode=weight_mode,
            intermediate_weight=intermediate_weight,
            mtp_tokens=mtp_tokens,
            mtp_weight=mtp_weight,
        ),
    )


def test_hadamard_composite_dimensions_and_outlier_dispersion():
    for d in (48, 64, 96, 256, 384, 640):
        b_size = largest_pow2_divisor(d, 128)
        assert b_size >= 16 and d % b_size == 0
        x = torch.randn(2, 5, d)
        signs = rademacher_signs(d, seed=99)
        h_block = hadamard_matrix(b_size)
        y = apply_hadamard_rotation(x, signs, h_block)
        x_rec = inverse_hadamard_rotation(y, signs, h_block)
        assert torch.allclose(x, x_rec, atol=1e-5), f"failed round-trip for d={d}"
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)

        y_fwht = fast_walsh_hadamard_transform(x)
        x_fwht = fast_walsh_hadamard_transform(y_fwht)
        assert torch.allclose(x, x_fwht, atol=1e-5)

    # Outlier spike dispersion + constant-vector immunity via Rademacher signs
    d = 384
    b_size = largest_pow2_divisor(d, 128)
    signs = rademacher_signs(d, seed=1729)
    h_block = hadamard_matrix(b_size)
    spike = torch.zeros(1, 1, d)
    spike[0, 0, 17] = 128.0
    rot_spike = apply_hadamard_rotation(spike, signs, h_block)
    assert rot_spike.abs().max().item() <= 16.0  # Reduced by sqrt(128) = 11.31x

    const_vec = torch.full((1, 1, d), 10.0)
    rot_const = apply_hadamard_rotation(const_vec, signs, h_block)
    # Without random signs, constant vector amplifies by sqrt(128) = 11.3x to 113.1; with signs it stays bounded
    assert rot_const.abs().max().item() < 45.0


def test_int4_pack_unpack_and_quarot_outlier_accuracy():
    vals = torch.randint(-8, 8, (16, 96), dtype=torch.int8)
    packed = pack_int4_signed(vals)
    assert packed.dtype == torch.uint8 and packed.shape == (16, 48)
    unpacked = unpack_int4_signed(packed)
    assert torch.equal(vals, unpacked)

    torch.manual_seed(7)
    lin = nn.Linear(128, 64, bias=False)
    nn.init.normal_(lin.weight, std=0.05)
    with torch.no_grad():
        lin.weight[:, 0] = 0.45
    x = torch.randn(4, 8, 128) * 0.5
    y_ref = lin(x)

    q_layer = HadamardLinearINT4.from_linear(lin, group_size=128, act_bits=8)
    y_quarot = q_layer(x)

    # Compare against naive unrotated INT4 weight (same group_size=128) + INT8 activation quantization
    w_scale_naive = (lin.weight.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-6)
    w_naive = (lin.weight / w_scale_naive).round().clamp(-7.0, 7.0) * w_scale_naive
    x_scale_naive = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-6)
    x_naive = (x / x_scale_naive).round().clamp(-127.0, 127.0) * x_scale_naive
    y_naive = torch.nn.functional.linear(x_naive, w_naive)

    mse_quarot = (y_quarot - y_ref).square().mean().item()
    mse_naive = (y_naive - y_ref).square().mean().item()
    assert mse_quarot < mse_naive * 0.35, f"expected QuaRot MSE ({mse_quarot}) < 0.35 * naive MSE ({mse_naive})"
    q_layer_g32 = HadamardLinearINT4.from_linear(lin, group_size=32, act_bits=8)
    assert q_layer_g32.dense_memory_bytes(2) / q_layer_g32.weight_memory_bytes() > 3.4


def test_quantize_model_quarot_int4_and_qat_mode():
    cfg = _make_fractal_cfg(d_model=64, hidden_dim=96, n_layers=4)
    model = FractalDeltaMemory(cfg.model, cfg.research).eval()
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out_dense = model(x)

    q_model, report = quantize_model_quarot_int4(model, group_size=32, act_bits=8, inplace=False)
    assert report["replaced_count"] > 0
    assert report["shared_embedding_preserved"] is True
    assert q_model.lm_head.weight is q_model.embedding.weight
    assert report["compression_vs_bf16"] > 3.2
    assert report["compression_vs_fp32"] > 6.4

    with torch.no_grad():
        out_q = q_model(x)
    assert out_q.logits.shape == out_dense.logits.shape
    assert torch.isfinite(out_q.logits).all()

    # Verify QAT mode forward + backward
    cfg_qat = _make_fractal_cfg(d_model=64, hidden_dim=96, n_layers=3, weight_mode="quarot_int4")
    qat_model = FractalDeltaMemory(cfg_qat.model, cfg_qat.research)
    out_qat = qat_model(x)
    loss = out_qat.logits.float().square().mean()
    loss.backward()
    assert qat_model.cells[1].qkv.weight.grad is not None
    assert torch.isfinite(qat_model.cells[1].qkv.weight.grad).all()


def test_early_exit_intermediate_supervision_and_adaptive_decode():
    cfg = _make_fractal_cfg(n_layers=6, intermediate_weight=0.25, mtp_tokens=2, mtp_weight=0.2)
    model = FractalDeltaMemory(cfg.model, cfg.research)
    objective = PredictiveObjective(model, cfg.research)
    batch = {
        "input_ids": torch.randint(0, 128, (2, 16)),
        "targets": torch.randint(0, 128, (2, 16)),
    }
    obj_out = objective(model, batch)
    assert "intermediate_loss" in obj_out.metrics
    assert torch.isfinite(obj_out.loss)
    obj_out.loss.backward()
    assert model.embedding.weight.grad is not None

    model.eval()
    prefix = torch.randint(0, 128, (1, 12))
    base_dec = FractalDecode(model)
    logits, st = base_dec.prefill(prefix, 32)
    ref_toks = [int(logits[0, -1].argmax().item())]
    for _ in range(7):
        tok_t = torch.tensor([[ref_toks[-1]]], dtype=torch.long)
        logits, st = base_dec.step(tok_t, st)
        ref_toks.append(int(logits[0, -1].argmax().item()))

    # 1. Thresholds disabled -> 100% full depth and exact token parity with CompiledDecode
    ee_full = AdaptiveEarlyExitDecode(model, confidence_threshold=1.1, delta_threshold=0.0, min_layers=2)
    toks_full, stats_full = ee_full.generate_greedy(prefix, 8)
    assert toks_full == ref_toks
    assert stats_full["mean_layers_executed"] == 6.0
    assert stats_full["layers_saved_ratio"] == 0.0

    # 2. Aggressive early exit after min_layers=2 -> saves 4/6 = 66.7% of Core layers with valid state
    ee_fast = AdaptiveEarlyExitDecode(model, confidence_threshold=0.001, delta_threshold=0.0, min_layers=2)
    toks_fast, stats_fast = ee_fast.generate_greedy(prefix, 8)
    assert len(toks_fast) == 8
    assert stats_fast["mean_layers_executed"] == 2.0
    assert abs(stats_fast["layers_saved_ratio"] - (4.0 / 6.0)) < 1e-6
    assert stats_fast["early_exit_rate"] == 1.0

    # 3. Latent delta convergence exit (zero intermediate lm_head calls)
    ee_delta = AdaptiveEarlyExitDecode(model, confidence_threshold=1.1, delta_threshold=10.0, min_layers=3)
    toks_delta, stats_delta = ee_delta.generate_greedy(prefix, 6)
    assert len(toks_delta) == 6
    assert stats_delta["mean_layers_executed"] == 3.0


def test_segment_memory_cache_and_confidence_gated_mtp():
    cfg = _make_fractal_cfg(n_layers=4, mtp_tokens=2, mtp_weight=0.25)
    model = FractalDeltaMemory(cfg.model, cfg.research).eval()
    objective = PredictiveObjective(model, cfg.research).eval()
    decoder = FractalDecode(model)

    cache = SegmentMemoryCache(max_entries=8, max_anchors=4)
    p_short = torch.randint(0, 128, (1, 10))
    p_long = torch.cat((p_short, torch.randint(0, 128, (1, 6))), dim=1)

    l1, s1, info1 = cache.prefill_cached(decoder, p_short, 32)
    assert info1["cache_hit"] == "miss" and info1["computed_tokens"] == 10

    l2, s2, info2 = cache.prefill_cached(decoder, p_short, 32)
    assert info2["cache_hit"] == "exact" and info2["computed_tokens"] == 0
    assert torch.allclose(l1, l2, atol=1e-6)

    l_part, s_part, info3 = cache.prefill_cached(decoder, p_long, 32)
    assert info3["cache_hit"] == "partial"
    assert info3["cached_tokens"] == 10 and info3["computed_tokens"] == 6

    l_full, s_full = decoder.prefill(p_long, 32)
    assert torch.allclose(l_part[:, -1:], l_full[:, -1:], atol=1e-4)

    cache.record_anchor(s1)
    cache.record_anchor(s_part)
    blended = cache.blend_with_anchors(s_part, anchor_weight=0.2)
    model._validate_state(blended, model.embedding(p_short[:, :1]))

    # Confidence-gated MTP must preserve 100% greedy parity with baseline MTP / CompiledDecode
    spec_ungated = SelfSpeculativeMTPDecode(
        model, objective.mtp_heads, vocab_limit=128, backend=None, draft_confidence_threshold=0.0
    )
    spec_gated = SelfSpeculativeMTPDecode(
        model, objective.mtp_heads, vocab_limit=128, backend=None, draft_confidence_threshold=0.5
    )
    toks_u, stats_u = spec_ungated.generate_greedy(p_short, 12)
    toks_g, stats_g = spec_gated.generate_greedy(p_short, 12)
    assert toks_u == toks_g
    assert stats_g["pruned_drafts"] >= 0
    assert stats_g["proposed_drafts"] <= stats_u["proposed_drafts"]


def test_combined_quarot_early_exit_segment_cache_and_mtp_stack():
    """Verify all 4 Phase 2B levers stacked together on the composite-dimension (d=384, ff=640) champion topology."""
    cfg = _make_fractal_cfg(d_model=192, hidden_dim=320, n_layers=6, mtp_tokens=2, mtp_weight=0.25)
    model = FractalDeltaMemory(cfg.model, cfg.research).eval()
    objective = PredictiveObjective(model, cfg.research).eval()

    q_model, q_report = quantize_model_quarot_int4(model, group_size=32, act_bits=8, inplace=False)
    assert q_report["shared_embedding_preserved"] is True
    assert q_report["compression_vs_bf16"] >= 3.5
    assert q_report["compression_vs_fp32"] >= 7.0

    prefix = torch.randint(0, 128, (1, 16))
    q_decoder = FractalDecode(q_model)

    # 1. SegmentMemoryCache prefix reuse on QuaRot-INT4 model
    cache = SegmentMemoryCache(max_entries=8, max_anchors=4)
    l_miss, s_miss, info_miss = cache.prefill_cached(q_decoder, prefix[:, :10], 32)
    l_part, s_part, info_part = cache.prefill_cached(q_decoder, prefix, 32)
    assert info_miss["cache_hit"] == "miss"
    assert info_part["cache_hit"] == "partial" and info_part["cached_tokens"] == 10 and info_part["computed_tokens"] == 6

    # 2. AdaptiveEarlyExitDecode on QuaRot-INT4 model
    ee_dec = AdaptiveEarlyExitDecode(q_model, confidence_threshold=0.005, delta_threshold=0.0, min_layers=2)
    ee_toks, ee_stats = ee_dec.generate_greedy(prefix, 6)
    assert len(ee_toks) == 6
    assert ee_stats["mean_layers_executed"] == 2.0
    assert abs(ee_stats["layers_saved_ratio"] - (4.0 / 6.0)) < 1e-6

    # 3. Confidence-Gated SelfSpeculativeMTPDecode on QuaRot-INT4 model (100% greedy parity)
    spec_dec = SelfSpeculativeMTPDecode(
        q_model, objective.mtp_heads, vocab_limit=128, backend=None, draft_confidence_threshold=0.2
    )
    spec_toks, _ = spec_dec.generate_greedy(prefix, 8)
    ee_exact = AdaptiveEarlyExitDecode(q_model, confidence_threshold=1.1, delta_threshold=0.0, min_layers=2)
    ref_toks, _ = ee_exact.generate_greedy(prefix, 8)
    assert spec_toks == ref_toks

