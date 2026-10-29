"""Verification tests for Etapa 1 (non-invasive magnitude/gradient telemetry) and Etapa 2
(compressed_native vs compressed_grouped without JEPA + cache_context_projection) per
docs/archive/PLAN_FINAL_EXECUTION_2026_10_28.md.
"""

import copy
import json

import pytest
import torch

from llm2.architectures.fractal_recursive import FractalLatentModel, fractal_parameter_formula
from llm2.architectures.latent_recursive import LatentRecursiveModel
from llm2.cli import main as cli_main
from llm2.config import ExperimentConfig, ModelConfig, ResearchConfig
from llm2.model import TransformerLM, count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research.accounting import estimate_research_flops
from llm2.research.campaign import campaign_config, main as campaign_main
from llm2.research.diagnostics import diagnose_model
from llm2.runtime import attention_context, capture_rng
from llm2.trainer import Trainer
from test_training import assert_nested_equal, training_config


def _state(trainer):
    keys = (
        "step",
        "tokens_processed",
        "sequences_processed",
        "input_tokens_processed",
        "training_flops_estimate",
        "validation_flops_estimate",
        "curriculum_probe_flops_estimate",
    )
    return copy.deepcopy((
        trainer.model.state_dict(),
        trainer.objective.state_dict(),
        trainer.optimizer.state_dict(),
        trainer.scheduler.state_dict(),
        trainer.sampler.state_dict(),
        capture_rng(),
        {key: trainer.counters[key] for key in keys},
    ))


@attention_context(True)
def test_etapa1_diagnose_model_has_zero_side_effects_and_reports_padding_and_null_grads(tiny_config):
    tiny_config.max_seq_len = 33
    tiny_config.dropout = 0.2
    model = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution="compressed_grouped",
            iterations=2,
            jepa_weight=0.0,
        ),
    )
    # Put model in train mode with pre-existing non-zero parameter gradients.
    model.train()
    ids = torch.randint(0, 32, (2, 33))
    targets = torch.randint(0, 32, (2, 33))
    init_loss = CausalLanguageModeling()(model, {"input_ids": ids, "targets": targets}).loss
    init_loss.backward()

    weights_before = copy.deepcopy(model.state_dict())
    grads_before = {name: (None if p.grad is None else p.grad.detach().clone()) for name, p in model.named_parameters()}
    modes_before = [m.training for m in model.modules()]
    rng_before = capture_rng()

    diag = diagnose_model(
        model,
        ids,
        targets=targets,
        batch_source="unit_test_validation_window",
        include_gradients=True,
        max_samples=16,
    )

    # Verify strict restoration of weights, gradients, module modes, and RNG.
    assert_nested_equal(weights_before, model.state_dict())
    for name, p in model.named_parameters():
        if grads_before[name] is None:
            assert p.grad is None, name
        else:
            assert torch.equal(grads_before[name], p.grad), f"Gradient mutated on {name}"
    assert [m.training for m in model.modules()] == modes_before
    assert_nested_equal(rng_before, capture_rng())

    # Verify batch metadata and Etapa 1 telemetry schema.
    meta = diag["batch_metadata"]
    assert meta["batch_source"] == "unit_test_validation_window"
    assert meta["batch_size"] == 2
    assert meta["seq_len"] == 33
    assert meta["evaluated_tokens"] == 66
    assert meta["eval_mode_dropout_disabled"] is True
    assert meta["include_gradients"] is True

    telemetry = diag["stage1_telemetry"]
    assert telemetry is not None
    assert "state_norm_over_projected_context" in telemetry["ratio_definitions"]
    assert len(telemetry["iterations"]) == 2

    iter1 = telemetry["iterations"][0]
    iter2 = telemetry["iterations"][1]
    assert iter1["iteration"] == 1
    assert iter2["iteration"] == 2

    # At iteration 1 (step_index=0), state_in is zero -> state_in RMS == 0 and state_norm RMS == 0.
    for s_name, valid_len in (("s0", 33), ("s1", 9), ("s2", 3)):
        s_entry1 = iter1["scales"][s_name]
        assert s_entry1["valid_length"] == valid_len
        assert s_entry1["activations_rms"]["state_in"] == 0.0
        assert s_entry1["activations_rms"]["state_norm"] == 0.0
        assert s_entry1["activations_rms"]["projected_context"] > 0.0
        assert s_entry1["ratios"]["core_branch_delta_over_state_in"] is None
        # Initial state is a constant zero tensor (requires_grad=False), so state_in grad is null with reason,
        # whereas state_norm_out depends on state_norm.weight (requires_grad=True) and equals core_input_sum grad!
        assert s_entry1["gradients"]["state_in"]["grad_rms"] is None
        assert "constant zero tensor" in s_entry1["gradients"]["state_in"]["null_reason"]
        assert s_entry1["gradients"]["state_norm_out"]["grad_rms"] > 0.0
        assert s_entry1["gradients"]["core_input_sum"]["grad_rms"] > 0.0
        assert abs(
            s_entry1["gradients"]["state_norm_out"]["grad_rms"]
            - s_entry1["gradients"]["core_input_sum"]["grad_rms"]
        ) < 1e-6
        assert s_entry1["gradients"]["post_interaction_state"]["grad_rms"] > 0.0

    # At iteration 2 (step_index=1), state_in > 0 and state_norm > 0; s0 post_interaction_state receives grad, while s1/s2 at final iteration are unused by lm_head.
    for s_name in ("s0", "s1", "s2"):
        s_entry2 = iter2["scales"][s_name]
        assert s_entry2["activations_rms"]["state_in"] > 0.0
        assert s_entry2["activations_rms"]["state_norm"] > 0.0
        assert s_entry2["ratios"]["state_norm_over_projected_context"] > 1.0
        assert s_entry2["ratios"]["core_branch_delta_over_state_in"] > 0.0
        assert s_entry2["gradients"]["state_in"]["grad_rms"] > 0.0
        assert s_entry2["gradients"]["state_norm_out"]["grad_rms"] > 0.0
        assert s_entry2["gradients"]["core_input_sum"]["grad_rms"] > 0.0

    assert iter2["scales"]["s0"]["gradients"]["post_interaction_state"]["grad_rms"] > 0.0
    for macro_scale in ("s1", "s2"):
        assert iter2["scales"][macro_scale]["gradients"]["post_interaction_state"]["grad_rms"] is None
        assert "not consumed by lm_head" in iter2["scales"][macro_scale]["gradients"]["post_interaction_state"]["null_reason"]

    # Verify right-padding on s2 in compressed_grouped mode is excluded from valid s2 stats and reported separately.
    pad1 = iter1["padding_telemetry"]["s2"]
    assert pad1["valid_positions"] == 3
    assert pad1["padded_positions"] == 6
    assert abs(pad1["padded_fraction"] - 6.0 / 9.0) < 1e-6

    # Verify native mode has no padding telemetry and matches s2 valid_length.
    native_model = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution="compressed_native",
            iterations=2,
            jepa_weight=0.0,
        ),
    )
    native_model.load_state_dict(model.state_dict())
    diag_native = diagnose_model(native_model, ids, targets=targets, batch_source="unit_test_native")
    assert diag_native["stage1_telemetry"]["iterations"][0]["padding_telemetry"] is None
    for s_name in ("s0", "s1", "s2"):
        # Valid-position activation RMS at iteration 1 and 2 must match between grouped and native in eval mode!
        torch.testing.assert_close(
            diag["stage1_telemetry"]["iterations"][1]["scales"][s_name]["activations_rms"]["state_norm"],
            diag_native["stage1_telemetry"]["iterations"][1]["scales"][s_name]["activations_rms"]["state_norm"],
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("scale_execution", ["compressed_grouped", "compressed_native"])
@attention_context(True)
def test_etapa2_cache_context_projection_equivalence_and_causality(tiny_config, scale_execution):
    tiny_config.max_seq_len = 65
    tiny_config.dropout = 0.0
    uncached = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution=scale_execution,
            iterations=2,
            cache_context_projection=False,
        ),
    )
    with torch.no_grad():
        uncached.block.interaction.mixing.normal_(0, 0.3)
        uncached.block.interaction.gate.normal_(-1, 0.3)
        uncached.block.conditioning.film.normal_(0, 0.2)
        uncached.block.comp4.prior.normal_(0, 0.2)
        uncached.block.comp16.prior.normal_(0, 0.2)
    cached = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution=scale_execution,
            iterations=2,
            cache_context_projection=True,
        ),
    )
    cached.load_state_dict(uncached.state_dict())

    # Count actual context_projection calls during K=2 forward.
    proj_calls = []
    hook = cached.block.cores[0].context_projection.register_forward_hook(lambda m, a, o: proj_calls.append(a[0].shape))
    try:
        cached(torch.randint(0, 32, (2, 33)))
    finally:
        hook.remove()
    # With cache_context_projection=True, c0, c1, c2 are projected once before the loop (3 calls total across K=2).
    assert len(proj_calls) == 3

    for length in [1, 4, 15, 16, 17, 31, 32, 33, 65]:
        uncached.zero_grad(set_to_none=True)
        cached.zero_grad(set_to_none=True)
        ids = torch.randint(0, 32, (2, length))
        out_u, out_c = uncached(ids), cached(ids)
        torch.testing.assert_close(out_u.logits, out_c.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out_u.hidden_states, out_c.hidden_states, rtol=1e-5, atol=1e-6)
        out_u.logits.square().mean().backward()
        out_c.logits.square().mean().backward()
        for (name, p_u), (other, p_c) in zip(uncached.named_parameters(), cached.named_parameters()):
            assert name == other
            assert (p_u.grad is None) == (p_c.grad is None), name
            if p_u.grad is not None:
                assert torch.isfinite(p_c.grad).all(), name
                torch.testing.assert_close(p_u.grad, p_c.grad, rtol=1e-4, atol=1e-6, msg=name)

    # Strict prefix causality under future token mutations and prefix truncation.
    cached.eval()
    for length in [4, 16, 17, 33, 65]:
        ids = torch.randint(0, 32, (2, length))
        with torch.no_grad():
            original = cached(ids).logits
            for cut in [1, 3, 4, 15, 16, 17, length - 1]:
                if cut >= length:
                    continue
                changed = ids.clone()
                changed[:, cut:] = (changed[:, cut:] + 5) % 32
                torch.testing.assert_close(original[:, :cut], cached(changed).logits[:, :cut], rtol=0, atol=0)
                torch.testing.assert_close(original[:, :cut], cached(ids[:, :cut]).logits, rtol=3e-5, atol=2e-6)


def test_etapa2_analytical_flops_and_parameter_invariance():
    # Match Kaggle T4 campaign dimensions (vocab=257, d=384, h=1024, heads=6, T=256, B=32, core_layers=1).
    cfg = ModelConfig(vocab_size=257, d_model=384, n_layers=6, n_heads=6, hidden_dim=1024, max_seq_len=256)
    obj = CausalLanguageModeling()
    variants = {
        "G0": ResearchConfig(architecture="fractal", scale_execution="compressed_grouped", iterations=2, cache_context_projection=False),
        "G1": ResearchConfig(architecture="fractal", scale_execution="compressed_grouped", iterations=2, cache_context_projection=True),
        "N0": ResearchConfig(architecture="fractal", scale_execution="compressed_native", iterations=2, cache_context_projection=False),
        "N1": ResearchConfig(architecture="fractal", scale_execution="compressed_native", iterations=2, cache_context_projection=True),
    }
    step_gflops = {}
    for key, r_cfg in variants.items():
        with torch.device("meta"):
            model = FractalLatentModel(cfg, r_cfg)
        assert count_parameters(model) == 2_168_076
        assert fractal_parameter_formula(cfg, r_cfg) == 2_168_076
        est = estimate_research_flops(model, obj, 256)
        step_gflops[key] = (est.training_per_sequence * 32) / 1e9

    # Verify exact analytical GFLOPs per step at B=32, T=256:
    # G0 = 328.281883 GFLOPs, G1 = 316.051292 GFLOPs (context projected before padding: 336 pos vs 384),
    # N0 = 291.816604 GFLOPs, N1 = 282.303922 GFLOPs
    assert abs(step_gflops["G0"] - 328.281883) < 1e-4
    assert abs(step_gflops["G1"] - 316.051292) < 1e-4
    assert abs(step_gflops["N0"] - 291.816604) < 1e-4
    assert abs(step_gflops["N1"] - 282.303922) < 1e-4

    # Verify via actual forward hooks on small dimensions (T=256, B=1) that context_projection
    # processes the exact number of token positions counted by the analytical FLOPs estimator.
    small_cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, hidden_dim=32, max_seq_len=256)
    expected_positions = {"G0": 768, "G1": 336, "N0": 672, "N1": 336}
    ids = torch.randint(0, 32, (1, 256))
    for key, r_cfg in variants.items():
        real_model = FractalLatentModel(small_cfg, r_cfg)
        real_model.eval()
        observed_positions = []
        h = real_model.block.cores[0].context_projection.register_forward_hook(
            lambda m, a, o: observed_positions.append(int(a[0].shape[0] * a[0].shape[1]))
        )
        try:
            with torch.no_grad():
                real_model(ids)
        finally:
            h.remove()
        assert sum(observed_positions) == expected_positions[key], (key, observed_positions)


@pytest.mark.parametrize("scale_execution", ["compressed_grouped", "compressed_native"])
@attention_context(True)
def test_etapa1_cached_context_telemetry_regression_matches_uncached_rms_and_reports_scopes(tiny_config, scale_execution):
    tiny_config.max_seq_len = 33
    tiny_config.dropout = 0.0
    uncached = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution=scale_execution,
            iterations=2,
            jepa_weight=0.0,
            cache_context_projection=False,
        ),
    )
    cached = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution=scale_execution,
            iterations=2,
            jepa_weight=0.0,
            cache_context_projection=True,
        ),
    )
    cached.load_state_dict(uncached.state_dict())

    ids = torch.randint(0, 32, (2, 33))
    targets = torch.randint(0, 32, (2, 33))
    diag_u = diagnose_model(uncached, ids, targets=targets, batch_source="reg_uncached")
    diag_c = diagnose_model(cached, ids, targets=targets, batch_source="reg_cached")

    for it_idx in range(2):
        for s_name in ("s0", "s1", "s2"):
            su = diag_u["stage1_telemetry"]["iterations"][it_idx]["scales"][s_name]
            sc = diag_c["stage1_telemetry"]["iterations"][it_idx]["scales"][s_name]
            # Regression check: cached G1/N1 must NOT record s2's projection on s0!
            # Every scale and iteration must have identical projected_context RMS between uncached and cached.
            torch.testing.assert_close(
                su["activations_rms"]["projected_context"],
                sc["activations_rms"]["projected_context"],
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                su["ratios"]["state_norm_over_projected_context"],
                sc["ratios"]["state_norm_over_projected_context"],
                rtol=1e-5,
                atol=1e-6,
            )
            # Scope checks:
            assert su["gradients"]["projected_context"]["scope"] == "per_call"
            assert su["gradients"]["projected_context_call_input"]["scope"] == "per_call"
            assert su["gradients"]["cached_projected_context_accumulated"]["grad_rms"] is None

            assert sc["gradients"]["projected_context"]["scope"] == "accumulated_across_iterations"
            assert sc["gradients"]["projected_context_call_input"]["scope"] == "per_call"
            assert sc["gradients"]["cached_projected_context_accumulated"]["scope"] == "accumulated_across_iterations"

            # Per-call gradient contribution to projected_context in cached mode must equal uncached per-call grad!
            torch.testing.assert_close(
                su["gradients"]["projected_context"]["grad_rms"],
                sc["gradients"]["projected_context_call_input"]["grad_rms"],
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                su["gradients"]["projected_context"]["grad_l2_norm"],
                sc["gradients"]["projected_context_call_input"]["grad_l2_norm"],
                rtol=1e-5,
                atol=1e-6,
            )


def test_etapa3_paired_common_init_protocol_and_sampler_alignment(training_config, tmp_path):
    from llm2.research.factory import build_backbone
    from llm2.research.pairing import iter_common_named_buffers, iter_common_named_parameters
    from llm2.runtime import setup_runtime

    base = training_config
    base.model.max_seq_len = base.data.seq_len = 17
    recurrent_aliases = [
        "lrm_no_jepa_k1",
        "lrm_no_jepa_k2",
        "flrm_compressed_grouped_no_jepa_k1",
        "flrm_compressed_grouped_no_jepa_k2",
        "flrm_compressed_native_cached_no_jepa_k2",
    ]

    # 1. Verify that WITHOUT paired_common_init, LRM and FLRM have different common weights under the same seed.
    cfg_lrm_unpaired = campaign_config(
        base, "lrm_no_jepa_k2", base.data.path, tmp_path / "u-lrm",
        seed=42, steps=2, device="cpu", max_seconds=0, paired_common_init=False,
    )
    cfg_flrm_unpaired = campaign_config(
        base, "flrm_compressed_grouped_no_jepa_k2", base.data.path, tmp_path / "u-flrm",
        seed=42, steps=2, device="cpu", max_seconds=0, paired_common_init=False,
    )
    setup_runtime(cfg_lrm_unpaired.train)
    m_lrm_u = build_backbone(cfg_lrm_unpaired)
    setup_runtime(cfg_flrm_unpaired.train)
    m_flrm_u = build_backbone(cfg_flrm_unpaired)
    assert not torch.equal(m_lrm_u.core.attn.qkv.weight if hasattr(m_lrm_u.core, "attn") else m_lrm_u.core.blocks[0].attn.qkv.weight,
                           m_flrm_u.block.cores[0].blocks[0].attn.qkv.weight)

    # 2. Verify that WITH paired_common_init=True, all 5 recurrent models share bit-for-bit identical common parameters & buffers,
    # distinct storage pointers across instances, intact internal weight tying, identical FLRM-exclusive weights across FLRM variants,
    # identical post-init RNG state, and identical initial RandomBatchStream training batches.
    models = {}
    post_init_rngs = {}
    first_batches = {}
    for alias in recurrent_aliases:
        cfg = campaign_config(
            base, alias, base.data.path, tmp_path / f"paired-{alias}",
            seed=42, steps=2, device="cpu", max_seconds=0, paired_common_init=True,
        )
        setup_runtime(cfg.train)
        m = build_backbone(cfg)
        post_init_rngs[alias] = capture_rng()
        models[alias] = m
        assert m.lm_head.weight is m.embedding.weight
        assert m.paired_init_provenance is not None
        assert m.paired_init_provenance["paired_common_init"] is True
        assert m.paired_init_provenance["common_parameter_tensor_count"] == 12

        with Trainer(cfg) as trainer:
            first_batches[alias] = trainer.sampler.next(torch.device("cpu"))

    ref_alias = "lrm_no_jepa_k1"
    ref_model = models[ref_alias]
    ref_common_params = dict(iter_common_named_parameters(ref_model))
    ref_common_bufs = dict(iter_common_named_buffers(ref_model))
    ref_sha = ref_model.paired_init_provenance["common_combined_sha256"]

    for alias, m in models.items():
        assert m.paired_init_provenance["common_combined_sha256"] == ref_sha
        assert_nested_equal(post_init_rngs[ref_alias], post_init_rngs[alias])
        assert torch.equal(first_batches[ref_alias]["input_ids"], first_batches[alias]["input_ids"])
        assert torch.equal(first_batches[ref_alias]["targets"], first_batches[alias]["targets"])
        for cname, cparam in iter_common_named_parameters(m):
            assert torch.equal(ref_common_params[cname], cparam), (alias, cname)
            if alias != ref_alias:
                assert ref_common_params[cname].data_ptr() != cparam.data_ptr(), (alias, cname)
        for bname, cbuf in iter_common_named_buffers(m):
            assert torch.equal(ref_common_bufs[bname], cbuf), (alias, bname)

    # All FLRM variants of the same seed must also have identical FLRM-exclusive weights.
    flrm_aliases = [a for a in recurrent_aliases if a.startswith("flrm")]
    ref_excl_sha = models[flrm_aliases[0]].paired_init_provenance["exclusive_combined_sha256"]
    assert ref_excl_sha is not None
    for alias in flrm_aliases[1:]:
        assert models[alias].paired_init_provenance["exclusive_combined_sha256"] == ref_excl_sha

    # 3. Verify CUDA RNG isolation when CUDA is initialized (simulated mock if CUDA is unavailable locally).
    from unittest.mock import patch
    cuda_state = {"state": [torch.tensor([42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)]}
    def mock_get_all():
        return [s.clone() for s in cuda_state["state"]]
    def mock_set_all(states):
        cuda_state["state"] = [s.clone() for s in states]
    def mock_seed_all(s):
        t = torch.zeros(16, dtype=torch.uint8)
        t[0], t[1] = s % 256, (s // 256) % 256
        cuda_state["state"] = [t]

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_initialized", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_rng_state_all", side_effect=mock_get_all), \
         patch("torch.cuda.set_rng_state_all", side_effect=mock_set_all), \
         patch("torch.cuda.manual_seed_all", side_effect=mock_seed_all):
        cfg_lrm = campaign_config(base, "lrm_no_jepa_k1", base.data.path, tmp_path / "mock-lrm", seed=42, steps=2, device="cpu", max_seconds=0, paired_common_init=True)
        cfg_flrm = campaign_config(base, "flrm_compressed_grouped_no_jepa_k2", base.data.path, tmp_path / "mock-flrm", seed=42, steps=2, device="cpu", max_seconds=0, paired_common_init=True)
        setup_runtime(cfg_lrm.train)
        m_lrm = build_backbone(cfg_lrm)
        rng_lrm = capture_rng()
        setup_runtime(cfg_flrm.train)
        m_flrm = build_backbone(cfg_flrm)
        rng_flrm = capture_rng()
        assert_nested_equal(rng_lrm, rng_flrm)


@pytest.mark.parametrize(
    "alias,expected_exec,expected_cache",
    [
        ("flrm_compressed_native_no_jepa_k2", "compressed_native", False),
        ("flrm_compressed_grouped_cached_no_jepa_k2", "compressed_grouped", True),
        ("flrm_compressed_native_cached_no_jepa_k2", "compressed_native", True),
    ],
)
def test_etapa2_exact_resume_with_dropout_and_accumulation(training_config, tmp_path, alias, expected_exec, expected_cache):
    base = training_config
    base.model.max_seq_len = base.data.seq_len = 17
    cfg = campaign_config(
        base,
        alias,
        base.data.path,
        tmp_path / f"run-{alias}",
        seed=42,
        steps=4,
        device="cpu",
        max_seconds=0,
        batch_size=base.data.batch_size,
        grad_accum_steps=base.train.grad_accum_steps,
    )
    cfg.research.gradient_checkpointing = True
    assert cfg.research.scale_execution == expected_exec
    assert cfg.research.cache_context_projection is expected_cache
    assert cfg.research.jepa_weight == 0.0
    assert cfg.model.dropout > 0 and cfg.train.grad_accum_steps > 1

    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        before = _state(trainer)
        metrics = trainer.train_step()
        expected = _state(trainer)
    torch.manual_seed(999)
    cfg.train.output_dir = str(tmp_path / f"resume-{alias}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert_nested_equal(before, _state(trainer))
        resumed_metrics = trainer.train_step()
        assert_nested_equal(expected, _state(trainer))
        for key in ("training_loss", "grad_norm", "lr", "tokens_processed", "training_flops_estimate"):
            assert metrics[key] == resumed_metrics[key]


def test_etapa1_and_etapa2_batch_profile_alternating_repeats_and_init_diagnostics(training_config, tmp_path):
    cfg = training_config
    cfg_path = tmp_path / "base.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    out_dir = tmp_path / "profile-campaign"
    campaign_main([
        "--execute",
        "--config", str(cfg_path),
        "--data", str(cfg.data.path),
        "--output", str(out_dir),
        "--device", "cpu",
        "--profile-batches",
        "--batch-matrix", "2x1",
        "--profile-backends", "eager",
        "--profile-repeats", "2",
        "--diagnose-init",
        "--warmup-steps", "1",
        "--measure-steps", "2",
        "--models",
        "flrm_compressed_grouped_no_jepa_k2",
        "flrm_compressed_native_no_jepa_k2",
        "flrm_compressed_grouped_cached_no_jepa_k2",
        "baseline_small",
    ])
    batch_profiles = json.loads((out_dir / "batch_profiles.json").read_text(encoding="utf-8"))
    # 4 models * 1 batch format * 1 backend * 2 repeats = 8 profile runs.
    assert len(batch_profiles) == 8
    rep1_models = [r["model"] for r in batch_profiles[:4]]
    rep2_models = [r["model"] for r in batch_profiles[4:]]
    # Verify alternating order between repetition 1 and repetition 2!
    assert rep2_models == list(reversed(rep1_models))

    # Verify init diagnostics JSONs were generated for all 4 models.
    for m in rep1_models:
        diag_path = out_dir / "diagnostics" / f"init-{m}-seed-42.json"
        assert diag_path.is_file()
        diag_data = json.loads(diag_path.read_text(encoding="utf-8"))
        assert diag_data["batch_metadata"]["batch_source"] == "validation_first_window_b1"
        if m.startswith("flrm"):
            assert diag_data["stage1_telemetry"] is not None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "alias",
    [
        "flrm_compressed_native_no_jepa_k2",
        "flrm_compressed_grouped_cached_no_jepa_k2",
        "flrm_compressed_native_cached_no_jepa_k2",
    ],
)
def test_etapa2_cuda_fp32_save_continue(training_config, tmp_path, alias):
    base = training_config
    base.model.max_seq_len = base.data.seq_len = 17
    cfg = campaign_config(
        base,
        alias,
        base.data.path,
        tmp_path / f"cuda-{alias}",
        seed=42,
        steps=4,
        device="cuda",
        max_seconds=0,
        batch_size=base.data.batch_size,
        grad_accum_steps=base.train.grad_accum_steps,
    )
    with Trainer(cfg) as trainer:
        metrics = trainer.train_step()
        assert metrics["peak_vram_allocated_bytes"] > 0
        checkpoint = trainer.save()
    cfg.train.output_dir = str(tmp_path / f"cuda-resume-{alias}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert torch.isfinite(torch.tensor(trainer.train_step()["training_loss"]))
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())

