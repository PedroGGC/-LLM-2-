"""Verification tests for Campanha 0005: Asymmetric V-Cycle, TorchScript fusion, and Causal NCM."""

import copy

import pytest
import torch

from llm2.architectures.fractal_recursive import FractalLatentModel, fractal_parameter_formula
from llm2.config import ExperimentConfig, ModelConfig, ResearchConfig, load_config
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research.campaign import campaign_config
from llm2.research.diagnostics import diagnose_model
from llm2.runtime import attention_context, capture_rng
from llm2.trainer import Trainer
from test_training import assert_nested_equal, training_config


def _state(trainer):
    keys = ("step", "tokens_processed", "sequences_processed", "input_tokens_processed",
            "training_flops_estimate", "validation_flops_estimate", "curriculum_probe_flops_estimate")
    return copy.deepcopy((trainer.model.state_dict(), trainer.objective.state_dict(),
        trainer.optimizer.state_dict(), trainer.scheduler.state_dict(), trainer.sampler.state_dict(),
        capture_rng(), {key: trainer.counters[key] for key in keys}))


@pytest.mark.parametrize("ncm_slots", [0, 16])
@attention_context(True)
def test_vcycle_parameter_counts_and_formula(ncm_slots):
    # Match Kaggle T4 campaign dimensions (vocab=257, d=384, h=1024, heads=6).
    cfg = ModelConfig(vocab_size=257, d_model=384, n_layers=6, n_heads=6, hidden_dim=1024, max_seq_len=256)
    research = ResearchConfig(
        architecture="fractal",
        scale_execution="compressed_grouped",
        iterations=2,
        v_cycle=True,
        ncm_slots=ncm_slots,
        fused_film=True,
    )
    model = FractalLatentModel(cfg, research)
    expected = 2_168_076 + ncm_slots * 384
    assert count_parameters(model) == expected
    assert fractal_parameter_formula(cfg, research) == expected
    assert model.lm_head.weight is model.embedding.weight


@attention_context(True)
def test_fused_film_and_native_numerical_equivalence(tiny_config):
    tiny_config.max_seq_len = 65
    tiny_config.dropout = 0.0
    unfused = FractalLatentModel(
        tiny_config,
        ResearchConfig(architecture="fractal", scale_execution="compressed_grouped", iterations=2, fused_film=False),
    )
    with torch.no_grad():
        unfused.block.interaction.mixing.normal_(0, 0.3)
        unfused.block.interaction.gate.normal_(-1, 0.3)
        unfused.block.conditioning.film.normal_(0, 0.2)
        unfused.block.comp4.prior.normal_(0, 0.2)
        unfused.block.comp16.prior.normal_(0, 0.2)
    fused = FractalLatentModel(
        tiny_config,
        ResearchConfig(architecture="fractal", scale_execution="compressed_grouped", iterations=2, fused_film=True),
    )
    fused.load_state_dict(unfused.state_dict())
    for length in [1, 4, 15, 16, 17, 31, 32, 33, 65]:
        unfused.zero_grad(set_to_none=True)
        fused.zero_grad(set_to_none=True)
        ids = torch.randint(0, 32, (2, length))
        a, b = unfused(ids), fused(ids)
        torch.testing.assert_close(a.logits, b.logits, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(a.hidden_states, b.hidden_states, rtol=3e-5, atol=2e-6)
        a.logits.square().mean().backward()
        b.logits.square().mean().backward()
        for (name, x), (other, y) in zip(unfused.named_parameters(), fused.named_parameters()):
            assert name == other
            assert (x.grad is None) == (y.grad is None), name
            if x.grad is not None:
                assert torch.isfinite(y.grad).all(), name
                torch.testing.assert_close(x.grad, y.grad, rtol=3e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("ncm_slots", [0, 16])
@attention_context(True)
def test_vcycle_shapes_and_analytical_cost_reduction(tiny_config, ncm_slots):
    tiny_config.max_seq_len = 33
    research = ResearchConfig(
        architecture="fractal",
        scale_execution="compressed_grouped",
        iterations=2,
        v_cycle=True,
        ncm_slots=ncm_slots,
        fused_film=True,
    )
    model = FractalLatentModel(tiny_config, research)
    core_shapes = []
    projection_shapes = []
    core_hook = model.block.cores[0].register_forward_pre_hook(lambda m, args: core_shapes.append(args[0].shape[:2]))
    projection_hook = model.block.interaction.projection.register_forward_pre_hook(
        lambda m, args: projection_shapes.append(args[0].shape[:2])
    )
    try:
        model(torch.zeros(2, 33, dtype=torch.long))
    finally:
        core_hook.remove()
        projection_hook.remove()
    # Iteration 0 runs (2, 33) and (4, 9); Iteration 1 (V-Cycle) skips (2, 33) and runs only (4, 9)!
    assert core_shapes == [(2, 33), (4, 9), (4, 9)]
    # Fused interaction packs all 3 scales (33 + 9 + 3 = 45) into a single linear projection per iteration!
    assert projection_shapes == [(2, 45), (2, 45)]

    # Verify analytical FLOP reduction vs symmetric K=2
    symmetric = FractalLatentModel(
        tiny_config,
        ResearchConfig(architecture="fractal", scale_execution="compressed_grouped", iterations=2, v_cycle=False),
    )
    sym_fwd, _, _ = symmetric.analytical_cost(33)
    vcycle_fwd, _, steps = model.analytical_cost(33)
    assert steps == 2
    assert vcycle_fwd < sym_fwd


@pytest.mark.parametrize("ncm_slots", [0, 16])
@attention_context(True)
def test_vcycle_and_ncm_strict_prefix_causality(tiny_config, ncm_slots):
    tiny_config.max_seq_len = 65
    tiny_config.dropout = 0.0
    model = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution="compressed_grouped",
            iterations=2,
            v_cycle=True,
            ncm_slots=ncm_slots,
            fused_film=True,
        ),
    )
    with torch.no_grad():
        model.block.interaction.mixing.normal_(0, 0.3)
        model.block.interaction.gate.normal_(-1, 0.3)
        model.block.conditioning.film.normal_(0, 0.2)
        model.block.comp4.prior.normal_(0, 0.2)
        model.block.comp16.prior.normal_(0, 0.2)
        if model.block.ncm is not None:
            model.block.ncm.memory.normal_(0, 0.1)
    model.eval()
    for length in [4, 16, 17, 33, 65]:
        ids = torch.randint(0, 32, (2, length))
        with torch.no_grad():
            original = model(ids).logits
            for cut in [1, 3, 4, 15, 16, 17, length - 1]:
                if cut >= length:
                    continue
                changed = ids.clone()
                changed[:, cut:] = (changed[:, cut:] + 7) % 32
                torch.testing.assert_close(original[:, :cut], model(changed).logits[:, :cut], rtol=0, atol=0)
                torch.testing.assert_close(original[:, :cut], model(ids[:, :cut]).logits, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("ncm_slots", [0, 16])
def test_vcycle_exact_resume_dropout_accumulation(training_config, tmp_path, ncm_slots):
    cfg = training_config
    cfg.model.max_seq_len = cfg.data.seq_len = 17
    cfg.train.accumulation_weighting = "tokens"
    cfg.research = ResearchConfig(
        architecture="fractal",
        scale_execution="compressed_grouped",
        iterations=2,
        v_cycle=True,
        ncm_slots=ncm_slots,
        fused_film=True,
        gradient_checkpointing=True,
    )
    assert cfg.model.dropout > 0 and cfg.train.grad_accum_steps > 1
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        before = _state(trainer)
        metrics = trainer.train_step()
        expected = _state(trainer)
    torch.manual_seed(999)
    cfg.train.output_dir = str(tmp_path / f"resume-vcycle-{ncm_slots}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert_nested_equal(before, _state(trainer))
        resumed_metrics = trainer.train_step()
        assert_nested_equal(expected, _state(trainer))
        for key in ("training_loss", "grad_norm", "lr", "tokens_processed", "training_flops_estimate"):
            assert metrics[key] == resumed_metrics[key]


@pytest.mark.parametrize("ncm_slots", [0, 16])
def test_vcycle_memorization_and_diagnostics(tiny_config, ncm_slots):
    model = FractalLatentModel(
        tiny_config,
        ResearchConfig(
            architecture="fractal",
            scale_execution="compressed_grouped",
            iterations=2,
            v_cycle=True,
            ncm_slots=ncm_slots,
            fused_film=True,
        ),
    )
    ids = torch.arange(18).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    objective = CausalLanguageModeling()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = objective(model, batch).loss.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert objective(model, batch).loss.item() < initial / 15
    if model.block.ncm is not None:
        assert model.block.ncm.memory.grad is not None
        assert torch.isfinite(model.block.ncm.memory.grad).all()
        assert model.block.ncm.memory.grad.abs().sum() > 0

    diag = diagnose_model(model, torch.randint(0, 32, (2, 17)))
    assert len(diag["iterations"]) == 2
    assert [value["samples"] for value in diag["scales"]] == [34, 10, 4]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("ncm_slots", [0, 16])
def test_vcycle_cuda_fp32_save_continue(training_config, tmp_path, ncm_slots):
    cfg = training_config
    cfg.model.max_seq_len = cfg.data.seq_len = 17
    cfg.train.device, cfg.train.precision, cfg.train.deterministic = "cuda", "fp32", False
    cfg.research = ResearchConfig(
        architecture="fractal",
        scale_execution="compressed_grouped",
        iterations=2,
        v_cycle=True,
        ncm_slots=ncm_slots,
        fused_film=True,
        jepa_weight=0.0,
    )
    with Trainer(cfg) as trainer:
        metrics = trainer.train_step()
        assert metrics["peak_vram_allocated_bytes"] > 0
        checkpoint = trainer.save()
    cfg.train.output_dir = str(tmp_path / f"cuda-vcycle-{ncm_slots}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert torch.isfinite(torch.tensor(trainer.train_step()["training_loss"]))
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_campaign_configs_for_vcycle_models(tmp_path):
    base = ExperimentConfig()
    for name, expected_slots in [("flrm_vcycle_k2", 0), ("flrm_vcycle_ncm_k2", 16)]:
        cfg = campaign_config(
            base,
            name,
            tmp_path / "data",
            tmp_path / name,
            seed=42,
            steps=100,
            device="cpu",
            max_seconds=0,
            batch_size=32,
            grad_accum_steps=1,
        )
        assert cfg.research.architecture == "fractal"
        assert cfg.research.scale_execution == "compressed_grouped"
        assert cfg.research.iterations == 2
        assert cfg.research.jepa_weight == 0.0
        assert cfg.research.v_cycle is True
        assert cfg.research.fused_film is True
        assert cfg.research.ncm_slots == expected_slots
        assert cfg.data.batch_size == 32
        assert cfg.train.grad_accum_steps == 1
