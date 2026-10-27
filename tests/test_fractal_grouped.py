"""Equivalence, causal gradients and independent resume contracts for grouped scales."""

import copy

import pytest
import torch

from llm2.architectures.fractal_recursive import FractalLatentModel, fractal_parameter_formula
from llm2.config import ResearchConfig
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research.accounting import estimate_research_flops
from llm2.research.diagnostics import diagnose_model
from llm2.runtime import capture_rng, attention_context
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


@pytest.mark.parametrize("iterations,conditioned", [(1, False), (1, True), (3, True)])
@attention_context(True)
def test_grouped_outputs_gradients_counts_and_causality(tiny_config, iterations, conditioned):
    tiny_config.dropout = 0.
    reference_r = ResearchConfig(architecture="fractal", iterations=iterations,
                                scale_conditioning=conditioned, record_states=True)
    batched_r = copy.deepcopy(reference_r)
    batched_r.scale_execution = "batched"
    torch.manual_seed(42)
    reference = FractalLatentModel(tiny_config, reference_r)
    if conditioned:
        with torch.no_grad():
            reference.block.conditioning.film.normal_(0, .2)
    with torch.no_grad():
        reference.block.interaction.mixing.normal_(0, .2)
    grouped = FractalLatentModel(tiny_config, batched_r)
    grouped.load_state_dict(reference.state_dict())
    assert count_parameters(grouped) == count_parameters(reference) == fractal_parameter_formula(tiny_config, reference_r)
    assert grouped.embedding.weight is grouped.lm_head.weight
    ids = torch.randint(0, 32, (2, 13))
    a, b = reference(ids), grouped(ids)
    torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=2e-7)
    for x, y in zip(a.intermediate_states, b.intermediate_states):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=2e-6)
    a.logits.square().mean().backward()
    b.logits.square().mean().backward()
    for (name, x), (other, y) in zip(reference.named_parameters(), grouped.named_parameters()):
        assert name == other
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=3e-7)
    assert estimate_research_flops(reference, CausalLanguageModeling(), 13).training_per_sequence == estimate_research_flops(grouped, CausalLanguageModeling(), 13).training_per_sequence
    grouped.eval()
    original = grouped(ids)
    for cut in (1, 3, 4, 5, 12):
        changed = ids.clone()
        changed[:, cut:] = (changed[:, cut:]+7) % 32
        altered = grouped(changed)
        torch.testing.assert_close(original.logits[:, :cut], altered.logits[:, :cut], rtol=0, atol=0)
        torch.testing.assert_close(original.hidden_states[:, :cut], altered.hidden_states[:, :cut], rtol=0, atol=0)
    torch.testing.assert_close(original.hidden_states, grouped(ids, return_logits=False).hidden_states, rtol=0, atol=0)


@pytest.mark.parametrize("execution", ["reference", "batched"])
def test_fractal_execution_exact_resume_full_state(training_config, tmp_path, execution):
    cfg = training_config
    cfg.train.accumulation_weighting = "tokens"
    cfg.research = ResearchConfig(architecture="fractal", scale_execution=execution,
        iterations=2, latent_steps=1, latent_warmup_steps=1,
        gradient_checkpointing=True, jepa_weight=.1, mtp_tokens=1, mtp_weight=.1)
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        expected_loss = trainer.train_step()["training_loss"]
        expected = copy.deepcopy((trainer.model.state_dict(), trainer.objective.state_dict(),
            trainer.optimizer.state_dict(), trainer.scheduler.state_dict(), trainer.sampler.state_dict(), capture_rng()))
    cfg.train.output_dir = str(tmp_path / f"resume-{execution}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert trainer.train_step()["training_loss"] == expected_loss
        assert_nested_equal(expected, (trainer.model.state_dict(), trainer.objective.state_dict(),
            trainer.optimizer.state_dict(), trainer.scheduler.state_dict(), trainer.sampler.state_dict(), capture_rng()))


def test_grouped_memorization_and_diagnostics(tiny_config):
    model = FractalLatentModel(tiny_config, ResearchConfig(architecture="fractal", scale_execution="batched"))
    ids = torch.arange(12).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    objective = CausalLanguageModeling()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    initial = objective(model, batch).loss.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        objective(model, batch).loss.backward()
        optimizer.step()
    assert objective(model, batch).loss.item() < initial/15
    result = diagnose_model(model, ids, max_samples=12)
    assert len(result["scales"]) == 3
    assert len(result["linear_cka"]) == 3
    assert len(result["iterations"]) == 2
    assert result["film"]["gamma_gain_min"] > .9


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("execution", ["reference", "batched"])
def test_fractal_cuda_fp32_checkpoint_continue(training_config, tmp_path, execution):
    cfg = training_config
    cfg.train.device, cfg.train.precision, cfg.train.deterministic = "cuda", "fp32", False
    cfg.train.accumulation_weighting = "tokens"
    cfg.research = ResearchConfig(architecture="fractal", scale_execution=execution, jepa_weight=.1)
    with Trainer(cfg) as trainer:
        metrics = trainer.train_step()
        assert metrics["peak_vram_allocated_bytes"] > 0
        checkpoint = trainer.save()
    cfg.train.output_dir = str(tmp_path / f"cuda-fp32-{execution}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert torch.isfinite(torch.tensor(trainer.train_step()["training_loss"]))
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
