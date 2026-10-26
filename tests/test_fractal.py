import copy

import pytest
import torch

from llm2.config import ResearchConfig
from llm2.architectures.fractal_recursive import (FractalLatentModel, FractalScaleEncoder,
                                                 fractal_parameter_formula)
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.generation import generate
from llm2.research.accounting import estimate_research_flops
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


@pytest.mark.parametrize("shared,conditioned,iterations", [(False, False, 1), (True, False, 1), (True, True, 1), (True, True, 3)])
def test_fractal_ablations(tiny_config, shared, conditioned, iterations):
    r = ResearchConfig(architecture="fractal", share_scales=shared, scale_conditioning=conditioned,
                       iterations=iterations, record_states=True)
    model = FractalLatentModel(tiny_config, r)
    assert count_parameters(model) == fractal_parameter_formula(tiny_config, r)
    assert len(model.block.cores) == (1 if shared else 3)
    for batch, length in [(1, 3), (2, 13)]:
        ids = torch.randint(0, 32, (batch, length))
        model.zero_grad(set_to_none=True)
        output = model(ids)
        assert output.logits.shape == (batch, length, 32)
        assert len(output.intermediate_states) == iterations
        output.logits.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        for prefix in [1, length // 2, length - 1]:
            changed = ids.clone()
            changed[:, prefix:] = (changed[:, prefix:] + 7) % 32
            torch.testing.assert_close(model(ids).logits[:, :prefix], model(changed).logits[:, :prefix], rtol=0, atol=0)
        model.train()
    output = generate(model, ids[:1, :2], max_seq_len=16, max_new_tokens=3, temperature=0)
    assert output.shape == (1, 5)


def test_causal_pooling_matches_manual_prefix():
    x = torch.randn(2, 13, 4)
    scales = FractalScaleEncoder([1, 4, 0])(x)
    for time in range(13):
        torch.testing.assert_close(scales[1][:, time], x[:, max(0, time-3):time+1].mean(1))
        torch.testing.assert_close(scales[2][:, time], x[:, :time+1].mean(1))


def test_fractal_memorization_and_sharing_cost(tiny_config):
    r = ResearchConfig(architecture="fractal", iterations=2)
    model = FractalLatentModel(tiny_config, r)
    objective = CausalLanguageModeling()
    ids = torch.arange(12).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = objective(model, batch).loss.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        objective(model, batch).loss.backward()
        optimizer.step()
    assert objective(model, batch).loss.item() < initial / 15
    r2 = copy.deepcopy(r)
    r2.iterations = 4
    deeper = FractalLatentModel(tiny_config, r2)
    assert count_parameters(model) == count_parameters(deeper)
    assert estimate_research_flops(deeper, objective, 8).training_per_sequence > estimate_research_flops(model, objective, 8).training_per_sequence


def test_fractal_resume_jepa_and_checkpointing(training_config, tmp_path):
    cfg = training_config
    cfg.research = ResearchConfig(architecture="fractal", iterations=2, jepa_weight=0.2,
                                  gradient_checkpointing=True, latent_steps=1, latent_warmup_steps=1)
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        expected_loss = trainer.train_step()["training_loss"]
        expected = copy.deepcopy((trainer.model.state_dict(), trainer.objective.state_dict(),
                                  trainer.optimizer.state_dict(), trainer.sampler.state_dict()))
    cfg.train.output_dir = str(tmp_path / "resumed-fractal")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert trainer.train_step()["training_loss"] == expected_loss
        assert_nested_equal(expected, (trainer.model.state_dict(), trainer.objective.state_dict(),
                                       trainer.optimizer.state_dict(), trainer.sampler.state_dict()))


def test_fractal_adaptive_curriculum_resume(training_config, tmp_path):
    cfg = training_config
    cfg.model.max_seq_len = cfg.data.seq_len = 128
    cfg.research = ResearchConfig(architecture="fractal", iterations=2, jepa_weight=0.1,
                                  curriculum="adaptive", max_difficulty=1, curriculum_interval=1, probe_examples=1)
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        path = trainer.last_checkpoint
        expected_metrics = trainer.train_step()
        expected = copy.deepcopy((trainer.model.state_dict(), trainer.objective.state_dict(),
                                  trainer.optimizer.state_dict(), trainer.sampler.state_dict()))
    cfg.train.output_dir = str(tmp_path / "fractal-curriculum-resume")
    with Trainer(cfg, resume=path) as trainer:
        assert trainer.train_step()["training_loss"] == expected_metrics["training_loss"]
        assert_nested_equal(expected, (trainer.model.state_dict(), trainer.objective.state_dict(),
                                       trainer.optimizer.state_dict(), trainer.sampler.state_dict()))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fractal_cuda_bf16(training_config):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    cfg = training_config
    cfg.train.device, cfg.train.precision, cfg.train.deterministic = "cuda", "bf16", False
    cfg.research = ResearchConfig(architecture="fractal", jepa_weight=0.1)
    with Trainer(cfg) as trainer:
        metrics = trainer.train_step()
        assert metrics["peak_vram_allocated_bytes"] > 0
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
