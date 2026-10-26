import copy

import pytest
import torch

from llm2.config import ResearchConfig
from llm2.architectures.latent_recursive import LatentRecursiveModel
from llm2.generation import generate
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research.accounting import estimate_research_flops
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("shape", [(1, 3), (2, 12)])
def test_recursive_shapes_causality_backward(tiny_config, steps, shape):
    r = ResearchConfig(architecture="recursive", iterations=steps, record_states=True)
    model = LatentRecursiveModel(tiny_config, r)
    ids = torch.randint(0, 32, shape)
    out = model(ids)
    assert out.logits.shape == (*shape, 32)
    assert out.state.value.shape == (*shape, 32)
    assert out.state.iterations == len(out.intermediate_states) == steps
    out.logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    changed = ids.clone()
    prefix = shape[1] // 2
    changed[:, prefix:] = (changed[:, prefix:] + 1) % 32
    torch.testing.assert_close(model(ids).logits[:, :prefix], model(changed).logits[:, :prefix], rtol=0, atol=0)
    torch.testing.assert_close(model(ids).logits, model(ids).logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cross-segment"):
        model(ids, state=out.state)


def test_sharing_cost_and_untied_control(tiny_config):
    models = [LatentRecursiveModel(tiny_config, ResearchConfig(architecture="recursive", iterations=n)) for n in (1, 4)]
    assert count_parameters(models[0]) == count_parameters(models[1])
    untied = LatentRecursiveModel(tiny_config, ResearchConfig(architecture="untied", iterations=4))
    assert len({id(c.context_projection.weight) for c in untied.cores}) == 4
    assert count_parameters(untied) > count_parameters(models[1])
    for model in models:
        assert model.embedding.weight is model.lm_head.weight
    objective = CausalLanguageModeling()
    a, b = [estimate_research_flops(m, objective, 8) for m in models]
    assert b.training_per_sequence > a.training_per_sequence
    assert b.internal_iterations == 4
    # Same unrolled operations, copied into distinct parameter tensors.
    untied.embedding.load_state_dict(models[1].embedding.state_dict())
    untied.norm.load_state_dict(models[1].norm.state_dict())
    for core in untied.cores:
        core.load_state_dict(models[1].core.state_dict())
    ids = torch.randint(0, 32, (2, 8))
    torch.testing.assert_close(models[1](ids).logits, untied(ids).logits, rtol=0, atol=0)


def test_checkpointing_and_truncation_explicit(tiny_config):
    r = ResearchConfig(architecture="recursive", iterations=3)
    plain = LatentRecursiveModel(tiny_config, r)
    r2 = copy.deepcopy(r)
    r2.gradient_checkpointing = True
    recomputed = LatentRecursiveModel(tiny_config, r2)
    recomputed.load_state_dict(plain.state_dict())
    ids = torch.randint(0, 32, (2, 8))
    for model in (plain, recomputed):
        model(ids).logits.square().mean().backward()
    for a, b in zip(plain.parameters(), recomputed.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert estimate_research_flops(recomputed, CausalLanguageModeling(), 8).recomputation_forward > 0
    r2.detach_every = 1
    truncated = LatentRecursiveModel(tiny_config, r2)
    truncated.load_state_dict(plain.state_dict())
    truncated(ids).logits.square().mean().backward()
    assert torch.isfinite(truncated.core.gate.grad).all()
    assert not torch.equal(truncated.core.gate.grad, plain.core.gate.grad)


@pytest.mark.parametrize("architecture", ["recursive", "untied"])
def test_recursive_memorization_and_generation(tiny_config, architecture):
    model = LatentRecursiveModel(tiny_config, ResearchConfig(architecture=architecture, iterations=2))
    objective = CausalLanguageModeling()
    ids = torch.arange(12).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = objective(model, batch).loss.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch).loss
        loss.backward()
        optimizer.step()
    assert objective(model, batch).loss.item() < initial / 15
    out = generate(model, ids[:1, :2], max_seq_len=16, max_new_tokens=4, temperature=0)
    assert out.shape == (1, 6)


@pytest.mark.parametrize("jepa", [0.0, 0.2])
def test_recursive_exact_resume(training_config, tmp_path, jepa):
    cfg = training_config
    cfg.research = ResearchConfig(architecture="recursive", iterations=2, jepa_weight=jepa,
                                  latent_steps=1, latent_warmup_steps=1, gradient_checkpointing=True)
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        expected_metrics = trainer.train_step()
        expected = copy.deepcopy((trainer.model.state_dict(), trainer.objective.state_dict(),
                                  trainer.optimizer.state_dict(), trainer.sampler.state_dict()))
    cfg.train.output_dir = str(tmp_path / "resume")
    with Trainer(cfg, resume=checkpoint) as trainer:
        metrics = trainer.train_step()
        assert_nested_equal(expected, (trainer.model.state_dict(), trainer.objective.state_dict(),
                                       trainer.optimizer.state_dict(), trainer.sampler.state_dict()))
        assert metrics["training_loss"] == expected_metrics["training_loss"]
        assert metrics["training_flops_estimate"] == expected_metrics["training_flops_estimate"]

