"""Scientific interventions: gradients, causal boundaries, cached math and strict resume."""

import copy

import pytest
import torch

from llm2.architectures.fractal_recursive import CrossScaleInteraction, FractalLatentModel, CausalScaleCompressor
from llm2.architectures.latent_recursive import LatentRecursiveModel, SharedRecursiveCore
from llm2.config import ResearchConfig
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.runtime import attention_context
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal
from test_compressed_optimization import _state


def test_context_normalization_formula_dtype_and_no_parameters(tiny_config):
    plain = SharedRecursiveCore(tiny_config)
    normalized = SharedRecursiveCore(tiny_config, context_normalization="rms")
    normalized.load_state_dict(plain.state_dict())
    assert count_parameters(plain) == count_parameters(normalized)
    x = torch.randn(2, 9, tiny_config.d_model)
    projected = plain.context_projection(x)
    assert torch.equal(plain.project_context(x), projected)
    expected = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + tiny_config.norm_eps)
    torch.testing.assert_close(normalized.project_context(x), expected, rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = normalized.project_context(x)
        assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    zero = normalized.project_context(torch.zeros_like(x))
    assert torch.equal(zero, torch.zeros_like(zero))


@pytest.mark.parametrize("architecture,mode", [("recursive", "reference"), ("fractal", "compressed_native"), ("fractal", "compressed_grouped")])
@attention_context(True)
def test_normalized_cached_outputs_gradients_and_causality(tiny_config, architecture, mode):
    tiny_config.dropout = 0.
    tiny_config.max_seq_len = 33
    constructor = LatentRecursiveModel if architecture == "recursive" else FractalLatentModel
    kwargs = dict(architecture=architecture, scale_execution=mode, context_normalization="rms")
    plain = constructor(tiny_config, ResearchConfig(**kwargs))
    cached = constructor(tiny_config, ResearchConfig(**kwargs, cache_context_projection=True))
    cached.load_state_dict(plain.state_dict())
    for length in (1, 4, 16, 17, 33):
        ids = torch.randint(0, tiny_config.vocab_size, (2, length))
        plain.zero_grad(); cached.zero_grad()
        a, b = plain(ids).logits, cached(ids).logits
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward(); b.square().mean().backward()
        for (name, p), (_, q) in zip(plain.named_parameters(), cached.named_parameters()):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                assert torch.isfinite(q.grad).all(), name
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
        if length > 1:
            cut = length // 2
            changed = ids.clone(); changed[:, cut:] = (changed[:, cut:] + 1) % tiny_config.vocab_size
            torch.testing.assert_close(cached(changed).logits[:, :cut], b[:, :cut], rtol=0, atol=0)


@pytest.mark.parametrize("path", ["reference", "batched", "native", "fused_native"])
def test_no_exchange_preserves_own_transform_and_blocks_other_inputs(tiny_config, path):
    interaction = CrossScaleInteraction(3, tiny_config.d_model, exchange=False, fused=path == "fused_native")
    states = tuple(torch.randn(2, 17, tiny_config.d_model, requires_grad=True) for _ in range(3))
    compressors = (CausalScaleCompressor(4, tiny_config.d_model), CausalScaleCompressor(16, tiny_config.d_model))
    if path in ("native", "fused_native"):
        outputs = interaction.forward_native(states, compressors)
        own = (states[0], compressors[0].compress(states[1]), compressors[1].compress(states[2]))
    elif path == "batched":
        outputs = interaction.forward_batched(torch.stack(states)).unbind(0); own = states
    else:
        outputs = interaction(states); own = states
    for i, value in enumerate(outputs):
        expected = own[i] + interaction.gate[i].sigmoid() * interaction.projection(own[i])
        torch.testing.assert_close(value, expected)
        gradients = torch.autograd.grad(value.sum(), states, retain_graph=True, allow_unused=True)
        for j, gradient in enumerate(gradients):
            if i == j:
                assert gradient is not None and gradient.abs().sum() > 0
            else:
                assert gradient is None or torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("film", [False, True])
def test_film_and_exchange_disabled_contract(tiny_config, film):
    model = FractalLatentModel(tiny_config, ResearchConfig(architecture="fractal", scale_execution="compressed_native",
        context_normalization="rms", scale_conditioning=film, cross_scale_exchange=False))
    assert (model.block.conditioning is not None) == film
    if film:
        x = torch.randn(2, 8, tiny_config.d_model)
        assert torch.equal(model.block.conditioning(x, 0), x)
    assert model.block.interaction.exchange is False


@pytest.mark.parametrize("architecture,film,exchange", [("recursive", True, True), ("fractal", True, True),
                                                       ("fractal", False, False)])
def test_new_controls_exact_resume_dropout_accumulation(training_config, tmp_path, architecture, film, exchange):
    cfg = training_config
    cfg.model.max_seq_len = cfg.data.seq_len = 17
    cfg.train.accumulation_weighting = "tokens"
    cfg.research = ResearchConfig(architecture=architecture, context_normalization="rms", scale_conditioning=film,
        cross_scale_exchange=exchange, scale_execution="compressed_native" if architecture == "fractal" else "reference",
        cache_context_projection=True, paired_common_init=True)
    assert cfg.model.dropout > 0 and cfg.train.grad_accum_steps > 1
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        before = _state(trainer)
        metrics = trainer.train_step(); expected = _state(trainer)
    cfg.train.output_dir = str(tmp_path / "resumed")
    with Trainer(cfg, resume=checkpoint) as resumed:
        assert_nested_equal(before, _state(resumed))
        result = resumed.train_step()
        assert_nested_equal(expected, _state(resumed))
        assert metrics["training_loss"] == result["training_loss"]


@pytest.mark.parametrize("architecture", ["recursive", "fractal"])
def test_normalized_memorization(tiny_config, architecture):
    cls = LatentRecursiveModel if architecture == "recursive" else FractalLatentModel
    research = ResearchConfig(architecture=architecture, context_normalization="rms",
        cross_scale_exchange=architecture != "fractal", scale_conditioning=False,
        scale_execution="compressed_native" if architecture == "fractal" else "reference")
    model = cls(tiny_config, research)
    ids = torch.arange(18).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    objective = CausalLanguageModeling(); optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    initial = objective(model, batch).loss.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch).loss
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
    assert objective(model, batch).loss.item() < initial / 15


@pytest.mark.parametrize("kwargs", [{"context_normalization": "layer"}, {"context_normalization": "rms", "architecture": "transformer"},
                                   {"cross_scale_exchange": False}])
def test_reject_ignored_or_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        ResearchConfig(**kwargs)
