import pytest
import torch

from llm2.config import ExperimentConfig, ModelConfig
from llm2.model import (CausalSDPA, RMSNorm, RotaryEmbedding, TransformerLM,
                       count_parameters, transformer_parameter_formula)
from llm2.objectives import CausalLanguageModeling


def test_shapes_tying_and_backward(tiny_config):
    model = TransformerLM(tiny_config)
    x = torch.randint(0, 32, (2, 12))
    output = model(x)
    assert output.logits.shape == (2, 12, 32)
    assert output.hidden_states.shape == (2, 12, 32)
    assert model.embedding.weight is model.lm_head.weight
    result = CausalLanguageModeling()(model, {"input_ids": x, "targets": x.roll(-1, 1)})
    result.loss.backward()
    assert result.token_count == 24
    assert torch.isfinite(result.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_causality(tiny_config):
    model = TransformerLM(tiny_config).eval()
    x = torch.randint(0, 32, (2, 16))
    changed = x.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 32
    torch.testing.assert_close(model(x).logits[:, :8], model(changed).logits[:, :8], rtol=0, atol=0)
    kernel = CausalSDPA()
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    actual = kernel(q, k, v)
    scores = q @ k.transpose(-2, -1) / (8**0.5)
    scores.masked_fill_(~torch.ones(16, 16, dtype=torch.bool).tril(), -float("inf"))
    torch.testing.assert_close(actual, scores.softmax(-1) @ v)


def test_numeric_stability_and_rope():
    x = (torch.randn(2, 4, 32) * 1e10).requires_grad_()
    result = RMSNorm(32)(x)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(RMSNorm(32)(torch.zeros_like(x))).all()
    rope = RotaryEmbedding(8, 32, 10000)
    q = torch.randn(2, 4, 16, 8)
    torch.testing.assert_close(rope(q).square().sum(-1), q.square().sum(-1))
    torch.testing.assert_close(rope(q)[..., 0, :], q[..., 0, :])


def test_eval_disables_dropout(tiny_config):
    tiny_config.dropout = 0.5
    model = TransformerLM(tiny_config).eval()
    x = torch.randint(0, 32, (2, 8))
    torch.testing.assert_close(model(x).logits, model(x).logits, rtol=0, atol=0)


def test_reference_parameter_count():
    config = ModelConfig()
    model = TransformerLM(config)
    assert count_parameters(model) == transformer_parameter_formula(config) == 33_990_144
    x = torch.randint(0, config.vocab_size, (1, 8))
    loss = CausalLanguageModeling()(model, {"input_ids": x, "targets": x.roll(-1, 1)}).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_bad_config_and_context(tiny_config):
    with pytest.raises(ValueError):
        ModelConfig(d_model=31)
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"train": {"max_step": 10}})
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"train": {"compile": "false"}})
    with pytest.raises(ValueError):
        TransformerLM(tiny_config)(torch.zeros(2, 33, dtype=torch.long))

