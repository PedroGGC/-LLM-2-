import copy

import pytest
import torch
from torch.nn import functional as F

from llm2.config import ResearchConfig
from llm2.model import TransformerLM
from llm2.research.objectives import PredictiveObjective
from llm2.research.accounting import estimate_research_flops
from llm2.config import ExperimentConfig


@pytest.mark.parametrize("jepa,mtp", [(0.2, 0), (0.0, 2), (0.2, 2)])
def test_predictive_forward_backward_causal_and_cost(tiny_config, jepa, mtp):
    model = TransformerLM(tiny_config)
    r = ResearchConfig(jepa_weight=jepa, mtp_tokens=mtp, mtp_weight=0.1 if mtp else 0)
    objective = PredictiveObjective(model, r).train()
    ids = torch.randint(0, 32, (2, 12))
    result = objective(model, {"input_ids": ids[:, :-1], "targets": ids[:, 1:]})
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in objective.parameters() if p.requires_grad)
    if jepa:
        assert not objective.target.training
        assert all(p.grad is None and not p.requires_grad for p in objective.target.parameters())
        assert "collapse_fraction" in result.metrics and result.metrics["representation_std"] > 0
    model.eval()
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 32
    torch.testing.assert_close(model(ids).logits[:, :6], model(changed).logits[:, :6], rtol=0, atol=0)
    cost = estimate_research_flops(model, objective, 12)
    assert cost.auxiliary_forward > 0
    assert (cost.target_forward > 0) == bool(jepa)


def test_ema_and_target_stop_gradient(tiny_config):
    model = TransformerLM(tiny_config)
    objective = PredictiveObjective(model, ResearchConfig(jepa_weight=0.2, ema_decay=0.5))
    original = copy.deepcopy(objective.target.state_dict())
    with torch.no_grad():
        model.embedding.weight.add_(1)
    objective.after_optimizer_step(model)
    torch.testing.assert_close(objective.target.embedding.weight, original["embedding.weight"] + 0.5)
    assert objective.ema_updates.item() == 1
    assert objective.target.embedding.weight is objective.target.lm_head.weight


def test_mtp_alignment_matches_explicit_offsets(tiny_config):
    model = TransformerLM(tiny_config)
    objective = PredictiveObjective(model, ResearchConfig(mtp_tokens=2, mtp_weight=0.3))
    ids = torch.randint(0, 32, (2, 12))
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    result = objective(model, batch)
    hidden = model(batch["input_ids"]).hidden_states
    expected = sum(F.cross_entropy(head(hidden[:, :-k]).flatten(0, 1), batch["targets"][:, k:].flatten())
                   for k, head in enumerate(objective.mtp_heads, start=1)) / 2
    assert result.metrics["mtp_loss"] == pytest.approx(expected.item())
    assert result.loss.item() == pytest.approx(result.metrics["lm_loss"] + 0.3 * expected.item())


def test_legacy_config_stays_legacy():
    cfg = ExperimentConfig()
    assert "research" not in cfg.to_dict()
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        ResearchConfig(mtp_tokens=1)
    with pytest.raises(ValueError):
        ResearchConfig(architecture="unknown")
