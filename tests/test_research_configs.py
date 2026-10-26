from pathlib import Path

import pytest
import torch

from llm2.config import load_config, ExperimentConfig, ResearchConfig
from llm2.model import count_parameters
from llm2.research.factory import build_backbone, build_objective
from llm2.research.accounting import estimate_research_flops
from llm2.trainer import Trainer


def test_every_executable_research_yaml_builds_and_counts():
    paths = list(Path("configs/experiments").glob("*.yaml")) + list(Path("configs/profiles").glob("*.yaml"))
    for path in paths:
        if path.name == "matrix.yaml":
            continue
        cfg = load_config(path)
        with torch.device("meta"):
            model = build_backbone(cfg)
            objective = build_objective(cfg, model)
        assert count_parameters(model) > 0
        assert estimate_research_flops(model, objective, cfg.data.seq_len).training_per_sequence > 0
    counts = {}
    for path in Path("configs/profiles").glob("*.yaml"):
        cfg = load_config(path)
        with torch.device("meta"):
            counts[path.stem] = count_parameters(build_backbone(cfg))
    assert 5_000_000 <= counts["a100_7m"] <= 10_000_000
    assert counts["a100_34m"] == 33_990_144
    assert 50_000_000 <= counts["a100_77m"] <= 100_000_000
    assert 300_000_000 <= counts["future_352m"] <= 500_000_000


def test_invalid_research_yaml_and_planning_guard():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"research":{"unrecognized":True}})
    with pytest.raises(ValueError):
        ResearchConfig(scales=[1, [4]])
    with pytest.raises(ValueError, match="planning_only"):
        Trainer(load_config("configs/profiles/future_352m.yaml"))
