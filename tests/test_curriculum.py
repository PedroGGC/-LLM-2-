import copy
from dataclasses import replace

import pytest
import torch

from llm2.config import ResearchConfig
from llm2.curriculum.skills import Skill, SkillRegistry, default_registry
from llm2.curriculum.generators import generate_problem, problem_set, solve, verify_v1
from llm2.curriculum.controller import CurriculumController
from llm2.curriculum.stream import MixedCurriculumStream, pack_problem
from llm2.data import RandomBatchStream, TextTokenizer
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


def test_dependency_graph_validates_missing_and_cycles():
    registry = default_registry()
    assert len(registry.order) == 9
    for key, skill in registry.skills.items():
        assert all(registry.order.index(dependency) < registry.order.index(key) for dependency in skill.dependencies)
    for skills in [[Skill("a", ("missing",))], [Skill("a", ("b",)), Skill("b", ("a",))], [Skill("a"), Skill("a")]]:
        with pytest.raises(ValueError):
            SkillRegistry(skills)


def test_procedural_splits_reproducible_and_verified():
    registry = default_registry()
    sets = {split: problem_set(registry, split, seed=7, examples=3, difficulties=(1, 2, 3))
            for split in ("train", "probe", "val", "test")}
    for split, problems in sets.items():
        assert [p.to_dict() for p in problems] == [p.to_dict() for p in problem_set(registry, split, seed=7, examples=3, difficulties=(1, 2, 3))]
        for problem in problems:
            assert verify_v1(problem, problem.answer)
            assert not verify_v1(problem, "incorrect")
            # The verifier never just compares with the potentially corrupted answer field.
            assert verify_v1(replace(problem, answer="corrupted"), problem.answer)
    for a in sets:
        for b in sets:
            if a != b:
                assert not {p.identifier for p in sets[a]} & {p.identifier for p in sets[b]}
    assert generate_problem("composition", 1, 1, "test").family == "B"


def test_controller_feedback_exploration_difficulty_and_resume():
    registry = default_registry()
    controller = CurriculumController(registry, max_difficulty=3, seed=19, exploration=0.2)
    before = controller.distribution()
    for _ in range(4):
        controller.update("arithmetic", 1, 0.95, split="probe")
        controller.update("arithmetic", 2, 0.05, split="probe")
    after = controller.distribution()
    assert before != after
    assert after[("arithmetic", 2)] > after[("arithmetic", 1)]
    assert sum(after.values()) == pytest.approx(1)
    assert min(after.values()) >= 0.2/len(after)
    with pytest.raises(ValueError, match="dedicated probe"):
        controller.update("arithmetic", 1, 1.0, split="test")
    state = controller.state_dict()
    expected = [controller.sample() for _ in range(20)]
    other = CurriculumController(registry, max_difficulty=3, seed=999, exploration=0.2)
    other.load_state_dict(state)
    assert [other.sample() for _ in range(20)] == expected
    assert_nested_equal(controller.state_dict(), other.state_dict())
    fixed = CurriculumController(registry, policy="fixed", fixed_steps=2)
    for _ in range(5):
        fixed.sample()
    assert fixed.distribution()[("arithmetic", 3)] > fixed.distribution()[("arithmetic", 1)]


def test_complete_problem_answer_masks_and_oversize():
    problem = generate_problem("arithmetic", 1, 12)
    ids, targets = pack_problem(problem, TextTokenizer(), 128)
    visible = targets[targets != -100].tolist()
    assert TextTokenizer().decode(visible) == problem.answer
    assert ids.shape == targets.shape == (128,)
    with pytest.raises(ValueError, match="exceeds context"):
        pack_problem(problem, TextTokenizer(), 4)


@pytest.mark.parametrize("architecture,policy", [("transformer", "uniform"), ("transformer", "adaptive"), ("recursive", "adaptive")])
def test_mixed_curriculum_exact_resume(training_config, tmp_path, architecture, policy):
    cfg = training_config
    cfg.data.seq_len = cfg.model.max_seq_len = 128
    cfg.research = ResearchConfig(architecture=architecture, curriculum=policy, max_difficulty=1,
                                  curriculum_interval=1, probe_examples=1, text_fraction=0.5)
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        path = trainer.last_checkpoint
        expected_metrics = trainer.train_step()
        expected = copy.deepcopy((trainer.model.state_dict(), trainer.optimizer.state_dict(), trainer.sampler.state_dict()))
        assert trainer.counters["curriculum_probe_flops_estimate"] > 0
        assert trainer.sampler.controller.history
        assert "effective_text_example_fraction" in expected_metrics
    cfg.train.output_dir = str(tmp_path / "resumed")
    with Trainer(cfg, resume=path) as resumed:
        metrics = resumed.train_step()
        assert metrics["training_loss"] == expected_metrics["training_loss"]
        assert metrics["training_flops_estimate"] == expected_metrics["training_flops_estimate"]
        assert_nested_equal(expected, (resumed.model.state_dict(), resumed.optimizer.state_dict(), resumed.sampler.state_dict()))


@pytest.mark.parametrize("limit,expected", [("tokens", "token_budget"), ("flops", "learning_flop_budget"), ("time", "time_budget")])
def test_budget_stops_at_boundary_and_can_resume(training_config, tmp_path, limit, expected):
    cfg = training_config
    cfg.research = ResearchConfig()
    if limit == "tokens":
        cfg.research.max_tokens = cfg.data.batch_size*cfg.data.seq_len*cfg.train.grad_accum_steps
    elif limit == "flops":
        cfg.research.max_train_flops = 1
    else:
        cfg.research.max_seconds = 0.000001
        cfg.research.shutdown_reserve_seconds = 0
    with Trainer(cfg) as trainer:
        trainer.fit()
        assert trainer.stop_reason == expected
        assert trainer.counters["step"] == (1 if limit == "tokens" else 0)
        checkpoint = trainer.last_checkpoint
        assert checkpoint.is_dir()
    cfg.train.output_dir = str(tmp_path / "extended-budget")
    cfg.research.max_seconds = 0
    cfg.research.max_tokens = cfg.research.max_train_flops = 0
    with Trainer(cfg, resume=checkpoint) as resumed:
        resumed.fit(until_step=2)
        assert resumed.counters["step"] == 2
