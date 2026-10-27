"""Regression cases for the Kaggle campaign and masked objective contracts."""

import copy
import json

import pytest
import torch

from llm2.cli import main
from llm2.config import ResearchConfig
from llm2.interfaces import Objective, ObjectiveOutput
from llm2.model import TransformerLM
from llm2.research.accounting import estimate_research_flops
from llm2.research.objectives import PredictiveObjective
from llm2.research.profiling import profile_step
from llm2.runtime import capture_rng
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


def test_teacher_hidden_only_preserves_targets_and_corrects_cost(tiny_config):
    model = TransformerLM(tiny_config).eval()
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    expected = model(ids).hidden_states
    assert model(ids, return_logits=False).logits is None
    torch.testing.assert_close(expected, model(ids, return_logits=False).hidden_states, rtol=0, atol=0)
    r = ResearchConfig(jepa_weight=0.2)
    objective = PredictiveObjective(model, r).eval()
    calls = []
    hook = objective.target.lm_head.register_forward_hook(lambda *args: calls.append(1))
    batch = {"input_ids": ids, "targets": ids}
    hidden_result = objective(model, batch)
    hidden_cost = estimate_research_flops(model, objective, 8)
    assert not calls
    r.jepa_hidden_only = False
    full_result = objective(model, batch)
    full_cost = estimate_research_flops(model, objective, 8)
    hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(hidden_result.loss, full_result.loss, rtol=0, atol=0)
    assert full_cost.target_forward-hidden_cost.target_forward == 2*8*tiny_config.d_model*tiny_config.vocab_size


def test_jepa_answer_mask_selects_answer_and_eos_without_padding(tiny_config):
    targets = torch.tensor([[-100, -100, 7, 8, 31, -100, -100, -100]])
    assert PredictiveObjective.jepa_mask(targets, 1).nonzero()[:, 1].tolist() == [2, 3, 4]
    assert PredictiveObjective.jepa_mask(targets, 2).nonzero()[:, 1].tolist() == [2, 3]
    model = TransformerLM(tiny_config)
    objective = PredictiveObjective(model, ResearchConfig(jepa_weight=0.2, jepa_offset=2))
    result = objective(model, {"input_ids": torch.randint(0, 32, (1, 8)), "targets": targets})
    assert result.metrics["jepa_pairs"] == 2
    result.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    empty = objective(model, {"input_ids": torch.randint(0, 32, (1, 8)),
                              "targets": torch.full((1, 8), -100)})
    assert empty.metrics["jepa_pairs"] == 0
    assert empty.metrics["variance_penalty"] == 0
    assert torch.isfinite(empty.loss)


class FixedBatches:
    def __init__(self, batches):
        self.batches = iter(batches)

    def next(self, device):
        return {k: v.to(device) for k, v in next(self.batches).items()}


def test_token_accumulation_matches_union_with_unequal_masks(training_config, tmp_path):
    cfg = training_config
    cfg.train.accumulation_weighting = "tokens"
    cfg.model.dropout = 0
    cfg.data.batch_size = 1
    ids = torch.randint(0, 257, (2, 8), generator=torch.Generator().manual_seed(17))
    targets = ids.clone()
    targets[1, 1:] = -100
    batches = [{"input_ids": ids[i:i+1], "targets": targets[i:i+1]} for i in range(2)]
    with Trainer(cfg) as trainer:
        trainer.sampler = FixedBatches(batches)
        trainer.train_step()
        expected = copy.deepcopy(trainer.model.state_dict())
    cfg.data.batch_size = 2
    cfg.train.grad_accum_steps = 1
    cfg.train.output_dir = str(tmp_path / "union")
    with Trainer(cfg) as trainer:
        trainer.sampler = FixedBatches([{"input_ids": ids, "targets": targets}])
        trainer.train_step()
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=1e-4, atol=3e-7)


class SplitLoss(Objective):
    def forward(self, model, batch):
        marker = batch["input_ids"][0, 0].float()
        theta = model.embedding.weight[0, 0]
        lm = theta*marker
        auxiliary = theta.square()*(marker+2)
        return ObjectiveOutput(lm+auxiliary, int((batch["targets"] != -100).sum()),
                               language_loss=lm)


def test_token_weighting_does_not_rescale_auxiliary(training_config):
    cfg = training_config
    cfg.train.accumulation_weighting = "tokens"
    cfg.data.batch_size = 1
    targets = torch.ones(1, 8, dtype=torch.long)
    short = targets.clone()
    short[:, 1:] = -100
    with Trainer(cfg, objective=SplitLoss()) as trainer:
        theta = float(trainer.model.embedding.weight[0, 0].detach())
        trainer.sampler = FixedBatches([{"input_ids": targets, "targets": targets},
                                       {"input_ids": targets*3, "targets": short}])
        result = trainer.train_step()
        assert result["optimization_loss"] == pytest.approx(theta*(8+3)/9+theta**2*4, abs=1e-8)


@pytest.mark.parametrize("architecture", ["transformer", "recursive"])
def test_token_accumulation_exact_resume(training_config, tmp_path, architecture):
    cfg = training_config
    cfg.train.accumulation_weighting = "tokens"
    cfg.research = ResearchConfig(architecture=architecture, jepa_weight=0.1,
                                  gradient_checkpointing=architecture == "recursive",
                                  latent_steps=1, latent_warmup_steps=1)
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        expected_loss = trainer.train_step()["training_loss"]
        expected = copy.deepcopy((trainer.model.state_dict(), trainer.objective.state_dict(),
                                  trainer.optimizer.state_dict(), trainer.scheduler.state_dict(),
                                  trainer.sampler.state_dict(), capture_rng()))
    cfg.train.output_dir = str(tmp_path / "token-resume")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert trainer.train_step()["training_loss"] == expected_loss
        assert_nested_equal(expected, (trainer.model.state_dict(), trainer.objective.state_dict(),
                                       trainer.optimizer.state_dict(), trainer.scheduler.state_dict(),
                                       trainer.sampler.state_dict(), capture_rng()))


def test_cli_exports_full_window_losses_without_overwrite(training_config, tmp_path, capsys):
    with Trainer(training_config) as trainer:
        trainer.fit(until_step=1)
        checkpoint = trainer.last_checkpoint
    out, windows = tmp_path / "full.json", tmp_path / "windows.jsonl"
    main(["evaluate", "--checkpoint", str(checkpoint), "--output", str(out),
          "--windows-output", str(windows)])
    capsys.readouterr()
    result = json.loads(out.read_text())
    rows = [json.loads(line) for line in windows.read_text().splitlines()]
    assert result["validation_scope"] == "all_full_windows"
    assert result["checkpoint_model_sha256"]
    assert result["dataset_sha256"]
    assert len(rows) == result["validation_sequences"]
    assert sum(r["supervised_tokens"] for r in rows) == result["validation_tokens"]
    assert sum(r["nll_sum"] for r in rows)/result["validation_tokens"] == pytest.approx(result["validation_loss"], abs=1e-6)
    assert rows[0]["target_start_offset"] == 1
    assert rows[-1]["target_start_offset"] == (len(rows)-1)*training_config.data.seq_len+1
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main(["evaluate", "--checkpoint", str(checkpoint), "--output", str(out)])
    assert out.read_bytes() == before


def test_warm_profile_excludes_trace_from_timing(training_config, tmp_path):
    result = profile_step(training_config, tmp_path / "warm-profile", warmup_steps=1, measure_steps=2)
    assert result["timing"]["measured_steps"] == 2
    assert len(result["unprofiled_samples"]) == 2
    assert result["step_metrics"]["step_tokens"] > 0
    assert result["observed_attention_operators"]
    with pytest.raises(ValueError, match="max_steps"):
        profile_step(training_config, tmp_path / "too-long", warmup_steps=6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("architecture", ["transformer", "recursive"])
def test_cuda_fp32_checkpoint_and_continue(training_config, tmp_path, architecture):
    cfg = training_config
    cfg.train.device, cfg.train.precision, cfg.train.deterministic = "cuda", "fp32", False
    cfg.train.accumulation_weighting = "tokens"
    cfg.research = ResearchConfig(architecture=architecture, jepa_weight=0.1)
    with Trainer(cfg) as trainer:
        metrics = trainer.train_step()
        assert metrics["peak_vram_allocated_bytes"] > 0
        checkpoint = trainer.save()
    cfg.train.output_dir = str(tmp_path / "cuda-fp32-resume")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert torch.isfinite(torch.tensor(trainer.train_step()["training_loss"]))
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
