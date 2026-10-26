import copy
import json
import random

import numpy as np
import pytest
import torch

from llm2.config import (DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, TrainConfig)
from llm2.data import prepare_data
from llm2.metrics import estimate_transformer_flops
from llm2.interfaces import Backbone, ModelOutput, Objective, ObjectiveOutput
from llm2.model import TransformerLM
from llm2.objectives import CausalLanguageModeling
from llm2.runtime import capture_rng
from llm2.trainer import Trainer


@pytest.fixture
def training_config(tmp_path):
    corpus = tmp_path / "raw.txt"
    corpus.write_text("\n\n".join(f"Research paragraph {i}, repeated patterns and controlled data. " * 5
                                for i in range(20)), encoding="utf-8")
    prepare_data([corpus], tmp_path / "data", kind="byte")
    return ExperimentConfig(
        model=ModelConfig(vocab_size=257, d_model=32, n_layers=2, n_heads=4,
                          hidden_dim=64, max_seq_len=16, dropout=0.2),
        data=DataConfig(path=str(tmp_path / "data"), seq_len=8, batch_size=2, eval_batch_size=2),
        optimizer=OptimizerConfig(lr=1e-3, warmup_steps=2, schedule_steps=12),
        train=TrainConfig(output_dir=str(tmp_path / "run"), max_steps=6, grad_accum_steps=2,
                          eval_interval=2, eval_batches=2, checkpoint_interval=2, log_interval=1),
    )


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_memorize_minibatch(tiny_config):
    model = TransformerLM(tiny_config)
    objective = CausalLanguageModeling()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    ids = torch.arange(16).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    initial = objective(model, batch).loss.item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    final = objective(model, batch).loss.item()
    assert final < 0.05 and final < initial / 20


def test_exact_resume_next_step_and_full_state(training_config, tmp_path):
    cfg = training_config
    with Trainer(cfg) as uninterrupted:
        uninterrupted.fit(until_step=2)
        checkpoint = uninterrupted.last_checkpoint
        before = {"model": copy.deepcopy(uninterrupted.model.state_dict()),
                  "optimizer": copy.deepcopy(uninterrupted.optimizer.state_dict()),
                  "scheduler": copy.deepcopy(uninterrupted.scheduler.state_dict()),
                  "sampler": copy.deepcopy(uninterrupted.sampler.state_dict()),
                  "rng": capture_rng(), "counters": uninterrupted.counters.copy()}
        expected_metrics = uninterrupted.train_step()
        expected = {"model": copy.deepcopy(uninterrupted.model.state_dict()),
                    "optimizer": copy.deepcopy(uninterrupted.optimizer.state_dict()),
                    "scheduler": copy.deepcopy(uninterrupted.scheduler.state_dict()),
                    "sampler": copy.deepcopy(uninterrupted.sampler.state_dict()), "rng": capture_rng()}
    # Perturb all global RNGs; constructor initialization must not change restored RNG state.
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    cfg.train.output_dir = str(tmp_path / "resumed")
    with Trainer(cfg, resume=checkpoint) as resumed:
        actual_before = {"model": resumed.model.state_dict(), "optimizer": resumed.optimizer.state_dict(),
                         "scheduler": resumed.scheduler.state_dict(), "sampler": resumed.sampler.state_dict(),
                         "rng": capture_rng(), "counters": resumed.counters.copy()}
        assert_nested_equal(before, actual_before)
        metrics = resumed.train_step()
        actual = {"model": resumed.model.state_dict(), "optimizer": resumed.optimizer.state_dict(),
                  "scheduler": resumed.scheduler.state_dict(), "sampler": resumed.sampler.state_dict(),
                  "rng": capture_rng()}
        assert_nested_equal(expected, actual)
        for name in ("training_loss", "grad_norm", "lr", "tokens_processed", "training_flops_estimate"):
            assert metrics[name] == expected_metrics[name]
        assert resumed.model.embedding.weight is resumed.model.lm_head.weight


def test_validation_preserves_rng_modes_sampler(training_config):
    with Trainer(training_config) as trainer:
        trainer.model.train()
        trainer.model.layers[0].eval()
        trainer.train_step()
        trainer.model.layers[0].eval()
        modes = [m.training for m in trainer.model.modules()]
        rng, sampler = capture_rng(), copy.deepcopy(trainer.sampler.state_dict())
        a, b = trainer.validate(2), trainer.validate(2)
        assert a == b
        assert modes == [m.training for m in trainer.model.modules()]
        assert_nested_equal(rng, capture_rng())
        assert_nested_equal(sampler, trainer.sampler.state_dict())


def test_metrics_and_estimated_flops(training_config):
    config = training_config
    expected = estimate_transformer_flops(config.model, config.data.seq_len)
    with Trainer(config) as trainer:
        trainer.fit(until_step=2)
        assert trainer.counters["tokens_processed"] == 2 * 2 * 2 * 8
        assert trainer.counters["training_flops_estimate"] == 2 * 2 * 2 * expected.training_per_sequence
        assert trainer.counters["gpu_time_seconds"] is None
        assert trainer.counters["peak_vram_allocated_bytes"] is None
    path = trainer.output_dir
    metadata = json.loads((path / "run.json").read_text())
    assert metadata["flops_estimate"]["is_estimate"] is True
    assert metadata["model_parameters_calculated"] == sum(p.numel() for p in trainer.model.parameters())
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["validation_loss"] > 0
    assert rows[-1]["tokens_per_second"] > 0
    assert list((path / "tensorboard").glob("events.*"))


def test_flop_formula():
    c = ModelConfig(vocab_size=16, d_model=8, n_heads=2, n_layers=1, hidden_dim=16)
    s = 4
    # Explicit matrix multiplications: qkv + output + 3 FFN + QK/AV + head.
    forward = 2*s*8*(3*8) + 2*s*8*8 + 3*2*s*8*16 + 2*2*s*s*8 + 2*s*8*16
    estimate = estimate_transformer_flops(c, s)
    assert estimate.forward_per_sequence == forward
    assert estimate.training_per_sequence == 3 * forward
    assert estimate.is_estimate


def test_incompatible_resume_and_checkpoint_corruption(training_config, tmp_path):
    with Trainer(training_config) as trainer:
        trainer.fit(until_step=2)
        path = trainer.last_checkpoint
    changed = copy.deepcopy(training_config)
    changed.train.output_dir = str(tmp_path / "other")
    changed.optimizer.lr *= 2
    with pytest.raises(ValueError, match="configuration"):
        Trainer(changed, resume=path)
    with (path / "training_state.pt").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        Trainer(training_config, resume=path)


def test_dataset_change_rejected(training_config, tmp_path):
    with Trainer(training_config) as trainer:
        trainer.fit(until_step=1)
        path = trainer.last_checkpoint
    corpus = tmp_path / "new.txt"
    corpus.write_text("A different first document. " * 30 + "\n\n" + "A second different document. " * 30)
    prepare_data([corpus], tmp_path / "other-data", kind="byte")
    training_config.data.path = str(tmp_path / "other-data")
    with pytest.raises(ValueError, match="dataset fingerprint"):
        Trainer(training_config, resume=path)


def test_environment_change_rejected(training_config):
    with Trainer(training_config) as trainer:
        trainer.fit(until_step=1)
        path = trainer.last_checkpoint
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["environment"]["packages"]["torch"] = "different-version"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="environment changed"):
        Trainer(training_config, resume=path)


def test_failed_step_cannot_be_checkpointed(training_config):
    with Trainer(training_config) as trainer:
        with torch.no_grad():
            trainer.model.embedding.weight.fill_(float("nan"))
        with pytest.raises(FloatingPointError):
            trainer.train_step()
        assert trainer.counters["step"] == 0
        with pytest.raises(RuntimeError, match="boundaries"):
            trainer.save()


class OtherBackbone(Backbone):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(257, 8)
        self.projection = torch.nn.Linear(8, 257)

    def forward(self, input_ids, *, state=None):
        return ModelOutput(logits=self.projection(self.embedding(input_ids)))


class ParameterizedObjective(Objective):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("calls", torch.tensor(0))
        self.base = CausalLanguageModeling()

    def forward(self, model, batch):
        if self.training:
            self.calls.add_(1)
        result = self.base(model, batch)
        return ObjectiveOutput(loss=result.loss * self.scale.square(), token_count=result.token_count)


def test_backbone_independence_and_objective_state(training_config, tmp_path):
    with Trainer(training_config, model_factory=OtherBackbone, objective_factory=ParameterizedObjective) as trainer:
        metrics = trainer.train_step()
        assert metrics["training_flops_estimate"] is None  # Never apply a Transformer formula to another model.
        assert trainer.model_parameters == 257 * 8 + 8 * 257 + 257
        assert trainer.total_parameters == trainer.model_parameters + 1
        checkpoint = trainer.save()
        objective_state = copy.deepcopy(trainer.objective.state_dict())
        trainer.train_step()
        expected_objective = copy.deepcopy(trainer.objective.state_dict())
        expected_model = copy.deepcopy(trainer.model.state_dict())
    training_config.train.output_dir = str(tmp_path / "custom-resume")
    with Trainer(training_config, model=OtherBackbone(), objective=ParameterizedObjective(), resume=checkpoint) as resumed:
        assert_nested_equal(objective_state, resumed.objective.state_dict())
        resumed.train_step()
        assert_nested_equal(expected_model, resumed.model.state_dict())
        assert_nested_equal(expected_objective, resumed.objective.state_dict())


def test_gradient_accumulation_equivalent_batch(training_config, tmp_path):
    training_config.model.dropout = 0.0
    with Trainer(training_config) as accumulated:
        accumulated.train_step()
        expected = copy.deepcopy(accumulated.model.state_dict())
    training_config.data.batch_size *= training_config.train.grad_accum_steps
    training_config.train.grad_accum_steps = 1
    training_config.train.output_dir = str(tmp_path / "large-batch")
    with Trainer(training_config) as large_batch:
        large_batch.train_step()
        for name, value in large_batch.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=1e-4, atol=2e-7)


def test_log_rewind_preserves_abandoned_attempt(training_config):
    with Trainer(training_config) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
    metrics_file = trainer.output_dir / "metrics.jsonl"
    with metrics_file.open("a") as stream:
        stream.write('{"step": 3, "training_loss": 123}\n')
        stream.write('{"step":')
    with Trainer(training_config, resume=checkpoint):
        active = [json.loads(line) for line in metrics_file.read_text().splitlines()]
        assert all(row["step"] <= 2 for row in active)
        assert list(trainer.output_dir.glob("metrics-before-resume-*.jsonl"))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16(training_config):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    training_config.train.device = "cuda"
    training_config.train.precision = "bf16"
    training_config.train.deterministic = False
    with Trainer(training_config) as trainer:
        result = trainer.train_step()
        assert math_isfinite(result["training_loss"])
        assert result["gpu_time_seconds"] > 0
        assert result["peak_vram_allocated_bytes"] > 0
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def math_isfinite(value):
    import math
    return math.isfinite(value)


@pytest.mark.compile
def test_compile_eager_contract(training_config):
    # Exercises Dynamo graph capture portably; Inductor/CUDA needs hardware validation.
    training_config.train.compile = True
    training_config.train.compile_backend = "eager"
    with Trainer(training_config) as trainer:
        assert math_isfinite(trainer.train_step()["training_loss"])

