import copy
import json
from pathlib import Path
import time
import zipfile

import pytest
import torch

from llm2.config import load_config, ResearchConfig
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research import cycle2
from llm2.research.cycle2_protocol import MODELS, SEEDS, configuration, configuration_stats, ordered_jobs, paired_fingerprints
from llm2.research.session import SessionBudget, SessionBudgetExceeded, SessionCampaign
from llm2.research.cycle2_report import write_report
from llm2.research.factory import build_backbone
from llm2.runtime import attention_context
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal
from test_compressed_optimization import _state

ROOT = Path(__file__).resolve().parents[1]
FACTORIAL = MODELS[2:]


def test_frozen_matrix_counts_pairing_and_schedule(tmp_path):
    base = load_config(ROOT / "configs/experiments/cycle2/base.yaml")
    configs = {name: configuration(base, name, tmp_path / "data", tmp_path / name, 81) for name in MODELS}
    assert len(ordered_jobs(SEEDS, ["0", "1"])) == 18
    for cfg in configs.values():
        assert cfg.train.max_steps == cfg.optimizer.schedule_steps == 2000
        assert cfg.data.batch_size * cfg.train.grad_accum_steps == 32
        assert cfg.train.compile and cfg.train.compile_backend == "inductor"
        assert cfg.research.context_normalization == "none"
        assert cfg.research.jepa_weight == cfg.research.mtp_weight == cfg.research.ncm_slots == 0
        assert not cfg.research.v_cycle and not cfg.research.inv_vcycle
        assert cfg.research.paired_common_init == (cfg.research.architecture != "transformer")
    assert [configuration_stats(configs[name])["registered_parameters"] for name in MODELS] == [
        2164224, 2017536, 2168076, 2165772, 2168076, 2165772]
    result = paired_fingerprints(configs)
    assert result["verified_common_equal"] and result["verified_non_film_exclusive_equal"]
    jobs = ordered_jobs(SEEDS, ["0", "1"])
    for name in MODELS:
        assert [j["gpu_slot"] for j in jobs if j["model"] == name][0] != [j["gpu_slot"] for j in jobs if j["model"] == name][1]
    assert [j["model"] for j in jobs[:6]] != [j["model"] for j in jobs[6:12]]
    base.optimizer.lr = .01
    with pytest.raises(ValueError, match="frozen"):
        configuration(base, MODELS[0], tmp_path / "data", tmp_path / "changed", 81)


@pytest.mark.parametrize("name", FACTORIAL)
@attention_context(True)
def test_factorial_causal_backward_observers_and_memorization(training_config, tmp_path, name):
    cfg = configuration(training_config, name, training_config.data.path, tmp_path / name, 81, device="cpu", smoke=True)
    cfg.model.dropout = 0.
    model = build_backbone(cfg)
    ids = torch.arange(18).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    output = model(batch["input_ids"])
    changed = batch["input_ids"].clone(); changed[:, 8:] += 3
    torch.testing.assert_close(model(changed).logits[:, :8], output.logits[:, :8], rtol=0, atol=0)
    if not cfg.research.cross_scale_exchange:
        assert len(output.state.scale_values) == 3
        assert all(not value.requires_grad for value in output.state.scale_values)
        assert [value.shape[1] for value in output.state.scale_values] == [17, 5, 2]
        def perturb_macro(module, inputs, result):
            return result + 100 if inputs[0].shape[1] < 17 else result
        hook = model.block.cores[0].register_forward_hook(perturb_macro)
        try:
            torch.testing.assert_close(model(batch["input_ids"]).logits, output.logits, rtol=0, atol=0)
        finally:
            hook.remove()
    objective = CausalLanguageModeling()
    initial = objective(model, batch).loss.item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert objective(model, batch).loss.item() < initial/15


@pytest.mark.parametrize("name", FACTORIAL)
def test_factorial_exact_resume_dropout_accumulation(training_config, tmp_path, name):
    cfg = configuration(training_config, name, training_config.data.path, tmp_path / "run", 81, device="cpu", smoke=True)
    assert cfg.model.dropout > 0 and cfg.train.grad_accum_steps == 2
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        before = _state(trainer); checkpoint = trainer.last_checkpoint
        expected_metrics = trainer.train_step(); expected = _state(trainer)
    cfg.train.output_dir = str(tmp_path / "resumed")
    with Trainer(cfg, resume=checkpoint) as resumed:
        assert_nested_equal(before, _state(resumed))
        actual = resumed.train_step()
        assert_nested_equal(expected, _state(resumed))
        assert actual["training_loss"] == expected_metrics["training_loss"]
        assert actual["parameters_with_gradients"] == expected_metrics["parameters_with_gradients"]
        if not cfg.research.cross_scale_exchange:
            assert actual["parameters_with_gradients"] < actual["model_parameters_calculated"]


@pytest.mark.compile
def test_compiled_model_keeps_macro_outputs_when_exchange_off(training_config, tmp_path):
    cfg = configuration(training_config, FACTORIAL[-1], training_config.data.path, tmp_path / "run", 81, device="cpu", smoke=True)
    cfg.model.dropout = 0.
    model = build_backbone(cfg)
    compiled = torch.compile(model, backend="aot_eager")
    ids = torch.randint(0, 257, (2, 17))
    reference, output = model(ids), compiled(ids)
    assert len(output.state.scale_values) == 3
    for a,b in zip(reference.state.scale_values, output.state.scale_values):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(reference.logits, output.logits,rtol=0,atol=0)
    output.logits.square().mean().backward()
    assert model.block.interaction.mixing.grad is None


def test_session_budget_counts_bootstrap_and_stops_before_reserve():
    now = [10.]
    budget = SessionBudget(100., 10., elapsed=20., clock=lambda: now[0])
    assert budget.work_remaining == 70.
    now[0] += 60.
    assert budget.work_remaining == 10.
    with pytest.raises(SessionBudgetExceeded): budget.require(10.)
    now[0] += 11.
    with pytest.raises(SessionBudgetExceeded): budget.require()


def test_watchdog_interrupts_blocking_subprocess(tmp_path):
    budget = SessionBudget(3., .5)
    campaign = SessionCampaign(ROOT, tmp_path / "session", device="cpu", gpus=[], budget=budget)
    started = time.monotonic()
    with pytest.raises(SessionBudgetExceeded):
        campaign.command(["-c", "import time; time.sleep(30)"], "blocked.log")
    assert time.monotonic()-started < 6
    assert not campaign.processes


def test_export_is_exclusive_and_excludes_weights(tmp_path):
    folder = tmp_path / "run";folder.mkdir()
    (folder / "campaign.json").write_text('{}')
    (folder / "training_state.pt").write_bytes(b"private state")
    campaign = type("Fake", (), {"output": folder})()
    archive = cycle2.export(campaign)
    original = archive.read_bytes()
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["campaign.json"]
        assert z.testzip() is None
    with pytest.raises(FileExistsError): cycle2.export(campaign)
    assert archive.read_bytes() == original


def test_plan_has_no_side_effects(tmp_path):
    output = tmp_path / "must-not-exist"
    args = cycle2.parser().parse_args(["--input", "missing.txt", "--output", str(output)])
    result = cycle2.plan(args)
    assert result["execution"] == "not_started" and len(result["jobs"]) == 18
    assert not output.exists()


def test_materialized_jobs_record_physical_parent_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-FIRST,GPU-SECOND")
    monkeypatch.setattr(cycle2, "paired_fingerprints", lambda configs: {"verified_for_this_test": True})
    campaign = SessionCampaign(ROOT, tmp_path / "mapping", device="cuda", gpus=["0", "1"],
                               budget=SessionBudget())
    args = cycle2.parser().parse_args(["--data", str(tmp_path / "data"), "--seeds", "81"])
    cycle2.materialize(campaign, args, tmp_path / "data")
    assert [j["physical_gpu"] for j in campaign.manifest["runs"]] == ["GPU-FIRST", "GPU-SECOND"]*3
    assert [j["gpu_slot"] for j in campaign.manifest["runs"]] == ["0", "1"]*3


def test_partial_campaign_never_claims_completeness(tmp_path, monkeypatch, training_config):
    args = cycle2.parser().parse_args(["--data", training_config.data.path, "--execute", "--device", "cpu",
        "--engineering-smoke", "--skip-tests", "--output", str(tmp_path / "partial"), "--seeds", "81"])
    def stop(*args, **kwargs): raise SessionBudgetExceeded("simulated budget boundary before pilot")
    monkeypatch.setattr(cycle2, "pilot", stop)
    result = cycle2.execute(args)
    assert result["status"] == "partial_budget"
    with zipfile.ZipFile(result["results_zip"]) as archive:
        report = json.loads(archive.read("report/report.json"))
        assert report["complete_seed_blocks"] == 0 and not report["confirmatory"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", FACTORIAL)
def test_cycle2_cuda_fp32_compiled_backward_checkpoint(training_config, tmp_path, name):
    cfg = configuration(training_config, name, training_config.data.path, tmp_path / "cuda", 81, device="cpu", smoke=True)
    cfg.train.device = "cuda"; cfg.train.deterministic = False
    cfg.train.compile = True; cfg.train.compile_backend = "inductor"
    with Trainer(cfg) as trainer:
        metrics = trainer.train_step()
        assert torch.isfinite(torch.tensor(metrics["training_loss"]))
        assert metrics["peak_vram_allocated_bytes"] > 0
        checkpoint = trainer.save()
    cfg.train.output_dir = str(tmp_path / "cuda-resume")
    with Trainer(cfg, resume=checkpoint) as resumed:
        assert torch.isfinite(torch.tensor(resumed.train_step()["training_loss"]))
