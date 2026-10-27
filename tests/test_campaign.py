import json
from pathlib import Path
import sys
import zipfile

import pytest
import torch
import yaml

from llm2.research.campaign import Campaign, campaign_config, export_campaign, parser, run_campaign, worker_environment
from llm2.research.diagnostics import diagnose_model, linear_cka
from llm2.model import TransformerLM
from llm2.runtime import capture_rng
from test_training import training_config, assert_nested_equal


def test_worker_environment_and_explicit_configs(training_config, tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = worker_environment(root, "1")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONPATH"] == str(root / "src")
    cfg = campaign_config(training_config, "flrm_batched", training_config.data.path,
                          tmp_path / "run", seed=17, steps=2, device="cpu", max_seconds=0)
    assert cfg.research.scale_execution == "batched"
    assert cfg.research.architecture == "fractal"
    assert cfg.train.accumulation_weighting == "tokens"
    assert cfg.train.seed == 17
    assert cfg.research.curriculum == "off"
    lrm = campaign_config(training_config, "lrm", training_config.data.path,
                         tmp_path / "lrm", seed=42, steps=2, device="cpu", max_seconds=0)
    assert lrm.research.iterations+lrm.research.latent_steps == 2
    assert lrm.research.jepa_weight == 0.1


def test_failed_process_keeps_full_error_and_never_selects_old_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    campaign = Campaign(root, tmp_path / "campaign", device="cpu", gpus=[])
    old = tmp_path / "old-run/checkpoints/step-00002000"
    old.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="command exited 7"):
        campaign.command(["-c", "print('unfiltered traceback detail'); raise SystemExit(7)"], "failure.log")
    assert "unfiltered traceback detail" in (campaign.output / "logs/failure.log").read_text()
    assert not campaign.manifest["runs"]
    archive = export_campaign(campaign.output)
    with zipfile.ZipFile(archive) as bundle:
        assert "logs/failure.log" in bundle.namelist()
    with pytest.raises(FileExistsError):
        Campaign(root, campaign.output, device="cpu", gpus=[])
    with pytest.raises(FileExistsError):
        export_campaign(campaign.output)


def test_cka_and_diagnostics_preserve_rng_modes(training_config):
    torch.manual_seed(42)
    value = torch.randn(12, 8)
    assert linear_cka(value, value*3+7) == pytest.approx(1)
    assert linear_cka(torch.zeros_like(value), value) is None
    model = TransformerLM(training_config.model).train()
    model.layers[0].eval()
    modes = [module.training for module in model.modules()]
    ids = torch.randint(0, 257, (1, 8))
    rng = capture_rng()
    result = diagnose_model(model, ids, max_samples=8)
    assert result["scales"][0]["samples"] == 8
    assert result["linear_cka"][0][0] == pytest.approx(1)
    assert modes == [module.training for module in model.modules()]
    assert_nested_equal(rng, capture_rng())


def test_tiny_campaign_end_to_end_in_separate_processes(training_config, tmp_path):
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump(training_config.to_dict()), encoding="utf-8")
    output = tmp_path / "fresh-campaign"
    args = parser().parse_args(["--execute", "--config", str(config),
        "--data", training_config.data.path, "--output", str(output), "--device", "cpu",
        "--models", "baseline", "--steps", "1", "--max-seconds", "0",
        "--no-benchmark", "--no-diagnostics"])
    result = run_campaign(args)
    assert result["status"] == "complete"
    manifest = json.loads((output / "campaign.json").read_text())
    assert manifest["status"] == "complete"
    job = manifest["runs"][0]
    assert job["status"] == "complete"
    evaluation = json.loads(Path(job["evaluation"]).read_text())
    assert evaluation["validation_scope"] == "all_full_windows"
    assert evaluation["dataset_sha256"] == manifest["dataset_sha256"]
    assert (output / "report/comparisons.csv").is_file()
    with zipfile.ZipFile(result["results_zip"]) as archive:
        assert "campaign.json" in archive.namelist()
        assert any(name.endswith("windows.jsonl") for name in archive.namelist())
        assert not any(name.endswith((".pt", ".safetensors", ".bin")) for name in archive.namelist())
    with pytest.raises(FileExistsError):
        run_campaign(args)
