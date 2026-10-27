import copy
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
from llm2.objectives import CausalLanguageModeling
from llm2.research.objectives import PredictiveObjective
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


def test_worker_environment_and_explicit_configs(training_config, tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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


def test_failed_process_keeps_full_error_and_never_selects_old_checkpoint(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    campaign = Campaign(root, tmp_path / "campaign", device="cpu", gpus=[])
    old = tmp_path / "old-run/checkpoints/step-00002000"
    old.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="command exited 7"):
        campaign.command(["-c", "print('unfiltered traceback detail'); raise SystemExit(7)"], "failure.log")
    assert "Etapa concluída" not in capsys.readouterr().out
    assert "unfiltered traceback detail" in (campaign.output / "logs/failure.log").read_text(encoding="utf-8")
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
    manifest = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    job = manifest["runs"][0]
    assert job["status"] == "complete"
    evaluation = json.loads(Path(job["evaluation"]).read_text(encoding="utf-8"))
    assert evaluation["validation_scope"] == "all_full_windows"
    assert evaluation["dataset_sha256"] == manifest["dataset_sha256"]
    assert (output / "report/comparisons.csv").is_file()
    with zipfile.ZipFile(result["results_zip"]) as archive:
        assert "campaign.json" in archive.namelist()
        assert any(name.endswith("windows.jsonl") for name in archive.namelist())
        assert not any(name.endswith((".pt", ".safetensors", ".bin")) for name in archive.namelist())
    with pytest.raises(FileExistsError):
        run_campaign(args)


def test_gpu_slot_respects_outer_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one,GPU-two")
    assert worker_environment(Path.cwd(), "1")["CUDA_VISIBLE_DEVICES"] == "GPU-two"


@pytest.mark.parametrize(("visible", "gpu", "expected"), [
    ("0", "0", "0"), ("3,7", "1", "7"),
    (" GPU-one , GPU-two ", "0", "GPU-one"),
    ("MIG-GPU-one/1/0,MIG-GPU-two/1/0", "1", "MIG-GPU-two/1/0"),
])
def test_gpu_slot_maps_parent_mask(monkeypatch, visible, gpu, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert worker_environment(Path.cwd(), gpu)["CUDA_VISIBLE_DEVICES"] == expected


@pytest.mark.parametrize(("visible", "gpu"), [
    ("0", "1"), ("3,7", "2"), ("", "0"), ("-1", "0"), ("0,", "0"),
])
def test_gpu_slot_cannot_escape_parent_mask(monkeypatch, visible, gpu):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="unavailable under CUDA_VISIBLE_DEVICES"):
        worker_environment(Path.cwd(), gpu)


@pytest.mark.parametrize("gpu", ["-1", "invalid", "1.5"])
def test_gpu_slot_requires_nonnegative_logical_index(monkeypatch, gpu):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="nonnegative logical index"):
        worker_environment(Path.cwd(), gpu)


@pytest.mark.parametrize("visible", ["0", "", "-1"])
def test_worker_without_gpu_preserves_parent_mask(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert worker_environment(Path.cwd())["CUDA_VISIBLE_DEVICES"] == visible


def test_profiles_include_baseline_and_all_compressed_modes_once(training_config, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    campaign = Campaign(root, tmp_path / "profile-campaign", device="cpu", gpus=[])
    (campaign.output / "configs").mkdir()
    models = ["baseline_small", "flrm_compressed", "flrm_compressed_native", "flrm_compressed_grouped",
              "flrm_compressed_grouped_no_jepa"]
    jobs, commands = [], []
    for seed in [42, 43]:
        for model in models:
            cfg = campaign_config(training_config, model, training_config.data.path,
                tmp_path / f"{model}-{seed}", seed=seed, steps=2, device="cpu", max_seconds=0, iterations=2)
            if model.startswith("flrm"):
                assert cfg.research.scale_execution == model.removeprefix("flrm_").removesuffix("_no_jepa")
                assert cfg.research.jepa_weight == (0. if model.endswith("_no_jepa") else .1)
            path = campaign.output / "configs" / f"{model}-{seed}.yaml"
            path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
            jobs.append({"id": f"{model}-{seed}", "model": model, "config": str(path)})
    def command(arguments, log_name, *, gpu=None):
        commands.append((arguments, gpu))
        output = Path(arguments[arguments.index("--output")+1])
        output.mkdir(parents=True)
        (output / "profile.json").write_text(json.dumps({"timing": {"mean_tokens_per_second": 123.}}))
    monkeypatch.setattr(campaign, "command", command)
    campaign.profile_runs(jobs, 10, 30)
    assert len(commands) == len(models)
    assert all(gpu is None for _, gpu in commands)
    assert all(str(args[args.index("--warmup-steps")+1]) == "10" for args, _ in commands)
    assert all(str(args[args.index("--measure-steps")+1]) == "30" for args, _ in commands)
    assert all("profile" in job for job in jobs[:len(models)])
    assert all("profile" not in job for job in jobs[len(models):])
    assert parser().parse_args([]).models == ["flrm_compressed_grouped_no_jepa", "flrm_compressed_grouped", "baseline_small"]


def test_grouped_jepa_ablation_changes_only_objective_and_preserves_initial_student(training_config, tmp_path):
    configs = [campaign_config(training_config, model, training_config.data.path,
        tmp_path / model, seed=42, steps=2, device="cpu", max_seconds=0, iterations=2)
        for model in ("flrm_compressed_grouped_no_jepa", "flrm_compressed_grouped")]
    resolved = []
    for cfg in configs:
        raw = cfg.to_dict()
        raw["train"].pop("output_dir")
        raw["research"].pop("jepa_weight")
        resolved.append(raw)
        assert cfg.research.iterations+cfg.research.latent_steps == 2
        assert cfg.research.scale_execution == "compressed_grouped"
        assert cfg.research.mtp_weight == cfg.research.intermediate_weight == 0
    assert resolved[0] == resolved[1]
    with Trainer(configs[0]) as trainer:
        assert type(trainer.objective) is CausalLanguageModeling
        assert not hasattr(trainer.objective, "target") and not hasattr(trainer.objective, "predictor")
        assert trainer.total_parameters == trainer.model_parameters
        student = copy.deepcopy(trainer.model.state_dict())
        batch = trainer.sampler.next()
        backbone_count = trainer.model_parameters
    with Trainer(configs[1]) as trainer:
        assert type(trainer.objective) is PredictiveObjective
        assert hasattr(trainer.objective, "target") and hasattr(trainer.objective, "predictor")
        assert trainer.model_parameters == backbone_count < trainer.total_parameters
        assert_nested_equal(student, trainer.model.state_dict())
        assert_nested_equal(batch, trainer.sampler.next())


def test_legacy_no_jepa_identifiers_keep_original_architectures(training_config, tmp_path):
    for model, architecture in [("flrm_no_jepa", "fractal"), ("lrm_no_jepa", "recursive")]:
        cfg = campaign_config(training_config, model, training_config.data.path,
            tmp_path / model, seed=42, steps=2, device="cpu", max_seconds=0)
        assert cfg.research.architecture == architecture
        assert cfg.research.scale_execution == "reference" and cfg.research.jepa_weight == 0
