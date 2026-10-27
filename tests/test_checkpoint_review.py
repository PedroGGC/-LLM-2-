import json
from pathlib import Path
import shutil

import pytest

from llm2.data import PreparedData, sha256_file
from llm2.research.campaign import Campaign, parser
from llm2.research.checkpoint_review import review_checkpoints, verified_source
from llm2.research.presentation import progress, comparison_text
from llm2.trainer import Trainer
from test_training import training_config


@pytest.fixture
def saved(training_config, tmp_path):
    root = Path(__file__).resolve().parents[1]
    original = tmp_path / "original"
    campaign = Campaign(root, original, device="cpu", gpus=[])
    run_id = "baseline-seed-42"
    training_config.train.seed = 42
    training_config.train.output_dir = str(original / "runs" / run_id)
    with Trainer(training_config) as trainer:
        trainer.train_step(); trainer.save()
        trainer.train_step(); trainer.save()
    campaign.manifest.update(dataset_sha256=PreparedData(training_config.data.path).fingerprint,
        runs=[{"id":run_id, "model":"baseline", "seed":42}], status="complete")
    campaign.save()
    args = parser().parse_args(["--execute", "--review-campaign", str(original), "--evaluation-root", str(root),
        "--data", training_config.data.path, "--device", "cpu", "--review-steps", "1", "2",
        "--output", str(tmp_path / "review")])
    return original, args


def test_review_real_evaluations_then_verified_reuse_without_training(saved, tmp_path, monkeypatch, capsys):
    original, args = saved
    checkpoints = list(original.rglob("*.safetensors")) + list(original.rglob("training_state.pt"))
    before = {str(p):sha256_file(p) for p in checkpoints}
    monkeypatch.setattr(Campaign, "train_worker", lambda *a, **kw: pytest.fail("review must never train"))
    result = review_checkpoints(args)
    assert result["status"] == "complete"
    output = Path(result["campaign"])
    manifest = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 2
    assert not any(r["reused"] for r in manifest["runs"])
    assert all(r["validation_tokens"] > 0 for r in manifest["runs"])
    assert "menor = melhor" in (output / "resumo-pt.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        review_checkpoints(args)
    for row in manifest["runs"]:
        destination = original / "runs/baseline-seed-42/evaluations" / f"step-{row['step']:08d}"
        shutil.copytree(Path(row["evaluation"]).parent, destination)
    args.output = str(tmp_path / "review-reused")
    monkeypatch.setattr(Campaign, "command", lambda *a, **kw: pytest.fail("verified cache must avoid subprocesses"))
    second = review_checkpoints(args)
    second_manifest = json.loads((Path(second["campaign"]) / "campaign.json").read_text(encoding="utf-8"))
    assert all(row["reused"] for row in second_manifest["runs"])
    assert before == {str(p):sha256_file(p) for p in checkpoints}
    old = original / "runs/baseline-seed-42/evaluations/step-00000001/language.json"
    corrupted = json.loads(old.read_text(encoding="utf-8"))
    corrupted["checkpoint_model_sha256"] = "wrong"
    old.write_text(json.dumps(corrupted), encoding="utf-8")
    args.output = str(tmp_path / "review-stale")
    with pytest.raises(RuntimeError, match="interrompida"):
        review_checkpoints(args)
    failed = json.loads((Path(args.output) / "campaign.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and not failed["runs"]
    assert Path(args.output+"-results.zip").is_file()
    assert "PROCESSO CONCLUÍDO COM SUCESSO" not in capsys.readouterr().out


def test_review_missing_weights_and_wrong_data_are_explicit(saved, tmp_path):
    original, args = saved
    manifest_path = original / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["dataset_sha256"]
    manifest["dataset_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        review_checkpoints(args)
    manifest["dataset_sha256"] = expected
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    weight = original / "runs/baseline-seed-42/checkpoints/step-00000001/model.safetensors"
    weight.unlink()
    with pytest.raises(FileNotFoundError, match="ZIP de análise contém só metadados"):
        review_checkpoints(args)
    assert not Path(args.output).exists()


def test_review_rejects_source_difference(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text("original")
    (tmp_path / "pyproject.toml").write_text("original")
    expected = {p.relative_to(tmp_path).as_posix():sha256_file(p)
                for p in (tmp_path / "src/example.py", tmp_path / "pyproject.toml")}
    assert verified_source(tmp_path, expected) == expected
    (tmp_path / "src/example.py").write_text("changed")
    with pytest.raises(ValueError, match="código original"):
        verified_source(tmp_path, expected)


def test_console_distinguishes_partial_loss_and_missing_measurements():
    text = progress({"step":100, "training_loss":2., "tokens_per_second_mean":1234,
                     "validation_loss_partial":1.5},
                    {"model":"flrm_batched", "seed":42, "max_steps":200, "physical_gpu":"1"}, 60.)
    assert "GPU física 1" in text and "50%" in text and "PARCIAL" in text
    assert "maior = melhor" in text and "menor = melhor" in text
    table = comparison_text([{"model":"baseline", "seed":42, "validation_loss":1.2}])
    assert "n/d" in table and "Loss completa" in table
