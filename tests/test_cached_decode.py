import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile

import pytest
import torch
from safetensors.torch import save_model

from llm2.research import decode_benchmark as bench
from llm2.research.context_protocol import configuration
from llm2.research.decode_cache import RecurrentDecode, TransformerDecode, generate, verify_logits
from llm2.research.factory import build_backbone
from llm2.research.stories_worker import score
from llm2.runtime import attention_context
from test_stories_data import fixture_data


@pytest.mark.parametrize("variant,adapter", [
    ("transformer_medium", TransformerDecode), ("context_untied2", RecurrentDecode),
])
def test_cache_equivalence_positions_and_reset(tmp_path, variant, adapter):
    cfg = configuration(variant, tmp_path, tmp_path, 341, device="cpu")
    model = build_backbone(cfg).eval()
    decoder = adapter(model)
    tokens = torch.randint(0, 256, (2, 101))
    with torch.inference_mode(), attention_context(True):
        for length in (1, 31, 64, 77):
            report = verify_logits(decoder, tokens[:, :length], tokens[:, length:])
            assert report["max_abs_logit_error"] < 1e-5
        prefix = tokens[:1, :77]
        first, generated = generate(decoder, prefix, 8)
        second, repeated = generate(decoder, prefix, 8)
        assert torch.equal(generated, repeated) and generated.shape == (1, 8)
        assert first["state_bytes"] == second["state_bytes"]
        if adapter is TransformerDecode:
            assert first["state_bytes"] == 2 * 2 * 84 * 80 * 4
            _, full = decoder.prefill(prefix, capacity=77)
            with pytest.raises(ValueError, match="available capacity"):
                decoder.step(prefix[:, :1], full)
        else:
            assert first["state_bytes"] == 12480
            assert all(cell.execution == "chunked" for cell in model.cells)


@pytest.mark.parametrize("variant", bench.VARIANTS)
def test_frozen_safetensors_to_benchmark_report(tmp_path, variant, monkeypatch):
    data = fixture_data(tmp_path / "data")
    cfg = configuration(variant, data.root, tmp_path, 341, device="cpu")
    model = build_backbone(cfg).eval()
    archive_path = tmp_path / "checkpoints.zip"
    folder = "runs/job/checkpoints/step-00006000"
    weight_path = tmp_path / "model.safetensors"
    save_model(model, weight_path)
    weight = weight_path.read_bytes()
    metadata = b'{"schema_version":1}'
    job = {
        "config": cfg.to_dict(), "folder": folder, "architecture_version": 1,
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "weight_sha256": hashlib.sha256(weight).hexdigest(),
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(folder + "/metadata.json", metadata)
        archive.writestr(folder + "/model.safetensors", weight)
        archive.writestr(folder + "/training_state.pt", b"untrusted pickle payload")
    original_open = ZipFile.open

    def only_inference(archive, name, *args, **kwargs):
        assert not name.endswith(".pt")
        return original_open(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", only_inference)
    device = torch.device("cpu")
    restored = bench.load_frozen_model(archive_path, job, device)
    with torch.inference_mode(), attention_context(True):
        original = score(model, data, "val", 256, 32, device)["nll"]
        result = {"variant": variant, "seed": 341, "checkpoint_nll_t256": original, "profiles": []}
        path = tmp_path / "worker.json"
        bench.evaluate(restored, data, device, path, result, lengths=(33, 96), new_bytes=8)
    value = json.loads(path.read_text())
    assert value["status"] == "complete" and value["nll_difference_from_checkpoint"] == 0
    for profile in value["profiles"]:
        assert profile["parity"]["positions"] == 9 and len(profile["samples"]) == 7
        assert profile["peak_vram_bytes"] is None and profile["energy"]["board_joules"] is None
    with pytest.raises(ValueError, match="SHA-256 dos pesos"):
        bench.load_frozen_model(archive_path, {**job, "weight_sha256": "0" * 64}, device)


def test_missing_weights_publish_failure_without_starting_workers(tmp_path, monkeypatch):
    def missing(*args):
        raise FileNotFoundError("pesos ausentes")
    monkeypatch.setattr(bench, "locate_weights", missing)
    meta = tmp_path / "bootstrap.json"
    meta.write_text("{}")
    output = tmp_path / "llm2-campaign-0034"
    args = argparse.Namespace(seconds=900, started_at=time.monotonic(), gpus=["0", "1"],
                              output=str(output), operator_metadata=meta, weights=None,
                              rebuild=False, reference="unused")
    ref = {"campaign": "llm2-campaign-0033", "dataset_sha256": "fixture", "jobs": {}}
    with pytest.raises(RuntimeError, match="Consulte o ZIP"):
        bench.run_campaign(args, ref)
    with ZipFile(tmp_path / "llm2-campaign-0034-results.zip") as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("campaign.json"))
        assert manifest["status"] == "failed" and manifest["runs"] == []
        assert manifest["optimizer_steps"] == 0
        assert not json.loads(archive.read("comparison.json"))["twofold_long_generation"]


def test_locate_uploaded_archive_and_reject_ambiguity(tmp_path):
    work, inputs = tmp_path / "working", tmp_path / "input"
    work.mkdir()
    target = inputs / "first" / "llm2-campaign-0033-checkpoints.zip"
    target.parent.mkdir(parents=True)
    target.touch()
    assert bench.locate_weights(None, work, inputs) == target
    second = inputs / "second" / target.name
    second.parent.mkdir()
    second.touch()
    with pytest.raises(ValueError, match="Mais de um ZIP"):
        bench.locate_weights(None, work, inputs)
    assert bench.locate_weights(str(second), work, inputs) == second
