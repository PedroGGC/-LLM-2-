import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import json
import time
from zipfile import ZipFile

import pytest
import torch

from llm2.research import decode_benchmark as bench, decode_bootstrap as bootstrap
from llm2.research.context_protocol import configuration
from llm2.research.stories_data import write_dataset
from llm2.runtime import source_provenance


@pytest.mark.parametrize("interrupted", [False, True])
def test_rebuild_trains_then_decodes_and_archives_new_weights(tmp_path, monkeypatch, interrupted):
    splits = {}
    for split in ("train", "cal_val", "val", "test"):
        raw = (split + " A little fox found a red ball.\n").encode() * 10
        splits[split] = {hashlib.sha256(raw).hexdigest(): raw}
    data = write_dataset(tmp_path / "data", splits, spec={"engineering_smoke": True})
    provenance = source_provenance()
    ref = {"campaign": "llm2-campaign-0033", "revision": provenance["git"]["commit"],
           "dataset_sha256": data.fingerprint, "source_sha256": provenance["source_sha256"], "jobs": {}}
    for variant in bench.VARIANTS:
        for seed in bench.SEEDS:
            job = f"{variant}-seed-{seed}"
            cfg = configuration(variant, "previous_data", "previous_output", seed, lr=.001, device="cpu", smoke=True)
            cfg.model.max_seq_len = 512
            cfg.data.seq_len = 256
            ref["jobs"][job] = {"variant": variant, "seed": seed, "config": cfg.to_dict(),
                                "architecture_version": 1, "nll_t256": 1.2, "weight_sha256": "a" * 64}
    original = tmp_path / "reference.json"
    original.write_text(json.dumps(ref))
    meta = tmp_path / "bootstrap.json"
    meta.write_text("{}")
    output = tmp_path / "llm2-campaign-0035"
    args = argparse.Namespace(seconds=900, started_at=time.monotonic(), gpus=["0", "1"],
                              output=str(output), operator_metadata=meta, rebuild=True,
                              reference=original, cache=tmp_path / "cache")
    monkeypatch.setattr(bench, "prepare", lambda *args, **kwargs: data)

    def local_pair(campaign, gpus, commands, *, training=None):
        for job in ref["jobs"]:
            command = commands(job)
            if training is not None:
                assert "--train-job" in command
                bench.rebuild_job(campaign.output, job, ref, data.root, smoke=True)
                if interrupted:
                    raise bench.SessionBudgetExceeded("test deadline")
            else:
                assert "--job" in command
                reference = bench.read_json(command[command.index("--reference") + 1])
                assert reference["rebuild_of"] == ref["campaign"]
                assert reference["campaign"] == output.name
                selected = reference["jobs"][job]
                assert selected["weight_sha256"] != ref["jobs"][job]["weight_sha256"]
                assert selected["original_weight_sha256"] == ref["jobs"][job]["weight_sha256"]
                device = torch.device("cpu")
                model = bench.load_frozen_model(campaign.output, selected, device)
                result = {"variant": selected["variant"], "seed": selected["seed"],
                          "status": "running", "profiles": [], "checkpoint_nll_t256": selected["nll_t256"]}
                bench.evaluate(model, data, device, campaign.output / (job + ".json"),
                               result, lengths=(31,), new_bytes=8)

    monkeypatch.setattr(bench, "paired_jobs", local_pair)
    bench.run_campaign(args, ref)
    with ZipFile(tmp_path / (output.name + "-results.zip")) as results, \
         ZipFile(tmp_path / (output.name + "-checkpoints.zip")) as weights:
        assert results.testzip() is None and weights.testzip() is None
        assert not any(name.endswith((".pt", ".safetensors")) for name in results.namelist())
        campaign = json.loads(results.read("campaign.json"))
        assert campaign["status"] == ("partial_budget" if interrupted else "complete")
        assert campaign["optimizer_steps"] == (4 if interrupted else 16)
        assert sum(name.endswith("model.safetensors") for name in weights.namelist()) == (1 if interrupted else 4)
        if not interrupted:
            reference = json.loads(weights.read("decode-reference.json"))
            assert all(bench.read_weight(Path(weights.filename), job) for job in reference["jobs"].values())
            assert len(campaign["runs"]) == 4
            assert all(row["status"] == "complete" for row in campaign["runs"])
        else:
            assert campaign["runs"] == []


def test_rebuild_bootstrap_dispatches_and_downloads_both_archives(tmp_path, monkeypatch):
    source_files = {"src/llm2/model.py": "hash"}
    operator_meta = {"source_files_sha256": {
        name: "hash" for name in (bootstrap.BENCHMARK, bootstrap.REFERENCE,
                                 "src/llm2/research/decode_cache.py", "src/llm2/research/decode_bootstrap.py")}}
    revision = "a" * 40
    roots = {}
    for label, rev, metadata in (("operator", revision, operator_meta),
                                 ("frozen", bootstrap.SCIENTIFIC_REVISION, {"source_files_sha256": source_files})):
        parent = tmp_path / label
        root = parent / "source"
        root.mkdir(parents=True)
        roots[rev] = (root, parent, metadata)
    reference = roots[revision][0] / bootstrap.REFERENCE
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"revision": bootstrap.SCIENTIFIC_REVISION, "source_sha256": source_files}))
    output = tmp_path / "llm2-campaign-0035"
    received, cleaned, downloads = [], [], []

    def bounded(command, **kwargs):
        if "--rebuild" in command:
            received.extend(command)
            output.mkdir()
            (output / "campaign.json").write_text(json.dumps({"status": "complete", "optimizer_steps": 24000}))
            for suffix in ("results", "checkpoints"):
                (tmp_path / f"{output.name}-{suffix}.zip").touch()

    helper = {
        "_BootstrapSession": lambda seconds, started: SimpleNamespace(remaining=lambda **kw: seconds),
        "_hardware": lambda: (tmp_path, ["0", "1"], "fixture"),
        "checkout_git_source": lambda rev, *a, **kw: roots[rev],
        "bounded_command": bounded,
    }
    guard = {"active_workers": lambda: [], "remove_checkout": lambda base, path: cleaned.append(path)}
    monkeypatch.setattr(bootstrap, "fetch", lambda name, digest: helper if name == "cycle2_bootstrap" else guard)
    monkeypatch.setattr(bootstrap.importlib.metadata, "version", lambda _: "fixture")
    monkeypatch.setattr(bootstrap.runpy, "run_path", lambda path:
                        {"download_file": downloads.append} if path.endswith("notebook_download.py")
                        else {"numbered_output": lambda *a, **kw: output})
    value = bootstrap.bootstrap_git(revision, run=True, run_number=35, session_seconds=3600, rebuild=True)
    assert value["optimizer_steps"] == 24000 and value["status"] == "complete"
    assert "--rebuild" in received and "--weights" not in received
    assert Path(received[received.index("--operator-metadata") + 1]).name == "bootstrap.json"
    assert [path.name for path in downloads] == [output.name + "-results.zip", output.name + "-checkpoints.zip"]
    assert cleaned == [roots[bootstrap.SCIENTIFIC_REVISION][1], roots[revision][1]]
