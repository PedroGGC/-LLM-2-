import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

from llm2.research.cycle2_bootstrap import extract_verified, bounded_command, bootstrap
from llm2.research import cycle2, cycle2_report
from llm2.research.cycle2_protocol import MODELS


def archive(path, name, payload=b"source"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_extract_checks_hash_and_refuses_existing(tmp_path):
    path = tmp_path / "source.zip"
    digest = archive(path, "src/llm2/test.py")
    with pytest.raises(ValueError, match="hash"):
        extract_verified(path, "0"*64, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    hashes = extract_verified(path, digest, tmp_path / "ok")
    assert hashes == {"src/llm2/test.py": hashlib.sha256(b"source").hexdigest()}
    with pytest.raises(FileExistsError): extract_verified(path, digest, tmp_path / "ok")


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "C:/escape.py", "a\\..\\escape.py"])
def test_source_archive_traversal_rejected_before_extraction(tmp_path, name):
    path = tmp_path / "bad.zip"
    digest = archive(path, name)
    with pytest.raises(ValueError): extract_verified(path, digest, tmp_path / "unpack")
    assert not (tmp_path / "unpack").exists()


def test_symlink_and_duplicate_members_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    link = zipfile.ZipInfo("symlink")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    digest = archive(path, link, b"outside")
    with pytest.raises(ValueError): extract_verified(path, digest, tmp_path / "unpack")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.py", b"one")
        with pytest.warns(UserWarning): z.writestr("x.py", b"two")
    with pytest.raises(ValueError): extract_verified(path, hashlib.sha256(path.read_bytes()).hexdigest(), tmp_path / "other")


def test_bootstrap_plan_is_inert(monkeypatch):
    from llm2.research import cycle2_bootstrap
    def unwanted(*args, **kwargs): raise AssertionError("external action during plan")
    monkeypatch.setattr(cycle2_bootstrap, "urlopen", unwanted)
    monkeypatch.setattr(cycle2_bootstrap.subprocess, "Popen", unwanted)
    assert bootstrap("0"*64, run=False) == {"status": "not_started"}


def test_bootstrap_timeout_kills_own_process_group(tmp_path):
    with pytest.raises(TimeoutError):
        bounded_command([sys.executable, "-c", "import time;time.sleep(30)"], cwd=tmp_path, timeout=.2)


def test_post_extract_source_tampering_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cycle2, "ROOT", tmp_path)
    source = tmp_path / "model.py";source.write_text("original")
    metadata = tmp_path / "bootstrap.json"
    metadata.write_text(json.dumps({"source_files_sha256": {"model.py": hashlib.sha256(source.read_bytes()).hexdigest()}}))
    cycle2.verify_bootstrap(metadata)
    source.write_text("modified")
    with pytest.raises(ValueError, match="changed"): cycle2.verify_bootstrap(metadata)


def test_factorial_report_uses_only_complete_seed_blocks(tmp_path, monkeypatch):
    rows = []
    for i,model in enumerate(MODELS):
        rows.append({"model": model, "seed": 81, "validation_loss": [1.39,1.405,1.40,1.41,1.43,1.44][i],
                     "train_wall_seconds": 100., "registered_parameters": 100, "parameters_in_tensors_with_gradients": 90,
                     "training_targets": 1000, "stop_reason": "max_steps"})
    rows.append(dict(rows[0],seed=82))
    monkeypatch.setattr(cycle2_report, "collect_rows", lambda campaign: (rows, []))
    fake = type("Campaign", (), {"output": tmp_path, "manifest": {"runs": [], "seeds": [81,82], "status": "partial_budget"}})()
    result = cycle2_report.write_report(fake)
    assert result["complete_seed_blocks"] == 1 and not result["confirmatory"] and not result["cost_equalized"]
    assert result["effects"]["exchange_effect_film_on"]["mean"] == pytest.approx(-.03)
    assert result["effects"]["film_effect_exchange_on"]["mean"] == pytest.approx(-.01)
    assert result["effects"]["interaction_difference_in_differences"]["mean"] == pytest.approx(0.)
    assert result["effects"]["exchange_effect_film_on"]["ci95_t_exploratory"] is None
