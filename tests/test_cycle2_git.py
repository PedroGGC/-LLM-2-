import ast
import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import time
import urllib.request

import pytest

from llm2.research import cycle2, cycle2_bootstrap as bootstrap
from llm2.research.session import SessionBudget


ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "scripts/build_cycle2_release.py"))


def test_two_hour_defaults_and_download_time_counted(monkeypatch):
    now = [160.]
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: now[0])
    session = bootstrap._BootstrapSession(started_at=100.)
    assert session.remaining() == 6840.
    assert session.remaining(export=True) == 7140.
    assert SessionBudget().seconds == 7200.
    assert cycle2.parser().parse_args(["--input", "unused"]).session_seconds == 7200.
    now[0] = 7000.
    with pytest.raises(TimeoutError):
        session.remaining()


@pytest.mark.parametrize("seconds", [0, 300, -1, 14401, float("nan"), float("inf")])
def test_bad_duration_rejected_before_external_actions(monkeypatch, seconds):
    def unwanted():
        raise AssertionError("hardware/setup started before validating duration")
    monkeypatch.setattr(bootstrap, "_hardware", unwanted)
    with pytest.raises(ValueError):
        bootstrap.bootstrap_git("a"*40, run=True, session_seconds=seconds)


def test_git_bootstrap_is_inert_and_requires_full_sha(monkeypatch):
    def unwanted(*args, **kwargs):
        raise AssertionError("unexpected external action")
    monkeypatch.setattr(bootstrap, "_hardware", unwanted)
    monkeypatch.setattr(bootstrap, "bounded_command", unwanted)
    assert bootstrap.bootstrap_git("a"*40) == {"status": "not_started"}
    with pytest.raises(ValueError, match="40-character"):
        bootstrap.bootstrap_git("experiment/fractal-lrm", run=True)


@pytest.mark.skipif(not shutil.which("git"), reason="git is not installed")
def test_git_bootstrap_clones_exact_revision_and_hashes_sources(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repository, stderr=subprocess.STDOUT, text=True).strip()
    git("init", "-b", "experiment/fractal-lrm")
    (repository / "model.py").write_text("version_one = True\n")
    git("add", "model.py")
    git("-c", "user.name=Cycle2 Test", "-c", "user.email=cycle2@example.invalid", "commit", "-m", "first")
    revision = git("rev-parse", "HEAD")
    (repository / "model.py").write_text("version_two = True\n")
    git("add", "model.py")
    git("-c", "user.name=Cycle2 Test", "-c", "user.email=cycle2@example.invalid", "commit", "-m", "second")
    work = tmp_path / "working"
    work.mkdir()
    monkeypatch.setattr(bootstrap, "_hardware", lambda: (work, ["0", "1"], "unchanged"))
    original = bootstrap.bounded_command
    def local_command(arguments, **kwargs):
        arguments = list(arguments)
        if arguments[:2] == ["git", "clone"]:
            arguments[-2] = str(repository)
        return original(arguments, **kwargs)
    monkeypatch.setattr(bootstrap, "bounded_command", local_command)
    def inspect(source, parent, metadata, session, hardware, run_number):
        assert (source / "model.py").read_text() == "version_one = True\n"
        assert metadata["source_git_revision"] == revision
        assert metadata["source_files_sha256"] == {"model.py": hashlib.sha256((source / "model.py").read_bytes()).hexdigest()}
        assert session.limit == 7200. and session.reserve == 300.
        assert run_number == 17
        return {"verified": True}
    monkeypatch.setattr(bootstrap, "_run_source", inspect)
    assert bootstrap.bootstrap_git(revision, run=True) == {"verified": True}


def test_setup_passes_same_two_hour_clock_to_runner(tmp_path, monkeypatch):
    source = tmp_path / "source"
    research = source / "src/llm2/research"
    research.mkdir(parents=True)
    parent = tmp_path / "parent"
    parent.mkdir()
    work = tmp_path / "working"
    work.mkdir()
    payload = b"verified corpus"
    (research / "cycle2_protocol.py").write_text(
        'CORPUS_URL = "https://example.invalid/corpus"\n'
        f'CORPUS_SHA256 = "{hashlib.sha256(payload).hexdigest()}"\n')
    (research / "naming.py").write_text(
        'def numbered_output(work, **kwargs):\n    return work / "llm2-campaign-0017"\n')
    now = [160.]
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(bootstrap.importlib.metadata, "version", lambda name: "unchanged")
    monkeypatch.setattr(bootstrap, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    commands = []
    def command(arguments, **kwargs):
        commands.append((list(map(str, arguments)), kwargs))
        if "pip" in arguments:
            now[0] += 20.
    monkeypatch.setattr(bootstrap, "bounded_command", command)
    session = bootstrap._BootstrapSession(started_at=100.)
    bootstrap._run_source(source, parent, {"source_files_sha256": {}}, session,
                          (work, ["0", "1"], "unchanged"), 17)
    runner, options = commands[-1]
    assert runner[runner.index("--session-seconds")+1] == "7200.0"
    assert runner[runner.index("--export-reserve-seconds")+1] == "300.0"
    assert float(runner[runner.index("--bootstrap-elapsed-seconds")+1]) == 80.
    assert options["timeout"] == 7120.
    metadata = json.loads((parent / "bootstrap.json").read_text())
    assert metadata["session_limit_seconds"] == 7200.
    assert metadata["setup_elapsed_seconds"] == 80.


def test_short_cell_active_download_and_default_notebook_inert(tmp_path, monkeypatch):
    revision = "a"*40
    builder = BUILD["build_git"]
    metadata = builder(tmp_path, revision)
    assert metadata["session_seconds"] == 7200
    cell = (tmp_path / "rodar_ciclo2_kaggle_2h.py").read_text(encoding="utf-8")
    ast.parse(cell)
    assert len(cell.splitlines()) < 40
    requests = []
    def download(url, **kwargs):
        requests.append(url)
        return io.BytesIO(b"def bootstrap_git(revision, **kwargs):\n    global result\n    result = (revision, kwargs)\n")
    monkeypatch.setattr(urllib.request, "urlopen", download)
    namespace = {}
    exec(compile(cell, "short-cell", "exec"), namespace)
    assert requests == [f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{revision}/src/llm2/research/cycle2_bootstrap.py"]
    actual_revision, settings = namespace["namespace"]["result"]
    assert actual_revision == revision and settings["session_seconds"] == 7200
    assert settings["run"] is True and settings["run_number"] == 17
    assert settings["started_at"] <= time.monotonic()
    def unwanted(*args, **kwargs):
        raise AssertionError("inactive notebook attempted external action")
    monkeypatch.setattr(urllib.request, "urlopen", unwanted)
    notebook = json.loads((tmp_path / "rodar_ciclo2_kaggle_2h.ipynb").read_text(encoding="utf-8"))
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            assert entry["execution_count"] is None and entry["outputs"] == []
            exec(compile("".join(entry["source"]), "notebook-cell", "exec"), {})
    with pytest.raises(FileExistsError):
        builder(tmp_path, revision)
