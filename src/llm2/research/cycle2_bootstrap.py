"""Standalone Kaggle bootstrap; standard library only until the verified package is loaded."""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import urlopen
import zipfile


def extract_verified(archive_path, expected_sha256, destination):
    archive_path, destination = Path(archive_path), Path(destination).resolve()
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Source ZIP hash differs from the tested release. Upload the correct package.")
    if destination.exists():
        raise FileExistsError("refusing to reuse an existing source checkout")
    hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) > 2000 or len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > 60*1024**2:
            raise ValueError("unexpected source archive size or duplicate entries")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in info.filename or ":" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16) or not name.parts):
                raise ValueError("unsafe source archive member")
        if archive.testzip() is not None:
            raise ValueError("source archive CRC failed")
        destination.mkdir(parents=True)
        for info in archive.infolist():
            path = destination / info.filename
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            payload = archive.read(info)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            hashes[info.filename] = hashlib.sha256(payload).hexdigest()
    return hashes


def bounded_command(arguments, *, cwd, timeout, env=None):
    if timeout <= 0:
        raise TimeoutError("session work time exhausted")
    process = subprocess.Popen(list(map(str, arguments)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               start_new_session=os.name == "posix")
    expired = threading.Event()

    def stop():
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()

    def expire():
        expired.set(); stop()

    timer = threading.Timer(timeout, expire)
    timer.daemon = True; timer.start()
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
        if expired.is_set():
            raise TimeoutError("global session deadline reached; complete checkpoints remain in the output folder")
        if code:
            raise subprocess.CalledProcessError(code, arguments)
    finally:
        timer.cancel()
        process.stdout.close()
        stop()  # Also stop leftover compiler descendants in this command's own process group.


class _BootstrapSession:
    def __init__(self, seconds=7200., started_at=None):
        now = time.monotonic()
        self.limit, self.reserve = float(seconds), 300.
        self.started = now if started_at is None else float(started_at)
        if (not all(math.isfinite(v) for v in (self.limit, self.started))
                or not self.reserve < self.limit <= 14400 or not 0 <= self.started <= now):
            raise ValueError("invalid session limit/start; limit must exceed 300 s and be at most 14400 s")
        self.remaining()

    @property
    def elapsed(self):
        return time.monotonic()-self.started

    def remaining(self, export=False):
        result = self.limit - self.elapsed - (0 if export else self.reserve)
        if result <= 0:
            raise TimeoutError("global session time exhausted")
        return result


def _hardware():
    work = Path("/kaggle/working")
    if not work.is_dir():
        raise RuntimeError("Use this bootstrap in a Kaggle notebook, with Internet and T4 GPU enabled.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle T4 GPU before running.")
    gpus = [str(i) for i in range(min(2, torch.cuda.device_count()))]
    if any("T4" not in torch.cuda.get_device_name(int(gpu)) for gpu in gpus):
        raise RuntimeError("This frozen protocol requires T4; other hardware needs separate validation.")
    before_torch = importlib.metadata.version("torch")
    return work, gpus, before_torch


def bootstrap(expected_sha256, *, run=False, run_number=17, archive=None,
              session_seconds=7200., started_at=None):
    """Run a verified ZIP; default maximum is two hours, not a minimum duration."""
    if not run:
        print("Execution disabled; no setup or training started.")
        return {"status": "not_started"}
    session = _BootstrapSession(session_seconds, started_at)
    work, gpus, before_torch = _hardware()
    candidates = [Path(archive)] if archive else sorted(Path("/kaggle/input").rglob("llm2-cycle2-source-*.zip"))
    candidates = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha256]
    if not candidates:
        raise FileNotFoundError("Attach the supplied llm2-cycle2-source ZIP as a Kaggle input dataset.")
    parent = Path(tempfile.mkdtemp(prefix="llm2-cycle2-code-", dir=work))
    source = parent / "source"
    hashes = extract_verified(candidates[0], expected_sha256, source)
    metadata = {"source_mode": "zip", "source_archive_sha256": expected_sha256,
                "source_archive": str(candidates[0]), "source_files_sha256": hashes}
    return _run_source(source, parent, metadata, session, (work, gpus, before_torch), run_number)


def bootstrap_git(revision, *, run=False, run_number=17, session_seconds=7200., started_at=None):
    """Clone the exact published revision; include cell download time in the session."""
    if not run:
        print("Execution disabled; no setup or training started.")
        return {"status": "not_started"}
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("use a complete 40-character commit SHA")
    session = _BootstrapSession(session_seconds, started_at)
    hardware = _hardware()
    parent = Path(tempfile.mkdtemp(prefix="llm2-cycle2-git-", dir=hardware[0]))
    source = parent / "source"
    url = "https://github.com/PedroGGC/-LLM-2-.git"
    bounded_command(["git", "clone", "--no-checkout", "--single-branch", "--branch",
                     "experiment/fractal-lrm", url, source], cwd=parent, timeout=session.remaining())
    bounded_command(["git", "checkout", "--detach", revision], cwd=source, timeout=session.remaining())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True,
                                   timeout=session.remaining()).strip()
    if head != revision:
        raise ValueError("checkout revision differs from the requested commit")
    bounded_command(["git", "diff", "--exit-code", "HEAD"], cwd=source, timeout=session.remaining())
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=source,
                                    timeout=session.remaining()).decode("utf-8").split("\0")
    hashes = {}
    for name in filter(None, names):
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError(f"unsupported source member: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"source_mode": "git", "source_git_url": url, "source_git_revision": revision,
                "source_files_sha256": hashes}
    return _run_source(source, parent, metadata, session, hardware, run_number)


def _run_source(source, parent, metadata, session, hardware, run_number):
    work, gpus, before_torch = hardware
    print(f"Cycle 2: six variants, seeds 81/82/83, serial training, pilot first. "
          f"Session maximum: {session.limit/3600:g} hours; may finish earlier.", flush=True)
    session.remaining()
    print("Verified source. Installing project dependencies with the current Python.", flush=True)
    bounded_command([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(source)+"[dev,research]"],
                    cwd=source, timeout=session.remaining())
    if importlib.metadata.version("torch") != before_torch:
        raise RuntimeError("Installation changed Torch; restart and review the environment before training.")
    # Read fixed JSON-string constants from the verified source without importing stale llm2 modules.
    source_globals = {}
    for line in (source / "src/llm2/research/cycle2_protocol.py").read_text(encoding="utf-8").splitlines():
        if line.startswith(("CORPUS_SHA256 =", "CORPUS_URL =")):
            name, value = line.split("=", 1)
            source_globals[name.strip()] = json.loads(value.strip())
    raw = parent / "tinyshakespeare.txt"
    with urlopen(source_globals["CORPUS_URL"], timeout=min(60, session.remaining())) as response:
        payload = response.read(2*1024**2 + 1)
    if hashlib.sha256(payload).hexdigest() != source_globals["CORPUS_SHA256"]:
        raise ValueError("Downloaded corpus differs from campaign 0016")
    with raw.open("xb") as stream:
        stream.write(payload)
    numbering = runpy.run_path(str(source / "src/llm2/research/naming.py"))["numbered_output"]
    output = numbering(work, number=run_number, search_roots=[Path("/kaggle/input")])
    metadata.update({"source_root": str(source), "torch_preserved": before_torch,
                "corpus_url": source_globals["CORPUS_URL"], "corpus_sha256": source_globals["CORPUS_SHA256"],
                "setup_elapsed_seconds": session.elapsed, "session_limit_seconds": session.limit,
                "bootstrap_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else None})
    metadata_file = parent / "bootstrap.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    command = [sys.executable, str(source / "scripts/run_cycle2.py"), "--execute", "--input", raw,
               "--output", output, "--gpus", *gpus, "--session-seconds", session.limit,
               "--export-reserve-seconds", session.reserve, "--bootstrap-metadata", metadata_file,
               "--bootstrap-elapsed-seconds", session.elapsed]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        bounded_command(command, cwd=source, timeout=session.remaining(export=True), env=environment)
    finally:
        archive_path = output.with_name(output.name+"-results.zip")
        print(f"\nCampaign folder: {output}\nResults ZIP (if export completed): {archive_path}", flush=True)
        print("Save the complete output folder as well; the analysis ZIP excludes weights/optimizer.", flush=True)
    return {"campaign": str(output), "session_elapsed_seconds": session.elapsed}
