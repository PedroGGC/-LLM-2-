"""Pinned Kaggle bootstrap for the 0033 frozen-weight generation comparison."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from urllib.request import urlopen

HELPER_REVISION = "1c8ce2abb8f47d3167454b3503598854b7756dad"
HELPER_SHA256 = "5bc8f2ddcaf3cacbae401cbe077b494ddc9fc90c798417e3f1686d2635f925b7"
GUARD_SHA256 = "8c05edcf6d0450e273ac2f1fc118a8167bc9833490f841c9cfb1d57d96262007"
SCIENTIFIC_REVISION = "db507b5258615e94cab7620dc5816fb31c36d8c7"
REFERENCE = "docs/results/campaign_0033_decode_reference.json"
BENCHMARK = "src/llm2/research/decode_benchmark.py"


def fetch(name, digest):
    url = f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{HELPER_REVISION}/src/llm2/research/{name}.py"
    with urlopen(url, timeout=60) as response:
        raw = response.read(1024**2 + 1)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("pinned helper checksum mismatch")
    namespace = {"__name__": "decode_helper_" + name}
    exec(compile(raw.decode("utf-8"), url, "exec"), namespace)
    return namespace


def bootstrap_git(revision, *, run=False, run_number=34, session_seconds=1800, checkpoints=None, started_at=None):
    if not run:
        return {"status": "not_started"}
    if (not re.fullmatch(r"[0-9a-f]{40}", revision) or type(run_number) is not int
            or run_number < 34 or not 300 < float(session_seconds) <= 1800):
        raise ValueError("invalid revision, campaign number or session limit")
    started_at = time.monotonic() if started_at is None else started_at
    helper = fetch("cycle2_bootstrap", HELPER_SHA256)
    session = helper["_BootstrapSession"](session_seconds, started_at)
    hardware = helper["_hardware"]()
    work, gpus, before_torch = hardware
    if len(gpus) != 2:
        raise RuntimeError("Ative GPU T4 x2.")
    guard = fetch("kaggle_light", GUARD_SHA256)
    if guard["active_workers"]():
        raise RuntimeError("Há workers do projeto ativos.")
    parents = []
    try:
        operator, parent, operator_meta = helper["checkout_git_source"](revision, session, hardware, shallow=True)
        parents.append(parent)
        frozen, frozen_parent, frozen_meta = helper["checkout_git_source"](
            SCIENTIFIC_REVISION, session, hardware, shallow=True)
        parents.append(frozen_parent)
        globals()["download_file"] = runpy.run_path(str(frozen / "src/llm2/research/notebook_download.py"))["download_file"]
        reference = json.loads((operator / REFERENCE).read_text(encoding="utf-8"))
        frozen_hashes = {name: digest for name, digest in frozen_meta["source_files_sha256"].items()
                         if (name.startswith("src/") and name.endswith(".py")) or name == "pyproject.toml"}
        if reference["revision"] != SCIENTIFIC_REVISION or reference["source_sha256"] != frozen_hashes:
            raise ValueError("frozen source does not match the audited checkpoints")
        helper["bounded_command"](
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "-e", str(frozen) + "[research]"],
            cwd=frozen, timeout=session.remaining(),
        )
        if importlib.metadata.version("torch") != before_torch:
            raise RuntimeError("dependency installation changed Torch")
        number = runpy.run_path(str(frozen / "src/llm2/research/naming.py"))["numbered_output"]
        while True:
            try:
                output = number(work, number=run_number, search_roots=[Path("/kaggle/input")])
                break
            except FileExistsError:
                run_number += 1
        metadata = parent / "decode-bootstrap.json"
        metadata.write_text(json.dumps({
            "operator_revision": revision, "scientific_revision": SCIENTIFIC_REVISION,
            "operator_sha256": {name: operator_meta["source_files_sha256"][name] for name in (
                BENCHMARK, REFERENCE, "src/llm2/research/decode_cache.py", "src/llm2/research/decode_bootstrap.py")},
            "torch_preserved": before_torch, "download_mode": "notebook_embedded_base64",
        }, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, operator / BENCHMARK, "--reference", operator / REFERENCE,
                   "--output", output, "--cache", work / "llm2-tinystories-prefix-cache",
                   "--operator-metadata", metadata, "--gpus", *gpus,
                   "--seconds", str(session_seconds), "--started-at", str(started_at)]
        if checkpoints:
            command.extend(["--weights", checkpoints])
        env = os.environ.copy()
        env.update(PYTHONPATH=str(frozen / "src"), PYTHONUNBUFFERED="1",
                   PYTHONIOENCODING="utf-8", LLM2_CYCLE4="1")
        bundle = work / f"{output.name}-results.zip"
        print(f"Diagnóstico {output.name} com os pesos finais da 0033 nas duas T4.", flush=True)
        try:
            helper["bounded_command"](command, cwd=frozen, env=env, timeout=session.remaining(export=True))
        finally:
            if bundle.is_file():
                try:
                    download_file(bundle)
                except Exception as exc:
                    print(f"Resultado em {bundle}. Botão indisponível: {exc}", flush=True)
        return {"campaign": str(output), "results": str(bundle), "optimizer_steps": 0}
    finally:
        for parent in reversed(parents):
            try:
                guard["remove_checkout"](work, parent)
            except Exception as exc:
                print(f"Checkout temporário preservado em {parent}: {exc}", flush=True)
