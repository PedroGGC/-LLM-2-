"""Paste/run through the pinned bootstrap cell in docs/KAGGLE_T4_CAMPAIGN.md.

Runs ONLY in Kaggle's browser notebook. No local GPU or local environment is needed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import urlopen
import uuid

PIN = os.environ.get("LLM2_REVISION", "")
SETTINGS = globals().get("LLM2_SETTINGS", {})
if not re.fullmatch(r"[0-9a-f]{40}", PIN):
    raise ValueError("Set LLM2_REVISION to the exact published 40-character commit from the delivery.")
WORK = Path("/kaggle/working")
if not WORK.is_dir():
    raise RuntimeError("Run this cell in Kaggle Notebooks with Internet and GPU T4x2 enabled.")

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("In Kaggle Settings select GPU T4 x2; this campaign requires two visible CUDA GPUs.")
torch_release = tuple(int(part) for part in str(torch.__version__).split("+")[0].split(".")[:2])
if not (torch_release[0] == 2 and torch_release[1] >= 5):
    raise RuntimeError("Use a Kaggle image with CUDA PyTorch >=2.5,<3; this bootstrap will not replace Torch automatically.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

REPO = WORK / f"llm2-{PIN[:12]}"
URL = "https://github.com/PedroGGC/-LLM-2-.git"
if not REPO.exists():
    subprocess.run(["git", "clone", "--no-checkout", "--single-branch",
                    "--branch", "experiment/fractal-lrm", URL, str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", PIN], check=True)
head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
if head != PIN:
    raise RuntimeError("Existing checkout has another commit; choose a fresh directory, do not reset old runs.")
subprocess.run(["git", "-C", str(REPO), "diff", "--exit-code", "HEAD"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-e", str(REPO)+"[dev,research]"], check=True)

SESSION = uuid.uuid4().hex[:12]
OUTPUT = WORK / f"llm2-campaign-{SESSION}"
command = [sys.executable, str(REPO / "scripts/run_kaggle.py"), "--execute",
           "--output", str(OUTPUT), "--device", "cuda", "--gpus", "0", "1",
           "--steps", str(SETTINGS.get("steps", 2000)),
           "--max-seconds", str(SETTINGS.get("max_seconds", 1800)),
           "--models", *SETTINGS.get("models", ["baseline", "lrm", "flrm_reference", "flrm_batched"]),
           "--seeds", *map(str, SETTINGS.get("seeds", [42]))]

data = SETTINGS.get("data")
inputs = SETTINGS.get("input", [])
if data and inputs:
    raise ValueError("Choose an existing prepared data folder OR raw input files.")
if data:
    command += ["--data", str(data)]
elif inputs:
    command += ["--input", *map(str, inputs)]
else:
    # ~1.1 MB, immutable public source revision; no large dataset download.
    corpus_revision = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
    corpus_url = f"https://raw.githubusercontent.com/karpathy/char-rnn/{corpus_revision}/data/tinyshakespeare/input.txt"
    folder = WORK / f"llm2-input-{SESSION}"
    folder.mkdir(exist_ok=False)
    with urlopen(corpus_url, timeout=60) as response:
        content = response.read(2*1024*1024+1)
    if not content or len(content) > 2*1024*1024:
        raise ValueError("Unexpected corpus size; download limited to 2 MiB.")
    content.decode("utf-8")
    corpus = folder / "tinyshakespeare.txt"
    corpus.write_bytes(content)
    provenance = {"url": corpus_url, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    (folder / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    command += ["--input", str(corpus)]
    print("Corpus:", provenance)

if SETTINGS.get("profile", True):
    command += ["--profile"]
if SETTINGS.get("cpu_tests", True):
    command += ["--cpu-tests"]
if SETTINGS.get("cuda_tests", True):
    command += ["--cuda-tests"]
if not SETTINGS.get("benchmark", True):
    command += ["--no-benchmark"]
if not SETTINGS.get("diagnostics", True):
    command += ["--no-diagnostics"]
if SETTINGS.get("evaluate_checkpoints"):
    command += ["--evaluate-checkpoints", *map(str, SETTINGS["evaluate_checkpoints"])]
print("Pinned commit:", PIN)
print("Fresh output:", OUTPUT)
print("Models:", SETTINGS.get("models", ["baseline", "lrm", "flrm_reference", "flrm_batched"]))
print("CPU/GPU tests and profiling will be recorded; failed commands stop the campaign.")
# Child imports THIS source, irrespective of an earlier editable installation in the notebook.
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO / "src")
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
subprocess.run(command, cwd=REPO, env=env, check=True)
archive = OUTPUT.parent / f"{OUTPUT.name}-results.zip"
print("\nFinished. Download this analysis bundle and send it back:")
print(archive)
print("Checkpoint tensors remain in the campaign directory. Save a Kaggle version with outputs to preserve them.")
try:
    from IPython.display import display, FileLink
    display(FileLink(str(archive)))
except ImportError:
    pass
