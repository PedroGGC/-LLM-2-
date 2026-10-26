"""Reproducibility, device/precision policy and environment provenance."""

from contextlib import nullcontext
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import zipfile

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import sha256_file


def setup_runtime(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use the CPU config locally")
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is not supported by this runtime/device")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    return torch.device(config.device)


def autocast_context(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def attention_context(deterministic):
    # Exact resume contract uses the math backend, including on CUDA.
    return sdpa_kernel(SDPBackend.MATH) if deterministic else nullcontext()


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG topology differs from this runtime")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _git(args, root):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def environment():
    packages = ("torch", "numpy", "PyYAML", "tokenizers", "safetensors", "tensorboard", "pytest")
    return {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
            "packages": {p: importlib.metadata.version(p) for p in packages
                         if importlib.util.find_spec("pytest" if p == "pytest" else
                                                    "yaml" if p == "PyYAML" else p)},
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src").rglob("*.py")) if (root / "src").exists() else []
    paths += [p for p in (root / "pyproject.toml",) if p.exists()]
    try:
        git = {"commit": _git(["rev-parse", "HEAD"], root),
               "status": _git(["status", "--porcelain"], root),
               "diff": _git(["diff", "HEAD", "--", "src", "configs", "pyproject.toml"], root)}
    except FileNotFoundError:
        git = {"commit": None, "status": None, "diff": None}
    configs = sorted((root / "configs").rglob("*.yaml")) if (root / "configs").exists() else []
    return {"git": git, "source_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p)
                                           for p in paths},
            "versioned_configs_sha256": {p.relative_to(root / "configs").as_posix(): sha256_file(p) for p in configs}}


def save_source_snapshot(path, provenance):
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        names = list(provenance["source_sha256"])
        names += [f"configs/{name}" for name in provenance["versioned_configs_sha256"]]
        for name in names:
            archive.write(root / name, arcname=name)

