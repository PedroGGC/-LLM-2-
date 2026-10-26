"""Immutable atomic checkpoint directories. State files are trusted pickle only."""

import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_model, save_model

from .data import sha256_file, write_json
from .runtime import capture_rng, restore_rng


def class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def resume_contract(config):
    raw = config.to_dict()
    raw["data"].pop("path")  # Content is identified by the manifest SHA-256.
    for field in ("output_dir", "max_steps", "eval_interval", "eval_batches",
                  "checkpoint_interval", "log_interval"):
        raw["train"].pop(field)
    if "research" in raw:
        for name in ("max_seconds", "shutdown_reserve_seconds", "max_tokens", "max_train_flops"):
            raw["research"].pop(name)
    return raw


def save_checkpoint(path, *, model, objective, optimizer, scheduler, sampler, counters,
                    config, data_fingerprint, environment_info, provenance, flops):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable checkpoint exists: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        save_model(model, str(temporary / "model.safetensors"), metadata={"format": "pt"})
        weight_files = ["model.safetensors", "training_state.pt"]
        if objective.state_dict():
            save_model(objective, str(temporary / "objective.safetensors"), metadata={"format": "pt"})
            weight_files.append("objective.safetensors")
        state = {"schema_version": 1, "objective": objective.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "sampler": sampler.state_dict(), "rng": capture_rng(), "counters": counters.copy()}
        torch.save(state, temporary / "training_state.pt")
        metadata = {"schema_version": 1, "config": config.to_dict(),
                    "resume_contract": resume_contract(config), "dataset_sha256": data_fingerprint,
                    "model_class": class_name(model), "objective_class": class_name(objective),
                    "architecture_version": getattr(model, "architecture_version", 1),
                    "environment": environment_info, "provenance": provenance, "flops": flops,
                    "files": {name: sha256_file(temporary / name)
                              for name in weight_files}}
        write_json(temporary / "metadata.json", metadata)
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    latest_tmp = path.parent / ".latest.tmp"
    write_json(latest_tmp, {"checkpoint": path.name})
    os.replace(latest_tmp, path.parent / "latest.json")
    return path


def read_checkpoint_metadata(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    for name in metadata["files"]:
        if name not in ("model.safetensors", "training_state.pt", "objective.safetensors"):
            raise ValueError("unexpected checkpoint artifact")
        if sha256_file(path / name) != metadata["files"][name]:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    return metadata


def load_checkpoint(path, *, model, objective, optimizer, scheduler, sampler, config,
                    data_fingerprint, environment_info, provenance, flops):
    metadata = read_checkpoint_metadata(path)
    if metadata["resume_contract"] != resume_contract(config):
        raise ValueError("incompatible configuration for exact resume")
    if metadata["dataset_sha256"] != data_fingerprint:
        raise ValueError("dataset fingerprint changed")
    if metadata["model_class"] != class_name(model) or metadata["objective_class"] != class_name(objective):
        raise ValueError("model/objective implementation changed")
    if metadata.get("architecture_version", 1) != getattr(model, "architecture_version", 1):
        raise ValueError("architecture version changed")
    if metadata["environment"] != environment_info:
        raise ValueError("environment changed; exact resume requires the same runtime")
    if metadata["provenance"]["source_sha256"] != provenance["source_sha256"]:
        raise ValueError("source files changed; exact resume requires the same implementation")
    if metadata["flops"] != flops:
        raise ValueError("FLOP estimator changed")
    # weights_only=False is explicit: only load a trusted local training state.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["schema_version"] != 1:
        raise ValueError("unsupported training state schema")
    if state["counters"]["step"] > config.train.max_steps:
        raise ValueError("max_steps is earlier than checkpoint")
    load_model(model, str(Path(path) / "model.safetensors"), device=str(next(model.parameters()).device))
    objective.load_state_dict(state["objective"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampler"])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state["rng"])
    return state["counters"], metadata

