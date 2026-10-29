"""Serial Cycle 2 runner: full factorial, prior pilot, bounded session, fresh outputs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

import torch
import yaml

from ..config import load_config
from ..data import prepare_data, sha256_file, write_json
from .cycle2_protocol import (MODELS, SEEDS, CORPUS_SHA256, configuration, configuration_stats,
                              ordered_jobs, paired_fingerprints, verify_dataset)
from .cycle2_report import write_report
from .campaign import worker_environment
from .naming import numbered_output
from .session import SessionBudget, SessionBudgetExceeded, SessionCampaign

ROOT = Path(__file__).resolve().parents[3]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="TinyShakespeare UTF-8 file, verified by SHA-256")
    source.add_argument("--data", help="prepared dataset; token hashes must match 0016")
    result.add_argument("--config", default=str(ROOT / "configs/experiments/cycle2/base.yaml"))
    result.add_argument("--output", help="fresh campaign directory; numbered by default")
    result.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=SEEDS)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--gpus", nargs="+", default=["0", "1"])
    result.add_argument("--execute", action="store_true", help="without this flag: inspect only, no files/network/training")
    result.add_argument("--session-seconds", type=float, default=7200.)
    result.add_argument("--export-reserve-seconds", type=float, default=300.)
    result.add_argument("--bootstrap-elapsed-seconds", type=float, default=0.)
    result.add_argument("--bootstrap-metadata", help="verified source-package and setup provenance")
    result.add_argument("--engineering-smoke", action="store_true", help="tiny CPU-only engineering run; NOT the scientific protocol")
    result.add_argument("--skip-tests", action="store_true", help="explicit engineering escape; recorded, not recommended for Kaggle")
    return result


def plan(args):
    if not 0 < args.session_seconds <= 14400:
        raise ValueError("session must be positive and at most 14400 seconds")
    if args.engineering_smoke and args.device != "cpu":
        raise ValueError("engineering smoke is CPU only")
    if args.device == "cpu" and not args.engineering_smoke:
        raise ValueError("scientific protocol requires CUDA; use --engineering-smoke for CPU")
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus or any(not s.isdecimal() for s in args.gpus):
        raise ValueError("GPU slots must be unique nonnegative logical indices")
    budget = SessionBudget(args.session_seconds, args.export_reserve_seconds,
                           elapsed=args.bootstrap_elapsed_seconds)
    base = load_config(args.config)
    output = Path(args.output or ROOT / "runs/PLANNED-cycle2").resolve()
    data = Path(args.data or output / "data").resolve()
    jobs = ordered_jobs(args.seeds, args.gpus if args.device == "cuda" else [None])
    for job in jobs:
        cfg = configuration(base, job["model"], data, output / "runs" / job["id"], job["seed"],
                            device=args.device, smoke=args.engineering_smoke)
        job.update(config_resolved=cfg.to_dict(), **configuration_stats(cfg))
    return {"execution": "not_started", "cycle": 2, "seeds": args.seeds,
            "models": list(MODELS), "jobs": jobs, "session_budget": budget.snapshot(),
            "protocol": "engineering_smoke" if args.engineering_smoke else "cycle2_t4_fp32_2000",
            "cost_equalized": False, "schedule_policy": "fixed 2000 steps; serial jobs and alternating GPUs"}


def verify_bootstrap(path):
    if not path:
        return None
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, digest in metadata["source_files_sha256"].items():
        file = (ROOT / name).resolve()
        if not file.is_relative_to(ROOT) or sha256_file(file) != digest:
            raise ValueError(f"source package changed after verification: {name}")
    return metadata


def prepare(campaign, args):
    campaign.budget.require()
    verify_bootstrap(args.bootstrap_metadata)
    if args.bootstrap_metadata:
        shutil.copyfile(args.bootstrap_metadata, campaign.output / "bootstrap.json")
    if args.data:
        path = Path(args.data).resolve()
    else:
        raw = Path(args.input).resolve()
        if not args.engineering_smoke and sha256_file(raw) != CORPUS_SHA256:
            raise ValueError("TinyShakespeare source differs from 0016")
        path = campaign.output / "data"
        prepare_data([raw], path, kind="byte", seed=42)
        source_dir = campaign.output / "input_sources"
        source_dir.mkdir()
        shutil.copyfile(raw, source_dir / "0-tinyshakespeare.txt")
        campaign.manifest["raw_sources"] = [{"sha256": sha256_file(raw), "bytes": raw.stat().st_size,
                                              "bundled": True, "path": str(raw)}]
    data = verify_dataset(path, smoke=args.engineering_smoke)
    if min(len(v) for v in data.arrays.values()) <= (17 if args.engineering_smoke else 256):
        raise ValueError("dataset splits too short")
    export_data = campaign.output / "data"
    export_data.mkdir(exist_ok=True)
    if path != export_data:
        for name in ("manifest.json", "tokenizer.json"):
            shutil.copyfile(path / name, export_data / name)
    campaign.manifest.update(dataset_path=str(path), dataset_sha256=data.fingerprint,
                             dataset_manifest=data.manifest, status="checking")
    campaign.save()
    return path


def materialize(campaign, args, data):
    directory = campaign.output / "configs"
    directory.mkdir()
    base = load_config(args.config)
    jobs = ordered_jobs(args.seeds, campaign.gpus)
    for job in jobs:
        run = campaign.output / "runs" / job["id"]
        cfg = configuration(base, job["model"], data, run, job["seed"], device=args.device, smoke=args.engineering_smoke)
        file = directory / f"{job['id']}.yaml"
        file.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True), encoding="utf-8")
        job["physical_gpu"] = (worker_environment(ROOT, job["gpu_slot"]).get("CUDA_VISIBLE_DEVICES")
                               if args.device == "cuda" else None)
        job.update(directory=str(run), config=str(file), max_steps=cfg.train.max_steps,
                   checkpoint_interval=cfg.train.checkpoint_interval, progress_interval=cfg.train.checkpoint_interval,
                   iterations=cfg.research.iterations, batch_size=cfg.data.batch_size,
                   grad_accum_steps=cfg.train.grad_accum_steps, **configuration_stats(cfg))
    campaign.manifest["runs"] = jobs
    campaign.manifest["paired_init_verification"] = {}
    for seed in args.seeds:
        campaign.budget.require()
        configs = {job["model"]: load_config(job["config"]) for job in jobs if job["seed"] == seed}
        campaign.manifest["paired_init_verification"][str(seed)] = paired_fingerprints(configs)
    campaign.save()


def pilot(campaign, args):
    """All six arms on each available physical GPU, BEFORE any scientific training."""
    records = []
    warmup, measured = (0, 1) if args.engineering_smoke else (10, 30)
    for slot_index, gpu in enumerate(campaign.gpus):
        for model in MODELS:
            campaign.budget.require()
            job = next(j for j in campaign.manifest["runs"] if j["model"] == model)
            destination = campaign.output / "pilots" / f"{model}-gpu-{slot_index}"
            start = time.monotonic()
            campaign.command(["-m", "llm2", "profile", "--config", job["config"], "--execute",
                              "--warmup-steps", warmup, "--measure-steps", measured, "--output", destination],
                             f"pilot-{model}-gpu-{slot_index}.log", gpu=gpu)
            value = json.loads((destination / "profile.json").read_text(encoding="utf-8"))
            records.append({"model": model, "gpu_slot": gpu, "profile": str(destination / "profile.json"),
                            "mean_step_seconds": value["timing"]["mean_step_seconds"],
                            "warmup_seconds": value["warmup_timing"]["total_warmup_seconds"],
                            "subprocess_wall_seconds": time.monotonic()-start,
                            "measured_steps": measured, "warmup_steps": warmup})
            campaign.manifest["pilot"] = records
            campaign.save()
    for job in campaign.manifest["runs"]:
        record = next(r for r in records if r["model"] == job["model"] and r["gpu_slot"] == job["gpu_slot"])
        job["estimated_job_seconds"] = (record["mean_step_seconds"] * job["max_steps"] * 1.35
                                        + record["warmup_seconds"] + (5 if args.engineering_smoke else 180))
    campaign.manifest["pilot_completed_before_training"] = True
    campaign.manifest["estimated_science_seconds"] = sum(j["estimated_job_seconds"] for j in campaign.manifest["runs"])
    campaign.save()


def scientific_jobs(campaign, args):
    for seed in args.seeds:
        block = [j for j in campaign.manifest["runs"] if j["seed"] == seed]
        # Avoid starting a seed block that the conservative pilot estimate says will not fit.
        campaign.budget.require(sum(j["estimated_job_seconds"] for j in block))
        for job in block:
            campaign.budget.require(job["estimated_job_seconds"])
            campaign.manifest["status"] = "training"
            campaign.save()
            campaign.command(["-m", "llm2", "diagnose", "--config", job["config"], "--device", args.device,
                              "--samples", "128", "--batch-size", "4", "--output",
                              campaign.output / "diagnostics" / f"init-{job['id']}.json"],
                             f"init-{job['id']}-diagnose.log", gpu=job["gpu_slot"])
            campaign.train_worker([job], job["gpu_slot"])
            if job["stop_reason"] != "max_steps":
                raise SessionBudgetExceeded("training ended before the fixed horizon")
            campaign.manifest["status"] = "evaluating"
            campaign.evaluate_run(job, evaluate_all_checkpoints=True, diagnose_steps=[job["max_steps"]], diagnose_batch_size=4)
    campaign.manifest["status"] = "complete"


def export(campaign):
    """Atomic exclusive publication; no weights/data binaries or previous archives overwritten."""
    destination = campaign.output.parent / f"{campaign.output.name}-results.zip"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".cycle2-export-", suffix=".zip", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(campaign.output.rglob("*")):
                if path.is_file() and (path.suffix in (".json", ".jsonl", ".yaml", ".txt", ".md", ".csv", ".log", ".png") or path.name == "source.zip"):
                    archive.write(path, arcname=path.relative_to(campaign.output).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("export CRC failed")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def execute(args, *, started=None):
    started = time.monotonic() if started is None else started
    budget = SessionBudget(args.session_seconds, args.export_reserve_seconds,
                           elapsed=args.bootstrap_elapsed_seconds + time.monotonic()-started)
    if args.device == "cuda":
        if not torch.cuda.is_available() or any(int(slot) >= torch.cuda.device_count() for slot in args.gpus):
            raise RuntimeError("requested CUDA GPU slots are unavailable")
        if any("T4" not in torch.cuda.get_device_name(int(slot)) for slot in args.gpus):
            raise RuntimeError("Cycle 2 protocol targets T4; other hardware needs a separately recorded experiment")
    output = Path(args.output) if args.output else numbered_output(ROOT / "runs")
    campaign = SessionCampaign(ROOT, output, device=args.device, gpus=args.gpus, budget=budget)
    campaign.manifest.update(cycle=2, seeds=args.seeds, models=list(MODELS),
        protocol="engineering_smoke" if args.engineering_smoke else "cycle2_t4_fp32_2000",
        scheduling="serial, rotated order by seed, GPU alternation, all evaluations on first GPU",
        cost_equalized=False, pilot_completed_before_training=False, tests_skipped=args.skip_tests,
        limitations="Exploratory same-corpus factorial, three seeds; different registered/active parameters; estimated FLOPs exclude elementwise and overestimate backward for disconnected branches. No new reasoning/general superiority claim.")
    error = None
    try:
        (campaign.output / "diagnostics").mkdir()
        data = prepare(campaign, args)
        materialize(campaign, args, data)
        if not args.skip_tests:
            campaign.command(["-m", "pytest", "-q", "-m", "not cuda"], "tests-cpu.log", gpu=campaign.gpus[0])
            if args.device == "cuda":
                campaign.command(["-m", "pytest", "-q", "-m", "cuda", "-k", "fp32"], "tests-cuda-fp32.log", gpu=campaign.gpus[0])
        pilot(campaign, args)
        scientific_jobs(campaign, args)
    except SessionBudgetExceeded as exc:
        campaign.manifest.update(status="partial_budget", error=str(exc))
        for job in campaign.manifest["runs"]:
            if job["status"] in ("training", "failed"):
                job["status"] = "interrupted_budget"
                checkpoints = sorted((Path(job["directory"]) / "checkpoints").glob("step-*"))
                job["last_complete_checkpoint"] = str(checkpoints[-1]) if checkpoints else None
    except Exception as exc:
        error = exc
        campaign.manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        campaign.abort()
        campaign.manifest["session_budget"] = budget.snapshot()
        campaign.manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        campaign.save()
        try:
            write_report(campaign)
        except Exception as report_error:
            campaign.manifest.update(report_error=str(report_error), status="failed")
            error = error or report_error
            campaign.save()
        archive = export(campaign)
        print(f"\nStatus: {campaign.manifest['status']}\nResultados: {archive}\nPreserve também a pasta completa com checkpoints: {campaign.output}", flush=True)
    if error is not None:
        raise RuntimeError(f"Cycle 2 failed; partial evidence exported to {archive}") from error
    return {"status": campaign.manifest["status"], "results_zip": str(archive), "campaign": str(campaign.output)}


def main(argv=None, *, started=None):
    args = parser().parse_args(argv)
    description = plan(args)
    if not args.execute:
        print(json.dumps(description, ensure_ascii=False, indent=2))
        return description
    return execute(args, started=started)
