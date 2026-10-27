"""Explicit, failure-aware campaigns for Kaggle T4 notebooks (or tiny CPU verification).

All work uses fresh directories. Workers use the current interpreter and checkout.
A campaign is a package comparison, not automatically an iso-parameter/FLOP ablation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import uuid
import zipfile

import torch
import yaml

from ..checkpoint import read_checkpoint_metadata
from ..config import ExperimentConfig, ResearchConfig, load_config
from ..data import PreparedData, prepare_data, sha256_file, write_json
from ..runtime import source_provenance


MODELS = ("baseline", "baseline_small", "baseline_jepa", "lrm", "lrm_no_jepa",
          "flrm_reference", "flrm_batched", "flrm_no_jepa")


def worker_environment(root, gpu=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(root) / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def campaign_config(base, model, data, output, *, seed, steps, device, max_seconds):
    if model not in MODELS:
        raise ValueError(f"unknown campaign model: {model}")
    cfg = ExperimentConfig.from_dict(base.to_dict())
    cfg.data.path, cfg.train.output_dir = str(Path(data).resolve()), str(Path(output).resolve())
    cfg.train.seed, cfg.train.max_steps = seed, steps
    cfg.optimizer.schedule_steps = max(cfg.optimizer.schedule_steps, steps)
    cfg.train.device, cfg.train.precision, cfg.train.deterministic = device, "fp32", device == "cpu"
    cfg.train.compile = False
    cfg.train.console_progress = True
    cfg.train.accumulation_weighting = "tokens"
    cfg.train.checkpoint_interval = max(1, min(500, steps))
    architecture = "transformer" if model.startswith("baseline") else "recursive" if model.startswith("lrm") else "fractal"
    jepa = 0.1 if model in ("baseline_jepa", "lrm", "flrm_reference", "flrm_batched") else 0
    cfg.research = ResearchConfig(architecture=architecture, iterations=1,
        latent_steps=1 if architecture == "recursive" else 0, jepa_weight=jepa,
        scale_execution="batched" if model == "flrm_batched" else "reference",
        max_seconds=max_seconds, shutdown_reserve_seconds=min(30., max_seconds/10) if max_seconds else 0.)
    if model == "baseline_small":
        # At D=384 this is 2,164,224 parameters; inspect the resolved config for other widths.
        cfg.model.n_layers = 1
        cfg.model.hidden_dim = (cfg.model.hidden_dim*5)//4
    return ExperimentConfig.from_dict(cfg.to_dict())


def export_campaign(directory):
    """A small analysis bundle, not a resumable checkpoint archive."""
    directory = Path(directory).resolve()
    destination = directory.parent / f"{directory.name}-results.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix in (".json", ".jsonl", ".yaml", ".txt", ".md", ".csv", ".svg", ".log")
                                   or path.name == "source.zip"):
                archive.write(path, arcname=path.relative_to(directory).as_posix())
    return destination


class Campaign:
    def __init__(self, root, output, *, device, gpus):
        self.root, self.output, self.device = Path(root).resolve(), Path(output).resolve(), device
        self.output.mkdir(parents=True, exist_ok=False)
        self.gpus = gpus if device == "cuda" else [None]
        self.lock, self.cancelled = threading.RLock(), threading.Event()
        self.processes = set()
        self.manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "preparing", "root": str(self.root), "python": sys.executable,
            "provenance": source_provenance(), "device": device, "gpu_slots": self.gpus, "runs": [],
            "limitations": "Package comparison: model sizes/depths/JEPA differ. A single seed or short profile cannot establish superiority. Performance with concurrent jobs can include shared-host effects. ZIP excludes checkpoint tensors and is not resumable."}
        self.save()

    def save(self):
        with self.lock:
            temporary = self.output / ".campaign.tmp"
            write_json(temporary, self.manifest)
            os.replace(temporary, self.output / "campaign.json")

    def command(self, arguments, log_name, *, gpu=None):
        if self.cancelled.is_set():
            raise RuntimeError("campaign cancelled after another worker failed")
        path = self.output / "logs" / log_name
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, *map(str, arguments)]
        env = worker_environment(self.root, gpu)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"command": command, "cwd": str(self.root),
                                     "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES")})+"\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            with self.lock:
                self.processes.add(process)
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    if line.startswith("TRAIN "):
                        print(f"[{log_name}] {line.rstrip()}", flush=True)
                returncode = process.wait()
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                with self.lock:
                    self.processes.discard(process)
            if returncode:
                tail = "\n".join(path.read_text(encoding="utf-8").splitlines()[-40:])
                raise RuntimeError(f"command exited {returncode}; complete log: {path}\n{tail}")

    def abort(self):
        self.cancelled.set()
        with self.lock:
            for process in tuple(self.processes):
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass

    def train_worker(self, jobs, gpu):
        for job in jobs:
            if self.cancelled.is_set():
                return
            job["status"] = "training"
            self.save()
            try:
                self.command(["-m", "llm2", "train", "--config", job["config"]],
                             f"{job['id']}-train.log", gpu=gpu)
                directory = Path(job["directory"])
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                checkpoint = Path(summary["checkpoint"]).resolve()
                if checkpoint.parent != (directory / "checkpoints").resolve():
                    raise ValueError("checkpoint does not belong to this fresh run")
                metadata = read_checkpoint_metadata(checkpoint)
                expected = load_config(job["config"])
                if metadata["config"] != expected.to_dict():
                    raise ValueError("checkpoint config differs from campaign run")
                if metadata["dataset_sha256"] != self.manifest["dataset_sha256"]:
                    raise ValueError("checkpoint dataset differs from campaign")
                if metadata["provenance"]["source_sha256"] != self.manifest["provenance"]["source_sha256"]:
                    raise ValueError("worker imported different source code")
                job.update(status="trained", checkpoint=str(checkpoint), stop_reason=summary["stop_reason"],
                           step=summary["step"], checkpoint_model_sha256=metadata["files"]["model.safetensors"])
                self.save()
                print(f"Trained {job['id']}: step={summary['step']} reason={summary['stop_reason']}", flush=True)
            except Exception as exc:
                job.update(status="failed", error=str(exc))
                self.save()
                self.abort()
                raise

    def evaluate_run(self, job, *, benchmark=True, diagnostics=True):
        # Sequential post-training measurements on the SAME physical GPU for every model.
        gpu = self.gpus[0]
        checkpoint = Path(job["checkpoint"])
        destination = Path(job["directory"]) / "evaluations" / checkpoint.name
        destination.mkdir(parents=True, exist_ok=False)
        common = ["--checkpoint", str(checkpoint), "--device", self.device, "--precision", "fp32"]
        self.command(["-m", "llm2", "evaluate", *common, "--output", destination / "language.json",
                      "--windows-output", destination / "windows.jsonl"],
                     f"{job['id']}-evaluate.log", gpu=gpu)
        result = json.loads((destination / "language.json").read_text(encoding="utf-8"))
        if result["validation_scope"] != "all_full_windows" or result["dataset_sha256"] != self.manifest["dataset_sha256"]:
            raise ValueError("final evaluation scope/dataset mismatch")
        if result["checkpoint_model_sha256"] != job["checkpoint_model_sha256"]:
            raise ValueError("final evaluation belongs to different weights")
        if benchmark:
            self.command(["-m", "llm2", "benchmark", *common, "--prompt", "First Citizen:",
                          "--warmup-tokens", "8", "--max-new-tokens", "64",
                          "--output", destination / "benchmark.json"],
                         f"{job['id']}-benchmark.log", gpu=gpu)
            self.command(["-m", "llm2", "generate", *common, "--prompt", "First Citizen:",
                          "--seed", "42", "--temperature", "0.8", "--max-new-tokens", "64"],
                         f"{job['id']}-generation.log", gpu=gpu)
        if diagnostics:
            self.command(["-m", "llm2", "diagnose", *common, "--samples", "128",
                          "--output", destination / "diagnostics.json"],
                         f"{job['id']}-diagnostics.log", gpu=gpu)
        job.update(status="complete", evaluation=str(destination / "language.json"),
                   full_validation_loss=result["validation_loss"], validation_tokens=result["validation_tokens"],
                   evaluation_physical_gpu=gpu)
        self.save()

    def profile_runs(self, jobs, warmup, measured):
        selected = [j for j in jobs if j["model"] in ("flrm_reference", "flrm_batched")]
        if not selected:
            selected = jobs[:1]
        # One seed per execution mode, and never overlapping GPU profiles.
        seen = set()
        for job in selected:
            if job["model"] in seen:
                continue
            seen.add(job["model"])
            cfg = load_config(job["config"])
            cfg.train.console_progress = False
            cfg.train.max_steps = max(cfg.train.max_steps, warmup+measured+1)
            cfg.optimizer.schedule_steps = max(cfg.optimizer.schedule_steps, cfg.train.max_steps)
            cfg.research.max_seconds = cfg.research.max_train_flops = cfg.research.max_tokens = 0
            config_path = self.output / "configs" / f"profile-{job['id']}.yaml"
            config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
            self.command(["-m", "llm2", "profile", "--config", config_path,
                          "--output", self.output / "profiles" / job["id"], "--execute",
                          "--warmup-steps", str(warmup), "--measure-steps", str(measured)],
                         f"{job['id']}-profile.log", gpu=self.gpus[0])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true", help="explicitly run the selected campaign")
    result.add_argument("--config", default="configs/kaggle_t4_1h_baseline.yaml")
    result.add_argument("--data", help="existing prepared dataset; never overwritten")
    result.add_argument("--input", nargs="+", help="raw UTF-8 files; prepare new byte data inside this campaign")
    result.add_argument("--output", help="new campaign directory")
    result.add_argument("--models", nargs="+", choices=MODELS,
                        default=["baseline", "lrm", "flrm_reference", "flrm_batched"])
    result.add_argument("--seeds", nargs="+", type=int, default=[42])
    result.add_argument("--steps", type=int, default=2000)
    result.add_argument("--max-seconds", type=float, default=1800.)
    result.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    result.add_argument("--gpus", nargs="+", default=["0", "1"])
    result.add_argument("--profile", action="store_true")
    result.add_argument("--warmup-steps", type=int, default=3)
    result.add_argument("--measure-steps", type=int, default=5)
    result.add_argument("--no-benchmark", action="store_true")
    result.add_argument("--no-diagnostics", action="store_true")
    result.add_argument("--cpu-tests", action="store_true")
    result.add_argument("--cuda-tests", action="store_true")
    result.add_argument("--evaluate-checkpoints", nargs="*", default=[],
                        help="optional old checkpoints; original prepared data must match their fingerprints")
    return result


def run_campaign(args):
    root = Path(__file__).resolve().parents[3]
    if args.steps <= 0 or args.max_seconds < 0 or not args.seeds:
        raise ValueError("positive steps, nonnegative time budget and at least one seed required")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("models and seeds must not contain duplicates")
    if bool(args.data) == bool(args.input):
        raise ValueError("provide exactly one of --data or --input")
    base = load_config(Path(args.config) if Path(args.config).is_absolute() else root / args.config)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this campaign is meant for Kaggle T4x2")
        if len(set(args.gpus)) != len(args.gpus):
            raise ValueError("GPU slots must be distinct")
        if any(not g.isdecimal() or int(g) >= torch.cuda.device_count() for g in args.gpus):
            raise ValueError("GPU slots must be indices visible to the notebook process")
    if any(m.startswith("flrm") for m in args.models):
        from ..architectures import fractal_recursive  # Fails early on the common branch.
    if not args.execute:
        return {"status": "not_started", "models": args.models, "seeds": args.seeds,
                "next": "Add --execute to run exactly this selection."}
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output = Path(args.output).resolve() if args.output else root / "runs" / f"kaggle-{identifier}"
    campaign = Campaign(root, output, device=args.device, gpus=args.gpus)
    failure = None
    try:
        if args.data:
            data_path = Path(args.data).resolve()
        else:
            data_path = output / "data"
            prepare_data(args.input, data_path, kind="byte", seed=42)
        data = PreparedData(data_path)
        if min(len(x) for x in data.arrays.values()) <= base.data.seq_len:
            raise ValueError("dataset splits are too short for the selected context")
        campaign.manifest.update(dataset_sha256=data.fingerprint, dataset_path=str(data_path),
                                 dataset_manifest=data.manifest, status="checking")
        if data.tokenizer.vocab_size > base.model.vocab_size:
            raise ValueError("prepared tokenizer exceeds model vocabulary; choose a matching --config")
        if args.input:
            sources = output / "input_sources"
            sources.mkdir()
            campaign.manifest["raw_sources"] = []
            for index, raw_path in enumerate(map(Path, args.input)):
                entry = {"path": str(raw_path.resolve()), "sha256": sha256_file(raw_path),
                         "bytes": raw_path.stat().st_size, "bundled": raw_path.stat().st_size <= 2*1024*1024}
                if entry["bundled"]:
                    (sources / f"{index}-{raw_path.name}").write_bytes(raw_path.read_bytes())
                campaign.manifest["raw_sources"].append(entry)
        campaign.save()
        for index, checkpoint in enumerate(args.evaluate_checkpoints):
            destination = output / "existing_evaluations" / str(index)
            campaign.command(["-m", "llm2", "evaluate", "--checkpoint", checkpoint,
                              "--data", data_path, "--device", args.device, "--precision", "fp32",
                              "--output", destination / "language.json",
                              "--windows-output", destination / "windows.jsonl"],
                             f"existing-{index}-evaluate.log", gpu=campaign.gpus[0])
        if args.cpu_tests:
            campaign.command(["-m", "pytest", "-q", "-m", "not cuda"], "tests-cpu.log", gpu=campaign.gpus[0])
        if args.cuda_tests and args.device == "cuda":
            campaign.command(["-m", "pytest", "-q", "-m", "cuda"], "tests-cuda.log", gpu=campaign.gpus[0])
        (output / "configs").mkdir()
        jobs = campaign.manifest["runs"]
        for seed in args.seeds:
            for model in args.models:
                run_id = f"{model}-seed-{seed}"
                slot = campaign.gpus[len(jobs) % len(campaign.gpus)]
                directory = output / "runs" / run_id
                cfg = campaign_config(base, model, data_path, directory, seed=seed, steps=args.steps,
                                      device=args.device, max_seconds=args.max_seconds)
                path = output / "configs" / f"{run_id}.yaml"
                path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True), encoding="utf-8")
                jobs.append({"id": run_id, "model": model, "seed": seed, "physical_gpu": slot,
                             "directory": str(directory), "config": str(path), "status": "pending"})
        campaign.manifest["status"] = "training"
        campaign.save()
        with ThreadPoolExecutor(max_workers=len(campaign.gpus)) as workers:
            futures = [workers.submit(campaign.train_worker, jobs[i::len(campaign.gpus)], gpu)
                       for i, gpu in enumerate(campaign.gpus)]
            for future in futures:
                future.result()
        campaign.manifest["status"] = "evaluating"
        campaign.save()
        for job in jobs:
            campaign.evaluate_run(job, benchmark=not args.no_benchmark, diagnostics=not args.no_diagnostics)
        if len({job["validation_tokens"] for job in jobs}) != 1:
            raise ValueError("models were not evaluated on the same number of tokens")
        if args.profile:
            campaign.profile_runs(jobs, args.warmup_steps, args.measure_steps)
        from .reporting import generate_report
        generate_report([job["directory"] for job in jobs], output / "report")
        campaign.manifest["status"] = "complete"
    except Exception as exc:
        failure = exc
        campaign.abort()
        campaign.manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        campaign.save()
        archive = export_campaign(output)
        print(f"Campaign: {output}\nAnalysis ZIP: {archive}", flush=True)
    if failure is not None:
        raise RuntimeError(f"campaign failed; logs and partial results preserved in {archive}") from failure
    return {"status": "complete", "campaign": str(output), "results_zip": str(archive),
            "models": len(campaign.manifest["runs"])}


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(run_campaign(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
