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
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile

import torch
import yaml

from ..checkpoint import read_checkpoint_metadata
from ..config import ExperimentConfig, ResearchConfig, load_config
from ..data import PreparedData, prepare_data, sha256_file, write_json
from ..runtime import source_provenance
from .presentation import MODEL_NAMES, batch_comparison_text, comparison_text, legend, number, progress, stage
from .naming import numbered_output


MODELS = (
    "baseline", "baseline_small", "baseline_jepa",
    "lrm", "lrm_no_jepa", "lrm_no_jepa_k1", "lrm_no_jepa_k2",
    "flrm_reference", "flrm_batched",
    "flrm_compressed", "flrm_compressed_native",
    "flrm_compressed_native_no_jepa", "flrm_compressed_native_no_jepa_k1", "flrm_compressed_native_no_jepa_k2",
    "flrm_compressed_native_cached_no_jepa_k2",
    "flrm_compressed_grouped", "flrm_compressed_grouped_no_jepa", "flrm_no_jepa",
    "flrm_compressed_grouped_no_jepa_k1", "flrm_compressed_grouped_no_jepa_k2",
    "flrm_compressed_grouped_cached_no_jepa_k2",
    "flrm_fused_k2", "flrm_ncm_k2",
    "flrm_vcycle_k2", "flrm_vcycle_ncm_k2", "flrm_fused_ncm_k2", "flrm_inv_vcycle_ncm_k2",
)

FIXED_ITERATIONS = {
    "lrm_no_jepa_k1": 1,
    "lrm_no_jepa_k2": 2,
    "flrm_compressed_native_no_jepa_k1": 1,
    "flrm_compressed_native_no_jepa_k2": 2,
    "flrm_compressed_native_cached_no_jepa_k2": 2,
    "flrm_compressed_grouped_no_jepa_k1": 1,
    "flrm_compressed_grouped_no_jepa_k2": 2,
    "flrm_compressed_grouped_cached_no_jepa_k2": 2,
    "flrm_fused_k2": 2,
    "flrm_ncm_k2": 2,
    "flrm_vcycle_k2": 2,
    "flrm_vcycle_ncm_k2": 2,
    "flrm_fused_ncm_k2": 2,
    "flrm_inv_vcycle_ncm_k2": 2,
}
DEFAULT_MODELS = ["flrm_compressed_grouped_no_jepa_k1", "flrm_compressed_grouped_no_jepa_k2", "baseline_small"]


def worker_environment(root, gpu=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(root) / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if gpu is not None:
        if not str(gpu).isdecimal():
            raise ValueError(f"GPU must be a nonnegative logical index, got {gpu!r}")
        index = int(gpu)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            env["CUDA_VISIBLE_DEVICES"] = str(index)
        else:
            slots = [slot.strip() for slot in visible.split(",")]
            if any(not slot or slot == "-1" for slot in slots) or index >= len(slots):
                raise ValueError(
                    f"GPU logical index {index} is unavailable under "
                    f"CUDA_VISIBLE_DEVICES={visible!r}; choose an index within the parent mask"
                )
            env["CUDA_VISIBLE_DEVICES"] = slots[index]
    return env


def campaign_config(base, model, data, output, *, seed, steps, device, max_seconds, iterations=2,
                    batch_size=None, grad_accum_steps=None, compile=False, paired_common_init=False):
    if model not in MODELS:
        raise ValueError(f"unknown campaign model: {model}")
    cfg = ExperimentConfig.from_dict(base.to_dict())
    if batch_size is not None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        cfg.data.batch_size = batch_size
    if grad_accum_steps is not None:
        if type(grad_accum_steps) is not int or grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be a positive integer, got {grad_accum_steps!r}")
        cfg.train.grad_accum_steps = grad_accum_steps
    cfg.data.path, cfg.train.output_dir = str(Path(data).resolve()), str(Path(output).resolve())
    cfg.train.seed, cfg.train.max_steps = seed, steps
    cfg.optimizer.schedule_steps = max(cfg.optimizer.schedule_steps, steps)
    cfg.train.device, cfg.train.precision, cfg.train.deterministic = device, "fp32", device == "cpu"
    cfg.train.compile = bool(compile)
    cfg.train.console_progress = True
    cfg.train.accumulation_weighting = "tokens"
    cfg.train.checkpoint_interval = max(1, min(500, steps))
    architecture = "transformer" if model.startswith("baseline") else "recursive" if model.startswith("lrm") else "fractal"
    # Fixed-K aliases remain distinct jobs/reports but share exactly one backbone.
    selected = model.rsplit("_k", 1)[0] if model in FIXED_ITERATIONS else model
    effective_iterations = 1 if architecture == "transformer" else FIXED_ITERATIONS.get(model, iterations)
    variant = selected.removesuffix("_no_jepa").removesuffix("_cached")
    is_grouped_ablation = model in (
        "flrm_fused_k2", "flrm_ncm_k2",
        "flrm_vcycle_k2", "flrm_vcycle_ncm_k2", "flrm_fused_ncm_k2", "flrm_inv_vcycle_ncm_k2",
    )
    is_fused_film = model in ("flrm_fused_k2", "flrm_vcycle_k2", "flrm_vcycle_ncm_k2", "flrm_fused_ncm_k2", "flrm_inv_vcycle_ncm_k2")
    is_vcycle = model in ("flrm_vcycle_k2", "flrm_vcycle_ncm_k2")
    is_inv_vcycle = model == "flrm_inv_vcycle_ncm_k2"
    ncm_slots = 16 if model in ("flrm_ncm_k2", "flrm_vcycle_ncm_k2", "flrm_fused_ncm_k2", "flrm_inv_vcycle_ncm_k2") else 0
    is_cached_ctx = model in ("flrm_compressed_native_cached_no_jepa_k2", "flrm_compressed_grouped_cached_no_jepa_k2")
    uses_jepa = not is_grouped_ablation and not selected.endswith("_no_jepa") and (
        selected in ("baseline_jepa", "lrm", "flrm_reference", "flrm_batched") or selected.startswith("flrm_compressed"))
    jepa = 0.1 if uses_jepa else 0.
    scale_exec = (
        "compressed_grouped" if is_grouped_ablation
        else variant.removeprefix("flrm_") if variant.startswith("flrm_compressed")
        else "batched" if model == "flrm_batched"
        else "reference"
    )
    use_paired = bool(paired_common_init) and architecture in ("recursive", "fractal")
    cfg.research = ResearchConfig(
        architecture=architecture,
        iterations=effective_iterations,
        latent_steps=0,
        jepa_weight=jepa,
        scale_execution=scale_exec,
        v_cycle=is_vcycle,
        inv_vcycle=is_inv_vcycle,
        ncm_slots=ncm_slots,
        fused_film=is_fused_film,
        cache_context_projection=is_cached_ctx,
        paired_common_init=use_paired,
        max_seconds=max_seconds,
        shutdown_reserve_seconds=min(30., max_seconds / 10) if max_seconds else 0.,
    )
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

    def command(self, arguments, log_name, *, gpu=None, job=None):
        if self.cancelled.is_set():
            raise RuntimeError("campaign cancelled after another worker failed")
        path = self.output / "logs" / log_name
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, *map(str, arguments)]
        env = worker_environment(self.root, gpu)
        action = next((name for name in ("train", "evaluate", "benchmark", "generate", "diagnose", "profile")
                       if name in arguments), None)
        title = {"train": "Treinamento", "evaluate": "Validação completa (loss: menor = melhor)",
                 "benchmark": "Velocidade de geração (maior = melhor)", "generate": "Exemplo de texto gerado",
                 "diagnose": "Diagnóstico das representações", "profile": "Perfil isolado de velocidade/memória"}.get(action, "Verificação de testes")
        print(f"\n▶ {title} — {log_name}\n  Log técnico completo: {path}", flush=True)
        started = time.monotonic()
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
                        try:
                            metrics = json.loads(line[6:])
                        except (ValueError, TypeError):
                            continue  # The original line is already preserved in the technical log.
                        interval = job.get("progress_interval") or 500
                        if job and (metrics["step"] % interval == 0 or metrics["step"] >= job["max_steps"]):
                            print(progress(metrics, job, time.monotonic()-started), flush=True)
                    elif action == "generate" or re.search(r"\d+ (passed|failed|skipped|deselected)", line):
                        print(line.rstrip(), flush=True)
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
        print(f"✓ Etapa concluída: {log_name}", flush=True)

    def abort(self):
        self.cancelled.set()
        with self.lock:
            for process in tuple(self.processes):
                if process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        pass

    def train_worker(self, jobs, gpu):
        for job in jobs:
            if self.cancelled.is_set():
                return
            with self.lock:
                job["status"] = "training"
                self.save()
            try:
                self.command(["-m", "llm2", "train", "--config", job["config"]],
                             f"{job['id']}-train.log", gpu=gpu, job=job)
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
                with self.lock:
                    job.update(status="trained", checkpoint=str(checkpoint), stop_reason=summary["stop_reason"],
                               step=summary["step"], checkpoint_model_sha256=metadata["files"]["model.safetensors"])
                    self.save()
                reason = {"max_steps": "atingiu o número de passos planejado",
                          "requested_step": "atingiu o passo solicitado",
                          "time_budget": "atingiu o limite de tempo; encerramento antecipado",
                          "token_budget": "atingiu o orçamento de alvos; encerramento antecipado",
                          "learning_flop_budget": "atingiu o orçamento estimado de computação; encerramento antecipado",
                          }.get(summary["stop_reason"], summary["stop_reason"])
                print(f"✓ {MODEL_NAMES.get(job['model'], job['model'])}: {summary['step']} passos; {reason}.", flush=True)
            except Exception as exc:
                with self.lock:
                    job.update(status="failed", error=str(exc))
                    self.save()
                self.abort()
                raise

    def evaluate_run(self, job, *, benchmark=True, diagnostics=True, evaluate_all_checkpoints=False,
                     diagnose_steps=None, diagnose_batch_size=1):
        # Sequential post-training measurements on the SAME physical GPU for every model.
        gpu = self.gpus[0]
        final_checkpoint = Path(job["checkpoint"]).resolve()
        run_dir = Path(job["directory"])
        ckpt_dir = run_dir / "checkpoints"
        if evaluate_all_checkpoints and ckpt_dir.is_dir():
            checkpoints = sorted(
                (p.resolve() for p in ckpt_dir.glob("step-*") if p.is_dir()),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if final_checkpoint not in checkpoints:
                checkpoints.append(final_checkpoint)
        else:
            checkpoints = [final_checkpoint]

        diag_step_set = None if diagnose_steps is None else {int(s) for s in diagnose_steps}
        checkpoint_evaluations = []

        for ckpt in checkpoints:
            step_num = int(ckpt.name.split("-")[-1])
            is_final = (ckpt == final_checkpoint)
            meta = read_checkpoint_metadata(ckpt)
            destination = run_dir / "evaluations" / ckpt.name
            destination.mkdir(parents=True, exist_ok=False)
            common = ["--checkpoint", str(ckpt), "--device", self.device, "--precision", "fp32"]
            self.command(["-m", "llm2", "evaluate", *common, "--output", destination / "language.json",
                          "--windows-output", destination / "windows.jsonl"],
                         f"{job['id']}-{ckpt.name}-evaluate.log" if not is_final else f"{job['id']}-evaluate.log",
                         gpu=gpu)
            result = json.loads((destination / "language.json").read_text(encoding="utf-8"))
            if result["validation_scope"] != "all_full_windows" or result["dataset_sha256"] != self.manifest["dataset_sha256"]:
                raise ValueError("evaluation scope/dataset mismatch")
            if result["checkpoint_model_sha256"] != meta["files"]["model.safetensors"]:
                raise ValueError("evaluation belongs to different weights")
            if is_final and result["checkpoint_model_sha256"] != job["checkpoint_model_sha256"]:
                raise ValueError("final evaluation belongs to different weights")
            print(f"  ✓ [Passo {step_num}] Loss LM COMPLETA: {number(result['validation_loss'], 6)} (menor = melhor); "
                  f"{number(result['validation_tokens'], 0)} alvos avaliados.", flush=True)
            if is_final and benchmark:
                self.command(["-m", "llm2", "benchmark", *common, "--prompt", "First Citizen:",
                              "--warmup-tokens", "8", "--max-new-tokens", "64",
                              "--output", destination / "benchmark.json"],
                             f"{job['id']}-benchmark.log", gpu=gpu)
                self.command(["-m", "llm2", "generate", *common, "--prompt", "First Citizen:",
                              "--seed", "42", "--temperature", "0.8", "--max-new-tokens", "64"],
                             f"{job['id']}-generation.log", gpu=gpu)
            should_diagnose = diagnostics and (is_final if diag_step_set is None else (step_num in diag_step_set))
            if should_diagnose:
                diag_cmd = ["-m", "llm2", "diagnose", *common, "--samples", "128",
                            "--batch-size", str(diagnose_batch_size),
                            "--output", destination / "diagnostics.json"]
                self.command(diag_cmd,
                             f"{job['id']}-{ckpt.name}-diagnostics.log" if not is_final else f"{job['id']}-diagnostics.log",
                             gpu=gpu)
            checkpoint_evaluations.append({
                "step": step_num,
                "checkpoint": str(ckpt),
                "checkpoint_model_sha256": meta["files"]["model.safetensors"],
                "validation_loss": result["validation_loss"],
                "validation_tokens": result["validation_tokens"],
                "evaluation": str(destination / "language.json"),
                "diagnostics": str(destination / "diagnostics.json") if should_diagnose else None,
            })
            if is_final:
                job.update(status="complete", evaluation=str(destination / "language.json"),
                           full_validation_loss=result["validation_loss"], validation_tokens=result["validation_tokens"],
                           evaluation_physical_gpu=worker_environment(self.root, gpu).get("CUDA_VISIBLE_DEVICES"))
        job["checkpoint_evaluations"] = checkpoint_evaluations
        self.save()

    def profile_runs(self, jobs, warmup, measured):
        selected = jobs
        # One seed per execution mode and batch format, never overlapping GPU profiles.
        seen = set()
        for job in selected:
            cfg = load_config(job["config"])
            profile_key = (job["model"], cfg.data.batch_size, cfg.train.grad_accum_steps)
            if profile_key in seen:
                continue
            seen.add(profile_key)
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
            profile_path = self.output / "profiles" / job["id"] / "profile.json"
            timing = json.loads(profile_path.read_text(encoding="utf-8"))["timing"]
            job["profile"] = str(profile_path)
            self.save()
            print(f"  {MODEL_NAMES.get(job['model'], job['model'])}: "
                  f"{number(timing['mean_tokens_per_second'], 0)} alvos/s (maior = melhor); "
                  f"{measured} amostras isoladas, após {warmup} passos de aquecimento.", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true", help="explicitly run the selected campaign")
    result.add_argument("--config", default="configs/kaggle_t4_1h_baseline.yaml")
    result.add_argument("--data", help="existing prepared dataset; never overwritten")
    result.add_argument("--input", nargs="+", help="raw UTF-8 files; prepare new byte data inside this campaign")
    result.add_argument("--output", help="new campaign directory")
    result.add_argument("--models", nargs="+", choices=MODELS,
                        default=DEFAULT_MODELS)
    result.add_argument("--progress-interval", type=int, default=500)
    result.add_argument("--seeds", nargs="+", type=int, default=[42])
    result.add_argument("--steps", type=int, default=2000)
    result.add_argument("--iterations", type=int, default=2)
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
    result.add_argument("--review-campaign", help="evaluate saved checkpoints only; never starts training")
    result.add_argument("--evaluation-root", help="original verified source checkout for saved checkpoints")
    result.add_argument("--review-steps", nargs="+", type=int, default=[500, 1000, 1500, 2000])
    result.add_argument("--batch-size", type=int, default=None, help="override per-device microbatch size")
    result.add_argument("--grad-accum-steps", type=int, default=None, help="override gradient accumulation steps")
    result.add_argument("--compile", action="store_true", help="enable torch.compile for training")
    result.add_argument("--profile-batches", action="store_true", help="run batch matrix profiling study")
    result.add_argument("--batch-matrix", nargs="+", default=["8x4", "16x2", "32x1"],
                        help="batch matrix combinations in BxA format (microbatch x accum)")
    result.add_argument("--profile-backends", nargs="+", choices=("eager", "inductor"), default=None,
                        help="backends to profile in batch profile mode (defaults tocompile flag)")
    result.add_argument("--profile-repeats", type=int, default=1,
                        help="number of alternating-order repetitions per backend in batch profile mode")
    result.add_argument("--diagnose-init", action="store_true",
                        help="run Etapa 1 non-invasive magnitude/gradient diagnostics at random initialization")
    result.add_argument("--paired-common-init", action="store_true",
                        help="initialize common parameters identically across LRM and FLRM per seed")
    result.add_argument("--evaluate-all-checkpoints", action="store_true",
                        help="evaluate all saved milestone checkpoints (e.g. 500, 1000, 1500, 2000) on full validation")
    result.add_argument("--diagnose-steps", nargs="+", type=int, default=None,
                        help="specific checkpoint steps to run diagnostics on (defaults to final step only)")
    result.add_argument("--diagnose-batch-size", type=int, default=1,
                        help="number of contiguous validation windows in the fixed diagnostic batch")
    return result


def parse_batch_matrix(matrix):
    combinations = []
    for item in matrix:
        match = re.fullmatch(r"(\d+)[xX*](\d+)", item.strip())
        if not match:
            raise ValueError(f"Formato de lote inválido: {item!r}. Use o formato BxA (ex: '8x4', '16x2', '32x1').")
        b, a = int(match.group(1)), int(match.group(2))
        if b <= 0 or a <= 0:
            raise ValueError(f"Lote e acumulação devem ser positivos: {item!r}")
        combinations.append((b, a))
    return combinations


def run_batch_profile(args):
    root = Path(__file__).resolve().parents[3]
    if args.warmup_steps < 0 or args.measure_steps < 1:
        raise ValueError("profiling requires nonnegative warmup and at least one measured step")
    if getattr(args, "profile_repeats", 1) < 1:
        raise ValueError("profile_repeats must be at least 1")
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must not contain duplicates")
    if bool(args.data) == bool(args.input):
        raise ValueError("provide exactly one of --data or --input")
    batch_matrix = parse_batch_matrix(args.batch_matrix)
    backends = getattr(args, "profile_backends", None)
    if not backends:
        backends = ["inductor" if getattr(args, "compile", False) else "eager"]
    repeats = getattr(args, "profile_repeats", 1)
    base = load_config(Path(args.config) if Path(args.config).is_absolute() else root / args.config)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this profile is meant for Kaggle T4x2")
    if any(m.startswith("flrm") for m in args.models):
        from ..architectures import fractal_recursive
    if not args.execute:
        return {"status": "not_started", "models": args.models, "batch_matrix": args.batch_matrix,
                "backends": backends, "repeats": repeats,
                "next": "Add --execute to run exactly this batch profile."}
    output = Path(args.output).resolve() if args.output else numbered_output(root / "runs")
    campaign = Campaign(root, output, device=args.device, gpus=args.gpus)
    legend()
    stage(1, 4, "PREPARAR E CONFERIR OS DADOS")
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

    stage(2, 4, "TESTES DE INTEGRAÇÃO E TELEMETRIA DE INICIALIZAÇÃO (ETAPA 1)")
    if args.cpu_tests:
        campaign.command(["-m", "pytest", "-q", "-m", "not cuda"], "tests-cpu.log", gpu=campaign.gpus[0])
    if args.cuda_tests and args.device == "cuda":
        campaign.command(["-m", "pytest", "-q", "-m", "cuda", "-k", "fp32"],
                         "tests-cuda-fp32.log", gpu=campaign.gpus[0])

    (output / "configs").mkdir(parents=True, exist_ok=True)
    (output / "profiles").mkdir(parents=True, exist_ok=True)
    gpu_slot = campaign.gpus[0]

    init_diagnostics = []
    if getattr(args, "diagnose_init", False):
        (output / "diagnostics").mkdir(parents=True, exist_ok=True)
        diag_bs = getattr(args, "diagnose_batch_size", 1)
        for seed in args.seeds:
            for model in args.models:
                diag_id = f"init-{model}-seed-{seed}"
                cfg = campaign_config(base, model, data_path, output / "runs" / diag_id, seed=seed,
                                      steps=max(100, args.warmup_steps + args.measure_steps + 1),
                                      device=args.device, max_seconds=0, iterations=args.iterations,
                                      batch_size=batch_matrix[0][0], grad_accum_steps=batch_matrix[0][1],
                                      compile=False,
                                      paired_common_init=getattr(args, "paired_common_init", False))
                cfg_path = output / "configs" / f"{diag_id}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True), encoding="utf-8")
                diag_out = output / "diagnostics" / f"{diag_id}.json"
                campaign.command(["-m", "llm2", "diagnose", "--config", cfg_path,
                                  "--data", data_path, "--device", args.device, "--precision", "fp32",
                                  "--samples", "128", "--batch-size", str(diag_bs), "--output", diag_out],
                                 f"{diag_id}-diagnose.log", gpu=gpu_slot)
                init_diagnostics.append({"model": model, "seed": seed, "path": str(diag_out)})
        campaign.manifest["init_diagnostics"] = init_diagnostics

    for index, checkpoint in enumerate(getattr(args, "evaluate_checkpoints", []) or []):
        dest_dir = output / "checkpoint_diagnostics" / str(index)
        dest_dir.mkdir(parents=True, exist_ok=True)
        campaign.command(["-m", "llm2", "diagnose", "--checkpoint", checkpoint,
                          "--data", data_path, "--device", args.device, "--precision", "fp32",
                          "--samples", "128", "--output", dest_dir / "diagnostics.json"],
                         f"ckpt-{index}-diagnose.log", gpu=gpu_slot)

    stage(3, 4, "PERFIL DE MATRIZ DE LOTES ISOLADO (GPU 0)")
    results = []
    base_plan = [(model, b, a) for model in args.models for b, a in batch_matrix]
    total_profiles = len(base_plan) * len(backends) * repeats
    count = 0
    for backend in backends:
        use_compile = backend == "inductor"
        for rep in range(1, repeats + 1):
            # Alternate execution order across repetitions to control for thermal/cache drift on GPU 0.
            ordered_plan = base_plan if (rep % 2 == 1) else list(reversed(base_plan))
            for model, b, a in ordered_plan:
                count += 1
                suffix = f"-{backend}-r{rep}" if (len(backends) > 1 or repeats > 1) else ""
                run_id = f"{model}-b{b}a{a}{suffix}"
                cfg = campaign_config(base, model, data_path, output / "runs" / run_id, seed=args.seeds[0],
                                      steps=max(100, args.warmup_steps + args.measure_steps + 1),
                                      device=args.device, max_seconds=0, iterations=args.iterations,
                                      batch_size=b, grad_accum_steps=a, compile=use_compile)
                cfg.train.console_progress = False
                cfg.optimizer.schedule_steps = max(cfg.optimizer.schedule_steps, cfg.train.max_steps)
                cfg.research.max_seconds = cfg.research.max_train_flops = cfg.research.max_tokens = 0
                cfg_path = output / "configs" / f"profile-{run_id}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True), encoding="utf-8")
                dest = output / "profiles" / run_id
                print(
                    f"[{count}/{total_profiles}] Medindo {MODEL_NAMES.get(model, model)} | "
                    f"Backend {backend} (rep {rep}/{repeats}) | Lote {b}×{a} ({b*a} seqs/passo)...",
                    flush=True,
                )
                campaign.command(["-m", "llm2", "profile", "--config", cfg_path,
                                  "--output", dest, "--execute",
                                  "--warmup-steps", str(args.warmup_steps),
                                  "--measure-steps", str(args.measure_steps)],
                                 f"{run_id}-profile.log", gpu=gpu_slot)
                profile_data = json.loads((dest / "profile.json").read_text(encoding="utf-8"))
                timing = profile_data["timing"]
                warmup_timing = profile_data.get("warmup_timing", {})
                step_metrics = profile_data.get("step_metrics", {})
                peak_vram = step_metrics.get("peak_vram_allocated_bytes")
                peak_vram_res = step_metrics.get("peak_vram_reserved_bytes")
                entry = {
                    "model": model,
                    "backend": backend,
                    "compile": use_compile,
                    "repeat": rep,
                    "batch_size": b,
                    "grad_accum_steps": a,
                    "effective_sequences": b * a,
                    "tokens_per_second": timing.get("mean_tokens_per_second"),
                    "median_tokens_per_second": timing.get("median_tokens_per_second"),
                    "mean_step_seconds": timing.get("mean_step_seconds"),
                    "median_step_seconds": timing.get("median_step_seconds"),
                    "stdev_step_seconds": timing.get("stdev_step_seconds"),
                    "first_warmup_step_seconds": warmup_timing.get("first_step_seconds"),
                    "total_warmup_seconds": warmup_timing.get("total_warmup_seconds"),
                    "peak_vram_allocated_bytes": peak_vram,
                    "peak_vram_reserved_bytes": peak_vram_res,
                    "cuda_launch_kernel_calls": profile_data.get("cuda_launch_kernel_calls"),
                    "analytical_step_flops_estimate": profile_data.get("analytical_step_flops_estimate"),
                    "profile_path": str(dest / "profile.json"),
                }
                results.append(entry)
                ms_val = entry["mean_step_seconds"] * 1000 if entry["mean_step_seconds"] is not None else None
                vram_mib = peak_vram / (1024 * 1024) if peak_vram is not None else None
                print(f"  ✓ {MODEL_NAMES.get(model, model)} [{backend} | r{rep} | {b}×{a}]: "
                      f"{number(entry['tokens_per_second'], 0)} alvos/s, "
                      f"{number(ms_val, 1)} ms/passo, "
                      f"VRAM alocada {number(vram_mib, 1)} MiB", flush=True)

    stage(4, 4, "TABELA COMPARATIVA DE LOTES — THROUGHPUT, TEMPO E MEMÓRIA")
    report_text = batch_comparison_text(results)
    print(report_text, flush=True)
    write_json(output / "batch_profiles.json", results)
    (output / "batch_profile_report.md").write_text(report_text, encoding="utf-8")
    campaign.manifest.update(status="complete", batch_profiles=results)
    campaign.save()
    archive = export_campaign(output)
    print("\nCONCLUÍDO. Baixe este ZIP e envie para análise:")
    print(archive)
    return {"status": "complete", "results": results, "archive": str(archive)}


def run_campaign(args):
    if getattr(args, "profile_batches", False):
        return run_batch_profile(args)
    root = Path(__file__).resolve().parents[3]
    if args.steps <= 0 or args.max_seconds < 0 or not args.seeds:
        raise ValueError("positive steps, nonnegative time budget and at least one seed required")
    if args.profile and (args.warmup_steps < 0 or args.measure_steps < 1):
        raise ValueError("profiling requires nonnegative warmup and at least one measured step")
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
    output = Path(args.output).resolve() if args.output else numbered_output(root / "runs")
    campaign = Campaign(root, output, device=args.device, gpus=args.gpus)
    failure = None
    try:
        legend()
        stage(1, 7, "PREPARAR E CONFERIR OS DADOS")
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
        stage(2, 7, "TESTES DE CORREÇÃO — CPU E GPU")
        if not (args.cpu_tests or args.cuda_tests):
            print("Testes não solicitados nesta execução.", flush=True)
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
            # This release targets T4 FP32. Do not gate it on BF16 emulation/native availability.
            campaign.command(["-m", "pytest", "-q", "-m", "cuda", "-k", "fp32"],
                             "tests-cuda-fp32.log", gpu=campaign.gpus[0])
        (output / "configs").mkdir()
        jobs = campaign.manifest["runs"]
        use_paired_init = getattr(args, "paired_common_init", False)
        for seed in args.seeds:
            for model in args.models:
                run_id = f"{model}-seed-{seed}"
                slot = campaign.gpus[len(jobs) % len(campaign.gpus)]
                directory = output / "runs" / run_id
                cfg = campaign_config(base, model, data_path, directory, seed=seed, steps=args.steps,
                                      device=args.device, max_seconds=args.max_seconds, iterations=args.iterations,
                                      batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
                                      compile=getattr(args, "compile", False),
                                      paired_common_init=use_paired_init)
                path = output / "configs" / f"{run_id}.yaml"
                path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True), encoding="utf-8")
                jobs.append({"id": run_id, "model": model, "seed": seed, "gpu_slot": slot,
                             "max_steps": cfg.train.max_steps, "checkpoint_interval": cfg.train.checkpoint_interval, "progress_interval": args.progress_interval,
                             "iterations": cfg.research.iterations+cfg.research.latent_steps,
                             "batch_size": cfg.data.batch_size, "grad_accum_steps": cfg.train.grad_accum_steps,
                             "paired_common_init": bool(cfg.research.paired_common_init),
                             "physical_gpu": worker_environment(root, slot).get("CUDA_VISIBLE_DEVICES"),
                             "directory": str(directory), "config": str(path), "status": "pending"})
        if use_paired_init:
            from .factory import build_backbone
            from ..runtime import setup_runtime
            paired_verification = {}
            for seed in args.seeds:
                seed_jobs = [j for j in jobs if j["seed"] == seed and j["paired_common_init"]]
                if not seed_jobs:
                    continue
                hashes = {}
                exclusive_hashes = {}
                for j in seed_jobs:
                    j_cfg = load_config(j["config"])
                    j_cfg.train.device = "cpu"
                    j_cfg.train.__post_init__()
                    setup_runtime(j_cfg.train)
                    m_inst = build_backbone(j_cfg)
                    prov = getattr(m_inst, "paired_init_provenance", None)
                    if not prov:
                        raise RuntimeError(f"paired_init_provenance missing for {j['id']}")
                    hashes[j["model"]] = prov["common_combined_sha256"]
                    if prov.get("exclusive_combined_sha256"):
                        exclusive_hashes[j["model"]] = prov["exclusive_combined_sha256"]
                unique_common = set(hashes.values())
                if len(unique_common) != 1:
                    raise RuntimeError(f"paired_common_init verification failed for seed {seed}: {hashes}")
                paired_verification[str(seed)] = {
                    "verified_equal_common_weights": True,
                    "common_combined_sha256": next(iter(unique_common)),
                    "models_verified": list(hashes.keys()),
                    "exclusive_combined_sha256_by_model": exclusive_hashes,
                }
                print(f"  ✓ Pareamento de pesos comuns verificado para seed {seed}: SHA256={next(iter(unique_common))[:16]}... ({len(hashes)} modelos)", flush=True)
            campaign.manifest["paired_init_verification"] = paired_verification
        if getattr(args, "diagnose_init", False):
            (output / "diagnostics").mkdir(parents=True, exist_ok=True)
            diag_bs = getattr(args, "diagnose_batch_size", 1)
            init_diagnostics = []
            for job in jobs:
                diag_out = output / "diagnostics" / f"init-{job['id']}.json"
                campaign.command(["-m", "llm2", "diagnose", "--config", job["config"],
                                  "--data", data_path, "--device", args.device, "--precision", "fp32",
                                  "--samples", "128", "--batch-size", str(diag_bs), "--output", diag_out],
                                 f"init-{job['id']}-diagnose.log", gpu=campaign.gpus[0])
                init_diagnostics.append({"job_id": job["id"], "model": job["model"], "seed": job["seed"], "path": str(diag_out)})
            campaign.manifest["init_diagnostics"] = init_diagnostics
        campaign.manifest["status"] = "training"
        campaign.save()
        stage(3, 7, "TREINAMENTO — MODELOS DISTRIBUÍDOS ENTRE AS GPUs")
        for job in jobs:
            print(f"  {MODEL_NAMES.get(job['model'], job['model'])} / seed {job['seed']} → "
                  f"{'GPU física '+str(job['physical_gpu']) if args.device == 'cuda' else 'CPU'}", flush=True)
        print(f"Progresso nos marcos principais (a cada {args.progress_interval} passos, onde os checkpoints são salvos no disco).", flush=True)
        with ThreadPoolExecutor(max_workers=len(campaign.gpus)) as workers:
            futures = [workers.submit(campaign.train_worker, jobs[i::len(campaign.gpus)], gpu)
                       for i, gpu in enumerate(campaign.gpus)]
            for future in futures:
                future.result()
        campaign.manifest["status"] = "evaluating"
        campaign.save()
        stage(4, 7, "VALIDAÇÃO COMPLETA E GERAÇÃO — UMA GPU POR VEZ")
        print("A segunda GPU pode ficar ociosa nesta etapa. Isso é esperado.", flush=True)
        for index, job in enumerate(jobs, 1):
            print(f"\nModelo {index}/{len(jobs)}: {MODEL_NAMES.get(job['model'], job['model'])} / seed {job['seed']}", flush=True)
            campaign.evaluate_run(
                job,
                benchmark=not args.no_benchmark,
                diagnostics=not args.no_diagnostics,
                evaluate_all_checkpoints=getattr(args, "evaluate_all_checkpoints", False),
                diagnose_steps=getattr(args, "diagnose_steps", None),
                diagnose_batch_size=getattr(args, "diagnose_batch_size", 1),
            )
        if len({job["validation_tokens"] for job in jobs}) != 1:
            raise ValueError("models were not evaluated on the same number of tokens")
        stage(5, 7, "PERFIL DE DESEMPENHO ISOLADO")
        if args.profile:
            campaign.profile_runs(jobs, args.warmup_steps, args.measure_steps)
        else:
            print("Perfil não solicitado nesta execução.", flush=True)
        stage(6, 7, "COMPARAÇÃO FINAL — MÉTRICAS EXPLICADAS")
        from .reporting import generate_report
        generate_report([job["directory"] for job in jobs], output / "report")
        rows = []
        for job in jobs:
            directory = Path(job["directory"])
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            benchmark_path = Path(job["evaluation"]).parent / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else {}
            rows.append({**summary, "model": job["model"], "seed": job["seed"],
                         "validation_loss": job["full_validation_loss"],
                         "generation_tokens_per_second": benchmark.get("tokens_per_second")})
        text = comparison_text(rows)
        (output / "report/resumo-pt.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        campaign.manifest["status"] = "complete"
    except Exception as exc:
        failure = exc
        campaign.abort()
        campaign.manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        stage(7, 7, "EXPORTAR RESULTADOS" if failure is None else "EXPORTAR RESULTADOS PARCIAIS — HOUVE UMA FALHA")
        campaign.save()
        archive = export_campaign(output)
        print(f"Pasta da campanha: {output}\nZIP para enviar à análise: {archive}", flush=True)
    if failure is not None:
        raise RuntimeError(f"campaign failed; logs and partial results preserved in {archive}") from failure
    return {"status": "complete", "campaign": str(output), "results_zip": str(archive),
            "models": len(campaign.manifest["runs"])}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.review_campaign:
            from .checkpoint_review import review_checkpoints
            result = review_checkpoints(args)
        else:
            result = run_campaign(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"\n❌ PROCESSO INTERROMPIDO: {exc}\nConfira o motivo e os logs indicados acima.", flush=True)
        raise
    if result["status"] == "complete":
        print("\n✓ PROCESSO CONCLUÍDO COM SUCESSO. O ZIP de análise está indicado acima.", flush=True)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
