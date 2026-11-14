"""Dual-Battery (TinyStories + byte-binding-v2) & Compiled Decode benchmark for IA2 candidates."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch

from ..architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from ..checkpoint import read_checkpoint_metadata
from ..config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from ..runtime import environment
from .binding_worker import BindingTrainer, score as score_binding
from .decode_cache import RecurrentDecode, generate, verify_logits
from .decode_optimization import CompiledDecode, LocalTransformerDecode
from .stories_data import prepare as prepare_stories
from .stories_worker import StoryTrainer, score as score_stories


RATES = (0.0003, 0.001, 0.003)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    group: str  # "baseline_1x" | "baseline_2x" | "candidate" | "recurrent_ref"
    architecture: str
    d_model: int
    n_layers: int
    n_heads: int
    hidden_dim: int
    shared: bool = True
    beta_scale: float = 2.0


ARMS = (
    ArmSpec("transformer_1x", "baseline_1x", "transformer", d_model=64, n_layers=2, n_heads=4, hidden_dim=96),
    ArmSpec("fractal_delta_1x", "candidate", "fractal_delta", d_model=80, n_layers=4, n_heads=4, hidden_dim=72, shared=True, beta_scale=2.0),
    ArmSpec("transformer_2x", "baseline_2x", "transformer", d_model=80, n_layers=3, n_heads=4, hidden_dim=128),
    ArmSpec("context_shared2", "recurrent_ref", "context_memory", d_model=80, n_layers=2, n_heads=4, hidden_dim=144, shared=True),
    ArmSpec("fractal_delta_med", "candidate", "fractal_delta", d_model=104, n_layers=4, n_heads=4, hidden_dim=128, shared=True, beta_scale=2.0),
)


def _build_research_config(arm: ArmSpec, *, smoke=False):
    if arm.architecture == "transformer":
        return ResearchConfig(architecture="transformer", scale_conditioning=False)
    if arm.architecture == "context_memory":
        return ResearchConfig(
            architecture="context_memory",
            share_weights=arm.shared,
            scale_conditioning=False,
            memory_dim=4 if smoke else 16,
            state_chunk_size=32,
        )
    return ResearchConfig(
        architecture=arm.architecture,
        share_scales=arm.shared,
        scale_conditioning=False,
        memory_dim=8 if smoke else 64,
        state_execution="chunked",
        state_chunk_size=32,
        fractal_beta_scale=arm.beta_scale,
    )


def make_story_config(arm: ArmSpec, data_root, output_dir, *, seed=341, lr=0.001, smoke=False, device="cuda", calibration=False):
    steps = (2 if calibration else 4) if smoke else (400 if calibration else 6000)
    d = 16 if smoke else arm.d_model
    ff = 24 if smoke else arm.hidden_dim
    heads = 2 if smoke else arm.n_heads
    research = _build_research_config(arm, smoke=smoke)
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=257,
            d_model=d,
            n_layers=arm.n_layers,
            n_heads=heads,
            hidden_dim=ff,
            max_seq_len=64 if smoke else 8192,
            dropout=0.05,
        ),
        data=DataConfig(
            path=str(Path(data_root).resolve()),
            seq_len=32 if smoke else 256,
            batch_size=2 if smoke else 32,
            eval_batch_size=2 if smoke else 32,
        ),
        optimizer=OptimizerConfig(
            lr=lr,
            weight_decay=0.01,
            warmup_steps=1 if smoke else (50 if calibration else 100),
            schedule_steps=steps,
        ),
        train=TrainConfig(
            seed=931 if calibration else seed,
            device=device,
            precision="fp32",
            deterministic=device == "cpu",
            compile=not smoke,
            max_steps=steps,
            grad_accum_steps=2 if smoke else 1,
            eval_interval=steps if calibration else max(1, steps // 4),
            checkpoint_interval=steps if calibration else max(1, steps // 2),
            log_interval=1 if smoke else 300,
            output_dir=str(Path(output_dir).resolve()),
            cpu_threads=2,
            accumulation_weighting="tokens",
            console_progress=not calibration,
        ),
        research=research,
    )


def make_binding_config(arm: ArmSpec, output_dir, *, seed=311, lr=0.003, smoke=False, device="cuda"):
    steps = 4 if smoke else 3000
    d = 16 if smoke else arm.d_model
    ff = 24 if smoke else arm.hidden_dim
    heads = 2 if smoke else arm.n_heads
    research = _build_research_config(arm, smoke=smoke)
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=257,
            d_model=d,
            n_layers=arm.n_layers,
            n_heads=heads,
            hidden_dim=ff,
            max_seq_len=64 if smoke else 8192,
            dropout=0.05,
        ),
        data=DataConfig(
            path=str(Path(output_dir).resolve() / "procedural-data-not-materialized"),
            seq_len=32 if smoke else 256,
            batch_size=2 if smoke else 16,
            eval_batch_size=1,
        ),
        optimizer=OptimizerConfig(
            lr=lr,
            weight_decay=0.01,
            warmup_steps=1 if smoke else 50,
            schedule_steps=steps,
        ),
        train=TrainConfig(
            seed=seed,
            device=device,
            precision="fp32",
            deterministic=device == "cpu",
            compile=not smoke,
            max_steps=steps,
            grad_accum_steps=2 if smoke else 1,
            eval_interval=max(1, steps // 4),
            checkpoint_interval=max(1, steps // 2),
            log_interval=1 if smoke else 300,
            eval_batches=2,
            output_dir=str(Path(output_dir).resolve()),
            cpu_threads=2,
            accumulation_weighting="tokens",
            console_progress=True,
        ),
        research=research,
    )


def select_story_lr(arm: ArmSpec, data_root, cal_root: Path, *, smoke=False, device="cuda"):
    selection_file = cal_root / arm.name / "selection.json"
    if selection_file.exists():
        return json.loads(selection_file.read_text(encoding="utf-8"))
    trials = []
    for lr in RATES:
        cfg = make_story_config(arm, data_root, cal_root / arm.name / str(lr), lr=lr, smoke=smoke, device=device, calibration=True)
        with StoryTrainer(cfg, calibration=True, smoke=smoke) as trainer:
            trainer.fit()
            nll = float(trainer.evaluations[-1]["nll"])
            trials.append({"lr": lr, "nll": nll})
            ckpt = Path(trainer.last_checkpoint)
            meta = read_checkpoint_metadata(ckpt)
            for fname in meta["files"]:
                p = ckpt / fname
                if p.is_file() and not p.is_symlink():
                    p.unlink()
    best = min(trials, key=lambda t: (t["nll"], t["lr"]))
    result = {"lr": best["lr"], "trials": trials}
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    selection_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def benchmark_decode(model, device, *, smoke=False):
    was_training = model.training
    model.eval()
    prefix_len = 24 if smoke else 7936
    new_bytes = 8 if smoke else 256
    span = 16 if smoke else 256
    gen = torch.Generator().manual_seed(91127)
    prefix = torch.randint(0, 256, (1, prefix_len), generator=gen).to(device)
    cont = torch.randint(0, 256, (1, 12 if smoke else 32), generator=gen).to(device)
    try:
        if type(model).__name__ == "TransformerLM":
            eager = LocalTransformerDecode(model, span=span)
        elif type(model) is FractalDeltaMemory:
            eager = FractalDecode(model)
        else:
            eager = RecurrentDecode(model)
        results = {}
        for mode_name, use_compiled in (("eager", False), ("inductor", True)):
            decoder = CompiledDecode(eager, backend="inductor") if use_compiled else eager
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            parity = verify_logits(decoder, prefix, cont)
            first_step_wall = time.perf_counter() - t0
            timing, _ = generate(decoder, prefix, new_bytes)
            peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            results[mode_name] = {
                "prefix_bytes": prefix_len,
                "new_bytes": new_bytes,
                "parity_max_abs_err": parity["max_abs_logit_error"],
                "decode_seconds": timing["decode_seconds"],
                "first_byte_seconds": timing["first_byte_seconds"],
                "total_seconds": timing["total_seconds"],
                "bytes_per_sec": (new_bytes - 1) / max(timing["decode_seconds"], 1e-9),
                "state_bytes": timing["state_bytes"],
                "peak_vram_bytes": peak_vram,
                "warmup_parity_seconds": first_step_wall,
                "cudagraph": getattr(decoder, "use_cudagraph", False),
            }
        return results
    finally:
        model.train(was_training)


def run_arm(arm: ArmSpec, data_root, work_root: Path, *, smoke=False, device="cuda"):
    print(f"\n=== [{arm.name}] ({arm.group}) starting on {device} ===", flush=True)
    arm_root = work_root / arm.name
    arm_root.mkdir(parents=True, exist_ok=True)
    cal = select_story_lr(arm, data_root, work_root / "calibration", smoke=smoke, device=device)
    print(f"[{arm.name}] calibrated LR = {cal['lr']} (trials={cal['trials']})", flush=True)

    story_cfg = make_story_config(arm, data_root, arm_root / "stories", lr=cal["lr"], smoke=smoke, device=device)
    with StoryTrainer(story_cfg, smoke=smoke) as trainer:
        params = trainer.model_parameters
        story_t0 = time.perf_counter()
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        trainer.fit()
        story_wall = time.perf_counter() - story_t0
        story_vram = torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None
        val_256 = score_stories(trainer.model, trainer.data, "val", 32 if smoke else 256, 2 if smoke else 32, trainer.device)
        long_len = 48 if smoke else 3219
        try:
            val_long = score_stories(trainer.model, trainer.data, "val", long_len, 1 if smoke else 4, trainer.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            val_long = {"nll": None, "error": "CUDA OOM"}
        decode_stats = benchmark_decode(trainer.model, trainer.device, smoke=smoke)
        print(
            f"[{arm.name}] TinyStories params={params} | val@256={val_256['nll']:.4f} | val@{long_len}={val_long.get('nll')} | "
            f"decode_inductor={decode_stats['inductor']['bytes_per_sec']:.1f} B/s",
            flush=True,
        )

    bind_cfg = make_binding_config(arm, arm_root / "binding", lr=cal["lr"], smoke=smoke, device=device)
    with BindingTrainer(bind_cfg, smoke=smoke) as btrainer:
        bind_t0 = time.perf_counter()
        if btrainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(btrainer.device)
        btrainer.fit()
        bind_wall = time.perf_counter() - bind_t0
        bind_vram = torch.cuda.max_memory_allocated(btrainer.device) if btrainer.device.type == "cuda" else None
        lengths = (32, 64) if smoke else (256, 1024, 4096, 8192)
        bind_scores = []
        for length in lengths:
            try:
                pt = score_binding(
                    btrainer.model,
                    length,
                    examples=2 if smoke else 64,
                    seed=92231 + length,
                    device=btrainer.device,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                pt = {"length": length, "error": "CUDA OOM", "accuracy": None, "paired_accuracy": None}
            compact = {k: v for k, v in pt.items() if k not in ("examples_out", "per_sequence")}
            bind_scores.append(compact)
            if pt.get("accuracy") is not None:
                print(
                    f"[{arm.name}] Binding len={length}: acc={100*pt['accuracy']:.1f}% | paired={100*pt['paired_accuracy']:.1f}%",
                    flush=True,
                )

    return {
        "name": arm.name,
        "group": arm.group,
        "architecture": arm.architecture,
        "parameters": params,
        "state_bytes_per_seq": (
            btrainer.model.state_bytes_per_sequence() if hasattr(btrainer.model, "state_bytes_per_sequence") else None
        ),
        "stories": {
            "lr": cal["lr"],
            "calibration": cal["trials"],
            "wall_seconds": story_wall,
            "peak_vram_bytes": story_vram,
            "val_256_nll": val_256["nll"],
            "val_long_nll": val_long.get("nll"),
            "val_long_length": long_len,
        },
        "decode": decode_stats,
        "binding": {
            "wall_seconds": bind_wall,
            "peak_vram_bytes": bind_vram,
            "scores": bind_scores,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--deadline-seconds", type=float, default=5400.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [arm for arm in ARMS if not args.arms or arm.name in args.arms]
    if not selected:
        raise ValueError("no matching benchmark arms")

    if args.data_cache is not None and (Path(args.data_cache) / "manifest.json").exists():
        data_root = Path(args.data_cache)
    elif args.smoke and args.data_cache is None:
        from tests.test_stories_data import fixture_data
        data_root = fixture_data(out_dir / "smoke_stories").root
    else:
        data_root = prepare_stories(Path(args.data_cache or (out_dir / "stories_cache"))).root

    started = time.monotonic()
    reports = []
    deferred = []
    for arm in selected:
        if reports and (time.monotonic() - started) >= args.deadline_seconds:
            deferred.append(arm.name)
            continue
        reports.append(run_arm(arm, data_root, out_dir, smoke=args.smoke, device=args.device))
        payload = {
            "status": "running",
            "smoke": args.smoke,
            "environment": environment(),
            "wall_seconds": time.monotonic() - started,
            "arms": reports,
            "deferred_arms": deferred,
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload["status"] = "partial_budget" if deferred else "complete"
    payload["wall_seconds"] = time.monotonic() - started
    payload["deferred_arms"] = deferred
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{payload['status'].upper()} in {payload['wall_seconds']/60:.1f} min -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
