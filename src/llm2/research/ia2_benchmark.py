"""Dual-Battery (TinyStories + byte-binding-v2) & Compiled Decode benchmark for IA2 candidates."""

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from ..checkpoint import read_checkpoint_metadata
from ..config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from ..data import sha256_file
from ..runtime import attention_context, environment
from ..trainer import Trainer
from .binding_data import TASK_ID as BINDING_TASK_ID, generate as generate_binding
from .binding_worker import BindingTrainer, score as score_binding
from .decode_cache import RecurrentDecode, generate, verify_logits
from .decode_optimization import CompiledDecode, LocalTransformerDecode
from .stories_data import StoriesData, StoryStream, prepare as prepare_stories, window as story_window
from .stories_worker import StoryTrainer, score as score_stories


RATES = (0.0003, 0.001, 0.003)
BANDS = (("1-256", 0, 256), ("257-512", 256, 512), ("513-2048", 512, 2048), ("2049+", 2048, 10**9))
UNIFIED_WRITES = ((0, 4), (1000, 8), (2000, 16), (3000, 32))


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
    fractal_window: int = 0
    fractal_value_residual: bool = False
    fractal_golden_phase: bool = False
    fractal_free_energy: bool = False
    fractal_weight_mode: str = "dense"
    memory_dim: int = 64
    chunk_size: int = 32


ARMS = (
    ArmSpec("transformer_1x", "baseline_1x", "transformer", d_model=64, n_layers=2, n_heads=4, hidden_dim=96),
    ArmSpec("fractal_delta_1x", "candidate", "fractal_delta", d_model=80, n_layers=4, n_heads=2, hidden_dim=120, shared=True, beta_scale=2.0),
    ArmSpec("transformer_2x", "baseline_2x", "transformer", d_model=80, n_layers=3, n_heads=4, hidden_dim=128),
    ArmSpec("context_shared2", "recurrent_ref", "context_memory", d_model=80, n_layers=2, n_heads=4, hidden_dim=144, shared=True),
    ArmSpec("fractal_delta_med", "candidate", "fractal_delta", d_model=104, n_layers=4, n_heads=2, hidden_dim=184, shared=True, beta_scale=2.0),
    ArmSpec("fractal_hybrid_med", "candidate", "fractal_delta", d_model=104, n_layers=4, n_heads=2, hidden_dim=112, shared=True, beta_scale=2.0, fractal_window=32),
    ArmSpec("fractal_pure_med", "candidate", "fractal_delta", d_model=104, n_layers=6, n_heads=2, hidden_dim=176, shared=True, beta_scale=2.0, fractal_value_residual=True),
    ArmSpec("fractal_golden_med", "candidate", "fractal_delta", d_model=104, n_layers=6, n_heads=2, hidden_dim=176, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True),
    ArmSpec("fractal_free_energy_med", "candidate", "fractal_delta", d_model=104, n_layers=6, n_heads=2, hidden_dim=164, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_free_energy=True),
    ArmSpec("fractal_ternary_med", "candidate", "fractal_delta", d_model=104, n_layers=6, n_heads=2, hidden_dim=176, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_weight_mode="ternary"),
    ArmSpec("fractal_quat_med", "candidate", "fractal_delta", d_model=160, n_layers=6, n_heads=4, hidden_dim=432, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_weight_mode="quaternion"),
    ArmSpec("transformer_big", "baseline_big", "transformer", d_model=128, n_layers=6, n_heads=4, hidden_dim=288),
    ArmSpec("fractal_delta_big", "candidate", "fractal_delta", d_model=256, n_layers=4, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, memory_dim=96, chunk_size=64),
    ArmSpec("fractal_pure_big", "candidate", "fractal_delta", d_model=256, n_layers=6, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, fractal_value_residual=True, memory_dim=96, chunk_size=64),
    ArmSpec("fractal_golden_big", "candidate", "fractal_delta", d_model=256, n_layers=6, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, memory_dim=96, chunk_size=64),
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
        memory_dim=8 if smoke else arm.memory_dim,
        state_execution="chunked",
        state_chunk_size=arm.chunk_size,
        fractal_beta_scale=arm.beta_scale,
        fractal_window=arm.fractal_window,
        fractal_value_residual=arm.fractal_value_residual,
        fractal_golden_phase=arm.fractal_golden_phase,
        fractal_free_energy=arm.fractal_free_energy,
        fractal_weight_mode=arm.fractal_weight_mode,
    )


def make_story_config(arm: ArmSpec, data_root, output_dir, *, seed=341, lr=0.001, smoke=False, device="cuda", calibration=False, story_seq_len=256):
    steps = (2 if calibration else 4) if smoke else (400 if calibration else 6000)
    d = 16 if smoke else arm.d_model
    ff = 24 if smoke else arm.hidden_dim
    heads = 2 if smoke else arm.n_heads
    seq_len = 32 if smoke else story_seq_len
    batch = 2 if smoke else max(1, (256 * 32) // seq_len)
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
            seq_len=seq_len,
            batch_size=batch,
            eval_batch_size=batch,
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


def select_story_lr(arm: ArmSpec, data_root, cal_root: Path, *, smoke=False, device="cuda", story_seq_len=256):
    selection_file = cal_root / arm.name / "selection.json"
    if selection_file.exists():
        return json.loads(selection_file.read_text(encoding="utf-8"))
    trials = []
    for lr in RATES:
        cfg = make_story_config(arm, data_root, cal_root / arm.name / str(lr), lr=lr, smoke=smoke, device=device, calibration=True, story_seq_len=story_seq_len)
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


def _make_decode_adapter(model, span):
    if type(model).__name__ == "TransformerLM":
        return LocalTransformerDecode(copy.deepcopy(model).eval(), span=span)
    if type(model) is FractalDeltaMemory:
        return FractalDecode(model)
    return RecurrentDecode(model)


def _collate_padded(pairs, device):
    width = max(len(x) for x, _ in pairs)
    bx = np.full((len(pairs), width), 256, dtype=np.int64)
    by = np.full((len(pairs), width), -100, dtype=np.int64)
    for row_idx, (x, y) in enumerate(pairs):
        n = len(x)
        bx[row_idx, :n] = x
        by[row_idx, :n] = y
    return torch.from_numpy(bx).to(device), torch.from_numpy(by).to(device)


@torch.no_grad()
def score_stories_bands_and_prefix(model, data, split, device, *, smoke=False):
    """Evaluate position-band NLL and causal distant-prefix intervention on full split documents."""
    was_training = model.training
    model.eval()
    max_len = 64 if smoke else 8192
    batch_docs = 2 if smoke else 8
    band_nll = {name: 0.0 for name, _, _ in BANDS}
    band_bytes = {name: 0 for name, _, _ in BANDS}
    try:
        docs = data.docs[split]
        with attention_context(device.type == "cpu"):
            for start in range(0, len(docs), batch_docs):
                pairs = [
                    story_window(data, split, idx, 0, min(docs[idx]["bytes"], max_len))
                    for idx in range(start, min(start + batch_docs, len(docs)))
                ]
                bx, by = _collate_padded(pairs, device)
                logits = model(bx).logits.float()
                loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), by, ignore_index=-100, reduction="none")
                valid = by != -100
                pos = torch.arange(by.shape[1], device=device)[None, :]
                for name, lo, hi in BANDS:
                    mask = valid & (pos >= lo) & (pos < hi)
                    band_nll[name] += float(loss[mask].sum())
                    band_bytes[name] += int(mask.sum())

            min_len = 48 if smoke else 640
            mid = 12 if smoke else 256
            eval_start = 24 if smoke else 512
            eligible = [i for i, r in enumerate(docs) if r["bytes"] >= min_len]
            if len(eligible) < 2:
                prefix_report = {
                    "eligible_docs": len(eligible),
                    "eval_start": eval_start,
                    "prefix_mid": mid,
                    "eval_bytes": 0,
                    "full_nll": None,
                    "truncated_nll": None,
                    "swapped_nll": None,
                    "delta_vs_truncated": None,
                    "delta_vs_swapped": None,
                }
            else:
                full_pairs, trunc_pairs, swap_pairs = [], [], []
                sums = {"full": 0.0, "truncated": 0.0, "swapped": 0.0}
                counts = {"full": 0, "truncated": 0, "swapped": 0}

                def _flush(key, batch_pairs):
                    bx, by = _collate_padded(batch_pairs, device)
                    logits = model(bx).logits.float()
                    loss = torch.nn.functional.cross_entropy(
                        logits.transpose(1, 2), by, ignore_index=-100, reduction="none"
                    )
                    mask = by != -100
                    sums[key] += float(loss[mask].sum())
                    counts[key] += int(mask.sum())

                for pos_i, idx in enumerate(eligible):
                    row = docs[idx]
                    orow = docs[eligible[(pos_i + 1) % len(eligible)]]
                    L = min(row["bytes"], max_len)
                    raw = np.asarray(data.arrays[split][row["offset"] : row["offset"] + L], dtype=np.int64)
                    other = np.asarray(data.arrays[split][orow["offset"] : orow["offset"] + mid], dtype=np.int64)
                    x_full = np.empty(L, dtype=np.int64)
                    x_full[0] = 256
                    x_full[1:] = raw[:-1]
                    y_full = np.full(L, -100, dtype=np.int64)
                    y_full[eval_start:] = raw[eval_start:]
                    full_pairs.append((x_full, y_full))

                    trunc_len = L - mid
                    x_trunc = np.empty(trunc_len, dtype=np.int64)
                    x_trunc[0] = 256
                    x_trunc[1:] = raw[mid:-1]
                    y_trunc = np.full(trunc_len, -100, dtype=np.int64)
                    y_trunc[eval_start - mid :] = raw[eval_start:]
                    trunc_pairs.append((x_trunc, y_trunc))

                    x_swap = x_full.copy()
                    x_swap[1 : mid + 1] = other
                    swap_pairs.append((x_swap, y_full))

                    if len(full_pairs) == batch_docs or pos_i == len(eligible) - 1:
                        _flush("full", full_pairs)
                        _flush("truncated", trunc_pairs)
                        _flush("swapped", swap_pairs)
                        full_pairs.clear()
                        trunc_pairs.clear()
                        swap_pairs.clear()

                if not (counts["full"] == counts["truncated"] == counts["swapped"] and counts["full"] > 0):
                    raise ValueError("distant-prefix intervention target byte count mismatch")
                n = counts["full"]
                full_nll = sums["full"] / n
                trunc_nll = sums["truncated"] / n
                swap_nll = sums["swapped"] / n
                prefix_report = {
                    "eligible_docs": len(eligible),
                    "eval_start": eval_start,
                    "prefix_mid": mid,
                    "eval_bytes": n,
                    "full_nll": full_nll,
                    "truncated_nll": trunc_nll,
                    "swapped_nll": swap_nll,
                    "delta_vs_truncated": trunc_nll - full_nll,
                    "delta_vs_swapped": swap_nll - full_nll,
                }
        bands_out = {
            name: {"bytes": band_bytes[name], "nll": (band_nll[name] / band_bytes[name]) if band_bytes[name] else None}
            for name, _, _ in BANDS
        }
        return {"bands": bands_out, "prefix_intervention": prefix_report}
    finally:
        model.train(was_training)


class UnifiedStream:
    """Deterministic 75% TinyStories + 25% Byte-Binding-v2 batch mixer."""

    def __init__(self, data, cfg):
        self.data = data
        self.length = cfg.data.seq_len
        self.batch = cfg.data.batch_size
        self.accumulation = cfg.train.grad_accum_steps
        self.smoke = self.length < 256
        self.bind_batch = max(1, self.batch // 4)
        self.story_batch = self.batch - self.bind_batch
        if self.story_batch < 1:
            raise ValueError("unified batch_size must be at least 2")
        story_cfg = copy.deepcopy(cfg)
        story_cfg.data.batch_size = self.story_batch
        self.story_stream = StoryStream(data, story_cfg, calibration=False)
        self.bind_rng = np.random.default_rng(cfg.train.seed + 101)
        self.draws = 0

    def next(self, device):
        story = self.story_stream.next(device)
        step = self.draws // self.accumulation
        writes = 2 if self.smoke else next(w for th, w in reversed(UNIFIED_WRITES) if step >= th)
        writes = min(writes, self.length // 8)
        queries = min(16, self.length // 8)
        bind = generate_binding(self.bind_rng, self.bind_batch, self.length, writes=writes, queries=queries, device=device)
        self.draws += 1
        return {
            "input_ids": torch.cat((story["input_ids"], bind["input_ids"]), dim=0),
            "targets": torch.cat((story["targets"], bind["targets"]), dim=0),
        }

    def state_dict(self):
        return {
            "task": f"unified:{BINDING_TASK_ID}",
            "length": self.length,
            "batch": self.batch,
            "accumulation": self.accumulation,
            "draws": self.draws,
            "story": self.story_stream.state_dict(),
            "bind_rng": copy.deepcopy(self.bind_rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        cur = self.state_dict()
        if any(state[k] != cur[k] for k in ("task", "length", "batch", "accumulation")):
            raise ValueError("incompatible unified sampler")
        self.draws = state["draws"]
        self.story_stream.load_state_dict(state["story"])
        self.bind_rng.bit_generator.state = copy.deepcopy(state["bind_rng"])


class UnifiedTrainer(Trainer):
    """Single-checkpoint trainer mixing TinyStories and Byte-Binding-v2."""

    def __init__(self, cfg, *, smoke=False, **kwargs):
        self.smoke = smoke
        self.evaluations = []
        data = StoriesData(cfg.data.path, require_pinned=not smoke)
        super().__init__(cfg, data=data, sampler_factory=lambda d, c: UnifiedStream(d, c), **kwargs)

    def validate(self, max_batches=None):
        story = score_stories(
            self.model, self.data, "val", self.config.data.seq_len, self.config.data.eval_batch_size, self.device
        )
        bind = score_binding(
            self.model,
            self.config.data.seq_len,
            examples=2 if self.smoke else 32,
            seed=91131,
            device=self.device,
            writes=2 if self.smoke else 32,
        )
        cost = self._evaluation_cost()
        if cost is not None:
            self.counters["validation_flops_estimate"] += cost * (story["windows"] + 2 * bind["examples"])
        self._update_experiment_wall()
        self._update_peak_memory()
        point = {
            "step": self.counters["step"],
            "story_nll": story["nll"],
            "binding_acc": bind["accuracy"],
            "binding_paired_acc": bind["paired_accuracy"],
            "training_wall_seconds": self.counters["train_wall_seconds"],
        }
        self.evaluations.append(point)
        print(
            f"[unified step={point['step']}] story_val={story['nll']:.4f} | "
            f"bind_acc={100*bind['accuracy']:.1f}% | bind_paired={100*bind['paired_accuracy']:.1f}%",
            flush=True,
        )
        return {
            "validation_loss": story["nll"],
            "binding_accuracy": bind["accuracy"],
            "paired_accuracy": bind["paired_accuracy"],
        }


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
        eager = _make_decode_adapter(model, span=span)
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
        return results, eager
    finally:
        model.train(was_training)


def sample_story_text(model, device, *, smoke=False, decoder=None):
    prompts = (
        "Once upon a time, a little ",
        "One day, Lily found a shiny ",
    )
    new_bytes = 16 if smoke else 96
    was_training = model.training
    model.eval()
    try:
        if decoder is None:
            decoder = _make_decode_adapter(model, span=32 if smoke else 256)
        samples = []
        with torch.inference_mode():
            for prompt in prompts:
                raw = list(prompt.encode("utf-8"))
                prefix = torch.tensor([raw], dtype=torch.long, device=device)
                logits, state = decoder.prefill(prefix, len(raw) + new_bytes)
                gen = []
                for _ in range(new_bytes):
                    nxt = int(logits[:, -1, :256].argmax(-1).item())
                    gen.append(nxt)
                    logits, state = decoder.step(torch.tensor([[nxt]], dtype=torch.long, device=device), state)
                samples.append({"prompt": prompt, "completion": bytes(gen).decode("utf-8", errors="replace")})
        return samples
    finally:
        model.train(was_training)


def _evaluate_stories_and_decode(model, data, device, *, story_seq_len, eval_batch_size, smoke, arm_label):
    val_256 = score_stories(model, data, "val", 32 if smoke else 256, 2 if smoke else 32, device)
    val_512 = (
        score_stories(model, data, "val", 48 if smoke else story_seq_len, 2 if smoke else eval_batch_size, device)
        if story_seq_len != 256
        else None
    )
    long_len = 48 if smoke else 3219
    try:
        val_long = score_stories(model, data, "val", long_len, 1 if smoke else 4, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        val_long = {"nll": None, "error": "CUDA OOM"}
    diagnostics = score_stories_bands_and_prefix(model, data, "val", device, smoke=smoke)
    decode_stats, eager_decoder = benchmark_decode(model, device, smoke=smoke)
    samples = sample_story_text(model, device, smoke=smoke, decoder=eager_decoder)
    b = diagnostics["bands"]
    p = diagnostics["prefix_intervention"]
    band_str = " ".join(f"{k}={v['nll']:.4f}" if v["nll"] is not None else f"{k}=None" for k, v in b.items())
    print(
        f"[{arm_label}] TinyStories val@256={val_256['nll']:.4f} | "
        + (f"val@{story_seq_len}={val_512['nll']:.4f} | " if val_512 else "")
        + f"val@{long_len}={val_long.get('nll')} | decode_inductor={decode_stats['inductor']['bytes_per_sec']:.1f} B/s",
        flush=True,
    )
    print(
        f"[{arm_label}] Bands: {band_str} | Prefix delta_trunc={p['delta_vs_truncated']} delta_swap={p['delta_vs_swapped']}",
        flush=True,
    )
    return val_256, val_512, val_long, long_len, diagnostics, decode_stats, samples


def _evaluate_binding_suite(model, device, *, smoke, binding_hard_writes, arm_label):
    lengths = (32, 64) if smoke else (256, 1024, 4096, 8192)
    bind_scores = []
    for length in lengths:
        try:
            pt = score_binding(
                model,
                length,
                examples=2 if smoke else 64,
                seed=92231 + length,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pt = {"length": length, "error": "CUDA OOM", "accuracy": None, "paired_accuracy": None}
        compact = {k: v for k, v in pt.items() if k not in ("examples_out", "per_sequence")}
        bind_scores.append(compact)
        if pt.get("accuracy") is not None:
            print(
                f"[{arm_label}] Binding len={length} w={compact['writes']}: acc={100*pt['accuracy']:.1f}% | paired={100*pt['paired_accuracy']:.1f}%",
                flush=True,
            )
    if binding_hard_writes > 0:
        hard_len = 64 if smoke else 8192
        hard_w = min(binding_hard_writes, hard_len // 8)
        try:
            pt = score_binding(
                model,
                hard_len,
                examples=2 if smoke else 64,
                seed=99231 + hard_len,
                device=device,
                writes=hard_w,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pt = {"length": hard_len, "writes": hard_w, "error": "CUDA OOM", "accuracy": None, "paired_accuracy": None}
        compact = {k: v for k, v in pt.items() if k not in ("examples_out", "per_sequence")}
        bind_scores.append(compact)
        if pt.get("accuracy") is not None:
            print(
                f"[{arm_label}] Binding HARD len={hard_len} w={compact['writes']}: acc={100*pt['accuracy']:.1f}% | paired={100*pt['paired_accuracy']:.1f}%",
                flush=True,
            )
    return bind_scores


def run_arm(
    arm: ArmSpec,
    data_root,
    work_root: Path,
    *,
    smoke=False,
    device="cuda",
    story_seq_len=256,
    binding_hard_writes=0,
    mode="separate",
    seed=341,
    lr_override=None,
):
    arm_label = f"{arm.name}:s{seed}" if (seed != 341 or mode == "unified") else arm.name
    print(f"\n=== [{arm_label}] ({arm.group}, mode={mode}) starting on {device} ===", flush=True)
    arm_root = work_root / (f"{arm.name}_s{seed}" if (seed != 341 or mode == "unified") else arm.name)
    arm_root.mkdir(parents=True, exist_ok=True)
    if lr_override is not None:
        cal = {"lr": float(lr_override), "trials": [{"lr": float(lr_override), "nll": None, "source": "cli_override"}]}
    else:
        cal = select_story_lr(arm, data_root, work_root / "calibration", smoke=smoke, device=device, story_seq_len=story_seq_len)
    print(f"[{arm_label}] LR = {cal['lr']} (trials={cal['trials']})", flush=True)

    if mode == "unified":
        unified_cfg = make_story_config(
            arm, data_root, arm_root / "unified", seed=seed, lr=cal["lr"], smoke=smoke, device=device, story_seq_len=story_seq_len
        )
        with UnifiedTrainer(unified_cfg, smoke=smoke) as trainer:
            params = trainer.model_parameters
            state_bytes = trainer.model.state_bytes_per_sequence() if hasattr(trainer.model, "state_bytes_per_sequence") else None
            t0 = time.perf_counter()
            if trainer.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(trainer.device)
            trainer.fit()
            train_wall = time.perf_counter() - t0
            train_vram = torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None
            ckpt_sha = sha256_file(Path(trainer.last_checkpoint) / "model.safetensors") if trainer.last_checkpoint else None
            val_256, val_512, val_long, long_len, diagnostics, decode_stats, samples = _evaluate_stories_and_decode(
                trainer.model,
                trainer.data,
                trainer.device,
                story_seq_len=story_seq_len,
                eval_batch_size=unified_cfg.data.eval_batch_size,
                smoke=smoke,
                arm_label=arm_label,
            )
            bind_scores = _evaluate_binding_suite(
                trainer.model, trainer.device, smoke=smoke, binding_hard_writes=binding_hard_writes, arm_label=arm_label
            )
        return {
            "name": arm.name,
            "seed": seed,
            "mode": mode,
            "checkpoint_sha256": ckpt_sha,
            "group": arm.group,
            "architecture": arm.architecture,
            "parameters": params,
            "state_bytes_per_seq": state_bytes,
            "stories": {
                "lr": cal["lr"],
                "calibration": cal["trials"],
                "train_seq_len": 32 if smoke else story_seq_len,
                "wall_seconds": train_wall,
                "peak_vram_bytes": train_vram,
                "val_256_nll": val_256["nll"],
                "val_train_len_nll": val_512["nll"] if val_512 else None,
                "val_long_nll": val_long.get("nll"),
                "val_long_length": long_len,
                "bands": diagnostics["bands"],
                "prefix_intervention": diagnostics["prefix_intervention"],
                "samples": samples,
            },
            "decode": decode_stats,
            "binding": {
                "wall_seconds": train_wall,
                "peak_vram_bytes": train_vram,
                "scores": bind_scores,
            },
        }

    story_cfg = make_story_config(
        arm, data_root, arm_root / "stories", seed=seed, lr=cal["lr"], smoke=smoke, device=device, story_seq_len=story_seq_len
    )
    with StoryTrainer(story_cfg, smoke=smoke) as trainer:
        params = trainer.model_parameters
        story_t0 = time.perf_counter()
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        trainer.fit()
        story_wall = time.perf_counter() - story_t0
        story_vram = torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None
        story_ckpt_sha = sha256_file(Path(trainer.last_checkpoint) / "model.safetensors") if trainer.last_checkpoint else None
        val_256, val_512, val_long, long_len, diagnostics, decode_stats, samples = _evaluate_stories_and_decode(
            trainer.model,
            trainer.data,
            trainer.device,
            story_seq_len=story_seq_len,
            eval_batch_size=story_cfg.data.eval_batch_size,
            smoke=smoke,
            arm_label=arm_label,
        )

    bind_cfg = make_binding_config(arm, arm_root / "binding", seed=seed, lr=cal["lr"], smoke=smoke, device=device)
    with BindingTrainer(bind_cfg, smoke=smoke) as btrainer:
        bind_t0 = time.perf_counter()
        if btrainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(btrainer.device)
        btrainer.fit()
        bind_wall = time.perf_counter() - bind_t0
        bind_vram = torch.cuda.max_memory_allocated(btrainer.device) if btrainer.device.type == "cuda" else None
        bind_ckpt_sha = sha256_file(Path(btrainer.last_checkpoint) / "model.safetensors") if btrainer.last_checkpoint else None
        state_bytes = btrainer.model.state_bytes_per_sequence() if hasattr(btrainer.model, "state_bytes_per_sequence") else None
        bind_scores = _evaluate_binding_suite(
            btrainer.model, btrainer.device, smoke=smoke, binding_hard_writes=binding_hard_writes, arm_label=arm_label
        )

    return {
        "name": arm.name,
        "seed": seed,
        "mode": mode,
        "checkpoint_sha256": story_ckpt_sha,
        "binding_checkpoint_sha256": bind_ckpt_sha,
        "group": arm.group,
        "architecture": arm.architecture,
        "parameters": params,
        "state_bytes_per_seq": state_bytes,
        "stories": {
            "lr": cal["lr"],
            "calibration": cal["trials"],
            "train_seq_len": 32 if smoke else story_seq_len,
            "wall_seconds": story_wall,
            "peak_vram_bytes": story_vram,
            "val_256_nll": val_256["nll"],
            "val_train_len_nll": val_512["nll"] if val_512 else None,
            "val_long_nll": val_long.get("nll"),
            "val_long_length": long_len,
            "bands": diagnostics["bands"],
            "prefix_intervention": diagnostics["prefix_intervention"],
            "samples": samples,
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
    parser.add_argument("--mode", choices=("separate", "unified"), default="separate")
    parser.add_argument("--seeds", nargs="*", type=int, default=[341])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--deadline-seconds", type=float, default=5400.0)
    parser.add_argument("--story-seq-len", type=int, default=256)
    parser.add_argument("--binding-hard-writes", type=int, default=0)
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
    payload = {
        "status": "running",
        "mode": args.mode,
        "smoke": args.smoke,
        "environment": environment(),
        "wall_seconds": 0.0,
        "arms": reports,
        "deferred_arms": deferred,
    }
    runs = [(arm, seed) for seed in args.seeds for arm in selected]
    for arm, seed in runs:
        if reports and (time.monotonic() - started) >= args.deadline_seconds:
            deferred.append(f"{arm.name}:s{seed}")
            continue
        reports.append(
            run_arm(
                arm,
                data_root,
                out_dir,
                smoke=args.smoke,
                device=args.device,
                story_seq_len=args.story_seq_len,
                binding_hard_writes=args.binding_hard_writes,
                mode=args.mode,
                seed=seed,
                lr_override=args.lr,
            )
        )
        payload = {
            "status": "running",
            "mode": args.mode,
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
