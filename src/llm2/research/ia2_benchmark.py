"""Dual-Battery (TinyStories + byte-binding-v2) & Compiled Decode benchmark for IA2 candidates."""

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from ..architectures.quarot import quantize_model_quarot_int4
from ..checkpoint import read_checkpoint_metadata
from ..config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from ..data import sha256_file
from ..runtime import attention_context, environment
from ..trainer import Trainer
from .binding_data import TASK_ID as BINDING_TASK_ID, generate as generate_binding
from .binding_worker import BindingTrainer, score as score_binding
from .decode_cache import RecurrentDecode, generate, verify_logits
from .decode_optimization import (
    AdaptiveEarlyExitDecode,
    CompiledDecode,
    LocalTransformerDecode,
    NVMLPowerMonitor,
    SegmentMemoryCache,
    SelfSpeculativeMTPDecode,
    restore_power_limit,
    try_pin_l2_weights,
    try_set_power_limit,
)
from .stories_data import StoriesData, StoryStream, _doc_tokens, prepare as prepare_stories, prepare_bpe as prepare_bpe_stories, window as story_window
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
    fractal_leaky_budget: bool = False
    fractal_assoc_delta: bool = False
    fractal_assoc_val_dim: int = 0
    fractal_prelude_coda: bool = False
    fractal_kda: bool = False
    fractal_assoc_multilag: bool = False
    fractal_decoupled_gates: bool = False
    fractal_hybrid_stride: int = 0
    fractal_exit_gate: bool = False
    exit_gate_weight: float = 0.0
    intermediate_weight: float = 0.0
    optimizer_kind: str = "adamw"
    mtp_tokens: int = 0
    mtp_weight: float = 0.0
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
    ArmSpec("fractal_golden_classic_med", "candidate", "fractal_delta", d_model=104, n_layers=6, n_heads=2, hidden_dim=176, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True),
    ArmSpec("fractal_free_energy_med", "candidate", "fractal_delta", d_model=104, n_layers=6, n_heads=2, hidden_dim=164, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_free_energy=True),
    ArmSpec("fractal_ternary_med", "candidate", "fractal_delta", d_model=104, n_layers=6, n_heads=2, hidden_dim=176, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_weight_mode="ternary"),
    ArmSpec("fractal_quat_med", "candidate", "fractal_delta", d_model=160, n_layers=6, n_heads=4, hidden_dim=432, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_weight_mode="quaternion"),
    ArmSpec("transformer_big", "baseline_big", "transformer", d_model=128, n_layers=6, n_heads=4, hidden_dim=288),
    ArmSpec("fractal_delta_big", "candidate", "fractal_delta", d_model=256, n_layers=4, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, memory_dim=96, chunk_size=64),
    ArmSpec("fractal_pure_big", "candidate", "fractal_delta", d_model=256, n_layers=6, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, fractal_value_residual=True, memory_dim=96, chunk_size=64),
    ArmSpec("fractal_golden_big", "candidate", "fractal_delta", d_model=256, n_layers=6, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, memory_dim=96, chunk_size=64),
    ArmSpec("fractal_golden_classic_big", "candidate", "fractal_delta", d_model=256, n_layers=6, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, memory_dim=96, chunk_size=64),
    ArmSpec("fractal_golden_classic_v2_big", "candidate", "fractal_delta", d_model=256, n_layers=6, n_heads=4, hidden_dim=544, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, memory_dim=96, chunk_size=64),
    ArmSpec("transformer_bpe_med", "baseline_2x", "transformer", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024),
    ArmSpec("fractal_classic_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, memory_dim=96, chunk_size=32),
    ArmSpec("fractal_pcc_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, fractal_prelude_coda=True, fractal_assoc_multilag=True, memory_dim=96, chunk_size=32),
    ArmSpec("fractal_pcc_kda_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, fractal_prelude_coda=True, fractal_kda=True, fractal_assoc_multilag=True, memory_dim=96, chunk_size=32),
    ArmSpec("fractal_pcc_mtp_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, fractal_prelude_coda=True, fractal_assoc_multilag=True, mtp_tokens=2, mtp_weight=0.25, memory_dim=96, chunk_size=32),
    ArmSpec("fractal_pcc_muon_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, fractal_prelude_coda=True, fractal_assoc_multilag=True, optimizer_kind="muon", memory_dim=96, chunk_size=32),
    ArmSpec("fractal_pcc_mtp_muon_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, fractal_prelude_coda=True, fractal_assoc_multilag=True, optimizer_kind="muon", mtp_tokens=2, mtp_weight=0.25, memory_dim=96, chunk_size=32),
    ArmSpec("fractal_pcc_full_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, fractal_prelude_coda=True, fractal_kda=True, fractal_assoc_multilag=True, optimizer_kind="muon", mtp_tokens=2, mtp_weight=0.25, memory_dim=96, chunk_size=32),
    ArmSpec("fractal_v2_bpe", "candidate", "fractal_delta", d_model=384, n_layers=8, n_heads=6, hidden_dim=1024, shared=True, beta_scale=2.0, fractal_value_residual=True, fractal_golden_phase=True, fractal_leaky_budget=True, fractal_assoc_delta=True, fractal_assoc_val_dim=48, fractal_prelude_coda=True, fractal_assoc_multilag=True, fractal_decoupled_gates=True, fractal_window=64, fractal_hybrid_stride=4, fractal_exit_gate=True, exit_gate_weight=0.10, intermediate_weight=0.20, optimizer_kind="muon", mtp_tokens=2, mtp_weight=0.25, memory_dim=96, chunk_size=32),
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
        fractal_leaky_budget=arm.fractal_leaky_budget,
        fractal_assoc_delta=arm.fractal_assoc_delta,
        fractal_assoc_val_dim=arm.fractal_assoc_val_dim,
        fractal_prelude_coda=arm.fractal_prelude_coda,
        fractal_kda=arm.fractal_kda,
        fractal_assoc_multilag=arm.fractal_assoc_multilag,
        fractal_decoupled_gates=arm.fractal_decoupled_gates,
        fractal_hybrid_stride=arm.fractal_hybrid_stride,
        fractal_exit_gate=arm.fractal_exit_gate,
        exit_gate_weight=arm.exit_gate_weight,
        intermediate_weight=arm.intermediate_weight,
        mtp_tokens=arm.mtp_tokens,
        mtp_weight=arm.mtp_weight,
    )


def _manifest_vocab_size(data_root, default: int = 257) -> int:
    manifest = Path(data_root) / "manifest.json"
    if manifest.is_file():
        return int(json.loads(manifest.read_text(encoding="utf-8"))["vocab_size"])
    return int(default)


def make_story_config(arm: ArmSpec, data_root, output_dir, *, seed=341, lr=0.001, smoke=False, device="cuda", calibration=False, story_seq_len=256, precision="fp32", vocab_size=None, max_steps=None):
    steps = (2 if calibration else 4) if smoke else (400 if calibration else (int(max_steps) if max_steps is not None else 6000))
    d = 16 if smoke else arm.d_model
    ff = 24 if smoke else arm.hidden_dim
    heads = 2 if smoke else arm.n_heads
    seq_len = 32 if smoke else story_seq_len
    batch = 2 if smoke else max(1, (256 * 32) // seq_len)
    resolved_vocab = _manifest_vocab_size(data_root) if vocab_size is None else int(vocab_size)
    research = _build_research_config(arm, smoke=smoke)
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=resolved_vocab,
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
            kind=arm.optimizer_kind,
            lr=lr,
            weight_decay=0.01,
            warmup_steps=1 if smoke else (50 if calibration else 100),
            schedule_steps=steps,
        ),
        train=TrainConfig(
            seed=931 if calibration else seed,
            device=device,
            precision="fp32" if device == "cpu" else precision,
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


def make_binding_config(arm: ArmSpec, output_dir, *, seed=311, lr=0.003, smoke=False, device="cuda", precision="fp32", vocab_size=257):
    steps = 4 if smoke else 3000
    d = 16 if smoke else arm.d_model
    ff = 24 if smoke else arm.hidden_dim
    heads = 2 if smoke else arm.n_heads
    research = _build_research_config(arm, smoke=smoke)
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=int(vocab_size),
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
            kind=arm.optimizer_kind,
            lr=lr,
            weight_decay=0.01,
            warmup_steps=1 if smoke else 50,
            schedule_steps=steps,
        ),
        train=TrainConfig(
            seed=seed,
            device=device,
            precision="fp32" if device == "cpu" else precision,
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


def select_story_lr(arm: ArmSpec, data_root, cal_root: Path, *, smoke=False, device="cuda", story_seq_len=256, precision="fp32", vocab_size=None):
    selection_file = cal_root / arm.name / "selection.json"
    if selection_file.exists():
        return json.loads(selection_file.read_text(encoding="utf-8"))
    trials = []
    for lr in RATES:
        cfg = make_story_config(arm, data_root, cal_root / arm.name / str(lr), lr=lr, smoke=smoke, device=device, calibration=True, story_seq_len=story_seq_len, precision=precision, vocab_size=vocab_size)
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


def _collate_padded(pairs, device, *, bos_id=256):
    width = max(len(x) for x, _ in pairs)
    bx = np.full((len(pairs), width), bos_id, dtype=np.int64)
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
        bos_id = data.bos_id
        with attention_context(device.type == "cpu"):
            for start in range(0, len(docs), batch_docs):
                pairs = [
                    story_window(data, split, idx, 0, min(_doc_tokens(docs[idx]), max_len))
                    for idx in range(start, min(start + batch_docs, len(docs)))
                ]
                bx, by = _collate_padded(pairs, device, bos_id=bos_id)
                logits = model(bx).logits.float()
                loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), by, ignore_index=-100, reduction="none")
                valid = by != -100
                pos = torch.arange(by.shape[1], device=device)[None, :]
                for name, lo, hi in BANDS:
                    mask = valid & (pos >= lo) & (pos < hi)
                    band_nll[name] += float(loss[mask].sum())
                    band_bytes[name] += int(mask.sum())

            min_len = 36 if smoke else (192 if data.vocab_size > 257 else 640)
            mid = 12 if smoke else (64 if data.vocab_size > 257 else 256)
            eval_start = 24 if smoke else (128 if data.vocab_size > 257 else 512)
            eligible = [i for i, r in enumerate(docs) if _doc_tokens(r) >= min_len]
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
                    bx, by = _collate_padded(batch_pairs, device, bos_id=bos_id)
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
                    L = min(_doc_tokens(row), max_len)
                    raw = np.asarray(data.arrays[split][row["offset"] : row["offset"] + L], dtype=np.int64)
                    other = np.asarray(data.arrays[split][orow["offset"] : orow["offset"] + mid], dtype=np.int64)
                    x_full = np.empty(L, dtype=np.int64)
                    x_full[0] = bos_id
                    x_full[1:] = raw[:-1]
                    y_full = np.full(L, -100, dtype=np.int64)
                    y_full[eval_start:] = raw[eval_start:]
                    full_pairs.append((x_full, y_full))

                    trunc_len = L - mid
                    x_trunc = np.empty(trunc_len, dtype=np.int64)
                    x_trunc[0] = bos_id
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

    def __init__(self, data, cfg, *, curriculum=False):
        self.data = data
        self.length = cfg.data.seq_len
        self.batch = cfg.data.batch_size
        self.max_seq_len = cfg.model.max_seq_len
        self.max_steps = cfg.train.max_steps
        self.accumulation = cfg.train.grad_accum_steps
        self.smoke = self.length < 256
        self.curriculum = curriculum
        self.bind_batch = max(1, self.batch // 4)
        self.story_batch = self.batch - self.bind_batch
        if self.story_batch < 1:
            raise ValueError("unified batch_size must be at least 2")
        story_cfg = copy.deepcopy(cfg)
        story_cfg.data.batch_size = self.story_batch
        self.story_stream = StoryStream(data, story_cfg, calibration=False)
        self.stage_streams = {}
        if self.curriculum:
            stages = [(self.length * 2, max(2, self.batch // 2))]
            if not self.smoke and self.batch >= 4:
                stages.append((self.length * 4, max(2, self.batch // 4)))
            for cur_len, cur_batch in stages:
                if cur_len <= self.max_seq_len and cur_batch >= 2:
                    scfg = copy.deepcopy(cfg)
                    scfg.data.seq_len = cur_len
                    scfg.data.batch_size = cur_batch - max(1, cur_batch // 4)
                    scfg.train.seed = cfg.train.seed + cur_len
                    self.stage_streams[cur_len] = StoryStream(data, scfg, calibration=False)
        self.bind_rng = np.random.default_rng(cfg.train.seed + 101)
        self.draws = 0

    def _stage(self, step: int) -> tuple[int, int]:
        if not self.curriculum:
            return self.length, self.batch
        if self.smoke:
            cur_len = self.length * 2
            if step >= 1 and cur_len in self.stage_streams:
                return cur_len, max(2, self.batch // 2)
            return self.length, self.batch
        stage2 = max(1, self.max_steps // 2)
        stage3 = max(stage2 + 1, (3 * self.max_steps) // 4)
        if step >= stage3 and (self.length * 4) in self.stage_streams:
            return self.length * 4, max(2, self.batch // 4)
        if step >= stage2 and (self.length * 2) in self.stage_streams:
            return self.length * 2, max(2, self.batch // 2)
        return self.length, self.batch

    def next(self, device):
        step = self.draws // self.accumulation
        cur_len, cur_batch = self._stage(step)
        cur_bind = max(1, cur_batch // 4)
        stream = self.story_stream if cur_len == self.length else self.stage_streams[cur_len]
        story = stream.next(device)
        stage3 = max(1, (3 * self.max_steps) // 4)
        if self.smoke:
            writes = 2
        elif self.curriculum and step >= stage3 and cur_len >= 2048:
            writes = 48
        else:
            writes = next(w for th, w in reversed(UNIFIED_WRITES) if step >= th)
        writes = min(writes, cur_len // 8)
        queries = min(16, cur_len // 8)
        bind = generate_binding(self.bind_rng, cur_bind, cur_len, writes=writes, queries=queries, device=device)
        self.draws += 1
        return {
            "input_ids": torch.cat((story["input_ids"], bind["input_ids"]), dim=0),
            "targets": torch.cat((story["targets"], bind["targets"]), dim=0),
        }

    def state_dict(self):
        state = {
            "task": f"unified:{BINDING_TASK_ID}",
            "length": self.length,
            "batch": self.batch,
            "accumulation": self.accumulation,
            "draws": self.draws,
            "story": self.story_stream.state_dict(),
            "bind_rng": copy.deepcopy(self.bind_rng.bit_generator.state),
        }
        if self.curriculum:
            state["curriculum"] = True
            state["stages"] = {str(k): s.state_dict() for k, s in self.stage_streams.items()}
        return state

    def load_state_dict(self, state):
        cur = self.state_dict()
        if any(state.get(k) != cur.get(k) for k in ("task", "length", "batch", "accumulation", "curriculum")):
            raise ValueError("incompatible unified sampler")
        self.draws = state["draws"]
        self.story_stream.load_state_dict(state["story"])
        if self.curriculum:
            for k, stream in self.stage_streams.items():
                stream.load_state_dict(state["stages"][str(k)])
        self.bind_rng.bit_generator.state = copy.deepcopy(state["bind_rng"])


class UnifiedTrainer(Trainer):
    """Single-checkpoint trainer mixing TinyStories and Byte-Binding-v2."""

    def __init__(self, cfg, *, smoke=False, unified_curriculum=False, **kwargs):
        self.smoke = smoke
        self.unified_curriculum = unified_curriculum
        self.evaluations = []
        data = StoriesData(cfg.data.path, require_pinned=not smoke)
        super().__init__(cfg, data=data, sampler_factory=lambda d, c: UnifiedStream(d, c, curriculum=unified_curriculum), **kwargs)

    def validate(self, max_batches=None):
        eff_max = 2 if (self.smoke and max_batches is None) else max_batches
        story = score_stories(
            self.model, self.data, "val", self.config.data.seq_len, self.config.data.eval_batch_size, self.device, max_batches=eff_max
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
            "story_bpb": story["bits_per_byte"],
            "binding_acc": bind["accuracy"],
            "binding_paired_acc": bind["paired_accuracy"],
            "training_wall_seconds": self.counters["train_wall_seconds"],
        }
        self.evaluations.append(point)
        print(
            f"[unified step={point['step']}] story_val={story['nll']:.4f} bpb={story['bits_per_byte']:.4f} | "
            f"bind_acc={100*bind['accuracy']:.1f}% | bind_paired={100*bind['paired_accuracy']:.1f}%",
            flush=True,
        )
        return {
            "validation_loss": story["nll"],
            "bits_per_byte": story["bits_per_byte"],
            "binding_accuracy": bind["accuracy"],
            "paired_accuracy": bind["paired_accuracy"],
        }


def benchmark_decode(
    model,
    device,
    *,
    smoke=False,
    vocab_limit=256,
    mtp_heads=None,
    val_prefix=None,
    measure_energy=False,
    pin_l2=False,
):
    was_training = model.training
    model.eval()
    prefix_len = 24 if smoke else 7936
    new_bytes = 8 if smoke else 256
    span = 16 if smoke else 256
    gen = torch.Generator().manual_seed(91127)
    prefix = torch.randint(0, vocab_limit, (1, prefix_len), generator=gen).to(device)
    cont = torch.randint(0, vocab_limit, (1, 12 if smoke else 32), generator=gen).to(device)
    l2_pin_info = None
    if pin_l2 and isinstance(model, FractalDeltaMemory):
        core_idx = 1 if len(model.cells) > 1 else 0
        l2_pin_info = try_pin_l2_weights(list(model.cells[core_idx].parameters()), device)
    try:
        eager = _make_decode_adapter(model, span=span)
        results = {}
        if l2_pin_info is not None:
            results["l2_pin"] = l2_pin_info
        compiled_decoder = None
        for mode_name, use_compiled in (("eager", False), ("inductor", True)):
            decoder = CompiledDecode(eager, backend="inductor") if use_compiled else eager
            if use_compiled:
                compiled_decoder = decoder
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            parity = verify_logits(decoder, prefix, cont)
            first_step_wall = time.perf_counter() - t0
            monitor = NVMLPowerMonitor(device) if (measure_energy and use_compiled) else None
            if monitor is not None:
                with monitor:
                    timing, _ = generate(decoder, prefix, new_bytes)
                nvml_stats = monitor.summary
            else:
                timing, _ = generate(decoder, prefix, new_bytes)
                nvml_stats = None
            peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            toks_s = (new_bytes - 1) / max(timing["decode_seconds"], 1e-9)
            mj_per_tok = (
                (nvml_stats["energy_joules"] * 1000.0) / max(new_bytes - 1, 1)
                if nvml_stats is not None and nvml_stats.get("energy_joules") is not None
                else None
            )
            results[mode_name] = {
                "prefix_bytes": prefix_len,
                "new_bytes": new_bytes,
                "parity_max_abs_err": parity["max_abs_logit_error"],
                "decode_seconds": timing["decode_seconds"],
                "first_byte_seconds": timing["first_byte_seconds"],
                "total_seconds": timing["total_seconds"],
                "bytes_per_sec": toks_s,
                "state_bytes": timing["state_bytes"],
                "peak_vram_bytes": peak_vram,
                "warmup_parity_seconds": first_step_wall,
                "cudagraph": getattr(decoder, "use_cudagraph", False),
                "nvml": nvml_stats,
                "mj_per_token": mj_per_tok,
            }
        if mtp_heads is not None and len(mtp_heads) > 0 and isinstance(model, FractalDeltaMemory) and compiled_decoder is not None:
            real_prefix = val_prefix.to(device) if val_prefix is not None else prefix[:, : (24 if smoke else 256)]
            spec_n = new_bytes
            with torch.inference_mode():
                # Standard greedy baseline on the exact same real validation prefix
                base_cap = real_prefix.shape[1] + spec_n + 1
                b_logits, b_state = compiled_decoder.prefill(real_prefix, base_cap)
                base_tokens = []
                monitor_b = NVMLPowerMonitor(device) if measure_energy else None
                t_b0 = time.perf_counter()
                if monitor_b is not None:
                    monitor_b.__enter__()
                try:
                    for idx in range(spec_n):
                        tok = int(b_logits[0, -1, :vocab_limit].argmax(dim=-1).item())
                        base_tokens.append(tok)
                        if idx + 1 < spec_n:
                            b_logits, b_state = compiled_decoder.step(torch.tensor([[tok]], dtype=torch.long, device=device), b_state)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    base_s = max(time.perf_counter() - t_b0, 1e-9)
                finally:
                    if monitor_b is not None:
                        monitor_b.__exit__(None, None, None)
                spec_dec = SelfSpeculativeMTPDecode(model, mtp_heads, vocab_limit=vocab_limit, backend="inductor")
                spec_tokens, spec_stats = spec_dec.generate_greedy(real_prefix, spec_n, measure_energy=measure_energy)
                if spec_tokens != base_tokens:
                    raise RuntimeError("SelfSpeculativeMTPDecode greedy parity mismatch against standard decode")
                base_tok_s = spec_n / base_s
                base_nvml = None if monitor_b is None else monitor_b.summary
                base_mj = (
                    (base_nvml["energy_joules"] * 1000.0) / spec_n
                    if base_nvml is not None and base_nvml.get("energy_joules") is not None
                    else None
                )
                results["speculative_mtp"] = {
                    **spec_stats,
                    "prefix_tokens": int(real_prefix.shape[1]),
                    "greedy_exact_match": True,
                    "baseline_real_decode_seconds": base_s,
                    "baseline_real_tokens_per_sec": base_tok_s,
                    "baseline_real_nvml": base_nvml,
                    "baseline_real_mj_per_token": base_mj,
                    "speedup_vs_baseline": spec_stats["tokens_per_sec"] / max(base_tok_s, 1e-9),
                }
        if isinstance(model, FractalDeltaMemory):
            real_prefix = val_prefix.to(device) if val_prefix is not None else prefix[:, : (24 if smoke else 256)]
            ee_n = new_bytes
            with torch.inference_mode():
                ee_dec = AdaptiveEarlyExitDecode(
                    model,
                    confidence_threshold=0.70,
                    delta_threshold=0.05,
                    min_layers=2,
                    vocab_limit=vocab_limit,
                )
                _, ee_stats = ee_dec.generate_greedy(real_prefix, ee_n, measure_energy=measure_energy)
                results["adaptive_early_exit"] = ee_stats

                seg_cache = SegmentMemoryCache(max_entries=16, max_anchors=4)
                f_dec = FractalDecode(model)
                _, _, miss_info = seg_cache.prefill_cached(f_dec, real_prefix, real_prefix.shape[1] + ee_n + 1)
                _, _, hit_info = seg_cache.prefill_cached(f_dec, real_prefix, real_prefix.shape[1] + ee_n + 1)
                results["segment_memory_cache"] = {"first_lookup": miss_info, "second_lookup": hit_info}

                _, q_report = quantize_model_quarot_int4(model, group_size=32, act_bits=8, inplace=False)
                results["quarot_int4"] = q_report
        return results, eager
    finally:
        model.train(was_training)


def sample_story_text(model, device, *, smoke=False, decoder=None, tokenizer=None):
    prompts = (
        "Once upon a time, a little ",
        "One day, Lily found a shiny ",
    )
    is_bpe = tokenizer is not None and tokenizer.kind == "bpe"
    new_items = (12 if smoke else 64) if is_bpe else (16 if smoke else 96)
    was_training = model.training
    model.eval()
    try:
        if decoder is None:
            decoder = _make_decode_adapter(model, span=32 if smoke else 256)
        samples = []
        with torch.inference_mode():
            for prompt in prompts:
                raw = tokenizer.encode(prompt) if is_bpe else list(prompt.encode("utf-8"))
                prefix = torch.tensor([raw], dtype=torch.long, device=device)
                logits, state = decoder.prefill(prefix, len(raw) + new_items)
                gen = []
                for _ in range(new_items):
                    step_logits = logits[:, -1] if is_bpe else logits[:, -1, :256]
                    nxt = int(step_logits.argmax(-1).item())
                    if is_bpe and nxt == tokenizer.eos_id:
                        break
                    gen.append(nxt)
                    logits, state = decoder.step(torch.tensor([[nxt]], dtype=torch.long, device=device), state)
                text = tokenizer.decode(gen) if is_bpe else bytes(gen).decode("utf-8", errors="replace")
                samples.append({"prompt": prompt, "completion": text})
        return samples
    finally:
        model.train(was_training)


def _evaluate_stories_and_decode(
    model,
    data,
    device,
    *,
    story_seq_len,
    eval_batch_size,
    smoke,
    arm_label,
    mtp_heads=None,
    measure_energy=False,
    pin_l2=False,
):
    max_b = 2 if smoke else None
    val_256 = score_stories(model, data, "val", 32 if smoke else 256, 2 if smoke else 32, device, max_batches=max_b)
    val_512 = (
        score_stories(model, data, "val", 48 if smoke else story_seq_len, 2 if smoke else eval_batch_size, device, max_batches=max_b)
        if story_seq_len != 256
        else None
    )
    long_len = 48 if smoke else 3219
    try:
        val_long = score_stories(model, data, "val", long_len, 1 if smoke else 4, device, max_batches=max_b)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        val_long = {"nll": None, "bits_per_byte": None, "error": "CUDA OOM"}
    diagnostics = score_stories_bands_and_prefix(model, data, "val", device, smoke=smoke)
    vocab_limit = 256 if data.schema == "stories-doc-bytes-v1" else data.vocab_size
    val_prefix = None
    val_arr = getattr(data, "arrays", {}).get("val") if hasattr(data, "arrays") else None
    if val_arr is not None and len(val_arr) >= (24 if smoke else 256):
        plen = 24 if smoke else 256
        val_prefix = torch.tensor(np.asarray(val_arr[:plen], dtype=np.int64), dtype=torch.long).unsqueeze(0)
    decode_stats, eager_decoder = benchmark_decode(
        model,
        device,
        smoke=smoke,
        vocab_limit=vocab_limit,
        mtp_heads=mtp_heads,
        val_prefix=val_prefix,
        measure_energy=measure_energy,
        pin_l2=pin_l2,
    )
    samples = sample_story_text(model, device, smoke=smoke, decoder=eager_decoder, tokenizer=data.tokenizer)
    b = diagnostics["bands"]
    p = diagnostics["prefix_intervention"]
    band_str = " ".join(f"{k}={v['nll']:.4f}" if v["nll"] is not None else f"{k}=None" for k, v in b.items())
    print(
        f"[{arm_label}] TinyStories val@256={val_256['nll']:.4f} (bpb={val_256['bits_per_byte']:.4f}) | "
        + (f"val@{story_seq_len}={val_512['nll']:.4f} (bpb={val_512['bits_per_byte']:.4f}) | " if val_512 else "")
        + f"val@{long_len}={val_long.get('nll')} | decode_inductor={decode_stats['inductor']['bytes_per_sec']:.1f} tok/s",
        flush=True,
    )
    print(
        f"[{arm_label}] Bands: {band_str} | Prefix delta_trunc={p['delta_vs_truncated']} delta_swap={p['delta_vs_swapped']}",
        flush=True,
    )
    if "speculative_mtp" in decode_stats:
        sp = decode_stats["speculative_mtp"]
        print(
            f"[{arm_label}] Speculative MTP: tok/s={sp['tokens_per_sec']:.1f} (base={sp['baseline_real_tokens_per_sec']:.1f}, "
            f"speedup={sp['speedup_vs_baseline']:.2f}x, accept={100*sp['draft_acceptance_rate']:.1f}%, "
            f"tok/cycle={sp['mean_tokens_per_cycle']:.2f}, mj/tok={sp.get('mj_per_token')})",
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
    precision="fp32",
    unified_curriculum=False,
    max_steps=None,
    measure_energy=False,
    pin_l2=False,
):
    arm_label = f"{arm.name}:s{seed}" if (seed != 341 or mode == "unified") else arm.name
    print(f"\n=== [{arm_label}] ({arm.group}, mode={mode}, precision={precision}, curriculum={unified_curriculum}) starting on {device} ===", flush=True)
    arm_root = work_root / (f"{arm.name}_s{seed}" if (seed != 341 or mode == "unified") else arm.name)
    arm_root.mkdir(parents=True, exist_ok=True)
    vocab_size = _manifest_vocab_size(data_root)
    if lr_override is not None:
        cal = {"lr": float(lr_override), "trials": [{"lr": float(lr_override), "nll": None, "source": "cli_override"}]}
    else:
        cal = select_story_lr(arm, data_root, work_root / "calibration", smoke=smoke, device=device, story_seq_len=story_seq_len, precision=precision, vocab_size=vocab_size)
    print(f"[{arm_label}] LR = {cal['lr']} (trials={cal['trials']})", flush=True)

    if mode == "unified":
        unified_cfg = make_story_config(
            arm, data_root, arm_root / "unified", seed=seed, lr=cal["lr"], smoke=smoke, device=device, story_seq_len=story_seq_len, precision=precision, vocab_size=vocab_size, max_steps=max_steps
        )
        with UnifiedTrainer(unified_cfg, smoke=smoke, unified_curriculum=unified_curriculum) as trainer:
            params = trainer.model_parameters
            total_params = trainer.total_parameters
            state_bytes = trainer.model.state_bytes_per_sequence() if hasattr(trainer.model, "state_bytes_per_sequence") else None
            t0 = time.perf_counter()
            if trainer.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(trainer.device)
            trainer.fit()
            train_wall = time.perf_counter() - t0
            train_vram = torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None
            ckpt_sha = sha256_file(Path(trainer.last_checkpoint) / "model.safetensors") if trainer.last_checkpoint else None
            mtp_heads = getattr(trainer.objective, "mtp_heads", None)
            val_256, val_512, val_long, long_len, diagnostics, decode_stats, samples = _evaluate_stories_and_decode(
                trainer.model,
                trainer.data,
                trainer.device,
                story_seq_len=story_seq_len,
                eval_batch_size=unified_cfg.data.eval_batch_size,
                smoke=smoke,
                arm_label=arm_label,
                mtp_heads=mtp_heads if mtp_heads is not None and len(mtp_heads) > 0 else None,
                measure_energy=measure_energy,
                pin_l2=pin_l2,
            )
            bind_scores = _evaluate_binding_suite(
                trainer.model, trainer.device, smoke=smoke, binding_hard_writes=binding_hard_writes, arm_label=arm_label
            )
        return {
            "name": arm.name,
            "seed": seed,
            "mode": mode,
            "precision": unified_cfg.train.precision,
            "optimizer": arm.optimizer_kind,
            "vocab_size": vocab_size,
            "checkpoint_sha256": ckpt_sha,
            "group": arm.group,
            "architecture": arm.architecture,
            "parameters": params,
            "total_parameters": total_params,
            "state_bytes_per_seq": state_bytes,
            "stories": {
                "lr": cal["lr"],
                "calibration": cal["trials"],
                "train_seq_len": 32 if smoke else story_seq_len,
                "wall_seconds": train_wall,
                "peak_vram_bytes": train_vram,
                "val_256_nll": val_256["nll"],
                "val_256_bpb": val_256["bits_per_byte"],
                "val_train_len_nll": val_512["nll"] if val_512 else None,
                "val_train_len_bpb": val_512["bits_per_byte"] if val_512 else None,
                "val_long_nll": val_long.get("nll"),
                "val_long_bpb": val_long.get("bits_per_byte"),
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
        arm, data_root, arm_root / "stories", seed=seed, lr=cal["lr"], smoke=smoke, device=device, story_seq_len=story_seq_len, precision=precision, vocab_size=vocab_size, max_steps=max_steps
    )
    with StoryTrainer(story_cfg, smoke=smoke) as trainer:
        params = trainer.model_parameters
        total_params = trainer.total_parameters
        story_t0 = time.perf_counter()
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        trainer.fit()
        story_wall = time.perf_counter() - story_t0
        story_vram = torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None
        story_ckpt_sha = sha256_file(Path(trainer.last_checkpoint) / "model.safetensors") if trainer.last_checkpoint else None
        mtp_heads = getattr(trainer.objective, "mtp_heads", None)
        val_256, val_512, val_long, long_len, diagnostics, decode_stats, samples = _evaluate_stories_and_decode(
            trainer.model,
            trainer.data,
            trainer.device,
            story_seq_len=story_seq_len,
            eval_batch_size=story_cfg.data.eval_batch_size,
            smoke=smoke,
            arm_label=arm_label,
            mtp_heads=mtp_heads if mtp_heads is not None and len(mtp_heads) > 0 else None,
            measure_energy=measure_energy,
            pin_l2=pin_l2,
        )

    bind_cfg = make_binding_config(arm, arm_root / "binding", seed=seed, lr=cal["lr"], smoke=smoke, device=device, precision=precision, vocab_size=vocab_size)
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
        "precision": story_cfg.train.precision,
        "optimizer": arm.optimizer_kind,
        "vocab_size": vocab_size,
        "checkpoint_sha256": story_ckpt_sha,
        "binding_checkpoint_sha256": bind_ckpt_sha,
        "group": arm.group,
        "architecture": arm.architecture,
        "parameters": params,
        "total_parameters": total_params,
        "state_bytes_per_seq": state_bytes,
        "stories": {
            "lr": cal["lr"],
            "calibration": cal["trials"],
            "train_seq_len": 32 if smoke else story_seq_len,
            "wall_seconds": story_wall,
            "peak_vram_bytes": story_vram,
            "val_256_nll": val_256["nll"],
            "val_256_bpb": val_256["bits_per_byte"],
            "val_train_len_nll": val_512["nll"] if val_512 else None,
            "val_train_len_bpb": val_512["bits_per_byte"] if val_512 else None,
            "val_long_nll": val_long.get("nll"),
            "val_long_bpb": val_long.get("bits_per_byte"),
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
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--tokenizer", choices=("byte", "bpe"), default="byte")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--seeds", nargs="*", type=int, default=[341])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--deadline-seconds", type=float, default=5400.0)
    parser.add_argument("--story-seq-len", type=int, default=256)
    parser.add_argument("--binding-hard-writes", type=int, default=0)
    parser.add_argument("--unified-curriculum", action="store_true")
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--pin-l2", action="store_true")
    parser.add_argument("--power-limit-ratio", type=float, default=None)
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
        from .stories_data import write_bpe_dataset, write_dataset
        smoke_dir = out_dir / ("smoke_stories_bpe" if args.tokenizer == "bpe" else "smoke_stories")
        if (smoke_dir / "manifest.json").exists():
            data_root = smoke_dir
        else:
            splits = {}
            for split in ("train", "cal_val", "val", "test"):
                docs = [(f"{split}: Once upon a time, a little fox found a red ball. {i}\n").encode() * (i + 1) for i in range(4)]
                splits[split] = {hashlib.sha256(raw).hexdigest(): raw for raw in docs}
            if args.tokenizer == "bpe":
                data_root = write_bpe_dataset(smoke_dir, splits, spec={"smoke": True}, source_fingerprint="smoke", vocab_size=min(args.vocab_size, 512)).root
            else:
                data_root = write_dataset(smoke_dir, splits, spec={"engineering_smoke": True}).root
    elif args.tokenizer == "bpe":
        data_root = prepare_bpe_stories(Path(args.data_cache or (out_dir / "stories_cache")), vocab_size=args.vocab_size).root
    else:
        data_root = prepare_stories(Path(args.data_cache or (out_dir / "stories_cache"))).root

    started = time.monotonic()
    reports = []
    deferred = []
    power_cap = try_set_power_limit(args.device, args.power_limit_ratio) if args.power_limit_ratio is not None else None
    payload = {
        "status": "running",
        "mode": args.mode,
        "precision": args.precision,
        "tokenizer": args.tokenizer,
        "unified_curriculum": args.unified_curriculum,
        "measure_energy": args.measure_energy,
        "pin_l2": args.pin_l2,
        "power_cap": power_cap,
        "smoke": args.smoke,
        "environment": environment(),
        "wall_seconds": 0.0,
        "arms": reports,
        "deferred_arms": deferred,
    }
    runs = [(arm, seed) for seed in args.seeds for arm in selected]
    try:
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
                    precision=args.precision,
                    unified_curriculum=args.unified_curriculum,
                    max_steps=args.steps,
                    measure_energy=args.measure_energy,
                    pin_l2=args.pin_l2,
                )
            )
            payload = {
                "status": "running",
                "mode": args.mode,
                "precision": args.precision,
                "tokenizer": args.tokenizer,
                "unified_curriculum": args.unified_curriculum,
                "measure_energy": args.measure_energy,
                "pin_l2": args.pin_l2,
                "power_cap": power_cap,
                "smoke": args.smoke,
                "environment": environment(),
                "wall_seconds": time.monotonic() - started,
                "arms": reports,
                "deferred_arms": deferred,
            }
            (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        restore_power_limit(args.device, power_cap)

    payload["status"] = "partial_budget" if deferred else "complete"
    payload["wall_seconds"] = time.monotonic() - started
    payload["deferred_arms"] = deferred
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{payload['status'].upper()} in {payload['wall_seconds']/60:.1f} min -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
