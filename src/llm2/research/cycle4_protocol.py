"""Predeclared scale-removal diagnosis. Independent of the immutable Cycle 3."""
import hashlib
import json
from pathlib import Path

from ..config import ExperimentConfig, load_config
from .cycle2_protocol import configuration_stats
from .cycle3_protocol import DATA_HASHES, verify_dataset

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (211, 212, 213)
VARIANTS = ("flrm_fine", "flrm_4", "flrm_16", "flrm_full", "lrm_cached", "transformer_two")
FACTORS = {"flrm_fine": [], "flrm_4": [4], "flrm_16": [16], "flrm_full": [4, 16]}
PROTOCOL = {"cycle": 4, "seeds": SEEDS, "variants": VARIANTS, "factors": FACTORS,
    "steps": 6000, "milestones": [1500, 3000, 4500, 6000], "context": 256, "batch": 32,
    "precision": "fp32", "compiler": "inductor", "dataset_files": DATA_HASHES,
    "primary": "flrm_full-minus-flrm_fine", "quality_margin": .005,
    "simplification_loss_margin": .002, "speed_fraction": .10,
    "scope": "equal targets; exploratory component attribution, no general efficiency confirmation"}
PROTOCOL_ID = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def configuration(variant, data, output, seed, *, device="cuda", smoke=False):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("unknown Cycle 4 variant/seed")
    cfg = load_config(ROOT / "configs/experiments/cycle3/base.yaml")
    cfg.data.path = str(data)
    cfg.data.batch_size = 2 if smoke else 32
    cfg.train.output_dir = str(output)
    cfg.train.seed = seed
    cfg.train.device = device
    cfg.train.precision = "fp32"
    cfg.train.deterministic = device == "cpu"
    cfg.train.compile = device == "cuda"
    cfg.train.cpu_threads = 2
    cfg.train.grad_accum_steps = 2 if smoke else 1
    cfg.train.accumulation_weighting = "tokens"
    cfg.train.console_progress = True
    cfg.train.max_steps = cfg.optimizer.schedule_steps = 4 if smoke else 6000
    cfg.optimizer.warmup_steps = 1 if smoke else 50
    cfg.train.checkpoint_interval = cfg.train.eval_interval = cfg.train.log_interval = cfg.train.max_steps // 4
    cfg.train.eval_batches = 8
    cfg.research.architecture = "fractal" if variant in FACTORS else "recursive"
    cfg.research.scale_conditioning = False
    cfg.research.scale_execution = "compressed_native" if variant in FACTORS else "reference"
    cfg.research.compressed_factors = FACTORS.get(variant)
    cfg.research.cache_context_projection = True
    cfg.research.paired_common_init = True
    cfg.research.max_seconds = cfg.research.shutdown_reserve_seconds = 0.
    cfg.research.context_normalization = "none"
    cfg.research.iterations = 2
    cfg.model.n_layers = 2
    cfg.model.hidden_dim = 1024
    if variant == "transformer_two":
        cfg.research.architecture = "transformer"
        cfg.research.cache_context_projection = cfg.research.paired_common_init = False
        cfg.model.hidden_dim = 384
    if smoke:
        cfg.model.d_model = 32; cfg.model.n_heads = 4; cfg.model.hidden_dim = 64
        cfg.model.max_seq_len = cfg.data.seq_len = 17
    return ExperimentConfig.from_dict(cfg.to_dict())


def ordered_jobs(data, output, gpus, *, device="cuda", smoke=False):
    jobs = []
    for index, seed in enumerate(SEEDS):
        order = VARIANTS[index:] + VARIANTS[:index]
        for variant in order:
            identity = f"{variant}-seed-{seed}"
            cfg = configuration(variant, data, Path(output)/"runs"/identity, seed, device=device, smoke=smoke)
            stats = configuration_stats(cfg)
            jobs.append({"id": identity, "model": variant, "variant": variant, "arm": variant,
                "seed": seed, "status": "pending", "max_steps": cfg.train.max_steps,
                "gpu_slot": str(gpus[(VARIANTS.index(variant)+index)%len(gpus)]),
                "config_resolved": cfg.to_dict(), "directory": cfg.train.output_dir,
                "progress_interval": cfg.train.log_interval, "batch_size": cfg.data.batch_size,
                "grad_accum_steps": cfg.train.grad_accum_steps,
                "planned_training_flops_estimate": stats["step_flops_estimated"]*cfg.train.max_steps,
                **stats})
    return jobs
