"""Frozen Cycle 2: communication x FiLM, same targets as 0016; no new objectives."""

import copy
import hashlib
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..data import PreparedData, sha256_file
from ..model import count_parameters, transformer_parameter_formula
from .accounting import estimate_research_flops
from .campaign import campaign_config
from .factory import build_backbone, build_objective
from .pairing import iter_common_named_parameters, iter_exclusive_named_parameters


MODELS = ("baseline_small", "lrm_no_jepa_k2", "flrm_exchange_film", "flrm_exchange_no_film",
          "flrm_no_exchange_film", "flrm_no_exchange_no_film")
SEEDS = (81, 82, 83)
DATA_HASHES = {
    "train.bin": "c4bd3f90105078794114101eee4fc520495901556e5be2540e00099d4f9e6232",
    "val.bin": "4de78ff7f29f0ad075954f1a45a0f8c85dc097ddd603026f1496c0a4fe77a08b",
    "tokenizer.json": "f88f84a387d00bb8027db376d50fff07c86c4ee137f1678351521fefe0f8d252",
}
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def configuration(base, model, data, output, seed, *, device="cuda", smoke=False):
    if model not in MODELS:
        raise ValueError(f"unknown Cycle 2 model: {model}")
    architecture_alias = (model if model in MODELS[:2] else "flrm_compressed_native_cached_no_jepa_k2")
    base = ExperimentConfig.from_dict(base.to_dict())
    if not smoke:
        expected_model = {"vocab_size": 257, "d_model": 384, "n_layers": 6, "n_heads": 6,
                          "hidden_dim": 1024, "max_seq_len": 256, "dropout": .05,
                          "norm_eps": 1e-6, "rope_theta": 10000.}
        expected_optimizer = {"lr": .0004, "weight_decay": .05, "beta1": .9, "beta2": .95,
                              "eps": 1e-8, "min_lr_ratio": .1}
        if (any(getattr(base.model, key) != value for key, value in expected_model.items())
                or any(getattr(base.optimizer, key) != value for key, value in expected_optimizer.items())
                or base.data.seq_len != 256 or base.data.eval_batch_size != 8
                or base.train.grad_clip != 1. or base.train.compile_backend != "inductor"):
            raise ValueError("base differs from the frozen Cycle 2 protocol; use a separately named experiment")
    # Fixed horizon: never inherit a longer LR scheduler from a different experiment.
    steps = 4 if smoke else 2000
    base.train.max_steps = steps
    base.optimizer.schedule_steps = steps
    base.optimizer.warmup_steps = 1 if smoke else 50
    if smoke:
        base.model.d_model = 32; base.model.n_heads = 4; base.model.hidden_dim = 64
        base.model.max_seq_len = base.data.seq_len = 17
    cfg = campaign_config(base, architecture_alias, data, output, seed=seed, steps=steps,
                          device=device, max_seconds=0, batch_size=2 if smoke else 32,
                          grad_accum_steps=2 if smoke else 1, compile=not smoke, paired_common_init=True)
    cfg.research.context_normalization = "none"
    cfg.research.scale_conditioning = "no_film" not in model
    cfg.research.cross_scale_exchange = "no_exchange" not in model
    cfg.train.checkpoint_interval = 2 if smoke else 500
    cfg.train.eval_interval = 2 if smoke else 100
    cfg.train.log_interval = 1 if smoke else 20
    cfg.research.shutdown_reserve_seconds = 0.
    return ExperimentConfig.from_dict(cfg.to_dict())


def ordered_jobs(seeds, gpus):
    """One process at a time. Rotate order; each variant swaps GPUs across seed blocks."""
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        raise ValueError("select unique Cycle 2 seeds from 81/82/83")
    slots = list(gpus) or [None]
    jobs = []
    for seed in seeds:
        block = SEEDS.index(seed)
        order = MODELS[2*block:] + MODELS[:2*block]
        for model in order:
            slot = slots[(MODELS.index(model) + block) % len(slots)]
            jobs.append({"id": f"{model}-seed-{seed}", "model": model, "seed": seed,
                         "gpu_slot": slot, "physical_gpu": slot, "status": "pending"})
    return jobs


def verify_dataset(path, *, smoke=False):
    data = PreparedData(path)
    if not smoke and (data.manifest["files"] != DATA_HASHES or data.tokenizer.vocab_size != 257):
        raise ValueError("Cycle 2 requires the exact token files/splits from 0016; refusing a different corpus")
    return data


def configuration_stats(cfg):
    with torch.device("meta"):
        model = build_backbone(cfg)
        objective = build_objective(cfg, model)
    count = count_parameters(model)
    if cfg.research.architecture == "fractal":
        from ..architectures.fractal_recursive import fractal_parameter_formula
        formula = fractal_parameter_formula(cfg.model, cfg.research)
    elif cfg.research.architecture == "transformer":
        formula = transformer_parameter_formula(cfg.model)
    else:
        d, h, v = cfg.model.d_model, cfg.model.hidden_dim, cfg.model.vocab_size
        formula = v*d + d + d*d + 2*d + cfg.research.core_layers*(4*d*d + 3*d*h + 2*d)
    if count != formula:
        raise ValueError("parameter enumeration/formula mismatch")
    flops = estimate_research_flops(model, objective, cfg.data.seq_len)
    return {"registered_parameters": count, "parameters_formula": formula,
            "step_flops_estimated": flops.training_per_sequence * cfg.data.batch_size * cfg.train.grad_accum_steps,
            "flops_exclusions": flops.exclusions,
            "note": "registered parameters differ from tensors receiving gradients; backward estimate is conservative for disconnected branches"}


def paired_fingerprints(configs):
    """Actually hash initialized CPU tensors; ignore only FiLM keys absent in the off arms."""
    common, exclusive, provenance = {}, {}, {}
    for name, config in configs.items():
        if config.research.architecture == "transformer":
            continue
        model = build_backbone(config)
        def hashes(items):
            return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    for key, value in items}
        common[name] = hashes(iter_common_named_parameters(model))
        if config.research.architecture == "fractal":
            exclusive[name] = hashes((key, value) for key, value in iter_exclusive_named_parameters(model)
                                     if not key.startswith("block.conditioning."))
        provenance[name] = copy.deepcopy(model.paired_init_provenance)
    if any(value != next(iter(common.values())) for value in common.values()):
        raise ValueError("common initial parameters differ")
    if any(value != next(iter(exclusive.values())) for value in exclusive.values()):
        raise ValueError("non-FiLM exclusive parameters differ between factorial arms")
    return {"verified_common_equal": True, "verified_non_film_exclusive_equal": True,
            "provenance_by_model": provenance, "non_film_exclusive_hashes": exclusive}
