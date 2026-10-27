"""Strict, serializable configuration. Paths in YAML are relative to that YAML."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import get_type_hints
import math

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 9
    n_heads: int = 8
    hidden_dim: int = 1472
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self):
        _types(self)
        for key in ("vocab_size", "d_model", "n_layers", "n_heads", "hidden_dim", "max_seq_len"):
            if getattr(self, key) <= 0:
                raise ValueError(f"model.{key} must be positive")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be integral and even for RoPE")
        if not 0 <= self.dropout < 1 or self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("invalid dropout, RoPE theta or norm epsilon")


@dataclass
class DataConfig:
    path: str = "data/processed"
    seq_len: int = 512
    batch_size: int = 8
    eval_batch_size: int = 8

    def __post_init__(self):
        _types(self)
        if min(self.seq_len, self.batch_size, self.eval_batch_size) <= 0:
            raise ValueError("data dimensions must be positive")


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 100
    schedule_steps: int = 5000

    def __post_init__(self):
        _types(self)
        if self.lr <= 0 or self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer rates")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1 and 0 <= self.min_lr_ratio <= 1):
            raise ValueError("invalid beta or min_lr_ratio")
        if not 0 <= self.warmup_steps < self.schedule_steps:
            raise ValueError("require 0 <= warmup_steps < schedule_steps")


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"
    precision: str = "fp32"
    deterministic: bool = True
    compile: bool = False
    compile_backend: str = "inductor"
    max_steps: int = 5000
    grad_accum_steps: int = 4
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    log_interval: int = 10
    output_dir: str = "runs/baseline"
    cpu_threads: int = 1
    accumulation_weighting: str = "microbatch"
    console_progress: bool = False

    def __post_init__(self):
        _types(self)
        if self.device not in ("cpu", "cuda") or self.precision not in ("fp32", "bf16"):
            raise ValueError("device: cpu|cuda; precision: fp32|bf16")
        if self.precision == "bf16" and self.device != "cuda":
            raise ValueError("BF16 configuration requires CUDA; use FP32 for CPU tests")
        for key in ("max_steps", "grad_accum_steps", "eval_interval", "eval_batches",
                    "checkpoint_interval", "log_interval", "cpu_threads"):
            if getattr(self, key) <= 0:
                raise ValueError(f"train.{key} must be positive")
        if self.grad_clip <= 0 or self.seed < 0 or self.seed >= 2**32:
            raise ValueError("grad_clip must be positive and seed must fit uint32")
        if self.accumulation_weighting not in ("microbatch", "tokens"):
            raise ValueError("accumulation_weighting must be microbatch|tokens")


def _types(obj):
    for name, expected in get_type_hints(type(obj)).items():
        value = getattr(obj, name)
        valid = type(value) is expected
        if expected is float:
            valid = type(value) in (float, int) and math.isfinite(value)
        if not valid:
            raise ValueError(f"{type(obj).__name__}.{name}: expected {expected.__name__}")


@dataclass
class ResearchConfig:
    planning_only: bool = False
    architecture: str = "transformer"
    iterations: int = 2
    core_layers: int = 1
    share_weights: bool = True
    latent_steps: int = 0
    latent_warmup_steps: int = 0
    detach_every: int = 0
    gradient_checkpointing: bool = False
    record_states: bool = False
    lm_weight: float = 1.0
    jepa_weight: float = 0.0
    jepa_offset: int = 1
    ema_decay: float = 0.99
    use_ema: bool = True
    variance_weight: float = 0.1
    variance_target: float = 0.5
    intermediate_weight: float = 0.0
    mtp_weight: float = 0.0
    mtp_tokens: int = 0
    curriculum: str = "off"
    text_fraction: float = 0.5
    exploration: float = 0.1
    max_difficulty: int = 3
    curriculum_interval: int = 20
    probe_examples: int = 2
    fixed_curriculum_steps: int = 100
    max_seconds: float = 0.0
    shutdown_reserve_seconds: float = 30.0
    max_tokens: int = 0
    max_train_flops: int = 0
    scales: list = field(default_factory=lambda: [1, 4, 0])
    share_scales: bool = True
    scale_conditioning: bool = True
    scale_execution: str = "reference"
    jepa_hidden_only: bool = True
    v_cycle: bool = False
    ncm_slots: int = 0
    fused_film: bool = False

    def __post_init__(self):
        _types(self)
        if self.architecture not in ("transformer", "recursive", "untied", "fractal"):
            raise ValueError("unknown research architecture")
        if self.scale_execution not in ("reference", "batched", "compressed", "compressed_native", "compressed_grouped"):
            raise ValueError("unknown scale_execution")
        if self.scale_execution != "reference" and (self.architecture != "fractal" or not self.share_scales):
            raise ValueError("non-reference scales require a fractal model with shared scale weights")
        if self.scale_execution.startswith("compressed") and self.scales != [1, 4, 0]:
            raise ValueError("compressed modes require scales=[1,4,0]; native block factors are fixed at 1,4,16")
        if self.fused_film and self.architecture != "fractal":
            raise ValueError("fused_film requires a fractal model")
        if (self.v_cycle or self.ncm_slots > 0) and (
            self.architecture != "fractal" or not self.scale_execution.startswith("compressed")
        ):
            raise ValueError("v_cycle and ncm_slots require a compressed fractal model")
        for name in ("iterations", "core_layers", "jepa_offset", "max_difficulty", "curriculum_interval",
                     "probe_examples", "fixed_curriculum_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"research.{name} must be positive")
        for name in ("latent_steps", "latent_warmup_steps", "detach_every", "mtp_tokens", "max_tokens",
                     "max_train_flops", "ncm_slots"):
            if getattr(self, name) < 0:
                raise ValueError(f"research.{name} must be nonnegative")
        for name in ("jepa_weight", "variance_weight", "intermediate_weight", "mtp_weight", "max_seconds", "shutdown_reserve_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"research.{name} must be nonnegative")
        if self.lm_weight <= 0 or self.variance_target <= 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("LM must remain active; invalid EMA/variance settings")
        if not 0 <= self.text_fraction <= 1 or not 0 < self.exploration <= 1:
            raise ValueError("invalid text fraction or exploration floor")
        if self.curriculum not in ("off", "uniform", "fixed", "adaptive"):
            raise ValueError("curriculum must be off|uniform|fixed|adaptive")
        if (self.mtp_tokens > 0) != (self.mtp_weight > 0):
            raise ValueError("MTP requires both mtp_tokens and mtp_weight > 0")
        if any(type(s) is not int or s < 0 for s in self.scales):
            raise ValueError("scales are nonnegative integers; 0 denotes the full causal prefix")
        if not self.scales or self.scales[0] != 1 or len(set(self.scales)) != len(self.scales):
            raise ValueError("scales must be unique and start with local scale 1")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    research: ResearchConfig | None = None

    def __post_init__(self):
        if self.data.seq_len > self.model.max_seq_len:
            raise ValueError("data.seq_len exceeds model.max_seq_len")
        if self.train.max_steps > self.optimizer.schedule_steps:
            raise ValueError("max_steps exceeds fixed schedule_steps")
        if self.research is not None:
            if max(self.research.jepa_offset if self.research.jepa_weight else 0,
                   self.research.mtp_tokens) >= self.data.seq_len:
                raise ValueError("auxiliary prediction offsets must be shorter than seq_len")
            if self.research.curriculum != "off" and self.data.seq_len < 128:
                raise ValueError("procedural curriculum requires seq_len >= 128 to preserve complete problems")
            if self.research.architecture == "transformer" and self.research.intermediate_weight:
                raise ValueError("intermediate supervision requires a recursive backbone")

    def to_dict(self):
        raw = asdict(self)
        if self.research is None:
            raw.pop("research")  # Legacy configs keep their original sections; new fields use defaults.
        return raw

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a mapping")
        constructors = {"model": ModelConfig, "data": DataConfig,
                        "optimizer": OptimizerConfig, "train": TrainConfig, "research": ResearchConfig}
        unknown = set(raw) - constructors.keys()
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        values = {}
        for name, constructor in constructors.items():
            if name == "research" and raw.get(name) is None:
                values[name] = None
                continue
            section = raw.get(name, {})
            if not isinstance(section, dict):
                raise ValueError(f"{name} must be a mapping")
            unknown = set(section) - {f.name for f in fields(constructor)}
            if unknown:
                raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
            values[name] = constructor(**section)
        return cls(**values)


def load_config(path):
    path = Path(path).resolve()
    config = ExperimentConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    config.data.path = str((path.parent / config.data.path).resolve())
    config.train.output_dir = str((path.parent / config.train.output_dir).resolve())
    return config
