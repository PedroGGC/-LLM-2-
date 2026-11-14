"""Strict, serializable configuration. Paths in YAML are relative to that YAML."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import get_type_hints, get_args, get_origin, Union
from types import UnionType
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
    def matches(value, expected):
        origin = get_origin(expected)
        if origin in (Union, UnionType):
            return any(matches(value, item) for item in get_args(expected))
        if origin is list:
            return type(value) is list and all(matches(item, get_args(expected)[0]) for item in value)
        if expected is float:
            return type(value) in (float, int) and math.isfinite(value)
        return type(value) is expected
    for name, expected in get_type_hints(type(obj)).items():
        value = getattr(obj, name)
        if not matches(value, expected):
            raise ValueError(f"{type(obj).__name__}.{name}: expected {expected}")


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
    inv_vcycle: bool = False
    ncm_slots: int = 0
    fused_film: bool = False
    cache_context_projection: bool = False
    paired_common_init: bool = False
    context_normalization: str = "none"
    cross_scale_exchange: bool = True
    compressed_factors: list[int] | None = None
    state_rule: str = "gated_delta"
    state_execution: str = "chunked"
    state_chunk_size: int = 32
    state_factor: int = 4
    state_readout: str = "next"
    memory_pattern: str = "none"
    memory_slots: int = 2048
    memory_reads: int = 16
    memory_dim: int = 96
    associative_levels: int = 1
    associative_execution: str = "reference"
    associative_block_size: int = 32
    fractal_beta_scale: float = 2.0
    fractal_window: int = 0
    fractal_value_residual: bool = False

    def __post_init__(self):
        _types(self)
        if self.compressed_factors is not None:
            if self.compressed_factors not in ([], [4], [16], [4, 16]):
                raise ValueError("compressed_factors must be [], [4], [16], [4,16] or None")
            if self.architecture != "fractal" or self.scale_execution != "compressed_native":
                raise ValueError("compressed_factors requires compressed_native fractal execution")
            if self.compressed_factors != [4, 16] and (
                self.scale_conditioning or self.fused_film or self.v_cycle or self.inv_vcycle or self.ncm_slots
            ):
                raise ValueError("reduced compressed_factors requires FiLM/NCM/V-Cycles disabled")
        if self.context_normalization not in ("none", "rms"):
            raise ValueError("context_normalization must be none|rms")
        if self.context_normalization != "none" and self.architecture == "transformer":
            raise ValueError("context_normalization requires a recurrent architecture")
        if not self.cross_scale_exchange and self.architecture != "fractal":
            raise ValueError("cross_scale_exchange=False requires fractal architecture")
        if self.architecture not in ("transformer", "recursive", "untied", "fractal", "state_recurrent", "state_hierarchy", "transition_memory", "context_memory", "context_hierarchy", "fractal_delta"):
            raise ValueError("unknown research architecture")
        if not 0 < self.fractal_beta_scale <= 2.0:
            raise ValueError("fractal_beta_scale must be in (0, 2]")
        if self.fractal_window != 0 and (self.architecture != "fractal_delta" or not 4 <= self.fractal_window <= 128):
            raise ValueError("fractal_window must be 0 or 4..128 for fractal_delta")
        if self.fractal_value_residual and self.architecture != "fractal_delta":
            raise ValueError("fractal_value_residual requires fractal_delta")
        if self.associative_levels not in (1, 2):
            raise ValueError("associative_levels must be 1 or 2")
        if self.associative_levels != 1 and self.architecture != "transition_memory":
            raise ValueError("associative_levels requires transition_memory")
        if self.associative_execution not in ("reference", "blocked"):
            raise ValueError("unknown associative_execution")
        if self.associative_execution != "reference" and self.architecture != "transition_memory":
            raise ValueError("blocked execution requires transition_memory")
        if not 4 <= self.associative_block_size <= 128:
            raise ValueError("associative_block_size must be 4..128")
        if self.associative_execution == "blocked" and (self.state_factor < 2 or self.associative_block_size % self.state_factor):
            raise ValueError("associative block size must be a multiple of state_factor")
        if self.state_rule not in ("gated_delta", "rwkv7", "mlstm") or self.state_execution not in ("sequential", "chunked"):
            raise ValueError("unknown state rule/execution")
        if not 1 <= self.state_chunk_size <= 64 or self.state_factor < 2:
            raise ValueError("state chunk size must be 1..64 and factor >=2")
        if self.state_readout not in ("next", "immediate"):
            raise ValueError("state_readout must be next|immediate")
        if self.state_readout != "next" and self.architecture != "state_hierarchy":
            raise ValueError("immediate state_readout requires state_hierarchy")
        if self.memory_pattern not in ("none", "all", "uniform", "binary", "fibonacci"):
            raise ValueError("unknown episodic memory pattern")
        if self.memory_slots < 2 or not 2 <= self.memory_reads <= self.memory_slots or self.memory_dim < 2:
            raise ValueError("invalid memory capacity/read dimension")
        if self.memory_pattern != "none" and (self.architecture != "state_hierarchy" or self.state_readout != "immediate"):
            raise ValueError("episodic memory requires immediate state_hierarchy")
        if (self.architecture.startswith(("state_", "context_")) or self.architecture in ("transition_memory", "fractal_delta")) and (
            self.jepa_weight or self.mtp_weight or self.intermediate_weight or self.lm_weight != 1
            or self.latent_steps or self.latent_warmup_steps or self.detach_every
            or self.gradient_checkpointing or self.record_states or self.curriculum != "off"
            or self.context_normalization != "none" or self.cache_context_projection or self.paired_common_init
            or self.core_layers != 1 or self.iterations != 2 or self.scale_conditioning
        ):
            raise ValueError("state backbones require language-only objective and no legacy refinement options")
        if self.scale_execution not in ("reference", "batched", "compressed", "compressed_native", "compressed_grouped"):
            raise ValueError("unknown scale_execution")
        if self.scale_execution != "reference" and (self.architecture != "fractal" or not self.share_scales):
            raise ValueError("non-reference scales require a fractal model with shared scale weights")
        if self.scale_execution.startswith("compressed") and self.scales != [1, 4, 0]:
            raise ValueError("compressed modes require scales=[1,4,0]; native block factors are fixed at 1,4,16")
        if self.fused_film and self.architecture != "fractal":
            raise ValueError("fused_film requires a fractal model")
        if self.cache_context_projection and self.architecture not in ("recursive", "fractal"):
            raise ValueError("cache_context_projection requires a recursive or fractal model")
        if self.cache_context_projection and self.architecture == "fractal" and not self.share_scales:
            raise ValueError("cache_context_projection requires shared scale cores")
        if self.cache_context_projection and self.ncm_slots > 0:
            raise ValueError("cache_context_projection cannot be combined with ncm_slots > 0")
        if self.paired_common_init and self.architecture not in ("recursive", "fractal"):
            raise ValueError("paired_common_init requires a recursive or fractal model")
        if self.paired_common_init and not self.share_weights:
            raise ValueError("paired_common_init requires share_weights=True")
        if self.paired_common_init and self.architecture == "fractal" and not self.share_scales:
            raise ValueError("paired_common_init requires shared scale cores")
        if self.v_cycle and self.inv_vcycle:
            raise ValueError("v_cycle and inv_vcycle are mutually exclusive")
        if (self.v_cycle or self.inv_vcycle or self.ncm_slots > 0) and (
            self.architecture != "fractal" or not self.scale_execution.startswith("compressed")
        ):
            raise ValueError("v_cycle, inv_vcycle and ncm_slots require a compressed fractal model")
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
