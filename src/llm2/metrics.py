"""Calculated counts, explicitly approximate FLOPs, and persisted scalar metrics."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class FlopEstimate:
    forward_per_sequence: int
    training_per_sequence: int
    method: str = "dense_matmuls_forward_x3_v1"
    is_estimate: bool = True
    exclusions: str = ("norms, RoPE, activations, softmax, loss, optimizer, embedding lookup, "
                       "dropout, validation, generation, compile; attention counted as dense square")


def estimate_transformer_flops(config, seq_len):
    """A multiply-add counts as 2 FLOPs; backward estimated as 2x forward."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    d, h, n, v, s = (config.d_model, config.hidden_dim, config.n_layers, config.vocab_size, seq_len)
    projections = 8 * s * d * d
    swiglu = 6 * s * d * h
    attention = 4 * s * s * d
    logits = 2 * s * d * v  # Tied weights still incur a vocabulary projection.
    forward = n * (projections + swiglu + attention) + logits
    return FlopEstimate(forward, 3 * forward)


class MetricLogger:
    def __init__(self, root, purge_step=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.root / "tensorboard"), purge_step=purge_step)
        self.file = (self.root / "metrics.jsonl").open("a", encoding="utf-8")

    def log(self, step, **metrics):
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite metric: {name}")
                self.writer.add_scalar(name, value, step)
        self.file.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        self.file.flush()
        self.writer.flush()

    def close(self):
        self.writer.close()
        self.file.close()

