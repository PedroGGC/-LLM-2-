"""Analytical component estimates; sharing parameters does not remove operations."""

from dataclasses import dataclass

from ..metrics import estimate_transformer_flops
from ..model import TransformerLM


@dataclass(frozen=True)
class ResearchFlopEstimate:
    forward_per_sequence: int
    training_per_sequence: int
    backward_per_sequence: int
    backbone_forward: int
    auxiliary_forward: int
    target_forward: int
    recomputation_forward: int
    internal_iterations: int
    method: str = "research_dense_matmuls_v1"
    is_estimate: bool = True
    exclusions: str = ("elementwise/norms/RoPE/softmax/loss/optimizer; square dense attention; "
                       "backward=2x differentiable forward, conservative under detach; no compile cost")


def backbone_cost(model, seq_len):
    if type(model) is TransformerLM:
        return estimate_transformer_flops(model.config, seq_len).forward_per_sequence, 0, model.config.n_layers
    if hasattr(model, "analytical_cost"):
        return model.analytical_cost(seq_len)
    from ..architectures.latent_recursive import LatentRecursiveModel
    if type(model) is not LatentRecursiveModel:
        return None
    c, r, s = model.config, model.research, seq_len
    d, h, steps = c.d_model, c.hidden_dim, model.active_iterations()
    core = r.core_layers * (8*s*d*d + 6*s*d*h + 4*s*s*d) + 2*s*d*d
    forward = steps * core + 2*s*d*c.vocab_size
    recompute = steps * core if r.gradient_checkpointing and model.training else 0
    return forward, recompute, steps


def estimate_research_flops(model, objective, seq_len):
    cost = backbone_cost(model, seq_len)
    if cost is None:
        return None
    forward, recompute, steps = cost
    auxiliary, target = 0, 0
    if hasattr(objective, "research"):
        r, d, v, s = objective.research, model.config.d_model, model.config.vocab_size, seq_len
        if r.jepa_weight:
            auxiliary += 4 * (s - r.jepa_offset) * d * d
            target = backbone_cost(objective.target, seq_len)[0]
        auxiliary += sum(2 * (s-k) * d * v for k in range(1, r.mtp_tokens + 1))
        if r.intermediate_weight:
            auxiliary += steps * 2*s*d*v
    backward = 2 * (forward + auxiliary)
    return ResearchFlopEstimate(forward + auxiliary + target, forward + auxiliary + target + backward + recompute,
                                backward, forward, auxiliary, target, recompute, steps)
