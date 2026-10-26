"""Experimental aligned multiscale recurrence; every aggregation is prefix-causal.

Scale grids retain token resolution. This tests functional sharing, not compressed attention.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, TransformerLM
from .latent_recursive import LatentState, SharedRecursiveCore, LatentRecursiveModel


class FractalScaleEncoder(nn.Module):
    def __init__(self, windows):
        super().__init__()
        self.windows = tuple(windows)

    def forward(self, context):
        prefix = context.float().cumsum(1)
        time = context.shape[1]
        positions = torch.arange(1, time + 1, device=context.device).view(1, time, 1)
        scales = []
        for window in self.windows:
            if window == 1:
                scales.append(context)
            elif window == 0 or window >= time:
                scales.append((prefix / positions).to(context.dtype))
            else:
                previous = torch.cat((torch.zeros_like(prefix[:, :window]), prefix[:, :-window]), dim=1)
                scales.append(((prefix - previous) / positions.clamp(max=window)).to(context.dtype))
        return tuple(scales)


class ScaleConditioning(nn.Module):
    def __init__(self, scales, dim):
        super().__init__()
        self.film = nn.Parameter(torch.zeros(scales, 2, dim))

    def forward(self, value, index):
        gamma, beta = self.film[index].unbind(0)
        return value * (1 + 0.1 * gamma.tanh()) + 0.1 * beta


class CrossScaleInteraction(nn.Module):
    def __init__(self, scales, dim):
        super().__init__()
        self.mixing = nn.Parameter(torch.zeros(scales, scales))
        self.projection = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.full((scales,), -2.0))

    def forward(self, states):
        weights = self.mixing.softmax(-1)
        # Mixing is at identical token positions; states at position t see only <=t.
        return tuple(value + self.gate[i].sigmoid() * self.projection(
            sum(weights[i, j] * other for j, other in enumerate(states)))
            for i, value in enumerate(states))


class SharedFractalCore(SharedRecursiveCore):
    """The exact common recursive operation, reused across scales without duplicate weights."""


class FractalRecursiveBlock(nn.Module):
    def __init__(self, config, research):
        super().__init__()
        count = len(research.scales)
        self.shared = research.share_scales
        self.cores = nn.ModuleList([SharedFractalCore(config, research.core_layers)
                                   for _ in range(1 if self.shared else count)])
        self.conditioning = ScaleConditioning(count, config.d_model) if research.scale_conditioning else None
        self.interaction = CrossScaleInteraction(count, config.d_model)

    def forward(self, states, contexts):
        values = []
        for index, (state, context) in enumerate(zip(states, contexts)):
            if self.conditioning is not None:
                context = self.conditioning(context, index)
            values.append(self.cores[0 if self.shared else index](state, context))
        return self.interaction(tuple(values))


class FractalLatentModel(Backbone):
    architecture_version = 1

    def __init__(self, config, research):
        super().__init__()
        if not research.share_weights:
            raise ValueError("fractal recursion shares across depth; use share_scales for F1/F2 ablations")
        self.config, self.research = config, research
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = FractalScaleEncoder(research.scales)
        self.block = FractalRecursiveBlock(config, research)
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(TransformerLM._init_weights)
        self.lm_head.weight = self.embedding.weight
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

    def active_iterations(self):
        return LatentRecursiveModel.active_iterations(self)

    def set_training_step(self, step):
        self.training_step.fill_(step)

    def forward(self, input_ids, *, state=None, iterations=None):
        if state is not None:
            raise ValueError("cross-segment state is unsupported; reset for every independent input")
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must be [batch, 1..max_seq_len]")
        steps = self.active_iterations() if iterations is None else iterations
        if type(steps) is not int or steps <= 0:
            raise ValueError("iterations must be positive")
        contexts = self.encoder(self.embedding(input_ids))
        states = tuple(torch.zeros_like(context) for context in contexts)
        history = []
        for index in range(steps):
            if self.training and self.research.detach_every and index and index % self.research.detach_every == 0:
                states = tuple(value.detach() for value in states)
            if self.training and self.research.gradient_checkpointing:
                states = checkpoint(self.block, states, contexts, use_reentrant=False, preserve_rng_state=True)
            else:
                states = self.block(states, contexts)
            if self.research.record_states or self.research.intermediate_weight:
                history.append(self.norm(states[0]))
        hidden = self.norm(states[0])
        return ModelOutput(logits=self.lm_head(hidden), hidden_states=hidden,
                           state=LatentState(states[0], steps), intermediate_states=tuple(history))

    def analytical_cost(self, seq_len):
        d, h, s = self.config.d_model, self.config.hidden_dim, seq_len
        count, steps = len(self.research.scales), self.active_iterations()
        per_core = self.research.core_layers*(8*s*d*d + 6*s*d*h + 4*s*s*d) + 2*s*d*d
        per_iteration = count * (per_core + 2*s*d*d)
        forward = steps * per_iteration + 2*s*d*self.config.vocab_size
        recompute = steps * per_iteration if self.training and self.research.gradient_checkpointing else 0
        return forward, recompute, steps


def fractal_parameter_formula(config, research):
    d, h, n = config.d_model, config.hidden_dim, len(research.scales)
    core = d*d + 2*d + research.core_layers*(4*d*d + 3*d*h + 2*d)
    return (config.vocab_size*d + d + (1 if research.share_scales else n)*core
            + d*d + n*n + n + (2*n*d if research.scale_conditioning else 0))
