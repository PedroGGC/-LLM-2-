"""Experimental aligned and compressed multiscale recurrence; every aggregation is prefix-causal.

Supports reference, batched and compressed scale execution modes.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, TransformerLM
from .latent_recursive import LatentState, SharedRecursiveCore, LatentRecursiveModel


class CausalScaleCompressor(nn.Module):
    """Causal block pooling and causal upsampling with strict future isolation."""

    def __init__(self, factor, dim):
        super().__init__()
        self.factor = factor
        self.dim = dim
        self.prior = nn.Parameter(torch.zeros(1, 1, dim))

    def compress(self, x):
        B, T, D = x.shape
        C = self.factor
        pad = (C - (T % C)) % C
        if pad > 0:
            x = torch.cat([x, x.new_zeros(B, pad, D)], dim=1)
        T_pad = x.shape[1]
        chunks = x.view(B, T_pad // C, C, D)
        if pad > 0:
            counts = torch.full((T_pad // C, 1), C, dtype=x.dtype, device=x.device)
            counts[-1, 0] = C - pad
            compressed = chunks.sum(dim=2) / counts.unsqueeze(0)
        else:
            compressed = chunks.mean(dim=2)
        return compressed

    def upsample_causal(self, compressed, target_len):
        B, T_comp, D = compressed.shape
        C = self.factor
        shifted = torch.cat([self.prior.expand(B, 1, D), compressed[:, :-1, :]], dim=1)
        upsampled = shifted.repeat_interleave(C, dim=1)[:, :target_len, :]
        return upsampled


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

    def forward_batched(self, contexts):
        gamma, beta = self.film.unbind(1)
        return contexts * (1 + 0.1 * gamma.tanh()[:, None, None, :]) + 0.1 * beta[:, None, None, :]


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

    def forward_batched(self, states):
        weights = self.mixing.softmax(-1)
        with torch.autocast(device_type=states.device.type, enabled=False):
            mixed = torch.einsum("ij,jbtd->ibtd", weights.float(), states.float()).to(states.dtype)
        return states + self.gate.sigmoid()[:, None, None, None] * self.projection(mixed)


class SharedFractalCore(SharedRecursiveCore):
    """The exact common recursive operation, reused across scales without duplicate weights."""


class FractalRecursiveBlock(nn.Module):
    def __init__(self, config, research):
        super().__init__()
        count = len(research.scales)
        self.shared = research.share_scales
        self.execution = research.scale_execution
        if self.execution in ("batched", "compressed") and not self.shared:
            raise ValueError(f"{self.execution} execution requires shared scale cores")
        self.cores = nn.ModuleList([SharedFractalCore(config, research.core_layers)
                                   for _ in range(1 if self.shared else count)])
        self.conditioning = ScaleConditioning(count, config.d_model) if research.scale_conditioning else None
        self.interaction = CrossScaleInteraction(count, config.d_model)
        if self.execution == "compressed":
            self.comp4 = CausalScaleCompressor(4, config.d_model)
            self.comp16 = CausalScaleCompressor(16, config.d_model)

    def forward(self, states, contexts):
        if self.execution == "compressed":
            s0, s1, s2 = states
            c0, c1, c2 = contexts
            T = c0.shape[1]
            if self.conditioning is not None:
                c0 = self.conditioning(c0, 0)
                c1 = self.conditioning(c1, 1)
                c2 = self.conditioning(c2, 2)
            v0 = self.cores[0](s0, c0)
            v1 = self.cores[0](s1, c1)
            v2 = self.cores[0](s2, c2)
            u1 = self.comp4.upsample_causal(v1, T)
            u2 = self.comp16.upsample_causal(v2, T)
            mixed = self.interaction((v0, u1, u2))
            next_s0 = mixed[0]
            next_s1 = self.comp4.compress(mixed[1])
            next_s2 = self.comp16.compress(mixed[2])
            return (next_s0, next_s1, next_s2)
        if self.execution == "batched":
            stacked_states, stacked_contexts = torch.stack(states), torch.stack(contexts)
            if self.conditioning is not None:
                stacked_contexts = self.conditioning.forward_batched(stacked_contexts)
            shape = stacked_states.shape
            values = self.cores[0](stacked_states.flatten(0, 1), stacked_contexts.flatten(0, 1))
            return self.interaction.forward_batched(values.reshape(shape)).unbind(0)
        values = []
        for index, (state, context) in enumerate(zip(states, contexts)):
            if self.conditioning is not None:
                context = self.conditioning(context, index)
            values.append(self.cores[0 if self.shared else index](state, context))
        return self.interaction(tuple(values))


class FractalLatentModel(Backbone):
    architecture_version = 1
    supports_hidden_only = True

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

    def forward(self, input_ids, *, state=None, iterations=None, return_logits=True):
        if state is not None:
            raise ValueError("cross-segment state is unsupported; reset for every independent input")
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must be [batch, 1..max_seq_len]")
        steps = self.active_iterations() if iterations is None else iterations
        if type(steps) is not int or steps <= 0:
            raise ValueError("iterations must be positive")
        if self.research.scale_execution == "compressed":
            c0 = self.embedding(input_ids)
            c1 = self.block.comp4.compress(c0)
            c2 = self.block.comp16.compress(c0)
            contexts = (c0, c1, c2)
            states = (torch.zeros_like(c0), torch.zeros_like(c1), torch.zeros_like(c2))
        else:
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
        return ModelOutput(logits=self.lm_head(hidden) if return_logits else None, hidden_states=hidden,
                           state=LatentState(states[0], steps), intermediate_states=tuple(history))

    def analytical_cost(self, seq_len, *, include_logits=True):
        d, h, s = self.config.d_model, self.config.hidden_dim, seq_len
        count, steps = len(self.research.scales), self.active_iterations()
        if self.research.scale_execution == "compressed":
            s1 = (s + 3) // 4
            s2 = (s + 15) // 16
            lengths = (s, s1, s2)
            per_iteration = sum(self.research.core_layers*(8*si*d*d + 6*si*d*h + 4*si*si*d) + 2*si*d*d for si in lengths)
            per_iteration += 2*s*d*d
        else:
            per_core = self.research.core_layers*(8*s*d*d + 6*s*d*h + 4*s*s*d) + 2*s*d*d
            per_iteration = count * (per_core + 2*s*d*d)
        forward = steps * per_iteration + (2*s*d*self.config.vocab_size if include_logits else 0)
        recompute = steps * per_iteration if self.training and self.research.gradient_checkpointing else 0
        return forward, recompute, steps


def fractal_parameter_formula(config, research):
    d, h, n = config.d_model, config.hidden_dim, len(research.scales)
    core = d*d + 2*d + research.core_layers*(4*d*d + 3*d*h + 2*d)
    extra = 2 * d if research.scale_execution == "compressed" else 0
    return (config.vocab_size*d + d + (1 if research.share_scales else n)*core
            + d*d + n*n + n + (2*n*d if research.scale_conditioning else 0) + extra)
