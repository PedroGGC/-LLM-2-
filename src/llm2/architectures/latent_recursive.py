"""Causal continuous-state refinement with explicitly shared neural operations.

Inspired by recurrence/latent computation, not a reproduction of TRM/HRM/Coconut.
No state is reused between forwards. Full BPTT is the default.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..interfaces import Backbone, ModelOutput
from ..model import DecoderBlock, RMSNorm, TransformerLM


@dataclass
class LatentState:
    value: torch.Tensor
    iterations: int


class SharedRecursiveCore(nn.Module):
    def __init__(self, config, depth=1):
        super().__init__()
        self.state_norm = RMSNorm(config.d_model, config.norm_eps)
        self.context_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(depth)])
        self.gate = nn.Parameter(torch.full((config.d_model,), -2.0))

    def project_context(self, context):
        return self.context_projection(context)

    def forward(self, state, context, *, context_is_projected: bool = False):
        projected_context = context if context_is_projected else self.context_projection(context)
        value = self.state_norm(state) + projected_context
        for block in self.blocks:
            value = block(value)
        return state + self.gate.sigmoid() * value


class LatentRecursiveModel(Backbone):
    architecture_version = 1
    supports_hidden_only = True

    def __init__(self, config, research):
        super().__init__()
        self.config, self.research = config, research
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.shared = research.share_weights and research.architecture != "untied"
        if self.shared:
            self.core = SharedRecursiveCore(config, research.core_layers)
        else:
            self.cores = nn.ModuleList([SharedRecursiveCore(config, research.core_layers)
                                       for _ in range(research.iterations + research.latent_steps)])
        self.apply(TransformerLM._init_weights)
        self.lm_head.weight = self.embedding.weight
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

    def active_iterations(self):
        extra = self.research.latent_steps
        if self.training and self.research.latent_warmup_steps:
            step = 0 if self.training_step.device.type == "meta" else int(self.training_step)
            extra = min(extra, step // self.research.latent_warmup_steps)
        return self.research.iterations + extra

    def set_training_step(self, step):
        self.training_step.fill_(step)

    def forward(self, input_ids, *, state=None, iterations=None, return_logits=True):
        if state is not None:
            raise ValueError("cross-segment state is unsupported; each independent input starts from zero")
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must be [batch, 1..max_seq_len]")
        steps = self.active_iterations() if iterations is None else iterations
        if type(steps) is not int or steps <= 0 or (not self.shared and steps > len(self.cores)):
            raise ValueError("invalid iteration count")
        context = self.embedding(input_ids)
        value = torch.zeros_like(context)
        context_is_projected = bool(self.shared and getattr(self.research, "cache_context_projection", False))
        if context_is_projected:
            context = self.core.project_context(context)
        history = []
        for index in range(steps):
            if self.training and self.research.detach_every and index and index % self.research.detach_every == 0:
                value = value.detach()
            core = self.core if self.shared else self.cores[index]
            if self.training and self.research.gradient_checkpointing:
                value = checkpoint(
                    core,
                    value,
                    context,
                    context_is_projected=context_is_projected,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                value = core(value, context, context_is_projected=context_is_projected)
            if self.research.record_states or self.research.intermediate_weight:
                history.append(self.norm(value))
        hidden = self.norm(value)
        return ModelOutput(logits=self.lm_head(hidden) if return_logits else None, hidden_states=hidden,
                           state=LatentState(value, steps), intermediate_states=tuple(history))

    def analytical_cost(self, seq_len, *, include_logits=True):
        c, r, s = self.config, self.research, seq_len
        d, h, steps = c.d_model, c.hidden_dim, self.active_iterations()
        blocks_cost = r.core_layers * (8 * s * d * d + 6 * s * d * h + 4 * s * s * d)
        ctx_proj_cost = 2 * s * d * d
        if self.shared and getattr(r, "cache_context_projection", False):
            forward_iterations = steps * blocks_cost + ctx_proj_cost
            recompute = steps * blocks_cost if r.gradient_checkpointing and self.training else 0
        else:
            forward_iterations = steps * (blocks_cost + ctx_proj_cost)
            recompute = forward_iterations if r.gradient_checkpointing and self.training else 0
        forward = forward_iterations + (2 * s * d * c.vocab_size if include_logits else 0)
        return forward, recompute, steps
