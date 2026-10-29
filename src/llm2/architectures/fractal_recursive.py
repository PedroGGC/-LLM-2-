"""Experimental aligned and compressed multiscale recurrence; every aggregation is prefix-causal.

Supports reference, batched and compressed scale execution modes, including asymmetric V-Cycle
refinement, vectorized FiLM/CrossScale kernel fusion, and Neural Context Memory (NCM) on macro scales.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, TransformerLM
from .latent_recursive import LatentState, SharedRecursiveCore, LatentRecursiveModel
from .ncm import NeuralContextMemory


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

    def compress_stacked(self, x3: torch.Tensor) -> torch.Tensor:
        """Compress 3 stacked fine-scale tensors (3, B, T, D) in a single batched op."""
        S, B, T, D = x3.shape
        flat = x3.reshape(S * B, T, D)
        comp = self.compress(flat)
        return comp.reshape(S, B, comp.shape[1], D)

    def shift_causal(self, compressed):
        B, _, D = compressed.shape
        return torch.cat([self.prior.expand(B, 1, D), compressed[:, :-1, :]], dim=1)

    def upsample_causal(self, compressed, target_len):
        shifted = self.shift_causal(compressed)
        return shifted.repeat_interleave(self.factor, dim=1)[:, :target_len, :]


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
    def __init__(self, scales, dim, *, fused=False):
        super().__init__()
        self.film = nn.Parameter(torch.zeros(scales, 2, dim))
        self.fused = fused

    def forward(self, value, index):
        gamma, beta = self.film[index].unbind(0)
        if self.fused:
            return torch.addcmul(0.1 * beta, value, 1.0 + 0.1 * gamma.tanh())
        return value * (1 + 0.1 * gamma.tanh()) + 0.1 * beta

    def forward_scales(self, c0, c1, c2):
        """Condition all 3 scales at once with a single vectorized tanh/gain computation."""
        gains = 1.0 + 0.1 * self.film[:, 0].tanh()
        biases = 0.1 * self.film[:, 1]
        return (
            torch.addcmul(biases[0], c0, gains[0]),
            torch.addcmul(biases[1], c1, gains[1]),
            torch.addcmul(biases[2], c2, gains[2]),
        )

    def forward_batched(self, contexts):
        gamma, beta = self.film.unbind(1)
        return contexts * (1 + 0.1 * gamma.tanh()[:, None, None, :]) + 0.1 * beta[:, None, None, :]


class CrossScaleInteraction(nn.Module):
    def __init__(self, scales, dim, *, fused=False, exchange=True):
        super().__init__()
        self.exchange = exchange
        self.mixing = nn.Parameter(torch.zeros(scales, scales))
        self.projection = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.full((scales,), -2.0))
        self.fused = fused

    def forward(self, states):
        if not self.exchange:
            return tuple(value + self.gate[i].sigmoid() * self.projection(value)
                         for i, value in enumerate(states))
        weights = self.mixing.softmax(-1)
        # Mixing is at identical token positions; states at position t see only <=t.
        return tuple(value + self.gate[i].sigmoid() * self.projection(
            sum(weights[i, j] * other for j, other in enumerate(states)))
            for i, value in enumerate(states))

    def forward_batched(self, states):
        if not self.exchange:
            return states + self.gate.sigmoid()[:, None, None, None] * self.projection(states)
        weights = self.mixing.softmax(-1)
        with torch.autocast(device_type=states.device.type, enabled=False):
            mixed = torch.einsum("ij,jbtd->ibtd", weights.float(), states.float()).to(states.dtype)
        return states + self.gate.sigmoid()[:, None, None, None] * self.projection(mixed)

    def forward_native(self, states, compressors, *, raw_macro=None):
        """Pool before each coarse projection, preserving the delayed read semantics.

        Mean(W x) = W mean(x). Pooling each input once per output scale avoids
        projecting all three outputs at fine resolution; rounding can differ.
        """
        if not self.exchange:
            own = (states[0], *(comp.compress(states[i + 1])
                               for i, comp in enumerate(compressors)))
            return self.forward(own)
        if self.fused and len(states) == 3 and len(compressors) == 2:
            s0, s1, s2 = states
            comp4, comp16 = compressors
            t0 = s0.shape[1]
            s_fine = torch.stack((s0, s1, s2), dim=0)
            if raw_macro is not None:
                v1, v2 = raw_macro
                # Exact algebraic shortcut:
                # comp4(upsample4(v1, T)) == shift_causal(v1)
                # comp4(upsample16(v2, T)) == repeat_interleave(shift_causal(v2), 4)[:T1]
                # comp16(upsample16(v2, T)) == shift_causal(v2)
                c1_v0 = comp4.compress(s0)
                t1 = c1_v0.shape[1]
                c1_u1 = comp4.shift_causal(v1)
                s_v2 = comp16.shift_causal(v2)
                c1_u2 = s_v2.repeat_interleave(4, dim=1)[:, :t1, :]
                c1_3 = torch.stack((c1_v0, c1_u1, c1_u2), dim=0)
                # Only compress (s0, u1) at factor 16 (2 tensors instead of 3), and reuse s_v2 for u2:
                c2_01 = comp16.compress_stacked(torch.stack((s0, s1), dim=0))
                t2 = s_v2.shape[1]
                c2_3 = torch.cat((c2_01, s_v2.unsqueeze(0)), dim=0)
            else:
                c1_3 = comp4.compress_stacked(s_fine)
                c2_3 = comp16.compress_stacked(s_fine)
                t1, t2 = c1_3.shape[2], c2_3.shape[2]
            weights = self.mixing.softmax(-1)
            # Compute only the needed scale row per resolution (no wasted 2/3 combinations):
            m0 = torch.einsum("j,jbtd->btd", weights[0], s_fine)
            m1 = torch.einsum("j,jbtd->btd", weights[1], c1_3)
            m2 = torch.einsum("j,jbtd->btd", weights[2], c2_3)
            packed_mixed = torch.cat((m0, m1, m2), dim=1)
            # Single Linear projection call instead of 3 separate calls:
            projected = self.projection(packed_mixed)
            g = self.gate.sigmoid()
            p0, p1, p2 = projected[:, :t0], projected[:, t0:t0 + t1], projected[:, t0 + t1:]
            return (
                torch.addcmul(s0, p0, g[0]),
                torch.addcmul(c1_3[1], p1, g[1]),
                torch.addcmul(c2_3[2], p2, g[2]),
            )
        weights = self.mixing.softmax(-1)
        outputs = []
        for i, compressor in enumerate((None, *compressors)):
            values = states if compressor is None else tuple(compressor.compress(x) for x in states)
            mixed = sum(weights[i, j] * value for j, value in enumerate(values))
            outputs.append(values[i] + self.gate[i].sigmoid() * self.projection(mixed))
        return tuple(outputs)


class SharedFractalCore(SharedRecursiveCore):
    """The exact common recursive operation, reused across scales without duplicate weights."""


class FractalRecursiveBlock(nn.Module):
    def __init__(self, config, research):
        super().__init__()
        count = len(research.scales)
        self.shared = research.share_scales
        self.execution = research.scale_execution
        self.v_cycle = research.v_cycle
        self.inv_vcycle = research.inv_vcycle
        self.fused_film = research.fused_film
        if self.execution != "reference" and not self.shared:
            raise ValueError(f"{self.execution} execution requires shared scale cores")
        self.cores = nn.ModuleList([SharedFractalCore(config, research.core_layers,
                                   context_normalization=research.context_normalization)
                                   for _ in range(1 if self.shared else count)])
        self.conditioning = (
            ScaleConditioning(count, config.d_model, fused=research.fused_film)
            if research.scale_conditioning
            else None
        )
        self.interaction = CrossScaleInteraction(count, config.d_model, fused=research.fused_film,
                                                 exchange=research.cross_scale_exchange)
        if self.execution.startswith("compressed"):
            self.comp4 = CausalScaleCompressor(4, config.d_model)
            self.comp16 = CausalScaleCompressor(16, config.d_model)
        self.ncm = (
            NeuralContextMemory(research.ncm_slots, config.d_model, init_parameters=False)
            if research.ncm_slots > 0
            else None
        )

    def forward(
        self,
        states,
        contexts,
        step_index: int = 0,
        preconditioned: bool = False,
        context_is_projected: bool = False,
    ):
        if self.execution.startswith("compressed"):
            s0, s1, s2 = states
            c0, c1, c2 = contexts
            T = c0.shape[1]
            if self.conditioning is not None and not preconditioned and not context_is_projected:
                if self.fused_film:
                    c0, c1, c2 = self.conditioning.forward_scales(c0, c1, c2)
                else:
                    c0 = self.conditioning(c0, 0)
                    c1 = self.conditioning(c1, 1)
                    c2 = self.conditioning(c2, 2)
            skip_fine_core = (self.v_cycle and step_index > 0) or (self.inv_vcycle and step_index == 0)
            v0 = s0 if skip_fine_core else self.cores[0](s0, c0, context_is_projected=context_is_projected)
            if self.execution == "compressed_grouped":
                # Pad only the shorter coarse branch. Causal valid queries cannot
                # read this right padding. Dropout draws differ from separate calls.
                length = s1.shape[1]
                pad = length - s2.shape[1]
                padded_s2 = torch.nn.functional.pad(s2, (0, 0, 0, pad))
                padded_c2 = torch.nn.functional.pad(c2, (0, 0, 0, pad))
                grouped_s = torch.cat((s1, padded_s2))
                grouped_c = torch.cat((c1, padded_c2))
                if self.ncm is not None:
                    grouped_c = self.ncm(grouped_c)
                values = self.cores[0](grouped_s, grouped_c, context_is_projected=context_is_projected)
                if self.ncm is not None:
                    values = self.ncm(values)
                v1, v2 = values.chunk(2, dim=0)
                v2 = v2[:, :s2.shape[1]]
            else:
                c1_in = self.ncm(c1) if self.ncm is not None else c1
                c2_in = self.ncm(c2) if self.ncm is not None else c2
                v1 = self.cores[0](s1, c1_in, context_is_projected=context_is_projected)
                v2 = self.cores[0](s2, c2_in, context_is_projected=context_is_projected)
                if self.ncm is not None:
                    v1 = self.ncm(v1)
                    v2 = self.ncm(v2)
            u1 = self.comp4.upsample_causal(v1, T)
            u2 = self.comp16.upsample_causal(v2, T)
            if self.execution != "compressed":
                return self.interaction.forward_native((v0, u1, u2), (self.comp4, self.comp16), raw_macro=(v1, v2))
            mixed = self.interaction((v0, u1, u2))
            next_s0 = mixed[0]
            next_s1 = self.comp4.compress(mixed[1])
            next_s2 = self.comp16.compress(mixed[2])
            return (next_s0, next_s1, next_s2)
        if self.execution == "batched":
            stacked_states, stacked_contexts = torch.stack(states), torch.stack(contexts)
            if self.conditioning is not None and not preconditioned and not context_is_projected:
                stacked_contexts = self.conditioning.forward_batched(stacked_contexts)
            shape = stacked_states.shape
            values = self.cores[0](
                stacked_states.flatten(0, 1),
                stacked_contexts.flatten(0, 1),
                context_is_projected=context_is_projected,
            )
            return self.interaction.forward_batched(values.reshape(shape)).unbind(0)
        values = []
        for index, (state, context) in enumerate(zip(states, contexts)):
            if self.conditioning is not None and not preconditioned and not context_is_projected:
                context = self.conditioning(context, index)
            values.append(
                self.cores[0 if self.shared else index](
                    state,
                    context,
                    context_is_projected=context_is_projected,
                )
            )
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
        if self.block.ncm is not None:
            self.block.ncm.reset_parameters()
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
        preconditioned = False
        context_is_projected = bool(
            self.research.share_scales and getattr(self.research, "cache_context_projection", False)
        )
        if self.research.scale_execution.startswith("compressed"):
            c0 = self.embedding(input_ids)
            c1 = self.block.comp4.compress(c0)
            c2 = self.block.comp16.compress(c0)
            states = (torch.zeros_like(c0), torch.zeros_like(c1), torch.zeros_like(c2))
            if (self.research.fused_film or context_is_projected) and self.block.conditioning is not None:
                if self.research.fused_film:
                    c0, c1, c2 = self.block.conditioning.forward_scales(c0, c1, c2)
                else:
                    c0 = self.block.conditioning(c0, 0)
                    c1 = self.block.conditioning(c1, 1)
                    c2 = self.block.conditioning(c2, 2)
                preconditioned = True
            if context_is_projected:
                c0 = self.block.cores[0].project_context(c0)
                c1 = self.block.cores[0].project_context(c1)
                c2 = self.block.cores[0].project_context(c2)
            contexts = (c0, c1, c2)
        else:
            contexts = self.encoder(self.embedding(input_ids))
            states = tuple(torch.zeros_like(context) for context in contexts)
            if context_is_projected:
                if self.block.conditioning is not None:
                    contexts = tuple(
                        self.block.conditioning(context, index) for index, context in enumerate(contexts)
                    )
                    preconditioned = True
                contexts = tuple(self.block.cores[0].project_context(context) for context in contexts)
        history = []
        for index in range(steps):
            if self.training and self.research.detach_every and index and index % self.research.detach_every == 0:
                states = tuple(value.detach() for value in states)
            if self.training and self.research.gradient_checkpointing:
                states = checkpoint(
                    self.block,
                    states,
                    contexts,
                    index,
                    preconditioned,
                    context_is_projected,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                states = self.block(states, contexts, index, preconditioned, context_is_projected)
            if self.research.record_states or self.research.intermediate_weight:
                history.append(self.norm(states[0]))
        hidden = self.norm(states[0])
        return ModelOutput(logits=self.lm_head(hidden) if return_logits else None, hidden_states=hidden,
                           state=LatentState(states[0], steps), intermediate_states=tuple(history))

    def analytical_cost(self, seq_len, *, include_logits=True):
        d, h, s = self.config.d_model, self.config.hidden_dim, seq_len
        count, steps = len(self.research.scales), self.active_iterations()
        cached_ctx = bool(self.research.share_scales and getattr(self.research, "cache_context_projection", False))
        if self.research.scale_execution.startswith("compressed"):
            s1 = (s + 3) // 4
            s2 = (s + 15) // 16
            lengths = (s, s1, s2)
            # Grouped mode computes both short branches at ceil(T/4).
            full_core_lengths = (s, s1, s1) if self.research.scale_execution == "compressed_grouped" else lengths
            macro_core_lengths = (s1, s1) if self.research.scale_execution == "compressed_grouped" else (s1, s2)
            projection_positions = 3 * s if self.research.scale_execution == "compressed" else sum(lengths)
            projection_flops = 2 * projection_positions * d * d
            ncm_positions = (2 * s1) if self.research.scale_execution == "compressed_grouped" else (s1 + s2)
            ncm_flops = 2 * (4 * ncm_positions * self.research.ncm_slots * d) if self.research.ncm_slots > 0 else 0
            forward_iterations = 0
            for index in range(steps):
                skip_fine = (self.research.v_cycle and index > 0) or (self.research.inv_vcycle and index == 0)
                active_lengths = macro_core_lengths if skip_fine else full_core_lengths
                core_flops = sum(
                    self.research.core_layers * (8 * si * d * d + 6 * si * d * h + 4 * si * si * d)
                    + (0 if cached_ctx else 2 * si * d * d)
                    for si in active_lengths
                )
                forward_iterations += core_flops + projection_flops + ncm_flops
            # Static cached context projection occurs before grouped padding on unpadded scale lengths (s, s1, s2).
            static_ctx_flops = sum(2 * si * d * d for si in lengths) if cached_ctx else 0
            forward = static_ctx_flops + forward_iterations + (2 * s * d * self.config.vocab_size if include_logits else 0)
            recompute = forward_iterations if self.training and self.research.gradient_checkpointing else 0
            return forward, recompute, steps
        core_blocks = self.research.core_layers * (8 * s * d * d + 6 * s * d * h + 4 * s * s * d)
        per_core = core_blocks + (0 if cached_ctx else 2 * s * d * d)
        per_iteration = count * (per_core + 2 * s * d * d)
        static_ctx_flops = (count * 2 * s * d * d) if cached_ctx else 0
        forward = static_ctx_flops + steps * per_iteration + (2 * s * d * self.config.vocab_size if include_logits else 0)
        recompute = steps * per_iteration if self.training and self.research.gradient_checkpointing else 0
        return forward, recompute, steps


def fractal_parameter_formula(config, research):
    d, h, n = config.d_model, config.hidden_dim, len(research.scales)
    core = d*d + 2*d + research.core_layers*(4*d*d + 3*d*h + 2*d)
    extra = 2 * d if research.scale_execution.startswith("compressed") else 0
    ncm = research.ncm_slots * d
    return (config.vocab_size*d + d + (1 if research.share_scales else n)*core
            + d*d + n*n + n + (2*n*d if research.scale_conditioning else 0) + extra + ncm)
