"""Fractal recursive Gated Delta memory with multi-timescale causal state and lag-1 associative induction."""

from copy import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, SwiGLU
from .state_kernels import dplr_chunked, dplr_sequential


def _init_decay_bias(heads, min_tau=4.0, max_tau=4096.0):
    if heads == 1:
        taus = torch.tensor([max_tau], dtype=torch.float32)
    else:
        taus = torch.logspace(math.log10(min_tau), math.log10(max_tau), heads, dtype=torch.float32)
    rates = (-1.0 / taus).exp().clamp(1e-4, 1.0 - 1.0e-4)
    return torch.log(rates / (1.0 - rates))


def _assoc_sequential(q, k, v, prev_mem=None):
    b, t, key_dim = q.shape
    val_dim = v.shape[-1]
    mem = q.new_zeros(b, key_dim, val_dim) if prev_mem is None else prev_mem
    outputs = []
    for idx in range(t):
        mem = mem + k[:, idx].unsqueeze(2) * v[:, idx].unsqueeze(1)
        outputs.append(torch.bmm(q[:, idx].unsqueeze(1), mem).squeeze(1))
    return torch.stack(outputs, dim=1), mem.contiguous()


def _assoc_chunked(q, k, v, prev_mem=None, chunk_size=32):
    b, t, key_dim = q.shape
    val_dim = v.shape[-1]
    if t <= 2 or chunk_size <= 1:
        return _assoc_sequential(q, k, v, prev_mem)

    c = min(chunk_size, t)
    pad = (c - (t % c)) % c
    qf, kf, vf = q, k, v
    if pad:
        qf = F.pad(qf, (0, 0, 0, pad))
        kf = F.pad(kf, (0, 0, 0, pad))
        vf = F.pad(vf, (0, 0, 0, pad))

    n_chunks = qf.shape[1] // c
    qc = qf.reshape(b, n_chunks, c, key_dim)
    kc = kf.reshape(b, n_chunks, c, key_dim)
    vc = vf.reshape(b, n_chunks, c, val_dim)

    upper = torch.triu(torch.ones(c, c, device=q.device, dtype=torch.bool), diagonal=1)
    intra_scores = torch.matmul(qc, kc.transpose(-1, -2)).masked_fill(upper, 0.0)
    intra_out = torch.matmul(intra_scores, vc)
    chunk_deltas = torch.matmul(kc.transpose(-1, -2), vc)

    mem = q.new_zeros(b, key_dim, val_dim) if prev_mem is None else prev_mem
    outputs = []
    for idx in range(n_chunks):
        outputs.append(torch.matmul(qc[:, idx], mem) + intra_out[:, idx])
        mem = mem + chunk_deltas[:, idx]

    out = torch.cat(outputs, dim=1)[:, :t].contiguous()
    return out, mem.contiguous()


@dataclass
class FractalLevelState:
    memory: torch.Tensor
    tail: torch.Tensor


@dataclass
class FractalDeltaState:
    levels: tuple[FractalLevelState, ...]
    assoc_memory: torch.Tensor
    local_tail: torch.Tensor


class FractalLevelAdapter(nn.Module):
    """Lightweight per-level conditioning and Euler residual scaling (all 1D, no weight decay)."""

    def __init__(self, d_model, heads, level, depth):
        super().__init__()
        self.attn_scale = nn.Parameter(torch.ones(d_model))
        self.attn_shift = nn.Parameter(torch.zeros(d_model))
        self.ffn_scale = nn.Parameter(torch.ones(d_model))
        self.ffn_shift = nn.Parameter(torch.zeros(d_model))
        span = (level - 0.5 * (depth - 1)) * 0.5
        self.decay_bias = nn.Parameter(torch.full((heads,), span))
        self.beta_bias = nn.Parameter(torch.zeros(heads))
        self.euler_attn = nn.Parameter(torch.ones(d_model))
        self.euler_ffn = nn.Parameter(torch.ones(d_model))
        self.anchor_gate = nn.Parameter(torch.full((d_model,), -2.0)) if level > 0 else None


class FractalDeltaCell(nn.Module):
    def __init__(self, config, research):
        super().__init__()
        self.d_model = config.d_model
        self.heads = config.n_heads
        self.head_dim = self.d_model // self.heads
        self.hidden_dim = config.hidden_dim
        self.conv_kernel = 4
        self.dropout = config.dropout
        self.execution = research.state_execution
        self.chunk_size = research.state_chunk_size
        self.beta_scale = research.fractal_beta_scale

        self.attn_norm = RMSNorm(self.d_model, eps=config.norm_eps)
        self.conv = nn.Conv1d(self.d_model, self.d_model, self.conv_kernel, groups=self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.decay = nn.Linear(self.d_model, self.heads, bias=True)
        self.beta = nn.Linear(self.d_model, self.heads, bias=True)
        self.out_gate = nn.Linear(self.d_model, self.d_model, bias=False)
        self.head_gamma = nn.Parameter(torch.ones(self.d_model))
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.ffn_norm = RMSNorm(self.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(self.d_model, self.hidden_dim)
        self._reset_parameters(config.n_layers)

    def _reset_parameters(self, depth):
        proj_std = 0.02 / math.sqrt(2 * depth)
        nn.init.normal_(self.conv.weight, std=0.02)
        nn.init.zeros_(self.conv.bias)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.decay.weight, std=0.02)
        with torch.no_grad():
            self.decay.bias.copy_(_init_decay_bias(self.heads))
        nn.init.normal_(self.beta.weight, std=0.02)
        nn.init.constant_(self.beta.bias, -1.0)
        nn.init.normal_(self.out_gate.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=proj_std)
        nn.init.normal_(self.ffn.down.weight, std=proj_std)

    def forward(self, x, state=None, adapter=None):
        b, t, d = x.shape
        normed = self.attn_norm(x)
        if adapter is not None:
            normed = normed * adapter.attn_scale + adapter.attn_shift

        tail_width = self.conv_kernel - 1
        norm_f = normed.float().transpose(1, 2)
        prefix = norm_f.new_zeros(b, d, tail_width) if state is None else state.tail
        padded = torch.cat((prefix, norm_f), dim=2)
        new_tail = padded[:, :, -tail_width:].contiguous()
        conv_out = F.conv1d(padded, self.conv.weight.float(), self.conv.bias.float(), groups=d).transpose(1, 2)
        local = (normed.float() + conv_out).to(normed.dtype)

        q, k, v = self.qkv(local).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        key = F.normalize(k.float(), dim=-1, eps=1e-6)
        query = F.normalize(q.float(), dim=-1, eps=1e-6)

        decay_logits = self.decay(local).transpose(1, 2).float()
        beta_logits = self.beta(local).transpose(1, 2).float()
        if adapter is not None:
            decay_logits = decay_logits + adapter.decay_bias.view(1, self.heads, 1)
            beta_logits = beta_logits + adapter.beta_bias.view(1, self.heads, 1)

        log_w = F.logsigmoid(decay_logits).unsqueeze(-1)
        beta = (self.beta_scale * beta_logits.sigmoid()).unsqueeze(-1)
        w = log_w.exp()
        a = -w * beta * key
        values = beta * v.float()

        prev_mem = None if state is None else state.memory
        if self.execution == "sequential" or t == 1:
            y, new_mem = dplr_sequential(query, key, values, log_w, a, key, prev_mem)
        else:
            y, new_mem = dplr_chunked(query, key, values, log_w, a, key, prev_mem, chunk_size=self.chunk_size)
        new_mem = new_mem.float().contiguous()

        gamma = self.head_gamma.float().view(1, self.heads, 1, self.head_dim)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * gamma
        y = y.transpose(1, 2).contiguous().reshape(b, t, d).to(x.dtype)
        gated = y * F.silu(self.out_gate(normed))
        attn_update = F.dropout(self.out_proj(gated), p=self.dropout, training=self.training)
        if adapter is not None:
            attn_update = attn_update * adapter.euler_attn
        x = x + attn_update

        ffn_in = self.ffn_norm(x)
        if adapter is not None:
            ffn_in = ffn_in * adapter.ffn_scale + adapter.ffn_shift
        ffn_update = F.dropout(self.ffn(ffn_in), p=self.dropout, training=self.training)
        if adapter is not None:
            ffn_update = ffn_update * adapter.euler_ffn
        return x + ffn_update, FractalLevelState(new_mem, new_tail)


class FractalDeltaMemory(Backbone):
    supports_hidden_only = True

    def __init__(self, config, research):
        super().__init__()
        self.config = config
        self.research = research
        self.depth = config.n_layers
        self.shared = research.share_scales
        self.heads = config.n_heads
        self.head_dim = config.d_model // self.heads
        self.assoc_dim = research.memory_dim
        self.chunk_size = research.state_chunk_size

        d = config.d_model
        self.assoc_val_dim = min(d, 8 if d < 32 else 24)
        self.embedding = nn.Embedding(config.vocab_size, d)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Causal lag-1 shared-address associative induction channel.
        self.input_norm = RMSNorm(d, eps=config.norm_eps)
        self.local_conv = nn.Conv1d(d, d, 3, groups=d, bias=False)
        nn.init.normal_(self.local_conv.weight, std=0.02)
        self.local_norm = RMSNorm(d, eps=config.norm_eps)
        self.assoc_address = nn.Linear(d, self.assoc_dim, bias=False)
        self.assoc_value = nn.Linear(d, self.assoc_val_dim, bias=False)
        self.assoc_write_gate = nn.Linear(d, 1, bias=True)
        self.assoc_readout = nn.Linear(self.assoc_val_dim, d, bias=False)
        for layer in (self.assoc_address, self.assoc_value, self.assoc_write_gate, self.assoc_readout):
            nn.init.normal_(layer.weight, std=0.02)
        nn.init.constant_(self.assoc_write_gate.bias, -2.0)
        self.assoc_scale = nn.Parameter(torch.ones(d))

        cell_count = 1 if self.shared else self.depth
        self.cells = nn.ModuleList([FractalDeltaCell(config, research) for _ in range(cell_count)])
        self.adapters = (
            nn.ModuleList([FractalLevelAdapter(d, self.heads, lvl, self.depth) for lvl in range(self.depth)])
            if self.depth > 1
            else nn.ModuleList()
        )
        self.norm = RMSNorm(d, eps=config.norm_eps)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _validate_state(self, state, x):
        if not isinstance(state, FractalDeltaState) or len(state.levels) != self.depth:
            raise ValueError("invalid fractal delta state shape/dtype")
        b, _, d = x.shape
        mem_shape = (b, self.heads, self.head_dim, self.head_dim)
        tail_shape = (b, d, 3)
        assoc_checks = (
            (state.assoc_memory, (b, self.assoc_dim, self.assoc_val_dim)),
            (state.local_tail, (b, d, 2)),
        )
        for tensor, expected in assoc_checks:
            if tensor.shape != expected or tensor.dtype != torch.float32 or tensor.device != x.device:
                raise ValueError("invalid fractal delta state shape/dtype")
        for level_state in state.levels:
            if not isinstance(level_state, FractalLevelState):
                raise ValueError("invalid fractal delta state shape/dtype")
            for tensor, expected in ((level_state.memory, mem_shape), (level_state.tail, tail_shape)):
                if tensor.shape != expected or tensor.dtype != torch.float32 or tensor.device != x.device:
                    raise ValueError("invalid fractal delta state shape/dtype")

    def _run_assoc(self, raw_embed, state=None, *, execution="chunked"):
        b, t, d = raw_embed.shape
        x_norm = self.input_norm(raw_embed).float()
        x_norm_t = x_norm.transpose(1, 2)
        if state is None:
            local_tail = x_norm_t.new_zeros(b, d, 2)
            prev_mem = None
        else:
            local_tail = state.local_tail
            prev_mem = state.assoc_memory

        padded = torch.cat((local_tail, x_norm_t), dim=2)
        next_local_tail = padded[:, :, -2:].contiguous()
        conv_out = F.conv1d(padded, self.local_conv.weight.float(), None, groups=d).transpose(1, 2)
        local = self.local_norm((x_norm + conv_out).to(raw_embed.dtype)).float()

        shifted = padded[:, :, 1:-1].transpose(1, 2)

        q = F.normalize(F.linear(x_norm, self.assoc_address.weight.float()), dim=-1, eps=1e-6)
        k_base = F.normalize(F.linear(shifted, self.assoc_address.weight.float()), dim=-1, eps=1e-6)
        gate = torch.sigmoid(F.linear(local, self.assoc_write_gate.weight.float(), self.assoc_write_gate.bias.float()))
        k = k_base * gate
        v = F.linear(local, self.assoc_value.weight.float())

        if execution == "sequential" or t == 1:
            read, next_mem = _assoc_sequential(q, k, v, prev_mem)
        else:
            read, next_mem = _assoc_chunked(q, k, v, prev_mem, chunk_size=self.chunk_size)

        assoc_out = self.assoc_readout(read.to(raw_embed.dtype)) * self.assoc_scale
        return assoc_out, next_mem.float().contiguous(), next_local_tail

    def forward(self, input_ids, *, state=None, return_logits=True):
        raw = self.embedding(input_ids)
        if state is not None:
            self._validate_state(state, raw)
        exec_mode = self.cells[0].execution
        assoc_out, next_assoc_mem, next_local_tail = self._run_assoc(
            raw, state, execution=exec_mode
        )
        x = F.dropout(raw, p=self.config.dropout, training=self.training)
        anchor = x
        next_levels = []
        for level in range(self.depth):
            adapter = self.adapters[level] if self.depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.cells[0 if self.shared else level]
            prev = None if state is None else state.levels[level]
            x, current = cell(x, prev, adapter=adapter)
            next_levels.append(current)
        x = x + F.dropout(assoc_out, p=self.config.dropout, training=self.training)
        hidden = self.norm(x)
        logits = self.lm_head(hidden) if return_logits else None
        next_state = FractalDeltaState(
            tuple(next_levels),
            next_assoc_mem,
            next_local_tail,
        )
        return ModelOutput(logits=logits, hidden_states=hidden, state=next_state)

    def state_bytes_per_sequence(self):
        d = self.config.d_model
        levels_floats = self.depth * (self.heads * self.head_dim * self.head_dim + d * 3)
        assoc_floats = self.assoc_dim * self.assoc_val_dim + 2 * d
        return (levels_floats + assoc_floats) * 4

    def analytical_cost(self, seq_len, *, include_logits=True):
        d = self.config.d_model
        h = self.heads
        dh = self.head_dim
        ff = self.config.hidden_dim
        per_level = seq_len * (10 * d * d + 4 * d * h + 6 * d * ff + 6 * d * dh)
        assoc_flops = seq_len * (
            4 * d * self.assoc_dim
            + 4 * d * self.assoc_val_dim
            + 6 * d
            + 4 * self.assoc_dim * self.assoc_val_dim
        )
        forward = (
            self.depth * per_level
            + assoc_flops
            + (2 * seq_len * d * self.config.vocab_size if include_logits else 0)
        )
        return forward, 0, self.depth

    def parameter_formula(self):
        c = self.config
        d, h, ff, k = c.d_model, self.heads, c.hidden_dim, 4
        embed = c.vocab_size * d
        assoc_params = d + 3 * d + d + d * self.assoc_dim + d * self.assoc_val_dim + (d + 1) + self.assoc_val_dim * d + d
        cell = (
            d  # attn_norm
            + d * (k + 1)  # depthwise conv weight + bias
            + 3 * d * d  # qkv
            + (d + 1) * h  # decay
            + (d + 1) * h  # beta
            + 2 * d * d  # out_gate + out_proj
            + d  # head_gamma
            + d  # ffn_norm
            + 3 * d * ff  # SwiGLU
        )
        adapters = self.depth * (6 * d + 2 * h) + max(self.depth - 1, 0) * d if self.depth > 1 else 0
        final_norm = d
        return embed + assoc_params + len(self.cells) * cell + adapters + final_norm


class FractalDecode:
    """Incremental single-byte decode adapter for FractalDeltaMemory."""

    is_recurrent = True

    def __init__(self, model):
        if type(model) is not FractalDeltaMemory or model.training:
            raise ValueError("fractal decode adapter requires evaluation FractalDeltaMemory")
        self.model = model
        self.step_cells = tuple(copy(cell) for cell in model.cells)
        for cell in self.step_cells:
            cell.execution = "sequential"

    def prefill(self, ids, capacity):
        if not 0 < ids.shape[1] <= capacity <= self.model.config.max_seq_len:
            raise ValueError("invalid prefill capacity")
        out = self.model(ids, return_logits=False)
        return self.model.lm_head(out.hidden_states[:, -1:]), out.state

    def step(self, ids, state):
        if ids.shape != (state.levels[0].memory.shape[0], 1):
            raise ValueError("fractal recurrent step requires one byte")
        raw = self.model.embedding(ids)
        assoc_out, next_assoc_mem, next_local_tail = self.model._run_assoc(
            raw, state, execution="sequential"
        )
        x = raw
        anchor = x
        next_levels = []
        for level, prev in enumerate(state.levels):
            adapter = self.model.adapters[level] if self.model.depth > 1 else None
            if adapter is not None and adapter.anchor_gate is not None:
                x = x + adapter.anchor_gate.sigmoid() * anchor
            cell = self.step_cells[0 if self.model.shared else level]
            x, current = cell(x, prev, adapter=adapter)
            next_levels.append(current)
        x = x + assoc_out
        logits = self.model.lm_head(self.model.norm(x))
        next_state = FractalDeltaState(
            tuple(next_levels),
            next_assoc_mem,
            next_local_tail,
        )
        return logits, next_state
