"""Randomly initialized decoder-only Transformer; no loss or optimizer coupling."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .interfaces import AttentionKernel, Backbone, ModelOutput


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Accumulate in float32 even under CUDA autocast.
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, theta):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        cos = self.cos[:x.shape[-2]].float()
        sin = self.sin[:x.shape[-2]].float()
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)


class CausalSDPA(AttentionKernel):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )


class SelfAttention(nn.Module):
    def __init__(self, config, kernel=None):
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
        self.kernel = kernel if kernel is not None else CausalSDPA(config.dropout)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        result = self.kernel(self.rope(q), self.rope(k), v)
        return self.proj(result.transpose(1, 2).contiguous().view(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, config, kernel=None):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = SelfAttention(config, kernel)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class TransformerLM(Backbone):
    def __init__(self, config: ModelConfig, *, attention_factory=None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            DecoderBlock(config, None if attention_factory is None else attention_factory(config))
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embedding.weight
        # Scale residual projections by depth at random initialization.
        for block in self.layers:
            for projection in (block.attn.proj, block.ffn.down):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, *, state=None):
        if state is not None:
            raise ValueError("Transformer baseline has no recurrent state")
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must be [batch, 1..max_seq_len]")
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return ModelOutput(logits=self.lm_head(x), hidden_states=x)


def count_parameters(model):
    """Calculated unique parameters, so tied weights are counted once."""
    return sum(p.numel() for p in model.parameters())


def transformer_parameter_formula(config):
    d, h, n, v = config.d_model, config.hidden_dim, config.n_layers, config.vocab_size
    return v * d + n * (4 * d * d + 3 * d * h + 2 * d) + d

