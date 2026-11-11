"""Incremental inference adapters over the frozen 0033 models."""
from copy import copy
from dataclasses import dataclass
import time

import torch
from torch.nn import functional as F

from llm2.architectures.context_memory import ContextMemory, ContextState
from llm2.model import CausalSDPA, TransformerLM
from llm2.research.recall_worker import state_storage_bytes, sync


@dataclass
class KVState:
    length: int
    capacity: int
    layers: tuple


class TransformerDecode:
    def __init__(self, model):
        if type(model) is not TransformerLM or model.training:
            raise ValueError("KV adapter requires the evaluation Transformer baseline")
        if any(type(block.attn.kernel) is not CausalSDPA for block in model.layers):
            raise ValueError("KV adapter requires the original SDPA kernel")
        self.model = model

    def prefill(self, ids, capacity):
        batch, length = ids.shape
        if not 0 < length <= capacity <= self.model.config.max_seq_len:
            raise ValueError("invalid prefill capacity")
        x = self.model.embedding(ids)
        caches = []
        for block in self.model.layers:
            q, k, v = self.project(block, x)
            attention = block.attn
            q, k = attention.rope(q), attention.rope(k)
            cache = k.new_empty(2, batch, attention.heads, capacity, attention.head_dim)
            cache[0, :, :, :length].copy_(k)
            cache[1, :, :, :length].copy_(v)
            caches.append(cache)
            x = self.finish_block(block, x, attention.kernel(q, k, v))
        return self.last_logits(x), KVState(length, capacity, tuple(caches))

    def step(self, ids, state):
        if ids.shape != (state.layers[0].shape[1], 1) or state.length >= state.capacity:
            raise ValueError("KV step requires one byte and available capacity")
        pos = state.length
        x = self.model.embedding(ids)
        for block, cache in zip(self.model.layers, state.layers, strict=True):
            q, k, v = self.project(block, x)
            rope = block.attn.rope
            q, k = (self.rotate_at(rope, tensor, pos) for tensor in (q, k))
            cache[0, :, :, pos:pos + 1].copy_(k)
            cache[1, :, :, pos:pos + 1].copy_(v)
            # One query at the last valid position can see every stored key.
            read = F.scaled_dot_product_attention(
                q, cache[0, :, :, :pos + 1], cache[1, :, :, :pos + 1],
                is_causal=False, dropout_p=0.,
            )
            x = self.finish_block(block, x, read)
        state.length += 1
        return self.last_logits(x), state

    @staticmethod
    def project(block, x):
        batch, length, _ = x.shape
        attn = block.attn
        qkv = attn.qkv(block.attn_norm(x)).view(batch, length, 3, attn.heads, attn.head_dim)
        return (tensor.transpose(1, 2) for tensor in qkv.unbind(2))

    @staticmethod
    def rotate_at(rope, x, pos):
        cos, sin = rope.cos[pos:pos + 1].float(), rope.sin[pos:pos + 1].float()
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)

    @staticmethod
    def finish_block(block, x, read):
        x = x + block.attn.proj(read.transpose(1, 2).contiguous().view(*x.shape))
        return x + block.ffn(block.ffn_norm(x))

    def last_logits(self, x):
        return self.model.lm_head(self.model.norm(x[:, -1:]))


class RecurrentDecode:
    def __init__(self, model):
        if type(model) is not ContextMemory or model.hierarchy or model.training:
            raise ValueError("recurrent adapter requires the evaluation fine-scale model")
        self.model = model
        # Each copy selects the original one-step rule while sharing all tensors.
        self.step_cells = tuple(copy(cell) for cell in model.cells)
        for cell in self.step_cells:
            cell.execution = "sequential"

    def prefill(self, ids, capacity):
        if not 0 < ids.shape[1] <= capacity <= self.model.config.max_seq_len:
            raise ValueError("invalid prefill capacity")
        output = self.model(ids, return_logits=False)
        return self.model.lm_head(output.hidden_states[:, -1:]), output.state

    def step(self, ids, state):
        if ids.shape != (state.last_coarse.shape[0], 1):
            raise ValueError("recurrent step requires one byte")
        x = self.model.embedding(ids).float()
        states = []
        for index, previous in enumerate(state.cells):
            cell = self.step_cells[0 if self.model.shared else index]
            x, current = cell(x, previous)
            states.append(current)
        logits = self.model.lm_head(self.model.norm(x))
        return logits, ContextState(tuple(states), state.pending, state.last_coarse)


@torch.inference_mode()
def verify_logits(decoder, prefix, continuation):
    length = prefix.shape[1]
    complete = torch.cat((prefix, continuation), dim=1)
    reference = decoder.model(complete).logits[:, length - 1:]
    first, state = decoder.prefill(prefix, complete.shape[1])
    incremental = [first]
    for token in continuation.split(1, dim=1):
        logits, state = decoder.step(token, state)
        incremental.append(logits)
    actual = torch.cat(incremental, dim=1)
    torch.testing.assert_close(actual, reference, atol=5e-4, rtol=5e-4)
    return {
        "positions": actual.shape[1],
        "max_abs_logit_error": float((actual - reference).abs().max()),
        "top1_agreement": float((actual.argmax(-1) == reference.argmax(-1)).float().mean()),
        "atol": 5e-4,
        "rtol": 5e-4,
    }


@torch.inference_mode()
def generate(decoder, prefix, new_bytes):
    if new_bytes < 2:
        raise ValueError("benchmark needs a prefill and at least one decode step")
    device = prefix.device
    sync(device)
    start = time.perf_counter()
    logits, state = decoder.prefill(prefix, prefix.shape[1] + new_bytes - 1)
    token = logits[..., :256].argmax(-1)
    sync(device)
    first = time.perf_counter()
    tokens = [token]
    for _ in range(new_bytes - 1):
        logits, state = decoder.step(token, state)
        token = logits[..., :256].argmax(-1)
        tokens.append(token)
    sync(device)
    end = time.perf_counter()
    return {
        "first_byte_seconds": first - start,
        "decode_seconds": end - first,
        "total_seconds": end - start,
        "state_bytes": state_storage_bytes(state),
    }, torch.cat(tokens, dim=1)
