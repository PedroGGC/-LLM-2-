"""Local KV cache, full-graph step compilation and launch profiling."""
from dataclasses import dataclass
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from llm2.architectures.context_memory import ContextCellState, ContextState
from llm2.research.recall_worker import sync

if __package__:
    from .decode_cache import RecurrentDecode, TransformerDecode
else:
    from decode_cache import RecurrentDecode, TransformerDecode

WINDOW = 256


class WindowSDPA(nn.Module):
    """Exact causal attention over the last span positions per layer."""

    def __init__(self, span, tile=512):
        super().__init__()
        if span < 1 or tile < 1:
            raise ValueError("positive window and tile required")
        self.span = span
        self.tile = tile

    def forward(self, q, k, v):
        length = q.shape[-2]
        if length <= self.span:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        positions = torch.arange(length, device=q.device)
        chunks = []
        for start in range(0, length, self.tile):
            end = min(start + self.tile, length)
            left = max(0, start - self.span + 1)
            distance = positions[start:end, None] - positions[None, left:end]
            mask = (distance >= 0) & (distance < self.span)
            chunks.append(F.scaled_dot_product_attention(
                q[:, :, start:end], k[:, :, left:end], v[:, :, left:end],
                attn_mask=mask, dropout_p=0.,
            ))
        return torch.cat(chunks, dim=-2)


@dataclass
class WindowState:
    position: torch.Tensor
    layers: tuple


class LocalTransformerDecode(TransformerDecode):
    def __init__(self, model, span=WINDOW):
        super().__init__(model)
        self.span = span
        for block in model.layers:
            block.attn.kernel = WindowSDPA(span).eval()

    def prefill(self, ids, capacity):
        batch, length = ids.shape
        if not 0 < length <= capacity <= self.model.config.max_seq_len:
            raise ValueError("invalid local prefill capacity")
        x = self.model.embedding(ids)
        caches = []
        left = max(0, length - self.span)
        indices = torch.arange(left, length, device=x.device) % self.span
        for block in self.model.layers:
            q, k, v = self.project(block, x)
            attention = block.attn
            q, k = attention.rope(q), attention.rope(k)
            cache = k.new_zeros(2, batch, attention.heads, self.span, attention.head_dim)
            cache[0].index_copy_(2, indices, k[:, :, left:])
            cache[1].index_copy_(2, indices, v[:, :, left:])
            caches.append(cache)
            x = self.finish_block(block, x, attention.kernel(q, k, v))
        position = torch.tensor([length], device=ids.device)
        return self.last_logits(x), WindowState(position, tuple(caches))

    @staticmethod
    def rotate_at_tensor(rope, x, position):
        cos = rope.cos.index_select(0, position).float()
        sin = rope.sin.index_select(0, position).float()
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)

    def step(self, ids, state):
        if ids.shape != (state.layers[0].shape[1], 1):
            raise ValueError("local KV step requires one byte")
        position = state.position
        slot = position % self.span
        valid = (torch.arange(self.span, device=ids.device) <= position)[None, :]
        x = self.model.embedding(ids)
        for block, cache in zip(self.model.layers, state.layers, strict=True):
            q, k, v = self.project(block, x)
            rope = block.attn.rope
            q = self.rotate_at_tensor(rope, q, position)
            k = self.rotate_at_tensor(rope, k, position)
            cache[0].index_copy_(2, slot, k)
            cache[1].index_copy_(2, slot, v)
            # K and V share the same slot order. All valid slots precede the query.
            read = F.scaled_dot_product_attention(q, cache[0], cache[1], attn_mask=valid, dropout_p=0.)
            x = self.finish_block(block, x, read)
        return self.last_logits(x), WindowState(position + 1, state.layers)


def canonical_recurrent_state(state):
    def batch_view(tensor):
        # Batch size is one. Only the stride of its singleton axis changes.
        return tensor.as_strided(tensor.shape, (tensor.numel(), *tensor.stride()[1:]))

    cells = tuple(ContextCellState(batch_view(cell.memory), batch_view(cell.tail)) for cell in state.cells)
    return ContextState(cells, state.pending, state.last_coarse)


class CompiledDecode:
    def __init__(self, decoder, *, backend="inductor"):
        self.decoder = decoder
        self.model = decoder.model
        self.recurrent = isinstance(decoder, RecurrentDecode)
        step = decoder.step
        if self.recurrent:
            def recurrent_step(ids, state):
                logits, current = decoder.step(ids, state)
                return logits, canonical_recurrent_state(current)
            step = recurrent_step
        self.compiled_step = torch.compile(step, backend=backend, fullgraph=True, dynamic=False)

    def prefill(self, ids, capacity):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("compiled benchmark requires batch 1")
        logits, state = self.decoder.prefill(ids, capacity)
        return logits, canonical_recurrent_state(state) if self.recurrent else state

    def step(self, ids, state):
        if ids.shape != (1, 1):
            raise ValueError("compiled benchmark requires a single byte at batch 1")
        # A one-element view can carry the entire source sequence's stride.
        # Canonicalize it without copying or specializing the graph on prefix length.
        return self.compiled_step(ids.as_strided((1, 1), (1, 1)), state)


@torch.inference_mode()
def cold_step_seconds(decoder, prefix, continuation):
    _, state = decoder.prefill(prefix, prefix.shape[1] + continuation.shape[1])
    sync(prefix.device)
    start = time.perf_counter()
    decoder.step(continuation[:, :1], state)
    sync(prefix.device)
    return time.perf_counter() - start


@torch.inference_mode()
def profile_steps(decoder, prefix, continuation):
    steps = min(16, continuation.shape[1])
    _, state = decoder.prefill(prefix, prefix.shape[1] + steps)
    sync(prefix.device)
    activities = [ProfilerActivity.CPU]
    if prefix.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as profiler:
        for token in continuation[:, :steps].split(1, dim=1):
            _, state = decoder.step(token, state)
        sync(prefix.device)
    events = [event for event in profiler.events() if str(event.device_type) == "DeviceType.CUDA"]
    top = sorted(profiler.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True)[:12]
    return {
        "steps": steps,
        "cuda_events": len(events) if events else None,
        "cuda_event_total_us": sum(event.time_range.elapsed_us() for event in events) if events else None,
        "cpu_operations": [{"name": event.key, "count": event.count,
                            "self_cpu_us": event.self_cpu_time_total} for event in top],
        "scope": "separate forced decode steps; event durations can overlap",
    }
