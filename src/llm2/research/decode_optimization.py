"""Local KV cache, full-graph step compilation and launch profiling."""
from dataclasses import dataclass
import shutil
import sys
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


def resolve_compile_backend(backend="inductor", device=None):
    """Resolve compilation backend, falling back to aot_eager on Windows CPU when C++ compiler is absent."""
    if backend != "inductor":
        return backend
    if device is not None and getattr(device, "type", None) == "cuda":
        return "inductor"
    if sys.platform == "win32" and shutil.which("cl") is None:
        return "aot_eager"
    return "inductor"


def canonical_recurrent_state(state):
    """Enforce strictly contiguous memory layout on recurrent state tensors without as_strided layout distortion."""
    cells = tuple(
        ContextCellState(
            cell.memory.reshape(cell.memory.shape).contiguous(),
            cell.tail.reshape(cell.tail.shape).contiguous(),
        )
        for cell in state.cells
    )
    pending = state.pending.reshape(state.pending.shape).contiguous() if state.pending is not None else None
    last_coarse = state.last_coarse.reshape(state.last_coarse.shape).contiguous() if state.last_coarse is not None else None
    return ContextState(cells, pending, last_coarse)


def _iter_tensors(value):
    from dataclasses import fields, is_dataclass

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _iter_tensors(getattr(value, item.name))


def _map_tensors(value, fn):
    from dataclasses import fields, is_dataclass

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{item.name: _map_tensors(getattr(value, item.name), fn) for item in fields(value)})
    return value


class CompiledDecode:
    def __init__(self, decoder, *, backend="inductor"):
        self.decoder = decoder
        self.model = decoder.model
        self.recurrent = bool(getattr(decoder, "is_recurrent", isinstance(decoder, RecurrentDecode)))
        device = next(decoder.model.parameters()).device if list(decoder.model.parameters()) else None
        self.backend = resolve_compile_backend(backend, device)
        step = decoder.step
        if self.recurrent and isinstance(decoder, RecurrentDecode):
            def recurrent_step(ids, state):
                logits, current = decoder.step(ids, state)
                return logits, canonical_recurrent_state(current)
            step = recurrent_step
        self._raw_step = step
        self.compiled_step = torch.compile(step, backend=self.backend, fullgraph=True, dynamic=False)
        self.use_cudagraph = bool(self.recurrent and self.backend == "inductor" and device is not None and device.type == "cuda")
        self._graph = None
        self._graph_key = None
        self._static_token = None
        self._static_in_tensors = None
        self._static_out = None

    def prefill(self, ids, capacity):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("compiled benchmark requires batch 1")
        logits, state = self.decoder.prefill(ids, capacity)
        if self.recurrent and isinstance(self.decoder, RecurrentDecode):
            return logits, canonical_recurrent_state(state)
        return logits, state

    def _capture_cudagraph(self, token, state, key):
        static_token = token.clone()
        static_state = _map_tensors(state, torch.Tensor.clone)
        torch.cuda.synchronize(static_token.device)
        for _ in range(3):
            self.compiled_step(static_token, static_state)
        torch.cuda.synchronize(static_token.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.compiled_step(static_token, static_state)
        torch.cuda.synchronize(static_token.device)
        self._graph = graph
        self._graph_key = key
        self._static_token = static_token
        self._static_in_tensors = tuple(_iter_tensors(static_state))
        self._static_out = static_out

    def step(self, ids, state):
        if ids.shape != (1, 1):
            raise ValueError("compiled benchmark requires a single byte at batch 1")
        token = ids.reshape(1, 1).clone()
        if self.use_cudagraph:
            state_tensors = tuple(_iter_tensors(state))
            key = (token.dtype, tuple((t.shape, t.dtype) for t in state_tensors))
            try:
                if self._graph is None or self._graph_key != key:
                    self._capture_cudagraph(token, state, key)
                self._static_token.copy_(token)
                for dst, src in zip(self._static_in_tensors, state_tensors, strict=True):
                    dst.copy_(src)
                self._graph.replay()
                logits, next_state = self._static_out
                return logits.clone(), _map_tensors(next_state, torch.Tensor.clone)
            except Exception as exc:
                if "InvalidCxxCompiler" in str(exc) or "cl is not found" in str(exc):
                    self.use_cudagraph = False
                    self.backend = "aot_eager"
                    self.compiled_step = torch.compile(self._raw_step, backend="aot_eager", fullgraph=True, dynamic=False)
                else:
                    raise
        try:
            return self.compiled_step(token, state)
        except Exception as exc:
            if self.backend == "inductor" and ("InvalidCxxCompiler" in str(exc) or "cl is not found" in str(exc)):
                self.backend = "aot_eager"
                self.compiled_step = torch.compile(self._raw_step, backend="aot_eager", fullgraph=True, dynamic=False)
                return self.compiled_step(token, state)
            raise


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
