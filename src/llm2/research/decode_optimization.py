"""Local KV cache, full-graph step compilation, self-speculative MTP decode and NVML/L2 helpers."""
import ctypes
from dataclasses import dataclass
import shutil
import subprocess
import sys
import threading
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from llm2.architectures.context_memory import ContextCellState, ContextState
from llm2.architectures.fractal_delta import FractalDecode
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


def _load_nvml():
    """Load NVIDIA Management Library via pynvml or system shared library."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        return ("pynvml", pynvml)
    except Exception:
        pass
    for lib_name in ("libnvidia-ml.so.1", "nvml.dll"):
        try:
            lib = ctypes.CDLL(lib_name)
            if lib.nvmlInit_v2() == 0:
                return ("ctypes", lib)
        except Exception:
            continue
    return None


class NVMLPowerMonitor:
    """Measure real GPU power, energy, temperature and SM clock; returns None fields on CPU."""

    def __init__(self, device, *, sample_interval_s: float = 0.01):
        self.device = torch.device(device)
        self.sample_interval_s = sample_interval_s
        self._nvml = None
        self._handle = None
        self._samples_mw: list[float] = []
        self._temps_c: list[float] = []
        self._clocks_mhz: list[float] = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0
        self._e0_mj = None
        self._e1_mj = None
        self.summary: dict | None = None

    def _query(self):
        kind, lib = self._nvml
        idx = 0 if self.device.index is None else int(self.device.index)
        if kind == "pynvml":
            if self._handle is None:
                self._handle = lib.nvmlDeviceGetHandleByIndex(idx)
            mw = float(lib.nvmlDeviceGetPowerUsage(self._handle))
            temp = float(lib.nvmlDeviceGetTemperature(self._handle, 0))
            clock = float(lib.nvmlDeviceGetClockInfo(self._handle, 1))
            try:
                energy = int(lib.nvmlDeviceGetTotalEnergyConsumption(self._handle))
            except Exception:
                energy = None
            return mw, temp, clock, energy
        if self._handle is None:
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(idx), ctypes.byref(handle)) != 0:
                raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")
            self._handle = handle
        mw_c = ctypes.c_uint()
        temp_c = ctypes.c_uint()
        clock_c = ctypes.c_uint()
        energy_c = ctypes.c_ulonglong()
        if lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(mw_c)) != 0:
            raise RuntimeError("nvmlDeviceGetPowerUsage failed")
        t_val = float(temp_c.value) if lib.nvmlDeviceGetTemperature(self._handle, 0, ctypes.byref(temp_c)) == 0 else None
        c_val = float(clock_c.value) if lib.nvmlDeviceGetClockInfo(self._handle, 1, ctypes.byref(clock_c)) == 0 else None
        e_val = int(energy_c.value) if lib.nvmlDeviceGetTotalEnergyConsumption(self._handle, ctypes.byref(energy_c)) == 0 else None
        return float(mw_c.value), t_val, c_val, e_val

    def _poll(self):
        while not self._stop.wait(self.sample_interval_s):
            try:
                mw, temp, clock, _ = self._query()
                self._samples_mw.append(mw)
                if temp is not None:
                    self._temps_c.append(temp)
                if clock is not None:
                    self._clocks_mhz.append(clock)
            except Exception:
                break

    def __enter__(self):
        if self.device.type != "cuda":
            return self
        self._nvml = _load_nvml()
        if self._nvml is None:
            return self
        try:
            sync(self.device)
            mw, temp, clock, e0 = self._query()
            self._samples_mw.append(mw)
            if temp is not None:
                self._temps_c.append(temp)
            if clock is not None:
                self._clocks_mhz.append(clock)
            self._e0_mj = e0
            self._t0 = time.perf_counter()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        except Exception:
            self._nvml = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._nvml is None:
            self.summary = None
            return
        sync(self.device)
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            mw, temp, clock, e1 = self._query()
            self._samples_mw.append(mw)
            if temp is not None:
                self._temps_c.append(temp)
            if clock is not None:
                self._clocks_mhz.append(clock)
            self._e1_mj = e1
        except Exception:
            pass
        watts_mean = (sum(self._samples_mw) / len(self._samples_mw)) / 1000.0 if self._samples_mw else None
        energy_j = (
            (self._e1_mj - self._e0_mj) / 1000.0
            if self._e0_mj is not None and self._e1_mj is not None and self._e1_mj >= self._e0_mj
            else (watts_mean * elapsed if watts_mean is not None else None)
        )
        self.summary = {
            "source": "nvml_energy_counter" if (self._e0_mj is not None and self._e1_mj is not None) else "nvml_power_samples",
            "elapsed_seconds": elapsed,
            "samples": len(self._samples_mw),
            "watts_mean": watts_mean,
            "energy_joules": energy_j,
            "temp_c_mean": sum(self._temps_c) / len(self._temps_c) if self._temps_c else None,
            "temp_c_max": max(self._temps_c) if self._temps_c else None,
            "sm_clock_mhz_mean": sum(self._clocks_mhz) / len(self._clocks_mhz) if self._clocks_mhz else None,
        }


def try_set_power_limit(device, ratio: float | None):
    """Best-effort GPU power cap setup returning original limit for restoration."""
    dev = torch.device(device)
    if dev.type != "cuda" or ratio is None:
        return {"applied": False, "ratio": ratio, "original_watts": None, "target_watts": None, "reason": "disabled_or_non_cuda"}
    idx = str(0 if dev.index is None else int(dev.index))
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", idx, "--query-gpu=power.limit,power.min_limit,power.max_limit", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        cur_w, min_w, max_w = (float(x.strip()) for x in out.split(","))
        target_w = max(min_w, min(max_w, cur_w * float(ratio)))
        subprocess.check_call(["nvidia-smi", "-i", idx, "-pl", f"{target_w:.2f}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return {"applied": True, "ratio": float(ratio), "original_watts": cur_w, "target_watts": target_w, "reason": "ok"}
    except Exception as exc:
        return {"applied": False, "ratio": float(ratio), "original_watts": None, "target_watts": None, "reason": str(exc)}


def restore_power_limit(device, cap_info: dict | None):
    if not cap_info or not cap_info.get("applied") or cap_info.get("original_watts") is None:
        return
    dev = torch.device(device)
    idx = str(0 if dev.index is None else int(dev.index))
    try:
        subprocess.run(["nvidia-smi", "-i", idx, "-pl", f"{cap_info['original_watts']:.2f}"], check=False, timeout=5)
    except Exception:
        pass


def try_pin_l2_weights(tensors, device):
    """Best-effort CUDA L2 cache persisting-window setup for recurrent Core weights."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return {"supported": False, "pinned_bytes": 0, "reason": "non_cuda"}
    tensor_list = [t for t in tensors if isinstance(t, torch.Tensor) and t.is_cuda]
    total_bytes = sum(t.numel() * t.element_size() for t in tensor_list)
    if not total_bytes:
        return {"supported": False, "pinned_bytes": 0, "reason": "no_cuda_tensors"}
    for lib_name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            cudart = ctypes.CDLL(lib_name)
            # cudaLimitPersistingL2CacheSize = 0x06; best-effort reserve up to tensor bytes.
            err = cudart.cudaDeviceSetLimit(ctypes.c_int(0x06), ctypes.c_size_t(total_bytes))
            return {
                "supported": err == 0,
                "pinned_bytes": total_bytes if err == 0 else 0,
                "reason": "cudaDeviceSetLimit_ok" if err == 0 else f"cudaError={err}",
            }
        except Exception:
            continue
    return {"supported": False, "pinned_bytes": 0, "reason": "libcudart_not_found"}


class SelfSpeculativeMTPDecode:
    """Exact greedy self-speculative decoder reusing trained auxiliary MTP heads."""

    def __init__(self, model, mtp_heads, *, vocab_limit: int, backend: str | None = "inductor"):
        self.decoder = FractalDecode(model)
        self.model = model
        self.mtp_heads = nn.ModuleList(list(mtp_heads)).eval()
        self.k_drafts = len(self.mtp_heads)
        if self.k_drafts < 1:
            raise ValueError("SelfSpeculativeMTPDecode requires at least one MTP head")
        self.vocab_limit = int(vocab_limit)
        device = next(model.parameters()).device
        self.device = device
        self.backend = None if backend is None else resolve_compile_backend(backend, device)
        decoder = self.decoder
        heads = self.mtp_heads
        vlim = self.vocab_limit

        def verify_raw(block_ids, state):
            logits_b, hidden_b, stacked_states = decoder.step_hidden(block_ids, state, return_all_states=True)
            pred_main = logits_b[0, :, :vlim].argmax(dim=-1, keepdim=True)
            pred_drafts = [head(hidden_b)[0, :, :vlim].argmax(dim=-1, keepdim=True) for head in heads]
            toks_table = torch.cat([pred_main, *pred_drafts], dim=-1)
            return toks_table, stacked_states

        self._verify_raw = verify_raw
        self._verify_fn = (
            verify_raw
            if self.backend is None
            else torch.compile(verify_raw, backend=self.backend, fullgraph=True, dynamic=False)
        )
        self.use_cudagraph = bool(self.backend == "inductor" and device.type == "cuda")
        self._graph = None
        self._graph_key = None
        self._static_block = None
        self._static_in = None
        self._static_out = None

    def _capture(self, block_ids, state, key):
        static_block = block_ids.clone()
        static_state = _map_tensors(state, torch.Tensor.clone)
        torch.cuda.synchronize(self.device)
        for _ in range(3):
            self._verify_fn(static_block, static_state)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self._verify_fn(static_block, static_state)
        torch.cuda.synchronize(self.device)
        self._graph = graph
        self._graph_key = key
        self._static_block = static_block
        self._static_in = tuple(_iter_tensors(static_state))
        self._static_out = static_out

    def _run_verify(self, block_ids, state):
        if self.use_cudagraph:
            state_tensors = tuple(_iter_tensors(state))
            key = (block_ids.shape, block_ids.dtype, tuple((t.shape, t.dtype) for t in state_tensors))
            try:
                if self._graph is None or self._graph_key != key:
                    self._capture(block_ids, state, key)
                self._static_block.copy_(block_ids)
                for dst, src in zip(self._static_in, state_tensors, strict=True):
                    dst.copy_(src)
                self._graph.replay()
                toks_table, stacked_states = self._static_out
                return toks_table.clone(), _map_tensors(stacked_states, torch.Tensor.clone)
            except Exception as exc:
                if "InvalidCxxCompiler" in str(exc) or "cl is not found" in str(exc):
                    self.use_cudagraph = False
                    self.backend = "aot_eager"
                    self._verify_fn = torch.compile(self._verify_raw, backend="aot_eager", fullgraph=True, dynamic=False)
                else:
                    raise
        try:
            return self._verify_fn(block_ids, state)
        except Exception as exc:
            if self.backend == "inductor" and ("InvalidCxxCompiler" in str(exc) or "cl is not found" in str(exc)):
                self.backend = "aot_eager"
                self._verify_fn = torch.compile(self._verify_raw, backend="aot_eager", fullgraph=True, dynamic=False)
                return self._verify_fn(block_ids, state)
            raise

    @torch.inference_mode()
    def generate_greedy(self, prefix: torch.Tensor, new_tokens: int, *, measure_energy: bool = False):
        if prefix.ndim != 2 or prefix.shape[0] != 1:
            raise ValueError("SelfSpeculativeMTPDecode requires batch 1")
        if new_tokens < 1:
            return [], {"tokens": 0, "cycles": 0, "decode_seconds": 0.0, "tokens_per_sec": None}
        cap = prefix.shape[1] + new_tokens + self.k_drafts + 1
        logits0, hidden0, cur_state = self.decoder.prefill_hidden(prefix, cap)
        next_tok = int(logits0[0, -1, :self.vocab_limit].argmax(dim=-1).item())
        drafts = [int(h(hidden0)[0, -1, :self.vocab_limit].argmax(dim=-1).item()) for h in self.mtp_heads]
        # Warmup verify pass before timing when compiling/capturing CUDAGraph.
        if self.backend is not None:
            warm_block = torch.tensor([[next_tok, *drafts]], dtype=torch.long, device=self.device)
            self._run_verify(warm_block, cur_state)

        emitted: list[int] = []
        cycles = 0
        accepted_drafts = 0
        proposed_drafts = 0
        sync(self.device)
        monitor = NVMLPowerMonitor(self.device) if measure_energy else None
        if monitor is not None:
            monitor.__enter__()
        t0 = time.perf_counter()
        try:
            while len(emitted) < new_tokens:
                block = torch.tensor([[next_tok, *drafts]], dtype=torch.long, device=self.device)
                toks_table, stacked_states = self._run_verify(block, cur_state)
                rows = toks_table.tolist()
                cycles += 1
                emitted.append(next_tok)
                if len(emitted) >= new_tokens:
                    break
                proposed_drafts += self.k_drafts
                last_idx = self.k_drafts
                for m in range(self.k_drafts):
                    true_next = rows[m][0]
                    if true_next == drafts[m]:
                        emitted.append(true_next)
                        accepted_drafts += 1
                        if len(emitted) >= new_tokens:
                            last_idx = m + 1
                            break
                    else:
                        last_idx = m
                        break
                if len(emitted) >= new_tokens:
                    break
                next_tok = rows[last_idx][0]
                drafts = rows[last_idx][1:]
                cur_state = self.decoder.slice_step_state(stacked_states, last_idx)
            sync(self.device)
            decode_s = max(time.perf_counter() - t0, 1e-9)
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)
        nvml = None if monitor is None else monitor.summary
        mj_per_tok = (
            (nvml["energy_joules"] * 1000.0) / len(emitted)
            if nvml is not None and nvml.get("energy_joules") is not None and emitted
            else None
        )
        stats = {
            "tokens": len(emitted),
            "cycles": cycles,
            "proposed_drafts": proposed_drafts,
            "accepted_drafts": accepted_drafts,
            "draft_acceptance_rate": (accepted_drafts / proposed_drafts) if proposed_drafts else 0.0,
            "mean_tokens_per_cycle": len(emitted) / max(cycles, 1),
            "decode_seconds": decode_s,
            "tokens_per_sec": len(emitted) / decode_s,
            "cudagraph": self.use_cudagraph,
            "nvml": nvml,
            "mj_per_token": mj_per_tok,
        }
        return emitted, stats
