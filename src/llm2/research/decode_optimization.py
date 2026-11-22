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
from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaState, FractalLevelState
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
        if torch._dynamo.config.cache_size_limit < 64:
            torch._dynamo.config.cache_size_limit = 64
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


class _CUDAGraphRunner:
    """Reusable CUDAGraph wrapper for any pure-tensor/dataclass-state compiled callable."""

    def __init__(self, fn, device: torch.device):
        self.fn = fn
        self.device = device
        self._graphs: dict[tuple, tuple] = {}

    def __call__(self, *args):
        tensors = tuple(_iter_tensors(args))
        key = tuple((t.shape, t.dtype) for t in tensors)
        entry = self._graphs.get(key)
        if entry is None:
            static_args = _map_tensors(args, torch.Tensor.clone)
            static_in = tuple(_iter_tensors(static_args))
            torch.cuda.synchronize(self.device)
            for _ in range(3):
                self.fn(*static_args)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self.fn(*static_args)
            torch.cuda.synchronize(self.device)
            entry = (graph, static_in, static_out)
            self._graphs[key] = entry
        graph, static_in, static_out = entry
        for dst, src in zip(static_in, tensors, strict=True):
            dst.copy_(src)
        graph.replay()
        return _map_tensors(static_out, torch.Tensor.clone)


class SelfSpeculativeMTPDecode:
    """Exact greedy self-speculative decoder reusing trained auxiliary MTP heads with per-bucket CUDAGraph gating."""

    def __init__(
        self,
        model,
        mtp_heads,
        *,
        vocab_limit: int,
        backend: str | None = "inductor",
        draft_confidence_threshold: float = 0.0,
    ):
        self.decoder = FractalDecode(model)
        self.model = model
        self.mtp_heads = nn.ModuleList(list(mtp_heads)).eval()
        self.k_drafts = len(self.mtp_heads)
        if self.k_drafts < 1:
            raise ValueError("SelfSpeculativeMTPDecode requires at least one MTP head")
        if not 0.0 <= float(draft_confidence_threshold) <= 1.0:
            raise ValueError("draft_confidence_threshold must be in [0.0, 1.0]")
        self.draft_confidence_threshold = float(draft_confidence_threshold)
        self.vocab_limit = int(vocab_limit)
        device = next(model.parameters()).device
        self.device = device
        self.backend = None if backend is None else resolve_compile_backend(backend, device)
        decoder = self.decoder
        heads = self.mtp_heads
        vlim = self.vocab_limit
        track_probs = self.draft_confidence_threshold > 0.0

        def verify_raw(block_ids, state):
            logits_b, hidden_b, stacked_states = decoder.step_hidden(block_ids, state, return_all_states=True)
            pred_main = logits_b[0, :, :vlim].argmax(dim=-1, keepdim=True)
            pred_drafts = []
            prob_drafts = []
            for head in heads:
                h_logits = head(hidden_b)[0, :, :vlim]
                if track_probs:
                    p_max, tok_max = torch.softmax(h_logits.float(), dim=-1).max(dim=-1, keepdim=True)
                    pred_drafts.append(tok_max)
                    prob_drafts.append(p_max)
                else:
                    pred_drafts.append(h_logits.argmax(dim=-1, keepdim=True))
            toks_table = torch.cat([pred_main, *pred_drafts], dim=-1)
            probs_table = torch.cat(prob_drafts, dim=-1) if track_probs else pred_main.new_ones((pred_main.shape[0], 1), dtype=torch.float32)
            return toks_table, probs_table, stacked_states

        self._verify_raw = verify_raw
        if torch._dynamo.config.cache_size_limit < 64:
            torch._dynamo.config.cache_size_limit = 64
        self._verify_fn = (
            verify_raw
            if self.backend is None
            else torch.compile(verify_raw, backend=self.backend, fullgraph=True, dynamic=False)
        )
        self.use_cudagraph = bool(self.backend == "inductor" and device.type == "cuda")
        self._cg_runner = _CUDAGraphRunner(self._verify_fn, device) if self.use_cudagraph else None

    def _select_drafts_from_hidden(self, last_hidden: torch.Tensor) -> tuple[list[int], int]:
        """Propose contiguous drafts passing draft_confidence_threshold; returns (drafts, pruned_count)."""
        drafts: list[int] = []
        pruned = 0
        for idx, head in enumerate(self.mtp_heads):
            d_logits = head(last_hidden)[0, -1, :self.vocab_limit].float()
            if self.draft_confidence_threshold > 0.0:
                probs = torch.softmax(d_logits, dim=-1)
                top_prob, top_tok = probs.max(dim=-1)
                if float(top_prob.item()) < self.draft_confidence_threshold:
                    pruned = self.k_drafts - idx
                    break
                drafts.append(int(top_tok.item()))
            else:
                drafts.append(int(d_logits.argmax(dim=-1).item()))
        return drafts, pruned

    def _select_drafts_from_row(self, draft_toks: list[int], draft_probs: list[float]) -> tuple[list[int], int]:
        if self.draft_confidence_threshold <= 0.0:
            return draft_toks, 0
        selected: list[int] = []
        for idx, (tok, prob) in enumerate(zip(draft_toks, draft_probs)):
            if prob < self.draft_confidence_threshold:
                return selected, self.k_drafts - idx
            selected.append(tok)
        return selected, 0

    def _run_verify(self, block_ids, state):
        if self.use_cudagraph and self._cg_runner is not None:
            try:
                return self._cg_runner(block_ids, state)
            except Exception as exc:
                if "InvalidCxxCompiler" in str(exc) or "cl is not found" in str(exc):
                    self.use_cudagraph = False
                    self._cg_runner = None
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
        drafts, init_pruned = self._select_drafts_from_hidden(hidden0)
        # Warmup verify buckets before timing when compiling/capturing per-bucket CUDAGraphs.
        if self.backend is not None:
            bucket_lens = range(1, self.k_drafts + 2) if self.draft_confidence_threshold > 0.0 else (self.k_drafts + 1,)
            for b_len in bucket_lens:
                warm_block = torch.full((1, b_len), next_tok, dtype=torch.long, device=self.device)
                self._run_verify(warm_block, cur_state)

        emitted: list[int] = []
        cycles = 0
        accepted_drafts = 0
        proposed_drafts = 0
        pruned_drafts = init_pruned
        sync(self.device)
        monitor = NVMLPowerMonitor(self.device) if measure_energy else None
        if monitor is not None:
            monitor.__enter__()
        t0 = time.perf_counter()
        try:
            while len(emitted) < new_tokens:
                cur_k = len(drafts)
                block = torch.tensor([[next_tok, *drafts]], dtype=torch.long, device=self.device)
                toks_table, probs_table, stacked_states = self._run_verify(block, cur_state)
                rows = toks_table.tolist()
                prob_rows = probs_table.tolist() if self.draft_confidence_threshold > 0.0 else None
                cycles += 1
                emitted.append(next_tok)
                if len(emitted) >= new_tokens:
                    break
                proposed_drafts += cur_k
                last_idx = cur_k
                for m in range(cur_k):
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
                if prob_rows is not None:
                    drafts, step_pruned = self._select_drafts_from_row(rows[last_idx][1:], prob_rows[last_idx])
                    pruned_drafts += step_pruned
                else:
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
            "pruned_drafts": pruned_drafts,
            "draft_confidence_threshold": self.draft_confidence_threshold,
            "draft_acceptance_rate": (accepted_drafts / proposed_drafts) if proposed_drafts else 0.0,
            "mean_tokens_per_cycle": len(emitted) / max(cycles, 1),
            "decode_seconds": decode_s,
            "tokens_per_sec": len(emitted) / decode_s,
            "cudagraph": self.use_cudagraph,
            "nvml": nvml,
            "mj_per_token": mj_per_tok,
        }
        return emitted, stats


class AdaptiveEarlyExitDecode:
    """Adaptive depth (Early-Exit / ACT) decoder with optional 2-stage compiled CUDAGraph execution."""

    def __init__(
        self,
        model,
        *,
        confidence_threshold: float = 0.90,
        delta_threshold: float = 0.0,
        min_layers: int = 2,
        vocab_limit: int | None = None,
        backend: str | None = None,
    ):
        self.decoder = FractalDecode(model)
        self.model = model
        self.confidence_threshold = float(confidence_threshold)
        self.delta_threshold = float(delta_threshold)
        self.min_layers = max(1, min(int(min_layers), model.depth))
        self.vocab_limit = model.config.vocab_size if vocab_limit is None else int(vocab_limit)
        self.device = next(model.parameters()).device
        self.backend = None if backend is None else resolve_compile_backend(backend, self.device)
        self.use_cudagraph = bool(self.backend == "inductor" and self.device.type == "cuda" and self.min_layers < model.depth)
        self._compiled_mode = bool(self.backend is not None and self.min_layers < model.depth)
        self._prefix_runner = None
        self._skip_runner = None
        self._suffix_runner = None
        if self._compiled_mode:
            dec = self.decoder
            min_l = self.min_layers
            vlim = self.vocab_limit
            if torch._dynamo.config.cache_size_limit < 64:
                torch._dynamo.config.cache_size_limit = 64

            def _prefix_raw(tok, st):
                return dec.step_early_exit_prefix(tok, st, min_layers=min_l, vocab_limit=vlim)

            def _skip_raw(x, st, p_lvls, n_assoc, n_tail, n_win, n_phase):
                return dec.step_early_exit_skip(
                    x, st, p_lvls, n_assoc, n_tail, n_win, n_phase, min_layers=min_l
                )

            def _suffix_raw(x, a_out, v0, anc, c_phase, st, p_lvls, n_assoc, n_tail, n_win, n_phase):
                return dec.step_early_exit_suffix(
                    x, a_out, v0, anc, c_phase, st, p_lvls, n_assoc, n_tail, n_win, n_phase, min_layers=min_l
                )

            p_fn = torch.compile(_prefix_raw, backend=self.backend, fullgraph=True, dynamic=False)
            k_fn = torch.compile(_skip_raw, backend=self.backend, fullgraph=True, dynamic=False)
            s_fn = torch.compile(_suffix_raw, backend=self.backend, fullgraph=True, dynamic=False)
            if self.use_cudagraph:
                self._prefix_runner = _CUDAGraphRunner(p_fn, self.device)
                self._skip_runner = _CUDAGraphRunner(k_fn, self.device)
                self._suffix_runner = _CUDAGraphRunner(s_fn, self.device)
            else:
                self._prefix_runner = p_fn
                self._skip_runner = k_fn
                self._suffix_runner = s_fn

    def prefill(self, ids: torch.Tensor, capacity: int):
        return self.decoder.prefill(ids, capacity)

    def _warmup_compiled(self, tok: torch.Tensor, state) -> None:
        if not self._compiled_mode:
            return
        out_p = self._prefix_runner(tok, state)
        x, a_out, v0, anc, c_phase, p_lvls, n_assoc, n_tail, n_win, n_phase, _, _, _ = out_p
        _ = self._skip_runner(x, state, p_lvls, n_assoc, n_tail, n_win, n_phase)
        _ = self._suffix_runner(x, a_out, v0, anc, c_phase, state, p_lvls, n_assoc, n_tail, n_win, n_phase)

    def step(self, token: torch.Tensor, state):
        if self._compiled_mode:
            out_p = self._prefix_runner(token, state)
            x, a_out, v0, anc, c_phase, p_lvls, n_assoc, n_tail, n_win, n_phase, inter_logits, top_prob, rel_delta = out_p
            halted = False
            if self.delta_threshold > 0.0 and float(rel_delta.item()) <= self.delta_threshold:
                halted = True
            elif self.confidence_threshold < 1.0 and float(top_prob.item()) >= self.confidence_threshold:
                halted = True
            if halted:
                next_state = self._skip_runner(x, state, p_lvls, n_assoc, n_tail, n_win, n_phase)
                return inter_logits, next_state, self.min_layers
            logits, next_state = self._suffix_runner(
                x, a_out, v0, anc, c_phase, state, p_lvls, n_assoc, n_tail, n_win, n_phase
            )
            return logits, next_state, self.model.depth

        logits, _, next_state, layers_exec = self.decoder.step_adaptive(
            token,
            state,
            confidence_threshold=self.confidence_threshold,
            delta_threshold=self.delta_threshold,
            min_layers=self.min_layers,
            vocab_limit=self.vocab_limit,
        )
        return logits, next_state, layers_exec

    @torch.inference_mode()
    def generate_greedy(self, prefix: torch.Tensor, new_tokens: int, *, measure_energy: bool = False):
        if prefix.ndim != 2 or prefix.shape[0] != 1:
            raise ValueError("AdaptiveEarlyExitDecode requires batch 1")
        if new_tokens < 1:
            return [], {"tokens": 0, "mean_layers_executed": 0.0, "layers_saved_ratio": 0.0}
        cap = prefix.shape[1] + new_tokens + 1
        logits, state = self.prefill(prefix, cap)
        next_tok = int(logits[0, -1, :self.vocab_limit].argmax(dim=-1).item())
        if self._compiled_mode:
            self._warmup_compiled(torch.tensor([[next_tok]], dtype=torch.long, device=self.device), state)
        emitted: list[int] = [next_tok]
        layers_hist: list[int] = []
        early_exits = 0
        depth = self.model.depth

        sync(self.device)
        monitor = NVMLPowerMonitor(self.device) if measure_energy else None
        if monitor is not None:
            monitor.__enter__()
        t0 = time.perf_counter()
        try:
            while len(emitted) < new_tokens:
                tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
                logits, state, l_exec = self.step(tok_tensor, state)
                layers_hist.append(l_exec)
                if l_exec < depth:
                    early_exits += 1
                next_tok = int(logits[0, -1, :self.vocab_limit].argmax(dim=-1).item())
                emitted.append(next_tok)
            sync(self.device)
            decode_s = max(time.perf_counter() - t0, 1e-9)
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)

        nvml = None if monitor is None else monitor.summary
        steps_count = max(len(layers_hist), 1)
        mean_l = sum(layers_hist) / steps_count if layers_hist else float(depth)
        saved_ratio = max(0.0, 1.0 - (mean_l / max(depth, 1)))
        mj_per_tok = (
            (nvml["energy_joules"] * 1000.0) / len(emitted)
            if nvml is not None and nvml.get("energy_joules") is not None and emitted
            else None
        )
        stats = {
            "tokens": len(emitted),
            "decode_steps": len(layers_hist),
            "total_depth": depth,
            "mean_layers_executed": mean_l,
            "layers_saved_ratio": saved_ratio,
            "early_exit_steps": early_exits,
            "early_exit_rate": (early_exits / len(layers_hist)) if layers_hist else 0.0,
            "decode_seconds": decode_s,
            "tokens_per_sec": len(emitted) / decode_s,
            "cudagraph": self.use_cudagraph,
            "nvml": nvml,
            "mj_per_token": mj_per_tok,
        }
        return emitted, stats


class SegmentMemoryCache:
    """Prefix state cache & multi-segment memory anchor bank (arXiv:2602.24281 inspiration) for recurrent states."""

    def __init__(self, *, max_entries: int = 32, max_anchors: int = 8):
        if max_entries < 1 or max_anchors < 1:
            raise ValueError("max_entries and max_anchors must be positive")
        self.max_entries = int(max_entries)
        self.max_anchors = int(max_anchors)
        self._prefix_cache: dict[tuple[int, ...], tuple[torch.Tensor, object]] = {}
        self._anchors: list[object] = []

    @staticmethod
    def clone_state(state):
        return _map_tensors(state, lambda t: t.detach().clone())

    def clear(self) -> None:
        self._prefix_cache.clear()
        self._anchors.clear()

    def put_prefix(self, token_ids_1d: tuple[int, ...], logits: torch.Tensor, state) -> None:
        if len(self._prefix_cache) >= self.max_entries and token_ids_1d not in self._prefix_cache:
            oldest = next(iter(self._prefix_cache))
            self._prefix_cache.pop(oldest, None)
        self._prefix_cache[token_ids_1d] = (logits.detach().clone(), self.clone_state(state))

    @torch.inference_mode()
    def prefill_cached(self, decoder, prefix: torch.Tensor, capacity: int):
        """Reuse longest cached prefix state in O(1), computing only the uncached suffix."""
        if prefix.ndim != 2 or prefix.shape[0] != 1:
            raise ValueError("SegmentMemoryCache.prefill_cached requires batch=1")
        key = tuple(int(x) for x in prefix[0].tolist())
        if key in self._prefix_cache:
            logits, st = self._prefix_cache[key]
            return logits.clone(), self.clone_state(st), {
                "cache_hit": "exact",
                "cached_tokens": len(key),
                "computed_tokens": 0,
            }
        best_k = 0
        best_state = None
        for k in range(len(key) - 1, 0, -1):
            sub = key[:k]
            if sub in self._prefix_cache:
                best_k = k
                _, best_state = self._prefix_cache[sub]
                break
        if best_state is not None and best_k > 0:
            cur_state = self.clone_state(best_state)
            suffix = prefix[:, best_k:]
            if hasattr(decoder, "step_hidden"):
                logits, _, cur_state = decoder.step_hidden(suffix, cur_state, return_all_states=False)
                logits = logits[:, -1:]
            else:
                for tok in suffix.split(1, dim=1):
                    logits, cur_state = decoder.step(tok, cur_state)
            self.put_prefix(key, logits, cur_state)
            return logits, cur_state, {
                "cache_hit": "partial",
                "cached_tokens": best_k,
                "computed_tokens": len(key) - best_k,
            }
        logits, cur_state = decoder.prefill(prefix, capacity)
        self.put_prefix(key, logits, cur_state)
        return logits, cur_state, {
            "cache_hit": "miss",
            "cached_tokens": 0,
            "computed_tokens": len(key),
        }

    def record_anchor(self, state) -> None:
        """Store a segment-boundary FractalDeltaState snapshot in the anchor ring buffer."""
        if len(self._anchors) >= self.max_anchors:
            self._anchors.pop(0)
        self._anchors.append(self.clone_state(state))

    def blend_with_anchors(self, state, *, anchor_weight: float = 0.15, decay: float = 0.7):
        """Blend active FP32 recurrent memory matrices with exponentially weighted segment anchors."""
        if not 0.0 <= anchor_weight <= 1.0:
            raise ValueError("anchor_weight must be in [0.0, 1.0]")
        if not self._anchors or anchor_weight == 0.0:
            return self.clone_state(state)
        n = len(self._anchors)
        raw_w = [decay ** (n - 1 - idx) for idx in range(n)]
        w_sum = sum(raw_w)
        weights = [w / w_sum for w in raw_w]

        blended_levels = []
        for lvl_idx, cur_lvl in enumerate(state.levels):
            anchor_mem = torch.zeros_like(cur_lvl.memory, dtype=torch.float32)
            for w, anc in zip(weights, self._anchors):
                anchor_mem = anchor_mem + float(w) * anc.levels[lvl_idx].memory.float()
            new_mem = ((1.0 - anchor_weight) * cur_lvl.memory.float() + anchor_weight * anchor_mem).contiguous()
            blended_levels.append(FractalLevelState(new_mem, cur_lvl.tail.clone()))

        anchor_assoc = torch.zeros_like(state.assoc_memory, dtype=torch.float32)
        for w, anc in zip(weights, self._anchors):
            anchor_assoc = anchor_assoc + float(w) * anc.assoc_memory.float()
        new_assoc = ((1.0 - anchor_weight) * state.assoc_memory.float() + anchor_weight * anchor_assoc).contiguous()
        return FractalDeltaState(
            tuple(blended_levels),
            new_assoc,
            state.local_tail.clone(),
            state.window,
            None if state.phase_pos is None else state.phase_pos.clone(),
        )
