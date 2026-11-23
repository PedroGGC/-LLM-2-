"""O Ringue: apples-to-apples benchmark harness comparing Lumina V1 against open HuggingFace and GGUF contenders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Sequence
import urllib.request

import torch
from torch.nn import functional as F

from ..architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from .decode_optimization import CompiledDecode, NVMLPowerMonitor
from .external_eval import evaluate_hf_suite, evaluate_ruler_niah


@dataclass
class RingueEntry:
    name: str
    kind: str  # "lumina_v1" | "hf_teacher" | "gguf_server"
    unique_params: int | None
    quantization: str
    peak_vram_gib: float | None
    context_state_mib_16k: float | None
    decode_tok_per_s: float | None
    mj_per_token: float | None
    watts_mean: float | None
    lambada_acc: float | None = None
    lambada_ppl: float | None = None
    arc_easy_acc_norm: float | None = None
    hellaswag_acc_norm: float | None = None
    ruler_2k_nll: float | None = None
    ruler_8k_nll: float | None = None
    ruler_16k_nll: float | None = None
    composite_score: float | None = None
    score_per_gib: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_composite_score(
    *,
    lambada_acc: float | None,
    arc_easy_acc_norm: float | None,
    hellaswag_acc_norm: float | None,
) -> float | None:
    vals = [v * 100.0 for v in (lambada_acc, arc_easy_acc_norm, hellaswag_acc_norm) if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _estimate_kv_cache_mib_16k(model: torch.nn.Module) -> float | None:
    if isinstance(model, FractalDeltaMemory):
        return round(model.state_bytes_per_sequence() / (1024**2), 4)
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layers", None)
    n_kv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_heads", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_heads", None)
    if not (n_layers and n_kv and hidden and n_heads):
        return None
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
    # 2 (K,V) * layers * kv_heads * head_dim * 16384 tokens * 2 bytes (BF16/FP16)
    bytes_16k = 2 * int(n_layers) * int(n_kv) * int(head_dim) * 16384 * 2
    return round(bytes_16k / (1024**2), 2)


def _count_logical_parameters(model: torch.nn.Module) -> int:
    """Count unique parameters including 2 logical weights per byte in QuaRot INT4 packed_weight buffers."""
    total = sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())
    seen_bufs: set[int] = set()
    for name, buf in model.named_buffers():
        if name.endswith("packed_weight") and buf.dtype == torch.uint8 and buf.data_ptr() not in seen_bufs:
            seen_bufs.add(buf.data_ptr())
            total += int(buf.numel() * 2)
    return total


def _model_storage_bytes(model: torch.nn.Module) -> int:
    tensors = {t.data_ptr(): t for t in list(model.parameters()) + list(model.buffers())}
    return sum(int(t.numel() * t.element_size()) for t in tensors.values())


@torch.inference_mode()
def _measure_native_decode(
    model: torch.nn.Module,
    prefix: torch.Tensor,
    new_tokens: int,
    device: torch.device,
    *,
    compile_fractal: bool = True,
) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model_bytes = _model_storage_bytes(model)
    base_alloc = 0
    is_fractal = isinstance(model, FractalDeltaMemory)
    if is_fractal:
        dec = CompiledDecode(FractalDecode(model), backend="inductor") if (compile_fractal and device.type == "cuda") else FractalDecode(model)
        cap = prefix.shape[1] + new_tokens + 4
        # Warmup
        lg0, st0 = dec.prefill(prefix, cap)
        tok0 = torch.tensor([[int(lg0[0, -1].argmax().item())]], dtype=torch.long, device=device)
        _, _ = dec.step(tok0, st0)
        del lg0, st0, tok0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            base_alloc = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        monitor = NVMLPowerMonitor(device) if device.type == "cuda" else None
        if monitor is not None:
            monitor.__enter__()
        t0 = time.perf_counter()
        try:
            lg, st = dec.prefill(prefix, cap)
            nxt = int(lg[0, -1].argmax().item())
            emitted = [nxt]
            while len(emitted) < new_tokens:
                lg, st = dec.step(torch.tensor([[nxt]], dtype=torch.long, device=device), st)
                nxt = int(lg[0, -1].argmax().item())
                emitted.append(nxt)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = max(time.perf_counter() - t0, 1e-9)
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)
    else:
        # HuggingFace CausalLM with KV-cache
        _ = model.generate(prefix, max_new_tokens=4, do_sample=False, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            base_alloc = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        monitor = NVMLPowerMonitor(device) if device.type == "cuda" else None
        if monitor is not None:
            monitor.__enter__()
        t0 = time.perf_counter()
        try:
            out = model.generate(prefix, max_new_tokens=new_tokens, do_sample=False, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = max(time.perf_counter() - t0, 1e-9)
            emitted = out[0, prefix.shape[1] :].tolist()
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)

    nvml = None if monitor is None else monitor.summary
    n_out = max(len(emitted), 1)
    mj = (nvml["energy_joules"] * 1000.0) / n_out if (nvml and nvml.get("energy_joules") is not None) else None
    if device.type == "cuda":
        decode_delta = max(0, int(torch.cuda.max_memory_allocated(device)) - base_alloc)
        peak_gib = (model_bytes + decode_delta) / (1024**3)
    else:
        peak_gib = None
    return {
        "tokens": len(emitted),
        "decode_tok_per_s": round(len(emitted) / dt, 1),
        "mj_per_token": None if mj is None else round(mj, 2),
        "watts_mean": None if (nvml is None or nvml.get("watts_mean") is None) else round(nvml["watts_mean"], 2),
        "peak_vram_gib": None if peak_gib is None else round(peak_gib, 3),
    }


class _HFTokenizerAdapter:
    """Wrap any HuggingFace tokenizer to match llm2's simple .encode(str) / .decode(ids) / .eos_id contract."""

    def __init__(self, hf_tok):
        self._tok = hf_tok
        self.eos_id = int(hf_tok.eos_token_id) if getattr(hf_tok, "eos_token_id", None) is not None else 0

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)


def evaluate_native_contender(
    name: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    *,
    kind: str = "lumina_v1",
    quantization: str = "bf16",
    max_examples: int = 50,
    ruler_lengths: Sequence[int] = (2048, 8192),
    decode_tokens: int = 128,
    compile_fractal: bool = True,
) -> tuple[RingueEntry, dict]:
    """Run full Ringue evaluation on a native PyTorch model (FractalDelta V2 or HuggingFace CausalLM)."""
    model.eval()
    tok = tokenizer if hasattr(tokenizer, "eos_id") and not hasattr(tokenizer, "is_fast") else _HFTokenizerAdapter(tokenizer)
    unique_params = _count_logical_parameters(model)

    # Wrap HF model so score_continuation_nll's model.config.max_seq_len works uniformly
    if not hasattr(model.config, "max_seq_len"):
        model.config.max_seq_len = int(getattr(model.config, "max_position_embeddings", 4096))

    suite = evaluate_hf_suite(model, tok, device, max_examples=max_examples, ruler_lengths=ruler_lengths)
    if isinstance(model, FractalDeltaMemory):
        suite["ruler_niah_grm"] = evaluate_ruler_niah(
            model, tok, device, lengths=ruler_lengths, examples_per_len=8, use_grm=True
        )

    prefix_ids = tok.encode("In modern artificial intelligence research, efficient recurrent architectures")[:64] or [1, 2, 3, 4]
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    dec = _measure_native_decode(model, prefix, decode_tokens, device, compile_fractal=compile_fractal)

    lam = suite.get("lambada", {})
    arc = suite.get("arc_easy", {})
    hel = suite.get("hellaswag", {})
    ruler = suite.get("ruler_niah_grm") or suite.get("ruler_niah", {})

    comp = compute_composite_score(
        lambada_acc=lam.get("exact_match"),
        arc_easy_acc_norm=arc.get("acc_norm"),
        hellaswag_acc_norm=hel.get("acc_norm"),
    )
    vram = dec["peak_vram_gib"]
    score_per_gib = round(comp / max(vram, 0.05), 2) if (comp is not None and vram is not None) else None

    entry = RingueEntry(
        name=name,
        kind=kind,
        unique_params=unique_params,
        quantization=quantization,
        peak_vram_gib=vram,
        context_state_mib_16k=_estimate_kv_cache_mib_16k(model),
        decode_tok_per_s=dec["decode_tok_per_s"],
        mj_per_token=dec["mj_per_token"],
        watts_mean=dec["watts_mean"],
        lambada_acc=None if lam.get("exact_match") is None else round(lam["exact_match"] * 100.0, 2),
        lambada_ppl=None if lam.get("ppl") is None else round(lam["ppl"], 2),
        arc_easy_acc_norm=None if arc.get("acc_norm") is None else round(arc["acc_norm"] * 100.0, 2),
        hellaswag_acc_norm=None if hel.get("acc_norm") is None else round(hel["acc_norm"] * 100.0, 2),
        ruler_2k_nll=ruler.get("2048", {}).get("mean_answer_nll"),
        ruler_8k_nll=ruler.get("8192", {}).get("mean_answer_nll"),
        ruler_16k_nll=ruler.get("16384", {}).get("mean_answer_nll"),
        composite_score=comp,
        score_per_gib=score_per_gib,
    )
    return entry, {"suite": suite, "decode": dec}


def query_gguf_server_completion(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout_s: float = 60.0,
) -> dict:
    """Query an OpenAI-compatible llama-server / KoboldCpp GGUF endpoint for Ringue throughput & output evaluation."""
    url = base_url.rstrip("/") + "/v1/completions"
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = max(time.perf_counter() - t0, 1e-9)
    text = body.get("choices", [{}])[0].get("text", "")
    n_tok = int(body.get("usage", {}).get("completion_tokens") or max(len(text.split()), 1))
    return {"text": text, "tokens": n_tok, "seconds": round(dt, 4), "tok_per_s": round(n_tok / dt, 1)}


def format_ringue_markdown(entries: Sequence[RingueEntry]) -> str:
    """Format Ringue contender entries into a clean GitHub Markdown leaderboard."""
    headers = [
        "Modelo",
        "Quant",
        "Params",
        "VRAM Pico (GiB)",
        "Estado/KV @16K (MiB)",
        "tok/s",
        "mJ/tok",
        "LAMBADA %",
        "ARC-E %",
        "HellaSwag %",
        "Score",
        "Score/GiB",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |"]

    def _f(v, suffix=""):
        return "—" if v is None else f"{v}{suffix}"

    for e in entries:
        p_str = "—" if e.unique_params is None else f"{e.unique_params / 1e6:.1f}M"
        row = [
            e.name,
            e.quantization,
            p_str,
            _f(e.peak_vram_gib),
            _f(e.context_state_mib_16k),
            _f(e.decode_tok_per_s),
            _f(e.mj_per_token),
            _f(e.lambada_acc),
            _f(e.arc_easy_acc_norm),
            _f(e.hellaswag_acc_norm),
            _f(e.composite_score),
            _f(e.score_per_gib),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
