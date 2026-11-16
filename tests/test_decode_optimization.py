import json

import pytest
import torch
from torch.nn import functional as F

from llm2.research import decode_benchmark as bench
from llm2.research.context_protocol import configuration
from llm2.research.decode_cache import RecurrentDecode, generate, verify_logits
from llm2.research.decode_optimization import (
    CompiledDecode,
    LocalTransformerDecode,
    WindowSDPA,
    canonical_recurrent_state,
    resolve_compile_backend,
)
from llm2.research.factory import build_backbone
from llm2.runtime import attention_context
from test_stories_data import fixture_data


def test_tiled_local_attention_matches_independent_dense_mask():
    q, k, v = torch.randn(3, 2, 4, 47, 8).unbind()
    positions = torch.arange(47)
    distances = positions[:, None] - positions[None, :]
    mask = (distances >= 0) & (distances < 8)
    with attention_context(True):
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        actual = WindowSDPA(8, tile=11)(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("variant", bench.VARIANTS)
def test_step_captures_full_graph_across_cache_wraps_and_prefixes(tmp_path, variant):
    cfg = configuration(variant, tmp_path, tmp_path, 341, device="cpu", smoke=True)
    cfg.model.max_seq_len = 128
    model = build_backbone(cfg).eval()
    adapter = LocalTransformerDecode(model, span=8) if variant == "transformer_medium" else RecurrentDecode(model)
    traces = []

    def capture(graph, inputs):
        traces.append(graph)
        return graph.forward

    compiled = CompiledDecode(adapter, backend=capture)
    with attention_context(True):
        for length in (3, 7, 8, 9, 49):
            prefix, continuation = bench.paired_input(length, 29, torch.device("cpu"))
            parity = verify_logits(compiled, prefix, continuation)
            assert parity["positions"] == 30 and parity["max_abs_logit_error"] < 1e-5
        timing, sample = generate(compiled, prefix, 9)
        _, expected = generate(adapter, prefix, 9)
        assert torch.equal(sample, expected)
    assert len(traces) == 1
    if variant == "transformer_medium":
        assert timing["state_bytes"] == cfg.model.n_layers * 2 * 8 * cfg.model.d_model * 4 + 8


@pytest.mark.parametrize("variant", bench.VARIANTS)
def test_optimization_exports_mode_metrics_and_profiling(tmp_path, variant):
    data = fixture_data(tmp_path / "data")
    cfg = configuration(variant, data.root, tmp_path, 341, device="cpu", smoke=True)
    cfg.model.max_seq_len = 512
    model = build_backbone(cfg).eval()
    device = torch.device("cpu")
    val_length = max(row["bytes"] for row in data.docs["val"])
    nll = bench.score(model, data, "val", 256, 32, device)["nll"]
    local_nll = bench.score(model, data, "val", val_length, 4, device)["nll"]
    result = {"variant": variant, "seed": 341, "profiles": [], "checkpoint_nll_t256": nll}
    path = tmp_path / "worker.json"
    bench.evaluate_optimized(model, data, device, path, result, local_nll, lengths=(33,), new_bytes=8,
                             compile_backend="eager")
    report = json.loads(path.read_text())
    expected = ({"transformer_global_eager", "transformer_local_eager", "transformer_local_inductor"}
                if variant == "transformer_medium" else {"recurrent_eager", "recurrent_inductor"})
    assert {row["mode"] for row in report["profiles"]} == expected
    assert report["status"] == "complete" and not report["mode_errors"]
    for row in report["profiles"]:
        assert row["parity"]["positions"] == 9 and row["profiler"]["steps"] == 8
        assert row["profiler"]["cuda_events"] is None and row["peak_vram_bytes"] is None
        if row["mode"].endswith("inductor"):
            assert row["first_step_seconds"] > 0 and row["backend"] == "eager"


def test_target_compares_fastest_reference_and_requires_full_matrix():
    modes = [
        {"mode": "recurrent_eager", "total_seconds": 1.4},
        {"mode": "recurrent_inductor", "total_seconds": .25},
        {"mode": "transformer_global_eager", "total_seconds": 1.2},
        {"mode": "transformer_local_eager", "total_seconds": .6},
        {"mode": "transformer_local_inductor", "total_seconds": .4},
    ]
    rows = [{"seed": seed, "status": "complete", "profiles": [
        {"prefix_bytes": 7936, "decode_seconds": item["total_seconds"] * .99, **item} for item in modes
    ]} for seed in bench.SEEDS]
    result = bench.optimized_comparisons(rows)
    assert not result["twofold_long_generation"]
    assert all(row["reference_mode"] == "transformer_local_inductor" for row in result["pairs"])
    for row in rows:
        row["profiles"][1]["total_seconds"] = .15
    assert bench.optimized_comparisons(rows)["twofold_long_generation"]
    rows[0]["profiles"].pop()
    assert not bench.optimized_comparisons(rows)["twofold_long_generation"]


def test_canonical_recurrent_state_enforces_strict_contiguity():
    from llm2.architectures.context_memory import ContextCellState, ContextState

    # Construct intentionally non-contiguous tensors via transposition and slicing
    non_contig_memory = torch.randn(1, 16, 4, 20).transpose(1, 2)
    assert not non_contig_memory.is_contiguous()

    non_contig_tail = torch.randn(1, 10, 80)[:, -3:]
    assert non_contig_tail.stride() != (3 * 80, 80, 1)

    state = ContextState(
        cells=(ContextCellState(non_contig_memory, non_contig_tail),),
        pending=torch.randn(1, 5, 80)[:, :0],
        last_coarse=torch.randn(1, 80),
    )

    canonical = canonical_recurrent_state(state)

    for cell in canonical.cells:
        assert cell.memory.is_contiguous()
        assert cell.tail.is_contiguous()
        assert cell.tail.stride() == (3 * 80, 80, 1)
        assert cell.memory.stride() == (4 * 16 * 20, 16 * 20, 20, 1)
    if canonical.pending is not None:
        assert canonical.pending.is_contiguous()
    if canonical.last_coarse is not None:
        assert canonical.last_coarse.is_contiguous()


def test_recurrent_step_dynamo_zero_graph_breaks(tmp_path):
    cfg = configuration("context_untied2", tmp_path, tmp_path, 341, device="cpu", smoke=True)
    cfg.model.max_seq_len = 128
    model = build_backbone(cfg).eval()
    adapter = RecurrentDecode(model)

    prefix = torch.randint(0, 256, (1, 32))
    _, state = adapter.prefill(prefix, 48)
    token = torch.randint(0, 256, (1, 1))

    explanation = torch._dynamo.explain(adapter.step)(token, state)
    assert explanation.graph_break_count == 0
    assert explanation.op_count == 112


def test_backend_resolution_and_fallback():
    import shutil
    import sys

    # On CPU:
    resolved_cpu = resolve_compile_backend("inductor", device=torch.device("cpu"))
    if sys.platform == "win32" and shutil.which("cl") is None:
        assert resolved_cpu == "aot_eager"
    else:
        assert resolved_cpu == "inductor"

    # On CUDA:
    resolved_cuda = resolve_compile_backend("inductor", device=torch.device("cuda:0"))
    assert resolved_cuda == "inductor"

    # Other backends passed through:
    assert resolve_compile_backend("eager", device=torch.device("cpu")) == "eager"
    assert resolve_compile_backend("aot_eager", device=torch.device("cpu")) == "aot_eager"


@pytest.mark.parametrize("prefix_len", (256, 1024, 7936))
def test_compiled_recurrent_step_numerical_parity_all_prefix_lengths(tmp_path, prefix_len):
    cfg = configuration("context_untied2", tmp_path, tmp_path, 341, device="cpu", smoke=False)
    model = build_backbone(cfg).eval()
    compiled = CompiledDecode(RecurrentDecode(model), backend="inductor")

    prefix, continuation = bench.paired_input(prefix_len, 16, torch.device("cpu"))
    parity = verify_logits(compiled, prefix, continuation)

    assert parity["positions"] == 17
    assert parity["max_abs_logit_error"] < 5e-4
    assert parity["top1_agreement"] == 1.0


@torch.inference_mode()
def test_compiled_recurrent_decode_latency_vs_transformer_kv_cache(tmp_path):
    import time
    from llm2.research.recall_worker import state_storage_bytes

    cfg_r = configuration("context_untied2", tmp_path, tmp_path, 341, device="cpu", smoke=False)
    model_r = build_backbone(cfg_r).eval()
    dec_r = CompiledDecode(RecurrentDecode(model_r), backend="inductor")

    cfg_t = configuration("transformer_medium", tmp_path, tmp_path, 341, device="cpu", smoke=False)
    model_t = build_backbone(cfg_t).eval()
    dec_t = CompiledDecode(LocalTransformerDecode(model_t, span=256), backend="inductor")

    # Prefill
    prefix, continuation = bench.paired_input(256, 32, torch.device("cpu"))
    _, state_r = dec_r.prefill(prefix, 512)
    _, state_t = dec_t.prefill(prefix, 512)

    # State footprint verification: recurrent memory is O(1) constant (< 16 KB)
    recurrent_bytes = state_storage_bytes(state_r)
    transformer_bytes = state_storage_bytes(state_t)
    assert recurrent_bytes < transformer_bytes
    assert recurrent_bytes <= 16 * 1024
    assert transformer_bytes >= 64 * 1024

    # Warmup
    tok = continuation[:, :1]
    for _ in range(3):
        _, state_r = dec_r.step(tok, state_r)
        _, state_t = dec_t.step(tok, state_t)

    # Measure step latency
    t0 = time.perf_counter()
    for _ in range(10):
        _, state_r = dec_r.step(tok, state_r)
    r_step_ms = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        _, state_t = dec_t.step(tok, state_t)
    t_step_ms = (time.perf_counter() - t0) / 10 * 1000

    assert 0 < r_step_ms < 50.0
    assert 0 < t_step_ms < 50.0


@torch.inference_mode()
def test_fractal_step_hidden_and_all_states_parity():
    from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory
    from llm2.config import ModelConfig, ResearchConfig

    torch.manual_seed(77)
    cfg = ModelConfig(vocab_size=32, d_model=24, n_layers=4, n_heads=2, hidden_dim=48, max_seq_len=64, dropout=0.0)
    res = ResearchConfig(
        architecture="fractal_delta",
        share_scales=True,
        scale_conditioning=False,
        memory_dim=12,
        state_chunk_size=8,
        fractal_value_residual=True,
        fractal_golden_phase=True,
        fractal_leaky_budget=True,
        fractal_assoc_delta=True,
        fractal_assoc_val_dim=12,
        fractal_prelude_coda=True,
        fractal_assoc_multilag=True,
    )
    model = FractalDeltaMemory(cfg, res).eval()
    dec = FractalDecode(model)

    prefix = torch.randint(0, 32, (1, 10))
    block = torch.randint(0, 32, (1, 4))
    l0, s0 = dec.prefill(prefix, 20)

    # Sequential step-by-step reference
    seq_logits = []
    seq_states = []
    st = s0
    for idx in range(block.shape[1]):
        lg, st = dec.step(block[:, idx : idx + 1], st)
        seq_logits.append(lg)
        seq_states.append(st)
    ref_logits = torch.cat(seq_logits, dim=1)

    blk_logits, _, stacked_states = dec.step_hidden(block, s0, return_all_states=True)
    assert (blk_logits - ref_logits).abs().max().item() < 1e-5
    for idx, ref_st in enumerate(seq_states):
        sl = dec.slice_step_state(stacked_states, idx)
        assert torch.equal(sl.phase_pos, ref_st.phase_pos)
        assert (sl.assoc_memory - ref_st.assoc_memory).abs().max().item() < 1e-5
        assert (sl.local_tail - ref_st.local_tail).abs().max().item() < 1e-5
        for l_a, l_b in zip(sl.levels, ref_st.levels, strict=True):
            assert (l_a.memory - l_b.memory).abs().max().item() < 1e-5
            assert (l_a.tail - l_b.tail).abs().max().item() < 1e-5


@torch.inference_mode()
def test_self_speculative_mtp_exact_greedy_parity_and_acceptance():
    from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory
    from llm2.config import ModelConfig, ResearchConfig
    from llm2.research.decode_optimization import SelfSpeculativeMTPDecode

    torch.manual_seed(123)
    cfg = ModelConfig(vocab_size=24, d_model=24, n_layers=4, n_heads=2, hidden_dim=48, max_seq_len=64, dropout=0.0)
    res = ResearchConfig(
        architecture="fractal_delta",
        share_scales=True,
        scale_conditioning=False,
        memory_dim=12,
        state_chunk_size=8,
        fractal_value_residual=True,
        fractal_golden_phase=True,
        fractal_leaky_budget=True,
        fractal_assoc_delta=True,
        fractal_assoc_val_dim=12,
        fractal_prelude_coda=True,
        fractal_assoc_multilag=True,
    )
    model = FractalDeltaMemory(cfg, res).eval()
    base_dec = FractalDecode(model)
    prefix = torch.randint(0, 24, (1, 12))
    new_tokens = 18

    logits, st = base_dec.prefill(prefix, prefix.shape[1] + new_tokens + 4)
    ref_tokens = []
    for i in range(new_tokens):
        nxt = int(logits[0, -1].argmax(dim=-1).item())
        ref_tokens.append(nxt)
        if i + 1 < new_tokens:
            logits, st = base_dec.step(torch.tensor([[nxt]]), st)

    # Regime 1: random auxiliary heads (tests rejection rollback) across eager and compiled
    heads = torch.nn.ModuleList([torch.nn.Linear(24, 24, bias=False), torch.nn.Linear(24, 24, bias=False)]).eval()
    for backend in (None, "aot_eager"):
        spec = SelfSpeculativeMTPDecode(model, heads, vocab_limit=24, backend=backend)
        got, stats = spec.generate_greedy(prefix, new_tokens, measure_energy=False)
        assert got == ref_tokens
        assert stats["tokens"] == new_tokens
        assert stats["cycles"] <= new_tokens

    # Regime 2: tie heads[0] to exact next token on constant-hidden fixed point test via matching head[0] == lm_head
    # (exercises partial match acceptance path when consecutive greedy tokens repeat or coincide with head[0])
    heads_match0 = torch.nn.ModuleList([torch.nn.Linear(24, 24, bias=False), torch.nn.Linear(24, 24, bias=False)]).eval()
    heads_match0[0].weight.copy_(model.lm_head.weight)
    spec_m = SelfSpeculativeMTPDecode(model, heads_match0, vocab_limit=24, backend=None)
    got_m, _ = spec_m.generate_greedy(prefix, new_tokens)
    assert got_m == ref_tokens


def test_nvml_and_l2_pin_cpu_safe_and_smoke_benchmark(tmp_path):
    from llm2.research.decode_optimization import NVMLPowerMonitor, try_pin_l2_weights
    from llm2.research.ia2_benchmark import main as bench_main
    import json

    t = torch.zeros(4, 4)
    pin = try_pin_l2_weights([t], torch.device("cpu"))
    assert pin["supported"] is False
    assert pin["pinned_bytes"] == 0
    with NVMLPowerMonitor(torch.device("cpu")) as mon:
        _ = t + 1
    assert mon.summary is None

    out_dir = tmp_path / "bench_smoke_mtp"
    bench_main([
        "--output-dir", str(out_dir),
        "--arms", "fractal_pcc_mtp_muon_bpe",
        "--mode", "unified",
        "--tokenizer", "bpe",
        "--vocab-size", "384",
        "--measure-energy",
        "--pin-l2",
        "--smoke",
        "--device", "cpu",
    ])
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    dec = summary["arms"][0]["decode"]
    assert dec["l2_pin"]["supported"] is False
    assert dec["inductor"]["nvml"] is None
    assert dec["speculative_mtp"]["greedy_exact_match"] is True
    assert dec["speculative_mtp"]["nvml"] is None
