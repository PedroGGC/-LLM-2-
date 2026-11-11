import json

import pytest
import torch
from torch.nn import functional as F

from llm2.research import decode_benchmark as bench
from llm2.research.context_protocol import configuration
from llm2.research.decode_cache import RecurrentDecode, generate, verify_logits
from llm2.research.decode_optimization import CompiledDecode, LocalTransformerDecode, WindowSDPA
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
