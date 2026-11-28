"""Wave 1 verification tests: static-capacity Global KV, chunked prefill, GQA-DeltaNet t==1 fast path, and 16K-64K state scaling."""
import torch

from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory, SWAState
from llm2.architectures.state_kernels import dplr_chunked, dplr_sequential
from llm2.config import ModelConfig, ResearchConfig


def _build_small_hybrid(*, kv_quant: str = "none", delta_gqa: bool = True, max_seq_len: int = 128) -> FractalDeltaMemory:
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=64,
        n_layers=4,
        n_heads=8,
        hidden_dim=128,
        max_seq_len=max_seq_len,
        dropout=0.0,
        rope_theta=10000.0,
    )
    res = ResearchConfig(
        architecture="fractal_delta",
        share_scales=False,
        scale_conditioning=False,
        fractal_head_dim=8,
        fractal_hybrid_swa=True,
        fractal_swa_kv_heads=2,
        fractal_delta_gqa=delta_gqa,
        fractal_swa_global=True,
        fractal_swa_levels=[0, 3],
        fractal_swa_rope=True,
        fractal_qk_norm=True,
        fractal_kv_quant=kv_quant,
        fractal_assoc_channel=False,
        fractal_deltanet_rope=True,
        fractal_golden_phase=False,
    )
    model = FractalDeltaMemory(cfg, res).eval()
    return model


@torch.inference_mode()
def test_static_global_kv_parity_no_quant():
    model = _build_small_hybrid(kv_quant="none", delta_gqa=True)
    # Parameter formula invariant
    enum_params = sum(p.numel() for p in model.parameters())
    assert model.parameter_formula() == enum_params

    dec = FractalDecode(model)
    prefix = torch.randint(0, 64, (1, 10))
    cont = torch.randint(0, 64, (1, 6))
    cap = 24

    logits_dyn, st_dyn = dec.prefill(prefix, capacity=cap)
    logits_sta, st_sta = dec.prefill_static(prefix, capacity=cap, chunk_size=32)
    assert torch.allclose(logits_dyn, logits_sta, atol=1e-4, rtol=1e-4)

    # Verify static Global KV shapes stay fixed across decode steps
    for lvl_idx in (0, 3):
        lvl = st_sta.levels[lvl_idx]
        assert isinstance(lvl, SWAState)
        assert lvl.is_static is True
        assert lvl.k_tail.shape[2] == cap
        assert int(lvl.seen.item()) == 10

    for step_idx in range(cont.shape[1]):
        tok = cont[:, step_idx : step_idx + 1]
        logits_dyn, st_dyn = dec.step(tok, st_dyn)
        logits_sta, st_sta = dec.step(tok, st_sta)
        assert torch.allclose(logits_dyn, logits_sta, atol=1e-4, rtol=1e-4), (
            f"step {step_idx} max_diff={(logits_dyn - logits_sta).abs().max().item()}"
        )
        for lvl_idx in (0, 3):
            assert st_sta.levels[lvl_idx].k_tail.shape[2] == cap
            assert int(st_sta.levels[lvl_idx].seen.item()) == 10 + step_idx + 1


@torch.inference_mode()
def test_static_global_kv_parity_fp8_quant():
    model = _build_small_hybrid(kv_quant="fp8", delta_gqa=True)
    dec = FractalDecode(model)
    prefix = torch.randint(0, 64, (1, 10))
    cont = torch.randint(0, 64, (1, 6))
    cap = 24

    logits_dyn, st_dyn = dec.prefill(prefix, capacity=cap)
    logits_sta, st_sta = dec.prefill_static(prefix, capacity=cap, chunk_size=32)
    assert torch.allclose(logits_dyn, logits_sta, atol=1e-4, rtol=1e-4)

    for step_idx in range(cont.shape[1]):
        tok = cont[:, step_idx : step_idx + 1]
        logits_dyn, st_dyn = dec.step(tok, st_dyn)
        logits_sta, st_sta = dec.step(tok, st_sta)
        max_diff = (logits_dyn - logits_sta).abs().max().item()
        assert max_diff < 5e-2, f"fp8 static step {step_idx} diverged: max_diff={max_diff}"
        assert int(logits_dyn.argmax(dim=-1).item()) == int(logits_sta.argmax(dim=-1).item())
        for lvl_idx in (0, 3):
            assert st_sta.levels[lvl_idx].k_tail.shape[2] == cap
            assert st_sta.levels[lvl_idx].k_scale.shape[2] == cap


@torch.inference_mode()
def test_chunked_prefill_static_exact_parity():
    model = _build_small_hybrid(kv_quant="none", delta_gqa=True)
    dec = FractalDecode(model)
    prefix = torch.randint(0, 64, (1, 14))
    cont = torch.randint(0, 64, (1, 4))
    cap = 24

    logits_ref, st_ref = dec.prefill(prefix, capacity=cap)
    logits_chk, st_chk = dec.prefill_static(prefix, capacity=cap, chunk_size=4)
    assert torch.allclose(logits_ref, logits_chk, atol=1e-4, rtol=1e-4), (
        f"chunked prefill max_diff={(logits_ref - logits_chk).abs().max().item()}"
    )

    for step_idx in range(cont.shape[1]):
        tok = cont[:, step_idx : step_idx + 1]
        logits_ref, st_ref = dec.step(tok, st_ref)
        logits_chk, st_chk = dec.step(tok, st_chk)
        assert torch.allclose(logits_ref, logits_chk, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_dplr_sequential_t1_fast_path_parity():
    torch.manual_seed(7)
    for kv_heads in (8, 2):
        b, q_heads, d = 2, 8, 16
        q = torch.randn(b, q_heads, 1, d)
        k = torch.randn(b, kv_heads, 1, d)
        v = torch.randn(b, kv_heads, 1, d)
        log_w = -torch.rand(b, kv_heads, 1, 1) * 0.5
        a = -torch.rand(b, kv_heads, 1, d) * 0.1
        b_vec = torch.randn(b, kv_heads, 1, d) * 0.1
        init_mem = torch.randn(b, kv_heads, d, d)

        out_seq, mem_seq = dplr_sequential(q, k, v, log_w, a, b_vec, init_mem)
        out_chk, mem_chk = dplr_chunked(q, k, v, log_w, a, b_vec, init_mem, chunk_size=4, use_triton=False)
        assert torch.allclose(out_seq, out_chk, atol=1e-5, rtol=1e-5)
        assert torch.allclose(mem_seq, mem_chk, atol=1e-5, rtol=1e-5)


def test_state_bytes_scaling_16k_32k_64k():
    model = _build_small_hybrid(kv_quant="fp8", delta_gqa=True, max_seq_len=65536)
    s0 = model.state_bytes_per_sequence(seq_len=0)
    s16 = model.state_bytes_per_sequence(seq_len=16384)
    s32 = model.state_bytes_per_sequence(seq_len=32768)
    s64 = model.state_bytes_per_sequence(seq_len=65536)
    # Linear growth of Global KV-FP8 state across 16K -> 32K -> 64K
    assert (s32 - s16) == (s16 - s0)
    assert (s64 - s32) == 2 * (s32 - s16)


def test_qsa_microblocks_and_turbo4_static_decode():
    torch.manual_seed(11)
    model = _build_small_hybrid(kv_quant="fp8", delta_gqa=True, max_seq_len=512).eval()
    for cell in model.cells:
        if hasattr(cell, "qsa_block_size"):
            cell.kv_quant = "turbo4"
            cell.qsa_block_size = 4
            cell.qsa_top_blocks = 8
    dec = FractalDecode(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 48))
    with torch.inference_mode():
        lg, st = dec.prefill_static(ids, capacity=128, chunk_size=16)
        assert torch.isfinite(lg).all()
        nxt = lg[:, -1:].argmax(dim=-1)
        for _ in range(6):
            lg, st = dec.step(nxt, st)
            assert torch.isfinite(lg).all()
            nxt = lg[:, -1:].argmax(dim=-1)
