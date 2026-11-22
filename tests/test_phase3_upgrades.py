"""Unit and CUDA tests for Campaign 014 upgrades: gradient checkpointing, per-bucket CUDAGraph MTP & Early-Exit,
FineWeb-Edu train-only BPE corpus pipeline, external zero-shot + RULER evaluation, and #7 Engram CPU offload."""

import copy

import pytest
import torch

from llm2.architectures.engram_offload import EngramTableCPU, benchmark_semi_structured_24
from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from llm2.config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from llm2.research.decode_optimization import AdaptiveEarlyExitDecode, SelfSpeculativeMTPDecode
from llm2.research.external_eval import evaluate_lambada, evaluate_multiple_choice, evaluate_ruler_niah
from llm2.research.objectives import PredictiveObjective
from llm2.research.real_corpus import prepare_fineweb_edu


def _make_cfg(*, gradient_checkpointing: bool = False, n_layers: int = 4, vocab_size: int = 128) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=n_layers,
            n_heads=4,
            hidden_dim=96,
            max_seq_len=256,
            dropout=0.0,
        ),
        data=DataConfig(seq_len=32, batch_size=2, eval_batch_size=2),
        optimizer=OptimizerConfig(schedule_steps=100, warmup_steps=5),
        train=TrainConfig(max_steps=20, eval_interval=10, checkpoint_interval=10),
        research=ResearchConfig(
            architecture="fractal_delta",
            share_scales=True,
            scale_conditioning=False,
            fractal_prelude_coda=True,
            fractal_value_residual=True,
            fractal_golden_phase=True,
            fractal_assoc_delta=True,
            fractal_assoc_multilag=True,
            gradient_checkpointing=gradient_checkpointing,
            intermediate_weight=0.25,
            mtp_tokens=2,
            mtp_weight=0.25,
        ),
    )


def test_fractal_gradient_checkpointing_gradient_parity():
    torch.manual_seed(17)
    cfg_ref = _make_cfg(gradient_checkpointing=False)
    cfg_ck = _make_cfg(gradient_checkpointing=True)
    m_ref = FractalDeltaMemory(cfg_ref.model, cfg_ref.research)
    m_ck = FractalDeltaMemory(cfg_ck.model, cfg_ck.research)
    m_ck.load_state_dict(copy.deepcopy(m_ref.state_dict()))

    obj_ref = PredictiveObjective(m_ref, cfg_ref.research)
    obj_ck = PredictiveObjective(m_ck, cfg_ck.research)
    obj_ck.load_state_dict(copy.deepcopy(obj_ref.state_dict()))

    batch = {
        "input_ids": torch.randint(0, 128, (2, 24)),
        "targets": torch.randint(0, 128, (2, 24)),
    }
    out_ref = obj_ref(m_ref, batch)
    out_ref.loss.backward()
    out_ck = obj_ck(m_ck, batch)
    out_ck.loss.backward()

    torch.testing.assert_close(out_ck.loss.detach(), out_ref.loss.detach(), atol=1e-5, rtol=1e-5)
    for (n1, p1), (_, p2) in zip(m_ref.named_parameters(), m_ck.named_parameters()):
        assert p1.grad is not None and p2.grad is not None, f"missing grad for {n1}"
        torch.testing.assert_close(p2.grad, p1.grad, atol=1e-5, rtol=1e-4)


def test_compiled_bucket_confidence_gated_mtp_and_two_stage_early_exit():
    torch.manual_seed(29)
    cfg = _make_cfg(n_layers=6)
    model = FractalDeltaMemory(cfg.model, cfg.research).eval()
    obj = PredictiveObjective(model, cfg.research).eval()
    prefix = torch.randint(0, 128, (1, 12))

    mtp_ref = SelfSpeculativeMTPDecode(model, obj.mtp_heads, vocab_limit=128, backend=None, draft_confidence_threshold=0.0)
    mtp_gated = SelfSpeculativeMTPDecode(
        model, obj.mtp_heads, vocab_limit=128, backend="aot_eager", draft_confidence_threshold=0.25
    )
    toks_ref, _ = mtp_ref.generate_greedy(prefix, 10)
    toks_g, stats_g = mtp_gated.generate_greedy(prefix, 10)
    assert toks_ref == toks_g
    assert stats_g["pruned_drafts"] > 0

    ee_full = AdaptiveEarlyExitDecode(model, confidence_threshold=1.1, min_layers=2, backend="aot_eager")
    toks_ee_full, st_full = ee_full.generate_greedy(prefix, 10)
    assert toks_ee_full == toks_ref
    assert st_full["mean_layers_executed"] == 6.0

    ee_fast = AdaptiveEarlyExitDecode(model, confidence_threshold=0.0001, min_layers=2, backend="aot_eager")
    toks_ee_fast, st_fast = ee_fast.generate_greedy(prefix, 8)
    assert len(toks_ee_fast) == 8
    assert st_fast["mean_layers_executed"] == 2.0
    assert abs(st_fast["layers_saved_ratio"] - (4.0 / 6.0)) < 1e-6


def test_real_corpus_fineweb_dedup_split_and_train_only_bpe(tmp_path):
    base_docs = [
        f"Document number {i}: science, physics, and mathematics explain how electrical energy, thermal dissipation, and recurrent memory move across modern silicon chips and GPUs."
        for i in range(40)
    ]
    # Inject exact duplicates to verify global SHA-256 deduplication across splits
    docs_with_dups = base_docs[:10] + base_docs[:5] + base_docs[10:]
    data = prepare_fineweb_edu(
        tmp_path,
        train_bytes=1200,
        val_bytes=300,
        cal_val_bytes=300,
        test_bytes=300,
        vocab_size=320,
        docs_iter=docs_with_dups,
    )
    assert data.schema == "stories-doc-bpe-v1"
    assert data.vocab_size <= 320
    assert data.manifest["selection"]["duplicates_dropped"] == 5
    # Re-calling prepare_fineweb_edu reuses immutable directory cleanly
    data2 = prepare_fineweb_edu(
        tmp_path,
        train_bytes=1200,
        val_bytes=300,
        cal_val_bytes=300,
        test_bytes=300,
        vocab_size=320,
        docs_iter=[],
    )
    assert data2.fingerprint == data.fingerprint

    # Test external evaluation harness against this tokenizer & tiny model
    cfg = _make_cfg(n_layers=3, vocab_size=data.vocab_size)
    model = FractalDeltaMemory(cfg.model, cfg.research).eval()
    dev = torch.device("cpu")
    mc = evaluate_multiple_choice(
        model,
        data.tokenizer,
        [("Question: What moves heat?\nAnswer:", ["silicon chips", "wooden hats"], 0)],
        dev,
    )
    assert mc["examples"] == 1 and mc["acc"] in (0.0, 1.0)
    lam = evaluate_lambada(model, data.tokenizer, [base_docs[0]], dev)
    assert lam["examples"] == 1 and lam["mean_nll"] > 0.0
    ruler = evaluate_ruler_niah(model, data.tokenizer, dev, lengths=(64,), examples_per_len=2, use_segment_cache=True)
    assert "64" in ruler and ruler["64"]["examples"] == 2


def test_engram_table_cpu_offload_zero_vram_and_grad_flow():
    engram = EngramTableCPU(num_entries=4096, engram_dim=64, d_model=64, pin_memory=False)
    assert engram.host_table.device.type == "cpu"
    assert engram.host_table_bytes() == 4096 * 64 * 2
    assert engram.host_table_bytes() > engram.gpu_param_bytes() * 20

    hidden = torch.randn(2, 16, 64, requires_grad=True)
    token_ids = torch.randint(0, 128, (2, 16))
    out = engram(hidden, token_ids)
    assert out.shape == hidden.shape
    out.square().mean().backward()
    assert hidden.grad is not None and engram.proj.weight.grad is not None
    assert benchmark_semi_structured_24(64, 64, 16, torch.device("cpu"))["supported"] is False


@pytest.mark.cuda
def test_phase3_cuda_cudagraph_and_checkpointing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(351)

    cfg = _make_cfg(gradient_checkpointing=True, n_layers=6)
    model = FractalDeltaMemory(cfg.model, cfg.research).to(device)
    obj = PredictiveObjective(model, cfg.research).to(device)
    compiled = torch.compile(model, backend="inductor", dynamic=False)
    batch = {
        "input_ids": torch.randint(0, 128, (2, 64), device=device),
        "targets": torch.randint(0, 128, (2, 64), device=device),
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = obj(compiled, batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.embedding.weight.grad is not None and torch.isfinite(model.embedding.weight.grad).all()

    model.eval()
    obj.eval()
    prefix = torch.randint(0, 128, (1, 16), device=device)

    mtp_g = SelfSpeculativeMTPDecode(
        model, obj.mtp_heads, vocab_limit=128, backend="inductor", draft_confidence_threshold=0.25
    )
    assert mtp_g.use_cudagraph is True
    toks_g, st_g = mtp_g.generate_greedy(prefix, 12)
    assert len(toks_g) == 12 and st_g["cudagraph"] is True

    ee_cg = AdaptiveEarlyExitDecode(model, confidence_threshold=0.001, min_layers=2, backend="inductor")
    assert ee_cg.use_cudagraph is True
    toks_ee, st_ee = ee_cg.generate_greedy(prefix, 8)
    assert len(toks_ee) == 8 and st_ee["cudagraph"] is True and st_ee["mean_layers_executed"] == 2.0

    engram = EngramTableCPU(num_entries=8192, engram_dim=64, d_model=64, pin_memory=True).to(device)
    assert engram.host_table.device.type == "cpu"
    h_gpu = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)
    out_eng = engram(h_gpu, prefix)
    assert out_eng.device.type == "cuda" and out_eng.dtype == torch.bfloat16
