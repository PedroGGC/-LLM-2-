"""Unit & CUDA tests for Fase A (FractalDelta V2):
- Gated DeltaNet-2 decoupled erase/write gates (exact chunked vs sequential + Triton DPLR)
- Native inputs_embeds support (forward, prefill_embeds, step_hidden, grad flow for multimodal bridge)
- Hybrid 3:1 interleaved CausalLocalWindow + MTP return_all_states exact greedy parity
- Learned Exit Gate (Ouro/TaH2 self-distilled halting supervision + AdaptiveEarlyExitDecode)
- Real query-dependent GRM (Gated Residual Memory, arXiv:2602.24281) non-collapsing readout
"""

from __future__ import annotations

import pytest
import torch

from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from llm2.architectures.quarot import quantize_model_quarot_int4
from llm2.config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from llm2.optim import build_optimizer
from llm2.research.decode_optimization import (
    AdaptiveEarlyExitDecode,
    SegmentMemoryCache,
    SelfSpeculativeMTPDecode,
)
from llm2.research.factory import build_backbone, build_objective


def _make_v2_cfg(
    *,
    device: str = "cpu",
    precision: str = "fp32",
    d_model: int = 32,
    n_layers: int = 8,
    n_heads: int = 4,
    hidden_dim: int = 64,
    vocab_size: int = 64,
    seq_len: int = 32,
    dropout: float = 0.0,
    decoupled_gates: bool = True,
    hybrid_stride: int = 4,
    fractal_window: int = 8,
    exit_gate: bool = True,
    exit_gate_weight: float = 0.15,
    intermediate_weight: float = 0.2,
    mtp_tokens: int = 2,
    mtp_weight: float = 0.25,
    grad_ckpt: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max(128, seq_len * 2),
            dropout=dropout,
        ),
        data=DataConfig(seq_len=seq_len, batch_size=2, eval_batch_size=2),
        optimizer=OptimizerConfig(kind="adamw", lr=1e-3, warmup_steps=2, schedule_steps=20),
        train=TrainConfig(
            seed=42,
            device=device,
            precision=precision,
            deterministic=True,
            compile=False,
            max_steps=10,
            grad_accum_steps=1,
            eval_interval=10,
            checkpoint_interval=10,
            log_interval=5,
        ),
        research=ResearchConfig(
            architecture="fractal_delta",
            share_scales=True,
            scale_conditioning=False,
            fractal_prelude_coda=True,
            fractal_value_residual=True,
            fractal_golden_phase=True,
            fractal_assoc_delta=True,
            fractal_assoc_multilag=True,
            fractal_decoupled_gates=decoupled_gates,
            fractal_window=fractal_window,
            fractal_hybrid_stride=hybrid_stride,
            fractal_exit_gate=exit_gate,
            exit_gate_weight=exit_gate_weight,
            intermediate_weight=intermediate_weight,
            mtp_tokens=mtp_tokens,
            mtp_weight=mtp_weight,
            gradient_checkpointing=grad_ckpt,
            memory_dim=16,
            fractal_assoc_val_dim=8,
            state_chunk_size=8,
        ),
    )


def _unique_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())


def _state_nbytes(state) -> int:
    total = 0
    for lvl in state.levels:
        total += lvl.memory.numel() * lvl.memory.element_size()
        total += lvl.tail.numel() * lvl.tail.element_size()
    total += state.assoc_memory.numel() * state.assoc_memory.element_size()
    total += state.local_tail.numel() * state.local_tail.element_size()
    if state.window is not None:
        total += state.window.k_tail.numel() * state.window.k_tail.element_size()
        total += state.window.v_tail.numel() * state.window.v_tail.element_size()
        total += state.window.valid_tail.numel() * state.window.valid_tail.element_size()
    if state.hybrid_windows is not None:
        for hw in state.hybrid_windows:
            total += hw.k_tail.numel() * hw.k_tail.element_size()
            total += hw.v_tail.numel() * hw.v_tail.element_size()
            total += hw.valid_tail.numel() * hw.valid_tail.element_size()
    if state.phase_pos is not None:
        total += state.phase_pos.numel() * state.phase_pos.element_size()
    return total


def test_v2_parameter_formula_state_bytes_and_strict_causality():
    torch.manual_seed(101)
    cfg = _make_v2_cfg()
    model = build_backbone(cfg).eval()
    assert isinstance(model, FractalDeltaMemory)
    assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert model.parameter_formula() == _unique_param_count(model)
    assert model.hybrid_window_levels == (3, 7)

    ids = torch.randint(0, cfg.model.vocab_size, (1, 20))
    out = model(ids)
    assert torch.isfinite(out.logits).all()
    assert len(out.exit_logits) == cfg.model.n_layers - 1
    assert _state_nbytes(out.state) == model.state_bytes_per_sequence()

    # Strict causality: changing token at t=12 must not affect logits at t < 12.
    ids_pert = ids.clone()
    ids_pert[0, 12] = (ids_pert[0, 12] + 7) % cfg.model.vocab_size
    out_pert = model(ids_pert)
    assert torch.allclose(out.logits[:, :12], out_pert.logits[:, :12], atol=1e-6, rtol=0.0)
    assert not torch.allclose(out.logits[:, 12:], out_pert.logits[:, 12:], atol=1e-6, rtol=0.0)


def test_v2_decoupled_gates_chunked_vs_sequential_and_decode_parity():
    torch.manual_seed(202)
    cfg = _make_v2_cfg()
    model = build_backbone(cfg).eval()
    ids = torch.randint(0, cfg.model.vocab_size, (1, 19))

    out_chunked = model(ids)
    for cell in model.cells:
        cell.execution = "sequential"
    out_seq = model(ids)
    assert (out_chunked.logits - out_seq.logits).abs().max().item() < 1e-4

    # Restore chunked prefill and verify prefill + incremental step matches full forward.
    for cell in model.cells:
        cell.execution = "chunked"
    dec = FractalDecode(model)
    lg_pre, st = dec.prefill(ids[:, :11], 32)
    assert (lg_pre - out_chunked.logits[:, 10:11]).abs().max().item() < 1e-4
    step_logits = []
    for idx in range(11, 19):
        lg, st = dec.step(ids[:, idx : idx + 1], st)
        step_logits.append(lg)
    step_all = torch.cat(step_logits, dim=1)
    assert (step_all - out_chunked.logits[:, 11:]).abs().max().item() < 1e-4


def test_v2_inputs_embeds_equivalence_and_grad_flow():
    torch.manual_seed(303)
    cfg = _make_v2_cfg()
    model = build_backbone(cfg).eval()
    ids = torch.randint(0, cfg.model.vocab_size, (2, 16))

    out_ids = model(ids)
    embeds = model.embedding(ids).detach().clone().requires_grad_(True)
    out_emb = model(inputs_embeds=embeds)
    assert torch.equal(out_ids.logits, out_emb.logits)
    assert torch.equal(out_ids.hidden_states, out_emb.hidden_states)

    # Verify gradients flow cleanly back to inputs_embeds (multimodal projector readiness).
    out_emb.logits.sum().backward()
    assert embeds.grad is not None and torch.isfinite(embeds.grad).all()
    assert embeds.grad.abs().sum().item() > 0.0

    # Verify FractalDecode.prefill_embeds + step_hidden(inputs_embeds=...) parity.
    dec = FractalDecode(model)
    e1 = model.embedding(ids[:1]).detach()
    lg_pre, _, st = dec.prefill_embeds(e1[:, :10], capacity=32)
    lg_step, _, _ = dec.step_hidden(inputs_embeds=e1[:, 10:], state=st)
    assert (lg_pre - out_ids.logits[:1, 9:10]).abs().max().item() < 1e-4
    assert (lg_step - out_ids.logits[:1, 10:]).abs().max().item() < 1e-4

    # Validation errors on bad arguments.
    with pytest.raises(ValueError):
        model(ids, inputs_embeds=embeds)
    with pytest.raises(ValueError):
        model()
    with pytest.raises(ValueError):
        model(inputs_embeds=torch.zeros(1, 4, cfg.model.d_model + 4))


def test_v2_hybrid_3to1_mtp_exact_greedy_parity():
    torch.manual_seed(404)
    cfg = _make_v2_cfg(hybrid_stride=4, fractal_window=8, mtp_tokens=2, mtp_weight=0.25)
    model = build_backbone(cfg).eval()
    obj = build_objective(cfg, model).eval()
    prefix = torch.randint(0, cfg.model.vocab_size, (1, 12))

    # Reference greedy decode
    dec = FractalDecode(model)
    lg, st = dec.prefill(prefix, 64)
    nxt = int(lg[0, -1].argmax().item())
    ref = [nxt]
    for _ in range(15):
        lg, st = dec.step(torch.tensor([[nxt]]), st)
        nxt = int(lg[0, -1].argmax().item())
        ref.append(nxt)

    mtp = SelfSpeculativeMTPDecode(model, obj.mtp_heads, vocab_limit=cfg.model.vocab_size, backend=None, draft_confidence_threshold=0.0)
    emitted, stats = mtp.generate_greedy(prefix, 16)
    assert emitted == ref
    assert stats["tokens"] == 16


def test_v2_learned_exit_gate_training_and_halting():
    torch.manual_seed(505)
    cfg = _make_v2_cfg(exit_gate=True, exit_gate_weight=0.2, intermediate_weight=0.2, grad_ckpt=True)
    model = build_backbone(cfg).train()
    obj = build_objective(cfg, model).train()
    opt = build_optimizer(model, obj, cfg.optimizer)

    bx = torch.randint(0, cfg.model.vocab_size, (2, 16))
    by = torch.randint(0, cfg.model.vocab_size, (2, 16))
    opt.zero_grad(set_to_none=True)
    out = obj(model, {"input_ids": bx, "targets": by})
    assert torch.isfinite(out.loss)
    assert "exit_gate_loss" in out.metrics and "exit_gate_match_rate" in out.metrics
    out.loss.backward()
    assert model.exit_gate.weight.grad is not None and model.exit_gate.weight.grad.abs().sum().item() > 0.0
    opt.step()

    # Verify AdaptiveEarlyExitDecode respects learned exit_gate_threshold.
    model.eval()
    prefix = torch.randint(0, cfg.model.vocab_size, (1, 8))
    with torch.no_grad():
        model.exit_gate.weight.zero_()
        model.exit_gate.bias.fill_(10.0)
    ee_halt = AdaptiveEarlyExitDecode(
        model, confidence_threshold=1.01, delta_threshold=0.0, exit_gate_threshold=0.5, min_layers=4, backend=None
    )
    _, stats_halt = ee_halt.generate_greedy(prefix, 8)
    assert stats_halt["mean_layers_executed"] == 4.0
    assert stats_halt["early_exit_rate"] == 1.0

    with torch.no_grad():
        model.exit_gate.bias.fill_(-10.0)
    ee_full = AdaptiveEarlyExitDecode(
        model, confidence_threshold=1.01, delta_threshold=0.0, exit_gate_threshold=0.5, min_layers=4, backend=None
    )
    _, stats_full = ee_full.generate_greedy(prefix, 8)
    assert stats_full["mean_layers_executed"] == float(cfg.model.n_layers)
    assert stats_full["early_exit_rate"] == 0.0


def test_v2_query_dependent_grm_non_collapsing_readout():
    torch.manual_seed(606)
    cfg = _make_v2_cfg()
    model = build_backbone(cfg).eval()
    cache = SegmentMemoryCache(max_entries=8, max_anchors=4)

    ids_cat = torch.randint(0, cfg.model.vocab_size // 2, (1, 12))
    ids_dog = torch.randint(cfg.model.vocab_size // 2, cfg.model.vocab_size, (1, 12))
    out_cat = model(ids_cat)
    out_dog = model(ids_dog)

    # Construct orthogonal segment keys (simulating two distinct visual frames / chapters).
    d = cfg.model.d_model
    key_cat = torch.zeros(1, d)
    key_cat[0, : d // 2] = 1.0
    key_dog = torch.zeros(1, d)
    key_dog[0, d // 2 :] = 1.0

    cache.record_segment(out_cat.state, key_cat, tag="img_cat")
    cache.record_segment(out_dog.state, key_dog, tag="img_dog")

    q_cat = key_cat.unsqueeze(1)
    q_dog = key_dog.unsqueeze(1)
    read_cat, info_cat = cache.query_grm_readout(q_cat, model=model, top_k=1, temperature=0.1)
    read_dog, info_dog = cache.query_grm_readout(q_dog, model=model, top_k=1, temperature=0.1)

    assert info_cat["selected_tags"] == ["img_cat"]
    assert info_dog["selected_tags"] == ["img_dog"]
    assert torch.isfinite(read_cat).all() and torch.isfinite(read_dog).all()
    assert not torch.allclose(read_cat, read_dog, atol=1e-5)

    # Verify retrieve_grm_state preserves hybrid_windows and passes _validate_state.
    fused_state, st_info = cache.retrieve_grm_state(q_cat, active_state=out_dog.state, top_k=2, temperature=0.2, grm_weight=0.3)
    assert st_info["selected_tags"][0] == "img_cat"
    assert fused_state.hybrid_windows is not None and len(fused_state.hybrid_windows) == len(model.hybrid_windows)
    out_after = model(ids_dog[:, :4], state=fused_state)
    assert torch.isfinite(out_after.logits).all()


def test_v2_strict_exact_checkpoint_resume_cpu_fp32(tmp_path):
    """Strict resume equivalence (AGENTS.md): CPU FP32, active dropout, grad_accum=2, full FractalDelta V2."""
    import copy
    from llm2.research.ia2_benchmark import ARMS, UnifiedTrainer, make_story_config
    from test_stories_data import fixture_data

    data = fixture_data(tmp_path / "stories")
    arm = next(a for a in ARMS if a.name == "fractal_v2_bpe")

    def _cfg(out_dir):
        cfg = make_story_config(arm, data.root, out_dir, seed=355, lr=0.002, smoke=True, device="cpu")
        cfg.model.dropout = 0.2
        cfg.data.batch_size = 2
        cfg.data.eval_batch_size = 2
        cfg.train.max_steps = 2
        cfg.train.checkpoint_interval = 1
        cfg.train.eval_interval = 10
        cfg.train.grad_accum_steps = 2
        cfg.train.deterministic = True
        cfg.optimizer.kind = "adamw"
        cfg.research.fractal_window = 8
        cfg.research.fractal_hybrid_stride = 4
        return cfg

    with UnifiedTrainer(_cfg(tmp_path / "full"), smoke=True, unified_curriculum=False) as full:
        full.fit()
        full_state = {k: v.clone() for k, v in full.model.state_dict().items()}
        full_sched = copy.deepcopy(full.scheduler.state_dict())

    cfg_part = _cfg(tmp_path / "part")
    cfg_part.train.max_steps = 1
    with UnifiedTrainer(cfg_part, smoke=True, unified_curriculum=False) as part:
        part.fit()
        ckpt1 = part.last_checkpoint

    with UnifiedTrainer(_cfg(tmp_path / "resumed"), smoke=True, unified_curriculum=False, resume=ckpt1) as res:
        res.fit()
        res_state = {k: v.clone() for k, v in res.model.state_dict().items()}
        res_sched = copy.deepcopy(res.scheduler.state_dict())

    for key in full_state:
        assert torch.equal(full_state[key], res_state[key]), f"Mismatch on parameter {key}"
    assert full_sched == res_sched


@pytest.mark.cuda
def test_v2_cuda_bf16_compile_quarot_and_cudagraph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(707)
    cfg = _make_v2_cfg(
        device="cuda",
        precision="bf16",
        d_model=64,
        n_layers=8,
        n_heads=4,
        hidden_dim=128,
        vocab_size=128,
        seq_len=64,
        grad_ckpt=True,
    )
    model = build_backbone(cfg).to(device).train()
    obj = build_objective(cfg, model).to(device).train()
    opt = build_optimizer(model, obj, cfg.optimizer)
    comp = torch.compile(model, backend="inductor", dynamic=False)

    bx = torch.randint(0, 128, (2, 64), device=device)
    by = torch.randint(0, 128, (2, 64), device=device)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = obj(comp, {"input_ids": bx, "targets": by})
    assert torch.isfinite(out.loss)
    out.loss.backward()
    opt.step()

    model.eval()
    obj.eval()
    m_bf16 = build_backbone(cfg).to(device=device, dtype=torch.bfloat16).eval()
    m_bf16.load_state_dict(model.state_dict())
    q_m, _ = quantize_model_quarot_int4(m_bf16, group_size=32, act_bits=16, inplace=False, use_triton=True)
    q_m = q_m.to(device=device, dtype=torch.bfloat16).eval()

    prefix = torch.randint(0, 128, (1, 16), device=device)
    ee = AdaptiveEarlyExitDecode(
        q_m, confidence_threshold=1.01, exit_gate_threshold=0.05, min_layers=4, vocab_limit=128, backend="inductor"
    )
    toks_ee, st_ee = ee.generate_greedy(prefix, 12)
    assert len(toks_ee) == 12 and st_ee["cudagraph"] is True

    mtp_heads = obj.mtp_heads.to(device=device, dtype=torch.bfloat16).eval()
    mtp = SelfSpeculativeMTPDecode(
        m_bf16, mtp_heads, vocab_limit=128, backend="inductor", draft_confidence_threshold=0.2
    )
    toks_mtp, st_mtp = mtp.generate_greedy(prefix, 12)
    assert len(toks_mtp) == 12 and st_mtp["cudagraph"] is True
