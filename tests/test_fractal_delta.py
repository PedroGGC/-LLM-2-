import copy

import pytest
import torch

from llm2.architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from llm2.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    ResearchConfig,
    TrainConfig,
)
from llm2.research.accounting import estimate_research_flops
from llm2.research.decode_cache import generate, verify_logits
from llm2.research.decode_optimization import CompiledDecode
from llm2.research.factory import build_backbone
from llm2.research.recall_worker import state_storage_bytes
from llm2.research.stories_worker import StoryTrainer
from test_stories_data import fixture_data
from test_training import assert_nested_equal


def _fractal_cfg(
    data_root,
    output_dir,
    *,
    depth=4,
    shared=True,
    d_model=32,
    heads=4,
    hidden_dim=64,
    dropout=0.0,
    device="cpu",
    window=0,
    value_residual=False,
    golden_phase=False,
    free_energy=False,
    weight_mode="dense",
    leaky_budget=False,
    assoc_delta=False,
):
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=257,
            d_model=d_model,
            n_layers=depth,
            n_heads=heads,
            hidden_dim=hidden_dim,
            max_seq_len=128,
            dropout=dropout,
        ),
        data=DataConfig(
            path=str(data_root),
            seq_len=32,
            batch_size=2,
            eval_batch_size=2,
        ),
        optimizer=OptimizerConfig(
            lr=1e-3,
            weight_decay=0.1,
            warmup_steps=1,
            schedule_steps=20,
        ),
        train=TrainConfig(
            seed=341,
            max_steps=6,
            grad_accum_steps=2,
            eval_interval=10,
            checkpoint_interval=2,
            log_interval=10,
            output_dir=str(output_dir),
            device=device,
            precision="fp32",
            deterministic=True,
        ),
        research=ResearchConfig(
            architecture="fractal_delta",
            share_scales=shared,
            scale_conditioning=False,
            state_execution="chunked",
            state_chunk_size=16,
            fractal_beta_scale=2.0,
            fractal_window=window,
            fractal_value_residual=value_residual,
            fractal_golden_phase=golden_phase,
            fractal_free_energy=free_energy,
            fractal_weight_mode=weight_mode,
            fractal_leaky_budget=leaky_budget,
            fractal_assoc_delta=assoc_delta,
        ),
    )


@pytest.mark.parametrize(
    "shared,depth,window,value_residual,golden_phase,free_energy,weight_mode,leaky_budget,assoc_delta",
    [
        (True, 4, 0, False, False, False, "dense", False, False),
        (False, 2, 0, False, False, False, "dense", False, False),
        (True, 1, 0, False, False, False, "dense", False, False),
        (True, 4, 8, False, False, False, "dense", False, False),
        (True, 4, 0, True, False, False, "dense", False, False),
        (True, 4, 0, True, True, False, "dense", False, False),
        (True, 4, 0, True, True, False, "dense", True, True),
        (True, 4, 0, True, False, True, "dense", False, False),
        (True, 4, 0, True, False, False, "ternary", False, False),
        (True, 4, 0, True, False, False, "quaternion", False, False),
    ],
)
def test_fractal_delta_causality_streaming_chunked_vs_seq_and_params(
    tmp_path, shared, depth, window, value_residual, golden_phase, free_energy, weight_mode, leaky_budget, assoc_delta
):
    cfg = _fractal_cfg(
        tmp_path,
        tmp_path / "run",
        depth=depth,
        shared=shared,
        dropout=0.0,
        window=window,
        value_residual=value_residual,
        golden_phase=golden_phase,
        free_energy=free_energy,
        weight_mode=weight_mode,
        leaky_budget=leaky_budget,
        assoc_delta=assoc_delta,
    )
    torch.manual_seed(17)
    model = build_backbone(cfg).eval()
    assert type(model) is FractalDeltaMemory
    assert model.lm_head.weight is model.embedding.weight
    assert model.parameter_formula() == sum(p.numel() for p in model.parameters())

    ids = torch.randint(0, 257, (2, 53))
    full = model(ids)
    assert state_storage_bytes(model(ids[:1]).state) == model.state_bytes_per_sequence()

    for cut in (1, 3, 4, 5, 16, 17, 32):
        changed = ids.clone()
        changed[:, cut:] = (changed[:, cut:] + 9) % 257
        torch.testing.assert_close(model(changed).logits[:, :cut], full.logits[:, :cut], rtol=0, atol=0)

    # Streaming parity across tiny irregular chunks including T=1.
    outputs, state = [], None
    for size in (1, 2, 1, 5, 16, 3, 25):
        piece, ids = ids[:, :size], ids[:, size:]
        out = model(piece, state=state)
        outputs.append(out.logits)
        state = out.state
    torch.testing.assert_close(torch.cat(outputs, 1), full.logits, atol=5e-5, rtol=2e-4)
    for a, b in zip(state.levels, full.state.levels, strict=True):
        torch.testing.assert_close(a.memory, b.memory, atol=3e-5, rtol=2e-4)
        torch.testing.assert_close(a.tail, b.tail, atol=3e-5, rtol=2e-4)
    torch.testing.assert_close(state.assoc_memory, full.state.assoc_memory, atol=3e-5, rtol=2e-4)
    torch.testing.assert_close(state.local_tail, full.state.local_tail, atol=3e-5, rtol=2e-4)

    # Sequential vs chunked forward and backward gradient equivalence.
    seq_model = copy.deepcopy(model)
    for cell in seq_model.cells:
        cell.execution = "sequential"
    ids_ref = torch.randint(0, 257, (2, 41))
    out_chunk = model(ids_ref)
    out_seq = seq_model(ids_ref)
    torch.testing.assert_close(out_chunk.logits, out_seq.logits, atol=5e-5, rtol=2e-4)
    out_chunk.logits.square().mean().backward()
    out_seq.logits.square().mean().backward()
    for p_c, p_s in zip(model.parameters(), seq_model.parameters(), strict=True):
        assert p_c.grad is not None and torch.isfinite(p_c.grad).all()
        torch.testing.assert_close(p_c.grad, p_s.grad, atol=5e-5, rtol=1e-3)

    # Corrupted state rejection.
    bad = copy.copy(full.state)
    bad.levels = tuple(copy.copy(s) for s in full.state.levels)
    bad.levels[0].memory = bad.levels[0].memory.double()
    with pytest.raises(ValueError, match="state shape/dtype"):
        model(ids_ref[:, :1], state=bad)


def test_fractal_delta_eval_dropout_zero(tmp_path):
    cfg = _fractal_cfg(tmp_path, tmp_path / "drop", depth=3, shared=True, dropout=0.25)
    torch.manual_seed(23)
    model = build_backbone(cfg).eval()
    ids = torch.randint(0, 257, (2, 32))
    torch.testing.assert_close(model(ids).logits, model(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "window,value_residual,golden_phase,free_energy,weight_mode",
    [
        (0, False, False, False, "dense"),
        (8, False, False, False, "dense"),
        (0, True, False, False, "dense"),
        (0, True, True, False, "dense"),
        (0, True, False, True, "dense"),
        (0, True, False, False, "ternary"),
        (0, True, False, False, "quaternion"),
    ],
)
def test_fractal_delta_decode_compiled_parity(tmp_path, window, value_residual, golden_phase, free_energy, weight_mode):
    cfg = _fractal_cfg(
        tmp_path,
        tmp_path / "dec",
        depth=3,
        shared=True,
        window=window,
        value_residual=value_residual,
        golden_phase=golden_phase,
        free_energy=free_energy,
        weight_mode=weight_mode,
    )
    torch.manual_seed(31)
    model = build_backbone(cfg).eval()
    adapter = FractalDecode(model)
    traces = []

    def capture(graph, inputs):
        traces.append(graph)
        return graph.forward

    compiled = CompiledDecode(adapter, backend=capture)
    prefix = torch.randint(0, 256, (1, 19))
    cont = torch.randint(0, 256, (1, 12))
    parity = verify_logits(compiled, prefix, cont)
    assert parity["positions"] == 13 and parity["max_abs_logit_error"] < 1e-4
    _, sample_compiled = generate(compiled, prefix, 8)
    _, sample_eager = generate(adapter, prefix, 8)
    assert torch.equal(sample_compiled, sample_eager)
    assert len(traces) == 1


def test_fractal_delta_memorization_and_finite_gradients(tmp_path):
    data = fixture_data(tmp_path / "data")
    cfg = _fractal_cfg(data.root, tmp_path / "mem", depth=4, shared=True, dropout=0.0)
    with StoryTrainer(cfg, smoke=True) as trainer:
        assert estimate_research_flops(trainer.model, trainer.objective, cfg.data.seq_len) is not None
        batch = trainer.sampler.next("cpu")
        opt = torch.optim.AdamW(trainer.model.parameters(), lr=0.01)
        losses = []
        for _ in range(80):
            opt.zero_grad()
            out = trainer.objective(trainer.model, batch)
            out.loss.backward()
            assert torch.isfinite(out.loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.model.parameters())
            torch.nn.utils.clip_grad_norm_(trainer.parameters, 1.0)
            opt.step()
            losses.append(float(out.loss.detach()))
        assert losses[-1] < losses[0] * 0.35


@pytest.mark.parametrize(
    "golden_phase,free_energy,weight_mode",
    [
        (False, False, "dense"),
        (True, False, "dense"),
        (False, True, "dense"),
        (False, False, "ternary"),
        (False, False, "quaternion"),
    ],
)
def test_fractal_delta_exact_resume_dropout_and_accumulation(tmp_path, golden_phase, free_energy, weight_mode):
    data = fixture_data(tmp_path / "data")
    cfg = _fractal_cfg(
        data.root,
        tmp_path / "first",
        depth=3,
        shared=True,
        dropout=0.2,
        value_residual=True,
        golden_phase=golden_phase,
        free_energy=free_energy,
        weight_mode=weight_mode,
    )
    with StoryTrainer(cfg, smoke=True) as a:
        a.fit(until_step=2)
        checkpoint = a.last_checkpoint
        loss = a.train_step()["training_loss"]
        expected = copy.deepcopy(
            (a.model.state_dict(), a.optimizer.state_dict(), a.scheduler.state_dict(), a.sampler.state_dict())
        )
    cfg.train.output_dir = str(tmp_path / "resume")
    with StoryTrainer(cfg, smoke=True, resume=checkpoint) as b:
        assert b.train_step()["training_loss"] == loss
        assert_nested_equal(
            expected,
            (b.model.state_dict(), b.optimizer.state_dict(), b.scheduler.state_dict(), b.sampler.state_dict()),
        )


def test_fractal_delta_binding_recall_learning(tmp_path):
    import numpy as np
    from llm2.research.binding_data import generate as gen_binding, intervene
    from llm2.research.binding_worker import BindingTrainer

    cfg = _fractal_cfg(tmp_path, tmp_path / "bind", depth=4, shared=True, d_model=48, heads=2, hidden_dim=48, dropout=0.05)
    cfg.research.memory_dim = 32
    cfg.data.batch_size = 16
    cfg.data.seq_len = 32
    with BindingTrainer(cfg, smoke=True) as trainer:
        opt = torch.optim.AdamW(trainer.model.parameters(), lr=0.003, weight_decay=0.01)
        for _ in range(180):
            batch = trainer.sampler.next("cpu")
            opt.zero_grad()
            out = trainer.objective(trainer.model, batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters, 1.0)
            opt.step()
        trainer.model.eval()
        rng = np.random.default_rng(999)
        eb = gen_binding(rng, 32, 64, writes=4, queries=4, device="cpu")
        ib = intervene(eb)
        with torch.no_grad():
            p1 = trainer.model(eb["input_ids"]).logits.argmax(-1)
            p2 = trainer.model(ib["input_ids"]).logits.argmax(-1)
            mask = eb["targets"] >= 0
            c1 = p1[mask] == eb["targets"][mask]
            c2 = p2[mask] == ib["targets"][mask]
            acc = c1.float().mean().item()
            paired = (c1 & c2).float().mean().item()
        print(f"UNSEEN_ACC={acc:.4f} PAIRED_ACC={paired:.4f}")
        assert acc >= 0.90 and paired >= 0.85


def test_fractal_delta_golden_phase_high_offset_and_shared_init(tmp_path):
    cfg_pure = _fractal_cfg(tmp_path, tmp_path / "p", depth=4, value_residual=True, golden_phase=False)
    cfg_gold = _fractal_cfg(tmp_path, tmp_path / "g", depth=4, value_residual=True, golden_phase=True)
    torch.manual_seed(99)
    m_pure = build_backbone(cfg_pure).eval()
    torch.manual_seed(99)
    m_gold = build_backbone(cfg_gold).eval()
    sd_p = m_pure.state_dict()
    sd_g = m_gold.state_dict()
    extra_keys = set(sd_g) - set(sd_p)
    assert extra_keys == {f"adapters.{i}.phase_scale" for i in range(4)}
    assert set(sd_p).issubset(sd_g)
    for name in sd_p:
        assert torch.equal(sd_p[name], sd_g[name]), f"mismatch at {name}"
    assert m_gold.state_bytes_per_sequence() - m_pure.state_bytes_per_sequence() == 8

    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        s0 = m_gold(x[:, :4]).state
        assert s0.phase_pos.dtype == torch.int64 and int(s0.phase_pos.item()) == 4
        # Force past FP32 exact integer limit (2**24 = 16777216)
        high_pos = (1 << 25) + 3
        s_chunk_init = copy.deepcopy(s0)
        s_chunk_init.phase_pos.fill_(high_pos)
        s_step_init = copy.deepcopy(s0)
        s_step_init.phase_pos.fill_(high_pos)

        out_chunk = m_gold(x[:, 4:], state=s_chunk_init)
        assert int(out_chunk.state.phase_pos.item()) == high_pos + 6

        step_logits = []
        st = s_step_init
        for t in range(4, 10):
            o = m_gold(x[:, t : t + 1], state=st)
            st = o.state
            step_logits.append(o.logits)
        out_step = torch.cat(step_logits, dim=1)
        assert int(st.phase_pos.item()) == high_pos + 6
        assert torch.allclose(out_chunk.logits, out_step, atol=1e-5, rtol=1e-5)


def test_unified_trainer_causality_and_exact_resume(tmp_path):
    from llm2.research.ia2_benchmark import ARMS, UnifiedTrainer, make_story_config

    d = fixture_data(tmp_path / "stories")
    arm = next(a for a in ARMS if a.name == "fractal_golden_big")

    def _cfg(out_dir):
        cfg = make_story_config(arm, d.root, out_dir, seed=351, lr=0.003, smoke=True, device="cpu")
        cfg.model.dropout = 0.2
        cfg.data.batch_size = 4
        cfg.data.eval_batch_size = 2
        cfg.train.max_steps = 2
        cfg.train.checkpoint_interval = 1
        cfg.train.eval_interval = 10
        cfg.train.grad_accum_steps = 2
        cfg.train.deterministic = True
        return cfg

    with UnifiedTrainer(_cfg(tmp_path / "run_full"), smoke=True) as full:
        b0 = full.sampler.next("cpu")
        assert b0["input_ids"].shape == (4, 32)
        # First 3 rows are stories (dense targets), last row is binding (sparse targets with -100)
        assert (b0["targets"][:3] != -100).any()
        assert (b0["targets"][3] == -100).any() and (b0["targets"][3] != -100).any()
        # Reset sampler by constructing clean full run
    with UnifiedTrainer(_cfg(tmp_path / "full_clean"), smoke=True) as full:
        full.fit()
        full_state = {k: v.clone() for k, v in full.model.state_dict().items()}
        full_sched = copy.deepcopy(full.scheduler.state_dict())
        full_counters = dict(full.counters)

    cfg_part = _cfg(tmp_path / "part")
    cfg_part.train.max_steps = 1
    with UnifiedTrainer(cfg_part, smoke=True) as part:
        part.fit()
        ckpt1 = part.last_checkpoint
    with UnifiedTrainer(_cfg(tmp_path / "resumed"), smoke=True, resume=ckpt1) as res:
        res.fit()
        assert_nested_equal(full_state, res.model.state_dict())
        assert_nested_equal(full_sched, res.scheduler.state_dict())
        assert res.counters["step"] == full_counters["step"] == 2
        assert res.counters["tokens_processed"] == full_counters["tokens_processed"]


def test_local_transformer_decode_does_not_mutate_model(tmp_path):
    from llm2.model import TransformerLM
    from llm2.research.ia2_benchmark import _make_decode_adapter

    cfg = _fractal_cfg(tmp_path, tmp_path / "tr", depth=2, d_model=16, heads=2, hidden_dim=32)
    m = TransformerLM(cfg.model).eval()
    orig_types = [type(layer.attn.kernel) for layer in m.layers]
    dec = _make_decode_adapter(m, span=8)
    assert type(dec.model.layers[0].attn.kernel).__name__ == "WindowSDPA"
    assert [type(layer.attn.kernel) for layer in m.layers] == orig_types


def test_fractal_golden_classic_decode_and_bf16_autocast_invariants(tmp_path):
    cfg = _fractal_cfg(
        tmp_path,
        tmp_path / "classic",
        depth=6,
        shared=True,
        d_model=32,
        heads=4,
        hidden_dim=64,
        value_residual=True,
        golden_phase=True,
        leaky_budget=True,
        assoc_delta=True,
    )
    torch.manual_seed(2026)
    model = build_backbone(cfg).eval()
    assert model.parameter_formula() == sum(p.numel() for p in model.parameters())

    # Fast levels (0, 1, 2) have beta_bias == 0; slow levels (3, 4, 5) have negative beta_bias
    for lvl in range(3):
        assert torch.all(model.adapters[lvl].beta_bias == 0.0)
    for lvl in range(3, 6):
        assert torch.all(model.adapters[lvl].beta_bias < 0.0)

    ids = torch.randint(0, 256, (2, 24))
    decoder = FractalDecode(model)
    parity = verify_logits(decoder, ids[:, :16], ids[:, 16:])
    assert parity["max_abs_logit_error"] < 1e-4

    # Under BF16 autocast, state tensors must remain strictly float32 and valid for continuation
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out1 = model(ids[:, :12])
        assert out1.state.assoc_memory.dtype == torch.float32
        assert out1.state.local_tail.dtype == torch.float32
        for lvl_state in out1.state.levels:
            assert lvl_state.memory.dtype == torch.float32
            assert lvl_state.tail.dtype == torch.float32
        out2 = model(ids[:, 12:], state=out1.state)
        assert torch.isfinite(out2.logits.float()).all()


