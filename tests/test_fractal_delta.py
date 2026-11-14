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


def _fractal_cfg(data_root, output_dir, *, depth=4, shared=True, d_model=32, heads=4, hidden_dim=64, dropout=0.0, device="cpu"):
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
        ),
    )


@pytest.mark.parametrize("shared,depth", [(True, 4), (False, 2), (True, 1)])
def test_fractal_delta_causality_streaming_chunked_vs_seq_and_params(tmp_path, shared, depth):
    cfg = _fractal_cfg(tmp_path, tmp_path / "run", depth=depth, shared=shared, dropout=0.0)
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


def test_fractal_delta_decode_compiled_parity(tmp_path):
    cfg = _fractal_cfg(tmp_path, tmp_path / "dec", depth=3, shared=True)
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


def test_fractal_delta_exact_resume_dropout_and_accumulation(tmp_path):
    data = fixture_data(tmp_path / "data")
    cfg = _fractal_cfg(data.root, tmp_path / "first", depth=3, shared=True, dropout=0.2)
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

    cfg = _fractal_cfg(tmp_path, tmp_path / "bind", depth=4, shared=True, d_model=48, heads=4, hidden_dim=48, dropout=0.05)
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

