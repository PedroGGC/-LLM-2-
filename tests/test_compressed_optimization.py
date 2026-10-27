"""The old compressed graph is the numerical control, not a new architecture."""

import copy

import pytest
import torch

from llm2.architectures.fractal_recursive import FractalLatentModel, fractal_parameter_formula
from llm2.config import ResearchConfig
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research.diagnostics import diagnose_model
from llm2.runtime import attention_context, capture_rng
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


MODES = ["compressed", "compressed_native", "compressed_grouped"]
CHECKPOINT_CASES = [(mode, weight, 2) for mode in MODES for weight in (0., .1)] + [("compressed_grouped", 0., 1)]


@pytest.mark.parametrize("mode", MODES[1:])
@pytest.mark.parametrize("iterations", [1, 2])
@attention_context(True)
def test_compressed_optimization_outputs_gradients_and_boundaries(tiny_config, mode, iterations):
    tiny_config.max_seq_len = 65
    tiny_config.dropout = 0.
    reference = FractalLatentModel(tiny_config, ResearchConfig(
        architecture="fractal", scale_execution="compressed", iterations=iterations))
    # Exercise learned nonuniform mixing, FiLM, gates and both nonzero priors.
    with torch.no_grad():
        reference.block.interaction.mixing.normal_(0, .3)
        reference.block.interaction.gate.normal_(-1, .3)
        reference.block.conditioning.film.normal_(0, .2)
        reference.block.comp4.prior.normal_(0, .2)
        reference.block.comp16.prior.normal_(0, .2)
    optimized = FractalLatentModel(tiny_config, ResearchConfig(
        architecture="fractal", scale_execution=mode, iterations=iterations))
    optimized.load_state_dict(reference.state_dict())
    assert count_parameters(reference) == count_parameters(optimized) == fractal_parameter_formula(tiny_config, optimized.research)
    assert optimized.lm_head.weight is optimized.embedding.weight
    for length in [1, 4, 15, 16, 17, 31, 32, 33, 65]:
        reference.zero_grad(set_to_none=True)
        optimized.zero_grad(set_to_none=True)
        ids = torch.randint(0, 32, (2, length))
        a, b = reference(ids), optimized(ids)
        torch.testing.assert_close(a.logits, b.logits, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(a.hidden_states, b.hidden_states, rtol=3e-5, atol=3e-6)
        a.logits.square().mean().backward()
        b.logits.square().mean().backward()
        for (name, x), (other, y) in zip(reference.named_parameters(), optimized.named_parameters()):
            assert name == other
            assert (x.grad is None) == (y.grad is None), name
            if x.grad is not None:
                assert torch.isfinite(y.grad).all(), name
                torch.testing.assert_close(x.grad, y.grad, rtol=3e-4, atol=2e-6, msg=name)
    optimized.eval()
    ids = torch.randint(0, 32, (2, 33))
    with torch.no_grad():
        original = optimized(ids).logits
        for cut in range(1, 33):
            changed = ids.clone()
            changed[:, cut:] = (changed[:, cut:] + 7) % 32
            torch.testing.assert_close(original[:, :cut], optimized(changed).logits[:, :cut], rtol=0, atol=0)
            torch.testing.assert_close(original[:, :cut], optimized(ids[:, :cut]).logits, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("mode", MODES)
@attention_context(True)
def test_compressed_execution_shapes_and_cost(tiny_config, mode):
    tiny_config.max_seq_len = 33
    model = FractalLatentModel(tiny_config, ResearchConfig(architecture="fractal", scale_execution=mode, iterations=2))
    core_shapes, projection_shapes = [], []
    core_hook = model.block.cores[0].register_forward_pre_hook(lambda m, args: core_shapes.append(args[0].shape[:2]))
    projection_hook = model.block.interaction.projection.register_forward_pre_hook(lambda m, args: projection_shapes.append(args[0].shape[1]))
    try:
        model(torch.zeros(2, 33, dtype=torch.long))
    finally:
        core_hook.remove()
        projection_hook.remove()
    lengths = (33, 9, 3)
    expected_cores = [(2, 33), (4, 9)] if mode == "compressed_grouped" else [(2, n) for n in lengths]
    assert core_shapes == expected_cores * 2
    assert projection_shapes == ([33, 33, 33] if mode == "compressed" else list(lengths)) * 2
    d, h, s = tiny_config.d_model, tiny_config.hidden_dim, 33
    core_lengths = (33, 9, 9) if mode == "compressed_grouped" else lengths
    core = sum(8*n*d*d + 6*n*d*h + 4*n*n*d + 2*n*d*d for n in core_lengths)
    projection = 2*d*d*(3*s if mode == "compressed" else sum(lengths))
    expected = 2*(core+projection)+2*s*d*tiny_config.vocab_size
    assert model.analytical_cost(s) == (expected, 0, 2)


def _state(trainer):
    # Wall-clock/memory high-water counters are observations, not deterministic.
    keys = ("step", "tokens_processed", "sequences_processed", "input_tokens_processed",
            "training_flops_estimate", "validation_flops_estimate", "curriculum_probe_flops_estimate")
    return copy.deepcopy((trainer.model.state_dict(), trainer.objective.state_dict(),
        trainer.optimizer.state_dict(), trainer.scheduler.state_dict(), trainer.sampler.state_dict(),
        capture_rng(), {key: trainer.counters[key] for key in keys}))


@pytest.mark.parametrize("mode,jepa_weight,iterations", CHECKPOINT_CASES)
def test_compressed_exact_resume_dropout_ema_accumulation(training_config, tmp_path, mode, jepa_weight, iterations):
    cfg = training_config
    cfg.model.max_seq_len = cfg.data.seq_len = 17
    cfg.train.accumulation_weighting = "tokens"
    cfg.research = ResearchConfig(architecture="fractal", scale_execution=mode,
        iterations=iterations, gradient_checkpointing=True, jepa_weight=jepa_weight)
    assert cfg.model.dropout > 0 and cfg.train.grad_accum_steps > 1
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2)
        checkpoint = trainer.last_checkpoint
        before = _state(trainer)
        metrics = trainer.train_step()
        expected = _state(trainer)
    torch.manual_seed(999)
    cfg.train.output_dir = str(tmp_path / f"resume-{mode}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert_nested_equal(before, _state(trainer))
        resumed_metrics = trainer.train_step()
        assert_nested_equal(expected, _state(trainer))
        for key in ("training_loss", "grad_norm", "lr", "tokens_processed", "training_flops_estimate"):
            assert metrics[key] == resumed_metrics[key]


@pytest.mark.parametrize("mode,iterations", [(mode, 2) for mode in MODES] + [("compressed_grouped", 1)])
def test_compressed_memorization(tiny_config, mode, iterations):
    model = FractalLatentModel(tiny_config, ResearchConfig(architecture="fractal", scale_execution=mode, iterations=iterations))
    ids = torch.arange(18).repeat(2, 1)
    batch = {"input_ids": ids[:, :-1], "targets": ids[:, 1:]}
    objective = CausalLanguageModeling()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    initial = objective(model, batch).loss.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert objective(model, batch).loss.item() < initial/15


@pytest.mark.parametrize("scales", [[1, 4], [1, 4, 16], [1, 2, 0]])
def test_compressed_rejects_ignored_scale_config(scales):
    with pytest.raises(ValueError, match="native block factors"):
        ResearchConfig(architecture="fractal", scale_execution="compressed_native", scales=scales)


def test_compressed_diagnostics_do_not_compare_unaligned_scales(tiny_config):
    model = FractalLatentModel(tiny_config, ResearchConfig(architecture="fractal", scale_execution="compressed_native"))
    result = diagnose_model(model, torch.randint(0, 32, (2, 17)))
    assert [value["samples"] for value in result["scales"]] == [34, 10, 4]
    assert all(result["linear_cka"][i][j] is None for i in range(3) for j in range(3) if i != j)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compressed_cuda_fp32_equivalence(tiny_config):
    tiny_config.max_seq_len = 33
    reference = FractalLatentModel(tiny_config, ResearchConfig(
        architecture="fractal", scale_execution="compressed", iterations=2)).cuda()
    with torch.no_grad():
        reference.block.interaction.mixing.normal_(0, .3)
        reference.block.conditioning.film.normal_(0, .2)
        reference.block.comp4.prior.normal_(0, .2)
        reference.block.comp16.prior.normal_(0, .2)
    ids = torch.randint(0, 32, (2, 33), device="cuda")
    a = reference(ids).logits
    a.square().mean().backward()
    for mode in MODES[1:]:
        optimized = FractalLatentModel(tiny_config, ResearchConfig(
            architecture="fractal", scale_execution=mode, iterations=2)).cuda()
        optimized.load_state_dict(reference.state_dict())
        b = optimized(ids).logits
        torch.testing.assert_close(a, b, rtol=3e-4, atol=2e-5)
        b.square().mean().backward()
        for (name, x), (_, y) in zip(reference.named_parameters(), optimized.named_parameters()):
            assert (x.grad is None) == (y.grad is None), name
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-3, atol=2e-5, msg=name)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode,jepa_weight,iterations", CHECKPOINT_CASES)
def test_compressed_cuda_fp32_save_continue(training_config, tmp_path, mode, jepa_weight, iterations):
    cfg = training_config
    cfg.model.max_seq_len = cfg.data.seq_len = 17
    cfg.train.device, cfg.train.precision, cfg.train.deterministic = "cuda", "fp32", False
    cfg.research = ResearchConfig(architecture="fractal", scale_execution=mode, iterations=iterations, jepa_weight=jepa_weight)
    with Trainer(cfg) as trainer:
        metrics = trainer.train_step()
        assert metrics["peak_vram_allocated_bytes"] > 0
        checkpoint = trainer.save()
    cfg.train.output_dir = str(tmp_path / f"cuda-{mode}")
    with Trainer(cfg, resume=checkpoint) as trainer:
        assert torch.isfinite(torch.tensor(trainer.train_step()["training_loss"]))
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
