"""Unit & CUDA tests for Fase C:
- Vectorized O(T*W) unfold CausalLocalWindow exactness vs 1-token step & strict causality
- CivitAI prompt sanitizer + 4-part Lumina V1 mixture text-only SFT through LuminaMultimodalWrapper
- Rota 1 RADLADS/LoopUS distillation (SVD embedding warmstart, KL+CE+hidden+exit_gate+MTP, safetensors tied reload)
- O Ringue scoreboard metrics & O(1) vs O(L) 16K state comparison
"""

from __future__ import annotations

from pathlib import Path

import pytest
from safetensors.torch import load_model
import torch
from torch import nn

from llm2.architectures.fractal_delta import CausalLocalWindow
from llm2.config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from llm2.multimodal import (
    DEFAULT_TEXT_SFT_PRESETS,
    DEFAULT_VISION_PRESETS,
    LuminaMultimodalWrapper,
    MockVisionEncoder,
    MultimodalSample,
    VLMStageConfig,
    clean_civitai_prompt,
    train_vlm_stage,
)
from llm2.research.distill_convert import (
    DistillConfig,
    apply_surgery_freeze,
    run_distill_conversion,
    transplant_teacher_weights,
    warmstart_student_embeddings,
)
from llm2.research.factory import build_backbone
from llm2.research.ringue_eval import RingueEntry, _estimate_kv_cache_mib_16k, compute_composite_score, format_ringue_markdown


def _make_teacher_cfg(device: str = "cpu", d_model: int = 64, vocab_size: int = 128) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=3, n_heads=4, hidden_dim=d_model * 2, max_seq_len=128, dropout=0.0
        ),
        data=DataConfig(seq_len=32, batch_size=2, eval_batch_size=2),
        optimizer=OptimizerConfig(kind="adamw", lr=1e-3, warmup_steps=1, schedule_steps=10),
        train=TrainConfig(seed=7, device=device, precision="bf16" if device == "cuda" else "fp32", deterministic=True, max_steps=10, eval_interval=10, checkpoint_interval=10, log_interval=5),
        research=ResearchConfig(architecture="transformer", scale_conditioning=False),
    )


def _make_student_cfg(device: str = "cpu", d_model: int = 32, vocab_size: int = 128) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=8, n_heads=4, hidden_dim=d_model * 2, max_seq_len=128, dropout=0.0
        ),
        data=DataConfig(seq_len=32, batch_size=2, eval_batch_size=2),
        optimizer=OptimizerConfig(kind="adamw", lr=1e-3, warmup_steps=1, schedule_steps=10),
        train=TrainConfig(seed=7, device=device, precision="bf16" if device == "cuda" else "fp32", deterministic=True, max_steps=10, eval_interval=10, checkpoint_interval=10, log_interval=5),
        research=ResearchConfig(
            architecture="fractal_delta",
            share_scales=True,
            scale_conditioning=False,
            fractal_prelude_coda=True,
            fractal_value_residual=True,
            fractal_golden_phase=True,
            fractal_assoc_delta=True,
            fractal_assoc_multilag=True,
            fractal_decoupled_gates=True,
            fractal_window=8,
            fractal_hybrid_stride=4,
            fractal_exit_gate=True,
            exit_gate_weight=0.10,
            memory_dim=16,
            fractal_assoc_val_dim=8,
            state_chunk_size=8,
        ),
    )


def test_vectorized_unfold_local_window_exact_parity_and_causality():
    torch.manual_seed(11)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=4, n_heads=4, hidden_dim=64, max_seq_len=128, dropout=0.0)
    win = CausalLocalWindow(cfg, window=8).eval()
    x = torch.randn(2, 23, 32)

    y_full, st_full = win(x)
    # Compare against incremental 1-token step-by-step
    st = None
    step_outs = []
    for t in range(x.shape[1]):
        yt, st = win(x[:, t : t + 1], st)
        step_outs.append(yt)
    y_step = torch.cat(step_outs, dim=1)
    assert (y_full - y_step).abs().max().item() < 1e-5
    assert (st_full.k_tail - st.k_tail).abs().max().item() < 1e-5
    assert (st_full.v_tail - st.v_tail).abs().max().item() < 1e-5

    # Strict causality check
    x_pert = x.clone()
    x_pert[:, 15] += 3.0
    y_pert, _ = win(x_pert)
    assert torch.allclose(y_full[:, :15], y_pert[:, :15], atol=1e-6, rtol=0.0)
    assert not torch.allclose(y_full[:, 15:], y_pert[:, 15:], atol=1e-6, rtol=0.0)


def test_civitai_sanitizer_and_text_sft_through_multimodal_wrapper():
    raw = "score_9, score_8_up, (masterpiece:1.4), <lora:detailer_v2:0.8>, (red silk dress:1.25), warm sunset lighting, warm sunset lighting, 8k, BREAK standing on balcony"
    cleaned = clean_civitai_prompt(raw)
    assert "lora" not in cleaned.lower()
    assert "masterpiece" not in cleaned.lower()
    assert "score_9" not in cleaned.lower()
    assert "red silk dress" in cleaned
    assert "standing on balcony" in cleaned
    assert cleaned.count("warm sunset lighting") == 1
    assert "civitai_nsfw_3d" in DEFAULT_VISION_PRESETS
    assert "booru_to_nl" in DEFAULT_TEXT_SFT_PRESETS and "uncensor_ptbr" in DEFAULT_TEXT_SFT_PRESETS

    # Text-only SFT sample (images=[]) flows cleanly through LuminaMultimodalWrapper
    torch.manual_seed(22)
    student = build_backbone(_make_student_cfg("cpu", d_model=32, vocab_size=256))
    wrapper = LuminaMultimodalWrapper(student, MockVisionEncoder(vision_dim=16, grid_size=4, patch_size=16), downsample_factor=2)

    class _Tok:
        eos_id = 0

        @staticmethod
        def encode(s):
            return [min(255, b) for b in s.encode("utf-8")]

    text_samples = [
        MultimodalSample(images=[], prompt="Translate tags: 1girl, red_dress", response="A woman wearing a red dress.", source="booru_to_nl")
    ] * 4
    rep = train_vlm_stage(
        wrapper, text_samples, _Tok(), lambda imgs: torch.empty(0), VLMStageConfig(stage=2, lr=2e-3, max_steps=4, precision="fp32"), torch.device("cpu")
    )
    assert rep["steps_completed"] == 4 and rep["final_loss"] < rep["initial_loss"]


def test_rota1_distillation_svd_warmstart_and_safetensors_reload(tmp_path: Path):
    torch.manual_seed(33)
    teacher = build_backbone(_make_teacher_cfg("cpu", d_model=64, vocab_size=128)).eval()
    student = build_backbone(_make_student_cfg("cpu", d_model=32, vocab_size=128))
    mtp_heads = nn.ModuleList([nn.Linear(32, 128, bias=False) for _ in range(2)])

    t_snap = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    warm = warmstart_student_embeddings(teacher, student)
    assert warm["warmstarted"] is True and warm["mode"] == "svd_lowrank"
    assert student.embedding.weight.data_ptr() == student.lm_head.weight.data_ptr()

    bx = torch.randint(0, 128, (2, 24))
    by = torch.roll(bx, shifts=-1, dims=1)
    by[:, -1] = -100
    batches = [(bx, by)] * 16

    ckpt_path = tmp_path / "lumina_distilled.safetensors"
    rep = run_distill_conversion(
        teacher,
        student,
        batches,
        DistillConfig(max_steps=12, lr=4e-3, warmup_steps=2, grad_accum_steps=1, precision="fp32"),
        torch.device("cpu"),
        mtp_heads=mtp_heads,
        warmstart_embed=False,
        save_path=ckpt_path,
        log_interval=0,
    )
    assert rep["steps_completed"] == 12
    assert rep["final_metrics"]["total_loss"] < rep["initial_metrics"]["total_loss"]
    assert "kl_loss" in rep["final_metrics"] and "hidden_mse" in rep["final_metrics"]
    assert "exit_gate_loss" in rep["final_metrics"] and "mtp_loss" in rep["final_metrics"]
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, t_snap[k]), f"Teacher param mutated: {k}"

    # Verify safetensors reload preserves tied embedding/lm_head and bit-for-bit logits
    reloaded = build_backbone(_make_student_cfg("cpu", d_model=32, vocab_size=128)).eval()
    load_model(reloaded, str(ckpt_path))
    assert reloaded.embedding.weight.data_ptr() == reloaded.lm_head.weight.data_ptr()
    student.eval()
    with torch.no_grad():
        assert torch.allclose(student(bx).logits, reloaded(bx).logits, atol=1e-6)


def test_attention_surgery_transplant_and_ffn_freeze():
    torch.manual_seed(77)
    teacher = build_backbone(_make_teacher_cfg("cpu", d_model=32, vocab_size=128)).eval()
    s_cfg = _make_student_cfg("cpu", d_model=32, vocab_size=128)
    s_cfg.model.n_layers = 3
    s_cfg.research.share_scales = False
    s_cfg.research.fractal_prelude_coda = False
    student = build_backbone(s_cfg)

    info = transplant_teacher_weights(teacher, student)
    assert info["transplanted"] is True
    assert info["ffn_cells_copied"] == 3 and info["qkv_cells_copied"] == 3
    # Verify exact bit-for-bit FFN & embedding transplant
    assert torch.equal(student.embedding.weight, teacher.embedding.weight)
    assert torch.equal(student.cells[0].ffn.gate.weight, teacher.layers[0].ffn.gate.weight)
    assert torch.equal(student.cells[2].ffn.down.weight, teacher.layers[2].ffn.down.weight)

    emb_before = student.embedding.weight.detach().clone()
    ffn_before = student.cells[0].ffn.gate.weight.detach().clone()
    beta_before = student.cells[0].beta.weight.detach().clone()

    bx = torch.randint(0, 128, (2, 16))
    by = torch.roll(bx, shifts=-1, dims=1)
    by[:, -1] = -100
    rep = run_distill_conversion(
        teacher,
        student,
        [(bx, by)] * 8,
        DistillConfig(
            max_steps=8,
            lr=4e-3,
            warmup_steps=1,
            precision="fp32",
            surgery_transplant=False,
            freeze_ffn=True,
            freeze_embeddings=True,
            layer_mse_weight=1.0,
        ),
        torch.device("cpu"),
        warmstart_embed=False,
        log_interval=0,
    )
    assert rep["steps_completed"] == 8
    assert "layer_mse" in rep["final_metrics"]
    assert rep["final_metrics"]["total_loss"] < rep["initial_metrics"]["total_loss"]
    # Transplanted embeddings and SwiGLU FFNs must stay 100% frozen while Gated DeltaNet-2 gates adapt
    assert torch.equal(student.embedding.weight, emb_before)
    assert torch.equal(student.cells[0].ffn.gate.weight, ffn_before)
    assert not torch.equal(student.cells[0].beta.weight, beta_before)


def test_ringue_leaderboard_and_o1_vs_ol_16k_state():
    teacher = build_backbone(_make_teacher_cfg("cpu", d_model=64, vocab_size=128))
    student = build_backbone(_make_student_cfg("cpu", d_model=32, vocab_size=128))
    kv_t = _estimate_kv_cache_mib_16k(teacher)
    st_s = _estimate_kv_cache_mib_16k(student)
    assert kv_t is not None and st_s is not None and st_s < kv_t

    score = compute_composite_score(lambada_acc=0.50, arc_easy_acc_norm=0.60, hellaswag_acc_norm=0.40)
    assert score == 50.0
    md = format_ringue_markdown(
        [
            RingueEntry(
                name="Lumina-V1-Proto",
                kind="lumina_v1",
                unique_params=100_000_000,
                quantization="quarot_int4",
                peak_vram_gib=0.8,
                context_state_mib_16k=1.0,
                decode_tok_per_s=650.0,
                mj_per_token=65.0,
                watts_mean=50.0,
                lambada_acc=50.0,
                arc_easy_acc_norm=60.0,
                hellaswag_acc_norm=40.0,
                composite_score=50.0,
                score_per_gib=62.5,
            )
        ]
    )
    assert "Lumina-V1-Proto" in md and "Score/GiB" in md and "62.5" in md


@pytest.mark.cuda
def test_phase_c_cuda_bf16_distill_and_compiled_unfold_window():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(44)
    use_bf16 = bool(torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    prec = "bf16" if use_bf16 else "fp32"
    t_cfg = _make_teacher_cfg("cuda", d_model=64, vocab_size=128)
    s_cfg = _make_student_cfg("cuda", d_model=64, vocab_size=128)
    t_cfg.train.precision = prec
    s_cfg.train.precision = prec
    teacher = build_backbone(t_cfg).to(device).eval()
    student = build_backbone(s_cfg).to(device)
    compiled_student = torch.compile(student, backend="inductor", dynamic=False)

    bx = torch.randint(0, 128, (2, 64), device=device)
    by = torch.roll(bx, shifts=-1, dims=1)
    by[:, -1] = -100

    # Verify compiled vectorized unfold window works cleanly in mixed precision forward+backward
    with torch.autocast("cuda", dtype=amp_dtype):
        out = compiled_student(bx)
    assert torch.isfinite(out.logits).all()
    out.logits.float().sum().backward()

    # Verify Rota 1 distillation drops loss on CUDA
    rep = run_distill_conversion(
        teacher,
        student,
        [(bx, by)] * 6,
        DistillConfig(max_steps=6, lr=3e-3, warmup_steps=1, precision=prec),
        device,
        log_interval=0,
    )
    assert rep["steps_completed"] == 6 and rep["final_metrics"]["total_loss"] < rep["initial_metrics"]["total_loss"]
