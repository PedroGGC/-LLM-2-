"""Unit & CUDA tests for Fase B (Lumina V1 Multimodal Bridge):
- VisionSpatialProjector 2x2 / 4x4 pixel-unshuffle + 2D positional embedding + odd-grid NaFlex padding
- Strict 3-stage freezing (Stage 1: backbone bit-for-bit untouched; Stage 2: Prelude+adapters only)
- Multi-image & 8-frame GIF O(1) constant state bytes + <img_end> GRM anchors surviving 512 text tokens
- Generic JSONL + weighted stream mixer + refusal filtering
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from llm2.architectures.fractal_delta import FractalDecode
from llm2.config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from llm2.multimodal import (
    LuminaMultimodalWrapper,
    MockVisionEncoder,
    MultimodalSample,
    VLMStageConfig,
    VisionSpatialProjector,
    apply_vlm_stage_freeze,
    iter_jsonl_multimodal,
    mix_multimodal_streams,
    train_vlm_stage,
)
from llm2.research.decode_optimization import SegmentMemoryCache
from llm2.research.factory import build_backbone


class _ByteTok:
    eos_id = 0

    @staticmethod
    def encode(text: str) -> list[int]:
        return [min(255, b) for b in text.encode("utf-8")]


def _make_backbone(device: str = "cpu", d_model: int = 32, n_layers: int = 8) -> torch.nn.Module:
    cfg = ExperimentConfig(
        model=ModelConfig(
            vocab_size=256,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
            hidden_dim=d_model * 2,
            max_seq_len=1024,
            dropout=0.0,
        ),
        data=DataConfig(seq_len=64, batch_size=1, eval_batch_size=1),
        optimizer=OptimizerConfig(kind="adamw", lr=1e-3, warmup_steps=1, schedule_steps=10),
        train=TrainConfig(
            seed=42,
            device=device,
            precision="bf16" if device == "cuda" else "fp32",
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
            fractal_decoupled_gates=True,
            fractal_window=8,
            fractal_hybrid_stride=4,
            fractal_exit_gate=True,
            memory_dim=16,
            fractal_assoc_val_dim=8,
            state_chunk_size=8,
        ),
    )
    return build_backbone(cfg)


def _state_nbytes(state) -> int:
    total = 0
    for lvl in state.levels:
        total += lvl.memory.numel() * lvl.memory.element_size()
        total += lvl.tail.numel() * lvl.tail.element_size()
    total += state.assoc_memory.numel() * state.assoc_memory.element_size()
    total += state.local_tail.numel() * state.local_tail.element_size()
    if state.hybrid_windows is not None:
        for hw in state.hybrid_windows:
            total += hw.k_tail.numel() * hw.k_tail.element_size()
            total += hw.v_tail.numel() * hw.v_tail.element_size()
            total += hw.valid_tail.numel() * hw.valid_tail.element_size()
    if state.phase_pos is not None:
        total += state.phase_pos.numel() * state.phase_pos.element_size()
    return total


def test_spatial_projector_2x2_4x4_and_odd_grid_naflex():
    torch.manual_seed(11)
    p2 = VisionSpatialProjector(vision_dim=16, d_model=32, downsample_factor=2)
    p4 = VisionSpatialProjector(vision_dim=16, d_model=32, downsample_factor=4)

    # Exact even grid 8x8 = 64 patches -> 16 tokens (2x2) and 4 tokens (4x4)
    grid_even = torch.randn(2, 8, 8, 16, requires_grad=True)
    y2 = p2(grid_even)
    y4 = p4(grid_even)
    assert y2.shape == (2, 16, 32)
    assert y4.shape == (2, 4, 32)
    (y2.sum() + y4.sum()).backward()
    assert grid_even.grad is not None and torch.isfinite(grid_even.grad).all()

    # Odd SigLIP-384 style grid 27x27 -> pads to 28x28 -> 14x14=196 (2x2) and 7x7=49 (4x4)
    grid_odd = torch.randn(1, 27, 27, 16)
    assert p2(grid_odd).shape == (1, 196, 32)
    assert p4(grid_odd).shape == (1, 49, 32)


def test_multimodal_target_masking_and_stage1_stage2_freeze_invariants():
    torch.manual_seed(22)
    backbone = _make_backbone("cpu")
    vision = MockVisionEncoder(vision_dim=16, grid_size=4, patch_size=16)
    wrapper = LuminaMultimodalWrapper(backbone, vision, downsample_factor=2)

    img = torch.randn(1, 3, 64, 64)  # 4x4 patches -> 2x2=4 visual tokens + 2 markers = 6 tokens
    prompt_ids = torch.tensor([[1, 10, 11, 12]])
    resp_ids = torch.tensor([[20, 21, 22]])
    embeds, targets, spans = wrapper.build_multimodal_inputs(images=[img], prompt_ids=prompt_ids, response_ids=resp_ids)
    assert spans == [(0, 6)]
    # Total length before shift = 6 + 4 + 3 = 13 -> shifted inputs = 12
    assert embeds.shape == (1, 12, 32)
    assert targets.shape == (1, 12)
    # First 9 positions (6 visual + 3 prompt) must be masked -100; last 3 must equal resp_ids
    assert (targets[0, :9] == -100).all()
    assert torch.equal(targets[0, 9:], resp_ids[0])

    # Stage 1: backbone & vision encoder must remain BIT-FOR-BIT unchanged while loss decreases!
    backbone_before = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
    vision_before = {k: v.detach().clone() for k, v in vision.state_dict().items()}
    proj_before = wrapper.projector.fc2.weight.detach().clone()

    pil_img = Image.new("RGB", (64, 64), color=(180, 40, 90))
    samples = [MultimodalSample(images=[pil_img], prompt="What color?", response="Red square.", source="synth")] * 12

    def _pre(imgs):
        return torch.stack([torch.full((3, 64, 64), 0.7) for _ in imgs])

    rep1 = train_vlm_stage(
        wrapper,
        samples,
        _ByteTok(),
        _pre,
        VLMStageConfig(stage=1, lr=5e-3, max_steps=12, precision="fp32"),
        torch.device("cpu"),
    )
    assert rep1["steps_completed"] == 12
    assert rep1["final_loss"] < rep1["initial_loss"]
    for k, v in backbone.state_dict().items():
        assert torch.equal(v, backbone_before[k]), f"Stage 1 mutated frozen backbone param {k}"
    for k, v in vision.state_dict().items():
        assert torch.equal(v, vision_before[k]), f"Stage 1 mutated frozen vision param {k}"
    assert not torch.equal(wrapper.projector.fc2.weight, proj_before)

    # Stage 2: Prelude (cells[0]) + adapters unfreeze, Core (cells[1]) & Coda (cells[2]) stay BIT-FOR-BIT frozen!
    core_before = {k: v.detach().clone() for k, v in backbone.cells[1].state_dict().items()}
    coda_before = {k: v.detach().clone() for k, v in backbone.cells[2].state_dict().items()}
    prelude_before = backbone.cells[0].qkv.weight.detach().clone()

    rep2 = train_vlm_stage(
        wrapper,
        samples[:4],
        _ByteTok(),
        _pre,
        VLMStageConfig(stage=2, lr=2e-3, max_steps=4, precision="fp32"),
        torch.device("cpu"),
    )
    assert rep2["steps_completed"] == 4
    assert not torch.equal(backbone.cells[0].qkv.weight, prelude_before)
    for k, v in backbone.cells[1].state_dict().items():
        assert torch.equal(v, core_before[k]), f"Stage 2 mutated frozen Core param {k}"
    for k, v in backbone.cells[2].state_dict().items():
        assert torch.equal(v, coda_before[k]), f"Stage 2 mutated frozen Coda param {k}"


def test_multimodal_8frame_gif_o1_state_and_grm_anchors_after_512_tokens():
    torch.manual_seed(33)
    backbone = _make_backbone("cpu").eval()
    vision = MockVisionEncoder(vision_dim=16, grid_size=4, patch_size=16).eval()
    wrapper = LuminaMultimodalWrapper(backbone, vision, downsample_factor=2).eval()

    prompt_ids = torch.tensor([[1, 5, 6, 7]])
    img_1 = [torch.randn(1, 3, 64, 64)]
    frames_8 = [torch.full((1, 3, 64, 64), float(i - 4) * 0.5) + 0.05 * torch.randn(1, 3, 64, 64) for i in range(8)]

    cache_1 = SegmentMemoryCache(max_entries=4, max_anchors=16)
    _, _, st_1 = wrapper.prefill_multimodal(images=img_1, prompt_ids=prompt_ids, grm_cache=cache_1, image_tags=["single"])

    cache_8 = SegmentMemoryCache(max_entries=4, max_anchors=16)
    logits, _, st_8 = wrapper.prefill_multimodal(
        images=frames_8,
        prompt_ids=prompt_ids,
        grm_cache=cache_8,
        image_tags=[f"gif_f{i}" for i in range(8)],
    )

    # Strict O(1) constant VRAM state: 1 image and 8 GIF frames occupy IDENTICAL state bytes!
    assert _state_nbytes(st_1) == backbone.state_bytes_per_sequence()
    assert _state_nbytes(st_8) == backbone.state_bytes_per_sequence()
    assert len(cache_8._grm_segments) == 8

    # Generate 512 tokens and confirm <img_end> GRM anchors remain intact & queryable!
    dec = FractalDecode(backbone)
    cur_state = st_8
    nxt = int(logits[0, -1].argmax().item())
    for _ in range(512):
        step_logits, _, cur_state = dec.step_hidden(torch.tensor([[nxt]]), cur_state)
        nxt = int(step_logits[0, -1].argmax().item())

    assert _state_nbytes(cur_state) == backbone.state_bytes_per_sequence()
    assert len(cache_8._grm_segments) == 8

    # Query using frame 0's exact stored summary key -> must retrieve gif_f0 as Top-1!
    _, f0_key, _ = cache_8._grm_segments[0]
    readout, info = cache_8.query_grm_readout(f0_key.unsqueeze(1), model=backbone, top_k=2, temperature=0.1)
    assert info["selected_tags"][0] == "gif_f0"
    assert torch.isfinite(readout).all()


def test_jsonl_loader_refusal_filter_and_weighted_mixer(tmp_path: Path):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(img_a)
    Image.new("RGB", (32, 32), (0, 0, 255)).save(img_b)

    jsonl = tmp_path / "dataset.jsonl"
    rows = [
        {"image": "a.png", "messages": [{"role": "user", "content": "Describe"}, {"role": "assistant", "content": "A bright red square."}], "rating": "general"},
        {"image": "b.png", "prompt": "Describe", "response": "I'm sorry, I cannot assist with that request."},
        {"images": ["a.png", "b.png"], "caption_long": "Two panels showing red and blue squares.", "rating": "explicit"},
        {"id": "pose::a.png", "task": "pose_breakdown", "image": "a.png", "system": "Analyze pose.", "user": "Examine target:", "assistant": "Standing upright, bright red canvas."},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    loaded = list(iter_jsonl_multimodal(jsonl, filter_refusals=True))
    assert len(loaded) == 3
    assert loaded[0].response == "A bright red square."
    assert len(loaded[1].images) == 2 and loaded[1].rating == "explicit"
    assert "Analyze pose." in loaded[2].prompt and loaded[2].response == "Standing upright, bright red canvas."

    # Weighted mixer (3:1 ratio between stream A and stream B)
    s_gen = iter([MultimodalSample([], "p", f"gen_{i}", "finevision") for i in range(6)])
    s_art = iter([MultimodalSample([], "p", f"art_{i}", "curated_danbooru") for i in range(6)])
    mixed = list(mix_multimodal_streams([s_gen, s_art], weights=[0.75, 0.25], max_samples=8))
    sources = [m.source for m in mixed]
    assert sources.count("finevision") == 6
    assert sources.count("curated_danbooru") == 2


@pytest.mark.cuda
def test_multimodal_cuda_bf16_stage1_and_grm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(44)
    backbone = _make_backbone("cuda", d_model=64, n_layers=8).to(device)
    vision = MockVisionEncoder(vision_dim=32, grid_size=4, patch_size=16).to(device)
    wrapper = LuminaMultimodalWrapper(backbone, vision, downsample_factor=2).to(device)

    pil_img = Image.new("RGB", (64, 64), color=(120, 200, 60))
    samples = [MultimodalSample(images=[pil_img], prompt="Describe", response="Green frame.", source="cuda_synth")] * 6

    def _pre(imgs):
        return torch.stack([torch.full((3, 64, 64), 0.5) for _ in imgs])

    rep = train_vlm_stage(
        wrapper,
        samples,
        _ByteTok(),
        _pre,
        VLMStageConfig(stage=1, lr=5e-3, max_steps=6, precision="bf16"),
        device,
    )
    assert rep["steps_completed"] == 6
    assert rep["final_loss"] is not None and rep["final_loss"] < rep["initial_loss"]

    wrapper.eval()
    cache = SegmentMemoryCache(max_entries=4, max_anchors=8)
    frames = [torch.randn(1, 3, 64, 64, device=device) for _ in range(4)]
    _, _, st = wrapper.prefill_multimodal(images=frames, prompt_ids=torch.tensor([[1, 2, 3]], device=device), grm_cache=cache)
    assert _state_nbytes(st) == backbone.state_bytes_per_sequence()
    assert len(cache._grm_segments) == 4
