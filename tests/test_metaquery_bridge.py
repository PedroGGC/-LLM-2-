"""Unit & CUDA tests for Fase D (MetaQueryBridge for Nano Lumina image & 3D GIF generation):
- Static image (B, Q, dit_cond_dim) & multi-frame GIF (B, F, Q, dit_cond_dim) conditioning shapes
- Strict O(1) constant state bytes across 1 vs 4 reference images + GRM visual anchor readout
- Strict backbone & vision wrapper freeze invariant during MetaQuery alignment training
"""

from __future__ import annotations

import pytest
import torch

from llm2.config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, ResearchConfig, TrainConfig
from llm2.multimodal import (
    LuminaMultimodalWrapper,
    MetaQueryBridge,
    MockVisionEncoder,
    compute_metaquery_alignment_loss,
)
from llm2.research.decode_optimization import SegmentMemoryCache
from llm2.research.factory import build_backbone


def _make_v2_backbone(device: str = "cpu", d_model: int = 32) -> torch.nn.Module:
    cfg = ExperimentConfig(
        model=ModelConfig(
            vocab_size=128,
            d_model=d_model,
            n_layers=8,
            n_heads=4,
            hidden_dim=d_model * 2,
            max_seq_len=512,
            dropout=0.0,
        ),
        data=DataConfig(seq_len=32, batch_size=1, eval_batch_size=1),
        optimizer=OptimizerConfig(kind="adamw", lr=1e-3, warmup_steps=1, schedule_steps=10),
        train=TrainConfig(
            seed=42,
            device=device,
            precision="bf16" if device == "cuda" else "fp32",
            deterministic=True,
            max_steps=10,
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
    tot = sum(l.memory.numel() * l.memory.element_size() + l.tail.numel() * l.tail.element_size() for l in state.levels)
    tot += state.assoc_memory.numel() * state.assoc_memory.element_size()
    tot += state.local_tail.numel() * state.local_tail.element_size()
    if state.hybrid_windows:
        tot += sum(
            hw.k_tail.numel() * hw.k_tail.element_size()
            + hw.v_tail.numel() * hw.v_tail.element_size()
            + hw.valid_tail.numel() * hw.valid_tail.element_size()
            for hw in state.hybrid_windows
        )
    if state.phase_pos is not None:
        tot += state.phase_pos.numel() * state.phase_pos.element_size()
    return tot


def test_metaquery_static_image_and_3d_gif_conditioning_and_o1_state():
    torch.manual_seed(101)
    backbone = _make_v2_backbone("cpu", d_model=32).eval()
    vision = MockVisionEncoder(vision_dim=16, grid_size=4, patch_size=16).eval()
    wrapper = LuminaMultimodalWrapper(backbone, vision, downsample_factor=2).eval()
    bridge = MetaQueryBridge(backbone, num_queries=16, dit_cond_dim=128, dit_pooled_dim=64, max_frames=16).eval()

    prompt_ids = torch.tensor([[1, 7, 8, 9]])
    img_1 = [torch.randn(1, 3, 64, 64)]
    imgs_4 = [torch.randn(1, 3, 64, 64) for _ in range(4)]

    cache_1 = SegmentMemoryCache(max_entries=4, max_anchors=8)
    out_img = bridge(prompt_ids=prompt_ids, images=img_1, wrapper=wrapper, grm_cache=cache_1, num_frames=1)
    assert out_img.seq_cond.shape == (1, 16, 128)
    assert out_img.pooled_cond.shape == (1, 64)
    assert torch.isfinite(out_img.seq_cond).all() and torch.isfinite(out_img.pooled_cond).all()

    cache_4 = SegmentMemoryCache(max_entries=4, max_anchors=8)
    out_gif = bridge(prompt_ids=prompt_ids, images=imgs_4, wrapper=wrapper, grm_cache=cache_4, num_frames=8)
    assert out_gif.seq_cond.shape == (1, 8, 16, 128)
    assert out_gif.pooled_cond.shape == (1, 64)
    assert not torch.allclose(out_gif.seq_cond[:, 0], out_gif.seq_cond[:, 5])

    # Strict O(1) state size invariant regardless of number of reference images or output GIF frames
    assert _state_nbytes(out_img.state) == backbone.state_bytes_per_sequence()
    assert _state_nbytes(out_gif.state) == backbone.state_bytes_per_sequence()
    assert len(cache_4._grm_segments) == 4


def test_metaquery_strict_backbone_freeze_and_alignment_training():
    torch.manual_seed(202)
    backbone = _make_v2_backbone("cpu", d_model=32)
    vision = MockVisionEncoder(vision_dim=16, grid_size=4, patch_size=16)
    wrapper = LuminaMultimodalWrapper(backbone, vision, downsample_factor=2)
    bridge = MetaQueryBridge(backbone, num_queries=8, dit_cond_dim=48, dit_pooled_dim=24)

    freeze_info = bridge.freeze_backbone_for_metaquery(wrapper)
    assert freeze_info["trainable_params"] > 0 and freeze_info["frozen_params"] > 0

    backbone_snap = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
    proj_snap = {k: v.detach().clone() for k, v in wrapper.projector.state_dict().items()}
    mq_before = bridge.meta_queries.detach().clone()

    opt = torch.optim.AdamW([p for p in bridge.parameters() if p.requires_grad], lr=1e-2)
    prompt_ids = torch.tensor([[2, 3, 4, 5]])
    ref_img = [torch.full((1, 3, 64, 64), 0.4)]
    target_seq = torch.randn(1, 8, 48)
    target_pool = torch.randn(1, 24)

    losses = []
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        cache = SegmentMemoryCache(max_entries=4, max_anchors=4)
        out = bridge(prompt_ids=prompt_ids, images=ref_img, wrapper=wrapper, grm_cache=cache, num_frames=1)
        loss, _ = compute_metaquery_alignment_loss(out, target_seq, target_pool)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))

    assert losses[-1] < losses[0]
    assert not torch.equal(bridge.meta_queries, mq_before)
    for k, v in backbone.state_dict().items():
        assert torch.equal(v, backbone_snap[k]), f"Backbone mutated during MetaQuery training: {k}"
    for k, v in wrapper.projector.state_dict().items():
        assert torch.equal(v, proj_snap[k]), f"Projector mutated during MetaQuery training: {k}"


@pytest.mark.cuda
def test_metaquery_cuda_image_and_gif_conditioning():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(303)
    backbone = _make_v2_backbone("cuda", d_model=64).to(device)
    vision = MockVisionEncoder(vision_dim=32, grid_size=4, patch_size=16).to(device)
    wrapper = LuminaMultimodalWrapper(backbone, vision, downsample_factor=2).to(device)
    bridge = MetaQueryBridge(backbone, num_queries=16, dit_cond_dim=128, dit_pooled_dim=64).to(device)
    bridge.freeze_backbone_for_metaquery(wrapper)

    opt = torch.optim.AdamW([p for p in bridge.parameters() if p.requires_grad], lr=5e-3)
    prompt_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    imgs = [torch.randn(1, 3, 64, 64, device=device) for _ in range(2)]
    t_seq = torch.randn(1, 8, 16, 128, device=device)
    t_pool = torch.randn(1, 64, device=device)

    l0 = None
    l_last = None
    for step in range(6):
        opt.zero_grad(set_to_none=True)
        cache = SegmentMemoryCache(max_entries=4, max_anchors=4)
        out = bridge(prompt_ids=prompt_ids, images=imgs, wrapper=wrapper, grm_cache=cache, num_frames=8)
        loss, _ = compute_metaquery_alignment_loss(out, t_seq, t_pool)
        loss.backward()
        opt.step()
        if step == 0:
            l0 = float(loss.detach().item())
        l_last = float(loss.detach().item())

    assert l0 is not None and l_last is not None and l_last < l0
