"""Lumina V1 Multimodal bridge: SigLIP 2 vision encoder, 2D spatial projector, GRM visual anchors, and 3-stage VLM trainer."""

from .vision_bridge import (
    LuminaMultimodalWrapper,
    MockVisionEncoder,
    SigLIPVisionEncoder,
    VisionSpatialProjector,
)
from .vlm_data import (
    DEFAULT_VISION_PRESETS,
    MultimodalSample,
    iter_hf_multimodal,
    iter_jsonl_multimodal,
    mix_multimodal_streams,
)
from .vlm_trainer import VLMStageConfig, apply_vlm_stage_freeze, train_vlm_stage

__all__ = [
    "DEFAULT_VISION_PRESETS",
    "LuminaMultimodalWrapper",
    "MockVisionEncoder",
    "MultimodalSample",
    "SigLIPVisionEncoder",
    "VLMStageConfig",
    "VisionSpatialProjector",
    "apply_vlm_stage_freeze",
    "iter_hf_multimodal",
    "iter_jsonl_multimodal",
    "mix_multimodal_streams",
    "train_vlm_stage",
]
