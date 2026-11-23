"""Lumina V1 Multimodal bridge: SigLIP 2 vision encoder, 2D spatial projector, GRM visual anchors, and 3-stage VLM trainer."""

from .metaquery_head import (
    MetaQueryBridge,
    MetaQueryOutput,
    compute_metaquery_alignment_loss,
)
from .vision_bridge import (
    LuminaMultimodalWrapper,
    MockVisionEncoder,
    SigLIPVisionEncoder,
    VisionSpatialProjector,
)
from .vlm_data import (
    DEFAULT_TEXT_SFT_PRESETS,
    DEFAULT_VISION_PRESETS,
    MultimodalSample,
    clean_civitai_prompt,
    iter_hf_multimodal,
    iter_hf_text_sft,
    iter_jsonl_multimodal,
    mix_multimodal_streams,
)
from .vlm_trainer import VLMStageConfig, apply_vlm_stage_freeze, train_vlm_stage

__all__ = [
    "DEFAULT_TEXT_SFT_PRESETS",
    "DEFAULT_VISION_PRESETS",
    "LuminaMultimodalWrapper",
    "MetaQueryBridge",
    "MetaQueryOutput",
    "MockVisionEncoder",
    "MultimodalSample",
    "SigLIPVisionEncoder",
    "VLMStageConfig",
    "VisionSpatialProjector",
    "apply_vlm_stage_freeze",
    "clean_civitai_prompt",
    "compute_metaquery_alignment_loss",
    "iter_hf_multimodal",
    "iter_hf_text_sft",
    "iter_jsonl_multimodal",
    "mix_multimodal_streams",
    "train_vlm_stage",
]
