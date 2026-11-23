"""3-stage multimodal training recipe for Lumina V1 preserving text fluency and FP32 loss invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch.nn import functional as F

from .vision_bridge import LuminaMultimodalWrapper


@dataclass
class VLMStageConfig:
    stage: int  # 1=projector_only, 2=projector+prelude+adapters, 3=sft_full_backbone
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 100
    grad_clip: float = 1.0
    max_text_tokens: int = 256
    precision: str = "fp32"  # fp32 | bf16

    def __post_init__(self):
        if self.stage not in (1, 2, 3):
            raise ValueError("VLMStageConfig.stage must be 1, 2, or 3")
        if self.lr <= 0 or self.max_steps <= 0 or self.grad_clip <= 0:
            raise ValueError("lr, max_steps, and grad_clip must be positive")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32|bf16")


def apply_vlm_stage_freeze(wrapper: LuminaMultimodalWrapper, stage: int) -> dict[str, int]:
    """Enforce strict parameter freezing per VLM training stage (vision encoder always frozen)."""
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    wrapper.vision_encoder.requires_grad_(False)
    wrapper.backbone.requires_grad_(False)
    wrapper.projector.requires_grad_(True)
    wrapper.img_start_emb.requires_grad_(True)
    wrapper.img_end_emb.requires_grad_(True)

    if stage == 2:
        # Unfreeze Prelude (cell 0), associative bridge, and lightweight per-level adapters.
        wrapper.backbone.cells[0].requires_grad_(True)
        wrapper.backbone.adapters.requires_grad_(True)
        for mod in (
            wrapper.backbone.input_norm,
            wrapper.backbone.local_conv,
            wrapper.backbone.local_norm,
            wrapper.backbone.assoc_address,
            wrapper.backbone.assoc_value,
            wrapper.backbone.assoc_write_gate,
            wrapper.backbone.assoc_readout,
        ):
            mod.requires_grad_(True)
        wrapper.backbone.assoc_scale.requires_grad_(True)
        if wrapper.backbone.assoc_lag_mix is not None:
            wrapper.backbone.assoc_lag_mix.requires_grad_(True)
    elif stage == 3:
        wrapper.backbone.requires_grad_(True)

    trainable = {p.data_ptr(): p.numel() for p in wrapper.parameters() if p.requires_grad}
    frozen = {p.data_ptr(): p.numel() for p in wrapper.parameters() if not p.requires_grad}
    return {
        "stage": stage,
        "trainable_params": sum(trainable.values()),
        "frozen_params": sum(frozen.values()),
    }


def _encode_text(tokenizer, text: str, max_len: int) -> list[int]:
    ids = list(tokenizer.encode(text))
    if not ids:
        ids = [0]
    return ids[:max_len]


def train_vlm_stage(
    wrapper: LuminaMultimodalWrapper,
    samples: Iterable,
    tokenizer,
    preprocess_fn: Callable[[list], torch.Tensor],
    cfg: VLMStageConfig,
    device: torch.device,
) -> dict:
    """Run one stage of VLM training with FP32 CE loss masked strictly to response tokens."""
    if cfg.precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 VLM training requires CUDA")

    freeze_info = apply_vlm_stage_freeze(wrapper, cfg.stage)
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    wrapper.train()
    wrapper.vision_encoder.eval()
    if cfg.stage == 1:
        wrapper.backbone.eval()

    eos_id = int(getattr(tokenizer, "eos_id", 0))
    losses: list[float] = []
    tokens_supervised = 0
    step = 0

    for sample in samples:
        if step >= cfg.max_steps:
            break
        # Stage 1 freezes the entire backbone and trains only the vision projector; skip text-only rows cleanly.
        if cfg.stage == 1 and not sample.images:
            continue
        prompt_toks = [eos_id] + _encode_text(tokenizer, f"User: {sample.prompt}\nAssistant:", cfg.max_text_tokens // 2)
        resp_toks = _encode_text(tokenizer, f" {sample.response}", cfg.max_text_tokens // 2) + [eos_id]

        prompt_ids = torch.tensor([prompt_toks], dtype=torch.long, device=device)
        resp_ids = torch.tensor([resp_toks], dtype=torch.long, device=device)
        img_tensors = [preprocess_fn([img]).to(device=device) for img in sample.images]

        optimizer.zero_grad(set_to_none=True)
        use_amp = cfg.precision == "bf16" and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out, targets, _ = wrapper(images=img_tensors, prompt_ids=prompt_ids, response_ids=resp_ids)
        # Invariant (AGENTS.md): always compute cross-entropy loss in FP32.
        loss = F.cross_entropy(out.logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite VLM loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().item()))
        tokens_supervised += int((targets != -100).sum().item())
        step += 1

    return {
        **freeze_info,
        "steps_completed": step,
        "tokens_supervised": tokens_supervised,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
        "losses": losses,
    }
