"""Evaluation preserves training modes and RNG; no sampler or optimizer mutations."""

import math

import torch

from .runtime import attention_context, autocast_context, capture_rng, restore_rng


@torch.no_grad()
def evaluate(model, objective, batches, *, device="cpu", precision="fp32", deterministic=True):
    rng = capture_rng()
    modes = [(module, module.training) for root in (model, objective) for module in root.modules()]
    model.eval()
    objective.eval()
    loss_sum, tokens, num_batches = 0.0, 0, 0
    try:
        with autocast_context(device, precision), attention_context(deterministic):
            for batch in batches:
                result = objective(model, batch)
                value = float(result.loss)
                if not math.isfinite(value) or result.token_count <= 0:
                    raise FloatingPointError("invalid validation loss or token count")
                loss_sum += value * result.token_count
                tokens += result.token_count
                num_batches += 1
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
    if tokens == 0:
        raise ValueError("empty validation iterator")
    return {"validation_loss": loss_sum / tokens, "validation_tokens": tokens,
            "validation_batches": num_batches}

