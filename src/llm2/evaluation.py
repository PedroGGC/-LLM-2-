"""Evaluation preserves training modes and RNG; no sampler or optimizer mutations."""

import math

import torch

from .runtime import attention_context, autocast_context, capture_rng, restore_rng


@torch.no_grad()
def evaluate(model, objective, batches, *, device="cpu", precision="fp32", deterministic=True,
             window_callback=None):
    rng = capture_rng()
    modes = [(module, module.training) for root in (model, objective) for module in root.modules()]
    model.eval()
    objective.eval()
    loss_sum, tokens, num_batches = 0.0, 0, 0
    lm_sum = 0.0
    sequences = 0
    try:
        with autocast_context(device, precision), attention_context(deterministic):
            for batch in batches:
                result = objective(model, batch)
                value = float(result.loss)
                if not math.isfinite(value) or result.token_count <= 0:
                    raise FloatingPointError("invalid validation loss or token count")
                loss_sum += value * result.token_count
                lm_sum += result.metrics.get("lm_loss", value) * result.token_count
                if window_callback is not None:
                    if result.token_losses is None:
                        raise ValueError("per-window export requires an objective with token_losses")
                    valid = batch["targets"] != -100
                    counts = valid.sum(1).tolist()
                    sums = result.token_losses.double().sum(1).tolist()
                    for index, (count, nll) in enumerate(zip(counts, sums)):
                        window_callback({"window_index": sequences + index,
                                         "input_length": batch["input_ids"].shape[1],
                                         "supervised_tokens": count, "nll_sum": nll,
                                         "lm_loss": nll/count if count else None})
                tokens += result.token_count
                num_batches += 1
                sequences += batch["input_ids"].shape[0]
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
    if tokens == 0:
        raise ValueError("empty validation iterator")
    lm_loss = lm_sum / tokens
    return {"validation_loss": lm_loss, "validation_objective_loss": loss_sum / tokens,
            "validation_perplexity": math.exp(lm_loss) if lm_loss < 700 else None, "validation_tokens": tokens,
            "validation_sequences": sequences,
            "validation_batches": num_batches}
