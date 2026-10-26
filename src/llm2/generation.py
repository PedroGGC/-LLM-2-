"""Simple autoregressive sampling for baseline inspection; no inference server or KV cache."""

import math

import torch

from .runtime import attention_context, autocast_context


@torch.no_grad()
def generate(model, input_ids, *, max_seq_len, max_new_tokens=64, temperature=1.0,
             top_k=None, eos_id=None, vocab_size=None, seed=42, precision="fp32", deterministic=True):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("generation expects one nonempty prompt [1, time]")
    if max_new_tokens < 0 or max_seq_len <= 0 or temperature < 0 or not math.isfinite(temperature):
        raise ValueError("invalid generation length or temperature")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    rng = torch.Generator(device=input_ids.device).manual_seed(seed)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    ids = input_ids.clone()
    try:
        with autocast_context(ids.device, precision), attention_context(deterministic):
            for _ in range(max_new_tokens):
                logits = model(ids[:, -max_seq_len:]).logits[:, -1].float()
                if vocab_size is not None:
                    if not 0 < vocab_size <= logits.shape[-1]:
                        raise ValueError("invalid tokenizer vocabulary size")
                    logits = logits[:, :vocab_size]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("non-finite generation logits")
                if temperature == 0:
                    token = logits.argmax(-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        threshold = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                        logits.masked_fill_(logits < threshold, -float("inf"))
                    token = torch.multinomial(logits.softmax(-1), 1, generator=rng)
                ids = torch.cat((ids, token), dim=1)
                if eos_id is not None and token.item() == eos_id:
                    break
    finally:
        for module, mode in modes:
            module.training = mode
    return ids

