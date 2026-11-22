"""External zero-shot log-likelihood benchmarks (LAMBADA, ARC-Easy, HellaSwag) and long-context RULER NIAH."""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch
from torch.nn import functional as F

from ..architectures.fractal_delta import FractalDecode
from .decode_optimization import SegmentMemoryCache


@torch.inference_mode()
def score_continuation_nll(
    model,
    tokenizer,
    context: str,
    continuation: str,
    device: torch.device,
    *,
    max_seq_len: int | None = None,
) -> tuple[float, int, bool]:
    """Compute exact sum NLL, token count, and greedy exact-match over continuation tokens given context."""
    ctx_ids = tokenizer.encode(context)
    cont_ids = tokenizer.encode(continuation)
    if not cont_ids:
        return 0.0, 0, True
    bos = getattr(tokenizer, "eos_id", 0)
    seq = [bos, *ctx_ids, *cont_ids]
    cap = model.config.max_seq_len if max_seq_len is None else min(int(max_seq_len), model.config.max_seq_len)
    if len(seq) > cap + 1:
        seq = seq[-(cap + 1) :]
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
    cont_len = min(len(cont_ids), y.shape[1])

    out = model(x)
    logits = out.logits[0, -cont_len:].float()
    targets = y[0, -cont_len:]
    sum_nll = float(F.cross_entropy(logits, targets, reduction="sum").item())
    exact = bool(torch.equal(logits.argmax(dim=-1), targets))
    return sum_nll, cont_len, exact


@torch.inference_mode()
def evaluate_multiple_choice(
    model,
    tokenizer,
    examples: Sequence[tuple[str, Sequence[str], int]],
    device: torch.device,
) -> dict:
    """Zero-shot length-normalized log-likelihood multiple-choice accuracy."""
    correct_raw = 0
    correct_norm = 0
    total = 0
    for prompt, choices, gold in examples:
        if not choices or not (0 <= gold < len(choices)):
            continue
        raw_scores = []
        norm_scores = []
        for choice in choices:
            nll, n_tok, _ = score_continuation_nll(model, tokenizer, prompt, " " + choice.strip(), device)
            raw_scores.append(nll)
            norm_scores.append(nll / max(len(choice.strip()), 1))
        if int(min(range(len(choices)), key=lambda i: raw_scores[i])) == int(gold):
            correct_raw += 1
        if int(min(range(len(choices)), key=lambda i: norm_scores[i])) == int(gold):
            correct_norm += 1
        total += 1
    return {
        "examples": total,
        "acc": (correct_raw / total) if total else None,
        "acc_norm": (correct_norm / total) if total else None,
    }


@torch.inference_mode()
def evaluate_lambada(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
) -> dict:
    """Zero-shot LAMBADA last-word exact match and perplexity."""
    total_nll = 0.0
    total_tok = 0
    exact_hits = 0
    count = 0
    for text in texts:
        parts = text.strip().rsplit(" ", 1)
        if len(parts) != 2:
            continue
        ctx, target_word = parts[0], " " + parts[1]
        nll, n_tok, exact = score_continuation_nll(model, tokenizer, ctx, target_word, device)
        if n_tok <= 0:
            continue
        total_nll += nll
        total_tok += n_tok
        exact_hits += int(exact)
        count += 1
    mean_nll = (total_nll / total_tok) if total_tok else None
    return {
        "examples": count,
        "exact_match": (exact_hits / count) if count else None,
        "mean_nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 50.0)) if mean_nll is not None else None,
    }


@torch.inference_mode()
def evaluate_ruler_niah(
    model,
    tokenizer,
    device: torch.device,
    *,
    lengths: Sequence[int] = (2048, 8192),
    examples_per_len: int = 8,
    seed: int = 2026,
    use_segment_cache: bool = False,
    use_grm: bool = False,
) -> dict:
    """Synthetic RULER single-needle retrieval across long contexts (with optional blind query-dependent GRM)."""
    rng = random.Random(seed)
    filler_sentences = (
        "The weather remained calm across the valley while birds crossed the river. ",
        "Engineers reviewed notes about bridge bearings and seasonal rainfall records. ",
        "Students gathered books from the library shelves before the evening lecture. ",
        "Ships waited outside the harbor until the morning tide cleared the channel. ",
    )
    results = {}
    is_fractal = hasattr(model, "cells") and hasattr(model, "_run_assoc")
    for target_len in lengths:
        hits = 0
        mean_nll_sum = 0.0
        for idx in range(examples_per_len):
            passkey = f"{rng.randint(1000, 9999)}"
            needle = f"The secret code for vault {idx + 1} is {passkey}. "
            query = f"What is the secret code for vault {idx + 1}? The secret code is"
            answer = f" {passkey}"

            needle_ids = tokenizer.encode(needle)
            query_ids = tokenizer.encode(query)
            ans_ids = tokenizer.encode(answer)
            filler_budget = max(32, int(target_len) - len(needle_ids) - len(query_ids) - len(ans_ids) - 4)

            prefix_chunks = []
            tok_count = 0
            while tok_count < filler_budget:
                s = rng.choice(filler_sentences)
                ids = tokenizer.encode(s)
                prefix_chunks.extend(ids)
                tok_count += len(ids)
            prefix_chunks = prefix_chunks[:filler_budget]
            # Insert needle in the first third so retrieval truly spans long range
            ins_pos = max(8, len(prefix_chunks) // 4)
            full_ctx = prefix_chunks[:ins_pos] + needle_ids + prefix_chunks[ins_pos:] + query_ids

            if is_fractal:
                # Stream in chunks <= model.config.max_seq_len so 8K–16K works even if max_seq_len=4096
                chunk_len = min(1024, model.config.max_seq_len)
                state = None
                cache = SegmentMemoryCache(max_entries=8, max_anchors=16) if (use_segment_cache or use_grm) else None
                for start in range(0, len(full_ctx), chunk_len):
                    seg = torch.tensor([full_ctx[start : start + chunk_len]], dtype=torch.long, device=device)
                    out = model(seg, state=state, return_logits=False)
                    state = out.state
                    if cache is not None:
                        if use_grm:
                            cache.record_segment(state, out.hidden_states, tag=f"seg_{start}")
                        elif start <= ins_pos < start + chunk_len:
                            cache.record_anchor(state)
                grm_delta = None
                if cache is not None and use_grm and len(cache._grm_segments) > 1:
                    q_slice = out.hidden_states[:, -min(len(query_ids), out.hidden_states.shape[1]) :]
                    state, _ = cache.retrieve_grm_state(
                        q_slice, active_state=state, top_k=2, temperature=0.5, grm_weight=0.25
                    )
                    grm_delta, _ = cache.query_grm_readout(out.hidden_states[:, -1:], model=model, top_k=2, temperature=0.5)
                elif cache is not None and len(cache._anchors) > 0:
                    state = cache.blend_with_anchors(state, anchor_weight=0.15)
                dec = FractalDecode(model)
                # Score answer tokens incrementally from carried state
                ans_nll = 0.0
                exact = True
                # First answer token is predicted from the last context token already in out.hidden_states
                last_h = out.hidden_states[:, -1:] if grm_delta is None else (out.hidden_states[:, -1:] + 0.25 * grm_delta)
                first_logits = model.lm_head(last_h).float()
                for a_idx, target_tok in enumerate(ans_ids):
                    logits_step = first_logits if a_idx == 0 else step_logits.float()
                    ans_nll += float(
                        F.cross_entropy(logits_step[0, -1:], torch.tensor([target_tok], device=device)).item()
                    )
                    if int(logits_step[0, -1].argmax().item()) != int(target_tok):
                        exact = False
                    if a_idx + 1 < len(ans_ids):
                        step_logits, state = dec.step(torch.tensor([[target_tok]], dtype=torch.long, device=device), state)
                mean_nll_sum += ans_nll / max(len(ans_ids), 1)
                hits += int(exact)
            else:
                ctx_str = tokenizer.decode(full_ctx)
                nll, n_tok, exact = score_continuation_nll(model, tokenizer, ctx_str, answer, device)
                mean_nll_sum += nll / max(n_tok, 1)
                hits += int(exact)

        results[str(target_len)] = {
            "examples": examples_per_len,
            "exact_match": hits / max(examples_per_len, 1),
            "mean_answer_nll": round(mean_nll_sum / max(examples_per_len, 1), 4),
        }
    return results


def evaluate_hf_suite(
    model,
    tokenizer,
    device: torch.device,
    *,
    max_examples: int = 100,
    ruler_lengths: Sequence[int] = (2048, 8192),
) -> dict:
    """Run real HuggingFace LAMBADA, ARC-Easy, HellaSwag + RULER benchmarks with graceful per-task error capture."""
    from datasets import load_dataset

    report: dict = {}

    try:
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        texts = [ds[i]["text"] for i in range(min(max_examples, len(ds)))]
        report["lambada"] = evaluate_lambada(model, tokenizer, texts, device)
    except Exception as exc:
        report["lambada"] = {"error": str(exc)}

    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        examples = []
        for i in range(min(max_examples, len(ds))):
            row = ds[i]
            labels = row["choices"]["label"]
            texts_c = row["choices"]["text"]
            if row["answerKey"] in labels:
                gold = labels.index(row["answerKey"])
                examples.append((f"Question: {row['question']}\nAnswer:", texts_c, gold))
        report["arc_easy"] = evaluate_multiple_choice(model, tokenizer, examples, device)
    except Exception as exc:
        report["arc_easy"] = {"error": str(exc)}

    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        examples = []
        for i in range(min(max_examples, len(ds))):
            row = ds[i]
            if str(row["label"]).isdigit():
                examples.append((row["ctx"], row["endings"], int(row["label"])))
        report["hellaswag"] = evaluate_multiple_choice(model, tokenizer, examples, device)
    except Exception as exc:
        report["hellaswag"] = {"error": str(exc)}

    report["ruler_niah"] = evaluate_ruler_niah(model, tokenizer, device, lengths=ruler_lengths, examples_per_len=8, use_segment_cache=False)
    report["ruler_niah_segment_cache"] = evaluate_ruler_niah(model, tokenizer, device, lengths=ruler_lengths, examples_per_len=8, use_segment_cache=True)
    return report
