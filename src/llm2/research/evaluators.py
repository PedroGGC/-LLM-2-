"""Generated-answer exact match and instrumented inference, never used to control curriculum."""

from dataclasses import fields
import time

import torch

from ..curriculum.generators import Problem, generate_problem, problem_set, verify_v1
from ..curriculum.skills import default_registry
from ..generation import generate
from ..model import count_parameters
from ..runtime import autocast_context, attention_context, capture_rng, restore_rng
from .accounting import backbone_cost


def evaluate_reasoning(model, tokenizer, *, seed=42, examples=2, difficulty=1, max_new_tokens=32,
                       precision="fp32", deterministic=True, seen_examples=None, iterations=None):
    if examples <= 0 or difficulty <= 0:
        raise ValueError("examples and difficulty must be positive")
    registry = default_registry()
    groups = {"unseen_validation": problem_set(registry, "val", seed=seed, examples=examples, difficulties=(difficulty,)),
              "unseen_test_and_families": problem_set(registry, "test", seed=seed, examples=examples, difficulties=(difficulty,)),
              "longer_steps": [generate_problem("multistep", difficulty+2, seed+i, "test") for i in range(examples)],
              "longer_sequences": [generate_problem("sequence", difficulty+2, seed+i, "test") for i in range(examples)]}
    if seen_examples:
        keys = {field.name for field in fields(Problem)}
        groups["recent_training_instances"] = [Problem(**{k: v for k, v in value.items() if k in keys})
                                               for value in seen_examples[:9*examples]]
    rng = capture_rng()
    rows, totals = [], {}
    started = time.perf_counter()
    device = next(model.parameters()).device
    try:
        for group, problems in groups.items():
            for problem in problems:
                prompt = tokenizer.encode(problem.prompt)
                if len(prompt) > model.config.max_seq_len:
                    rows.append({"group": group, "skill": problem.skill, "identifier": problem.identifier,
                                 "status": "context_overflow", "correct": False})
                    continue
                ids = torch.tensor([prompt], device=device)
                output = generate(model, ids, max_seq_len=model.config.max_seq_len, max_new_tokens=max_new_tokens,
                                  temperature=0, eos_id=tokenizer.eos_id, vocab_size=tokenizer.vocab_size,
                                  seed=seed, precision=precision, deterministic=deterministic, iterations=iterations)
                candidate = tokenizer.decode(output[0, len(prompt):].tolist())
                rows.append({"group": group, "skill": problem.skill, "difficulty": problem.difficulty,
                             "family": problem.family, "identifier": problem.identifier,
                             "expected": problem.answer, "predicted": candidate,
                             "correct": verify_v1(problem, candidate), "status": "evaluated"})
        for group in groups:
            selected = [row for row in rows if row["group"] == group]
            totals[group] = {"examples": len(selected), "exact_match": sum(row["correct"] for row in selected)/len(selected),
                             "context_overflow": sum(row["status"] == "context_overflow" for row in selected)}
        per_skill = {}
        for skill in registry.order:
            selected = [row for row in rows if row["skill"] == skill and row["group"] != "recent_training_instances"]
            per_skill[skill] = {"examples": len(selected), "exact_match": sum(row["correct"] for row in selected)/len(selected)}
    finally:
        restore_rng(rng)
    return {"kind": "generated_answer_exact_match", "seed": seed, "iterations_override": iterations,
            "groups": totals, "skills": per_skill, "predictions": rows,
            "observed_wall_seconds": time.perf_counter()-started,
            "limitations": "Small procedural tasks only; no general language competence claim. Over-context cases count as failures and are reported. Test split never controls training."}


@torch.no_grad()
def benchmark_inference(model, tokenizer, prompt, *, max_new_tokens=16, warmup_tokens=0,
                        precision="fp32", deterministic=True):
    if max_new_tokens <= 0 or warmup_tokens < 0:
        raise ValueError("invalid benchmark token counts")
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt) or [tokenizer.eos_id]], device=device)
    kwargs = dict(max_seq_len=model.config.max_seq_len, temperature=0, eos_id=None,
                  vocab_size=tokenizer.vocab_size, precision=precision, deterministic=deterministic)
    if warmup_tokens:
        generate(model, ids, max_new_tokens=warmup_tokens, **kwargs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = {}
    output = generate(model, ids, max_new_tokens=max_new_tokens, timings=timings, **kwargs)
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    rng = capture_rng()
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with autocast_context(device, precision), attention_context(deterministic):
            state_output = model(output[:, -model.config.max_seq_len:])
        value = getattr(state_output.state, "value", None)
        state_bytes = value.numel()*value.element_size() if value is not None else 0
        sequence_costs = [backbone_cost(model, min(model.config.max_seq_len, ids.shape[1]+i))
                          for i in range(max_new_tokens)]
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
    flops = sum(cost[0] for cost in sequence_costs) if all(cost is not None for cost in sequence_costs) else None
    return {**timings, "device": str(device), "precision": precision,
            "parameters_calculated": count_parameters(model), "prompt_tokens": ids.shape[1],
            "max_context": model.config.max_seq_len, "warmup_tokens": warmup_tokens,
            "weight_bytes_calculated": sum(p.numel()*p.element_size() for p in model.parameters()),
            "returned_state_bytes_calculated": state_bytes, "persistent_cross_segment_state_bytes": 0,
            "internal_iterations": getattr(state_output.state, "iterations", 1),
            "generation_flops_estimate": flops,
            "flops_per_generated_token_estimate": flops/max_new_tokens if flops is not None else None,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
            "limitations": "Single-process, batch 1, eager full-window generation without KV cache; load excluded, requested warmup excluded. GPU allocator is not total board memory. State inspection is outside latency."}
