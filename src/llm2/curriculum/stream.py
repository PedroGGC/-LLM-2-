"""Mixed text/procedural batches, full-state sampling and isolated curriculum probes."""

import random

import torch

from ..runtime import attention_context, autocast_context, capture_rng, restore_rng
from .controller import CurriculumController
from .generators import generate_problem, problem_set
from .skills import default_registry


def pack_problem(problem, tokenizer, seq_len):
    prompt = tokenizer.encode(problem.prompt)
    answer = tokenizer.encode(problem.answer) + [tokenizer.eos_id]
    ids = prompt + answer
    if len(ids) > seq_len + 1:
        raise ValueError(f"procedural problem exceeds context; increase seq_len ({len(ids)} > {seq_len+1})")
    # Supervise only the complete answer; no truncation of the problem or solution.
    targets = [-100] * seq_len
    inputs = (ids + [tokenizer.eos_id] * (seq_len + 1 - len(ids)))[:-1]
    for index in range(len(prompt)-1, len(ids)-1):
        targets[index] = ids[index+1]
    return torch.tensor(inputs), torch.tensor(targets)


class MixedCurriculumStream:
    def __init__(self, text_stream, tokenizer, research, seed):
        self.text_stream, self.tokenizer, self.research = text_stream, tokenizer, research
        self.registry = default_registry()
        self.controller = CurriculumController(self.registry, research.curriculum,
            max_difficulty=research.max_difficulty, exploration=research.exploration, seed=seed+1,
            fixed_steps=research.fixed_curriculum_steps)
        self.rng = random.Random(seed)
        self.counts = {"text_examples": 0, "procedural_examples": 0, "text_targets": 0, "procedural_targets": 0}
        self.batches = 0
        self.last_source = "text"
        self.recent_problems = []

    def next(self, device="cpu"):
        self.batches += 1
        if self.rng.random() < self.research.text_fraction:
            self.last_source = "text"
            batch = self.text_stream.next(device)
        else:
            self.last_source = "procedural"
            examples = []
            for _ in range(self.text_stream.batch_size):
                skill, level = self.controller.sample()
                problem = generate_problem(skill, level, self.rng.getrandbits(63), "train")
                self.recent_problems.append(problem.to_dict())
                self.recent_problems = self.recent_problems[-128:]
                examples.append(pack_problem(problem, self.tokenizer, self.text_stream.seq_len))
            batch = {"input_ids": torch.stack([x for x, _ in examples]).to(device),
                     "targets": torch.stack([y for _, y in examples]).to(device)}
        source = self.last_source
        self.counts[f"{source}_examples"] += batch["input_ids"].shape[0]
        self.counts[f"{source}_targets"] += int((batch["targets"] != -100).sum())
        return batch

    def state_dict(self):
        return {"version": 1, "text": self.text_stream.state_dict(), "controller": self.controller.state_dict(),
                "rng": self.rng.getstate(), "counts": self.counts.copy(), "batches": self.batches,
                "last_source": self.last_source, "text_fraction": self.research.text_fraction,
                "recent_problems": list(self.recent_problems)}

    def load_state_dict(self, state):
        if state["version"] != 1 or state["text_fraction"] != self.research.text_fraction:
            raise ValueError("incompatible mixed sampler")
        self.text_stream.load_state_dict(state["text"])
        self.controller.load_state_dict(state["controller"])
        self.rng.setstate(state["rng"])
        self.counts, self.batches, self.last_source = state["counts"].copy(), state["batches"], state["last_source"]
        self.recent_problems = list(state["recent_problems"])

    @torch.no_grad()
    def probe(self, model, device, precision, deterministic):
        rng = capture_rng()
        modes = [(module, module.training) for module in model.modules()]
        model.eval()
        scores = {}
        examples = problem_set(self.registry, "probe", seed=19871, examples=self.research.probe_examples,
                               difficulties=range(1, min(self.research.max_difficulty, 3)+1))
        try:
            with autocast_context(device, precision), attention_context(deterministic):
                for problem in examples:
                    ids, targets = pack_problem(problem, self.tokenizer, self.text_stream.seq_len)
                    targets = targets.to(device)
                    logits = model(ids.unsqueeze(0).to(device)).logits[0]
                    mask = targets != -100
                    # Cheap teacher-forced answer-token accuracy is a probe, not final exact match.
                    score = float((logits.argmax(-1)[mask] == targets[mask]).float().mean())
                    key = (problem.skill, problem.difficulty)
                    scores.setdefault(key, []).append(score)
            for (skill, level), values in scores.items():
                self.controller.update(skill, level, sum(values)/len(values), split="probe")
        finally:
            for module, mode in modes:
                module.training = mode
            restore_rng(rng)
        return {"probe_answer_token_accuracy": sum(sum(v) for v in scores.values())/len(examples),
                "probe_sequences": len(examples)}
