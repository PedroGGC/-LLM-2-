"""Interpretable sampler with feedback, uncertainty, an exploration floor and revisitation."""

import copy
import math
import random


class CurriculumController:
    def __init__(self, registry, policy="adaptive", *, max_difficulty=3, exploration=0.1, seed=42, fixed_steps=100):
        if policy not in ("uniform", "fixed", "adaptive") or not 0 < exploration <= 1 or max_difficulty <= 0 or fixed_steps <= 0:
            raise ValueError("invalid curriculum configuration")
        self.registry, self.policy = registry, policy
        self.exploration, self.max_difficulty, self.fixed_steps = exploration, max_difficulty, fixed_steps
        self.cells = [(skill, level) for skill in registry.order
                      for level in range(1, min(max_difficulty, registry.skills[skill].max_difficulty) + 1)]
        self.records = {f"{skill}:{level}": {"visits": 0, "success": 0.0, "progress": 0.0}
                        for skill, level in self.cells}
        self.rng = random.Random(seed)
        self.draws, self.history = 0, []

    def distribution(self):
        total_visits = sum(record["visits"] for record in self.records.values())
        scores = []
        for skill, level in self.cells:
            record = self.records[f"{skill}:{level}"]
            if self.policy == "uniform":
                score = 1.0
            elif self.policy == "fixed":
                target = min(self.max_difficulty, 1 + self.draws // self.fixed_steps)
                score = 1.0 if level == target else 0.1
            else:
                uncertainty = math.sqrt(math.log(total_visits + 2) / (record["visits"] + 1))
                dependencies = self.registry.skills[skill].dependencies
                readiness = (sum(self.records[f"{dep}:1"]["success"] for dep in dependencies) / len(dependencies)) if dependencies else 1.0
                score = (0.5 + 1 - record["success"] + abs(record["progress"]) + uncertainty) * (0.5 + 0.5*readiness)
            scores.append(score)
        normalizer = sum(scores)
        return {(skill, level): self.exploration/len(scores) + (1-self.exploration)*score/normalizer
                for (skill, level), score in zip(self.cells, scores)}

    def sample(self):
        distribution = self.distribution()
        cell = self.rng.choices(list(distribution), weights=list(distribution.values()), k=1)[0]
        self.draws += 1
        return cell

    def update(self, skill, difficulty, success, *, split):
        if split != "probe":
            raise ValueError("curriculum feedback must come only from the dedicated probe split")
        if not math.isfinite(success) or not 0 <= success <= 1:
            raise ValueError("success must be finite and in [0, 1]")
        key = f"{skill}:{difficulty}"
        if key not in self.records:
            raise ValueError("unknown skill/difficulty")
        record = self.records[key]
        before = record["success"]
        record["success"] = success if record["visits"] == 0 else 0.8*before + 0.2*success
        record["progress"] = record["success"] - before
        record["visits"] += 1
        self.history.append({"cell": key, "score": success, "visits": record["visits"]})
        self.history = self.history[-256:]

    def _contract(self):
        return {"version": 1, "registry": self.registry.fingerprint, "policy": self.policy,
                "max_difficulty": self.max_difficulty, "exploration": self.exploration, "fixed_steps": self.fixed_steps}

    def state_dict(self):
        return copy.deepcopy({"contract": self._contract(), "records": self.records,
                              "history": self.history, "draws": self.draws, "rng": self.rng.getstate()})

    def load_state_dict(self, state):
        if state["contract"] != self._contract():
            raise ValueError("incompatible curriculum controller")
        self.records, self.history, self.draws = copy.deepcopy(state["records"]), copy.deepcopy(state["history"]), state["draws"]
        self.rng.setstate(state["rng"])
