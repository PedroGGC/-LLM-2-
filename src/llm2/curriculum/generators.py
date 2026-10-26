"""CPU-only deterministic tasks; verifiers recompute answers without eval or an LLM."""

from dataclasses import asdict, dataclass
import hashlib
import json
import random


SPLITS = {"train": 0, "probe": 1, "val": 2, "test": 3}


@dataclass(frozen=True)
class Problem:
    prompt: str
    answer: str
    skill: str
    difficulty: int
    seed: int
    split: str
    family: str
    metadata: dict

    @property
    def identifier(self):
        # Content ID deliberately excludes seed/split so identical questions cannot masquerade as disjoint.
        return hashlib.sha256((self.skill + "\n" + self.prompt).encode()).hexdigest()

    def to_dict(self):
        return {**asdict(self), "identifier": self.identifier,
                "verifier": "llm2.curriculum.generators.verify_v1"}


def solve(skill, metadata):
    m = metadata
    a, b = m.get("a", 0), m.get("b", 0)
    if skill == "arithmetic":
        value = a + b if m["operation"] == "+" else a - b
    elif skill == "comparison":
        value = ">" if a > b else "<" if a < b else "="
    elif skill == "conditional":
        value = a + b if a > b else a - b
    elif skill == "composition":
        value = (a + b)*m["c"] if m["family"] == "A" else a*b + m["c"]
    elif skill == "sequence":
        value = a + m["step"] * m["length"]
    elif skill == "symbolic":
        value = m["text"][::-1] if m["family"] == "A" else m["text"].upper()
    elif skill == "logic":
        value = "yes" if ((a < b and b < m["c"]) if m["family"] == "A" else
                          (a < b or b < m["c"])) else "no"
    elif skill == "multistep":
        value = a
        for operation, amount in m["operations"]:
            value = value + amount if operation == "+" else value * amount if operation == "*" else value - amount
    elif skill == "structured":
        value = ",".join(map(str, sorted(m["values"], reverse=m["family"] == "B")))
    else:
        raise ValueError(f"unknown skill: {skill}")
    return str(value)


def verify_v1(problem, candidate):
    return candidate.strip() == solve(problem.skill, problem.metadata)


def generate_problem(skill, difficulty, seed, split="train", *, family=None):
    if split not in SPLITS or type(difficulty) is not int or difficulty < 1 or seed < 0:
        raise ValueError("invalid procedural split, difficulty or seed")
    namespace = hashlib.sha256(f"v1:{split}:{skill}:{difficulty}:{seed}".encode()).digest()
    rng = random.Random(int.from_bytes(namespace[:8], "little"))
    # Each split/difficulty uses disjoint numeric bands, not just distinct RNG seeds.
    base = difficulty * 10000 + SPLITS[split] * 1000
    a, b, c = (base + rng.randrange(1, 100) for _ in range(3))
    fam = family or ("B" if split == "test" and skill in ("composition", "symbolic", "logic", "structured") else "A")
    if fam not in ("A", "B"):
        raise ValueError("family must be A or B")
    m = {"a": a, "b": b, "c": c, "family": fam}
    if skill == "arithmetic":
        m["operation"] = "+" if difficulty == 1 or rng.random() < 0.5 else "-"
        prompt = f"{a}{m['operation']}{b} = "
    elif skill == "comparison":
        prompt = f"Compare {a} and {b}. Answer <, > or =: "
    elif skill == "conditional":
        prompt = f"If {a}>{b}, add them; otherwise subtract second from first. Answer: "
    elif skill == "composition":
        prompt = f"({a}+{b})*{c} = " if fam == "A" else f"({a}*{b})+{c} = "
    elif skill == "sequence":
        m.update(step=rng.randrange(1, difficulty*5 + 1), length=3 + difficulty)
        values = [a + i*m["step"] for i in range(m["length"])]
        prompt = f"Continue by constant difference: {','.join(map(str, values))}, "
    elif skill == "symbolic":
        # A numeric identifier embeds the disjoint split band as ordinary task content.
        m["text"] = f"id{a}x{rng.randrange(1000)}"
        prompt = f"{'Reverse' if fam == 'A' else 'Uppercase'} {m['text']}: "
    elif skill == "logic":
        prompt = f"({a}<{b}) {'AND' if fam == 'A' else 'OR'} ({b}<{c})? yes/no: "
    elif skill == "multistep":
        m["operations"] = [(op, rng.randrange(1, 5)) for _ in range(difficulty) for op in ("+", "*", "-")]
        prompt = f"Start {a}; " + "; ".join(f"{op}{amount}" for op, amount in m["operations"]) + ". Result: "
    elif skill == "structured":
        m["values"] = [base + rng.randrange(1, 100) for _ in range(2 + difficulty)]
        prompt = f"Sort {'ascending' if fam == 'A' else 'descending'} {','.join(map(str, m['values']))}: "
    else:
        raise ValueError(f"unknown skill: {skill}")
    return Problem(prompt, solve(skill, m), skill, difficulty, seed, split, fam, m)


def problem_set(registry, split, *, seed=42, examples=2, difficulties=(1,)):
    problems, identifiers = [], set()
    for skill in registry.order:
        for difficulty in difficulties:
            accepted, attempt = 0, 0
            while accepted < examples:
                if attempt > examples * 100:
                    raise ValueError("could not generate enough distinct procedural instances")
                problem = generate_problem(skill, difficulty, seed + attempt, split)
                attempt += 1
                if problem.identifier not in identifiers:
                    identifiers.add(problem.identifier)
                    problems.append(problem)
                    accepted += 1
    return problems
