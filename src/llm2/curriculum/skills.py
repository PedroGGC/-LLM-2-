"""Small manual dependency graph; this is a sampling prior, not a neural knowledge map."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class Skill:
    identifier: str
    dependencies: tuple[str, ...] = ()
    max_difficulty: int = 3


class SkillRegistry:
    def __init__(self, skills):
        self.skills = {s.identifier: s for s in skills}
        if len(self.skills) != len(skills) or not self.skills:
            raise ValueError("skills must be nonempty and unique")
        visiting, visited, order = set(), set(), []

        def visit(key):
            if key not in self.skills:
                raise ValueError(f"missing dependency: {key}")
            if key in visiting:
                raise ValueError("skill graph contains a cycle")
            if key in visited:
                return
            visiting.add(key)
            for dependency in self.skills[key].dependencies:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)
            order.append(key)
        for key in sorted(self.skills):
            visit(key)
        self.order = tuple(order)
        raw = [asdict(self.skills[key]) for key in self.order]
        self.fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def default_registry():
    return SkillRegistry([
        Skill("arithmetic"), Skill("comparison"), Skill("conditional", ("comparison", "arithmetic")),
        Skill("composition", ("arithmetic",)), Skill("sequence", ("arithmetic",)),
        Skill("symbolic"), Skill("logic", ("comparison",)),
        Skill("multistep", ("composition", "conditional")), Skill("structured", ("comparison",)),
    ])
