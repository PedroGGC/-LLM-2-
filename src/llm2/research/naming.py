"""Readable, collision-safe output numbers; this module also runs via runpy."""

from pathlib import Path
import re


def numbered_output(parent, prefix="llm2-campaign", *, number=None, search_roots=()):
    """Reserve a number without creating the eventual exclusive output directory.

    Only files visible in these roots can provide history. An empty fresh session
    starts at one. Persistent reservation directories tolerate concurrent callers;
    failed runs can leave gaps. Nothing existing is deleted or overwritten.
    """
    if prefix not in ("llm2-campaign", "llm2-review"):
        raise ValueError("unknown numbered output prefix")
    if number is not None and (type(number) is not int or number < 1):
        raise ValueError("run_number deve ser um inteiro positivo ou omitido para numeração automática")
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    registry = parent / ".llm2-numbering" / prefix
    registry.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}-([0-9]+)(?:-results\.zip)?$")
    used = set()

    def observe(paths):
        for path in paths:
            match = pattern.fullmatch(path.name)
            if match:
                used.add(int(match[1]))

    observe(parent.glob(f"{prefix}-*"))
    for root in search_roots:
        root = Path(root)
        if root.is_dir():
            observe(root.rglob(f"{prefix}-*"))
    used.update(int(p.name) for p in registry.iterdir() if p.name.isdecimal())
    if prefix == "llm2-campaign":
        # A stopped bootstrap may have created only its input folder.
        for path in parent.glob("llm2-input-*"):
            suffix = path.name.removeprefix("llm2-input-")
            if suffix.isdecimal():
                used.add(int(suffix))
    candidate = number if number is not None else max(used, default=0) + 1
    while True:
        output = parent / f"{prefix}-{candidate:04d}"
        occupied = candidate in used or output.exists() or output.with_name(output.name+"-results.zip").exists()
        if prefix == "llm2-campaign":
            occupied |= (parent / f"llm2-input-{candidate:04d}").exists()
        if not occupied:
            try:
                (registry / f"{candidate:04d}").mkdir()
            except FileExistsError:
                occupied = True
        if not occupied:
            return output
        if number is not None:
            raise FileExistsError(f"O número {candidate:04d} já foi usado ou reservado. Escolha outro run_number.")
        candidate += 1
