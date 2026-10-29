"""Create a small source archive to upload from the Colab notebook (no datasets/weights)."""

import argparse
from pathlib import Path
import zipfile


SOURCE_SUFFIXES = {
    "src": {".py"},
    "tests": {".py"},
    "configs": {".yaml", ".yml", ".md"},
    "scripts": {".py"},
    "notebooks": {".ipynb"},
    "docs": {".md", ".json", ".csv", ".svg", ".png"},
    "requirements": {".txt"},
}
ROOT_FILES = ("pyproject.toml", "README.md", "AGENTS.md", "PLAN.md", "VALIDATION.md", ".gitignore")


def source_files(root):
    """Select source/documentation only, even when local outputs live beside them."""
    paths = []
    for folder, suffixes in SOURCE_SUFFIXES.items():
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink() and path.suffix in suffixes
                    and not any(part.startswith(".") or part == "__pycache__"
                                or part.endswith(".egg-info") for part in relative.parts)
                    and path.resolve().is_relative_to(root.resolve())):
                paths.append(path)
    paths.extend(root / name for name in ROOT_FILES
                 if (root / name).is_file() and not (root / name).is_symlink())
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new ZIP path; existing archives are never overwritten")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output).resolve() if args.output else root / "dist" / "llm2-lab-colab.zip"
    if destination.exists():
        parser.error("refusing to overwrite an existing archive; choose a new --output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(root):
            archive.write(path, arcname=path.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    main()

