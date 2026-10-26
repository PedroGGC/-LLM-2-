"""Create a small source archive to upload from the Colab notebook (no datasets/weights)."""

import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new ZIP path; existing archives are never overwritten")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output).resolve() if args.output else root / "dist" / "llm2-lab-colab.zip"
    if destination.exists():
        parser.error("refusing to overwrite an existing archive; choose a new --output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = [p for folder in ("src", "tests", "configs", "scripts", "notebooks", "docs", "requirements")
             for p in (root / folder).rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and ".egg-info" not in str(p)]
    paths += [root / name for name in ("pyproject.toml", "README.md", "AGENTS.md", "PLAN.md", "VALIDATION.md")
              if (root / name).exists()]
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, arcname=path.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    main()

