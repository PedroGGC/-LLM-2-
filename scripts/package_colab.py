"""Create a small source archive to upload from the Colab notebook (no datasets/weights)."""

from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "dist" / "llm2-lab-colab.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = [p for folder in ("src", "tests", "configs", "scripts", "notebooks", "docs", "requirements")
             for p in (root / folder).rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and ".egg-info" not in str(p)]
    paths += [root / name for name in ("pyproject.toml", "README.md", "AGENTS.md", "PLAN.md", "VALIDATION.md")
              if (root / name).exists()]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, arcname=path.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    main()

