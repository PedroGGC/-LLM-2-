"""Package logs/configs/reports without duplicating weights, optimizer or raw datasets."""

import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, destination = Path(args.run).resolve(), Path(args.output).resolve()
    if destination.exists():
        parser.error("refusing to overwrite an export")
    if not (root / "run.json").is_file() or destination.is_relative_to(root):
        parser.error("choose an existing run and an export outside that run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix in (".json", ".jsonl", ".yaml", ".txt", ".md", ".svg")
                                  or path.name == "source.zip" or path.name.startswith("events.")):
                archive.write(path, arcname=path.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    main()
