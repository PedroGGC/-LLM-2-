"""Deterministic Google Colab GPU runner for LLM2-Lab (MCP + Notebook compatible)."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


REPO_URL = "https://github.com/PedroGGC/-LLM-2-.git"
DEFAULT_BRANCH = "llm-ideas"


def checkout_repo(work_root: Path, branch: str = DEFAULT_BRANCH, revision: str | None = None) -> Path:
    work_root.mkdir(parents=True, exist_ok=True)
    suffix = revision[:12] if revision else branch.replace("/", "-")
    repo = work_root / f"llm2-{suffix}"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--single-branch", "--branch", branch, REPO_URL, str(repo)],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", str(repo), "fetch", "origin", branch], check=True)
        if not revision:
            subprocess.run(["git", "-C", str(repo), "checkout", branch], check=True)
            subprocess.run(["git", "-C", str(repo), "reset", "--hard", f"origin/{branch}"], check=True)
    if revision:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a 40-character hexadecimal git commit SHA")
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", revision], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision and head != revision:
        raise RuntimeError(f"checkout head {head} does not match requested revision {revision}")
    return repo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="/content/llm2_colab")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-repo", default=None, help="use an already checked-out local repository")
    parser.add_argument("--run-tests", action="store_true", help="run pytest and pytest -m cuda")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--mode", choices=("separate", "unified"), default="unified")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--tokenizer", choices=("byte", "bpe"), default="bpe")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--seeds", nargs="*", type=int, default=[351])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--story-seq-len", type=int, default=512)
    parser.add_argument("--binding-hard-writes", type=int, default=64)
    parser.add_argument("--unified-curriculum", action="store_true")
    parser.add_argument("--deadline-seconds", type=float, default=5400.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    import torch

    print("=" * 78)
    print("LLM2-Lab — Google Colab GPU Runner")
    print("=" * 78)
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"GPU {idx}: {props.name} ({props.total_memory / (1024**3):.2f} GiB, bf16={torch.cuda.is_bf16_supported()})")
    else:
        print("WARNING: CUDA not available; running on CPU")

    work_root = Path(args.work_root).resolve()
    if args.local_repo:
        repo = Path(args.local_repo).resolve()
    else:
        repo = checkout_repo(work_root, branch=args.branch, revision=args.revision)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", f"{repo}[dev,research]"], check=True)

    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    print(f"Repository: {repo} @ {head}")

    if args.run_tests:
        print("\n--- Running pytest suite ---")
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(repo), check=True)
        if torch.cuda.is_available():
            print("\n--- Running CUDA test suite (-m cuda) ---")
            subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "cuda"], cwd=str(repo), check=True)

    if args.arms:
        out_dir = Path(args.output_dir).resolve() if args.output_dir else (work_root / f"campaign_{head[:8]}")
        data_cache = work_root / "stories_cache"
        cmd = [
            sys.executable,
            "-m",
            "llm2.research.ia2_benchmark",
            "--output-dir",
            str(out_dir),
            "--data-cache",
            str(data_cache),
            "--mode",
            args.mode,
            "--precision",
            args.precision,
            "--tokenizer",
            args.tokenizer,
            "--vocab-size",
            str(args.vocab_size),
            "--story-seq-len",
            str(args.story_seq_len),
            "--binding-hard-writes",
            str(args.binding_hard_writes),
            "--deadline-seconds",
            str(args.deadline_seconds),
            "--arms",
            *args.arms,
            "--seeds",
            *map(str, args.seeds),
        ]
        if args.lr is not None:
            cmd += ["--lr", str(args.lr)]
        if args.steps is not None:
            cmd += ["--steps", str(args.steps)]
        if args.unified_curriculum:
            cmd.append("--unified-curriculum")
        if args.smoke:
            cmd.append("--smoke")
        print("\n--- Launching IA2 Benchmark ---")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(repo), check=True)
        summary_path = out_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print("\n=== SUMMARY JSON ===")
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
