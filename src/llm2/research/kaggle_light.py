"""Scoped ephemeral cleanup and storage inventory. Standalone, no pickle loading."""
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

CAMPAIGN = re.compile(r"llm2-campaign-[0-9]{4,}")
CHECKOUT = re.compile(r"llm2-cycle2-(?:git|code)-[A-Za-z0-9_-]+")
REMOTE = "https://github.com/PedroGGC/-LLM-2-.git"
PROTECTED = {".safetensors", ".pt", ".pth", ".ckpt"}


def root_directory(base):
    path = Path(base)
    if path.is_symlink():
        raise ValueError("working root cannot be a link")
    path = path.resolve()
    inputs = Path("/kaggle/input").resolve()
    if (not path.is_dir() or path.parent == path or path == Path.home().resolve()
            or path == inputs or path.is_relative_to(inputs)):
        raise ValueError("invalid working root")
    return path


def active_workers():
    found = []
    for process in Path("/proc").glob("[0-9]*"):
        if int(process.name) == os.getpid():
            continue
        try:
            args = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            cwd = str((process / "cwd").resolve())
        except OSError:
            continue
        if (re.search(r"(?:^|\s)(?:llm2(?:\.|\s)|\S*/scripts/run_(?:cycle|recurrent))", args)
                or re.search(r"/llm2-cycle2-(?:git|code)-[A-Za-z0-9_-]+(?:/|\s)", args + " " + cwd + "/")):
            found.append(int(process.name))
    return found


def tree_bytes(path):
    path = Path(path)
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    total = 0
    for folder, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(folder) / d).is_symlink()]
        for name in files:
            try:
                total += (Path(folder) / name).lstat().st_size
            except FileNotFoundError:
                pass  # Live compiler cache inventories are approximate.
    return total


def inventory(base="/kaggle/working"):
    base = root_directory(base)
    rows = [{"name": p.name, "bytes": tree_bytes(p)} for p in base.iterdir()]
    return sorted(rows, key=lambda r: r["bytes"], reverse=True)


def _safe_target(base, path):
    base = root_directory(base)
    path = Path(path)
    if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(base):
        raise ValueError("cleanup target is not an internal regular directory")
    relative = path.relative_to(base)
    for ancestor in (path, *path.parents):
        if ancestor == base:
            break
        if ancestor.is_symlink():
            raise ValueError("linked cleanup ancestor")
    if not relative.parts:
        raise ValueError("cannot remove working root")
    return base, path


def remove_cache(base, path, *, guard=active_workers):
    """Only recognized compiler trees; links inside are unlinked without traversal."""
    base, path = _safe_target(base, path)
    parts = path.relative_to(base).parts
    if not (len(parts) in (2, 3) and CAMPAIGN.fullmatch(parts[0])
            and parts[1] == "compiler-cache" and (len(parts) == 2 or re.fullmatch(r"gpu-[0-9]+", parts[2]))):
        raise ValueError("not an owned compiler cache")
    for folder, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(folder) / d).is_symlink()]
        if "checkpoints" in dirs or any(Path(n).suffix.lower() in PROTECTED for n in files):
            raise ValueError("scientific weights found in cache; preserve for inspection")
    if guard():
        raise RuntimeError("workers active; cleanup blocked")
    amount = tree_bytes(path)
    _safe_target(base, path)
    shutil.rmtree(path)
    return {"name": path.relative_to(base).as_posix(), "bytes": amount}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT,
                                   timeout=20, encoding="utf-8").strip()


def remove_checkout(base, parent, *, guard=active_workers):
    base, parent = _safe_target(base, parent)
    if parent.parent != base or not CHECKOUT.fullmatch(parent.name):
        raise ValueError("not a disposable project checkout")
    roots = [parent / n for n in ("source", "scientific-worker") if (parent / n).exists()]
    if not roots:
        raise ValueError("checkout absent; unknown directory preserved")
    for path in parent.iterdir():
        if path in roots:
            continue
        if not path.is_file() or path.name not in {"bootstrap.json", "bootstrap-state.json"}:
            raise ValueError("unknown file beside checkout; preserve")
    for root in roots:
        _safe_target(base, root)
        if root.is_symlink() or git(root, "remote", "get-url", "origin") != REMOTE:
            raise ValueError("foreign checkout")
        common = Path(git(root, "rev-parse", "--git-common-dir"))
        common = (root / common).resolve() if not common.is_absolute() else common.resolve()
        if not common.is_relative_to(parent):
            raise ValueError("Git metadata outside disposable checkout")
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("modified checkout preserved")
        published = False
        try:
            git(root, "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/experiment/fractal-lrm")
            published = True
        except subprocess.CalledProcessError:
            # Exact-SHA shallow fetches have FETCH_HEAD, no remote branch ref.
            try:
                published = git(root, "rev-parse", "HEAD") == git(root, "rev-parse", "FETCH_HEAD")
            except subprocess.CalledProcessError:
                pass
        if not published:
            raise ValueError("unpublished checkout commits preserved")
    for folder, dirs, files in os.walk(parent, followlinks=False):
        if any((Path(folder) / d).is_symlink() for d in dirs) or any((Path(folder) / n).is_symlink() for n in files):
            raise ValueError("linked checkout preserved")
        for name in files:
            p = Path(folder) / name
            if p.suffix.lower() in PROTECTED or p.suffix.lower() in {".bin", ".parquet"}:
                raise ValueError("scientific payload in checkout; preserve")
    # Ignored user files are not necessarily caches. Inspect them explicitly.
    for root in roots:
        ignored = git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z").split("\0")
        for name in filter(None, ignored):
            if not any(n in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
                       or n.endswith(".egg-info") for n in Path(name).parts):
                raise ValueError("unknown ignored file in checkout")
    if guard():
        raise RuntimeError("workers active; cleanup blocked")
    amount = tree_bytes(parent)
    _safe_target(base, parent)
    def readonly(function, path, error):
        target = Path(path)
        if (os.name != "nt" or not isinstance(error[1], PermissionError) or target.is_symlink()
                or not target.resolve().is_relative_to(parent)):
            raise error[1]
        target.chmod(target.stat().st_mode | stat.S_IWRITE)
        function(path)
    shutil.rmtree(parent, onerror=readonly)
    return {"name": parent.name, "bytes": amount}


def clean_working(base="/kaggle/working", *, apply=True, guard=active_workers):
    base = root_directory(base)
    if guard():
        raise RuntimeError("workers ativos; nenhum arquivo removido")
    before = inventory(base)
    print("Consumo antes da limpeza:", flush=True)
    for row in before[:20]:
        print(f"  {row['bytes'] / 1024**3:.3f} GiB  {row['name']}", flush=True)
    removed, preserved = [], []
    for parent in sorted(base.iterdir()):
        target = parent / "compiler-cache" if CAMPAIGN.fullmatch(parent.name) else parent
        action = remove_cache if CAMPAIGN.fullmatch(parent.name) else remove_checkout
        if parent.is_symlink() or not (CAMPAIGN.fullmatch(parent.name) or CHECKOUT.fullmatch(parent.name)) or not target.exists():
            continue
        if not apply:
            preserved.append({"name": str(target), "reason": "inventory only"})
            continue
        try:
            removed.append(action(base, target, guard=guard))
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            preserved.append({"name": str(target), "reason": str(exc)})
    receipt = {"deleted": removed, "preserved": preserved, "freed_file_bytes": sum(r["bytes"] for r in removed),
               "before": before, "after": inventory(base), "scope": "compiler caches and verified clean code only; scientific files preserved"}
    print(f"Liberados {receipt['freed_file_bytes'] / 1024**3:.3f} GiB em arquivos temporários.", flush=True)
    if preserved:
        print("Preservados por segurança:", json.dumps(preserved, ensure_ascii=False), flush=True)
    return receipt
