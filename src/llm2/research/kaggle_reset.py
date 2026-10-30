"""Standalone stdlib backup and externally verified reset of Kaggle working."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile

KIND = "llm2-working-backup-v1"
EPHEMERAL = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             ".virtual_documents", ".ipynb_checkpoints"}
WEIGHTS = {".pt", ".pth", ".ckpt", ".safetensors"}


def sha256_file(path):
    h = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def active_workers():
    found = []
    for process in Path("/proc").glob("[0-9]*"):
        if int(process.name) == os.getpid(): continue
        try: args = (process/"cmdline").read_bytes().split(b"\0")
        except OSError: continue
        words = [a.decode("utf-8", "replace") for a in args]
        if any(w == "llm2" or w.startswith("llm2.") or "scripts/run_cycle" in w
               or re.search(r"/llm2-cycle[234]-(?:git|code)-", w)
               or re.search(r"/llm2-[0-9a-f]{12}(?:/|$)", w) for w in words):
            found.append(int(process.name))
    return found


def root_directory(base):
    path = Path(base)
    if path.is_symlink(): raise ValueError("working root cannot be a symlink")
    path = path.resolve()
    if not path.is_dir() or path.parent == path or path == Path.home().resolve():
        raise ValueError("invalid working root")
    inputs = Path("/kaggle/input").resolve()
    if path == inputs or path.is_relative_to(inputs): raise ValueError("input is read-only data, never a cleanup target")
    return path


def inventory(base, *, exclude=()):
    """Hashes regular files; links are recorded, never traversed."""
    base = root_directory(base); excluded = {Path(p).resolve() for p in exclude}
    records = {}
    def walk(directory):
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(base).as_posix()
            if path.name in EPHEMERAL or (not path.is_symlink() and path.resolve() in excluded): continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                records[relative] = {"kind": "symlink", "target": os.readlink(path)}
            elif stat.S_ISDIR(info.st_mode):
                if not path.resolve().is_relative_to(base): raise ValueError("external directory")
                records[relative] = {"kind": "directory"}; walk(path)
            elif stat.S_ISREG(info.st_mode):
                if not path.resolve().is_relative_to(base): raise ValueError("external file")
                digest = sha256_file(path); after = path.lstat()
                if path.is_symlink() or (info.st_size, info.st_mtime_ns, info.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError("file changed during inventory: "+relative)
                records[relative] = {"kind": "file", "bytes": info.st_size,
                                     "mtime_ns": info.st_mtime_ns, "sha256": digest}
            else: raise ValueError("unsupported special file: "+relative)
    walk(base)
    return records


def checkpoint_selection(base, records):
    groups = {}
    for name, record in records.items():
        p = Path(name)
        if record["kind"] == "directory" and p.parent.name == "checkpoints" and re.fullmatch(r"step-[0-9]+", p.name):
            groups.setdefault(p.parent.as_posix(), []).append(p)
    omitted, verified, warnings = [], [], []
    for parent, checkpoints in groups.items():
        checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
        valid = True; complete = []
        for checkpoint in checkpoints:
            try:
                metadata = base/checkpoint/"metadata.json"
                if records[(checkpoint/"metadata.json").as_posix()]["kind"] != "file":
                    raise ValueError("linked checkpoint metadata")
                if metadata.stat().st_size > 10*1024**2: raise ValueError("oversized metadata")
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                files = meta["files"]
                if not isinstance(files, dict) or not {"model.safetensors", "training_state.pt"} <= files.keys():
                    raise ValueError("unknown or incomplete training checkpoint layout")
                for filename, digest in files.items():
                    if Path(filename).name != filename or "\\" in filename: raise ValueError("external checkpoint file")
                    observed = records[(checkpoint/filename).as_posix()]
                    if observed["kind"] != "file" or observed["sha256"] != digest: raise ValueError("checkpoint hash mismatch")
                complete.append(checkpoint)
            except (OSError, ValueError, KeyError, TypeError) as error:
                valid = False
                warnings.append({"checkpoint_group": parent, "reason": str(error),
                                 "action": "preserve entire unverified group"})
        if complete:
            verified.append(complete[-1].as_posix())
        if valid:
            omitted.extend(p.as_posix() for p in checkpoints[:-1])
    files = [n for n, r in records.items() if r["kind"] == "file" and
             not any(n == old or n.startswith(old+"/") for old in omitted)]
    unclassified = [n for n in files if Path(n).suffix.lower() in WEIGHTS and
                    not any(n.startswith(checkpoint+"/") for checkpoint in verified)]
    return files, omitted, verified, unclassified, warnings


def next_run_number(records):
    values = [22]
    for name in records:
        first = name.split("/", 1)[0]
        match = re.fullmatch(r"llm2-campaign-([0-9]+)(?:-.*)?", first)
        if match: values.append(int(match[1]))
        match = re.fullmatch(r"\.llm2-numbering/llm2-campaign/([0-9]+)", name)
        if match: values.append(int(match[1]))
    return max(values)+1


def read_backup(path, *, expected_root=None, verify=True):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "backup_manifest.json" not in names:
            raise ValueError("invalid backup members")
        if archive.getinfo("backup_manifest.json").file_size > 100*1024**2:
            raise ValueError("oversized backup manifest")
        manifest = json.loads(archive.read("backup_manifest.json"))
        if manifest["kind"] != KIND or (expected_root is not None and manifest["root"] != str(expected_root)):
            raise ValueError("backup root/kind mismatch")
        if set(names) != {"backup_manifest.json", *("working/"+n for n in manifest["selected_files"])}:
            raise ValueError("backup member set differs from manifest")
        if verify:
            if archive.testzip() is not None: raise ValueError("backup CRC failed")
            for name in manifest["selected_files"]:
                if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                    raise ValueError("unsafe backup member")
                h = hashlib.sha256()
                with archive.open("working/"+name) as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
                if h.hexdigest() != manifest["inventory"][name]["sha256"]:
                    raise ValueError("backup content hash failed: "+name)
        return manifest


def backup_working(base="/kaggle/working"):
    base = root_directory(base)
    workers = active_workers()
    if workers: raise RuntimeError("Workers ativos; aguarde o fim antes do backup: "+str(workers))
    # Reuse a matching verified backup after a repeated cell or kernel restart.
    for previous in sorted(base.glob("llm2-working-backup-*.zip"), reverse=True):
        try: manifest = read_backup(previous, expected_root=base)
        except (ValueError, OSError, KeyError, zipfile.BadZipFile): continue
        if inventory(base, exclude=(previous,)) == manifest["inventory"]:
            return backup_receipt(previous, manifest, reused=True)
    records = inventory(base)
    files, omitted, latest, unclassified, warnings = checkpoint_selection(base, records)
    required = sum(records[n]["bytes"] for n in files) + 512*1024**2
    if shutil.disk_usage(base).free < required:
        raise OSError("Espaço insuficiente para backup; nenhum dado foi apagado. Necessário: "+str(required))
    manifest = {"kind": KIND, "root": str(base), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": records, "selected_files": files, "omitted_older_checkpoints": omitted,
        "verified_latest_checkpoints": latest, "unclassified_weights_preserved": unclassified,
        "warnings": warnings, "next_run_number": next_run_number(records),
        "symlinks": "definitions recorded in inventory; external targets not followed"}
    descriptor, name = tempfile.mkstemp(prefix=".llm2-backup-writing-", suffix=".zip", dir=base)
    os.close(descriptor); temporary = Path(name)
    destination = base/("llm2-working-backup-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+os.urandom(3).hex()+".zip")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr("backup_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for index, relative in enumerate(files, 1):
                path = base/relative
                if path.is_symlink() or not path.resolve().is_relative_to(base): raise ValueError("file changed to external link")
                archive.write(path, "working/"+relative)
                if index % 100 == 0: print(f"Backup: {index}/{len(files)} arquivos", flush=True)
        read_backup(temporary, expected_root=base)
        if active_workers() or inventory(base, exclude=(temporary,)) != records:
            raise RuntimeError("Working mudou durante o backup; nada foi apagado.")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return backup_receipt(destination, manifest)


def backup_receipt(path, manifest, *, reused=False):
    result = {"status": "backup_ready", "path": str(path), "sha256": sha256_file(path),
              "bytes": path.stat().st_size, "next_run_number": manifest["next_run_number"],
              "latest_checkpoints": len(manifest["verified_latest_checkpoints"]), "reused": reused,
              "warnings": manifest["warnings"], "unclassified_weights": manifest["unclassified_weights_preserved"]}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("Baixe o ZIP, calcule seu SHA-256 local e execute MODO='limpar' com BACKUP_LOCAL_SHA256.", flush=True)
    return result


def _remove_directory(path, base):
    resolved = path.resolve()
    if path.is_symlink() or resolved.parent != base: raise ValueError("external deletion target")
    def readonly(function, name, exception):
        candidate = Path(name)
        if os.name != "nt" or not isinstance(exception[1], PermissionError) or candidate.is_symlink() or not candidate.resolve().is_relative_to(resolved):
            raise exception[1]
        candidate.chmod(candidate.stat().st_mode | stat.S_IWRITE); function(name)
    shutil.rmtree(path, onerror=readonly)


def clear_working(local_sha256, base="/kaggle/working"):
    base = root_directory(base)
    if not isinstance(local_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", local_sha256.strip()):
        raise ValueError("Informe o SHA-256 calculado na cópia local baixada.")
    if active_workers(): raise RuntimeError("Worker ativo; limpeza bloqueada.")
    match = None
    for path in base.glob("llm2-working-backup-*.zip"):
        if not path.is_symlink() and sha256_file(path) == local_sha256.strip().lower():
            match = path; break
    if match is None: raise ValueError("SHA local não corresponde a nenhum backup no working.")
    manifest = read_backup(match, expected_root=base)
    if inventory(base, exclude=(match,)) != manifest["inventory"]:
        raise RuntimeError("Inventário mudou após o backup; gere um novo backup antes de limpar.")
    if active_workers(): raise RuntimeError("Worker iniciou após a verificação; limpeza bloqueada.")
    before = shutil.disk_usage(base)
    # The exact root stays. Input and symlink destinations are never recursive targets.
    for child in list(base.iterdir()):
        if child.parent != base: raise ValueError("invalid child")
        if child.is_symlink(): child.unlink()
        elif child.is_dir(): _remove_directory(child, base)
        elif child.is_file(): child.unlink()
        else: raise ValueError("unsupported cleanup target")
    after = shutil.disk_usage(base)
    receipt = {"status": "cleaned", "root": str(base), "backup_local_sha256": local_sha256.lower().strip(),
        "next_run_number": manifest["next_run_number"], "remaining_entries": len(list(base.iterdir())),
        "filesystem_free_gib": after.free/1024**3, "filesystem_free_delta_gib": (after.free-before.free)/1024**3}
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt
