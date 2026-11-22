"""Real web corpus (FineWeb-Edu) streaming, SHA-256 deduplication, document split, and train-only BPE preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from .stories_data import StoriesData, write_bpe_dataset


def _iter_fineweb_texts(repo: str, config: str | None) -> Iterable[str]:
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    ds = load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)
    for row in ds:
        text = row.get("text")
        if isinstance(text, str):
            yield text


def partition_dedup_documents(
    texts: Iterable[str],
    *,
    train_bytes: int,
    val_bytes: int,
    cal_val_bytes: int,
    test_bytes: int,
    min_doc_bytes: int = 128,
) -> tuple[dict[str, dict[str, bytes]], dict]:
    """Deduplicate documents globally by SHA-256 and fill validation/test splits before train."""
    targets = (
        ("val", int(val_bytes)),
        ("cal_val", int(cal_val_bytes)),
        ("test", int(test_bytes)),
        ("train", int(train_bytes)),
    )
    splits: dict[str, dict[str, bytes]] = {name: {} for name, _ in targets}
    sizes = {name: 0 for name, _ in targets}
    seen: set[str] = set()
    duplicates = 0
    skipped_short = 0

    for text in texts:
        cleaned = text.strip()
        raw = cleaned.encode("utf-8", errors="ignore")
        if len(raw) < min_doc_bytes:
            skipped_short += 1
            continue
        key = hashlib.sha256(raw).hexdigest()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)

        placed = False
        for name, budget in targets:
            if sizes[name] < budget:
                splits[name][key] = raw
                sizes[name] += len(raw)
                placed = True
                break
        if not placed:
            break

    missing = [name for name, docs in splits.items() if not docs]
    if missing:
        raise ValueError(f"insufficient corpus documents to populate splits: {missing}")
    stats = {
        "split_bytes": sizes,
        "split_docs": {name: len(docs) for name, docs in splits.items()},
        "duplicates_dropped": duplicates,
        "short_dropped": skipped_short,
    }
    return splits, stats


def prepare_fineweb_edu(
    cache_dir: str | Path,
    *,
    train_bytes: int = 24 * 1024**2,
    val_bytes: int = 512 * 1024,
    cal_val_bytes: int = 512 * 1024,
    test_bytes: int = 512 * 1024,
    vocab_size: int = 8192,
    repo: str = "HuggingFaceFW/fineweb-edu",
    config: str | None = "sample-10BT",
    docs_iter: Iterable[str] | None = None,
) -> StoriesData:
    """Prepare or reuse an immutable train-only BPE dataset from FineWeb-Edu (or injected texts)."""
    root = Path(cache_dir).resolve()
    tag = hashlib.sha256(
        f"{repo}:{config}:{train_bytes}:{val_bytes}:{cal_val_bytes}:{test_bytes}:{vocab_size}".encode()
    ).hexdigest()[:12]
    out_dir = root / f"fineweb-bpe-v{int(vocab_size)}-{tag}"
    if (out_dir / "manifest.json").exists():
        return StoriesData(out_dir, require_pinned=False)

    stream = _iter_fineweb_texts(repo, config) if docs_iter is None else docs_iter
    splits, stats = partition_dedup_documents(
        stream,
        train_bytes=train_bytes,
        val_bytes=val_bytes,
        cal_val_bytes=cal_val_bytes,
        test_bytes=test_bytes,
    )
    spec = {
        "name": "fineweb-edu-bpe-doc-v1",
        "repo": repo,
        "config": config,
        "train_bytes": int(train_bytes),
        "val_bytes": int(val_bytes),
        "cal_val_bytes": int(cal_val_bytes),
        "test_bytes": int(test_bytes),
        "policy": "global SHA-256 dedup; val/cal_val/test filled before train; train-only BPE",
    }
    return write_bpe_dataset(
        out_dir,
        splits,
        spec=spec,
        source_fingerprint=tag,
        selection=stats,
        vocab_size=vocab_size,
    )
