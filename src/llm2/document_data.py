"""Whole documents with explicit splits; no re-splitting or validation-trained tokenizer."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import __version__
from .data import TextTokenizer, sha256_file, write_json


def prepare_documents(inputs, output, *, kind="byte", vocab_size=8192):
    """JSONL rows: {id, text, split: train|val|test}; preserve text bytes exactly."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {output}")
    if kind not in ("byte", "bpe"):
        raise ValueError("kind must be byte or bpe")
    documents = {split: {} for split in ("train", "val", "test")}
    owners, identifiers, sources = {}, set(), []
    duplicates = 0
    for path in sorted({Path(p).resolve() for p in inputs}):
        sources.append({"name": path.name, "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"id", "text", "split"}:
                    raise ValueError(f"expected id/text/split at {path.name}:{line_number}")
                identifier, text, split = row["id"], row["text"], row["split"]
                if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                        or not isinstance(text, str) or not text.strip() or split not in documents):
                    raise ValueError("invalid/duplicate document id, text or explicit split")
                identifiers.add(identifier)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in owners and owners[digest] != split:
                    raise ValueError("document leakage between explicit splits")
                if digest in owners:
                    duplicates += 1
                    continue
                owners[digest] = split
                documents[split][digest] = (identifier, text)
    if any(not docs for docs in documents.values()):
        raise ValueError("nonempty train, val and test document splits required")
    tokenizer = TextTokenizer() if kind == "byte" else TextTokenizer.train(
        (documents["train"][h][1] for h in sorted(documents["train"])), vocab_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tokenizer.save(staging / "tokenizer.json")
        manifest = {
            "schema_version": 2, "dtype": "<u4", "sources": sources,
            "preparation": {"llm2_version": __version__, "python": platform.python_version(),
                            "numpy": np.__version__, "tokenizers": importlib.metadata.version("tokenizers"),
                            "code_sha256": sha256_file(__file__)},
            "document_policy": "whole JSONL text unchanged; exact dedup within split; cross-split duplicates rejected",
            "packing_policy": "concatenated documents + EOS; cross-document causal attention; no padding",
            "tokenizer_kind": kind, "vocab_size": tokenizer.vocab_size, "eos_id": tokenizer.eos_id,
            "requested_vocab_size": vocab_size if kind == "bpe" else 257,
            "duplicates_removed": duplicates, "splits": {}, "files": {},
        }
        for split, docs in documents.items():
            count, raw_bytes, ids = 0, 0, []
            with (staging / f"{split}.bin").open("wb") as stream:
                for digest in sorted(docs):
                    identifier, text = docs[digest]
                    tokens = tokenizer.encode(text) + [tokenizer.eos_id]
                    stream.write(np.asarray(tokens, dtype="<u4").tobytes())
                    count += len(tokens)
                    raw_bytes += len(text.encode("utf-8"))
                    ids.append(identifier)
            manifest["splits"][split] = {"tokens": count, "raw_utf8_bytes": raw_bytes,
                                         "document_hashes": sorted(docs), "document_ids": ids}
        manifest["files"] = {name: sha256_file(staging / name)
                             for name in ("train.bin", "val.bin", "test.bin", "tokenizer.json")}
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest
