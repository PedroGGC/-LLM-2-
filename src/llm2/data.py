"""Offline document splits, train-only tokenization and resumable packed sampling."""

import hashlib
import importlib.metadata
import json
import random
import re
import shutil
import tempfile
import platform
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from . import __version__


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Hash-bearing artifacts must have identical bytes on Windows and Linux.
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


class TextTokenizer:
    """Byte baseline (257 symbols) or byte-level BPE, both lossless for UTF-8 text."""

    def __init__(self, kind="byte", engine=None):
        if kind not in ("byte", "bpe") or (kind == "bpe" and engine is None):
            raise ValueError("invalid tokenizer")
        self.kind = kind
        self.engine = engine
        if engine is not None:
            # Literal reserved-token text is encoded as ordinary bytes; EOS is appended explicitly.
            engine.encode_special_tokens = True
        self.eos_id = 256 if kind == "byte" else engine.token_to_id("<|endoftext|>")
        self.vocab_size = 257 if kind == "byte" else engine.get_vocab_size()

    @classmethod
    def train(cls, documents, vocab_size=8192):
        if vocab_size < 257:
            raise ValueError("byte-level BPE vocabulary must be at least 257")
        engine = Tokenizer(models.BPE())
        engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        engine.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, show_progress=False,
            special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        engine.train_from_iterator(documents, trainer=trainer)
        return cls("bpe", engine)

    def encode(self, text):
        return list(text.encode("utf-8")) if self.kind == "byte" else self.engine.encode(text).ids

    def decode(self, ids):
        if self.kind == "byte":
            return bytes(i for i in ids if i != self.eos_id).decode("utf-8", errors="replace")
        return self.engine.decode(list(ids), skip_special_tokens=True)

    def save(self, path):
        write_json(path, {"kind": self.kind, "engine": None if self.engine is None
                          else json.loads(self.engine.to_str())})

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        engine = None if raw["kind"] == "byte" else Tokenizer.from_str(json.dumps(raw["engine"]))
        return cls(raw["kind"], engine)


def prepare_data(inputs, output, *, kind="bpe", vocab_size=8192, val_fraction=0.1, seed=42):
    """One document per blank-line-separated paragraph. Exact duplicates removed globally."""
    if not 0 < val_fraction < 1 or seed < 0:
        raise ValueError("require 0 < val_fraction < 1 and a nonnegative seed")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {output}")
    documents, sources = {}, []
    for path in sorted({Path(p).resolve() for p in inputs}):
        text = path.read_text(encoding="utf-8")
        sources.append({"name": path.name, "sha256": sha256_file(path)})
        # Whitespace-only blank lines are document separators as well.
        for document in re.split(r"\n\s*\n", text):
            document = document.strip()
            if document:
                digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
                documents[digest] = document
    if len(documents) < 2:
        raise ValueError("at least two distinct documents required for an isolated split")
    keys = sorted(documents)
    random.Random(seed).shuffle(keys)
    n_val = min(len(keys) - 1, max(1, int(len(keys) * val_fraction)))
    split_keys = {"train": keys[n_val:], "val": keys[:n_val]}
    train_docs = [documents[k] for k in split_keys["train"]]
    if kind not in ("byte", "bpe"):
        raise ValueError("kind must be byte or bpe")
    tokenizer = TextTokenizer() if kind == "byte" else TextTokenizer.train(train_docs, vocab_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tokenizer.save(staging / "tokenizer.json")
        manifest = {
            "schema_version": 1, "seed": seed, "val_fraction": val_fraction,
            "preparation": {"llm2_version": __version__, "python": platform.python_version(),
                            "numpy": np.__version__, "tokenizers": importlib.metadata.version("tokenizers"),
                            "code_sha256": sha256_file(__file__)},
            "sources": sources, "document_policy": "blank-line paragraphs; strip; exact dedup before split",
            "packing_policy": "concatenated documents + EOS; cross-document causal attention; no padding",
            "dtype": "<u4", "tokenizer_kind": kind, "vocab_size": tokenizer.vocab_size,
            "requested_vocab_size": vocab_size if kind == "bpe" else 257,
            "eos_id": tokenizer.eos_id, "splits": {}, "files": {},
        }
        for split, doc_keys in split_keys.items():
            count = 0
            with (staging / f"{split}.bin").open("wb") as stream:
                for key in doc_keys:
                    ids = tokenizer.encode(documents[key]) + [tokenizer.eos_id]
                    stream.write(np.asarray(ids, dtype="<u4").tobytes())
                    count += len(ids)
            manifest["splits"][split] = {"tokens": count, "document_hashes": doc_keys}
        for name in ("train.bin", "val.bin", "tokenizer.json"):
            manifest["files"][name] = sha256_file(staging / name)
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


class PreparedData:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.fingerprint = sha256_file(self.root / "manifest.json")
        m = self.manifest
        if type(m["schema_version"]) is not int or m["schema_version"] not in (1, 2) or m["dtype"] != "<u4":
            raise ValueError("unsupported dataset schema")
        splits = ("train", "val") if m["schema_version"] == 1 else ("train", "val", "test")
        expected = {"tokenizer.json", *(f"{split}.bin" for split in splits)}
        if set(m["splits"]) != set(splits):
            raise ValueError("manifest must contain exactly the required splits")
        if set(m["files"]) != expected:
            raise ValueError("manifest must contain exactly the required artifacts")
        for name in sorted(expected):
            if sha256_file(self.root / name) != m["files"][name]:
                raise ValueError(f"dataset checksum mismatch: {name}")
        seen = set()
        for split in splits:
            hashes = m["splits"][split]["document_hashes"]
            if len(hashes) != len(set(hashes)) or seen.intersection(hashes):
                raise ValueError("document leakage or duplicate hashes across splits")
            if m["schema_version"] == 2 and (not hashes or any(
                    not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None for h in hashes)):
                raise ValueError("invalid document hashes")
            seen.update(hashes)
        self.tokenizer = TextTokenizer.load(self.root / "tokenizer.json")
        if self.tokenizer.vocab_size != m["vocab_size"] or self.tokenizer.eos_id != m["eos_id"]:
            raise ValueError("tokenizer metadata mismatch")
        self.arrays = {}
        for split in splits:
            path = self.root / f"{split}.bin"
            if path.stat().st_size != m["splits"][split]["tokens"] * 4 or path.stat().st_size == 0:
                raise ValueError(f"invalid token file size: {split}")
            tokens = np.memmap(path, dtype="<u4", mode="r")
            if int(tokens.max()) >= m["vocab_size"]:
                raise ValueError("token out of vocabulary")
            self.arrays[split] = tokens


class RandomBatchStream:
    """Uniform random contiguous windows, with replacement, using an independent CPU RNG."""

    def __init__(self, tokens, seq_len, batch_size, seed):
        if seq_len <= 0 or batch_size <= 0 or len(tokens) <= seq_len:
            raise ValueError("split requires at least seq_len + 1 tokens and positive dimensions")
        self.tokens, self.seq_len, self.batch_size = tokens, seq_len, batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.batches = 0

    def next(self, device="cpu"):
        starts = torch.randint(len(self.tokens) - self.seq_len, (self.batch_size,), generator=self.generator)
        windows = np.stack([self.tokens[int(i):int(i) + self.seq_len + 1] for i in starts])
        packed = torch.from_numpy(windows.astype(np.int64))
        self.batches += 1
        return {"input_ids": packed[:, :-1].to(device), "targets": packed[:, 1:].to(device)}

    def state_dict(self):
        return {"rng": self.generator.get_state(), "batches": self.batches,
                "seq_len": self.seq_len, "batch_size": self.batch_size, "length": len(self.tokens)}

    def load_state_dict(self, state):
        if any(state[k] != self.state_dict()[k] for k in ("seq_len", "batch_size", "length")):
            raise ValueError("incompatible sampler shape")
        self.generator.set_state(state["rng"].cpu())
        self.batches = state["batches"]


def validation_batches(tokens, seq_len, batch_size, max_batches=None, device="cpu"):
    """Fixed sequential windows, every full target window once; incomplete tail excluded."""
    windows = (len(tokens) - 1) // seq_len
    if windows < 1:
        raise ValueError("validation split requires at least seq_len + 1 tokens")
    stop = windows if max_batches is None else min(windows, max_batches * batch_size)
    for start in range(0, stop, batch_size):
        packed = np.stack([tokens[i * seq_len:(i + 1) * seq_len + 1]
                           for i in range(start, min(start + batch_size, stop))]).astype(np.int64)
        packed = torch.from_numpy(packed)
        yield {"input_ids": packed[:, :-1].to(device), "targets": packed[:, 1:].to(device)}

