"""Pinned WikiText-103 raw importer. Whole articles, official splits, no silent filtering."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen

from ..data import sha256_file, write_json
from ..document_data import prepare_documents


REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
# Audited exact duplicate: The Hustler (film), also present in the official test split.
# Only this documented training exclusion is permitted; unknown leakage still fails.
TRAIN_EXCLUSIONS = {"c10b5100b0181cc148fa30970a9acaac0780277653f3288ee4883ed1c85bb78f"}
FILES = {
    "train-00000-of-00002.parquet": "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212",
    "train-00001-of-00002.parquet": "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720",
    "validation-00000-of-00001.parquet": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    "test-00000-of-00001.parquet": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
}


def fetch_files(cache, *, download=False):
    cache = Path(cache)
    for name, digest in FILES.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"missing pinned shard: {path}; use --download explicitly")
            cache.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-103-raw-v1/{name}"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    with urlopen(url, timeout=60) as response:
                        shutil.copyfileobj(response, stream)
                if sha256_file(temporary) != digest:
                    raise ValueError(f"download checksum mismatch: {name}")
                # Exclusive destination: never replace a concurrently created cache file.
                with path.open("xb") as stream, temporary.open("rb") as source:
                    shutil.copyfileobj(source, stream)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if sha256_file(path) != digest:
            raise ValueError(f"pinned shard checksum mismatch: {name}")


def articles(lines):
    """Top-level '= Title =' headings delimit articles; paragraph/section boundaries do not."""
    current = []
    for text in lines:
        if not isinstance(text, str):
            raise ValueError("WikiText text column must contain strings")
        if re.fullmatch(r"= [^=].* =", text.strip()):
            if current:
                yield "".join(current)
            current = [text]
        elif current:
            current.append(text)
        elif text.strip():
            raise ValueError("nonempty WikiText preamble before first article")
    if current:
        yield "".join(current)


def parquet_lines(paths):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("WikiText import requires pyarrow; install with python -m pip install pyarrow") from exc
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
            yield from batch.column(0).to_pylist()


def import_wikitext(cache, output, *, train_bytes=32 * 1024**2, download=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    if type(train_bytes) is not int or train_bytes <= 0:
        raise ValueError("train_bytes must be a positive integer")
    fetch_files(cache, download=download)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        rows, metadata, owners = [], {}, {}
        # All articles are checked for leakage BEFORE training subset selection.
        for split, prefix in (("train", "train"), ("val", "validation"), ("test", "test")):
            docs = {}
            paths = [Path(cache) / name for name in sorted(FILES) if name.startswith(prefix + "-")]
            duplicates, excluded = 0, []
            for text in articles(parquet_lines(paths)):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if split == "train" and digest in TRAIN_EXCLUSIONS:
                    excluded.append(digest)
                    continue
                if digest in owners and owners[digest] != split:
                    raise ValueError("whole-article leakage across official WikiText splits")
                if digest in docs:
                    duplicates += 1
                owners[digest] = split
                docs[digest] = text
            selected, size = [], 0
            for digest in sorted(docs):
                if split == "train" and size >= train_bytes:
                    break
                text = docs[digest]
                selected.append(digest); size += len(text.encode("utf-8"))
                rows.append({"id": f"wikitext-103:{split}:{digest}", "split": split, "text": text})
            metadata[split] = {"available_articles": len(docs), "selected_articles": len(selected),
                               "selected_utf8_bytes": size, "duplicates_removed": duplicates,
                               "excluded_training_overlap_sha256": sorted(excluded)}
        raw = staging / "documents.jsonl"
        with raw.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = prepare_documents([raw], staging / "prepared", kind="byte")
        # Metadata is part of the immutable prepared-data identity, not an unhashed sidecar.
        manifest["corpus"] = {"name": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
                              "revision": REVISION, "shard_sha256": FILES,
                              "selection": "ascending SHA-256 of whole article; include final article intact",
                              "training_overlap_exclusions": sorted(TRAIN_EXCLUSIONS),
                              "requested_train_bytes": train_bytes, "splits": metadata}
        write_json(staging / "prepared" / "manifest.json", manifest)
        (staging / "prepared").rename(output)
    finally:
        shutil.rmtree(staging)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-bytes", type=int, default=32 * 1024**2)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    result = import_wikitext(args.cache, args.output, train_bytes=args.train_bytes, download=args.download)
    print(json.dumps(result["corpus"], indent=2))


if __name__ == "__main__":
    main()
