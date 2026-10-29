import hashlib
import json

import numpy as np
import pytest

from llm2.data import PreparedData, TextTokenizer, sha256_file
from llm2.document_data import prepare_documents
from llm2.cli import main, parser
from llm2.trainer import Trainer
from test_training import training_config


def documents(path, extra=()):
    rows = [{"id": split, "split": split, "text": f"  {split} article\n\nsecond paragraph Ω 🧪\n" * 30}
            for split in ("train", "val", "test")]
    rows += list(extra)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


@pytest.mark.parametrize("kind", ["byte", "bpe"])
def test_whole_documents_train_only_tokenizer_and_reproducibility(tmp_path, kind):
    raw = tmp_path / "documents.jsonl"; rows = documents(raw)
    a = prepare_documents([raw], tmp_path / "a", kind=kind, vocab_size=300)
    b = prepare_documents([raw], tmp_path / "b", kind=kind, vocab_size=300)
    assert a == b and a["schema_version"] == 2
    data = PreparedData(tmp_path / "a")
    assert set(data.arrays) == {"train", "val", "test"}
    for row in rows:
        split = row["split"]
        assert data.tokenizer.decode(data.arrays[split].tolist()) == row["text"]
        assert a["splits"][split]["document_hashes"] == [hashlib.sha256(row["text"].encode()).hexdigest()]
    if kind == "bpe":
        TextTokenizer.train([rows[0]["text"]], 300).save(tmp_path / "expected.json")
        assert sha256_file(tmp_path / "expected.json") == a["files"]["tokenizer.json"]
    with pytest.raises(FileExistsError):
        prepare_documents([raw], tmp_path / "a")


def test_explicit_cross_split_duplicates_rejected_and_same_split_deduped(tmp_path):
    raw = tmp_path / "docs.jsonl"; rows = documents(raw)
    documents(raw, [{"id": "copy", "split": "train", "text": rows[0]["text"]}])
    m = prepare_documents([raw], tmp_path / "ok")
    assert m["duplicates_removed"] == 1
    documents(raw, [{"id": "leak", "split": "test", "text": rows[0]["text"]}])
    with pytest.raises(ValueError, match="leakage"):
        prepare_documents([raw], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("tamper", ["binary", "splits", "hashes", "schema", "files"])
def test_schema_two_integrity_not_relaxed(tmp_path, tamper):
    raw = tmp_path / "docs.jsonl"; documents(raw)
    folder = tmp_path / "data"; m = prepare_documents([raw], folder)
    if tamper == "binary":
        (folder / "test.bin").write_bytes(b"bad")
    else:
        if tamper == "splits": del m["splits"]["test"]
        if tamper == "hashes": m["splits"]["test"]["document_hashes"] = m["splits"]["train"]["document_hashes"]
        if tamper == "schema": m["schema_version"] = 99
        if tamper == "files": del m["files"]["test.bin"]
        (folder / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError): PreparedData(folder)


def test_cli_default_val_and_reserved_test_are_identified(training_config, tmp_path):
    raw = tmp_path / "docs.jsonl"; documents(raw)
    folder = tmp_path / "explicit"; prepare_documents([raw], folder)
    cfg = training_config; cfg.data.path = str(folder)
    with Trainer(cfg) as trainer:
        trainer.train_step(); checkpoint = trainer.save()
    for split in ("val", "test"):
        output = tmp_path / f"{split}.json"
        main(["evaluate", "--checkpoint", str(checkpoint), "--split", split, "--output", str(output)])
        report = json.loads(output.read_text())
        expected = (len(PreparedData(folder).arrays[split]) - 1) // cfg.data.seq_len * cfg.data.seq_len
        assert report["evaluation_split"] == split
        assert report["validation_tokens"] == expected
    assert parser().parse_args(["evaluate", "--checkpoint", "x"]).split == "val"


def test_schema_one_rejects_test_without_fallback(training_config):
    with Trainer(training_config) as trainer:
        trainer.train_step(); checkpoint = trainer.save()
    with pytest.raises(SystemExit):
        main(["evaluate", "--checkpoint", str(checkpoint), "--split", "test"])
