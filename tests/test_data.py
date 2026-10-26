import json
import random

import numpy as np
import pytest
import torch

from llm2.data import (PreparedData, RandomBatchStream, TextTokenizer, prepare_data,
                       sha256_file, validation_batches)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("\n\n".join(f"Documento {i}: ciência, memória e atenção. " * 4 for i in range(20)),
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", ["byte", "bpe"])
def test_reproducible_train_only_tokenizer(corpus, tmp_path, kind):
    for name in ("a", "b"):
        prepare_data([corpus], tmp_path / name, kind=kind, vocab_size=300, seed=9)
    for name in ("manifest.json", "tokenizer.json", "train.bin", "val.bin"):
        assert sha256_file(tmp_path / "a" / name) == sha256_file(tmp_path / "b" / name)
    data = PreparedData(tmp_path / "a")
    train = data.manifest["splits"]["train"]["document_hashes"]
    val = data.manifest["splits"]["val"]["document_hashes"]
    assert not set(train) & set(val)
    assert len(train) + len(val) == 20
    if kind == "bpe":
        import hashlib
        docs = [d.strip() for d in corpus.read_text(encoding="utf-8").split("\n\n")]
        by_hash = {hashlib.sha256(d.encode()).hexdigest(): d for d in docs}
        trained = TextTokenizer.train([by_hash[k] for k in train], 300)
        trained.save(tmp_path / "independent.json")
        assert sha256_file(tmp_path / "independent.json") == sha256_file(tmp_path / "a/tokenizer.json")
    text = "Olá, ciência! 漢字 🧪\n  espaços <|endoftext|> literal"
    assert data.tokenizer.decode(data.tokenizer.encode(text)) == text
    with pytest.raises(FileExistsError):
        prepare_data([corpus], tmp_path / "a")


def test_sampler_state_and_independent_rng():
    tokens = np.arange(100, dtype=np.uint32)
    stream = RandomBatchStream(tokens, 8, 3, seed=12)
    batch = stream.next()
    torch.testing.assert_close(batch["targets"][:, :-1], batch["input_ids"][:, 1:])
    state = stream.state_dict()
    expected = stream.next()
    random.seed(100)
    torch.manual_seed(100)
    resumed = RandomBatchStream(tokens, 8, 3, seed=999)
    resumed.load_state_dict(state)
    actual = resumed.next()
    for key in actual:
        assert torch.equal(actual[key], expected[key])
    assert resumed.batches == stream.batches == 2
    with pytest.raises(ValueError):
        RandomBatchStream(tokens, 9, 3, 12).load_state_dict(state)


def test_validation_fixed_windows():
    batches = list(validation_batches(np.arange(29), 4, 3))
    assert [b["targets"].numel() for b in batches] == [12, 12, 4]
    assert torch.equal(torch.cat([b["targets"].flatten() for b in batches]), torch.arange(1, 29))


def test_tampering_rejected(corpus, tmp_path):
    prepare_data([corpus], tmp_path / "data", kind="byte")
    with (tmp_path / "data/train.bin").open("ab") as stream:
        stream.write(b"0000")
    with pytest.raises(ValueError, match="checksum"):
        PreparedData(tmp_path / "data")


def test_dedup_before_split(tmp_path):
    corpus = tmp_path / "duplicates.txt"
    corpus.write_text("same document\n\nsame document\n\nanother one", encoding="utf-8")
    m = prepare_data([corpus], tmp_path / "data", kind="byte")
    assert sum(len(s["document_hashes"]) for s in m["splits"].values()) == 2
    corpus.write_text("only one", encoding="utf-8")
    with pytest.raises(ValueError, match="two distinct"):
        prepare_data([corpus], tmp_path / "bad")

