import torch
import pytest

from llm2.config import ModelConfig


@pytest.fixture(autouse=True)
def stable_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(17)


@pytest.fixture
def tiny_config():
    return ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                       hidden_dim=64, max_seq_len=32)

