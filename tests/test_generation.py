import pytest
import torch

from llm2.generation import generate
from llm2.model import TransformerLM
from llm2.runtime import capture_rng


def test_generation_local_rng_context_and_vocabulary(tiny_config):
    model = TransformerLM(tiny_config).train()
    prompt = torch.tensor([[1, 2, 3]])
    before = capture_rng()["torch"]
    args = dict(max_seq_len=8, max_new_tokens=12, seed=9, top_k=5, vocab_size=16)
    a = generate(model, prompt, **args)
    b = generate(model, prompt, **args)
    assert torch.equal(a, b)
    assert torch.equal(a[:, :3], prompt)
    assert a.shape == (1, 15) and int(a.max()) < 16
    assert model.training
    assert torch.equal(before, capture_rng()["torch"])
    greedy = generate(model, prompt, max_seq_len=8, max_new_tokens=4, temperature=0)
    assert greedy.shape == (1, 7)
    with pytest.raises(ValueError):
        generate(model, prompt, max_seq_len=8, temperature=-1)


def test_generation_stops_on_eos(tiny_config):
    model = TransformerLM(tiny_config)
    for parameter in model.parameters():
        parameter.data.zero_()
    output = generate(model, torch.tensor([[1]]), max_seq_len=8, max_new_tokens=10,
                      temperature=0, eos_id=0)
    assert output.tolist() == [[1, 0]]

