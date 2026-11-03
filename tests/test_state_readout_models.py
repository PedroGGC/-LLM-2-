import copy
import hashlib
import json
import sys
import types
import subprocess
from pathlib import Path

import pytest
import torch

from llm2.architectures.state_recurrent import StateRecurrentModel
from llm2.config import ResearchConfig
from llm2.objectives import CausalLanguageModeling
from llm2.trainer import Trainer
from llm2.research.state_readout_worker import ReadoutTrainer
from test_training import training_config,assert_nested_equal

ROOT=Path(__file__).resolve().parents[1]


def research(mode):
    return ResearchConfig(architecture="state_recurrent" if mode=="fine" else "state_hierarchy",
        share_scales=False,state_rule="mlstm",state_chunk_size=4,scale_conditioning=False,
        state_readout="immediate" if mode=="immediate" else "next")


@pytest.mark.parametrize("mode",["fine","next","immediate"])
def test_causality_streaming_gradients_and_parameters(tiny_config,mode):
    cfg=copy.deepcopy(tiny_config);cfg.n_layers=1 if mode=="fine" else 2
    model=StateRecurrentModel(cfg,research(mode)).eval()
    ids=torch.randint(0,cfg.vocab_size,(2,17))
    expected=model(ids)
    assert len(model.cells)==cfg.n_layers
    assert model.parameter_formula()==sum(p.numel() for p in model.parameters())
    assert model.embedding.weight is model.lm_head.weight
    for boundary in (1,3,4,5,7,8,9,16):
        changed=ids.clone();changed[:,boundary:]=(changed[:,boundary:]+1)%cfg.vocab_size
        torch.testing.assert_close(model(changed).logits[:,:boundary],expected.logits[:,:boundary],rtol=1e-5,atol=1e-6)
    a=copy.deepcopy(model);b=copy.deepcopy(model)
    a(ids).logits.square().mean().backward()
    state=None;parts=[]
    for piece in (ids[:,:3],ids[:,3:4],ids[:,4:7],ids[:,7:12],ids[:,12:]):
        out=b(piece,state=state);state=out.state;parts.append(out.logits)
    actual=torch.cat(parts,1)
    torch.testing.assert_close(actual,expected.logits,atol=2e-6,rtol=2e-5)
    actual.square().mean().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        assert pa.grad is not None and torch.isfinite(pa.grad).all()
        torch.testing.assert_close(pa.grad,pb.grad,atol=3e-6,rtol=4e-4)
    assert len(state.memories)==cfg.n_layers
    torch.testing.assert_close(model(ids).logits,expected.logits,rtol=0,atol=0)
    assert model.architecture_version==(1 if mode=="next" else 2)


def test_only_readout_index_differs_and_fourth_position_changes(tiny_config):
    torch.manual_seed(251);a=StateRecurrentModel(tiny_config,research("next")).eval()
    torch.manual_seed(251);b=StateRecurrentModel(tiny_config,research("immediate")).eval()
    assert_nested_equal(a.state_dict(),b.state_dict())
    ids=torch.randint(0,tiny_config.vocab_size,(2,17))
    old,new=a(ids).logits,b(ids).logits
    keep=torch.arange(17)%4!=3
    torch.testing.assert_close(old[:,keep],new[:,keep],atol=0,rtol=0)
    assert not torch.equal(old[:,3],new[:,3])


@pytest.mark.parametrize("sharing",[False,True])
def test_default_matches_frozen_historical_forward(tiny_config,sharing,monkeypatch):
    # The delivery is a shallow checkout: historical objects need not be present.
    monkeypatch.setattr(subprocess,"check_output",lambda *a,**k:pytest.fail("historical Git access in preflight"))
    payload=(ROOT/"tests/fixtures/state_recurrent_v1.py.txt").read_bytes()
    reference=json.loads((ROOT/"docs/results/campaign_0027_diagnostic_reference.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(payload).hexdigest()==reference["source_sha256"]["src/llm2/architectures/state_recurrent.py"]
    name="llm2.architectures._historical_state_readout_test"
    module=types.ModuleType(name);sys.modules[name]=module
    try:
        exec(compile(payload,name,"exec"),module.__dict__)
        r=research("next");r.share_scales=sharing
        torch.manual_seed(44);old=module.StateRecurrentModel(tiny_config,r)
        torch.manual_seed(44);new=StateRecurrentModel(tiny_config,r)
        ids=torch.randint(0,tiny_config.vocab_size,(2,17))
        assert_nested_equal(old.state_dict(),new.state_dict())
        torch.testing.assert_close(old(ids).logits,new(ids).logits,atol=0,rtol=0)
        assert old.parameter_formula()==new.parameter_formula() and old.state_bytes_per_sequence()==new.state_bytes_per_sequence()
    finally:sys.modules.pop(name,None)


@pytest.mark.parametrize("mode",["fine","immediate"])
def test_resume_dropout_accumulation_and_incompatible_readout(training_config,tmp_path,mode):
    cfg=training_config;cfg.model.n_layers=1 if mode=="fine" else 2;cfg.research=research(mode)
    assert cfg.model.dropout>0 and cfg.train.grad_accum_steps>1
    with ReadoutTrainer(cfg) as original:
        original.fit(until_step=2);cp=original.last_checkpoint
        metric=original.train_step()
        expected=copy.deepcopy((original.model.state_dict(),original.optimizer.state_dict(),original.scheduler.state_dict(),original.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/"resumed")
    with ReadoutTrainer(cfg,resume=cp) as resumed:
        actual=resumed.train_step()
        assert_nested_equal(expected,(resumed.model.state_dict(),resumed.optimizer.state_dict(),resumed.scheduler.state_dict(),resumed.sampler.state_dict()))
        assert actual["training_loss"]==metric["training_loss"]
    if mode=="immediate":
        cfg.research.state_readout="next";cfg.train.output_dir=str(tmp_path/"incompatible")
        with pytest.raises(ValueError):ReadoutTrainer(cfg,resume=cp)


@pytest.mark.parametrize("mode",["fine","immediate"])
def test_memorization_and_finiteness(tiny_config,mode):
    cfg=copy.deepcopy(tiny_config);cfg.n_layers=1 if mode=="fine" else 2
    model=StateRecurrentModel(cfg,research(mode));objective=CausalLanguageModeling()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    ids=torch.arange(12).repeat(2,1);batch={"input_ids":ids[:,:-1],"targets":ids[:,1:]}
    initial=objective(model,batch).loss.item()
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True);loss=objective(model,batch).loss
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    assert objective(model,batch).loss.item()<initial/15


def test_readout_config_validation():
    with pytest.raises(ValueError,match="next\\|immediate"):ResearchConfig(state_readout="bad")
    with pytest.raises(ValueError,match="requires state_hierarchy"):ResearchConfig(state_readout="immediate")


@pytest.mark.cuda
@pytest.mark.parametrize("mode",["fine","immediate"])
def test_cuda_inductor_new_controls(tiny_config,mode):
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable; run on T4 before training")
    cfg=copy.deepcopy(tiny_config);cfg.n_layers=1 if mode=="fine" else 2
    model=StateRecurrentModel(cfg,research(mode)).cuda()
    ids=torch.randint(0,cfg.vocab_size,(2,17),device="cuda")
    expected=model(ids).logits;expected.square().mean().backward()
    gradients=[p.grad.clone() for p in model.parameters()];model.zero_grad(set_to_none=True)
    actual=torch.compile(model,backend="inductor",dynamic=False)(ids).logits
    actual.square().mean().backward();torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-3)
    for p,g in zip(model.parameters(),gradients):torch.testing.assert_close(p.grad,g,atol=2e-5,rtol=3e-3)
