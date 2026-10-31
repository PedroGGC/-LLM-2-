import copy
import pytest
import torch
from llm2.config import ResearchConfig
from llm2.architectures.state_kernels import memory_rule
from llm2.architectures.state_recurrent import StateRecurrentModel
from llm2.model import count_parameters
from llm2.objectives import CausalLanguageModeling
from llm2.research.accounting import estimate_research_flops
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal


@pytest.mark.parametrize("rule", ["gated_delta","rwkv7","mlstm"])
@pytest.mark.parametrize("length,chunk", [(1,4),(9,4),(17,8)])
def test_memory_rule_chunks_outputs_states_gradients(rule,length,chunk):
    torch.manual_seed(11)
    width = 4 if rule=="rwkv7" else 1
    tensors = [torch.randn(2,2,length,d)*.3 for d in (4,4,3,width,width)]
    versions = []
    for execution in ("sequential","chunked"):
        inputs = [x.clone().requires_grad_() for x in tensors]
        q,k,v,g1,g2 = inputs
        output,state = memory_rule(rule,q,k,v,(g1,g2),execution=execution,chunk_size=chunk)
        flat = state if isinstance(state,tuple) else (state,)
        (output.square().sum()+sum(s.square().sum()*.01 for s in flat)).backward()
        versions.append((output,flat,[x.grad for x in inputs]))
    for a,b in zip(versions[0],versions[1]):
        if isinstance(a,list) or isinstance(a,tuple):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-4)
        else:torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)


@pytest.mark.parametrize("rule", ["gated_delta","rwkv7","mlstm"])
@pytest.mark.parametrize("architecture,sharing", [("state_recurrent",False),("state_hierarchy",False),("state_hierarchy",True)])
def test_backbone_causality_backward_streaming_and_count(tiny_config,rule,architecture,sharing):
    r = ResearchConfig(architecture=architecture,state_rule=rule,share_scales=sharing,state_chunk_size=4,scale_conditioning=False)
    model = StateRecurrentModel(tiny_config,r)
    ids = torch.randint(0,32,(2,17)); model.eval()
    expected = model(ids)
    assert count_parameters(model)==model.parameter_formula()
    assert model.embedding.weight is model.lm_head.weight
    assert estimate_research_flops(model,CausalLanguageModeling(),17) is None
    for boundary in (1,3,4,5,8,16):
        changed=ids.clone();changed[:,boundary:]=(changed[:,boundary:]+1)%32
        torch.testing.assert_close(expected.logits[:,:boundary],model(changed).logits[:,:boundary],atol=1e-6,rtol=1e-5)
    state=None; outputs=[]
    for piece in (ids[:,:3],ids[:,3:8],ids[:,8:9],ids[:,9:]):
        result=model(piece,state=state);outputs.append(result.logits);state=result.state
    torch.testing.assert_close(torch.cat(outputs,1),expected.logits,atol=2e-6,rtol=2e-5)
    expected.logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # A different document starts with empty state; no hidden state is stored on modules.
    torch.testing.assert_close(model(ids).logits,expected.logits,rtol=0,atol=0)


@pytest.mark.parametrize("rule", ["gated_delta","rwkv7","mlstm"])
def test_hierarchy_shared_independent_initial_parity_and_safetensors(tiny_config,rule,tmp_path):
    from safetensors.torch import save_model,load_model
    torch.manual_seed(241)
    shared=StateRecurrentModel(tiny_config,ResearchConfig(architecture="state_hierarchy",state_rule=rule,scale_conditioning=False))
    torch.manual_seed(241)
    independent=StateRecurrentModel(tiny_config,ResearchConfig(architecture="state_hierarchy",state_rule=rule,share_scales=False,scale_conditioning=False))
    ids=torch.randint(0,32,(2,17))
    torch.testing.assert_close(shared(ids).logits,independent(ids).logits,rtol=0,atol=0)
    for module in (independent.embedding,):module.load_state_dict(shared.embedding.state_dict())
    independent.norm.load_state_dict(shared.norm.state_dict())
    for cell in independent.cells:cell.load_state_dict(shared.cells[0].state_dict())
    ids=torch.randint(0,32,(2,17))
    torch.testing.assert_close(shared(ids).logits,independent(ids).logits,rtol=0,atol=0)
    assert count_parameters(shared)<count_parameters(independent)
    save_model(shared,str(tmp_path/"weights.safetensors"))
    clone=copy.deepcopy(shared);load_model(clone,str(tmp_path/"weights.safetensors"))
    assert clone.lm_head.weight is clone.embedding.weight
    torch.testing.assert_close(clone(ids).logits,shared(ids).logits,atol=0,rtol=0)


@pytest.mark.parametrize("rule", ["gated_delta","rwkv7","mlstm"])
@pytest.mark.parametrize("architecture,sharing", [("state_recurrent",False),("state_hierarchy",False),("state_hierarchy",True)])
def test_state_exact_resume_dropout_accumulation(training_config,tmp_path,rule,architecture,sharing):
    cfg=training_config
    cfg.research=ResearchConfig(architecture=architecture,state_rule=rule,share_scales=sharing,state_chunk_size=4,scale_conditioning=False)
    with Trainer(cfg) as original:
        original.fit(until_step=2);checkpoint=original.last_checkpoint
        expected_metrics=original.train_step()
        expected=copy.deepcopy((original.model.state_dict(),original.optimizer.state_dict(),
                               original.scheduler.state_dict(),original.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/"resumed")
    with Trainer(cfg,resume=checkpoint) as resumed:
        actual=resumed.train_step()
        assert_nested_equal(expected,(resumed.model.state_dict(),resumed.optimizer.state_dict(),
                                     resumed.scheduler.state_dict(),resumed.sampler.state_dict()))
        assert actual["training_loss"]==expected_metrics["training_loss"]


@pytest.mark.parametrize("rule", ["gated_delta","rwkv7","mlstm"])
def test_state_memorization(tiny_config,rule):
    model=StateRecurrentModel(tiny_config,ResearchConfig(architecture="state_hierarchy",state_rule=rule,scale_conditioning=False))
    objective=CausalLanguageModeling(); optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    ids=torch.arange(12).repeat(2,1);batch={"input_ids":ids[:,:-1],"targets":ids[:,1:]}
    initial=objective(model,batch).loss.item()
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True);loss=objective(model,batch).loss
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    assert objective(model,batch).loss.item()<initial/15


@pytest.mark.cuda
@pytest.mark.parametrize("rule", ["gated_delta","rwkv7","mlstm"])
@pytest.mark.parametrize("sharing", [False,True])
def test_state_cuda_inductor(tiny_config,rule,sharing):
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable; not empirical GPU validation")
    model=StateRecurrentModel(tiny_config,ResearchConfig(architecture="state_hierarchy",state_rule=rule,
        share_scales=sharing,state_chunk_size=4,scale_conditioning=False)).cuda()
    ids=torch.randint(0,32,(2,17),device="cuda")
    expected=model(ids).logits;expected.square().mean().backward()
    gradients=[p.grad.clone() for p in model.parameters()];model.zero_grad(set_to_none=True)
    compiled=torch.compile(model,backend="inductor");actual=compiled(ids).logits
    actual.square().mean().backward();torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-5)
    for param,grad in zip(model.parameters(),gradients):
        torch.testing.assert_close(param.grad,grad,rtol=3e-3,atol=2e-5)


@pytest.mark.parametrize("rule", ["gated_delta","rwkv7","mlstm"])
def test_inference_state_does_not_retain_full_prefix_storage(tiny_config,rule):
    model=StateRecurrentModel(tiny_config,ResearchConfig(architecture="state_hierarchy",state_rule=rule,scale_conditioning=False)).eval()
    with torch.no_grad():state=model(torch.ones(1,31,dtype=torch.long)).state
    for tensor in (state.pending,state.last_coarse):
        assert tensor.untyped_storage().nbytes()==tensor.numel()*tensor.element_size()
    if rule=="mlstm":
        for memory in state.memories:
            assert memory[2].untyped_storage().nbytes()==memory[2].numel()*memory[2].element_size()
