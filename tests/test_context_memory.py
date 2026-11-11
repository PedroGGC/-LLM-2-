import copy
import collections
import pytest
import torch
from llm2.architectures.context_memory import retention_blocked,retention_sequential
from llm2.research.context_protocol import configuration,VARIANTS
from llm2.research.factory import build_backbone


@pytest.mark.parametrize('length',[1,31,67])
def test_affine_scan_forward_backward_states_and_extreme_decay(length):
    torch.manual_seed(733)
    values=[torch.randn(2,2,length,4),torch.randn(2,2,length,4),torch.randn(2,2,length,6),
            -torch.rand(2,2,length)*4,torch.randn(2,2,4,6)]
    a=[v.requires_grad_() for v in values];b=[v.detach().clone().requires_grad_() for v in a]
    ref=retention_sequential(*a);out=retention_blocked(*b,block_size=16)
    for x,y in zip(out,ref):torch.testing.assert_close(x,y,atol=1e-5,rtol=2e-4)
    sum(v.square().mean() for v in ref).backward();sum(v.square().mean() for v in out).backward()
    for x,y in zip(a,b):
        assert torch.isfinite(y.grad).all();torch.testing.assert_close(x.grad,y.grad,atol=1e-5,rtol=5e-4)
    for logs in (torch.zeros_like(a[3]),torch.full_like(a[3],-1000)):
        r=retention_sequential(*a[:3],logs,a[4]);o=retention_blocked(*a[:3],logs,a[4],block_size=16)
        for x,y in zip(o,r):torch.testing.assert_close(x,y,atol=5e-5,rtol=2e-4)


@pytest.mark.parametrize('variant',['context_single','context_shared2','context_untied2','hierarchy_shared','hierarchy_untied'])
def test_context_causality_streaming_backward_and_parameter_sharing(variant):
    cfg=configuration(variant,'unused','unused',341,smoke=True,device='cpu');cfg.model.dropout=0;cfg.model.max_seq_len=128
    torch.manual_seed(13);model=build_backbone(cfg).eval();ids=torch.randint(0,257,(2,67))
    full=model(ids)
    for cut in (1,3,4,5,31,32,35):
        changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+7)%257
        torch.testing.assert_close(model(changed).logits[:,:cut],full.logits[:,:cut],rtol=0,atol=0)
    outputs=[];state=None
    for piece in ids.split(3,1):
        out=model(piece,state=state);outputs.append(out.logits);state=out.state
    torch.testing.assert_close(torch.cat(outputs,1),full.logits,atol=4e-6,rtol=2e-4)
    for a,b in zip(state.cells,full.state.cells):
        torch.testing.assert_close(a.memory,b.memory,atol=3e-6,rtol=2e-4)
        torch.testing.assert_close(a.tail,b.tail,atol=3e-6,rtol=2e-4)
    full.logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.parameter_formula()==sum(p.numel() for p in model.parameters())
    assert model.lm_head.weight is model.embedding.weight
    if model.depth==2:
        assert full.state.cells[0].memory.data_ptr()!=full.state.cells[1].memory.data_ptr()
    # State reset gives the same second document as its isolated evaluation.
    torch.testing.assert_close(model(ids[1:]).logits,full.logits[1:],atol=3e-6,rtol=2e-4)
    for cell in full.state.cells:
        for value in (cell.memory,cell.tail):assert value.untyped_storage().nbytes()==value.numel()*value.element_size()
    assert full.state.pending.untyped_storage().nbytes()==full.state.pending.numel()*4
    wrong=copy.copy(full.state);wrong.cells=tuple(copy.copy(s) for s in full.state.cells)
    wrong.cells[0].memory=wrong.cells[0].memory.double()
    with pytest.raises(ValueError,match='state shape'):model(ids[:,:1],state=wrong)
    if model.hierarchy:
        with torch.no_grad():model.merge.zero_()
        assert not torch.allclose(model(ids).logits,full.logits,atol=1e-5)


@pytest.mark.parametrize('pair',[('context_shared2','context_untied2'),('hierarchy_shared','hierarchy_untied')])
def test_copied_shared_and_independent_start_with_equal_function(pair):
    models=[]
    for variant in pair:
        cfg=configuration(variant,'unused','unused',341,smoke=True,device='cpu');cfg.model.dropout=0
        torch.manual_seed(19);models.append(build_backbone(cfg).eval())
    ids=torch.randint(0,257,(2,32));a,b=models
    torch.testing.assert_close(a(ids).logits,b(ids).logits,atol=0,rtol=0)
    assert len(a.cells)==1 and len(b.cells)==2
    assert sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters())==sum(p.numel() for p in a.cells[0].parameters())


def test_order_collision_is_resolved_by_causal_state():
    a,b,tail=b'xxcatxx',b'xxdogxx',b'xxxxxxxx'
    ids=torch.tensor([[256,*(a+b+tail)],[256,*(b+a+tail)]])
    triples=lambda values:collections.Counter(tuple(values[i:i+3]) for i in range(len(values)-2))
    assert triples(ids[0].tolist())==triples(ids[1].tolist())
    for variant in ('context_single','context_shared2','hierarchy_shared'):
        cfg=configuration(variant,'unused','unused',341,smoke=True,device='cpu');cfg.model.dropout=0
        torch.manual_seed(10032);model=build_backbone(cfg).eval();logits=model(ids).logits[:,-1]
        assert (logits[0]-logits[1]).abs().max()>1e-5


@pytest.mark.cuda
@pytest.mark.parametrize('variant',['context_shared2','hierarchy_shared'])
def test_context_real_cuda_inductor_reference_forward_backward(variant):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    cfg=configuration(variant,'unused','unused',341,device='cuda');cfg.model.dropout=0
    torch.manual_seed(43);a=build_backbone(cfg).cuda().eval();b=copy.deepcopy(a)
    for cell in b.cells:cell.execution='sequential'
    ids=torch.randint(0,257,(2,256),device='cuda');compiled=torch.compile(a,backend='inductor')
    actual=compiled(ids).logits;expected=b(ids).logits
    torch.testing.assert_close(actual,expected,atol=5e-5,rtol=1e-3)
    actual.square().mean().backward();expected.square().mean().backward()
    for x,y in zip(a.parameters(),b.parameters()):
        assert torch.isfinite(x.grad).all();torch.testing.assert_close(x.grad,y.grad,atol=5e-5,rtol=3e-3)
