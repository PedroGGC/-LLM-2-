import copy
import pytest
import torch
from llm2.architectures.transition_memory import associative_reads,blocked_associative_reads
from llm2.research.stories_protocol import configuration
from llm2.research.factory import build_backbone


@pytest.mark.parametrize('phase',[0,1,3])
def test_blocked_kernel_equivalence_forward_backward_and_no_future(phase):
    torch.manual_seed(71)
    a=[torch.randn(2,67,8),torch.randn(2,67,8),torch.randn(2,67,12),torch.randn(2,8,12),torch.randn(2,8,12)]
    a=[v.requires_grad_() for v in a];b=[v.detach().clone().requires_grad_() for v in a]
    expected=associative_reads(*a,phase,4);actual=blocked_associative_reads(*b,phase,4,32)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)
    sum(x.square().mean() for x in expected).backward();sum(x.square().mean() for x in actual).backward()
    for x,y in zip(a,b):torch.testing.assert_close(x.grad,y.grad,atol=3e-5,rtol=3e-5)
    changed=[v.detach().clone() for v in b]
    for v in changed[:3]:v[:,35:]+=2
    nofuture=blocked_associative_reads(*changed,phase,4,32)
    for x,y in zip(actual[:2],nofuture[:2]):torch.testing.assert_close(x[:,:35],y[:,:35],atol=0,rtol=0)


@pytest.mark.parametrize('variant',['transition','pyramid_shared'])
def test_blocked_model_reference_and_streaming(variant):
    cfg=configuration(variant,'unused','unused',331,smoke=True,device='cpu');cfg.model.dropout=0
    torch.manual_seed(17);model=build_backbone(cfg).eval();ref=copy.deepcopy(model);ref.research.associative_execution='reference'
    ids=torch.randint(0,257,(2,61));actual=model(ids).logits;expected=ref(ids).logits
    torch.testing.assert_close(actual,expected,atol=4e-6,rtol=5e-5)
    actual.square().mean().backward();expected.square().mean().backward()
    for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=4e-6,rtol=6e-4)
    state=None;out=[]
    for piece in ids.split(7,1):
        result=model(piece,state=state);state=result.state;out.append(result.logits)
    torch.testing.assert_close(torch.cat(out,1),actual,atol=4e-6,rtol=5e-5)
    assert model.parameter_formula()==sum(p.numel() for p in model.parameters())
    assert model.architecture_version==2 and model.embedding.weight is model.lm_head.weight


@pytest.mark.cuda
@pytest.mark.parametrize('variant',['transition','pyramid_shared'])
def test_real_cuda_blocked_inductor_reference_and_gradients(variant):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    cfg=configuration(variant,'unused','unused',331,smoke=True,device='cpu');cfg.model.dropout=0
    torch.manual_seed(13);model=build_backbone(cfg).cuda().eval();ref=copy.deepcopy(model);ref.research.associative_execution='reference'
    ids=torch.randint(0,257,(2,64),device='cuda');compiled=torch.compile(model,backend='inductor')
    expected=ref(ids).logits;actual=compiled(ids).logits
    torch.testing.assert_close(actual,expected,atol=4e-5,rtol=8e-4)
    actual.square().mean().backward();expected.square().mean().backward()
    for a,b in zip(model.parameters(),ref.parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad,b.grad,atol=4e-5,rtol=2e-3)
