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


@pytest.mark.parametrize("w_dim", (1, 12))
def test_dplr_inter_chunk_scan_custom_op_cpu_parity_and_compile(w_dim):
    from llm2.architectures.state_kernels import dplr_chunked, dplr_sequential
    from llm2.architectures.triton_dplr import dplr_inter_chunk_scan

    torch.manual_seed(99)
    b, h, l, v, k, c = 2, 2, 19, 10, 12, 8
    inputs = [torch.randn(b, h, l, k) * 0.2, torch.randn(b, h, l, k) * 0.2, torch.randn(b, h, l, v) * 0.2,
              torch.randn(b, h, l, w_dim) * 0.2 - 0.5, torch.randn(b, h, l, k) * 0.1, torch.randn(b, h, l, k) * 0.1,
              torch.randn(b, h, v, k) * 0.1]

    refs = []
    for mode in ("seq", "chunk"):
        xs = [x.clone().requires_grad_() for x in inputs]
        q, kk, vv, lw, aa, bb, st = xs
        if mode == "seq":
            o, m = dplr_sequential(q, kk, vv, lw, aa, bb, st)
        else:
            o, m = dplr_chunked(q, kk, vv, lw, aa, bb, st, chunk_size=c, use_triton=False)
        (o.square().sum() + 0.05 * m.square().sum()).backward()
        refs.append((o.detach(), m.detach(), [x.grad.detach() for x in xs]))

    torch.testing.assert_close(refs[0][0], refs[1][0], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(refs[0][1], refs[1][1], atol=2e-5, rtol=1e-4)

    # Verify custom_op PyTorch fallback + autograd + torch.compile match loop exact
    n = 4
    kv = torch.randn(b, h, n, c, v, requires_grad=True)
    rp = torch.randn(b, h, n, c, k, requires_grad=True)
    md = torch.rand(b, h, n, 1, w_dim, requires_grad=True)
    st_t = torch.randn(b, h, n, k, k) * 0.05
    st_t.requires_grad_(True)
    ml = torch.randn(b, h, n, v, k, requires_grad=True)
    init = torch.randn(b, h, v, k, requires_grad=True)

    o_scan, m_scan = dplr_inter_chunk_scan(kv, rp, md, st_t, ml, init)
    (o_scan.square().sum() + 0.1 * m_scan.square().sum()).backward()
    grads_scan = [x.grad.clone() for x in (kv, rp, md, st_t, ml, init)]

    for x in (kv, rp, md, st_t, ml, init):
        x.grad = None
    mem = init
    outs = []
    for idx in range(n):
        outs.append(kv[:, :, idx] + rp[:, :, idx] @ mem.transpose(-1, -2))
        mem = mem * md[:, :, idx] + ml[:, :, idx] + mem @ st_t[:, :, idx]
    o_loop = torch.cat(outs, dim=-2)
    (o_loop.square().sum() + 0.1 * mem.square().sum()).backward()
    torch.testing.assert_close(o_scan, o_loop, atol=5e-5, rtol=3e-4)
    torch.testing.assert_close(m_scan, mem, atol=5e-5, rtol=3e-4)
    for g1, x in zip(grads_scan, (kv, rp, md, st_t, ml, init)):
        torch.testing.assert_close(g1, x.grad, atol=5e-5, rtol=3e-4)

    # Verify custom_op compiles without errors under torch.compile
    for x in (kv, rp, md, st_t, ml, init):
        x.grad = None
    compiled_scan = torch.compile(dplr_inter_chunk_scan, backend="aot_eager")
    o_comp, m_comp = compiled_scan(kv, rp, md, st_t, ml, init)
    (o_comp.square().sum() + 0.1 * m_comp.square().sum()).backward()
    torch.testing.assert_close(o_comp, o_scan, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(m_comp, m_scan, atol=1e-6, rtol=1e-6)
    for g1, x in zip(grads_scan, (kv, rp, md, st_t, ml, init)):
        torch.testing.assert_close(g1, x.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("v_dim,k_dim,w_dim,seq_len", [
    (64, 64, 1, 513),
    (48, 96, 1, 512),
    (64, 64, 64, 256),
])
def test_triton_dplr_chunked_cuda_forward_backward_parity(v_dim, k_dim, w_dim, seq_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; not empirical GPU validation")
    from llm2.architectures.state_kernels import dplr_chunked
    from llm2.architectures.triton_dplr import HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("Triton unavailable")
    torch.manual_seed(321)
    b, h, c = 2, 2, 32
    tensors = [
        torch.randn(b, h, seq_len, k_dim, device="cuda") * 0.15,
        torch.randn(b, h, seq_len, k_dim, device="cuda") * 0.15,
        torch.randn(b, h, seq_len, v_dim, device="cuda") * 0.15,
        torch.randn(b, h, seq_len, w_dim, device="cuda") * 0.2 - 1.0,
        torch.randn(b, h, seq_len, k_dim, device="cuda") * 0.08,
        torch.randn(b, h, seq_len, k_dim, device="cuda") * 0.08,
        torch.randn(b, h, v_dim, k_dim, device="cuda") * 0.08,
    ]
    outs = []
    for flag in (False, True):
        xs = [t.clone().requires_grad_() for t in tensors]
        o, m = dplr_chunked(*xs, chunk_size=c, use_triton=flag)
        (o.square().sum() + 0.05 * m.square().sum()).backward()
        outs.append((o.detach(), m.detach(), [x.grad.detach() for x in xs]))

    torch.testing.assert_close(outs[0][0], outs[1][0], atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(outs[0][1], outs[1][1], atol=2e-4, rtol=1e-3)
    for g_ref, g_tri in zip(outs[0][2], outs[1][2]):
        torch.testing.assert_close(g_ref, g_tri, atol=3e-4, rtol=2e-3)
