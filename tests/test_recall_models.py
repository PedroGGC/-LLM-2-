import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from llm2.architectures.episodic import fractions,sparse_indices
from llm2.config import ExperimentConfig
from llm2.research.factory import build_backbone
from llm2.research.recall_protocol import configuration,VARIANTS
from llm2.research.recall_data import generate,RecallStream
from llm2.research.recall_worker import RecallTrainer,score
from llm2.research.state_readout_worker import initial_digest
from test_training import assert_nested_equal


@pytest.mark.parametrize("pattern",["uniform","binary","fibonacci"])
def test_index_cardinality_reach_causality_and_storage(pattern):
    completed=torch.arange(0,300)
    indices,valid=sparse_indices(completed,64,16,fractions(pattern,16))
    for n,idx,mask in zip(completed.tolist(),indices.tolist(),valid.tolist()):
        chosen=[i for i,m in zip(idx,mask) if m]
        assert len(chosen)==min(n,16)==len(set(chosen))
        assert all(max(0,n-64)<=i<n for i in chosen)
        if chosen:assert max(chosen)==n-1 and min(chosen)==max(0,n-64)


def test_task_answers_are_external_reproducible_and_sampler_resumes():
    a=generate(np.random.default_rng(77),2,1024)
    b=generate(np.random.default_rng(77),2,1024)
    for k in a:assert torch.equal(a[k],b[k])
    for row in range(2):
        known={int(a['input_ids'][row,i+1]):(int(a['input_ids'][row,i+2]),i) for i in range(0,1024,4) if a['input_ids'][row,i]==128}
        assert len(known)==32
        positions=(a['targets'][row]!=-100).nonzero().flatten().tolist();assert len(positions)==16
        for p in positions:
            key=int(a['input_ids'][row,p]);value,write=known[key]
            assert int(a['targets'][row,p])==value and int(a['distances'][row,p])==p-1-write
            assert a['input_ids'][row,p-1]==129 and a['input_ids'][row,p+1]==130
    cfg=configuration('memory_all','unused',271,smoke=True,device='cpu')
    s=RecallStream(cfg);s.next('cpu');state=s.state_dict();expected=s.next('cpu')
    t=RecallStream(cfg);t.load_state_dict(state)
    for k,v in t.next('cpu').items():assert torch.equal(v,expected[k])


@pytest.mark.parametrize("variant",["memory_all","memory_uniform","memory_binary","memory_fibonacci"])
def test_causal_streaming_gradient_and_tied_parameter_counts(variant):
    cfg=configuration(variant,'unused',271,smoke=True,device='cpu');cfg.model.dropout=0.
    torch.manual_seed(21);model=build_backbone(cfg).eval()
    assert model.embedding.weight is model.lm_head.weight
    assert len(model.cells)==1 and model.parameter_formula()==sum(p.numel() for p in model.parameters())
    ids=torch.randint(0,257,(2,33));expected=model(ids).logits
    for boundary in (1,3,4,5,7,8,16,32):
        changed=ids.clone();changed[:,boundary:]=(changed[:,boundary:]+7)%257
        torch.testing.assert_close(model(changed).logits[:,:boundary],expected[:,:boundary],atol=2e-6,rtol=2e-5)
    a=copy.deepcopy(model);b=copy.deepcopy(model)
    a(ids).logits.square().mean().backward()
    state=None;outputs=[]
    for piece in (ids[:,:1],ids[:,1:4],ids[:,4:7],ids[:,7:16],ids[:,16:]):
        out=b(piece,state=state);state=out.state;outputs.append(out.logits)
    actual=torch.cat(outputs,1);torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    actual.square().mean().backward()
    for x,y in zip(a.parameters(),b.parameters()):
        assert x.grad is not None and y.grad is not None and torch.isfinite(x.grad).all()
        torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=6e-4)
    assert state.bank.shape[1]<=cfg.research.memory_slots
    torch.testing.assert_close(model(ids).logits,expected,atol=0,rtol=0)
    assert model.state_bytes_per_sequence()>=cfg.research.memory_slots*cfg.model.d_model*4


def test_patterns_have_identical_initial_parameters_and_memory_has_influence():
    digests=[]
    for v in VARIANTS:
        if not v.startswith('memory_'):continue
        cfg=configuration(v,'unused',271,smoke=True,device='cpu');cfg.model.dropout=0
        torch.manual_seed(41);model=build_backbone(cfg).eval();digests.append(initial_digest(model))
        ids=torch.randint(0,257,(2,32));prefix=model(ids[:,:16]).state
        altered=copy.copy(prefix);altered.bank=prefix.bank+torch.randn_like(prefix.bank)
        assert not torch.equal(model(ids[:,16:],state=prefix).logits,model(ids[:,16:],state=altered).logits)
    assert len(set(digests))==1


@pytest.mark.parametrize('variant',['memory_all','memory_fibonacci'])
def test_streaming_over_capacity_and_sequential_rule(variant):
    cfg=configuration(variant,'unused',271,smoke=True,device='cpu');cfg.model.dropout=0
    cfg.research.memory_slots=cfg.research.memory_reads=4
    torch.manual_seed(7);model=build_backbone(cfg).eval();ids=torch.randint(0,257,(1,61))
    expected=model(ids).logits;state=None;parts=[]
    for piece in ids.split(7,dim=1):
        out=model(piece,state=state);state=out.state;parts.append(out.logits)
    assert state.bank.shape[1]==4
    torch.testing.assert_close(torch.cat(parts,1),expected,atol=3e-6,rtol=4e-5)
    sequential=copy.deepcopy(model)
    for cell in sequential.cells:cell.execution='sequential'
    torch.testing.assert_close(sequential(ids).logits,expected,atol=3e-6,rtol=4e-5)


def test_finite_forward_backward_and_memorization():
    cfg=configuration('memory_all','unused',271,smoke=True,device='cpu');cfg.model.dropout=0
    torch.manual_seed(61);model=build_backbone(cfg)
    batch=generate(np.random.default_rng(19),2,32,writes=4,queries=4)
    mask=batch['targets']!=-100;opt=torch.optim.AdamW(model.parameters(),lr=.01)
    losses=[]
    for _ in range(80):
        opt.zero_grad();logits=model(batch['input_ids']).logits
        loss=torch.nn.functional.cross_entropy(logits[mask],batch['targets'][mask]);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]*.35


def test_exact_resume_with_dropout_and_accumulation(tmp_path):
    cfg=configuration('memory_fibonacci',tmp_path/'original',271,smoke=True,device='cpu')
    cfg.model.dropout=.2;assert cfg.train.grad_accum_steps==2
    with RecallTrainer(cfg,smoke=True) as a:
        a.fit(until_step=2);cp=a.last_checkpoint;loss=a.train_step()['training_loss']
        expected=copy.deepcopy((a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/'resumed')
    with RecallTrainer(cfg,smoke=True,resume=cp) as b:
        actual=b.train_step()['training_loss']
        assert actual==loss
        assert_nested_equal(expected,(b.model.state_dict(),b.optimizer.state_dict(),b.scheduler.state_dict(),b.sampler.state_dict()))
    cfg.research.memory_pattern='binary';cfg.train.output_dir=str(tmp_path/'wrong')
    with pytest.raises(ValueError,match='configuration'):RecallTrainer(cfg,smoke=True,resume=cp)


def test_development_forward_cost_is_not_reported_as_zero(tmp_path):
    cfg=configuration('transformer',tmp_path/'cost',271,smoke=True,device='cpu')
    with RecallTrainer(cfg,smoke=True) as trainer:
        cost=trainer._evaluation_cost();assert cost>0
        trainer.validate()
        assert trainer.counters['validation_flops_estimate']==2*cost


@pytest.mark.cuda
@pytest.mark.parametrize('variant',['memory_all','memory_fibonacci'])
def test_real_cuda_inductor_eager_and_streaming(variant):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    cfg=configuration(variant,'unused',271,smoke=True,device='cpu');cfg.model.dropout=0
    torch.manual_seed(3);model=build_backbone(cfg).cuda().eval();ids=torch.randint(0,257,(2,32),device='cuda')
    eager=model(ids).logits;compiled=torch.compile(model,backend='inductor')
    torch.testing.assert_close(compiled(ids).logits,eager,atol=3e-5,rtol=5e-4)
    model.train();loss=compiled(ids).logits.square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();p=model(ids[:,:7]);out=model(ids[:,7:],state=p.state)
    torch.testing.assert_close(torch.cat((p.logits,out.logits),1),eager,atol=3e-5,rtol=5e-4)
