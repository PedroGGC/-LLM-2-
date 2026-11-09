import copy
import numpy as np
import pytest
import torch
from llm2.architectures.transition_memory import associative_reads
from llm2.research.binding_data import generate,intervene,BindingStream
from llm2.research.binding_protocol import configuration
from llm2.research.binding_worker import BindingTrainer,score,training_benchmark
from llm2.research.factory import build_backbone
from llm2.research.recall_worker import state_storage_bytes
from llm2.runtime import capture_rng
from test_training import assert_nested_equal


@pytest.mark.parametrize("phase",[0,1,3])
def test_associative_chunk_kernel_matches_recurrence_and_gradients(phase):
    torch.manual_seed(32)
    source=[torch.randn(2,19,5),torch.randn(2,19,5),torch.randn(2,19,7),torch.randn(2,5,7),torch.randn(2,5,7)]
    a=[v.clone().requires_grad_() for v in source];b=[v.clone().requires_grad_() for v in source]
    actual=associative_reads(*a,phase)
    q,k,v,total,partial=b;fine=[];coarse=[];p=phase
    for i in range(q.shape[1]):
        delta=k[:,i,:,None]*v[:,i,None,:]
        coarse.append(torch.einsum('bk,bkv->bv',q[:,i],total-partial))
        total=total+delta;partial=partial+delta
        fine.append(torch.einsum('bk,bkv->bv',q[:,i],partial));p=(p+1)%4
        if p==0:partial=torch.zeros_like(partial)
    expected=(torch.stack(fine,1),torch.stack(coarse,1),total,partial)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    sum(x.square().mean() for x in actual).backward();sum(x.square().mean() for x in expected).backward()
    for x,y in zip(a,b):torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize("variant",["transition","pyramid_shared","pyramid_independent"])
def test_models_are_causal_streamable_and_count_unique_parameters(variant):
    cfg=configuration(variant,'unused',311,smoke=True,device='cpu');cfg.model.dropout=0
    torch.manual_seed(18);m=build_backbone(cfg).eval();ids=torch.randint(0,257,(2,37))
    assert m.embedding.weight is m.lm_head.weight
    assert m.parameter_formula()==sum(p.numel() for p in m.parameters())
    expected=m(ids).logits
    for boundary in (1,3,4,5,7,8,31,32):
        changed=ids.clone();changed[:,boundary:]=(changed[:,boundary:]+37)%257
        torch.testing.assert_close(m(changed).logits[:,:boundary],expected[:,:boundary],atol=2e-6,rtol=2e-5)
    a=copy.deepcopy(m);b=copy.deepcopy(m)
    a(ids).logits.square().mean().backward();state=None;outputs=[]
    for piece in (ids[:,:1],ids[:,1:3],ids[:,3:7],ids[:,7:16],ids[:,16:]):
        out=b(piece,state=state);state=out.state;outputs.append(out.logits)
    actual=torch.cat(outputs,1);torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    actual.square().mean().backward()
    for x,y in zip(a.parameters(),b.parameters()):
        assert x.grad is not None and torch.isfinite(x.grad).all()
        torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=4e-4)
    assert state_storage_bytes(state)==m.state_bytes_per_sequence()*2
    assert len(m.operators)==(2 if variant=='pyramid_independent' else 1)
    torch.testing.assert_close(m(ids).logits,expected,atol=0,rtol=0)


def test_shared_and_independent_initial_forward_equal_and_flat_neutral_partition():
    models=[]
    for v in ('transition','pyramid_shared','pyramid_independent'):
        cfg=configuration(v,'unused',311,smoke=True,device='cpu');cfg.model.dropout=0
        torch.manual_seed(55);models.append(build_backbone(cfg).eval())
    ids=torch.randint(0,257,(2,33));flat,shared,independent=models
    torch.testing.assert_close(flat(ids).logits,shared(ids).logits,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(shared(ids).logits,independent(ids).logits,atol=0,rtol=0)
    with torch.no_grad():independent.operators[1].readout.weight.add_(1)
    torch.testing.assert_close(independent(ids).logits[:,:4],shared(ids).logits[:,:4],atol=0,rtol=0)
    assert not torch.equal(independent(ids).logits[:,4:],shared(ids).logits[:,4:])


def test_task_unique_maps_counterfactuals_and_curriculum_resume():
    batch=generate(np.random.default_rng(771),2,256)
    alt=intervene(batch);mask=batch['targets']!=-100
    assert not torch.any(batch['targets'][mask]==alt['targets'][mask])
    for data in (batch,alt):
        for row in range(2):
            x=data['input_ids'][row];table={int(x[p+1]):int(x[p+2]) for p in range(0,256,4) if x[p]==128}
            assert len(table)==len(set(table.values()))==32
            for p in (data['targets'][row]!=-100).nonzero().flatten().tolist():
                assert table[int(x[p])]==int(data['targets'][row,p])
                assert x[p-1]==129 and x[p+1]==130
    cfg=configuration('transition','unused',311,smoke=True,device='cpu');cfg.data.seq_len=256
    a=BindingStream(cfg);a.draws=398
    assert int((a.next('cpu')['input_ids']==128).sum())==cfg.data.batch_size*4
    state=a.state_dict();expected=a.next('cpu');b=BindingStream(cfg);b.load_state_dict(state)
    for k,v in b.next('cpu').items():assert torch.equal(v,expected[k])
    assert int((a.next('cpu')['input_ids']==128).sum())==cfg.data.batch_size*8


@pytest.mark.parametrize('variant',['transition','pyramid_shared','pyramid_independent'])
def test_generalizes_to_new_maps_including_changed_queries(variant):
    cfg=configuration(variant,'unused',311,device='cpu');cfg.model.dropout=0
    torch.manual_seed(815);model=build_backbone(cfg);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    rng=np.random.default_rng(817)
    for _ in range(110):
        batch=generate(rng,8,128,writes=4,queries=8);mask=batch['targets']!=-100
        opt.zero_grad();logits=model(batch['input_ids']).logits
        loss=torch.nn.functional.cross_entropy(logits[mask],batch['targets'][mask]);loss.backward()
        assert torch.isfinite(loss);torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    result=score(model,128,examples=32,seed=88221,device=torch.device('cpu'),writes=4,queries=8)
    assert result['accuracy']>=.95 and result['paired_accuracy']>=.95


def test_resume_with_dropout_accumulation_and_benchmark_restore(tmp_path):
    cfg=configuration('pyramid_shared',tmp_path/'original',311,smoke=True,device='cpu');cfg.model.dropout=.2
    with BindingTrainer(cfg,smoke=True) as a:
        a.fit(until_step=2);checkpoint=a.last_checkpoint
        saved=copy.deepcopy((a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict(),a.counters))
        rng=capture_rng();training_benchmark(a,samples=2,warmup=1)
        assert_nested_equal(saved,(a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict(),a.counters))
        assert_nested_equal(rng,capture_rng())
        expected_loss=a.train_step()['training_loss']
        expected=copy.deepcopy((a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/'resumed')
    with BindingTrainer(cfg,smoke=True,resume=checkpoint) as b:
        assert b.train_step()['training_loss']==expected_loss
        assert_nested_equal(expected,(b.model.state_dict(),b.optimizer.state_dict(),b.scheduler.state_dict(),b.sampler.state_dict()))
    cfg.research.associative_levels=1;cfg.train.output_dir=str(tmp_path/'wrong')
    with pytest.raises(ValueError,match='configuration'):BindingTrainer(cfg,smoke=True,resume=checkpoint)


def test_measured_update_matches_trainer_math(tmp_path):
    from llm2.runtime import restore_rng
    cfg=configuration('transition',tmp_path/'math',311,smoke=True,device='cpu');cfg.model.dropout=.2
    with BindingTrainer(cfg,smoke=True) as trainer:
        before=copy.deepcopy((trainer.model.state_dict(),trainer.optimizer.state_dict(),trainer.scheduler.state_dict(),trainer.sampler.state_dict()))
        rng=capture_rng();trainer.train_step()
        expected=copy.deepcopy((trainer.model.state_dict(),trainer.optimizer.state_dict()))
        for obj,state in zip((trainer.model,trainer.optimizer,trainer.scheduler,trainer.sampler),before):obj.load_state_dict(state)
        restore_rng(rng);captured=[];original=trainer.optimizer.step
        def step(*a,**k):
            value=original(*a,**k);captured.append(copy.deepcopy((trainer.model.state_dict(),trainer.optimizer.state_dict())));return value
        trainer.optimizer.step=step
        training_benchmark(trainer,samples=1,warmup=0)
        trainer.optimizer.step=original
        assert_nested_equal(expected,captured[0])


@pytest.mark.cuda
@pytest.mark.parametrize('variant',['transition','pyramid_independent'])
def test_cuda_inductor_real_forward_backward_and_streaming(variant):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    cfg=configuration(variant,'unused',311,smoke=True,device='cpu');cfg.model.dropout=0
    torch.manual_seed(3);m=build_backbone(cfg).cuda().eval();ids=torch.randint(0,257,(2,32),device='cuda')
    expected=m(ids).logits;compiled=torch.compile(m,backend='inductor')
    torch.testing.assert_close(compiled(ids).logits,expected,atol=3e-5,rtol=5e-4)
    compiled(ids).logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    a=m(ids[:,:7]);b=m(ids[:,7:],state=a.state)
    torch.testing.assert_close(torch.cat((a.logits,b.logits),1),expected,atol=3e-5,rtol=5e-4)
