import copy
import torch
import pytest
from llm2.research.context_protocol import configuration
from llm2.research.stories_worker import StoryTrainer
from llm2.research.binding_worker import training_benchmark
from llm2.runtime import capture_rng
from test_stories_data import fixture_data
from test_training import assert_nested_equal


@pytest.mark.parametrize('variant',['context_shared2','hierarchy_shared'])
def test_context_memorization_and_finite_gradients(tmp_path,variant):
    data=fixture_data(tmp_path/'data');cfg=configuration(variant,data.root,tmp_path/'run',341,smoke=True,device='cpu');cfg.model.dropout=0
    with StoryTrainer(cfg,smoke=True) as trainer:
        batch=trainer.sampler.next('cpu');opt=torch.optim.AdamW(trainer.model.parameters(),lr=.01);loss=[]
        for _ in range(100):
            opt.zero_grad();out=trainer.objective(trainer.model,batch);out.loss.backward()
            assert torch.isfinite(out.loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.model.parameters())
            torch.nn.utils.clip_grad_norm_(trainer.parameters,1);opt.step();loss.append(float(out.loss.detach()))
        assert loss[-1]<loss[0]*.35


@pytest.mark.parametrize('variant',['context_shared2','hierarchy_untied'])
def test_context_exact_resume_dropout_accumulation_and_pilot(tmp_path,variant):
    data=fixture_data(tmp_path/'data');cfg=configuration(variant,data.root,tmp_path/'first',341,smoke=True,device='cpu');cfg.model.dropout=.2
    with StoryTrainer(cfg,smoke=True) as a:
        a.fit(until_step=2);checkpoint=a.last_checkpoint
        before=copy.deepcopy((a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict(),a.counters));rng=capture_rng()
        training_benchmark(a,samples=2,warmup=1)
        assert_nested_equal(before,(a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict(),a.counters));assert_nested_equal(rng,capture_rng())
        loss=a.train_step()['training_loss'];expected=copy.deepcopy((a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/'resume')
    with StoryTrainer(cfg,smoke=True,resume=checkpoint) as b:
        assert b.train_step()['training_loss']==loss
        assert_nested_equal(expected,(b.model.state_dict(),b.optimizer.state_dict(),b.scheduler.state_dict(),b.sampler.state_dict()))
