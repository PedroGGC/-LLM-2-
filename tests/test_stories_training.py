import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from llm2.research.stories_protocol import configuration
from llm2.research.stories_worker import StoryTrainer,score,calibrate,generations
from llm2.research.binding_worker import training_benchmark
from llm2.research.stories_ngram import evaluate_ngram
from llm2.runtime import capture_rng
from test_stories_data import fixture_data
from test_training import assert_nested_equal


def test_story_forward_backward_memorization_and_coverage(tmp_path):
    data=fixture_data(tmp_path/'data');cfg=configuration('transition',data.root,tmp_path/'run',331,smoke=True,device='cpu')
    cfg.model.dropout=0
    with StoryTrainer(cfg,smoke=True) as trainer:
        batch=trainer.sampler.next('cpu');opt=torch.optim.AdamW(trainer.model.parameters(),lr=.01)
        loss=[]
        for _ in range(100):
            opt.zero_grad();out=trainer.objective(trainer.model,batch);out.loss.backward()
            assert torch.isfinite(out.loss) and all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(trainer.parameters,1);opt.step();loss.append(float(out.loss.detach()))
        assert loss[-1]<loss[0]*.35
        result=score(trainer.model,trainer.data,'val',32,2,trainer.device)
        assert result['bytes']==data.manifest['splits']['val']['bytes']
        samples=generations(trainer,smoke=True);assert len(samples)==2 and all(s['new_bytes']==8 for s in samples)


def test_story_exact_resume_dropout_accumulation_and_restored_benchmark(tmp_path):
    data=fixture_data(tmp_path/'data');cfg=configuration('pyramid_shared',data.root,tmp_path/'original',331,smoke=True,device='cpu');cfg.model.dropout=.2
    with StoryTrainer(cfg,smoke=True) as a:
        a.fit(until_step=2);checkpoint=a.last_checkpoint
        before=copy.deepcopy((a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict(),a.counters));rng=capture_rng()
        training_benchmark(a,samples=2,warmup=1)
        assert_nested_equal(before,(a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict(),a.counters))
        assert_nested_equal(rng,capture_rng())
        loss=a.train_step()['training_loss'];expected=copy.deepcopy((a.model.state_dict(),a.optimizer.state_dict(),a.scheduler.state_dict(),a.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/'resumed')
    with StoryTrainer(cfg,smoke=True,resume=checkpoint) as b:
        assert b.train_step()['training_loss']==loss
        assert_nested_equal(expected,(b.model.state_dict(),b.optimizer.state_dict(),b.scheduler.state_dict(),b.sampler.state_dict()))
    cfg.train.output_dir=str(tmp_path/'wrong')
    with pytest.raises(ValueError,match='sampler'):StoryTrainer(cfg,smoke=True,calibration=True,resume=checkpoint)


def test_rate_selection_train_only_and_cache_integrity(tmp_path):
    data=fixture_data(tmp_path/'data');calls=[]
    original=StoryTrainer.validate
    def validate(self,*a,**k):calls.append(self.calibration);return original(self,*a,**k)
    from unittest.mock import patch
    with patch.object(StoryTrainer,'validate',validate):
        result=calibrate('transformer_matched',data.root,tmp_path/'campaign',smoke=True,device='cpu',deadline=None)
    assert calls==[True,True]
    assert result['lr']==min(result['trials'],key=lambda v:(v['nll'],v['lr']))['lr']
    assert all(v['evaluation']['split']=='cal_val' for v in result['trials'])
    assert not list((tmp_path/'campaign').rglob('*.pt')) and not list((tmp_path/'campaign').rglob('*.safetensors'))
    cached=calibrate('transformer_matched',data.root,tmp_path/'campaign',smoke=True,device='cpu',deadline=None)
    assert cached['lr']==result['lr']
    import json
    path=tmp_path/'campaign/calibration/transformer_matched/selection.json';value=json.loads(path.read_text())
    value['trials'][0]['evaluation']['split']='val';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='trials'):calibrate('transformer_matched',data.root,tmp_path/'campaign',smoke=True,device='cpu',deadline=None)


def test_ngram_is_train_only_and_same_coverage(tmp_path):
    data=fixture_data(tmp_path/'data');result=evaluate_ngram(data,32)
    assert result['validation_bytes']==data.manifest['splits']['val']['bytes']
    assert result['training_bytes']==data.manifest['splits']['train']['bytes']
    assert np.isfinite(result['nll']) and result['nll']>0
