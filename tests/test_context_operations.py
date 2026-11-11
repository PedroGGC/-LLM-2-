import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from llm2.config import ExperimentConfig,load_config
from llm2.research import context_protocol as recipe,stories_protocol as previous,stories_campaign as campaign
from llm2.research.stories_bootstrap import bootstrap_git
from llm2.research.campaign_storage import verify_pair
from llm2.research.factory import build_backbone
from test_stories_operations import args


def test_context_configs_counts_rotation_and_independent_recipe(tmp_path):
    rows=recipe.jobs(tmp_path,'unused',['0','1']);counts={}
    assert len(rows)==16 and recipe.PROTOCOL_ID!=previous.PROTOCOL_ID
    with torch.device('meta'):
        for row in rows:
            cfg=ExperimentConfig.from_dict(row['config_resolved']);model=build_backbone(cfg)
            counts[row['variant']]=sum(p.numel() for p in model.parameters())
            if hasattr(model,'parameter_formula'):assert counts[row['variant']]==model.parameter_formula()
            path=campaign.materialize(cfg,tmp_path/(row['id']+'.yaml'));assert load_config(path).to_dict()==cfg.to_dict()
            assert cfg.train.max_steps==cfg.optimizer.schedule_steps==6000
    assert counts=={'transformer_matched':86400,'context_single':85848,'transformer_medium':148960,
        'context_shared2':85848,'context_untied2':151056,'hierarchy_shared':85928,'hierarchy_untied':151136,'transformer':2164992}
    for variant in recipe.VARIANTS:
        assert {r['gpu_slot'] for r in rows if r['variant']==variant}=={'0','1'}
        cfg=recipe.configuration(variant,'unused','unused',341,calibration=True)
        assert cfg.train.seed==941 and cfg.train.max_steps==cfg.optimizer.schedule_steps==400
    with patch('llm2.research.stories_bootstrap.urlopen',side_effect=AssertionError('network')):
        for settings in ({'profile':'bad'},{'profile':'context','seeds':[331]},
                         {'profile':'context','seeds':[341],'run_number':32}):
            with pytest.raises(ValueError):bootstrap_git('a'*40,run=True,**settings)


@pytest.mark.parametrize('failure',['worker','deadline','disk'])
def test_context_failure_deadline_disk_export_partial(tmp_path,failure):
    output=tmp_path/'llm2-campaign-0033';a=args(output);a.seeds=[341]
    if failure=='disk':context=patch.object(campaign.shutil,'disk_usage',return_value=type('Usage',(),{'free':0})())
    else:context=patch.object(campaign,'run_pair',side_effect=campaign.SessionBudgetExceeded('deadline') if failure=='deadline' else RuntimeError('worker failure'))
    with context:
        if failure=='deadline':assert campaign.execute(a,recipe=recipe)['status']=='partial_budget'
        else:
            with pytest.raises(RuntimeError,match='evidence exported'):campaign.execute(a,recipe=recipe)
    pair=verify_pair(tmp_path/'llm2-campaign-0033-results.zip',tmp_path/'llm2-campaign-0033-checkpoints.zip')
    assert pair['campaign']['protocol_id']==recipe.PROTOCOL_ID
    assert pair['campaign']['status'] in ('failed','partial_budget')


def test_resource_milestone_requires_both_complete_seeds_and_quality(tmp_path):
    rows=[]
    for seed in recipe.SEEDS:
        for variant,seconds in [('transformer_matched',.02),('context_single',.009)]:
            rows.append({'variant':variant,'seed':seed,'id':f'{variant}-{seed}','status':'complete','step':6000,'gpu_slot':'0',
                'config_resolved':{'data':{'seq_len':256},'train':{'max_steps':6000}},'parameters':86000,
                'counters':{'tokens_processed':1,'train_wall_seconds':1,'peak_vram_allocated_bytes':None},
                'training_profile':{'median_step_seconds':seconds},'evaluations':[{'step':6000,'nll':1.,'bits_per_byte':1.4,'bytes':10,
                    'training_bytes':1,'training_wall_seconds':1}]})
    holder=SimpleNamespace(output=tmp_path,manifest={'runs':rows,'status':'complete'})
    def result():
        campaign.report(holder,recipe=recipe)
        values=json.loads((tmp_path/'report/report.json').read_text())['resource_milestones']
        return next(v['passed_both_seeds'] for v in values if v['variant']=='context_single' and v['reference']=='transformer_matched' and v['target_speed_factor']==2)
    assert result()
    rows[-1]['evaluations'][0]['nll']=1.003;assert not result()
    rows[-1]['evaluations'][0]['nll']=1.;rows[-1]['step']=3000;assert not result()
    rows[-1]['step']=6000;rows.pop();assert not result()


def test_full_cpu_context_campaign_calibration_texts_reuse_and_old_cycle_isolation(tmp_path):
    output=tmp_path/'llm2-campaign-0033';a=args(output);a.seeds=[341]
    value=campaign.execute(a,recipe=recipe)
    assert value['status']=='complete' and not value['pending_ids']
    pair=verify_pair(tmp_path/'llm2-campaign-0033-results.zip',tmp_path/'llm2-campaign-0033-checkpoints.zip',require_complete=True)
    assert len(pair['covered'])==8 and sum(n=='source.zip' for n in pair['results_members'])==1
    for j in value['runs']:
        assert j['step']==4 and j['counters']['tokens_processed']==512
        assert len(j['generation'])==2 and len(j['profiles'])==2
        assert j['protocol_id']==recipe.PROTOCOL_ID and j['config_resolved']['train']['seed']==341
        assert len(list((Path(j['directory'])/'checkpoints').glob('step-*')))==1
        assert j['optimization_probe']['final_logit_difference']>0
        for trial in j['calibration']['trials']:
            assert trial['checkpoint_metadata']['config']['train']['seed']==941
            assert trial['evaluation']['split']=='cal_val' and trial['calibration_state_retired']
    assert not list((output/'calibration').rglob('*.pt'))
    legacy=copy.deepcopy(value);legacy['cycle']=previous.CYCLE;legacy['dataset_sha256']='must-not-be-read'
    folder=tmp_path/'llm2-campaign-0032';folder.mkdir();(folder/'campaign.json').write_text(json.dumps(legacy))
    a=args(tmp_path/'llm2-campaign-0034');a.seeds=[341]
    with patch.object(campaign,'run_pair',side_effect=AssertionError('unneeded retraining')):
        again=campaign.execute(a,recipe=recipe)
    assert again['status']=='complete' and len(again['historical_runs'])==8 and not again['runs']
