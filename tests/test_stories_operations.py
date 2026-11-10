import argparse
import hashlib
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
import pytest
import torch
from llm2.config import ExperimentConfig,load_config
from llm2.research import stories_campaign as campaign
from llm2.research.stories_protocol import jobs,configuration,VARIANTS
from llm2.research.stories_bootstrap import bootstrap_git
from llm2.research.campaign_storage import verify_pair
from llm2.research.factory import build_backbone


def args(path):
    return argparse.Namespace(output=str(path),cache=str(path.parent/'cache'),device='cpu',gpus=['0','1'],seeds=[331],
        engineering_smoke=True,session_seconds=900,bootstrap_elapsed_seconds=0,bootstrap_metadata=None,started=time.monotonic())


def test_counts_configs_gpu_rotation_and_frozen_helpers(tmp_path):
    rows=jobs(tmp_path,'unused',['0','1']);counts={}
    with torch.device('meta'):
        for row in rows:
            cfg=ExperimentConfig.from_dict(row['config_resolved']);counts[row['variant']]=sum(p.numel() for p in build_backbone(cfg).parameters())
            path=campaign.materialize(cfg,tmp_path/(row['id']+'.yaml'));assert load_config(path).to_dict()==cfg.to_dict()
    assert counts=={'transformer':2164992,'transition':86785,'pyramid_shared':86979,'transformer_matched':86400}
    for variant in VARIANTS:assert {r['gpu_slot'] for r in rows if r['variant']==variant}=={'0','1'}
    from llm2.research import stories_bootstrap as boot
    for name,digest in [('cycle2_bootstrap',boot.HELPER_SHA256),('kaggle_light',boot.GUARD_SHA256)]:
        raw=subprocess.check_output(['git','show',f'{boot.HELPER_REVISION}:src/llm2/research/{name}.py'],cwd=Path(__file__).resolve().parents[1])
        assert hashlib.sha256(raw).hexdigest()==digest


def test_disabled_invalid_bootstrap_has_no_network():
    with patch('llm2.research.stories_bootstrap.urlopen',side_effect=AssertionError('network')):
        assert bootstrap_git('unused')=={'status':'not_started'}
        with pytest.raises(ValueError):bootstrap_git('bad',run=True)
        with pytest.raises(ValueError):bootstrap_git('a'*40,run=True,seeds=[311])


@pytest.mark.parametrize('failure',['worker','deadline','disk'])
def test_failure_deadline_disk_export_partial(tmp_path,failure):
    output=tmp_path/'llm2-campaign-0032';foreign=tmp_path/'keep.txt';foreign.write_text('keep')
    if failure=='disk':context=patch.object(campaign.shutil,'disk_usage',return_value=type('Usage',(),{'free':0})())
    else:context=patch.object(campaign,'run_pair',side_effect=campaign.SessionBudgetExceeded('deadline') if failure=='deadline' else RuntimeError('worker failure'))
    with context:
        if failure=='deadline':value=campaign.execute(args(output));assert value['status']=='partial_budget'
        else:
            with pytest.raises(RuntimeError,match='evidence exported'):campaign.execute(args(output))
    pair=verify_pair(tmp_path/'llm2-campaign-0032-results.zip',tmp_path/'llm2-campaign-0032-checkpoints.zip')
    assert pair['campaign']['status'] in ('failed','partial_budget') and foreign.read_text()=='keep'


def test_full_cpu_language_campaign_calibration_texts_and_native_reuse(tmp_path):
    output=tmp_path/'llm2-campaign-0032';value=campaign.execute(args(output))
    assert value['status']=='complete' and not value['pending_ids']
    pair=verify_pair(tmp_path/'llm2-campaign-0032-results.zip',tmp_path/'llm2-campaign-0032-checkpoints.zip',require_complete=True)
    assert len(pair['covered'])==4 and sum(n=='source.zip' for n in pair['results_members'])==1
    assert not any(n.endswith(('.safetensors','.pt','.bin')) for n in pair['results_members'])
    for j in value['runs']:
        assert j['step']==4 and j['counters']['tokens_processed']==512
        assert len(j['generation'])==2 and len(j['profiles'])==2
        assert j['config_resolved']['optimizer']['lr']==j['learning_rate']
        assert load_config(j['resolved_config_file']).to_dict()==j['config_resolved']
        assert j['evaluations'][-1]['bytes']==value['counting_baseline']['validation_bytes']
        assert len(list((Path(j['directory'])/'checkpoints').glob('step-*')))==1
    assert not list((output/'calibration').rglob('*.pt'))
    with patch.object(campaign,'run_pair',side_effect=AssertionError('unneeded retraining')):
        again=campaign.execute(args(tmp_path/'llm2-campaign-0033'))
    assert again['status']=='complete' and len(again['historical_runs'])==4 and not again['runs']
