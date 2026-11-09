import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
import pytest
from llm2.research import binding_campaign as campaign
from llm2.research.binding_bootstrap import bootstrap_git
from llm2.research.binding_protocol import jobs,configuration,PROTOCOL_ID
from llm2.research.binding_data import TASK_ID
from llm2.research.campaign_storage import verify_pair
from llm2.config import load_config


def args(output):
    return argparse.Namespace(output=str(output),device='cpu',gpus=['0','1'],seeds=[311],engineering_smoke=True,
        session_seconds=900,bootstrap_elapsed_seconds=0,bootstrap_metadata=None,started=time.monotonic())


def test_counts_roundtrip_and_gpu_rotation(tmp_path):
    rows=jobs(tmp_path,['0','1'])
    assert len(rows)==8
    from llm2.research.factory import build_backbone
    import torch
    counts={}
    with torch.device('meta'):
        for row in rows:
            cfg=configuration(row['variant'],tmp_path/row['id'],row['seed'])
            counts[row['variant']]=sum(p.numel() for p in build_backbone(cfg).parameters())
            path=campaign.materialize(cfg,tmp_path/(row['id']+'.yaml'))
            assert load_config(path).to_dict()==cfg.to_dict()
    assert counts=={'transformer':2164992,'transition':86785,'pyramid_shared':86979,'pyramid_independent':111652}
    for v in counts:
        assigned=[row['gpu_slot'] for row in rows if row['variant']==v]
        assert assigned==(['0','1'] if assigned[0]=='0' else ['1','0'])


def test_disabled_invalid_bootstrap_and_frozen_helpers():
    with patch('llm2.research.binding_bootstrap.urlopen',side_effect=AssertionError('network')):
        assert bootstrap_git('unused')=={'status':'not_started'}
        with pytest.raises(ValueError):bootstrap_git('bad',run=True)
        with pytest.raises(ValueError):bootstrap_git('a'*40,run=True,seeds=[271])
    from llm2.research import binding_bootstrap as b
    root=Path(__file__).resolve().parents[1]
    for name,expected in [('cycle2_bootstrap',b.HELPER_SHA256),('kaggle_light',b.GUARD_SHA256)]:
        raw=subprocess.check_output(['git','show',f'{b.HELPER_REVISION}:src/llm2/research/{name}.py'],cwd=root)
        assert hashlib.sha256(raw).hexdigest()==expected


def test_worker_failure_exports_partial_and_preserves_unrelated_data(tmp_path):
    foreign=tmp_path/'keep.txt';foreign.write_text('keep')
    output=tmp_path/'llm2-campaign-0031'
    with patch.object(campaign,'run_pair',side_effect=RuntimeError('injected failure')):
        with pytest.raises(RuntimeError,match='evidence exported'):campaign.execute(args(output))
    result=verify_pair(tmp_path/'llm2-campaign-0031-results.zip',tmp_path/'llm2-campaign-0031-checkpoints.zip')
    assert result['campaign']['status']=='failed' and result['campaign']['dataset_sha256']==TASK_ID
    assert foreign.read_text()=='keep'


def test_low_disk_exports_partial(tmp_path):
    output=tmp_path/'llm2-campaign-0031'
    usage=type('Usage',(),{'free':0})()
    with patch.object(campaign.shutil,'disk_usage',return_value=usage):
        with pytest.raises(RuntimeError):campaign.execute(args(output))
    value=verify_pair(tmp_path/'llm2-campaign-0031-results.zip',tmp_path/'llm2-campaign-0031-checkpoints.zip')
    assert value['campaign']['status']=='failed'


def test_deadline_preserves_export_reserve_and_partial_bundle(tmp_path):
    output=tmp_path/'llm2-campaign-0031'
    with patch.object(campaign,'run_pair',side_effect=campaign.SessionBudgetExceeded('injected deadline')):
        value=campaign.execute(args(output))
    assert value['status']=='partial_budget'
    pair=verify_pair(tmp_path/'llm2-campaign-0031-results.zip',tmp_path/'llm2-campaign-0031-checkpoints.zip')
    assert pair['campaign']['session_budget']['export_reserve_seconds']==300


def test_full_cpu_engineering_campaign_and_native_reuse(tmp_path):
    output=tmp_path/'llm2-campaign-0031'
    result=campaign.execute(args(output))
    assert result['status']=='complete' and not result['pending_ids']
    pair=verify_pair(tmp_path/'llm2-campaign-0031-results.zip',tmp_path/'llm2-campaign-0031-checkpoints.zip',require_complete=True)
    assert len(pair['covered'])==4
    assert sum(n=='source.zip' for n in pair['results_members'])==1
    assert not any(n.endswith(('.pt','.safetensors')) for n in pair['results_members'])
    for row in result['runs']:
        assert row['step']==4 and row['counters']['step']==4
        assert row['counters']['tokens_processed']==4*2*2*4
        assert len(row['profiles'])==2 and row['training_profile']['weights_and_optimizer_restored']
        assert len(list((Path(row['directory'])/'checkpoints').glob('step-*')))==1
    # New work uses verified original tensors rather than duplicating/retraining.
    with patch.object(campaign,'run_pair',side_effect=AssertionError('unneeded training')):
        reuse=campaign.execute(args(tmp_path/'llm2-campaign-0032'))
    assert reuse['status']=='complete' and len(reuse['historical_runs'])==4 and not reuse['runs']
    report=json.loads((tmp_path/'llm2-campaign-0032/report/report.json').read_text())
    assert report['protocol_id']==PROTOCOL_ID and len(report['rows'])==8
