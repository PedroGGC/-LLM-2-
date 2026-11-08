import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import zipfile
import pytest
from llm2.research import recall_campaign as run
from llm2.research.recall_protocol import configuration,jobs,VARIANTS,PROTOCOL_ID
from llm2.research import recall_bootstrap as bootstrap
from llm2.research.session import SessionBudgetExceeded
from llm2.research.campaign_storage import verify_pair
from llm2.config import load_config
from llm2.research.state_study import materialize


def args(output):
    return SimpleNamespace(output=str(output),device='cpu',gpus=['0','1'],engineering_smoke=True,seeds=[271],
        session_seconds=600,bootstrap_elapsed_seconds=0,started=run.time.monotonic(),bootstrap_metadata=None)


def test_parameters_and_seed_slot_rotation():
    alljobs=jobs(Path('out'),['0','1'])
    assert len(alljobs)==14
    count={j['variant']:j['parameters'] for j in alljobs}
    assert count['transformer']==2164992 and count['recurrent']==2174232
    assert count['hierarchy']==1136652
    assert len({count[v] for v in VARIANTS if v.startswith('memory_')})==1
    assert count['memory_all']==1284878
    for v in VARIANTS:
        assert [j['gpu_slot'] for j in alljobs if j['variant']==v] in (['0','1'],['1','0'])


def test_admission_keeps_whole_groups_and_horizon():
    rows=jobs(Path('out'),['0','1'],seeds=[271]);groups=run.batches(rows)
    p={'command_wall_seconds':30.,'median_step_seconds':.05,'validation_seconds':1.}
    profiles={v:(p,p) for v in VARIANTS}
    selected,cost=run.admit(groups,profiles,100000.,1.2)
    first=max(j['estimated_train_seconds'] for j in selected[0])+sum(j['estimated_post_seconds'] for j in selected[0])
    limited,_=run.admit(groups,profiles,first+.1,1.2)
    assert [j['variant'] for j in limited[0]]==['transformer','memory_all'] and len(limited)==1
    assert all(j['max_steps']==j['config_resolved']['optimizer']['schedule_steps']==3000 for j in rows)
    assert run.admit(groups,profiles,1.,1.2)==([],0.)


def test_procedural_config_paths_survive_yaml_resolution(tmp_path):
    for v in VARIANTS:
        cfg=configuration(v,tmp_path/v,271,smoke=True,device='cpu')
        path=materialize(cfg,tmp_path/'configs'/(v+'.yaml'))
        assert load_config(path).to_dict()==cfg.to_dict()


@pytest.mark.parametrize('failure',['worker','disk','deadline'])
def test_failure_exports_valid_partial_without_touching_other_data(tmp_path,monkeypatch,failure):
    preserved=tmp_path/'original.bin';preserved.write_bytes(b'keep')
    def fail(*a,**k):
        if failure=='deadline':raise SessionBudgetExceeded('injected deadline')
        raise RuntimeError('injected worker failure')
    monkeypatch.setattr(run,'worker',fail)
    if failure=='disk':monkeypatch.setattr(run.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    output=tmp_path/'llm2-campaign-0030'
    if failure=='deadline':result=run.execute(args(output));assert result['status']=='partial_budget'
    else:
        with pytest.raises(RuntimeError,match='evidence exported'):run.execute(args(output))
    assert preserved.read_bytes()==b'keep'
    result=tmp_path/'llm2-campaign-0030-results.zip';check=tmp_path/'llm2-campaign-0030-checkpoints.zip'
    verify_pair(result,check)
    with zipfile.ZipFile(result) as z:
        assert z.testzip() is None and 'report/report.html' in z.namelist()
        assert not any(n.endswith(('.pt','.safetensors')) for n in z.namelist())


def test_bootstrap_disabled_and_limits_do_not_fetch(monkeypatch):
    monkeypatch.setattr(bootstrap,'fetch',lambda *a:pytest.fail('network not allowed'))
    assert bootstrap.bootstrap_git('bad',run=False)=={'status':'not_started'}
    for settings in ({'revision':'bad'},{'revision':'a'*40,'session_seconds':7201},{'revision':'a'*40,'seeds':[999]}):
        with pytest.raises(ValueError):bootstrap.bootstrap_git(run=True,**settings)


def test_full_cpu_engineering_campaign_and_history_reuse(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),PYTHONIOENCODING='utf-8')
    def execute(number):
        output=tmp_path/f'llm2-campaign-{number:04d}'
        cp=subprocess.run([sys.executable,'-m','llm2.research.recall_campaign','--output',str(output),
            '--device','cpu','--engineering-smoke','--seeds','271','--session-seconds','600'],
            cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=400)
        assert cp.returncode==0,cp.stdout+'\n'+cp.stderr
        return output,json.loads((output/'campaign.json').read_text(encoding='utf-8'))
    output,c=execute(30)
    assert c['status']=='complete' and len(c['runs'])==7 and c['pending_ids']==[]
    assert c['concurrent_pilot'] and len(c['pilot_profiles'])==7
    memory=[j for j in c['runs'] if j['variant'].startswith('memory_')]
    assert len({j['initial_parameters_sha256'] for j in memory})==1
    for j in c['runs']:
        assert j['step']==4 and j['counters']['tokens_processed']==64
        assert j['post']['status']=='complete' and len(j['post']['heldout'])==2
        assert (Path(j['checkpoint'])/'model.safetensors').is_file()
        assert not (Path(j['directory'])/'checkpoints/step-00000002').exists()
    result=output.with_name(output.name+'-results.zip');check=output.with_name(output.name+'-checkpoints.zip')
    verify_pair(result,check,require_complete=True)
    with zipfile.ZipFile(result) as z:
        assert sum(n.endswith('source.zip') for n in z.namelist())==1
        assert not any('after-' in n or 'compiler-cache' in n or n.endswith(('.pt','.bin','.safetensors')) for n in z.namelist())
    _,again=execute(31)
    assert again['status']=='complete' and again['runs']==[] and len(again['historical_runs'])==7
    # A completed train with interrupted post is reused without repeating its steps.
    original=copy.deepcopy(c);c['runs'][0].pop('post');c['runs'][0]['status']='trained'
    (output/'campaign.json').write_text(json.dumps(c),encoding='utf-8')
    # Remove only the test-owned second manifest from discovery, preserving its files.
    (tmp_path/'llm2-campaign-0031/campaign.json').rename(tmp_path/'llm2-campaign-0031/campaign.saved.json')
    _,recovered=execute(32)
    assert recovered['runs']==[] and len(recovered['historical_runs'])==7 and recovered['status']=='complete'
    assert recovered['historical_runs'][0]['post']['status']=='complete'
