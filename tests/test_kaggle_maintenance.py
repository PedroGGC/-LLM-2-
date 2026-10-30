import hashlib
import json
from pathlib import Path
import subprocess
import runpy
import io
import time
import urllib.request
import zipfile

import pytest

from llm2.research import kaggle_maintenance as maintenance


@pytest.fixture
def cache(tmp_path,monkeypatch):
    monkeypatch.setattr(maintenance,'active_workers',lambda:[])
    parent=tmp_path/'llm2-cycle2-git-test';source=parent/'source';source.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=source,stderr=subprocess.STDOUT)
    git('init')
    (source/'.gitignore').write_text('__pycache__/\nruns/\n.env\n')
    (source/'core.py').write_text('value=1\n')
    git('add','.')
    git('-c','user.name=Maintenance Test','-c','user.email=maintenance@example.invalid','commit','-m','fixture')
    git('remote','add','origin',maintenance.REMOTE)
    (parent/'bootstrap.json').write_text('{}')
    return tmp_path,parent,source,git


def test_only_clean_code_cache_removed_and_data_archives_registry_preserved(cache):
    base,parent,source,git=cache
    (source/'__pycache__').mkdir();(source/'__pycache__/core.pyc').write_bytes(b'cache')
    outputs=base/'llm2-campaign-0021';outputs.mkdir();(outputs/'model.safetensors').write_bytes(b'weights')
    archive=base/'llm2-campaign-0021-after-01-results.zip';archive.write_bytes(b'keep archive')
    dataset=base/'dataset';dataset.mkdir();(dataset/'train.bin').write_bytes(b'data')
    reservation=base/'.llm2-numbering/llm2-campaign/0021';reservation.mkdir(parents=True)
    result=maintenance.clean_working(base,apply=True,input_root=base/'inputs')
    assert not source.exists() and (parent/'bootstrap.json').exists()
    assert (outputs/'model.safetensors').read_bytes()==b'weights' and archive.read_bytes()==b'keep archive'
    assert (dataset/'train.bin').read_bytes()==b'data' and reservation.exists()
    assert result['code_caches'][0]['deleted'] and result['next_run_number']==22


@pytest.mark.parametrize('change',['modified','untracked','ignored_payload','tracked_payload','wrong_remote'])
def test_non_reproducible_or_data_bearing_cache_is_preserved(cache,change):
    base,parent,source,git=cache
    if change=='modified':(source/'core.py').write_text('changed\n')
    if change=='untracked':(source/'notes.txt').write_text('keep\n')
    if change=='ignored_payload':
        (source/'runs').mkdir();(source/'runs/state.pt').write_bytes(b'state')
    if change=='tracked_payload':
        (source/'weights.safetensors').write_bytes(b'weights');git('add','weights.safetensors')
        git('-c','user.name=Maintenance Test','-c','user.email=maintenance@example.invalid','commit','-m','weights')
    if change=='wrong_remote':git('remote','set-url','origin','https://example.invalid/other.git')
    result=maintenance.clean_working(base,apply=True,input_root=base/'inputs')
    assert source.exists() and not result['code_caches'] and result['protected_caches']


def test_active_worker_blocks_cleanup_and_new_batch(cache,monkeypatch):
    base,parent,source,git=cache
    monkeypatch.setattr(maintenance,'active_workers',lambda:[{'pid':123,'program':'python'}])
    result=maintenance.clean_working(base,apply=True,input_root=base/'inputs')
    assert source.exists() and not result['safe_to_start_new_batch'] and not result['code_caches']


def test_target_outside_base_is_rejected_before_removal(cache,tmp_path):
    base,parent,source,git=cache
    other=base/'other';other.mkdir()
    with pytest.raises(ValueError,match='owned'):maintenance.code_cache_targets(parent,other)
    assert source.exists()


def test_dry_run_and_numbering_consider_archives_inputs_and_reservations(cache):
    base,parent,source,git=cache
    reservation=base/'.llm2-numbering/llm2-campaign/0022';reservation.mkdir(parents=True)
    inputs=base/'inputs';inputs.mkdir();(inputs/'llm2-campaign-0023-after-01-results.zip').write_bytes(b'placeholder')
    result=maintenance.clean_working(base,input_root=inputs)
    assert source.exists() and result['next_run_number']==24 and not result['code_caches'][0]['deleted']


def test_partial_checkpoint_blocks_restart_and_completed_export_is_recognized(tmp_path,monkeypatch):
    monkeypatch.setattr(maintenance,'active_workers',lambda:[])
    identity={'test':'fixture'}
    monkeypatch.setattr(maintenance,'SCIENCE_IDENTITY_SHA256',hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
    monkeypatch.setattr(maintenance,'FLRM_103_SHA256','a'*64)
    campaign=tmp_path/'llm2-campaign-0021';campaign.mkdir()
    jobs=[{'id':'anchor_flrm-seed-103','arm':'anchor_flrm','seed':103,'status':'complete',
           'step':8000,'checkpoint_model_sha256':'a'*64},
          {'id':'flops_transformer-seed-103','arm':'flops_transformer','seed':103,'status':'pending'}]
    (campaign/'campaign.json').write_text(json.dumps({'runs':jobs}))
    with zipfile.ZipFile(tmp_path/'llm2-campaign-0021-after-01-results.zip','w') as z:
        z.writestr('report/report.json',json.dumps({'identity':identity,'rows':[jobs[0]]}))
    before=maintenance.clean_working(tmp_path,input_root=tmp_path/'inputs')
    assert before['safe_to_start_new_batch'] and before['campaign_0021'][0]['complete_in_results_zip']
    state=campaign/'runs/flops_transformer-seed-103/checkpoints/step-00004868';state.mkdir(parents=True)
    after=maintenance.clean_working(tmp_path,input_root=tmp_path/'inputs')
    assert not after['safe_to_start_new_batch'] and after['warnings']
    assert state.exists()


def test_completed_without_export_is_not_assumed_pending(tmp_path,monkeypatch):
    monkeypatch.setattr(maintenance,'active_workers',lambda:[])
    campaign=tmp_path/'llm2-campaign-0021';campaign.mkdir()
    (campaign/'campaign.json').write_text(json.dumps({'runs':[{'id':'run','arm':'anchor_flrm','seed':103,
        'status':'complete','checkpoint_model_sha256':'a'*64}]}))
    result=maintenance.clean_working(tmp_path,input_root=tmp_path/'inputs')
    assert not result['safe_to_start_new_batch'] and 'ZIP' in result['warnings'][0]


def test_recovery_cell_requires_safe_status_before_network(monkeypatch):
    builder=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_cycle3_short_release.py'))
    def forbidden(*a,**k):raise AssertionError('download before maintenance check')
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    code=builder['cell']('a'*40,run_number=22,max_runs=1,seeds=[103],maintenance_guard=True)
    for namespace in [{},{'WORKING_INFO':{'safe_to_start_new_batch':False}}]:
        with pytest.raises(RuntimeError,match='manutencao'):exec(compile(code,'recovery','exec'),namespace)


def test_recovery_uses_next_number_and_single_pending_model(monkeypatch):
    builder=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_cycle3_short_release.py'))
    payload=b'def bootstrap_git(revision, **kwargs):\n    global captured\n    captured=(revision,kwargs)\n'
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:io.BytesIO(payload))
    started=time.monotonic()
    namespace={'WORKING_INFO':{'safe_to_start_new_batch':True,'next_run_number':24},
               'maintenance_namespace':{'active_workers':lambda:[]},'maintenance_started_at':started}
    code=builder['cell']('a'*40,run_number=22,max_runs=1,seeds=[103],maintenance_guard=True)
    exec(compile(code,'recovery','exec'),namespace)
    revision,settings=namespace['namespace']['captured']
    assert settings['run_number']==24 and settings['max_runs']==1 and settings['seeds']==[103]
    assert settings['started_at']==started and settings['session_seconds']==3600


def test_recovery_notebook_is_inert_and_includes_maintenance(tmp_path,monkeypatch):
    builder=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_cycle3_short_release.py'))
    result=builder['build'](tmp_path,'a'*40,run_number=22,max_runs=1,seeds=[103],maintenance_guard=True)
    def forbidden(*a,**k):raise AssertionError('inactive notebook downloaded')
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    notebook=json.loads((tmp_path/'rodar_ciclo3_kaggle_1h.ipynb').read_text(encoding='utf-8'))
    assert result['maintenance_guard'] and result['max_runs']==1
    namespace={}
    for cell in notebook['cells']:
        if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'notebook','exec'),namespace)
    assert len(notebook['cells'])==3
