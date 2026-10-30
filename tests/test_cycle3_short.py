import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import runpy

import pytest

from llm2.research import cycle3_short as short, cycle3_short_bootstrap as bootstrap
from llm2.research.cycle3_protocol import ordered_jobs
from llm2.research.session import SessionBudget, SessionBudgetExceeded

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def history():
    return json.loads((ROOT/'docs/results/campaign_0019_ledger.json').read_text())['report']


def test_queue_reuses_twelve_completed_runs_and_selects_primary_pair(history,tmp_path):
    jobs=ordered_jobs([101,102,103,104,105],['0','1'],tmp_path/'data',tmp_path/'out')
    selected,pending=short.select_jobs(jobs,history,2)
    assert [j['id'] for j in selected]==['anchor_flrm-seed-103','flops_transformer-seed-103']
    assert len(pending)==16
    assert [j['max_steps'] for j in selected]==[8000,19472]
    assert [j['gpu_slot'] for j in selected]==['0','1']
    extra=copy.deepcopy(history)
    extra['rows'] += [{'arm':j['arm'],'seed':j['seed']} for j in selected]
    selected,_=short.select_jobs(jobs,extra,2)
    assert [j['id'] for j in selected]==['flops_lrm-seed-103','flops_transformer_deep-seed-103']


def test_history_deduplicates_copies_but_rejects_conflicting_weights_and_identities(history):
    result=short.combine([history,copy.deepcopy(history)],history['identity'])
    assert len(result['rows'])==12 and len(result['curves'])==48
    assert result['complete_seeds']==[101,102] and not result['confirmation_eligible']
    for field in ['checkpoint_model_sha256','validation_loss','max_steps']:
        other=copy.deepcopy(history);other['rows'][0][field]='conflict'
        with pytest.raises(ValueError,match='conflicting'):short.combine([history,other],history['identity'])
    for field in ['dataset_sha256','source_sha256','versioned_configs_sha256']:
        other=copy.deepcopy(history);other['identity'][field]='changed'
        with pytest.raises(ValueError,match='incompatible'):short.combine([other],history['identity'])


def test_focused_validation_contains_training_invariants_and_compiled_selected_families():
    models=['baseline_small','flrm_exchange_no_film']
    cpu,gpu=short.validation_commands(models)
    assert 'tests/test_json_portability.py' in cpu
    for model in models:
        assert any('causal_backward_finite_memorization['+model+']' in s for s in cpu)
        assert any('strict_resume_dropout_accumulation['+model+']' in s for s in cpu)
        assert any('cuda_fp32_inductor_backward_checkpoint['+model+']' in s for s in gpu)
    assert not any('jepa' in s and 'no_jepa' not in s for s in cpu+gpu)
    assert short.validation_commands(models,'full')==[
        ['-m','pytest','-q','-m','not cuda'],['-m','pytest','-q','-m','cuda','-k','fp32']]


def test_one_hour_plan_has_no_files_or_network(tmp_path,monkeypatch):
    def unwanted(*a,**k):raise AssertionError('network started')
    monkeypatch.setattr(bootstrap,'urlopen',unwanted)
    assert bootstrap.bootstrap_git('x',run=False)=={'status':'not_started'}
    output=tmp_path/'absent'
    result=short.main(['--cache','absent','--output',str(output)])
    assert result['session_seconds']==3600 and result['max_runs']==2 and not output.exists()


@pytest.mark.parametrize('seconds',[0,300,3601,float('nan'),float('inf')])
def test_short_bootstrap_rejects_bad_settings_before_download(monkeypatch,seconds):
    def unwanted(*a,**k):raise AssertionError('network started')
    monkeypatch.setattr(bootstrap,'urlopen',unwanted)
    with pytest.raises(ValueError):bootstrap.bootstrap_git('a'*40,run=True,session_seconds=seconds)


def test_snapshot_contains_canonical_report_history_and_is_exclusive(tmp_path,history):
    folder=tmp_path/'campaign';folder.mkdir()
    (folder/'campaign.json').write_text('{}')
    report=folder/'reports/after-01'
    for category in ['report','history']:
        (report/category).mkdir(parents=True)
        (report/category/'report.json').write_text(json.dumps(history))
    c=SimpleNamespace(output=folder,manifest={'runs':[]})
    destination=tmp_path/'snapshot-results.zip'
    short.export(c,destination,report)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None and len(archive.namelist())==len(set(archive.namelist()))
        assert json.loads(archive.read('report/report.json'))['identity']==history['identity']
        assert json.loads(archive.read('history/report.json'))['identity']==history['identity']
    assert len(short.load_report(destination)['rows'])==12
    before=destination.read_bytes()
    with pytest.raises(FileExistsError):short.export(c,destination,report)
    assert destination.read_bytes()==before


def test_deadline_allows_one_completed_run_without_requiring_six():
    now=[0.];budget=SessionBudget(3600,300,elapsed=200,clock=lambda:now[0])
    budget.require(1200);now[0]+=1200
    budget.require(1200);now[0]+=1200
    with pytest.raises(SessionBudgetExceeded):budget.require(1200)
    assert budget.work_remaining==700


def test_short_report_keeps_independent_code_identity_and_full_seed_requirement(tmp_path,history):
    c=SimpleNamespace(output=tmp_path/'campaign',manifest={'runs':[],
        'provenance':{k:history['identity'][k] for k in ['source_sha256','versioned_configs_sha256']},
        'dataset_sha256':history['identity']['dataset_sha256'],'engineering_smoke':False,'tests_skipped':False})
    result=short.reports(c,history,c.output/'reports/after-01')
    assert result['identity']==history['identity'] and not result['confirmation_eligible']
    queue=json.loads((c.output/'reports/after-01/history/queue.json').read_text())
    assert queue['completed_runs']==12 and len(queue['pending_run_ids'])==18
    pairs=json.loads((c.output/'reports/after-01/history/available_pairs.json').read_text())
    assert pairs['effects_exploratory']['flops_transformer']['n']==2


def test_snapshot_rejects_duplicate_names_in_untrusted_archive(tmp_path):
    path=tmp_path/'duplicate.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('report/report.json','{}')
        with pytest.warns(UserWarning):archive.writestr('report/report.json','{}')
    with pytest.raises(ValueError,match='duplicate'):short.load_report(path)


def test_bootstrap_preserves_one_clock_and_separates_operator_worker(tmp_path,monkeypatch):
    from llm2.research.cycle2_bootstrap import _BootstrapSession
    operator=tmp_path/'source';research=operator/'src/llm2/research';research.mkdir(parents=True)
    payload=b'# helper';(research/'cycle2_bootstrap.py').write_bytes(payload)
    worker=tmp_path/'scientific-worker'
    naming=worker/'src/llm2/research/naming.py';naming.parent.mkdir(parents=True)
    naming.write_text('def numbered_output(work,**kwargs):\n    return work/"llm2-campaign-0020"\n')
    ledger=operator/'docs/results/campaign_0019_ledger.json';ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({'report':{'protocol_id':'frozen','identity':{
        'source_sha256':{'core.py':'unchanged'},'versioned_configs_sha256':{}}}}))
    now=[160.];commands=[]
    monkeypatch.setattr(bootstrap.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(bootstrap,'urlopen',lambda *a,**k:io.BytesIO(payload))
    monkeypatch.setattr(bootstrap.importlib.metadata,'version',lambda *a:'unchanged')
    monkeypatch.setattr(bootstrap,'checkout_hashes',lambda *a:(bootstrap.WORKER_REVISION,{'core.py':'unchanged'}))
    monkeypatch.setattr(bootstrap,'previous_results',lambda *a:[])
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    def command(arguments,**kwargs):
        commands.append((list(map(str,arguments)),kwargs))
        if 'pip' in arguments:now[0]+=20
        if 'archive' in arguments:Path(arguments[arguments.index('--output')+1]).write_bytes(b'operator source')
    def inject(code,namespace):
        namespace.update(_BootstrapSession=_BootstrapSession,_hardware=lambda:(tmp_path,['0','1'],'unchanged'),
            checkout_git_source=lambda *a:(operator,tmp_path,{'source_files_sha256':{'operator':'separate'}}),bounded_command=command)
    monkeypatch.setattr(bootstrap,'exec',inject,raising=False)
    bootstrap.bootstrap_git('a'*40,run=True,started_at=100.)
    arguments,options=commands[-1]
    assert arguments[arguments.index('--worker-root')+1]==str(worker)
    assert arguments[arguments.index('--session-seconds')+1]=='3600.0'
    assert float(arguments[arguments.index('--bootstrap-elapsed-seconds')+1])==80
    assert options['timeout']==3520
    assert options['env']['PYTHONPATH']==str(worker/'src')
    assert options['env']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    meta=json.loads((tmp_path/'bootstrap-short.json').read_text())
    assert meta['source_git_revision']==bootstrap.WORKER_REVISION and meta['operator_git_revision']=='a'*40
    assert meta['source_files_sha256']!=meta['operator_source_files_sha256']


def test_discovery_uses_full_and_progress_reports_only_for_same_protocol(tmp_path,history):
    for name,report,prefix in [('0019',history,'report'),('0020-after-01',history,'history'),('old',{'protocol_id':'other'},'report')]:
        with zipfile.ZipFile(tmp_path/f'llm2-campaign-{name}-results.zip','w') as archive:
            archive.writestr(prefix+'/report.json',json.dumps(report))
    found=bootstrap.previous_results(tmp_path,history['protocol_id'])
    assert len(found)==2


def test_short_delivery_is_inert_and_records_one_hour_cap(tmp_path):
    module=runpy.run_path(str(ROOT/'scripts/build_cycle3_short_release.py'))
    exec(compile(module['cell']('a'*40,False),'inactive','exec'),{})
    result=module['build'](tmp_path,'a'*40)
    assert result['max_runs']==2 and result['session_seconds']==3600
    assert 'cycle3_short_bootstrap.py' in (tmp_path/'rodar_ciclo3_kaggle_1h.py').read_text()
    with pytest.raises(FileExistsError):module['build'](tmp_path,'a'*40)
