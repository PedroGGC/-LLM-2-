import copy
import hashlib
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import pytest
import torch
import yaml

from llm2.cli import main as cli_main
from llm2.data import PreparedData
from llm2.document_data import prepare_documents
from llm2.objectives import CausalLanguageModeling
from llm2.research import cycle3, cycle3_bootstrap, wikitext
from llm2.research.cycle3_evaluate import bundle
from llm2.research.cycle3_protocol import ARMS, MODELS, SEEDS, configuration, ordered_jobs, paired_fingerprints, verify_dataset
from llm2.research.cycle3_report import analyze, merge
from llm2.research.factory import build_backbone
from llm2.research.session import SessionBudgetExceeded
from llm2.runtime import attention_context
from llm2.trainer import Trainer
from test_training import training_config, assert_nested_equal
from test_compressed_optimization import _state

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def explicit_data(tmp_path):
    raw=tmp_path/'documents.jsonl'
    raw.write_text(''.join(json.dumps({'id':s,'split':s,'text':(s+' article and patterns. ')*25})+'\n'
                           for s in ['train','val','test']),encoding='utf-8')
    destination=tmp_path/'documents'
    prepare_documents([raw],destination)
    return destination


def test_frozen_protocol_horizons_cost_and_initialization(tmp_path):
    jobs=ordered_jobs(SEEDS,['0','1'],tmp_path/'data',tmp_path/'output')
    assert len(jobs)==30
    first=jobs[:6]
    assert [j['max_steps'] for j in first]==[8000,8000,8000,19472,10620,11396]
    assert [j['registered_parameters'] for j in first]==[2164224,2017536,2165772,2164224,2017536,3639552]
    assert [j['arm'] for j in jobs[:6]]!=[j['arm'] for j in jobs[6:12]]
    for job in jobs:
        cfg=job['config_resolved']
        assert cfg['optimizer']['schedule_steps']==job['max_steps']
        assert cfg['data']['batch_size']*cfg['train']['grad_accum_steps']==32
        assert cfg['train']['compile'] and cfg['train']['precision']=='fp32'
        assert not cfg['research']['scale_conditioning'] and cfg['research']['cross_scale_exchange']
        assert cfg['research']['context_normalization']=='none'
        assert cfg['research']['jepa_weight']==cfg['research']['mtp_weight']==cfg['research']['ncm_slots']==0
        assert not cfg['research']['v_cycle'] and not cfg['research']['inv_vcycle']
        if job['arm'].startswith('flops_'):
            assert 0<=job['relative_flop_budget_shortfall']<.0003
    configs={name:configuration(name,tmp_path/'data',tmp_path/name,101) for name in MODELS}
    assert paired_fingerprints(configs)['verified_common_equal']
    assert len([j for j in first if j['shared_anchor']])==1


def test_plan_is_inert_and_smoke_not_science(tmp_path,monkeypatch):
    def unexpected(*a,**k): raise AssertionError('network during planning')
    monkeypatch.setattr(cycle3_bootstrap,'urlopen',unexpected)
    assert cycle3_bootstrap.bootstrap_git('x',run=False)=={'status':'not_started'}
    args=cycle3.parser().parse_args(['--cache','missing','--output',str(tmp_path/'absent')])
    result=cycle3.plan(args)
    assert result['session_budget']['limit_seconds']==14400 and len(result['jobs'])==30
    assert not (tmp_path/'absent').exists()
    args.device='cpu'
    with pytest.raises(ValueError,match='engineering'):cycle3.plan(args)


@pytest.mark.parametrize('name',MODELS)
@attention_context(True)
def test_selected_models_causal_backward_finite_memorization(tmp_path,name):
    cfg=configuration(name,tmp_path/'unused',tmp_path/'run',101,device='cpu',smoke=True)
    cfg.model.dropout=0
    model=build_backbone(cfg)
    ids=torch.arange(18).repeat(2,1)
    batch={'input_ids':ids[:,:-1],'targets':ids[:,1:]}
    output=model(batch['input_ids']).logits
    changed=batch['input_ids'].clone();changed[:,8:]+=3
    torch.testing.assert_close(model(changed).logits[:,:8],output[:,:8],atol=0,rtol=0)
    objective=CausalLanguageModeling();initial=objective(model,batch).loss.item()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True);loss=objective(model,batch).loss
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert objective(model,batch).loss.item()<initial/15


@pytest.mark.parametrize('name',MODELS)
def test_selected_models_strict_resume_dropout_accumulation(training_config,tmp_path,name):
    cfg=configuration(name,training_config.data.path,tmp_path/'run',101,device='cpu',smoke=True)
    assert cfg.model.dropout>0 and cfg.train.grad_accum_steps==2 and cfg.train.deterministic
    with Trainer(cfg) as trainer:
        trainer.fit(until_step=2);before=_state(trainer);checkpoint=trainer.last_checkpoint
        expected_metrics=trainer.train_step();expected=_state(trainer)
    cfg.train.output_dir=str(tmp_path/'resumed')
    with Trainer(cfg,resume=checkpoint) as resumed:
        assert_nested_equal(before,_state(resumed));actual=resumed.train_step()
        assert_nested_equal(expected,_state(resumed))
        assert actual['training_loss']==expected_metrics['training_loss']


def test_dataset_contract_and_real_pinned_hashes_optional(explicit_data):
    assert verify_dataset(explicit_data,smoke=True).manifest['schema_version']==2
    with pytest.raises(ValueError,match='pinned'):verify_dataset(explicit_data)


def fake_wikitext(monkeypatch):
    monkeypatch.setattr(wikitext,'fetch_files',lambda *a,**k:None)
    monkeypatch.setattr(wikitext,'parquet_lines',lambda paths:iter([f' = {paths[0].name} = \n',' body\n']))


def test_wikitext_atomic_manifest_before_publication(tmp_path,monkeypatch):
    fake_wikitext(monkeypatch)
    destination=tmp_path/'prepared'
    def fail(*a,**k):raise RuntimeError('injected manifest failure')
    monkeypatch.setattr(wikitext,'write_json',fail)
    with pytest.raises(RuntimeError,match='injected'):wikitext.import_wikitext(tmp_path,destination,train_bytes=1)
    assert not destination.exists() and not list(tmp_path.glob('.prepared-*'))


def test_wikitext_explicit_training_exclusion_unknown_leakage_and_articles(tmp_path,monkeypatch):
    assert list(wikitext.articles(['\n',' = A = \n',' paragraph\n',' = = Section = = \n','text\n',' = B = \n','end\n']))==[
        ' = A = \n paragraph\n = = Section = = \ntext\n',' = B = \nend\n']
    monkeypatch.setattr(wikitext,'fetch_files',lambda *a,**k:None)
    duplicate=' = Duplicate = \n body\n';digest=hashlib.sha256(duplicate.encode()).hexdigest()
    def lines(paths):
        prefix=paths[0].name.split('-')[0]
        return iter((duplicate.splitlines(keepends=True) if prefix in ('train','test') else [])+[f' = {prefix} = \n','body\n'])
    monkeypatch.setattr(wikitext,'parquet_lines',lines)
    with pytest.raises(ValueError,match='leakage'):wikitext.import_wikitext(tmp_path,tmp_path/'bad')
    monkeypatch.setattr(wikitext,'TRAIN_EXCLUSIONS',{digest})
    result=wikitext.import_wikitext(tmp_path,tmp_path/'good',train_bytes=1)
    assert result['corpus']['splits']['train']['excluded_training_overlap_sha256']==[digest]
    assert digest in result['splits']['test']['document_hashes']
    assert digest not in result['splits']['train']['document_hashes']


def test_bundle_matches_cli_and_backup_preserves_weights(explicit_data,tmp_path):
    output=tmp_path/'campaign';output.mkdir()
    cfg=configuration(MODELS[0],explicit_data,output/'runs'/'one',101,device='cpu',smoke=True)
    config=output/'config.yaml';config.write_text(yaml.safe_dump(cfg.to_dict()))
    with Trainer(cfg) as trainer:
        trainer.fit();checkpoint=trainer.last_checkpoint
    result=bundle(str(config),cfg.train.output_dir,device='cpu')
    assert not result['test_evaluated'] and [r['step'] for r in result['checkpoints']]==[1,2,3,4]
    standalone=tmp_path/'standalone.json'
    cli_main(['evaluate','--checkpoint',str(checkpoint),'--device','cpu','--output',str(standalone)])
    assert result['checkpoints'][-1]['validation_loss']==json.loads(standalone.read_text())['validation_loss']
    campaign=SimpleNamespace(output=output,manifest={'runs':[{'directory':cfg.train.output_dir}]})
    backup=cycle3.checkpoint_backup(campaign)
    with zipfile.ZipFile(backup) as archive:
        assert any(n.endswith('model.safetensors') for n in archive.namelist())
        assert any(n.endswith('training_state.pt') for n in archive.namelist())
        assert all('step-00000001' not in n for n in archive.namelist())
        assert archive.testzip() is None
    with pytest.raises(FileExistsError):cycle3.checkpoint_backup(campaign)


def rows_for(seeds):
    env={'python':'fixed','packages':{},'cuda_version':'test','cudnn_version':1,'device':'T4','cublas_workspace_config':':4096:8'}
    return [{'arm':arm,'seed':seed,'validation_loss':1.0 if arm=='anchor_flrm' else 1.01,
             'environment':env} for seed in seeds for arm in ARMS]


def test_confirmation_requires_five_complete_seeds_and_no_smoke():
    for seeds in [SEEDS[:1],SEEDS[:3],SEEDS[:4]]:
        value=analyze(rows_for(seeds),[],identity={})
        assert not value['confirmation_eligible'] and value['decision']=='inconclusive'
    value=analyze(rows_for(SEEDS),[],identity={})
    assert value['confirmation_eligible'] and value['decision']=='quality_advantage_at_matched_estimated_flops'
    assert not analyze(rows_for(SEEDS),[],identity={},smoke=True)['confirmation_eligible']
    with pytest.raises(ValueError,match='duplicate'):analyze(rows_for(SEEDS)+rows_for(SEEDS[:1]),[],identity={})
    assert len(analyze(rows_for(SEEDS)[:-1],[],identity={})['complete_seeds'])==4


def test_merge_rejects_dataset_code_runtime_and_duplicate_seeds(tmp_path):
    first=analyze(rows_for([101]),[],identity={'dataset':'a'})
    second=analyze(rows_for([102]),[],identity={'dataset':'a'})
    a=tmp_path/'a.json';b=tmp_path/'b.json'
    a.write_text(json.dumps(first));b.write_text(json.dumps(second))
    # Patch writer because compact synthetic rows intentionally omit display fields.
    from llm2.research import cycle3_report
    original=cycle3_report.save_report
    try:
        cycle3_report.save_report=lambda *a:None
        assert merge([a,b],tmp_path/'combined')['complete_seeds']==[101,102]
        with pytest.raises(ValueError,match='duplicate'):merge([a,a],tmp_path/'duplicate')
        second['identity']['dataset']='b';b.write_text(json.dumps(second))
        with pytest.raises(ValueError,match='incompatible'):merge([a,b],tmp_path/'changed')
    finally:cycle3_report.save_report=original


def test_partial_budget_exports_honestly(explicit_data,tmp_path,monkeypatch):
    args=cycle3.parser().parse_args(['--data',str(explicit_data),'--execute','--device','cpu','--engineering-smoke',
        '--skip-tests','--output',str(tmp_path/'partial'),'--seeds','101'])
    monkeypatch.setattr(cycle3,'pilot',lambda *a:(_ for _ in ()).throw(SessionBudgetExceeded('simulated')))
    result=cycle3.execute(args)
    assert result['status']=='partial_budget'
    with zipfile.ZipFile(result['results_zip']) as archive:
        report=json.loads(archive.read('report/report.json'))
        assert not report['confirmation_eligible'] and not report['complete_seeds']
        assert report['pending_seeds']==list(SEEDS) and not report['test_evaluated']


def test_short_release_inert_pinned_and_exclusive(tmp_path,monkeypatch):
    module=runpy.run_path(str(ROOT/'scripts/build_cycle3_release.py'))
    code=module['git_cell']('a'*40,run=False)
    exec(compile(code,'cell','exec'),{})
    assert '14400' in code and 'cycle3_bootstrap.py' in code
    result=module['build_git'](tmp_path/'release','a'*40)
    assert result['session_seconds']==14400
    with pytest.raises(FileExistsError):module['build_git'](tmp_path/'release','a'*40)


@pytest.mark.parametrize('seconds',[0,300,14401,float('nan'),float('inf')])
def test_bad_duration_fails_before_download(monkeypatch,seconds):
    def unwanted(*a,**k):raise AssertionError('unexpected download')
    monkeypatch.setattr(cycle3_bootstrap,'urlopen',unwanted)
    with pytest.raises(ValueError):cycle3_bootstrap.bootstrap_git('a'*40,run=True,session_seconds=seconds)


def test_bootstrap_four_hour_clock_dependency_group_and_seed_forwarding(tmp_path,monkeypatch):
    from llm2.research.cycle2_bootstrap import _BootstrapSession
    source=tmp_path/'source';research=source/'src/llm2/research';research.mkdir(parents=True)
    (research/'naming.py').write_text('def numbered_output(work, **kwargs):\n    return work / "llm2-campaign-0018"\n')
    payload=b'# verified helper'
    (research/'cycle2_bootstrap.py').write_bytes(payload)
    now=[160.];commands=[]
    monkeypatch.setattr(cycle3_bootstrap.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(cycle3_bootstrap,'urlopen',lambda *a,**k:io.BytesIO(payload))
    monkeypatch.setattr(cycle3_bootstrap.importlib.metadata,'version',lambda *a:'unchanged')
    def command(arguments,**kwargs):
        commands.append((list(map(str,arguments)),kwargs))
        if 'pip' in arguments:now[0]+=20
    def inject(code,namespace):
        namespace.update(_BootstrapSession=_BootstrapSession,_hardware=lambda:(tmp_path,['0','1'],'unchanged'),
            checkout_git_source=lambda *a:(source,tmp_path,{'source_files_sha256':{}}),bounded_command=command)
    monkeypatch.setattr(cycle3_bootstrap,'exec',inject,raising=False)
    cycle3_bootstrap.bootstrap_git('a'*40,run=True,started_at=100.,seeds=[103,104])
    assert commands[0][0][-1].endswith('[dev,research,wikitext]')
    arguments,options=commands[-1]
    assert arguments[arguments.index('--session-seconds')+1]=='14400.0'
    assert float(arguments[arguments.index('--bootstrap-elapsed-seconds')+1])==80
    assert arguments[arguments.index('--seeds')+1:arguments.index('--session-seconds')]==['103','104']
    assert options['timeout']==14320


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('name',MODELS)
def test_cycle3_cuda_fp32_inductor_backward_checkpoint(training_config,tmp_path,name):
    cfg=configuration(name,training_config.data.path,tmp_path/'cuda',101,device='cpu',smoke=True)
    cfg.train.device='cuda';cfg.train.deterministic=False;cfg.train.compile=True;cfg.train.compile_backend='inductor'
    with Trainer(cfg) as trainer:
        metrics=trainer.train_step();assert torch.isfinite(torch.tensor(metrics['training_loss']))
        assert metrics['peak_vram_allocated_bytes']>0;checkpoint=trainer.save()
    cfg.train.output_dir=str(tmp_path/'cuda-resume')
    with Trainer(cfg,resume=checkpoint) as resumed:
        assert torch.isfinite(torch.tensor(resumed.train_step()['training_loss']))
