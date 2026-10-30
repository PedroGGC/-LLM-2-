"""Operational batches; imported with the immutable scientific worker on PYTHONPATH."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
import zipfile

import torch
from llm2.checkpoint import read_checkpoint_metadata
from llm2.data import sha256_file, write_json
from llm2.runtime import environment
from llm2.research import cycle3 as science, cycle3_report as reporting
from llm2.research.cycle3_protocol import ROOT, ARMS, MODELS, SEEDS, PROTOCOL_ID
from llm2.research.session import SessionBudget, SessionBudgetExceeded, SessionCampaign

WORKER_REVISION = '8798f6b84714d2c6671d970685984270078a982c'
OPERATOR_ROOT = Path(__file__).resolve().parents[3]
PRIORITY = ('anchor_flrm','flops_transformer','flops_lrm','flops_transformer_deep','targets_lrm','targets_transformer')


def parser():
    p=science.parser()
    p.description=__doc__
    p.set_defaults(session_seconds=3600.,seeds=list(SEEDS[2:]))
    p.add_argument('--worker-root',default=str(ROOT))
    p.add_argument('--ledger',default=str(OPERATOR_ROOT/'docs/results/campaign_0019_ledger.json'))
    p.add_argument('--previous-results',nargs='*',default=[])
    p.add_argument('--max-runs',type=int,default=2,choices=(1,2))
    p.add_argument('--validation-mode',choices=('focused','full'),default='focused')
    return p


def combine(reports, identity):
    """Identical copies are one observation; conflicting replicas are never selected."""
    rows,curves={},{}
    for report in reports:
        if report['identity']!=identity or report['protocol_id']!=PROTOCOL_ID or report['test_evaluated']:
            raise ValueError('incompatible scientific code, configuration, dataset or protocol in history')
        for collection,values,keys in ((rows,report['rows'],('arm','seed')),
                                       (curves,report['curves'],('arm','seed','step'))):
            for row in values:
                key=tuple(row[k] for k in keys)
                if key in collection and collection[key]!=row:
                    raise ValueError(f'conflicting repeated evidence: {key}')
                collection[key]=row
    result=reporting.analyze(list(rows.values()),list(curves.values()),identity=identity,
        smoke=any(r['engineering_smoke'] for r in reports),tests_skipped=any(r['tests_skipped'] for r in reports))
    return result


def load_report(path):
    path=Path(path)
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as archive:
            if len(archive.namelist())!=len(set(archive.namelist())):
                raise ValueError('duplicate archive names')
            name='history/report.json' if 'history/report.json' in archive.namelist() else 'report/report.json'
            if archive.getinfo(name).file_size>10*1024**2:raise ValueError('oversized report')
            return json.loads(archive.read(name))
    value=json.loads(path.read_text(encoding='utf-8'))
    return value.get('report',value)


def select_jobs(jobs, history, maximum):
    done={(r['arm'],r['seed']) for r in history['rows']}
    seeds=list(dict.fromkeys(j['seed'] for j in jobs))
    lookup={(j['arm'],j['seed']):j for j in jobs}
    pending=[lookup[a,s] for s in seeds for a in PRIORITY if (a,s) in lookup and (a,s) not in done]
    return pending[:maximum],pending[maximum:]


def validation_commands(models,mode='focused',device='cuda'):
    if mode=='full':
        return [['-m','pytest','-q','-m','not cuda']]+([['-m','pytest','-q','-m','cuda','-k','fp32']] if device=='cuda' else [])
    commands=[['-m','pytest','-q','tests/test_json_portability.py',
               'tests/test_cycle3.py::test_frozen_protocol_horizons_cost_and_initialization',
               'tests/test_cycle3.py::test_bundle_matches_cli_and_backup_preserves_weights']]
    for model in models:
        commands[0]+=[f'tests/test_cycle3.py::test_selected_models_causal_backward_finite_memorization[{model}]',
                      f'tests/test_cycle3.py::test_selected_models_strict_resume_dropout_accumulation[{model}]']
    if device=='cuda':
        commands.append(['-m','pytest','-q',*[f'tests/test_cycle3.py::test_cycle3_cuda_fp32_inductor_backward_checkpoint[{m}]' for m in models]])
    return commands


@contextmanager
def heartbeat(campaign,label):
    stop=threading.Event()
    def tick():
        while not stop.wait(30):
            elapsed=campaign.budget.elapsed
            print(f'⏱ {label} | sessão {elapsed/60:.1f}/{campaign.budget.seconds/60:g} min | '
                  f'tempo de trabalho restante {campaign.budget.work_remaining/60:.1f} min',flush=True)
    thread=threading.Thread(target=tick,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join(timeout=1)


def pilot(campaign,args):
    records=[];durations={}
    warm,measure=(0,1) if args.engineering_smoke else (10,30)
    seen=set()
    for job in campaign.manifest['runs']:
        model,gpu=job['model'],job['gpu_slot']
        if (model,gpu) in seen:continue
        seen.add((model,gpu))
        destination=campaign.output/'pilots'/f'{model}-gpu-{gpu}'
        with science.phase(campaign,'pilot_training',model=model,gpu_slot=gpu):
            campaign.command(['-m','llm2','profile','--config',job['config'],'--execute','--warmup-steps',warm,
                              '--measure-steps',measure,'--output',destination],f'pilot-{model}-{gpu}.log',gpu=gpu)
        value=json.loads((destination/'profile.json').read_text(encoding='utf-8'))
        records.append({'model':model,'gpu_slot':gpu,'profile':str(destination/'profile.json'),
                        'mean_step_seconds':value['timing']['mean_step_seconds'],
                        'warmup_seconds':value['warmup_timing']['total_warmup_seconds']})
        if model not in durations:
            target=campaign.output/'pilots'/f'{model}-validation'
            with science.phase(campaign,'pilot_validation',model=model):
                campaign.command(['-m','llm2.research.cycle3_evaluate','--config',job['config'],
                    '--directory',target,'--device',args.device,'--timing-only'],f'pilot-{model}-val.log',gpu=campaign.gpus[0])
            durations[model]=json.loads((target/'timing.json').read_text())['validation_seconds']
    for job in campaign.manifest['runs']:
        row=next(r for r in records if (r['model'],r['gpu_slot'])==(job['model'],job['gpu_slot']))
        job['estimated_job_seconds']=(row['mean_step_seconds']*job['max_steps']*1.35+row['warmup_seconds']
            +4*durations[job['model']]*1.3+(5 if args.engineering_smoke else 120))
    campaign.manifest.update(pilot=records,pilot_completed_before_training=True,validation_duration_pilot=durations,
        compilation_measurement='warmup combines compilation/runtime warming; no isolated compile counter')
    campaign.save()


def reports(campaign,prior,destination):
    fake=SimpleNamespace(output=destination,manifest=campaign.manifest)
    current=reporting.write_report(fake)
    aggregate=combine([prior,current],prior['identity'])
    reporting.save_report(destination/'history',aggregate)
    lookup={(r['arm'],r['seed']):r for r in aggregate['rows']}
    pairs={}
    for arm in ARMS:
        if arm=='anchor_flrm':continue
        seeds=[s for s in SEEDS if ('anchor_flrm',s) in lookup and (arm,s) in lookup]
        pairs[arm]={'seeds':seeds,**reporting.effect([lookup['anchor_flrm',s]['validation_loss']-lookup[arm,s]['validation_loss'] for s in seeds])}
    write_json(destination/'history/available_pairs.json',{'effects_exploratory':pairs,
        'confirmation_requires_five_complete_blocks':True,'complete_blocks':aggregate['complete_seeds']})
    write_json(destination/'history/queue.json',{'completed_runs':len(aggregate['rows']),
        'pending_run_ids':[f'{a}-seed-{s}' for s in SEEDS for a in PRIORITY if (a,s) not in lookup],
        'next_action':'Attach the latest results ZIP as Input for the next one-hour batch.'})
    return aggregate


def export(campaign,destination,report_directory,*,checkpoints=False):
    """Exclusive snapshots, including the latest complete optimizer-step states."""
    descriptor,name=tempfile.mkstemp(prefix='.cycle3-batch-',suffix='.zip',dir=destination.parent)
    os.close(descriptor);temporary=Path(name)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED if checkpoints else zipfile.ZIP_DEFLATED) as archive:
            if checkpoints:
                paths=[]
                for job in campaign.manifest['runs']:
                    run=Path(job['directory'])
                    states=sorted((run/'checkpoints').glob('step-*'),key=lambda p:int(p.name.split('-')[-1]))
                    if states:
                        read_checkpoint_metadata(states[-1]);paths+=list(states[-1].iterdir())
                        paths += [run/n for n in ('run.yaml','run.json') if (run/n).exists()]
                paths += [campaign.output/n for n in ('campaign.json','data/manifest.json','data/tokenizer.json') if (campaign.output/n).exists()]
            else:
                paths=[p for p in campaign.output.rglob('*') if p.is_file() and
                       (p.suffix in ('.json','.jsonl','.yaml','.txt','.md','.csv','.log','.png') or p.name=='source.zip')]
            for p in sorted(paths):
                if p.is_file():archive.write(p,p.relative_to(campaign.output).as_posix())
            if not checkpoints:
                for category in ('report','history'):
                    for p in sorted((report_directory/category).rglob('*')):
                        if p.is_file():archive.write(p,category+'/'+p.relative_to(report_directory/category).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if len(archive.namelist())!=len(set(archive.namelist())) or archive.testzip() is not None:
                raise ValueError('snapshot integrity failed')
        os.link(temporary,destination)
    finally:temporary.unlink(missing_ok=True)
    return destination


def publish(campaign,prior,suffix):
    destination=campaign.output/'reports'/suffix
    aggregate=reports(campaign,prior,destination)
    stem=campaign.output.name+('' if suffix=='final' else '-'+suffix)
    analysis=export(campaign,campaign.output.parent/f'{stem}-results.zip',destination)
    backup=export(campaign,campaign.output.parent/f'{stem}-checkpoints.zip',destination,checkpoints=True)
    print(f'✓ Evidência salva: {analysis}\n✓ Checkpoints: {backup}',flush=True)
    return aggregate,analysis,backup


def execute(args,*,started=None):
    started=time.monotonic() if started is None else started
    budget=SessionBudget(args.session_seconds,args.export_reserve_seconds,
                         elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-started)
    if Path(args.worker_root).resolve()!=ROOT.resolve():raise ValueError('worker import path mismatch')
    if args.device=='cuda' and (not torch.cuda.is_available() or any('T4' not in torch.cuda.get_device_name(int(s)) for s in args.gpus)):
        raise ValueError('T4 worker required')
    campaign=SessionCampaign(ROOT,Path(args.output).resolve(),device=args.device,gpus=args.gpus,budget=budget)
    campaign.manifest.update(cycle=3,protocol_id=PROTOCOL_ID,seeds=args.seeds,models=list(MODELS),
        engineering_smoke=args.engineering_smoke,tests_skipped=args.skip_tests,test_evaluated=False,
        operational_mode='one_hour_batch',max_new_runs=args.max_runs,worker_revision=WORKER_REVISION,
        validation_scope=args.validation_mode,scheduling='serial, original GPU alternation; at most two new runs')
    identity={'protocol_id':PROTOCOL_ID,'dataset_sha256':None,
              'source_sha256':campaign.manifest['provenance']['source_sha256'],
              'versioned_configs_sha256':campaign.manifest['provenance']['versioned_configs_sha256']}
    prior=reporting.analyze([],[],identity=identity,smoke=args.engineering_smoke)
    error=None
    try:
        with science.phase(campaign,'dataset_and_configurations'):
            data=science.prepare(campaign,args)
            identity['dataset_sha256']=campaign.manifest['dataset_sha256']
            prior['identity']=identity
            if not args.engineering_smoke:
                if not args.bootstrap_metadata:raise ValueError('science requires verified bootstrap metadata')
                metadata=json.loads(Path(args.bootstrap_metadata).read_text())
                if metadata['source_git_revision']!=WORKER_REVISION:raise ValueError('scientific worker revision changed')
                archive=Path(metadata['operator_source_archive'])
                if sha256_file(archive)!=metadata['operator_source_archive_sha256']:raise ValueError('operator source archive changed')
                (campaign.output/'operator').mkdir();shutil.copyfile(archive,campaign.output/'operator/source.zip')
                campaign.manifest['operator_provenance']=metadata['operator_source_files_sha256']
                prior=combine([load_report(args.ledger),*[load_report(p) for p in args.previous_results]],identity)
                current_runtime=reporting.runtime_contract(environment())
                if any(reporting.runtime_contract(r['environment'])!=current_runtime for r in prior['rows']):
                    raise ValueError('runtime versions/hardware differ from previous evidence')
            science.materialize(campaign,args,data)
            jobs,pending=select_jobs(campaign.manifest['runs'],prior,args.max_runs)
            campaign.manifest.update(runs=jobs,previous_complete_runs=len(prior['rows']),
                                     remaining_after_selected=[j['id'] for j in pending])
            print('Lote planejado: '+', '.join(j['id'] for j in jobs)+f" | {len(prior['rows'])} resultados anteriores preservados",flush=True)
        if jobs and not args.skip_tests:
            commands=validation_commands(list(dict.fromkeys(j['model'] for j in jobs)),args.validation_mode,args.device)
            campaign.manifest['validation_commands']=commands
            for i,command in enumerate(commands):
                with science.phase(campaign,'validation_focused' if args.validation_mode=='focused' else 'validation_full'):
                    campaign.command(command,f'tests-{args.validation_mode}-{i}.log',gpu=campaign.gpus[0])
        if jobs:
            with heartbeat(campaign,'piloto'):pilot(campaign,args)
            print('Previsão conservadora dos treinos/avaliações: '+f"{sum(j['estimated_job_seconds'] for j in jobs)/60:.1f} min",flush=True)
        for index,job in enumerate(jobs,1):
            budget.require(job['estimated_job_seconds'])
            print(f"Lote {index}/{len(jobs)}: {job['id']} ({job['max_steps']} passos fixos)",flush=True)
            with science.phase(campaign,'training',job=job['id']),heartbeat(campaign,job['id']):
                campaign.train_worker([job],job['gpu_slot'])
            if job['step']!=job['max_steps'] or job['stop_reason']!='max_steps':
                raise SessionBudgetExceeded('fixed horizon not reached')
            with science.phase(campaign,'evaluation_generation_diagnostics',job=job['id']):
                science.evaluate_run(campaign,job,args)
            campaign.save();publish(campaign,prior,f'after-{index:02d}')
        campaign.manifest['status']='complete'
    except SessionBudgetExceeded as exc:
        campaign.manifest.update(status='partial_budget',error=str(exc))
    except Exception as exc:
        error=exc;campaign.manifest.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort();campaign.manifest['session_budget']=budget.snapshot();campaign.save()
        aggregate,analysis,backup=publish(campaign,prior,'final')
        print(f"Status do lote: {campaign.manifest['status']}; {len(aggregate['rows'])}/30 treinos preservados no histórico.\n"
              f'Baixe {analysis.name} e {backup.name}.',flush=True)
    if error:raise RuntimeError('short batch failed; partial evidence preserved') from error
    return {'status':campaign.manifest['status'],'results_zip':str(analysis),'checkpoints_zip':str(backup)}


def main(argv=None,*,started=None):
    args=parser().parse_args(argv)
    if not 300<args.session_seconds<=3600:raise ValueError('short session cap must exceed 300 and be at most 3600 s')
    if args.export_reserve_seconds<300:raise ValueError('preserve at least five minutes for export')
    science.plan(args)
    if not args.execute:
        return {'status':'not_started','session_seconds':args.session_seconds,'max_runs':args.max_runs,'worker_revision':WORKER_REVISION}
    if not args.output:raise ValueError('fresh output path required')
    return execute(args,started=started)


if __name__=='__main__':main()
