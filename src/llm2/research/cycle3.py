"""Bounded independent-corpus confirmation with predeclared target/FLOP cohorts."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
import zipfile

import torch
import yaml

from ..checkpoint import read_checkpoint_metadata
from ..config import load_config
from .campaign import worker_environment
from .cycle2 import export, verify_bootstrap
from .cycle3_protocol import (ROOT, SEEDS, MODELS, PROTOCOL, PROTOCOL_ID,
                              ordered_jobs, paired_fingerprints, verify_dataset)
from .cycle3_report import write_report
from .naming import numbered_output
from .session import SessionBudget, SessionBudgetExceeded, SessionCampaign


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', help='existing immutable prepared dataset')
    source.add_argument('--cache', help='pinned WikiText parquet cache')
    p.add_argument('--download', action='store_true')
    p.add_argument('--output')
    p.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--gpus', nargs='+', default=['0', '1'])
    p.add_argument('--execute', action='store_true')
    p.add_argument('--session-seconds', type=float, default=14400.)
    p.add_argument('--export-reserve-seconds', type=float, default=300.)
    p.add_argument('--bootstrap-elapsed-seconds', type=float, default=0.)
    p.add_argument('--bootstrap-metadata')
    p.add_argument('--engineering-smoke', action='store_true')
    p.add_argument('--skip-tests', action='store_true', help='explicitly recorded engineering escape')
    return p


def plan(args):
    if not 0 < args.session_seconds <= 14400:
        raise ValueError('session must be at most 14400 seconds')
    if (args.device == 'cpu') != args.engineering_smoke:
        raise ValueError('CPU requires --engineering-smoke; scientific protocol requires CUDA')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(not s.isdecimal() for s in args.gpus):
        raise ValueError('invalid logical GPU slots')
    budget = SessionBudget(args.session_seconds, args.export_reserve_seconds, elapsed=args.bootstrap_elapsed_seconds)
    output = Path(args.output or ROOT/'runs/PLANNED-cycle3').resolve()
    data = Path(args.data or output/'data').resolve()
    jobs = ordered_jobs(args.seeds, args.gpus, data, output, device=args.device, smoke=args.engineering_smoke)
    return {'execution': 'not_started', 'protocol_id': PROTOCOL_ID, 'protocol': PROTOCOL,
            'engineering_smoke': args.engineering_smoke, 'jobs': jobs, 'session_budget': budget.snapshot()}


@contextmanager
def phase(campaign, name, **details):
    started = time.monotonic()
    row = {'phase': name, **details}
    campaign.manifest.setdefault('phases', []).append(row)
    try:
        yield
    finally:
        row['wall_seconds'] = time.monotonic()-started
        campaign.save()


def prepare(campaign, args):
    verify_bootstrap(args.bootstrap_metadata)
    if args.bootstrap_metadata:
        shutil.copyfile(args.bootstrap_metadata, campaign.output/'bootstrap.json')
    if args.data:
        path = Path(args.data).resolve()
    else:
        path = campaign.output/'data'
        command = ['-m', 'llm2.research.wikitext', '--cache', args.cache, '--output', path]
        if args.download:
            command.append('--download')
        campaign.command(command, 'prepare-wikitext.log')
    data = verify_dataset(path, smoke=args.engineering_smoke)
    if min(map(len, data.arrays.values())) <= (17 if args.engineering_smoke else 256):
        raise ValueError('dataset splits too short')
    if path != campaign.output/'data':
        (campaign.output/'data').mkdir()
        for name in ('manifest.json', 'tokenizer.json'):
            shutil.copyfile(path/name, campaign.output/'data'/name)
    campaign.manifest.update(dataset_sha256=data.fingerprint, dataset_manifest=data.manifest,
                             dataset_path=str(path), status='checking')
    return path


def materialize(campaign, args, data):
    (campaign.output/'configs').mkdir()
    jobs = ordered_jobs(args.seeds, campaign.gpus, data, campaign.output, device=args.device, smoke=args.engineering_smoke)
    for job in jobs:
        file = campaign.output/'configs'/f"{job['id']}.yaml"
        file.write_text(yaml.safe_dump(job['config_resolved'], sort_keys=True), encoding='utf-8')
        job.update(config=str(file), directory=job['config_resolved']['train']['output_dir'],
                   physical_gpu=worker_environment(ROOT, job['gpu_slot']).get('CUDA_VISIBLE_DEVICES') if args.device=='cuda' else None)
    campaign.manifest['runs'] = jobs
    campaign.manifest['paired_init_verification'] = {}
    for seed in args.seeds:
        campaign.budget.require()
        configs = {j['id']: load_config(j['config']) for j in jobs if j['seed']==seed}
        campaign.manifest['paired_init_verification'][str(seed)] = paired_fingerprints(configs)
    campaign.save()


def pilot(campaign, args):
    records, durations = [], {}
    warm, measured = (0, 1) if args.engineering_smoke else (10, 30)
    for index, gpu in enumerate(campaign.gpus):
        for model in MODELS:
            job = next(j for j in campaign.manifest['runs'] if j['model']==model)
            destination = campaign.output/'pilots'/f'{model}-gpu-{index}'
            with phase(campaign, 'pilot_training', model=model, gpu_slot=gpu):
                campaign.command(['-m', 'llm2', 'profile', '--config', job['config'], '--execute',
                                  '--warmup-steps', warm, '--measure-steps', measured, '--output', destination],
                                 f'pilot-{model}-gpu-{index}.log', gpu=gpu)
            value = json.loads((destination/'profile.json').read_text(encoding='utf-8'))
            records.append({'model': model, 'gpu_slot': gpu, 'profile': str(destination/'profile.json'),
                            'mean_step_seconds': value['timing']['mean_step_seconds'],
                            'warmup_seconds': value['warmup_timing']['total_warmup_seconds']})
            campaign.manifest['pilot'] = records
            if index==0:
                target = campaign.output/'pilots'/f'{model}-validation'
                with phase(campaign, 'pilot_validation', model=model):
                    campaign.command(['-m', 'llm2.research.cycle3_evaluate', '--config', job['config'],
                                      '--directory', target, '--device', args.device, '--timing-only'],
                                     f'pilot-{model}-validation.log', gpu=gpu)
                durations[model] = json.loads((target/'timing.json').read_text())['validation_seconds']
    for job in campaign.manifest['runs']:
        row = next(r for r in records if (r['model'],r['gpu_slot'])==(job['model'],job['gpu_slot']))
        job['estimated_job_seconds'] = (row['mean_step_seconds']*job['max_steps']*1.35
            + row['warmup_seconds'] + 4*durations[job['model']]*1.3 + (5 if args.engineering_smoke else 120))
    campaign.manifest.update(pilot_completed_before_training=True, validation_duration_pilot=durations,
        forecast_formula='step_seconds*steps*1.35 + warmup + 4*full_val_seconds*1.3 + 120s overhead (5s smoke)',
        compilation_measurement='warmup includes compile/runtime warmup; no isolated compile time or recompilation counter')
    campaign.save()


def evaluate_run(campaign, job, args):
    gpu = campaign.gpus[0]
    campaign.command(['-m', 'llm2.research.cycle3_evaluate', '--config', job['config'],
                      '--directory', job['directory'], '--device', args.device], f"{job['id']}-evaluate.log", gpu=gpu)
    result = json.loads((Path(job['directory'])/'evaluation_bundle.json').read_text())
    if result['test_evaluated'] or result['split']!='val':
        raise ValueError('reserved test consulted by validation pipeline')
    records = result['checkpoints']
    if [r['step'] for r in records] != [job['max_steps']*i//4 for i in range(1,5)]:
        raise ValueError('incomplete milestone evaluation')
    expected_tokens = (campaign.manifest['dataset_manifest']['splits']['val']['tokens']-1)//load_config(job['config']).data.seq_len*load_config(job['config']).data.seq_len
    for row in records:
        value = json.loads(Path(row['evaluation']).read_text())
        meta = read_checkpoint_metadata(row['checkpoint'])
        if (value['evaluation_split']!='val' or value['validation_scope']!='all_full_windows'
                or value['dataset_sha256']!=campaign.manifest['dataset_sha256']
                or value['validation_tokens']!=expected_tokens
                or value['checkpoint_model_sha256']!=meta['files']['model.safetensors']):
            raise ValueError('evaluation dataset/scope/weights mismatch')
    final = records[-1]
    if final['checkpoint_model_sha256']!=job['checkpoint_model_sha256']:
        raise ValueError('final checkpoint mismatch')
    job.update(status='complete', checkpoint_evaluations=records, evaluation=final['evaluation'],
               full_validation_loss=final['validation_loss'], validation_tokens=final['validation_tokens'],
               evaluation_physical_gpu=worker_environment(ROOT,gpu).get('CUDA_VISIBLE_DEVICES') if args.device=='cuda' else None)
    campaign.save()


def scientific_jobs(campaign, args):
    for seed in args.seeds:
        block = [j for j in campaign.manifest['runs'] if j['seed']==seed]
        campaign.budget.require(sum(j['estimated_job_seconds'] for j in block))
        for job in block:
            campaign.budget.require(job['estimated_job_seconds'])
            with phase(campaign, 'training', job=job['id']):
                campaign.train_worker([job], job['gpu_slot'])
            if job['stop_reason']!='max_steps' or job['step']!=job['max_steps']:
                raise SessionBudgetExceeded('training ended before the fixed horizon')
            with phase(campaign, 'evaluation_generation_diagnostics', job=job['id']):
                evaluate_run(campaign, job, args)
    campaign.manifest['status'] = 'complete'


def checkpoint_backup(campaign):
    """Latest COMPLETE optimizer-step state per started run, including final weights."""
    destination = campaign.output.with_name(campaign.output.name+'-checkpoints.zip')
    descriptor, name = tempfile.mkstemp(prefix='.cycle3-checkpoints-', suffix='.zip', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            for job in campaign.manifest['runs']:
                run = Path(job['directory'])
                checkpoints = sorted((run/'checkpoints').glob('step-*'), key=lambda p:int(p.name.split('-')[-1]))
                if not checkpoints:
                    continue
                checkpoint = checkpoints[-1]
                read_checkpoint_metadata(checkpoint)
                for path in checkpoint.iterdir():
                    if path.is_file(): archive.write(path, path.relative_to(campaign.output).as_posix())
                for name in ('run.yaml', 'run.json'):
                    if (run/name).exists(): archive.write(run/name, (run/name).relative_to(campaign.output).as_posix())
            for name in ('campaign.json', 'data/manifest.json', 'data/tokenizer.json'):
                path = campaign.output/name
                if path.exists(): archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None: raise ValueError('checkpoint backup CRC failed')
        os.link(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def execute(args, *, started=None):
    started = time.monotonic() if started is None else started
    budget = SessionBudget(args.session_seconds,args.export_reserve_seconds,
                           elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-started)
    if args.device=='cuda':
        if not torch.cuda.is_available() or any(int(s)>=torch.cuda.device_count() for s in args.gpus):
            raise RuntimeError('requested CUDA slots unavailable')
        if any('T4' not in torch.cuda.get_device_name(int(s)) for s in args.gpus):
            raise RuntimeError('this frozen scientific protocol requires T4')
    output = Path(args.output).resolve() if args.output else numbered_output(ROOT/'runs')
    campaign = SessionCampaign(ROOT,output,device=args.device,gpus=args.gpus,budget=budget)
    campaign.manifest.update(cycle=3, protocol=PROTOCOL, protocol_id=PROTOCOL_ID, seeds=args.seeds,
        models=list(MODELS), engineering_smoke=args.engineering_smoke, tests_skipped=args.skip_tests,
        test_evaluated=False, pilot_completed_before_training=False,
        scheduling='serial; rotated arms, alternating physical GPUs; evaluations on first GPU',
        limitations='Validation-only independent corpus; estimated FLOP matching is not time/parameter matching. Five complete seeds required; test held out. Results ZIP excludes weights; separate checkpoint backup preserves latest complete states.')
    error = None
    try:
        with phase(campaign,'dataset_and_configurations'):
            data = prepare(campaign,args)
            materialize(campaign,args,data)
        if not args.skip_tests:
            with phase(campaign,'tests_cpu'):
                campaign.command(['-m','pytest','-q','-m','not cuda'],'tests-cpu.log',gpu=campaign.gpus[0])
            if args.device=='cuda':
                with phase(campaign,'tests_cuda_fp32'):
                    campaign.command(['-m','pytest','-q','-m','cuda','-k','fp32'],'tests-cuda-fp32.log',gpu=campaign.gpus[0])
        pilot(campaign,args)
        scientific_jobs(campaign,args)
    except SessionBudgetExceeded as exc:
        campaign.manifest.update(status='partial_budget',error=str(exc))
        for job in campaign.manifest['runs']:
            if job['status'] in ('training','failed'): job['status']='interrupted_budget'
    except Exception as exc:
        error = exc
        campaign.manifest.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort()
        campaign.manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        campaign.manifest['session_budget'] = budget.snapshot()
        campaign.save()
        try:
            write_report(campaign)
        except Exception as exc:
            error = error or exc
            campaign.manifest.update(status='failed',report_error=str(exc));campaign.save()
        backup = None
        try:
            backup = checkpoint_backup(campaign)
            campaign.manifest['checkpoint_backup'] = str(backup)
        except Exception as exc:
            error = error or exc
            campaign.manifest.update(status='failed',checkpoint_backup_error=str(exc))
        campaign.manifest['session_budget_after_backup'] = budget.snapshot()
        campaign.save()
        archive = export(campaign)
        print(f"\nStatus: {campaign.manifest['status']}\nResults: {archive}\nCheckpoints: {backup}",flush=True)
    if error is not None: raise RuntimeError(f'Cycle 3 failed; evidence exported to {archive}') from error
    return {'status':campaign.manifest['status'],'results_zip':str(archive),'checkpoints_zip':str(backup),'campaign':str(output)}


def main(argv=None, *, started=None):
    args = parser().parse_args(argv)
    description = plan(args)
    if not args.execute:
        print(json.dumps(description,indent=2));return description
    return execute(args,started=started)
