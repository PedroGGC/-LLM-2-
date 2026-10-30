"""Standalone one-hour bootstrap; scientific workers remain on the audited revision."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
from urllib.request import urlopen
import zipfile

WORKER_REVISION='8798f6b84714d2c6671d970685984270078a982c'


def checkout_hashes(source,session):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True,timeout=session.remaining()).strip()
    names=subprocess.check_output(['git','ls-files','-z'],cwd=source,timeout=session.remaining()).decode().split('\0')
    hashes={}
    for name in filter(None,names):
        path=source/name
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError('unsupported source member')
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return head,hashes


def previous_results(work,protocol_id):
    candidates=sorted(set(work.glob('llm2-campaign-*-results.zip'))|set(Path('/kaggle/input').rglob('llm2-campaign-*-results.zip')))
    selected=[]
    for path in candidates:
        with zipfile.ZipFile(path) as archive:
            name='history/report.json' if 'history/report.json' in archive.namelist() else 'report/report.json'
            if name not in archive.namelist():continue
            if archive.getinfo(name).file_size>10*1024**2:raise ValueError('oversized history report')
            if json.loads(archive.read(name)).get('protocol_id')==protocol_id:selected.append(path)
    return selected


def bootstrap_git(revision,*,run=False,run_number=20,session_seconds=3600.,max_runs=2,
                  seeds=(103,104,105),validation_mode='focused',started_at=None):
    if not run:return {'status':'not_started'}
    now=time.monotonic();started_at=now if started_at is None else float(started_at)
    if (not re.fullmatch(r'[0-9a-f]{40}',revision) or not math.isfinite(started_at)
            or not 0<=started_at<=now or not 300<float(session_seconds)<=3600
            or type(max_runs) is not int or max_runs not in (1,2)
            or type(run_number) is not int or run_number<=0 or validation_mode not in ('focused','full')
            or not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s not in range(101,106) for s in seeds)):
        raise ValueError('invalid short-session settings')
    url=f'https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{revision}/src/llm2/research/cycle2_bootstrap.py'
    with urlopen(url,timeout=60) as response:payload=response.read(1024**2+1)
    if len(payload)>1024**2:raise ValueError('oversized helper')
    helper={'__name__':'cycle3_short_helper'};exec(compile(payload.decode(),url,'exec'),helper)
    session=helper['_BootstrapSession'](session_seconds,started_at)
    hardware=helper['_hardware']();work,gpus,before_torch=hardware
    operator,parent,operator_meta=helper['checkout_git_source'](revision,session,hardware)
    if hashlib.sha256((operator/'src/llm2/research/cycle2_bootstrap.py').read_bytes()).digest()!=hashlib.sha256(payload).digest():
        raise ValueError('helper differs from pinned operator checkout')
    command=helper['bounded_command'];worker=parent/'scientific-worker'
    command(['git','-c','core.autocrlf=false','worktree','add','--detach',worker,WORKER_REVISION],cwd=operator,timeout=session.remaining())
    head,hashes=checkout_hashes(worker,session)
    if head!=WORKER_REVISION:raise ValueError('scientific worker revision changed')
    command(['git','diff','--exit-code','HEAD'],cwd=worker,timeout=session.remaining())
    ledger=json.loads((operator/'docs/results/campaign_0019_ledger.json').read_text(encoding='utf-8'))
    for name,digest in ledger['report']['identity']['source_sha256'].items():
        if hashes.get(name)!=digest:raise ValueError('scientific source differs from campaign 0019: '+name)
    for name,digest in ledger['report']['identity']['versioned_configs_sha256'].items():
        if hashes.get('configs/'+name)!=digest:raise ValueError('scientific configuration changed')
    command([sys.executable,'-m','pip','install','--quiet','-e',str(worker)+'[dev,research,wikitext]'],
            cwd=worker,timeout=session.remaining())
    if importlib.metadata.version('torch')!=before_torch:raise RuntimeError('installation changed Torch')
    output=runpy.run_path(str(worker/'src/llm2/research/naming.py'))['numbered_output'](
        work,number=run_number,search_roots=[Path('/kaggle/input')])
    source_archive=parent/'operator-source.zip'
    command(['git','archive','--format=zip','--output',source_archive,revision],cwd=operator,timeout=session.remaining())
    metadata={'source_mode':'git','source_git_revision':WORKER_REVISION,'source_files_sha256':hashes,
        'operator_git_revision':revision,'operator_source_files_sha256':operator_meta['source_files_sha256'],
        'operator_source_archive':str(source_archive),
        'operator_source_archive_sha256':hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        'session_limit_seconds':session.limit,'setup_elapsed_seconds':session.elapsed,'torch_preserved':before_torch}
    meta=parent/'bootstrap-short.json';meta.write_text(json.dumps(metadata,indent=2),encoding='utf-8',newline='\n')
    history=previous_results(work,ledger['report']['protocol_id'])
    arguments=[sys.executable,str(operator/'scripts/run_cycle3_short.py'),'--worker-root',worker,'--execute',
        '--cache',work/'llm2-wikitext-pinned-cache','--download','--output',output,'--gpus',*gpus,'--seeds',*seeds,
        '--max-runs',max_runs,'--session-seconds',session.limit,'--export-reserve-seconds',300,
        '--validation-mode',validation_mode,'--bootstrap-metadata',meta,'--bootstrap-elapsed-seconds',session.elapsed]
    if history:arguments+=['--previous-results',*history]
    env=os.environ.copy();env.update(PYTHONPATH=str(worker/'src'),PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    # Match the scientific Trainer's runtime policy before comparing environments.
    env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    print(f'Ciclo 3: até {max_runs} treinos novos; teto total {session.limit/60:g} min. '
          'Doze resultados da 0019 já preservados. Histórico adicional: '+str(len(history))+' ZIPs.',flush=True)
    try:command(arguments,cwd=worker,timeout=session.remaining(export=True),env=env)
    finally:print(f'Resultados e snapshots: {output.parent}/{output.name}*-results.zip; checkpoints em ZIP separado.',flush=True)
    return {'campaign':str(output),'session_elapsed_seconds':session.elapsed}
