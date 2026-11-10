"""Pinned stdlib Kaggle entry: bounded TinyStories language screen and downloads."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from urllib.request import urlopen

HELPER_REVISION='1c8ce2abb8f47d3167454b3503598854b7756dad'
HELPER_SHA256='5bc8f2ddcaf3cacbae401cbe077b494ddc9fc90c798417e3f1686d2635f925b7'
GUARD_SHA256='8c05edcf6d0450e273ac2f1fc118a8167bc9833490f841c9cfb1d57d96262007'


def fetch(name,digest):
    url=f'https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{HELPER_REVISION}/src/llm2/research/{name}.py'
    with urlopen(url,timeout=60) as response:raw=response.read(1024**2+1)
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('pinned helper checksum mismatch')
    ns={'__name__':'stories_pinned_'+name};exec(compile(raw.decode('utf-8'),url,'exec'),ns);return ns


def bootstrap_git(revision,*,run=False,run_number=32,session_seconds=7200,seeds=(331,332),started_at=None):
    if not run:return {'status':'not_started'}
    if (not re.fullmatch(r'[0-9a-f]{40}',revision) or type(run_number) is not int or run_number<32 or
        not 300<float(session_seconds)<=7200 or not seeds or len(set(seeds))!=len(seeds) or not set(seeds)<={331,332}):
        raise ValueError('exact revision, run>=32, seeds331/332 and limit<=2h required')
    print('TinyStories: quatro modelos x duas seeds; calibração train-only; T4x2, teto2h.',flush=True)
    started_at=time.monotonic() if started_at is None else started_at
    helper=fetch('cycle2_bootstrap',HELPER_SHA256);session=helper['_BootstrapSession'](session_seconds,started_at)
    hardware=helper['_hardware']();work,gpus,before_torch=hardware
    if len(gpus)!=2:raise RuntimeError('enable two T4 GPUs')
    guard=fetch('kaggle_light',GUARD_SHA256)
    if guard['active_workers']():raise RuntimeError('project workers active; do not duplicate campaign')
    parent=None
    try:
        root,parent,metadata=helper['checkout_git_source'](revision,session,hardware,shallow=True)
        globals()['download_file']=runpy.run_path(str(root/'src/llm2/research/notebook_download.py'))['download_file']
        helper['bounded_command']([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','-e',str(root)+'[dev,research]'],cwd=root,timeout=session.remaining())
        if importlib.metadata.version('torch')!=before_torch:raise RuntimeError('dependency install changed Torch')
        number=runpy.run_path(str(root/'src/llm2/research/naming.py'))['numbered_output']
        while True:
            try:output=number(work,number=run_number,search_roots=[Path('/kaggle/input')]);break
            except FileExistsError:run_number+=1
        metadata.update(torch_preserved=before_torch,session_seconds=session_seconds,download_mode='notebook_embedded_base64',
            scope='pinned 34MiB TinyStories prefixes; document-isolated byte LM; no old weights migrated')
        meta=parent/'bootstrap.json';meta.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')
        command=[sys.executable,'-m','llm2.research.stories_campaign','--output',str(output),
            '--cache',str(work/'llm2-tinystories-prefix-cache'),'--gpus',*gpus,'--seeds',*map(str,seeds),
            '--session-seconds',str(session_seconds),'--bootstrap-elapsed-seconds',str(session.elapsed),'--bootstrap-metadata',str(meta)]
        env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
        env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        try:helper['bounded_command'](command,cwd=root,timeout=session.remaining(export=True),env=env)
        finally:
            for kind in ('results','checkpoints'):
                path=work/f'{output.name}-{kind}.zip'
                if path.is_file():
                    try:download_file(path)
                    except Exception as exc:print(f'Arquivo preservado:{path}; botão indisponível:{exc}',flush=True)
        return {'campaign':str(output),'result':str(work/f'{output.name}-results.zip')}
    finally:
        if parent is not None:
            try:
                row=guard['remove_checkout'](work,parent)
                print(f"Checkout temporário removido:{row['bytes']/1024**2:.1f}MiB.",flush=True)
            except Exception as exc:print(f'Checkout preservado:{parent}: {exc}',flush=True)
