"""Publish fresh short Kaggle artifacts pinned to an already published commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def git_cell(revision, *, run=True, session_seconds=14400, run_number=18):
    if not re.fullmatch(r'[0-9a-f]{40}',revision): raise ValueError('full commit SHA required')
    if not 300<session_seconds<=14400 or type(run_number) is not int or run_number<=0:
        raise ValueError('invalid session/run number')
    return f'''import time
from urllib.request import urlopen

# CICLO 3: CONFIRMACAO INDEPENDENTE (KAGGLE T4 x2)
RUN_CAMPAIGN = {run!r}
LLM2_REVISION = {revision!r}
LLM2_SETTINGS = {{
    "run_number": {run_number},
    "session_seconds": {session_seconds},  # teto total de 4h; pode terminar antes
    "seeds": [101, 102, 103, 104, 105],
}}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{{LLM2_REVISION}}/src/llm2/research/cycle3_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        bootstrap = response.read(1024 * 1024 + 1)
    if len(bootstrap) > 1024 * 1024:
        raise ValueError("Bootstrap maior que o limite esperado")
    namespace = {{"__name__": "llm2_cycle3_pinned_bootstrap"}}
    exec(compile(bootstrap.decode("utf-8"), url, "exec"), namespace)
    namespace["bootstrap_git"](
        LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
    )
else:
    print("Execucao desativada. Defina RUN_CAMPAIGN = True para iniciar.")
'''


def build_git(output,revision,*,run_number=18):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    names=('rodar_ciclo3_kaggle_4h.py','rodar_ciclo3_kaggle_4h.ipynb','llm2-cycle3-git-4h.json')
    if any((output/n).exists() for n in names): raise FileExistsError('use a fresh release destination')
    staging=Path(tempfile.mkdtemp(prefix='.cycle3-release-',dir=output))
    (staging/names[0]).write_text(git_cell(revision,run_number=run_number),encoding='utf-8')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'cells':[{'id':'cycle3-guide','cell_type':'markdown','metadata':{},'source':[
            '# Ciclo 3 — WikiText-103\n','Ative Internet e GPU T4 x2. Defina RUN_CAMPAIGN=True.\n',
            'Teto total de quatro horas, incluindo preparo/testes/piloto/treino/avaliação/exportação.\n',
            'Baixe os dois ZIPs: resultados e checkpoints. Seeds pendentes podem exigir outra sessão.\n']},
            {'id':'cycle3-run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
             'source':git_cell(revision,run=False,run_number=run_number).splitlines(keepends=True)}]}
    (staging/names[1]).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    manifest={'release':'cycle3-git-4h','revision':revision,'run_number':run_number,'session_seconds':14400,'export_reserve_seconds':300,
              'files':{n:{'sha256':hashlib.sha256((staging/n).read_bytes()).hexdigest(),'bytes':(staging/n).stat().st_size} for n in names[:2]}}
    (staging/names[2]).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    for n in names:
        os.link(staging/n,output/n);(staging/n).unlink()
    staging.rmdir()
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--run-number',type=int,default=18)
    a=p.parse_args();print(json.dumps(build_git(a.output_dir,a.revision,run_number=a.run_number),indent=2))
