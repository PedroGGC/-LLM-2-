"""Fresh one-hour Kaggle cells; source and outputs remain pinned and exclusive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def cell(revision,run=True,*,run_number=20):
    if not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('full SHA required')
    if type(run_number) is not int or run_number<=0:raise ValueError('positive run number required')
    return f'''import time
from urllib.request import urlopen

# CICLO 3: LOTE CURTO (KAGGLE T4 x2)
RUN_CAMPAIGN = {run!r}
LLM2_REVISION = {revision!r}
LLM2_SETTINGS = {{
    "run_number": {run_number},
    "session_seconds": 3600,
    "max_runs": 2,
    "seeds": [103, 104, 105],
}}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{{LLM2_REVISION}}/src/llm2/research/cycle3_short_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        bootstrap = response.read(1024 * 1024 + 1)
    if len(bootstrap) > 1024 * 1024:
        raise ValueError("Bootstrap maior que o limite esperado")
    namespace = {{"__name__": "llm2_cycle3_short_bootstrap"}}
    exec(compile(bootstrap.decode("utf-8"), url, "exec"), namespace)
    namespace["bootstrap_git"](
        LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
    )
else:
    print("Execucao desativada.")
'''


def build(directory,revision,*,run_number=20):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    active=cell(revision,run_number=run_number);inactive=cell(revision,False,run_number=run_number)
    names=['rodar_ciclo3_kaggle_1h.py','rodar_ciclo3_kaggle_1h.ipynb','release.json']
    if any((directory/n).exists() for n in names):raise FileExistsError('use a fresh release folder')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'id':'guide','cell_type':'markdown','metadata':{},'source':[
            '# Ciclo 3 — lote de até uma hora\n','Ative Internet e T4 x2; mude RUN_CAMPAIGN para True.\n',
            'No máximo dois treinos novos. A evidência auditada da 0019 já está incluída.\n',
            'Nas próximas sessões, anexe o último ZIP de resultados como Input e use outro run_number.\n',
            'Baixe resultados e checkpoints; snapshots são salvos após cada treino.\n']},
            {'id':'run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
             'source':inactive.splitlines(keepends=True)}]}
    for name,value in [(names[0],active),(names[1],json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')]:
        with (directory/name).open('x',encoding='utf-8',newline='\n') as stream:stream.write(value)
    manifest={'revision':revision,'worker_revision':'8798f6b84714d2c6671d970685984270078a982c',
        'session_seconds':3600,'max_runs':2,'run_number':run_number,
        'files':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in names[:2]}}
    with (directory/names[2]).open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--run-number',type=int,default=20)
    a=p.parse_args();print(json.dumps(build(a.output_dir,a.revision,run_number=a.run_number),indent=2))
