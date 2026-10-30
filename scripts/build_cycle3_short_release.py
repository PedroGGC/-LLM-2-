"""Fresh one-hour Kaggle cells; source and outputs remain pinned and exclusive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def cell(revision,run=True,*,run_number=20,max_runs=2,seeds=(103,104,105),maintenance_guard=False):
    if not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('full SHA required')
    if type(run_number) is not int or run_number<=0:raise ValueError('positive run number required')
    if type(max_runs) is not int or max_runs not in (1,2):raise ValueError('one or two new runs required')
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s not in range(101,106) for s in seeds):
        raise ValueError('unique seeds 101..105 required')
    guard='''    info = globals().get("WORKING_INFO")
    if not info or not info["safe_to_start_new_batch"]:
        raise RuntimeError("Execute a manutencao e confira os avisos antes de continuar.")
    if maintenance_namespace["active_workers"]():
        raise RuntimeError("Ainda ha um worker ativo; aguarde sua conclusao.")
    LLM2_SETTINGS["run_number"] = info["next_run_number"]
''' if maintenance_guard else ''
    start='globals().get("maintenance_started_at", time.monotonic())' if maintenance_guard else 'time.monotonic()'
    return f'''import time
from urllib.request import urlopen

# CICLO 3: LOTE CURTO (KAGGLE T4 x2)
RUN_CAMPAIGN = {run!r}
LLM2_REVISION = {revision!r}
LLM2_SETTINGS = {{
    "run_number": {run_number},
    "session_seconds": 3600,
    "max_runs": {max_runs},
    "seeds": {list(seeds)!r},
}}

if RUN_CAMPAIGN:
    print("Celula de treino iniciada.", flush=True)
{guard}    started_at = {start}
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
    print("Bootstrap baixado; preparando ambiente.", flush=True)
    namespace["bootstrap_git"](
        LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
    )
else:
    print("Execucao desativada.")
'''


def maintenance_cell(revision,run=True):
    if not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('full SHA required')
    return f'''import time
from urllib.request import urlopen

RUN_MAINTENANCE = {run!r}
if RUN_MAINTENANCE:
    maintenance_started_at = time.monotonic()
    print("Inspecionando /kaggle/working; campanhas e ZIPs serao preservados.", flush=True)
    url = "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{revision}/src/llm2/research/kaggle_maintenance.py"
    with urlopen(url, timeout=60) as response:
        maintenance_source = response.read(1024 * 1024 + 1)
    if len(maintenance_source) > 1024 * 1024:
        raise ValueError("Arquivo de manutencao maior que o esperado")
    maintenance_namespace = {{"__name__": "llm2_kaggle_maintenance"}}
    exec(compile(maintenance_source.decode("utf-8"), url, "exec"), maintenance_namespace)
    WORKING_INFO = maintenance_namespace["clean_working"](apply=True)
else:
    print("Manutencao desativada.")
'''


def build(directory,revision,*,run_number=20,max_runs=2,seeds=(103,104,105),maintenance_guard=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    settings={'run_number':run_number,'max_runs':max_runs,'seeds':seeds,'maintenance_guard':maintenance_guard}
    active=cell(revision,**settings);inactive=cell(revision,False,**settings)
    names=['rodar_ciclo3_kaggle_1h.py','rodar_ciclo3_kaggle_1h.ipynb','release.json','limpar_working_kaggle.py']
    if any((directory/n).exists() for n in names):raise FileExistsError('use a fresh release folder')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'id':'guide','cell_type':'markdown','metadata':{},'source':[
            '# Ciclo 3 — lote de até uma hora\n','Ative Internet e T4 x2; mude RUN_CAMPAIGN para True.\n',
            'No máximo dois treinos novos. A evidência auditada da 0019 já está incluída.\n',
            'Nas próximas sessões, anexe o último ZIP de resultados como Input e use outro run_number.\n',
            'Baixe resultados e checkpoints; snapshots são salvos após cada treino.\n']},
            {'id':'run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
             'source':inactive.splitlines(keepends=True)}]}
    if maintenance_guard:
        notebook['cells'].insert(1,{'id':'maintenance','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':maintenance_cell(revision,False).splitlines(keepends=True)})
        notebook['cells'][0]['source'] += ['Execute primeiro a manutencao. O treino usa o proximo numero livre e bloqueia se houver worker/checkpoint parcial.\n']
    for name,value in [(names[0],active),(names[1],json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'),
                       (names[3],maintenance_cell(revision))]:
        with (directory/name).open('x',encoding='utf-8',newline='\n') as stream:stream.write(value)
    manifest={'revision':revision,'worker_revision':'8798f6b84714d2c6671d970685984270078a982c',
        'session_seconds':3600,'max_runs':max_runs,'run_number':run_number,'maintenance_guard':maintenance_guard,
        'files':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in [names[0],names[1],names[3]]}}
    with (directory/names[2]).open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--run-number',type=int,default=20)
    p.add_argument('--max-runs',type=int,default=2,choices=(1,2));p.add_argument('--seeds',type=int,nargs='+',default=[103,104,105])
    p.add_argument('--maintenance-guard',action='store_true')
    a=p.parse_args();print(json.dumps(build(a.output_dir,a.revision,run_number=a.run_number,max_runs=a.max_runs,
        seeds=a.seeds,maintenance_guard=a.maintenance_guard),indent=2))
