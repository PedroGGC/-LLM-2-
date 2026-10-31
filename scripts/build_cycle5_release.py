"""Build two short pinned cells: verified redundant-ZIP cleanup and Cycle 5."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
RESULTS_SHA="e89ff95f2c16518438ea3edbda536c50292265ad2e0bfdf769e564fd934565a2"
CHECKPOINTS_SHA="b41fec39a5c9c3cb7a04df86a13bcca06fbbb6448d9688101b1c6d4850d8919c"


def cells(revision,*,active=True):
    if not re.fullmatch(r"[0-9a-f]{40}",revision):raise ValueError("exact revision required")
    hashes={name:hashlib.sha256((ROOT/"src/llm2/research"/name).read_bytes()).hexdigest()
            for name in ("kaggle_reset.py","campaign_storage.py","cycle5_bootstrap.py")}
    clean=f'''import hashlib
from urllib.request import urlopen

# Remove apenas ZIPs after-NN redundantes da 0023; preserva pasta e definitivos.
RUN_CLEANUP = {active!r}
REVISION = {revision!r}
RESULTS_LOCAL_SHA256 = {RESULTS_SHA!r}
CHECKPOINTS_LOCAL_SHA256 = {CHECKPOINTS_SHA!r}

if RUN_CLEANUP:
    print("Conferindo definitivos e snapshots redundantes da 0023.", flush=True)
    modules = {{}}
    for name, expected in {{"kaggle_reset.py": {hashes['kaggle_reset.py']!r}, "campaign_storage.py": {hashes['campaign_storage.py']!r}}}.items():
        url = f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{{REVISION}}/src/llm2/research/{{name}}"
        with urlopen(url, timeout=60) as response:
            source = response.read(1024 * 1024 + 1)
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("Fonte diferente da revisao verificada: " + name)
        namespace = {{"__name__": "llm2_verified_maintenance"}}
        exec(compile(source.decode("utf-8"), url, "exec"), namespace)
        modules[name] = namespace
    LLM2_CLEANUP_INFO = modules["campaign_storage.py"]["clean_incrementals"](
        23, RESULTS_LOCAL_SHA256, CHECKPOINTS_LOCAL_SHA256,
        guard=modules["kaggle_reset.py"]["active_workers"],
        root_check=modules["kaggle_reset.py"]["root_directory"],
    )
else:
    print("Limpeza desativada.")
'''
    train=f'''import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = {active!r}
LLM2_REVISION = {revision!r}
LLM2_SETTINGS = {{"run_number": 24, "target_seconds": 7200, "session_seconds": 10800}}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Ciclo 5: 12 treinos; meta 2h, teto 3h; snapshot temporario aos 6/12.", flush=True)
    url = f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{{LLM2_REVISION}}/src/llm2/research/cycle5_bootstrap.py"
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != {hashes['cycle5_bootstrap.py']!r}:
        raise ValueError("Bootstrap diferente da revisao verificada.")
    namespace = {{"__name__": "llm2_cycle5_bootstrap"}}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    namespace["bootstrap_git"](LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS)
else:
    print("Treino desativado.")
'''
    return {"01_limpar_zips_redundantes_0023.py":clean,"02_rodar_controle_profundidade.py":train}


def build(directory,revision):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    scripts=cells(revision)
    for name,code in scripts.items():
        compile(code,name,"exec")
        with (directory/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(code)
    record={"revision":revision,"cycle":5,"seeds":[221,222,223],"jobs":12,"primary_full_evaluations":48,
        "fixed_steps":{"fine_k1":11372,"fine_k2":6000,"fine_k4":3084,"transformer_two":10496},
        "target_seconds":7200,"hard_seconds":10800,"snapshot_policy":"halfway temporary; delete after verified finals",
        "local_results_sha256":RESULTS_SHA,"local_checkpoints_sha256":CHECKPOINTS_SHA,
        "files":{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in scripts}}
    (directory/"release.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8",newline="\n")
    return record

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--revision",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args();print(json.dumps(build(a.output_dir,a.revision),indent=2))
