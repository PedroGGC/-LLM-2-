"""Build exactly two notebook entrypoint scripts, pinned to one public revision."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
NAMES=("01_backup_e_limpeza_kaggle.py","02_rodar_diagnostico_fractal.py")


def cells(revision,*,active=True):
    if not re.fullmatch(r"[0-9a-f]{40}",revision):raise ValueError("full revision required")
    hashes={name:hashlib.sha256((ROOT/"src/llm2/research"/name).read_bytes()).hexdigest()
        for name in ("kaggle_reset.py","cycle4_bootstrap.py")}
    backup=f'''import hashlib
from urllib.request import urlopen

# Primeiro execute em "backup". Baixe o ZIP e confira seu SHA-256 no computador.
# Depois execute esta mesma celula em "limpar", preenchendo o hash local.
MODO = {"backup" if active else "desativado"!r}
BACKUP_LOCAL_SHA256 = ""
REVISAO = {revision!r}

if MODO in ("backup", "limpar"):
    print("Preparando backup/limpeza do working.", flush=True)
    url = f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{{REVISAO}}/src/llm2/research/kaggle_reset.py"
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != {hashes["kaggle_reset.py"]!r}:
        raise ValueError("Fonte de backup diferente da revisao verificada.")
    reset = {{"__name__": "llm2_verified_reset"}}
    exec(compile(source.decode("utf-8"), url, "exec"), reset)
    if MODO == "backup":
        LLM2_BACKUP_INFO = reset["backup_working"]()
        print("PowerShell local: Get-FileHash -LiteralPath 'CAMINHO_DO_ZIP_BAIXADO' -Algorithm SHA256")
        try:
            from IPython.display import display, FileLink
            display(FileLink(LLM2_BACKUP_INFO["path"]))
        except ImportError:
            pass
    else:
        LLM2_CLEANUP_INFO = reset["clear_working"](BACKUP_LOCAL_SHA256)
elif MODO == "desativado":
    print("Backup/limpeza desativados.")
else:
    raise ValueError('MODO deve ser "backup" ou "limpar".')
'''
    train=f'''import hashlib
import time
from urllib.request import urlopen

RUN_CAMPAIGN = {active!r}
LLM2_REVISION = {revision!r}
LLM2_SETTINGS = {{
    "run_number": globals().get("LLM2_CLEANUP_INFO", {{}}).get("next_run_number"),
    "target_seconds": 7200,
    "session_seconds": 10800,
}}

if RUN_CAMPAIGN:
    started_at = time.monotonic()
    print("Diagnostico fractal: 18 treinos; meta 2h, teto 3h.", flush=True)
    if LLM2_SETTINGS["run_number"] is None:
        raise ValueError("Preencha run_number com o numero do recibo da limpeza (minimo 23).")
    url = (
        "https://raw.githubusercontent.com/PedroGGC/-LLM-2-/"
        f"{{LLM2_REVISION}}/src/llm2/research/cycle4_bootstrap.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != {hashes["cycle4_bootstrap.py"]!r}:
        raise ValueError("Bootstrap diferente da revisao verificada.")
    namespace = {{"__name__": "llm2_cycle4_bootstrap"}}
    exec(compile(source.decode("utf-8"), url, "exec"), namespace)
    namespace["bootstrap_git"](
        LLM2_REVISION, run=True, started_at=started_at, **LLM2_SETTINGS
    )
else:
    print("Treino desativado.")
'''
    return dict(zip(NAMES,(backup,train)))


def build(directory,revision):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    names=(*NAMES,"release.json")
    if any((directory/n).exists() for n in names):raise FileExistsError("use a fresh release directory")
    for name,code in cells(revision).items():
        compile(code,name,"exec")
        with (directory/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(code)
    manifest={"revision":revision,"target_seconds":7200,"session_seconds":10800,
        "seeds":[211,212,213],"steps":6000,"variants":6,"evaluations":72,
        "files":{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in NAMES}}
    with (directory/"release.json").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(manifest,indent=2)+"\n")
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True);p.add_argument("--revision",required=True)
    a=p.parse_args();print(json.dumps(build(a.output_dir,a.revision),indent=2))
