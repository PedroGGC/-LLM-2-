"""Pinned single-checkout training bootstrap; reuses immutable prepared data."""
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

HELPER_REVISION="1c8ce2abb8f47d3167454b3503598854b7756dad"
HELPER_SHA256="5bc8f2ddcaf3cacbae401cbe077b494ddc9fc90c798417e3f1686d2635f925b7"
GUARD_SHA256="8c05edcf6d0450e273ac2f1fc118a8167bc9833490f841c9cfb1d57d96262007"
VARIANTS=("hierarchy_next","hierarchy_immediate","recurrent","fine_only","transformer_two")


def existing_data(work):
    candidates=[Path(work)/"llm2-campaign-0027/data"]
    for path in sorted(Path(work).glob("llm2-campaign-*/campaign.json")):
        try:value=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,ValueError):continue
        if value.get("cycle")=="state_readout_v1" and value.get("dataset_path"):
            candidates.append(Path(value["dataset_path"]))
    for folder in candidates:
        if not folder.is_symlink() and all((folder/name).is_file() and not (folder/name).is_symlink()
                for name in ("manifest.json","tokenizer.json","train.bin","val.bin","test.bin")):
            return folder.resolve()
    return None


def bootstrap_git(revision,*,run=False,run_number=29,session_seconds=7200,variants=VARIANTS,started_at=None):
    if not run:return {"status":"not_started"}
    if (not re.fullmatch(r"[0-9a-f]{40}",revision) or type(run_number) is not int or run_number<29 or
            not 300<float(session_seconds)<=7200 or not variants or len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS)):
        raise ValueError("exact revision, valid variants/number>=29 and session<=2h required")
    print("Controle fino/leitura: até5treinos de6000passos, seed251, duas T4; limite2h.",flush=True)
    started_at=time.monotonic() if started_at is None else started_at
    url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{HELPER_REVISION}/src/llm2/research/cycle2_bootstrap.py"
    with urlopen(url,timeout=60) as response:source=response.read(1024**2+1)
    if hashlib.sha256(source).hexdigest()!=HELPER_SHA256:raise ValueError("checkout helper differs")
    helper={"__name__":"readout_checkout_helper"};exec(compile(source.decode("utf-8"),url,"exec"),helper)
    session=helper["_BootstrapSession"](session_seconds,started_at)
    hardware=helper["_hardware"]();work,gpus,before_torch=hardware
    if len(gpus)!=2:raise RuntimeError("enable two T4 GPUs")
    url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{HELPER_REVISION}/src/llm2/research/kaggle_light.py"
    with urlopen(url,timeout=60) as response:source=response.read(1024**2+1)
    if hashlib.sha256(source).hexdigest()!=GUARD_SHA256:raise ValueError("cleanup guard differs")
    guard={"__name__":"readout_cleanup_guard"};exec(compile(source.decode("utf-8"),url,"exec"),guard)
    if guard["active_workers"]():raise RuntimeError("project workers active; duplicate training blocked")
    parent=None
    try:
        root,parent,metadata=helper["checkout_git_source"](revision,session,hardware,shallow=True)
        globals()["download_file"]=runpy.run_path(str(root/"src/llm2/research/notebook_download.py"))["download_file"]
        return dispatch(root,parent,metadata,helper,session,hardware,run_number,session_seconds,variants)
    except Exception:
        if parent is not None:
            try:preserve_preflight_error(work,parent)
            except Exception as exc:print(f"Erro original permanece no log; recibo indisponível:{exc}",flush=True)
        raise
    finally:
        if parent is not None:
            try:
                row=guard["remove_checkout"](work,parent)
                print(f"Checkout temporário removido:{row['bytes']/1024**2:.1f}MiB.",flush=True)
            except Exception as exc:print(f"Checkout preservado:{parent}: {exc}",flush=True)


def preserve_preflight_error(work,parent):
    path=Path(parent)/"bootstrap.json"
    if not path.is_file():return None
    value=json.loads(path.read_text(encoding="utf-8"))
    if value.get("status")!="failed":return None
    receipt=Path(work)/f"llm2-readout-preflight-error-{time.time_ns()}.json"
    with receipt.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write("\n")
    print(f"Diagnóstico preservado:{receipt}",flush=True)
    if "download_file" in globals():download_file(receipt)
    return receipt


def dispatch(root,parent,metadata,helper,session,hardware,run_number,session_seconds,variants):
    work,gpus,before_torch=hardware
    data=existing_data(work);cache=work/"llm2-wikitext-verified-cache"
    meta=parent/"bootstrap.json"
    if data is None:
        helper["bounded_command"]([sys.executable,str(root/"src/llm2/research/wikitext_cache.py"),"--base",work,
            "--source",root/"src/llm2/research/wikitext.py","--source-sha256",metadata["source_files_sha256"]["src/llm2/research/wikitext.py"],
            "--report",meta],cwd=root,timeout=session.remaining())
    else:print(f"Dados preparados encontrados:{data}; hashes serão conferidos.",flush=True)
    helper["bounded_command"]([sys.executable,"-m","pip","install","--quiet","--no-cache-dir","-e",str(root)+"[dev,research,wikitext]"],cwd=root,timeout=session.remaining())
    if importlib.metadata.version("torch")!=before_torch:raise RuntimeError("installation changed Torch")
    numbering=runpy.run_path(str(root/"src/llm2/research/naming.py"))["numbered_output"]
    while True:
        try:output=numbering(work,number=run_number,search_roots=[Path("/kaggle/input")]);break
        except FileExistsError:run_number+=1
    metadata.update(torch_preserved=before_torch,session_seconds=session_seconds,download_mode="notebook_embedded_base64",
        scope="new training protocol; historical qualification not rewritten; all jobs start fresh")
    meta.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8",newline="\n")
    command=[sys.executable,"-m","llm2.research.state_readout","--output",output,"--gpus",*gpus,
        "--session-seconds",str(session_seconds),"--bootstrap-elapsed-seconds",str(session.elapsed),"--bootstrap-metadata",meta,
        "--variants",*variants]
    command.extend(["--data",str(data)] if data else ["--cache",str(cache),"--download"])
    env=os.environ.copy();env.update(PYTHONPATH=str(root/"src"),PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    try:helper["bounded_command"](command,cwd=root,timeout=session.remaining(export=True),env=env)
    finally:
        for kind in ("results","checkpoints"):
            path=work/f"{output.name}-{kind}.zip"
            if path.is_file():
                try:download_file(path)
                except Exception as exc:print(f"Arquivo preservado:{path}; botão indisponível:{exc}",flush=True)
    return {"campaign":str(output),"result":str(work/f"{output.name}-results.zip")}
