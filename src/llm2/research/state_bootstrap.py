"""Exact-revision Kaggle bootstrap; Stage A and B have separate session clocks."""
import importlib.metadata
import json
from pathlib import Path
import re
import runpy
import sys
import time
from urllib.request import urlopen


def bootstrap_git(revision,*,run=False,phase="qualify",run_number=25,platform_report=None,tuning_report=None,
                  seeds=(241,242,243),target_seconds=7200.,session_seconds=10800.,started_at=None):
    if not run:return {"status":"not_started"}
    if (not re.fullmatch(r"[0-9a-f]{40}",revision) or phase not in ("qualify","experiment")
        or type(run_number) is not int or run_number<24 or target_seconds!=7200 or session_seconds!=10800
        or not seeds or len(set(seeds))!=len(seeds) or not set(seeds).issubset((241,242,243))):
        raise ValueError("exact revision, valid phase/seeds, number>=24 and 2h/3h bounds required")
    started_at=time.monotonic() if started_at is None else started_at
    print(f"Recorrência/fractal: etapa {phase}; meta2h/teto3h (inclui preparação).",flush=True)
    url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{revision}/src/llm2/research/cycle2_bootstrap.py"
    with urlopen(url,timeout=60) as response:payload=response.read(1024**2+1)
    if len(payload)>1024**2:raise ValueError("oversized helper")
    helper={"__name__":"state_bootstrap_helper"};exec(compile(payload.decode("utf-8"),url,"exec"),helper)
    session=helper["_BootstrapSession"](session_seconds,started_at)
    hardware=helper["_hardware"]();work,gpus,before_torch=hardware
    if len(gpus)!=2:raise RuntimeError("two T4 GPUs required")
    source,parent,metadata=helper["checkout_git_source"](revision,session,hardware,shallow=True)
    if (source/"src/llm2/research/cycle2_bootstrap.py").read_bytes()!=payload:raise ValueError("helper/source mismatch")
    maintenance=runpy.run_path(str(source/"src/llm2/research/kaggle_light.py"))
    downloads=runpy.run_path(str(source/"src/llm2/research/notebook_download.py"))
    globals()["download_file"]=downloads["download_file"]
    if maintenance["active_workers"]():raise RuntimeError("project workers are active")
    command=helper["bounded_command"]
    try:
        return _run(revision,phase,run_number,platform_report,tuning_report,seeds,session_seconds,target_seconds,
                    session,hardware,source,parent,metadata,maintenance,command)
    finally:
        try:
            receipt=maintenance["remove_checkout"](work,parent)
            print(f"Checkout temporário removido: {receipt['bytes']/1024**2:.1f} MiB.",flush=True)
        except Exception as exc:
            print(f"Checkout preservado para inspeção: {parent}: {exc}",flush=True)


def _run(revision,phase,run_number,platform_report,tuning_report,seeds,session_seconds,target_seconds,
         session,hardware,source,parent,metadata,maintenance,command):
    work,gpus,before_torch=hardware
    command([sys.executable,"-m","pip","install","--quiet","--no-cache-dir","-e",str(source)+"[dev,research,wikitext]"],
            cwd=source,timeout=session.remaining())
    if importlib.metadata.version("torch")!=before_torch:raise RuntimeError("installation changed Torch")
    if phase=="experiment" and platform_report is None:
        candidates=sorted(work.glob("llm2-campaign-*/platforms.json"),key=lambda p:p.stat().st_mtime,reverse=True)
        # Source/versions/qualification checked again by the actual worker.
        for candidate in candidates:
            value=json.loads(candidate.read_text(encoding="utf-8"))
            if not value.get("engineering_smoke") and value.get("selected_rule") and value.get("qualification_version")==2:
                platform_report=str(candidate);break
        if platform_report is None:raise ValueError("run Stage A first or supply its platforms.json explicitly")
    numbering=runpy.run_path(str(source/"src/llm2/research/naming.py"))["numbered_output"]
    while True:
        try:output=numbering(work,number=run_number,search_roots=[Path("/kaggle/input")]);break
        except FileExistsError:run_number+=1
    metadata.update(source_root=str(source),torch_preserved=before_torch,phase=phase,
        checkout_bytes=maintenance["tree_bytes"](parent),checkout_retention="delete after command exits",
        compiler_retention="per-candidate cleanup and final cleanup; 2 GiB monitored limit",
        download_mode="notebook_embedded_base64; bounded parts for large files")
    meta=parent/"bootstrap-state.json";meta.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8",newline="\n")
    args=[sys.executable,"-m","llm2.research.state_study","--execute","--phase",phase,"--output",str(output),
        "--gpus",*gpus,"--session-seconds",str(session_seconds),"--target-seconds",str(target_seconds),
        "--bootstrap-elapsed-seconds",str(session.elapsed),"--bootstrap-metadata",str(meta)]
    if phase=="experiment":
        args.extend(["--platform-report",str(platform_report),"--cache",str(work/"llm2-wikitext-pinned-cache"),
                     "--download","--seeds",*map(str,seeds)])
        if tuning_report:args.extend(["--tuning-report",str(tuning_report)])
    import os
    env=os.environ.copy();env.update(PYTHONPATH=str(source/"src"),PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8",
        TORCH_LOGS="recompiles");env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    try:command(args,cwd=source,timeout=session.remaining(export=True),env=env)
    finally:
        print(f"Resultados: {output.name}-results.zip",flush=True)
        # Also available after a failed/partial campaign, once the worker exports.
        for kind in ("results","checkpoints"):
            archive=work/f"{output.name}-{kind}.zip"
            if archive.is_file():
                try:download_file(archive)
                except Exception as exc:print(f"Arquivo preservado; botão indisponível: {archive}: {exc}",flush=True)
    return {"campaign":str(output),"platform_report":str(output/"platforms.json") if phase=="qualify" else platform_report,
        "tuning_report":str(output/"campaign.json") if phase=="experiment" else None}
