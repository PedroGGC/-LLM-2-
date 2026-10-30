"""Standalone exact-revision Cycle 4 bootstrap, target 2h and hard limit 3h."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import time
from urllib.request import urlopen


def bootstrap_git(revision, *, run=False, run_number=None, target_seconds=7200., session_seconds=10800., started_at=None):
    if not run: return {"status": "not_started"}
    now=time.monotonic();started_at=now if started_at is None else float(started_at)
    if (not re.fullmatch(r"[0-9a-f]{40}",revision) or not math.isfinite(started_at) or not 0<=started_at<=now
        or target_seconds!=7200 or session_seconds!=10800
        or type(run_number) is not int or run_number<23):
        raise ValueError("Cycle 4 requires 2h target, 3h hard limit, exact revision and number >=23")
    url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{revision}/src/llm2/research/cycle2_bootstrap.py"
    with urlopen(url,timeout=60) as response: payload=response.read(1024**2+1)
    if len(payload)>1024**2:raise ValueError("oversized helper")
    helper={"__name__":"cycle4_helper"};exec(compile(payload.decode("utf-8"),url,"exec"),helper)
    session=helper["_BootstrapSession"](session_seconds,started_at)
    hardware=helper["_hardware"]()
    work,gpus,before_torch=hardware
    if len(gpus)!=2:raise ValueError("two isolated T4 devices required")
    source,parent,metadata=helper["checkout_git_source"](revision,session,hardware)
    if hashlib.sha256((source/"src/llm2/research/cycle2_bootstrap.py").read_bytes()).digest()!=hashlib.sha256(payload).digest():
        raise ValueError("helper differs from pinned source")
    # Do not import a stale editable llm2 from the notebook kernel.
    reset=runpy.run_path(str(source/"src/llm2/research/kaggle_reset.py"))
    if reset["active_workers"]():raise RuntimeError("another project worker is active")
    command=helper["bounded_command"]
    command([sys.executable,"-m","pip","install","--quiet","-e",str(source)+"[dev,research,wikitext]"],
        cwd=source,timeout=session.remaining())
    if importlib.metadata.version("torch")!=before_torch:raise RuntimeError("installation changed Torch")
    numbering=runpy.run_path(str(source/"src/llm2/research/naming.py"))["numbered_output"]
    candidate=run_number
    while True:
        try:
            output=numbering(work,number=candidate,search_roots=[Path("/kaggle/input")]);break
        except FileExistsError:
            candidate+=1
    metadata.update(source_root=str(source),torch_preserved=before_torch,
        setup_elapsed_seconds=session.elapsed,session_limit_seconds=session.limit,target_seconds=target_seconds)
    meta=parent/"bootstrap-cycle4.json"
    meta.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8",newline="\n")
    env=os.environ.copy()
    env.update(PYTHONPATH=str(source/"src"),PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8",
        LLM2_CYCLE4="1",LLM2_COMPILER_CACHE_ROOT=str(output/"compiler-cache"),TORCH_LOGS="recompiles")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    arguments=[sys.executable,str(source/"scripts/run_cycle4.py"),"--execute","--cache",work/"llm2-wikitext-pinned-cache",
        "--download","--output",output,"--gpus",*gpus,"--session-seconds",session.limit,
        "--target-seconds",target_seconds,"--export-reserve-seconds",300,
        "--bootstrap-metadata",meta,"--bootstrap-elapsed-seconds",session.elapsed]
    print("Ciclo 4: 18 treinos, meta 2h, teto 3h; validação e piloto antes dos treinos.",flush=True)
    try:command(arguments,cwd=source,timeout=session.remaining(export=True),env=env)
    finally:print(f"Resultados: {output.name}*-results.zip; baixe também {output.name}-checkpoints.zip",flush=True)
    return {"campaign":str(output),"session_elapsed_seconds":session.elapsed}
