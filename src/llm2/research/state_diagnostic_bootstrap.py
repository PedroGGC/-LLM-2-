"""Short Kaggle delivery: frozen 0027 weights, eager evaluation, one small ZIP."""
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

SCIENTIFIC_REVISION = "1c8ce2abb8f47d3167454b3503598854b7756dad"
HELPER_SHA256 = "5bc8f2ddcaf3cacbae401cbe077b494ddc9fc90c798417e3f1686d2635f925b7"
GUARD_SHA256 = "8c05edcf6d0450e273ac2f1fc118a8167bc9833490f841c9cfb1d57d96262007"
FILES = ("src/llm2/research/state_diagnostic.py", "src/llm2/research/state_diagnostic_bootstrap.py",
         "src/llm2/research/wikitext_cache.py", "docs/results/campaign_0027_diagnostic_reference.json",
         "tests/test_state_diagnostic.py")


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def bootstrap_git(revision, *, run=False, run_number=28, session_seconds=7200,
                  original_campaign="/kaggle/working/llm2-campaign-0027", started_at=None):
    if not run:return {"status":"not_started"}
    if (not re.fullmatch(r"[0-9a-f]{40}",revision) or type(run_number) is not int or run_number<28
            or not 300<float(session_seconds)<=10800):raise ValueError("invalid revision/number/session")
    started_at=time.monotonic() if started_at is None else started_at
    print("Diagnóstico0027: pesos congelados;11condições, duas T4; sem treino/compilação.",flush=True)
    url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{SCIENTIFIC_REVISION}/src/llm2/research/cycle2_bootstrap.py"
    with urlopen(url,timeout=60) as response:payload=response.read(1024**2+1)
    if hashlib.sha256(payload).hexdigest()!=HELPER_SHA256:raise ValueError("pinned helper differs")
    helper={"__name__":"diagnostic_checkout_helper"}
    exec(compile(payload.decode("utf-8"),url,"exec"),helper)
    session=helper["_BootstrapSession"](session_seconds,started_at)
    hardware=helper["_hardware"]();work,gpus,before_torch=hardware
    if len(gpus)!=2:raise RuntimeError("two T4 GPUs required")
    guard_url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{SCIENTIFIC_REVISION}/src/llm2/research/kaggle_light.py"
    with urlopen(guard_url,timeout=60) as response:guard_bytes=response.read(1024**2+1)
    if hashlib.sha256(guard_bytes).hexdigest()!=GUARD_SHA256:raise ValueError("maintenance guard differs")
    guard={"__name__":"diagnostic_maintenance"}
    exec(compile(guard_bytes.decode("utf-8"),guard_url,"exec"),guard)
    if guard["active_workers"]():raise RuntimeError("project workers active; duplicate diagnosis blocked")
    parents=[]
    try:
        operator,op_parent,op_meta=helper["checkout_git_source"](revision,session,hardware,shallow=True);parents.append(op_parent)
        worker,worker_parent,worker_meta=helper["checkout_git_source"](SCIENTIFIC_REVISION,session,hardware,shallow=True);parents.append(worker_parent)
        globals()["download_file"]=runpy.run_path(str(worker/"src/llm2/research/notebook_download.py"))["download_file"]
        return dispatch(operator,worker,op_parent,worker_parent,op_meta,worker_meta,helper,session,hardware,
                        run_number,session_seconds,original_campaign)
    finally:
        for parent in reversed(parents):
            try:
                removed=guard["remove_checkout"](work,parent)
                print(f"Checkout temporário removido: {removed['bytes']/1024**2:.1f}MiB.",flush=True)
            except Exception as exc:print(f"Checkout preservado: {parent}: {exc}",flush=True)


def dispatch(operator,worker,op_parent,worker_parent,op_meta,worker_meta,helper,session,hardware,
             run_number,session_seconds,original_campaign):
    work,gpus,before_torch=hardware
    for name in FILES:
        if sha(operator/name)!=op_meta["source_files_sha256"].get(name):raise ValueError("diagnostic source differs")
    reference=operator/"docs/results/campaign_0027_diagnostic_reference.json"
    ref=json.loads(reference.read_text(encoding="utf-8"))
    actual={name:digest for name,digest in worker_meta["source_files_sha256"].items()
            if name.startswith("src/") and name.endswith(".py") or name=="pyproject.toml"}
    if actual!=ref["source_sha256"] or ref["scientific_revision"]!=SCIENTIFIC_REVISION:
        raise ValueError("original scientific source mismatch")
    diagnostic=runpy.run_path(str(operator/"src/llm2/research/state_diagnostic.py"))
    inputs=diagnostic["inspect_inputs"](ref,original_campaign,[work,Path("/kaggle/input")])
    print("Quatro pesos finais/metadados conferidos; nenhum pickle será desserializado.",flush=True)
    cache=work/"llm2-wikitext-verified-cache"
    if not (Path(original_campaign)/"data/manifest.json").is_file():
        helper["bounded_command"]([sys.executable,str(operator/"src/llm2/research/wikitext_cache.py"),
            "--base",work,"--source",worker/"src/llm2/research/wikitext.py",
            "--source-sha256",actual["src/llm2/research/wikitext.py"],"--report",op_parent/"bootstrap.json"],
            cwd=worker,timeout=session.remaining())
    helper["bounded_command"]([sys.executable,"-m","pip","install","--quiet","--no-cache-dir","-e",
        str(worker)+"[dev,research,wikitext]"],cwd=worker,timeout=session.remaining())
    if importlib.metadata.version("torch")!=before_torch:raise RuntimeError("installation changed Torch")
    env=os.environ.copy();env.update(PYTHONPATH=str(worker/"src"),PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    helper["bounded_command"]([sys.executable,"-m","pytest","-q",str(operator/"tests/test_state_diagnostic.py"),
        "-k","not integration"],cwd=worker,timeout=session.remaining(),env=env)
    numbering=runpy.run_path(str(worker/"src/llm2/research/naming.py"))["numbered_output"]
    while True:
        try:
            output=numbering(work,number=run_number,search_roots=[Path("/kaggle/input")]);break
        except FileExistsError:run_number+=1
    operation={"revision":op_meta["source_git_revision"],"scientific_revision":SCIENTIFIC_REVISION,
        "files_sha256":{name:sha(operator/name) for name in FILES},"inputs_preflight":inputs,
        "policy":"eager frozen-weight sensitivity; not training, inference speed or trained ablation",
        "download_mode":"notebook_embedded_base64"}
    metadata=op_parent/"bootstrap.json"
    metadata.write_text(json.dumps(operation,indent=2)+"\n",encoding="utf-8",newline="\n")
    command=[sys.executable,str(operator/"src/llm2/research/state_diagnostic.py"),"--reference",reference,
        "--worker-source",worker,"--operator-root",operator,"--operator-metadata",metadata,
        "--original",original_campaign,"--output",output,"--cache",cache,"--gpus",*gpus,
        "--session-seconds",str(session_seconds),"--bootstrap-elapsed-seconds",str(session.elapsed)]
    try:helper["bounded_command"](command,cwd=worker,timeout=session.remaining(export=True),env=env)
    finally:
        bundle=work/(output.name+"-results.zip")
        if bundle.is_file():
            try:download_file(bundle)
            except Exception as exc:print(f"Resultado preservado: {bundle}; botão indisponível: {exc}",flush=True)
    return {"campaign":str(output),"results":str(work/(output.name+"-results.zip")),"optimizer_steps":0}
