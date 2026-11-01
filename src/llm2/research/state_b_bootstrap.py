"""Standalone notebook bootstrap: new operation, unchanged qualified worker."""
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
PLATFORM_SHA256 = "cf281d13a522724eabfe84e1e1a59b4244b5093971e9067867030db5b1894da3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_git(revision, *, run=False, run_number=27, seeds=(241,), tuning_report=None,
                  target_seconds=7200, session_seconds=10800, started_at=None):
    if not run:
        return {"status":"not_started"}
    if (not re.fullmatch(r"[0-9a-f]{40}",revision) or type(run_number) is not int or run_number<26
            or target_seconds!=7200 or session_seconds!=10800 or not seeds
            or len(set(seeds))!=len(seeds) or not set(seeds).issubset((241,242,243))):
        raise ValueError("exact revision, unique protocol seeds, number>=26 and 2h/3h bounds required")
    started_at = time.monotonic() if started_at is None else started_at
    print("Etapa B: mLSTM simples, hierarquia independente/compartilhada e Transformer; meta2h/teto3h.", flush=True)
    url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{SCIENTIFIC_REVISION}/src/llm2/research/cycle2_bootstrap.py"
    with urlopen(url,timeout=60) as response:
        payload=response.read(1024**2+1)
    if hashlib.sha256(payload).hexdigest()!=HELPER_SHA256:
        raise ValueError("unexpected pinned checkout helper")
    helper={"__name__":"state_b_checkout_helper"}
    exec(compile(payload.decode("utf-8"),url,"exec"),helper)
    session=helper["_BootstrapSession"](session_seconds,started_at)
    hardware=helper["_hardware"]()
    work,gpus,before_torch=hardware
    if len(gpus)!=2:
        raise RuntimeError("two T4 GPUs required")
    # Same stdlib guard as the qualified release; no torch/package import needed.
    guard_url=f"https://raw.githubusercontent.com/PedroGGC/-LLM-2-/{SCIENTIFIC_REVISION}/src/llm2/research/kaggle_light.py"
    with urlopen(guard_url,timeout=60) as response:
        guard_source=response.read(1024**2+1)
    if hashlib.sha256(guard_source).hexdigest()!="8c05edcf6d0450e273ac2f1fc118a8167bc9833490f841c9cfb1d57d96262007":
        raise ValueError("unexpected maintenance guard")
    maintenance={"__name__":"state_b_maintenance"}
    exec(compile(guard_source.decode("utf-8"),guard_url,"exec"),maintenance)
    if maintenance["active_workers"]():
        raise RuntimeError("project workers active; do not start duplicate training")
    parents=[]
    try:
        operator,op_parent,op_checkout=helper["checkout_git_source"](revision,session,hardware,shallow=True)
        parents.append(op_parent)
        worker,worker_parent,worker_checkout=helper["checkout_git_source"](SCIENTIFIC_REVISION,session,hardware,shallow=True)
        parents.append(worker_parent)
        if (worker/"src/llm2/research/cycle2_bootstrap.py").read_bytes()!=payload:
            raise ValueError("scientific helper/checkout mismatch")
        downloads=runpy.run_path(str(worker/"src/llm2/research/notebook_download.py"))
        globals()["download_file"]=downloads["download_file"]
        return _run(operator,worker,op_parent,worker_parent,op_checkout,worker_checkout,
                    helper,session,hardware,run_number,seeds,tuning_report,target_seconds,session_seconds)
    except Exception:
        try:
            if "op_parent" in locals():export_cache_failure(work,op_parent)
        except Exception as exc:
            print(f"Diagnóstico continua no log; não consegui exportar o recibo: {exc}",flush=True)
        raise
    finally:
        for parent in reversed(parents):
            try:
                row=maintenance["remove_checkout"](work,parent)
                print(f"Checkout temporário removido: {row['bytes']/1024**2:.1f} MiB.",flush=True)
            except Exception as exc:
                print(f"Checkout preservado para inspeção: {parent}: {exc}",flush=True)


def export_cache_failure(work,parent):
    path=Path(parent)/"bootstrap.json"
    if not path.is_file():return None
    report=json.loads(path.read_text(encoding="utf-8"))
    if report.get("status")!="failed":return None
    receipt=Path(work)/f"llm2-data-cache-error-{time.time_ns()}.json"
    with receipt.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False);stream.write("\n")
    print(f"Diagnóstico do cache preservado: {receipt.name}",flush=True)
    if "download_file" in globals():download_file(receipt)
    return receipt


def _run(operator,worker,op_parent,worker_parent,op_checkout,worker_checkout,
         helper,session,hardware,run_number,seeds,tuning_report,target_seconds,session_seconds):
    work,gpus,before_torch=hardware
    report=operator/"docs/results/campaign_0025_platforms.json"
    if sha(report)!=PLATFORM_SHA256:
        raise ValueError("campaign0025 qualification was changed")
    qualified=json.loads(report.read_text(encoding="utf-8"))
    # Compare the entire scientific source map. Do not rewrite the qualification.
    actual={name:d for name,d in worker_checkout["source_files_sha256"].items()
            if name.startswith("src/") and name.endswith(".py") or name=="pyproject.toml"}
    if qualified["source_sha256"]!=actual or qualified.get("selected_rule")!="mlstm":
        raise ValueError("qualified source map differs from scientific checkout")
    names=("src/llm2/research/state_b_operator.py","src/llm2/research/state_b_bootstrap.py","src/llm2/research/wikitext_cache.py",
           "docs/results/campaign_0025_platforms.json")
    for name in names:
        if sha(operator/name)!=op_checkout["source_files_sha256"].get(name):
            raise ValueError("operator checkout file hash mismatch")
    meta=op_parent/"bootstrap.json"
    # Recover raw inputs before spending time on installation or CUDA compilation.
    helper["bounded_command"]([sys.executable,str(operator/"src/llm2/research/wikitext_cache.py"),
        "--base",str(work),"--source",str(worker/"src/llm2/research/wikitext.py"),
        "--source-sha256",worker_checkout["source_files_sha256"]["src/llm2/research/wikitext.py"],
        "--report",str(meta)],cwd=worker,timeout=session.remaining())
    raw_cache=json.loads(meta.read_text(encoding="utf-8"))
    if raw_cache.get("status")!="verified":raise ValueError("raw corpus cache did not pass preflight")
    selected_cache=Path(raw_cache["selected_cache"])
    if selected_cache.is_symlink() or selected_cache.resolve().parent!=work.resolve():
        raise ValueError("raw cache path outside working")
    op_meta={"revision":op_checkout["source_git_revision"],"scientific_revision":SCIENTIFIC_REVISION,
             "files_sha256":{name:sha(operator/name) for name in names},
             "policy":"variant-scoped per-GPU cache; grouped tuning; paired train/isolated profiles",
             "qualified_platform_sha256":PLATFORM_SHA256,"raw_cache_preflight":raw_cache,
             "scope":"operation only; model/optimizer/sampler/evaluation/checkpoints from unchanged worker"}
    meta.write_text(json.dumps(op_meta,indent=2)+"\n",encoding="utf-8",newline="\n")
    helper["bounded_command"]([sys.executable,"-m","pip","install","--quiet","--no-cache-dir","-e",
                                str(worker)+"[dev,research,wikitext]"],cwd=worker,timeout=session.remaining())
    if importlib.metadata.version("torch")!=before_torch:
        raise RuntimeError("installation changed Torch")
    numbering=runpy.run_path(str(worker/"src/llm2/research/naming.py"))["numbered_output"]
    while True:
        try:
            output=numbering(work,number=run_number,search_roots=[Path("/kaggle/input")])
            break
        except FileExistsError:
            run_number+=1
    worker_checkout.update(source_root=str(worker),torch_preserved=before_torch,phase="experiment",
        operator_revision=op_meta["revision"],operator=op_meta,download_mode="notebook_embedded_base64",
        target_seconds=target_seconds,session_seconds=session_seconds,seeds=list(seeds))
    bootstrap=worker_parent/"bootstrap-state.json"
    bootstrap.write_text(json.dumps(worker_checkout,indent=2)+"\n",encoding="utf-8",newline="\n")
    command=[sys.executable,str(operator/"src/llm2/research/state_b_operator.py"),
        "--worker-source",str(worker),"--operator-root",str(operator),"--operator-metadata",str(meta),
        "--execute","--phase","experiment","--platform-report",str(report),"--output",str(output),
        "--gpus",*gpus,"--seeds",*map(str,seeds),"--cache",str(selected_cache),"--download",
        "--target-seconds",str(target_seconds),"--session-seconds",str(session_seconds),
        "--bootstrap-elapsed-seconds",str(session.elapsed),"--bootstrap-metadata",str(bootstrap)]
    if tuning_report:
        command.extend(["--tuning-report",str(tuning_report)])
    env=os.environ.copy()
    env.update(PYTHONPATH=str(worker/"src"),PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8",TORCH_LOGS="recompiles")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print(f"Operação: {op_meta['revision']} | ciência qualificada: {SCIENTIFIC_REVISION}",flush=True)
    try:
        helper["bounded_command"](command,cwd=worker,timeout=session.remaining(export=True),env=env)
    finally:
        for kind in ("results","checkpoints"):
            path=work/f"{output.name}-{kind}.zip"
            if path.is_file():
                try:download_file(path)
                except Exception as exc:print(f"Arquivo preservado: {path}; botão indisponível: {exc}",flush=True)
    return {"campaign":str(output),"scientific_revision":SCIENTIFIC_REVISION,
            "operator_revision":op_meta["revision"],"tuning_report":str(output/"campaign.json")}
