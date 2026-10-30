"""Bounded parallel scale diagnosis with isolated cost measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import threading
import time
import traceback
import zipfile

import torch
import yaml
from ..checkpoint import read_checkpoint_metadata
from ..config import load_config
from ..data import write_json
from .campaign import worker_environment
from .cycle3 import prepare
from .cycle3_short import heartbeat
from .cycle4_protocol import ROOT, PROTOCOL, PROTOCOL_ID, SEEDS, VARIANTS, ordered_jobs
from .cycle4_report import analyze, save as save_report
from .factory import build_backbone
from .pairing import compute_paired_init_provenance
from .session import SessionCampaign, SessionBudget, SessionBudgetExceeded


@contextmanager
def phase(campaign,name,**details):
    started=time.monotonic();row={"phase":name,**details}
    with campaign.lock:
        campaign.manifest.setdefault("phases",[]).append(row)
    try:
        yield
    finally:
        with campaign.lock:
            row["wall_seconds"]=time.monotonic()-started
            campaign.save()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data");source.add_argument("--cache")
    p.add_argument("--download",action="store_true");p.add_argument("--output")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--gpus",nargs="+",default=["0","1"])
    p.add_argument("--execute",action="store_true")
    p.add_argument("--engineering-smoke",action="store_true")
    p.add_argument("--skip-tests",action="store_true")
    p.add_argument("--session-seconds",type=float,default=10800.)
    p.add_argument("--target-seconds",type=float,default=7200.)
    p.add_argument("--export-reserve-seconds",type=float,default=300.)
    p.add_argument("--bootstrap-elapsed-seconds",type=float,default=0.)
    p.add_argument("--bootstrap-metadata")
    return p


def validate(args):
    if (args.device=="cpu") != args.engineering_smoke:
        raise ValueError("CPU is engineering smoke only; science requires CUDA")
    if len(args.gpus)!=2 or len(set(args.gpus))!=2 or not all(s.isdecimal() for s in args.gpus):
        raise ValueError("exactly two distinct logical slots required")
    if (not 300<args.session_seconds<=10800 or not 0<args.target_seconds<=args.session_seconds
        or args.export_reserve_seconds<300):
        raise ValueError("invalid session bounds")
    if not args.engineering_smoke and (args.session_seconds!=10800 or args.target_seconds!=7200 or args.skip_tests):
        raise ValueError("science requires 2h target/3h hard cap and focused tests")
    if args.execute and not args.engineering_smoke and not args.bootstrap_metadata:
        raise ValueError("science requires pinned bootstrap provenance")
    if args.execute and args.device=="cuda" and (not torch.cuda.is_available() or any(
        int(s)>=torch.cuda.device_count() or "T4" not in torch.cuda.get_device_name(int(s)) for s in args.gpus)):
        raise ValueError("two T4 GPUs required")


def materialize(campaign,args,data):
    directory=campaign.output/"configs";directory.mkdir()
    jobs=ordered_jobs(data,campaign.output,args.gpus,device=args.device,smoke=args.engineering_smoke)
    common={}
    for job in jobs:
        file=directory/(job["id"]+".yaml")
        file.write_text(yaml.safe_dump(job["config_resolved"],sort_keys=True),encoding="utf-8",newline="\n")
        job["config"]=str(file)
        job["checkpoint_interval"]=job["max_steps"]//4
        job["physical_gpu"]=(worker_environment(ROOT,job["gpu_slot"]).get("CUDA_VISIBLE_DEVICES")
                             if args.device=="cuda" else None)
        cfg=load_config(file)
        if cfg.research.paired_common_init:
            model=build_backbone(cfg)
            provenance=compute_paired_init_provenance(model,seed=job["seed"])
            current=provenance["common_combined_sha256"]
            if job["seed"] in common and common[job["seed"]]!=current:
                raise ValueError("common initial parameters differ across recurrent variants")
            common[job["seed"]]=current;job["initial_common_sha256"]=current
            del model
    campaign.manifest.update(runs=jobs,paired_init_verification=common,
        sampler_policy="RandomBatchStream(seed+1), identical T/batch/horizon per seed; private sampler per process")
    campaign.save()


def command(campaign,args,arguments,name,*,slot=None,job=None):
    return campaign.command(arguments,name,gpu=slot if args.device=="cuda" else None,job=job)


def tests(campaign,args):
    names=["tests/test_cycle4_models.py","tests/test_cycle4_operations.py","tests/test_kaggle_reset.py",
           "tests/test_json_portability.py"]
    calls=[["-m","pytest","-q",*names,"-m","not cuda"]]
    if args.device=="cuda":
        calls.append(["-m","pytest","-q","tests/test_cycle4_models.py","-m","cuda"])
    campaign.manifest["validation_commands"]=calls
    for i,call in enumerate(calls):
        with phase(campaign,"validation_focused"):
            command(campaign,args,call,f"tests-focused-{i}.log",slot=args.gpus[0])


def require_idle(campaign):
    with campaign.lock:
        if campaign.processes: raise RuntimeError("isolated measurement attempted while another worker was active")


def pilot(campaign,args):
    require_idle(campaign)
    records=[];durations={}
    for slot in args.gpus:
        for variant in VARIANTS:
            job=next(j for j in campaign.manifest["runs"] if j["variant"]==variant)
            folder=campaign.output/"pilots"/f"{variant}-gpu-{slot}"
            with phase(campaign,"pilot_isolated",variant=variant,gpu_slot=slot):
                command(campaign,args,["-m","llm2","profile","--execute","--config",job["config"],
                    "--output",folder,"--warmup-steps",0 if args.engineering_smoke else 10,
                    "--measure-steps",1 if args.engineering_smoke else 30],f"pilot-{variant}-{slot}.log",slot=slot)
            value=json.loads((folder/"profile.json").read_text())
            records.append({"variant":variant,"gpu_slot":slot,
                "mean_step_seconds":value["timing"]["mean_step_seconds"],
                "warmup_seconds":value["warmup_timing"]["total_warmup_seconds"]})
            if variant not in durations:
                target=campaign.output/"pilots"/f"{variant}-validation"
                command(campaign,args,["-m","llm2.research.cycle3_evaluate","--config",job["config"],
                    "--directory",target,"--device",args.device,"--timing-only"],f"pilot-{variant}-val.log",slot=slot)
                durations[variant]=json.loads((target/"timing.json").read_text())["validation_seconds"]
    # Disposable concurrent heavy pilots measure host contention, never quality.
    started=time.monotonic()
    def concurrent(slot):
        job=next(j for j in campaign.manifest["runs"] if j["variant"]=="flrm_full")
        folder=campaign.output/"pilots"/f"concurrent-full-{slot}"
        command(campaign,args,["-m","llm2","profile","--execute","--config",job["config"],"--output",folder,
            "--warmup-steps",0 if args.engineering_smoke else 10,"--measure-steps",1 if args.engineering_smoke else 30],
            f"pilot-concurrent-{slot}.log",slot=slot)
        return slot,json.loads((folder/"profile.json").read_text())["timing"]["mean_step_seconds"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent_records=dict(pool.map(concurrent,args.gpus))
    factor=max(1.,max(concurrent_records[s]/next(r["mean_step_seconds"] for r in records
        if r["variant"]=="flrm_full" and r["gpu_slot"]==s) for s in args.gpus))
    campaign.manifest.update(pilot=records,validation_duration_pilot=durations,
        parallel_pilot={"wall_seconds":time.monotonic()-started,"step_seconds":concurrent_records,"contention_factor":factor},
        pilot_completed_before_training=True)
    for job in campaign.manifest["runs"]:
        record=next(r for r in records if r["variant"]==job["variant"] and r["gpu_slot"]==job["gpu_slot"])
        job["estimated_job_seconds"]=(1.35*factor*record["mean_step_seconds"]*job["max_steps"]
            +record["warmup_seconds"]+4*durations[job["variant"]]*1.3+(5 if args.engineering_smoke else 120))
    campaign.save()


def forecast_seconds(jobs,slots):
    return max(sum(j["estimated_job_seconds"] for j in jobs if j["gpu_slot"]==slot) for slot in slots)


def required_storage(jobs,*,smoke=False):
    # Four milestones, a per-job archive and a final bundle; 16 bytes/parameter is conservative.
    return sum(j["registered_parameters"]*16*6 for j in jobs)+(0 if smoke else 1536*1024**2)


def admit(campaign,args):
    jobs=campaign.manifest["runs"]
    post=0 if args.engineering_smoke else 900
    prediction=forecast_seconds(jobs,args.gpus)+post
    campaign.manifest["forecast"]={"remaining_seconds":prediction,"safety_factor":1.35,
        "predicted_total_seconds":campaign.budget.elapsed+prediction+campaign.budget.reserve,
        "target_seconds":args.target_seconds,"hard_seconds":args.session_seconds}
    print("Previsão conservadora total: "+f"{campaign.manifest['forecast']['predicted_total_seconds']/60:.1f} min",flush=True)
    if campaign.manifest["forecast"]["predicted_total_seconds"]>args.target_seconds:
        print("A previsão usa parte da contingência até três horas; matriz e horizontes permanecem fixos.",flush=True)
    campaign.budget.require(prediction)
    reserve=0 if args.engineering_smoke else 2*1024**3
    needed=required_storage(jobs,smoke=args.engineering_smoke)+reserve
    campaign.manifest["storage_admission"]={"required_remaining_bytes":needed,
        "free_bytes":shutil.disk_usage(campaign.output).free,"reserve_bytes":reserve}
    if campaign.manifest["storage_admission"]["free_bytes"]<needed:
        raise OSError("insufficient space for fixed matrix, exports and 2 GiB reserve")
    campaign.save()


def report_data(campaign):
    rows=[];curves=[]
    for job in campaign.manifest["runs"]:
        if job["status"]!="complete":continue
        summary=json.loads((Path(job["directory"])/"summary.json").read_text())
        rows.append({"variant":job["variant"],"seed":job["seed"],"max_steps":job["max_steps"],
            "validation_loss":job["full_validation_loss"],"validation_tokens":job["validation_tokens"],
            "training_targets":summary["tokens_processed"],"training_flops_estimated":summary["training_flops_estimate"],
            "train_wall_seconds":summary["train_wall_seconds"],"timing_scope":"concurrent operational training",
            "peak_train_vram_bytes":summary["peak_vram_allocated_bytes"],"registered_parameters":job["registered_parameters"],
            "physical_gpu":job["physical_gpu"],"checkpoint_model_sha256":job["checkpoint_model_sha256"]})
        for point in job["checkpoint_evaluations"]:
            curves.append({"variant":job["variant"],"seed":job["seed"],"step":point["step"],
                "validation_loss":point["validation_loss"],"validation_tokens":point["validation_tokens"],
                "training_targets":point["step"]*job["batch_size"]*job["grad_accum_steps"]*job["config_resolved"]["data"]["seq_len"],
                "training_flops_estimated":point["step"]*job["step_flops_estimated"]})
    return analyze(rows,curves,campaign.manifest.get("isolated_profiles",[]),smoke=campaign.manifest["engineering_smoke"])


def archive_files(destination,files,*,base,extra=None,stored=False):
    descriptor,name=tempfile.mkstemp(prefix=".cycle4-export-",suffix=".zip",dir=destination.parent)
    os.close(descriptor);temporary=Path(name)
    try:
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(files)):
                if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
                    raise ValueError("export path outside campaign")
                if path.is_file():archive.write(path,path.relative_to(base).as_posix())
            for name,value in (extra or {}).items():archive.writestr(name,value)
        with zipfile.ZipFile(temporary) as archive:
            if len(archive.namelist())!=len(set(archive.namelist())) or archive.testzip() is not None:
                raise ValueError("export integrity failed")
        os.link(temporary,destination)
    finally:temporary.unlink(missing_ok=True)
    return destination


def checkpoint_files(campaign,jobs):
    files=[]
    for job in jobs:
        run=Path(job["directory"])
        states=sorted((run/"checkpoints").glob("step-*"),key=lambda p:int(p.name.split("-")[-1]))
        # Temporary incomplete checkpoint directories have a leading dot and cannot enter this set.
        if states:
            read_checkpoint_metadata(states[-1]);files.extend(states[-1].iterdir())
            files.extend(run/n for n in ("run.json","run.yaml") if (run/n).exists())
    return files+[campaign.output/n for n in ("campaign.json","data/manifest.json","data/tokenizer.json") if (campaign.output/n).exists()]


def publish(campaign,suffix,*,checkpoint_jobs=(),final=False):
    with campaign.lock:
        campaign.save();result=report_data(campaign)
        destination=campaign.output/"reports"/suffix
        save_report(destination,result,plot=final)
        files=[]
        shared=("campaign.json","bootstrap.json","data/manifest.json","data/tokenizer.json")
        files.extend(campaign.output/n for n in shared if (campaign.output/n).exists())
        for prefix in ("configs","logs","pilots","isolated","post"):
            folder=campaign.output/prefix
            if folder.exists():
                # Live logs and files are omitted while a different worker is active.
                if campaign.processes and prefix=="logs":continue
                if prefix=="post" and not final:continue
                files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in (
                    ".json",".jsonl",".yaml",".txt",".log",".png",".zip"))
        for job in campaign.manifest["runs"]:
            if job["status"]=="complete":
                files.extend(p for p in Path(job["directory"]).rglob("*") if p.is_file()
                    and p.suffix in (".json",".jsonl",".yaml",".txt",".log",".zip"))
        extra={"report/"+p.relative_to(destination).as_posix():p.read_bytes()
            for p in destination.rglob("*") if p.is_file()}
        stem=campaign.output.name+("" if final else "-"+suffix)
        results=archive_files(campaign.output.parent/(stem+"-results.zip"),files,base=campaign.output,extra=extra)
        backup=None
        if final or checkpoint_jobs:
            backup=archive_files(campaign.output.parent/(stem+"-checkpoints.zip"),
                checkpoint_files(campaign,campaign.manifest["runs"] if final else checkpoint_jobs),
                base=campaign.output,stored=True)
        print(f"Snapshot salvo: {results.name}"+(f" | {backup.name}" if backup else ""),flush=True)
        return results,backup


def train_parallel(campaign,args):
    publishing=threading.Lock();counter=[0]
    def worker(slot):
        for job in [j for j in campaign.manifest["runs"] if j["gpu_slot"]==slot]:
            if campaign.cancelled.is_set():return
            require_job_budget(campaign,args,job)
            with phase(campaign,"training_concurrent",job=job["id"],gpu_slot=slot):
                campaign.train_worker([job],slot if args.device=="cuda" else None)
            if job["stop_reason"]!="max_steps" or job["step"]!=job["max_steps"]:
                raise SessionBudgetExceeded("fixed horizon not reached")
            with phase(campaign,"full_validation_concurrent",job=job["id"],gpu_slot=slot):
                command(campaign,args,["-m","llm2.research.cycle4_evaluate","--config",job["config"],"--device",args.device],
                    job["id"]+"-evaluate.log",slot=slot)
            value=json.loads((Path(job["directory"])/"evaluation_bundle.json").read_text())
            if value["test_evaluated"] or value["split"]!="val":raise ValueError("invalid evaluation split")
            records=value["checkpoints"]
            if [r["step"] for r in records]!=[job["max_steps"]*i//4 for i in range(1,5)]:
                raise ValueError("missing evaluations")
            expected=(campaign.manifest["dataset_manifest"]["splits"]["val"]["tokens"]-1)//job["config_resolved"]["data"]["seq_len"]*job["config_resolved"]["data"]["seq_len"]
            for row in records:
                metadata=read_checkpoint_metadata(row["checkpoint"])
                detail=json.loads(Path(row["evaluation"]).read_text())
                if (row["checkpoint_model_sha256"]!=metadata["files"]["model.safetensors"]
                    or detail["dataset_sha256"]!=campaign.manifest["dataset_sha256"]
                    or detail["validation_tokens"]!=expected):
                    raise ValueError("validation content/weights mismatch")
            with campaign.lock:
                job.update(status="complete",checkpoint_evaluations=records,
                    full_validation_loss=records[-1]["validation_loss"],validation_tokens=expected)
                if records[-1]["checkpoint_model_sha256"]!=job["checkpoint_model_sha256"]:
                    raise ValueError("final weights mismatch")
            with publishing:
                counter[0]+=1
                publish(campaign,f"after-{counter[0]:02d}",checkpoint_jobs=[job])
    run_lanes(args.gpus,worker,campaign.abort)


def require_job_budget(campaign,args,job):
    campaign.budget.require(job["estimated_job_seconds"]+(0 if args.engineering_smoke else 900))


def run_lanes(slots,worker,on_error):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,slot) for slot in slots]
        try:
            for future in as_completed(futures):future.result()
        except BaseException:
            on_error()
            for future in futures:future.cancel()
            raise


def postprocess(campaign,args):
    require_idle(campaign)
    profiles=[]
    for variant in VARIANTS:
        job=next(j for j in campaign.manifest["runs"] if j["variant"]==variant and j["seed"]==SEEDS[0])
        for slot in args.gpus:
            folder=campaign.output/"isolated"/f"{variant}-gpu-{slot}"
            command(campaign,args,["-m","llm2","profile","--execute","--config",job["config"],"--output",folder,
                "--weights-checkpoint",job["checkpoint"],"--warmup-steps",0 if args.engineering_smoke else 10,
                "--measure-steps",1 if args.engineering_smoke else 90,"--energy-seconds",0 if args.engineering_smoke else 30],
                f"isolated-{variant}-{slot}.log",slot=slot)
            value=json.loads((folder/"profile.json").read_text())
            values=[p["step_train_wall_seconds"] for p in value["unprofiled_samples"]]
            chunks=[statistics.median(values[i:i+30]) for i in range(0,len(values),30)]
            row={"variant":variant,"gpu_slot":slot,"seed":SEEDS[0],"median_step_seconds":statistics.median(chunks),
                "repetition_medians":chunks,"energy":value["energy"],"host_memory":value["host_memory"],
                "scope":"isolated weights-only profiling clone, fresh optimizer; no concurrent workers"}
            benchmark=folder/"inference.json"
            command(campaign,args,["-m","llm2","benchmark","--checkpoint",job["checkpoint"],"--device",args.device,
                "--prompt","The history of","--warmup-tokens","8","--max-new-tokens","64","--output",benchmark],
                f"inference-{variant}-{slot}.log",slot=slot)
            row["inference"]=json.loads(benchmark.read_text())
            row["inference"]["kv_cache_bytes_calculated"]=0
            lengths=[job["config_resolved"]["data"]["seq_len"]]
            factors=job["config_resolved"]["research"]["compressed_factors"] or []
            lengths.extend((lengths[0]+factor-1)//factor for factor in factors)
            row["training_context_projection_cache_bytes_calculated"]=(
                job["batch_size"]*sum(lengths)*job["config_resolved"]["model"]["d_model"]*4
                if job["config_resolved"]["research"]["cache_context_projection"] else 0)
            row["cache_bytes_scope"]="temporary FP32 context projections per forward; no persistent sequence memory/KV cache"
            profiles.append(row);campaign.manifest["isolated_profiles"]=profiles;campaign.save()
    for job in campaign.manifest["runs"]:
        destination=campaign.output/"post"/job["id"];destination.mkdir(parents=True)
        command(campaign,args,["-m","llm2","diagnose","--checkpoint",job["checkpoint"],"--device",args.device,
            "--samples","128","--batch-size","4","--output",destination/"diagnostics.json"],
            job["id"]+"-diagnose.log",slot=args.gpus[0])
        if job["variant"] in ("flrm_full","lrm_cached"):
            command(campaign,args,["-m","llm2.research.cycle4_evaluate","--checkpoint",job["checkpoint"],
                "--device",args.device,"--output",destination/"depth"],job["id"]+"-depth.log",slot=args.gpus[0])


def execute(args,*,started=None):
    validate(args)
    started=time.monotonic() if started is None else started
    budget=SessionBudget(args.session_seconds,args.export_reserve_seconds,
        elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-started)
    campaign=SessionCampaign(ROOT,Path(args.output).resolve(),device=args.device,gpus=args.gpus,budget=budget)
    campaign.manifest.update(cycle=4,protocol=PROTOCOL,protocol_id=PROTOCOL_ID,
        engineering_smoke=args.engineering_smoke,tests_skipped=args.skip_tests,test_evaluated=False,
        target_seconds=args.target_seconds,post_reserve_seconds=0 if args.engineering_smoke else 900,
        scheduling="two independent single-GPU lanes; profiles isolated; parallel train time labelled operational",
        isolated_profiles=[])
    error=None
    old_flags={key:os.environ.get(key) for key in ("LLM2_CYCLE4","LLM2_COMPILER_CACHE_ROOT")}
    os.environ["LLM2_CYCLE4"]="1"
    os.environ["LLM2_COMPILER_CACHE_ROOT"]=str(campaign.output/"compiler-cache")
    try:
        with heartbeat(campaign,"ciclo 4"):
            with phase(campaign,"dataset_and_configurations"):
                data=prepare(campaign,args);materialize(campaign,args,data)
            if not args.skip_tests:tests(campaign,args)
            with phase(campaign,"pilot"):pilot(campaign,args)
            admit(campaign,args)
            train_parallel(campaign,args)
            with phase(campaign,"isolated_profiles_and_diagnostics"):postprocess(campaign,args)
            campaign.manifest["status"]="complete"
    except SessionBudgetExceeded as exc:
        campaign.manifest.update(status="partial_budget",error=str(exc))
    except Exception as exc:
        exhausted=budget.work_remaining<=0
        error=None if exhausted else exc
        campaign.manifest.update(status="partial_budget" if exhausted else "failed",
            error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort()
        campaign.manifest["session_budget"]=budget.snapshot()
        campaign.save()
        try:results,backup=publish(campaign,"final",final=True)
        finally:
            for key,value in old_flags.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
        print(f"Status: {campaign.manifest['status']} | {results.name} | {backup.name}",flush=True)
    if error:raise RuntimeError(f"Cycle 4 failed; evidence preserved in {results}") from error
    return {"status":campaign.manifest["status"],"results_zip":str(results),"checkpoints_zip":str(backup)}


def main(argv=None,*,started=None):
    args=parser().parse_args(argv);validate(args)
    if not args.execute:
        return {"execution":"not_started","protocol":PROTOCOL,"protocol_id":PROTOCOL_ID,
            "target_seconds":args.target_seconds,"hard_seconds":args.session_seconds}
    if not args.output:raise ValueError("fresh output directory required")
    return execute(args,started=started)

if __name__=="__main__":main()
