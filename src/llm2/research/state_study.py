"""Bounded Stage A qualification / Stage B scientific experiment, in separate sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
from types import SimpleNamespace
import torch
import yaml
from ..data import PreparedData,write_json
from ..runtime import environment,source_provenance
from .session import SessionBudget,SessionBudgetExceeded,SessionCampaign
from .cycle3_short import heartbeat
from .cycle4 import archive_files,checkpoint_files
from .cycle3 import prepare
from .state_protocol import ROOT,PROTOCOL,PROTOCOL_ID,RULES,VARIANTS,SEEDS,configuration,ordered_jobs
from .state_platforms import select,validate_report
from .state_report import analyze
from .document_windows import tuning_dataset
from .campaign_storage import prune_archives,verify_pair


def materialize(cfg,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=True),encoding="utf-8",newline="\n")
    return str(path)


def qualify(campaign,args):
    rows=[]
    for slot in (args.gpus if not args.engineering_smoke else ["0"]):
        for rule in RULES:
            destination=campaign.output/"qualification"/f"{rule}-{slot}"
            call=["-m","llm2.research.state_platforms","--execute","--rule",rule,"--slot","0","--output",str(destination)]
            if args.engineering_smoke:call.append("--engineering-smoke")
            campaign.command(call,f"qualify-{rule}-{slot}.log",gpu=slot if not args.engineering_smoke else None)
            record=json.loads((destination/"platforms.json").read_text(encoding="utf-8"))["rows"][0]
            record["slot"]=str(slot);rows.append(record)
            campaign.manifest["qualification_rows"]=rows;campaign.save()
    result={"protocol_id":PROTOCOL_ID,"implementation_scope":"common_shell_memory_rules",
        "engineering_smoke":args.engineering_smoke,"source_sha256":source_provenance()["source_sha256"],
        "environment":environment(),"rows":rows,"selected_rule":select(rows,engineering_smoke=args.engineering_smoke)}
    write_json(campaign.output/"platforms.json",result)
    campaign.manifest["selected_rule"]=result["selected_rule"]
    print(f"Plataforma selecionada: {result['selected_rule']}; implementação de regra no shell comum.",flush=True)
    if not args.engineering_smoke and result["selected_rule"] is None:
        raise RuntimeError("no candidate qualified on both T4s; preserve failures, do not select by CPU")


def worker(campaign,args,cfg,name,mode,slot,*,output=None,weights=None,job=None,energy_seconds=0.):
    path=materialize(cfg,campaign.output/"configs"/(name+".yaml"))
    command=["-m","llm2.research.state_worker","--config",path,"--mode",mode]
    if output is not None:command.extend(["--output",str(output)])
    if weights:command.extend(["--weights",str(weights)])
    if energy_seconds:command.extend(["--energy-seconds",str(energy_seconds)])
    if args.engineering_smoke:command.append("--engineering-smoke")
    campaign.command(command,name+".log",gpu=slot if not args.engineering_smoke else None,job=job)
    return json.loads((Path(output or cfg.train.output_dir)/"worker.json").read_text(encoding="utf-8"))


def schedule_batches(jobs,slots,*,remaining_seconds,post_seconds=1200.):
    """Admit whole seed blocks, preserving every variant/horizon and rotation."""
    admitted=[];pending=[];loads={str(s):0. for s in slots}
    for seed in SEEDS:
        block=[j for j in jobs if j["seed"]==seed]
        if not block:continue
        proposed=loads.copy()
        for job in block:proposed[job["gpu_slot"]]+=job["estimated_job_seconds"]
        if max(proposed.values())+post_seconds<=remaining_seconds:
            admitted.extend(block);loads=proposed
        else:pending.extend(block)
    return admitted,pending


def experiment(campaign,args):
    if args.engineering_smoke:rule="gated_delta"
    else:
        if not args.platform_report:raise ValueError("Stage B requires Stage A platforms.json")
        value=json.loads(Path(args.platform_report).read_text(encoding="utf-8"));rule=validate_report(value)
        write_json(campaign.output/"platforms.json",value)
    campaign.manifest["selected_rule"]=rule
    data=PreparedData(args.data) if args.engineering_smoke and args.data else None
    if data is None:data=PreparedData(prepare(campaign,args))
    else:
        campaign.manifest.update(dataset_sha256=data.fingerprint,dataset_manifest=data.manifest,dataset_path=str(data.root))
    campaign.save()
    tune_path=tuning_dataset(data.root,campaign.output/"tuning-data")
    # All variants are benchmarked separately on both physical GPUs before concurrent jobs.
    profiles={};pilot_configs={}
    for slot in args.gpus:
        for variant in VARIANTS:
            cfg=configuration(variant,rule,data.root,campaign.output/"pilots"/f"{variant}-{slot}",241,
                smoke=args.engineering_smoke,device=args.device)
            value=worker(campaign,args,cfg,f"pilot-{variant}-{slot}","pilot",slot)
            profiles[variant,slot]=value;pilot_configs[variant,slot]=cfg
    # A short two-lane pilot measures shared-host contention, not a speedup claim.
    def concurrent(slot):
        cfg=pilot_configs["recurrent",slot]
        return slot,worker(campaign,args,cfg,f"concurrent-{slot}","pilot",slot,
                           output=campaign.output/"pilots"/f"concurrent-{slot}")
    with ThreadPoolExecutor(max_workers=2) as pool:parallel=dict(pool.map(concurrent,args.gpus))
    contention=max(1.,max(parallel[s]["median_step_seconds"]/profiles["recurrent",s]["median_step_seconds"] for s in args.gpus))
    jobs=ordered_jobs(rule,data.root,campaign.output,args.gpus,device=args.device,smoke=args.engineering_smoke,seeds=args.seeds)
    for job in jobs:
        profile=profiles[job["variant"],job["gpu_slot"]]
        job["estimated_job_seconds"]=(1.35*contention*profile["median_step_seconds"]*job["max_steps"]
            +profile["warmup_seconds"]+2.6*profile["full_validation_seconds"]+(0 if args.engineering_smoke else 120))
    tuning_seconds=0 if args.engineering_smoke else sum(1.35*max(profiles[v,s]["median_step_seconds"] for s in args.gpus)*750*2
        +2*max(profiles[v,s]["warmup_seconds"]+profiles[v,s]["full_validation_seconds"] for s in args.gpus) for v in VARIANTS)
    if args.tuning_report:tuning_seconds=0
    admitted,pending=schedule_batches(jobs,args.gpus,remaining_seconds=campaign.budget.work_remaining-tuning_seconds,
                                     post_seconds=0 if args.engineering_smoke else 1200.)
    campaign.manifest.update(runs=jobs,admitted_job_ids=[j["id"] for j in admitted],pending_job_ids=[j["id"] for j in pending],
        pilot_profiles=[{"variant":v,"slot":s,**p} for (v,s),p in profiles.items()],contention_factor=contention,
        planned_tuning_seconds=tuning_seconds,forecast_margin=1.35)
    print(f"Admitidos {len(admitted)}/{len(jobs)} treinos, em blocos completos de seed; pendentes {len(pending)}.",flush=True)
    campaign.save()
    if not admitted:raise SessionBudgetExceeded("not even one whole seed fits; horizons unchanged")
    needed=sum(j["parameters"]*16*4 for j in admitted)+(0 if args.engineering_smoke else 2*1024**3)
    if shutil.disk_usage(campaign.output).free<needed:raise OSError("insufficient space including 2 GiB reserve")
    # Symmetric tuning, same number of attempts, original training documents only.
    rates={};tuning=[]
    if args.tuning_report:
        frozen=json.loads(Path(args.tuning_report).read_text(encoding="utf-8"))
        if (frozen.get("protocol_id")!=PROTOCOL_ID or frozen.get("selected_rule")!=rule
            or frozen.get("dataset_sha256")!=data.fingerprint
            or frozen.get("provenance",{}).get("source_sha256")!=campaign.manifest["provenance"]["source_sha256"]):
            raise ValueError("frozen tuning source/protocol/dataset/rule mismatch")
        rates=frozen.get("learning_rates",{});tuning=frozen.get("tuning",[])
        if set(rates)!=set(VARIANTS) or any(r not in (.0002,.0004) for r in rates.values()):
            raise ValueError("invalid frozen learning rates")
    else:
        for variant in VARIANTS:
            candidates=[]
            for rate in (.0002,.0004):
                cfg=configuration(variant,rule,tune_path,campaign.output/"tuning"/f"{variant}-{rate}",901,lr=rate,
                    smoke=args.engineering_smoke,device=args.device,steps=4 if args.engineering_smoke else 750)
                value=worker(campaign,args,cfg,f"tune-{variant}-{rate}","tune",args.gpus[0])
                nll=value["evaluations"][-1]["validation_loss"]
                candidates.append((nll,rate));tuning.append({"variant":variant,"lr":rate,"validation_loss":nll})
            rates[variant]=min(candidates)[1]
    campaign.manifest.update(learning_rates=rates,tuning=tuning);campaign.save()
    publishing=threading.Lock();completed=[0]
    def lane(slot):
        for job in [j for j in admitted if j["gpu_slot"]==slot]:
            if campaign.cancelled.is_set():return
            campaign.budget.require(job["estimated_job_seconds"]+(0 if args.engineering_smoke else 600))
            cfg=configuration(job["variant"],rule,data.root,job["directory"],job["seed"],lr=rates[job["variant"]],
                              smoke=args.engineering_smoke,device=args.device)
            job.update(status="training",config_resolved=cfg.to_dict());campaign.save()
            value=worker(campaign,args,cfg,job["id"],"train",slot,job=job)
            if value["source_sha256"]!=campaign.manifest["provenance"]["source_sha256"] or value["dataset_sha256"]!=data.fingerprint:
                raise ValueError("worker source/dataset mismatch")
            if value["stop_reason"]!="max_steps" or value["step"]!=job["max_steps"]:
                raise SessionBudgetExceeded("fixed horizon not completed")
            job.update(status="complete",checkpoint=value["checkpoint"],checkpoint_model_sha256=value["checkpoint_model_sha256"],
                evaluations=value["evaluations"],counters=value["counters"],flops_estimate=value["flops_estimate"],
                worker_environment=value["environment"])
            campaign.save()
            with publishing:
                completed[0]+=1
                if completed[0]==(len(admitted)+1)//2:
                    pair=publish(campaign,args,suffix="midway")
                    campaign.manifest["midway_archives"]=[str(p) for p in pair];campaign.save()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,slot) for slot in args.gpus]
        try:
            for future in as_completed(futures):future.result()
        except BaseException:
            campaign.abort();raise
    isolated=[]
    for variant in VARIANTS:
        job=next(j for j in admitted if j["variant"]==variant)
        for slot in args.gpus:
            cfg=configuration(variant,rule,data.root,job["directory"],job["seed"],lr=rates[variant],
                smoke=args.engineering_smoke,device=args.device)
            value=worker(campaign,args,cfg,f"post-{variant}-{slot}","pilot",slot,
                output=campaign.output/"post"/f"{variant}-{slot}",weights=job["checkpoint"],
                energy_seconds=0 if args.engineering_smoke else 30.)
            isolated.append({"variant":variant,"slot":slot,**value})
    campaign.manifest["isolated_profiles"]=isolated
    if pending:campaign.manifest["status"]="partial_planned_batches"


def publish(campaign,args,*,suffix=None):
    with campaign.lock:
        campaign.save()
        return _publish_locked(campaign,args,suffix=suffix)


def _publish_locked(campaign,args,*,suffix=None):
    rows=[]
    for job in campaign.manifest["runs"]:
        if job["status"]!="complete":continue
        final=job["evaluations"][-1]
        rows.append({"variant":job["variant"],"seed":job["seed"],"validation_loss":final["validation_loss"],
            "bits_per_byte":final["validation_bits_per_byte"],"validation_targets":final["validation_tokens"],
            "parameters":job["parameters"],"state_bytes_per_sequence":job["state_bytes_per_sequence"],
            "flops_estimate":job.get("flops_estimate"),"train_wall_seconds":job["counters"]["train_wall_seconds"],
            "timing_scope":"concurrent operational training","gpu_slot":job["gpu_slot"]})
    report=analyze(rows,campaign.manifest.get("isolated_profiles",[]),smoke=args.engineering_smoke)
    curves=[{"variant":j["variant"],"seed":j["seed"],**point} for j in campaign.manifest["runs"]
            if j["status"]=="complete" for point in j["evaluations"]]
    report["curves"]=curves
    report["identity"]={"selected_rule":campaign.manifest.get("selected_rule"),
        "source_sha256":campaign.manifest["provenance"]["source_sha256"],
        "dataset_sha256":campaign.manifest.get("dataset_sha256"),"learning_rates":campaign.manifest.get("learning_rates"),
        "environment":environment()}
    folder=campaign.output/"report";folder.mkdir(exist_ok=True)
    write_json(folder/"report.json",report)
    if curves:
        with (folder/"curves.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=curves[0]);writer.writeheader();writer.writerows(curves)
    if curves and not suffix:
        from .state_report import plot_curves
        plot_curves(curves,folder/"quality_by_cost.png",engineering_smoke=args.engineering_smoke)
    if rows:
        with (folder/"runs.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    files=[]
    for directory in ("configs","logs","qualification","pilots","post","tuning","tuning-data","report"):
        if suffix and directory=="logs":continue
        files.extend(p for p in (campaign.output/directory).rglob("*") if p.is_file() and p.suffix in (".json",".jsonl",".yaml",".txt",".log",".zip",".png",".csv"))
    included=[j for j in campaign.manifest["runs"] if not suffix or j["status"]=="complete"]
    for job in included:
        files.extend(p for p in Path(job["directory"]).rglob("*") if p.is_file() and p.suffix in (".json",".jsonl",".yaml",".txt",".log",".zip"))
    files.extend(p for p in campaign.output.glob("*") if p.is_file() and p.suffix in (".json",".csv"))
    stem=campaign.output.name+("-"+suffix if suffix else "")
    result=archive_files(campaign.output.parent/(stem+"-results.zip"),files,base=campaign.output)
    backup=None
    if campaign.manifest["runs"]:
        backup=archive_files(campaign.output.parent/(stem+"-checkpoints.zip"),
            checkpoint_files(campaign,included),base=campaign.output,stored=True)
        verify_pair(result,backup)
    return result,backup


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("qualify","experiment"),default="qualify")
    p.add_argument("--output",required=True);p.add_argument("--execute",action="store_true")
    p.add_argument("--platform-report");p.add_argument("--tuning-report");p.add_argument("--data");p.add_argument("--cache");p.add_argument("--download",action="store_true")
    p.add_argument("--engineering-smoke",action="store_true");p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--gpus",nargs=2,default=["0","1"]);p.add_argument("--seeds",nargs="+",type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument("--session-seconds",type=float,default=10800.);p.add_argument("--target-seconds",type=float,default=7200.)
    p.add_argument("--bootstrap-elapsed-seconds",type=float,default=0.);p.add_argument("--bootstrap-metadata")
    a=p.parse_args(argv)
    if (a.device=="cpu")!=a.engineering_smoke or len(set(a.gpus))!=2 or not all(s.isdecimal() for s in a.gpus):
        raise ValueError("CPU requires engineering smoke; two distinct GPU slots required")
    if not 300<a.session_seconds<=10800 or not 0<a.target_seconds<=a.session_seconds:raise ValueError("invalid session bounds")
    if not a.engineering_smoke and (a.session_seconds!=10800 or a.target_seconds!=7200):raise ValueError("science requires 2h target/3h cap")
    if a.phase=="experiment" and not(a.data or a.cache):raise ValueError("experiment requires data or pinned cache")
    if not a.execute:return {"status":"not_started","protocol":PROTOCOL,"protocol_id":PROTOCOL_ID}
    if not a.engineering_smoke and (not torch.cuda.is_available() or len(set(a.gpus))!=2 or any(
        int(s)>=torch.cuda.device_count() or "T4" not in torch.cuda.get_device_name(int(s)) for s in a.gpus)):
        raise RuntimeError("science requires two T4 GPUs")
    budget=SessionBudget(a.session_seconds,300.,elapsed=a.bootstrap_elapsed_seconds)
    campaign=SessionCampaign(ROOT,Path(a.output).resolve(),device=a.device,gpus=a.gpus,budget=budget)
    campaign.manifest.update(cycle="state_v1",protocol=PROTOCOL,protocol_id=PROTOCOL_ID,phase=a.phase,engineering_smoke=a.engineering_smoke,
        implementation_scope="common_shell_memory_rules",test_evaluated=False,target_seconds=a.target_seconds)
    previous=os.environ.get("LLM2_CYCLE4");os.environ["LLM2_CYCLE4"]="1"
    old_cache=os.environ.get("LLM2_COMPILER_CACHE_ROOT");os.environ["LLM2_COMPILER_CACHE_ROOT"]=str(campaign.output/"compiler-cache")
    error=None
    try:
        with heartbeat(campaign,"recorrência/fractal"):
            campaign.command(["-m","pytest","-q","tests/test_state_recurrent.py","tests/test_state_study.py","-m","not cuda"],"tests-state-cpu.log")
            if a.device=="cuda" and a.phase=="experiment":
                if not a.platform_report:raise ValueError("Stage B requires platforms.json before CUDA tests")
                selected=validate_report(json.loads(Path(a.platform_report).read_text(encoding="utf-8")))
                campaign.command(["-m","pytest","-q","tests/test_state_recurrent.py","-m","cuda","-k",selected],
                                 "tests-state-cuda.log",gpu=a.gpus[0])
            # Stage A records compile failures per candidate; one incompatible rule
            # must not prevent qualifying another. Each worker compares compiled
            # outputs/gradients against the independent sequential oracle.
            if a.phase=="qualify":qualify(campaign,a)
            else:experiment(campaign,a)
            if campaign.manifest["status"]!="partial_planned_batches":campaign.manifest["status"]="complete"
    except SessionBudgetExceeded as exc:campaign.manifest.update(status="partial_budget",error=str(exc))
    except Exception as exc:
        campaign.manifest.update(status="failed",error=str(exc),traceback=traceback.format_exc());error=exc
    finally:
        campaign.abort();campaign.manifest["session_budget"]=budget.snapshot();campaign.save()
        try:
            result,backup=publish(campaign,a)
            mid=[Path(p) for p in campaign.manifest.get("midway_archives",[]) if p and Path(p).exists()]
            if mid and backup:
                try:prune_archives(campaign.output.parent,mid,final_results=result,final_checkpoints=backup,
                    guard=lambda:list(campaign.processes))
                except Exception as cleanup_error:print(f"Definitivos preservados; temporários mantidos: {cleanup_error}",flush=True)
        finally:
            if previous is None:os.environ.pop("LLM2_CYCLE4",None)
            else:os.environ["LLM2_CYCLE4"]=previous
            if old_cache is None:os.environ.pop("LLM2_COMPILER_CACHE_ROOT",None)
            else:os.environ["LLM2_COMPILER_CACHE_ROOT"]=old_cache
        print(f"Status: {campaign.manifest['status']}; {campaign.output}",flush=True)
    if error:raise RuntimeError("study failed; evidence exported") from error
    return campaign.manifest


if __name__=="__main__":main()
