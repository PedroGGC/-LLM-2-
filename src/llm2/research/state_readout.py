"""Two-hour five-arm screen; warm/cold pilots, whole jobs, bounded disk use."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace

STARTED=time.monotonic()
import torch
import yaml

from ..checkpoint import read_checkpoint_metadata
from ..data import write_json,sha256_file
from ..runtime import environment,save_source_snapshot
from .cycle3 import prepare
from .cycle3_short import heartbeat
from .cycle4 import archive_files,checkpoint_files
from .campaign_storage import verify_pair
from .session import SessionCampaign,SessionBudget,SessionBudgetExceeded
from .state_b_operator import SlotCaches,family_for
from .state_study import storage_budget,discard_compiler_cache,materialize
from .state_readout_protocol import ROOT,PROTOCOL,PROTOCOL_ID,VARIANTS,PAIRS,configuration,jobs


class ReadoutCampaign(SessionCampaign):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from .kaggle_light import remove_cache,tree_bytes
        self.caches=SlotCaches(self,remove_cache,tree_bytes)

    def command(self,arguments,log_name,*,gpu=None,job=None,timeout=None):
        call=lambda:super(ReadoutCampaign,self).command(arguments,log_name,gpu=gpu,job=job,timeout=timeout)
        return call() if gpu is None else self.caches.run(gpu,family_for(SimpleNamespace(yaml=yaml),arguments),call)


def worker(campaign,args,cfg,name,mode,slot,*,output=None,weights=None,job=None):
    path=materialize(cfg,campaign.output/"configs"/(name+".yaml"))
    cmd=["-m","llm2.research.state_readout_worker","--config",path,"--mode",mode]
    if output:cmd.extend(["--output",str(output)])
    if weights:cmd.extend(["--weights",str(weights),"--energy-seconds","0" if args.engineering_smoke else "30"])
    if args.engineering_smoke:cmd.append("--engineering-smoke")
    campaign.command(cmd,name+".log",gpu=None if args.engineering_smoke else slot,job=job)
    return json.loads((Path(output or cfg.train.output_dir)/"worker.json").read_text(encoding="utf-8"))


def comparable_config(value):
    value=json.loads(json.dumps(value));value["data"].pop("path");value["train"].pop("output_dir")
    return value


def native_history(campaign,data,*,smoke=False):
    """Completed native runs only; never scan arbitrary old ZIPs or resume pickle."""
    rows={};current=environment()
    for path in sorted(campaign.output.parent.glob("llm2-campaign-*/campaign.json")):
        if path.parent==campaign.output or path.is_symlink():continue
        try:value=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,ValueError):
            print(f"Histórico opcional ilegível preservado:{path}",flush=True);continue
        if value.get("cycle")!="state_readout_v1" or value.get("protocol_id")!=PROTOCOL_ID:continue
        if value.get("engineering_smoke")!=smoke:continue
        if (value.get("dataset_sha256")!=data.fingerprint or
            value["provenance"]["source_sha256"]!=campaign.manifest["provenance"]["source_sha256"]):
            raise ValueError("readout history source/dataset differs; do not mix campaigns")
        if any(value.get("environment",{}).get(k)!=current.get(k) for k in ("python","packages","cuda_version","cudnn_version")):
            raise ValueError("readout history environment differs")
        for job in value.get("runs",[]):
            if job["status"]!="complete":continue
            expected=configuration(job["variant"],data.root,"unused",smoke=smoke,device="cpu" if smoke else "cuda")
            if comparable_config(job["config_resolved"])!=comparable_config(expected.to_dict()):
                raise ValueError("readout history configuration differs")
            if job["counters"]["step"]!=job["max_steps"] or not job["evaluations"][-1]["full"]:
                raise ValueError("readout history horizon incomplete")
            if job["variant"] in rows and rows[job["variant"]]["checkpoint_model_sha256"]!=job["checkpoint_model_sha256"]:
                raise ValueError("conflicting repeated seed251 results")
            rows[job["variant"]]={**job,"historical_campaign":str(path.parent),"history_manifest_sha256":sha256_file(path)}
    return list(rows.values())


def estimates(job,cold,warm,contention,*,cache_hot,smoke=False):
    if smoke:return 0.,0.
    # Timings are measured cold/warm; imports are included via command wall time.
    train_start=warm["command_wall_seconds"] if cache_hot else cold["command_wall_seconds"]
    train=1.35*(job["max_steps"]*cold["median_step_seconds"]*contention+train_start+
                2.6*cold["full_validation_seconds"]+60.)
    post=1.35*(warm["command_wall_seconds"]+30.+60.)
    return train,post


def admission(pairs,by_variant,profiles,hot,remaining,contention,*,smoke=False):
    selected=[];forecast=0.;active=dict(hot)
    for pair in pairs:
        group=[by_variant[v] for v in pair if v in by_variant]
        if not group:continue
        costs=[estimates(j,*profiles[j["variant"]],contention,
               cache_hot=active.get(j["gpu_slot"])==j["variant"],smoke=smoke) for j in group]
        duration=max(t for t,p in costs)+sum(p for t,p in costs)
        if forecast+duration>remaining:break
        selected.append(group);forecast+=duration
        for job,(train,post) in zip(group,costs):
            job.update(estimated_job_seconds=train,estimated_post_seconds=post)
            active[job["gpu_slot"]]=job["variant"]
    return selected,forecast


def report(campaign):
    evidence=campaign.manifest.get("historical_runs",[])+[j for j in campaign.manifest["runs"] if j["status"]=="complete"]
    rows=[];curves=[]
    for job in evidence:
        final=job["evaluations"][-1]
        rows.append({"variant":job["variant"],"seed":251,"validation_loss":final["validation_loss"],
            "bits_per_byte":final["validation_bits_per_byte"],"validation_targets":final["validation_tokens"],
            "parameters":job["parameters"],"state_bytes_per_sequence":job["state_bytes_per_sequence"],
            "train_wall_seconds":job["counters"]["train_wall_seconds"],
            "campaign":job.get("historical_campaign",str(campaign.output))})
        curves.extend({"variant":job["variant"],"seed":251,**point} for point in job["evaluations"])
    lookup={r["variant"]:r for r in rows};contrasts={}
    for a,b in (("hierarchy_immediate","hierarchy_next"),("fine_only","recurrent")):
        contrasts[a+"-minus-"+b]=lookup[a]["validation_loss"]-lookup[b]["validation_loss"] if a in lookup and b in lookup else None
    result={"protocol_id":PROTOCOL_ID,"rows":rows,"contrasts":contrasts,"curves":curves,
        "isolated_profiles":campaign.manifest.get("post_profiles",[]),"test_evaluated":False,
        "decision":"exploratory_one_seed" if len(rows)==5 else "incomplete_matrix",
        "limitations":["fixed LR recipe; new controls not individually tuned","equal targets, not iso-compute",
            "isolated profiles on one assigned GPU per arm; no two-GPU efficiency confirmation",
            "historical campaign timings remain separate; no generation benchmark","one seed, no confidence interval"]}
    folder=campaign.output/"report";folder.mkdir(exist_ok=True)
    write_json(folder/"report.json",result)
    for name,values in (("runs.csv",rows),("curves.csv",curves)):
        if values:
            keys=sorted({k for row in values for k in row})
            with (folder/name).open("w",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(values)
    for row in rows:print(f"{row['variant']}: NLL{row['validation_loss']:.7f}",flush=True)
    return result


def publish(campaign):
    campaign.save();report(campaign)
    files=[]
    for path in campaign.output.rglob("*"):
        if not path.is_file() or "compiler-cache" in path.parts:continue
        if path.suffix not in (".json",".jsonl",".yaml",".txt",".log",".zip",".csv"):continue
        # One canonical source snapshot; per-worker snapshots remain on disk only.
        if path.name=="source.zip" and path.parent!=campaign.output:continue
        files.append(path)
    results=archive_files(campaign.output.parent/(campaign.output.name+"-results.zip"),files,base=campaign.output)
    backup=archive_files(campaign.output.parent/(campaign.output.name+"-checkpoints.zip"),
        checkpoint_files(campaign,campaign.manifest["runs"]),base=campaign.output,stored=True)
    verify_pair(results,backup)
    print(f"Resultados: {results.name}; checkpoints finais: {backup.name}",flush=True)
    return results,backup


def execute(args):
    budget=SessionBudget(args.session_seconds,300.,elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-args.started)
    campaign=ReadoutCampaign(ROOT,Path(args.output).resolve(),device=args.device,gpus=args.gpus,budget=budget)
    campaign.manifest.update(cycle="state_readout_v1",protocol=PROTOCOL,protocol_id=PROTOCOL_ID,
        environment=environment(),engineering_smoke=args.engineering_smoke,test_evaluated=False,post_profiles=[])
    previous={name:os.environ.get(name) for name in ("LLM2_CYCLE4","LLM2_COMPILER_CACHE_ROOT")}
    os.environ.update(LLM2_CYCLE4="1",LLM2_COMPILER_CACHE_ROOT=str(campaign.output/"compiler-cache"))
    error=None
    try:
        with storage_budget(campaign,smoke=args.engineering_smoke),heartbeat(campaign,"controle fino/leitura"):
            campaign.command(["-m","pytest","-q","tests/test_state_readout_models.py","-m","not cuda"],"tests-readout-cpu.log") if not args.engineering_smoke else None
            if args.device=="cuda":campaign.command(["-m","pytest","-q","tests/test_state_readout_models.py","-m","cuda"],"tests-readout-cuda.log",gpu=args.gpus[0])
            data_path=prepare(campaign,args)
            from ..data import PreparedData
            data=PreparedData(data_path)
            save_source_snapshot(campaign.output/"source.zip",campaign.manifest["provenance"])
            history=native_history(campaign,data,smoke=args.engineering_smoke)
            campaign.manifest["historical_runs"]=history
            missing=[v for v in args.variants if v not in {j["variant"] for j in history}]
            current=jobs(data_path,campaign.output,args.gpus,smoke=args.engineering_smoke,device=args.device,variants=missing) if missing else []
            campaign.manifest["runs"]=current;campaign.save()
            needed=sum(j["parameters"]*48 for j in current)+(0 if args.engineering_smoke else 4*1024**3)
            campaign.manifest["storage_estimate"]={"required_free_bytes":needed,
                "scope":"checkpoint transient + final ZIP allowance48bytes/parameter;2GiB cache plus2GiB reserve"}
            if shutil.disk_usage(campaign.output).free<needed:raise OSError("insufficient space for checkpoints/cache plus reserve")
            by_variant={j["variant"]:j for j in current};profiles={};hot={}
            # Last pilots are the primary hierarchy pair, retaining their warm caches.
            order=[v for v in ("transformer_two","recurrent","fine_only","hierarchy_next","hierarchy_immediate") if v in by_variant]
            for variant in order:
                job=by_variant[variant];slot=job["gpu_slot"];values=[]
                for label in ("cold","warm"):
                    name=f"pilot-{variant}-{label}";cfg=configuration(variant,data_path,campaign.output/"pilots"/name,
                        smoke=args.engineering_smoke,device=args.device)
                    print(f"Piloto isolado {variant},GPU{slot},{label}.",flush=True)
                    began=time.perf_counter();p=worker(campaign,args,cfg,name,"pilot",slot)
                    p["command_wall_seconds"]=time.perf_counter()-began;values.append(p)
                profiles[variant]=values;hot[slot]=variant
                campaign.manifest["pilot_profiles"]={k:v for k,v in profiles.items()};campaign.save()
            contention=1.
            first=next(([by_variant[v] for v in pair if v in by_variant] for pair in PAIRS if len([v for v in pair if v in by_variant])==2),[])
            if first:
                def concurrent(job):
                    variant=job["variant"];name="concurrent-"+variant
                    cfg=configuration(variant,data_path,campaign.output/"pilots"/name,smoke=args.engineering_smoke,device=args.device)
                    return variant,worker(campaign,args,cfg,name,"pilot",job["gpu_slot"])
                with ThreadPoolExecutor(max_workers=2) as pool:parallel=dict(pool.map(concurrent,first))
                if not args.engineering_smoke:
                    contention=max(1.,max(p["median_step_seconds"]/profiles[v][1]["median_step_seconds"] for v,p in parallel.items()))
                for job in first:hot[job["gpu_slot"]]=job["variant"]
                campaign.manifest["concurrent_pilot"]=parallel
            admitted,forecast=admission(PAIRS,by_variant,profiles,hot,budget.work_remaining,contention,smoke=args.engineering_smoke)
            campaign.manifest.update(contention_factor=contention,forecast_margin=1.35,
                forecast_total_seconds=budget.elapsed+forecast+300,admitted_ids=[j["id"] for p in admitted for j in p])
            print(f"Previsão conservadora total: {(budget.elapsed+forecast+300)/60:.1f}min; limite120min; "
                  f"admitidos{sum(map(len,admitted))}/{len(current)} jobs inteiros.",flush=True)
            campaign.save()
            if current and not admitted:raise SessionBudgetExceeded("no complete comparison pair fits; pilots exported")
            def train(job):
                cfg=configuration(job["variant"],data_path,job["directory"],smoke=args.engineering_smoke,device=args.device)
                job["status"]="training";campaign.save()
                result=worker(campaign,args,cfg,job["id"],"train",job["gpu_slot"],job=job)
                if (result["step"]!=job["max_steps"] or result["stop_reason"]!="max_steps" or
                    result["source_sha256"]!=campaign.manifest["provenance"]["source_sha256"] or result["dataset_sha256"]!=data.fingerprint):
                    raise ValueError("training horizon/source/dataset differs")
                job.update(result,status="complete");campaign.save()
                return job
            for group in admitted:
                budget.require(max(j["estimated_job_seconds"] for j in group)+sum(j["estimated_post_seconds"] for j in group))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(train,j) for j in group]
                    try:
                        for f in as_completed(futures):f.result()
                    except BaseException:campaign.abort();raise
                paired=[j for j in history+[j for j in current if j["status"]=="complete"] if j["variant"].startswith("hierarchy_")]
                if len(paired)==2 and paired[0]["initial_parameters_sha256"]!=paired[1]["initial_parameters_sha256"]:
                    raise ValueError("hierarchy initial weights were not paired")
                # Both training workers stopped; all post measurements are isolated.
                for job in group:
                    cfg=configuration(job["variant"],data_path,job["directory"],smoke=args.engineering_smoke,device=args.device)
                    name="post-"+job["variant"]
                    p=worker(campaign,args,cfg,name,"pilot",job["gpu_slot"],output=campaign.output/"post"/name,weights=job["checkpoint"])
                    campaign.manifest["post_profiles"].append({"variant":job["variant"],"slot":job["gpu_slot"],**p});campaign.save()
            evidence=history+[j for j in current if j["status"]=="complete"]
            hierarchy=[j for j in evidence if j["variant"] in ("hierarchy_next","hierarchy_immediate")]
            if len(hierarchy)==2 and hierarchy[0]["initial_parameters_sha256"]!=hierarchy[1]["initial_parameters_sha256"]:
                raise ValueError("hierarchy initial weights were not paired")
            campaign.manifest["status"]="complete" if set(VARIANTS)<={j["variant"] for j in evidence} else "partial_planned"
    except SessionBudgetExceeded as exc:campaign.manifest.update(status="partial_budget",error=str(exc))
    except Exception as exc:
        error=exc;campaign.manifest.update(status="failed",error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort()
        try:discard_compiler_cache(campaign)
        except Exception as exc:campaign.manifest["cache_cleanup_error"]=str(exc)
        campaign.manifest["pending_variants"]=[v for v in VARIANTS if v not in {j["variant"] for j in campaign.manifest.get("historical_runs",[])+[j for j in campaign.manifest["runs"] if j["status"]=="complete"]}]
        campaign.manifest["session_budget"]=budget.snapshot();campaign.save()
        try:publish(campaign)
        finally:
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
        print(f"Status:{campaign.manifest['status']}; pendentes:{campaign.manifest['pending_variants']}",flush=True)
    if error:raise RuntimeError("readout campaign failed; evidence exported") from error
    return campaign.manifest


def main(argv=None):
    started=STARTED;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--data");p.add_argument("--cache");p.add_argument("--download",action="store_true")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda");p.add_argument("--gpus",nargs=2,default=["0","1"])
    p.add_argument("--session-seconds",type=float,default=7200);p.add_argument("--bootstrap-elapsed-seconds",type=float,default=0.)
    p.add_argument("--bootstrap-metadata");p.add_argument("--engineering-smoke",action="store_true")
    p.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS));args=p.parse_args(argv);args.started=started
    if not 300<args.session_seconds<=7200 or (args.device=="cpu")!=args.engineering_smoke or len(set(args.gpus))!=2:
        raise ValueError("two-hour limit and two slots required; CPU is explicit engineering smoke")
    if not args.engineering_smoke and (not args.bootstrap_metadata or not torch.cuda.is_available() or
            any(not slot.isdecimal() or int(slot)>=torch.cuda.device_count() or "T4" not in torch.cuda.get_device_name(int(slot)) for slot in args.gpus)):
        raise ValueError("science requires pinned bootstrap and two T4 GPUs")
    return execute(args)


if __name__=="__main__":main()
