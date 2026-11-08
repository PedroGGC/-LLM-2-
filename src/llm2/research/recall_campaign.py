"""Two-T4 retrieval screen; whole jobs, two-hour deadline and final archives only."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import yaml
STARTED=time.monotonic()
import torch
from ..checkpoint import read_checkpoint_metadata
from ..config import ExperimentConfig
from ..data import write_json,sha256_file
from ..runtime import environment,save_source_snapshot
from .session import SessionBudget,SessionBudgetExceeded,SessionCampaign
from .state_b_operator import SlotCaches
from .state_study import storage_budget,discard_compiler_cache,materialize
from .cycle3_short import heartbeat
from .cycle4 import archive_files,checkpoint_files
from .campaign_storage import verify_pair
from .recall_protocol import VARIANTS,PAIRS,SEEDS,PROTOCOL,PROTOCOL_ID,jobs,configuration
from .recall_data import TASK_ID

ROOT=Path(__file__).resolve().parents[3]


class RecallCampaign(SessionCampaign):
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        from .kaggle_light import remove_cache,tree_bytes
        self.caches=SlotCaches(self,remove_cache,tree_bytes)
    def command(self,arguments,log_name,*,gpu=None,job=None,timeout=None):
        family="preflight"
        if "--config" in arguments:
            cfg=yaml.safe_load(Path(arguments[arguments.index("--config")+1]).read_text(encoding="utf-8"))
            relevant={"model":cfg["model"],"research":cfg["research"],"batch":cfg["data"]["batch_size"],"context":cfg["data"]["seq_len"]}
            if relevant["research"]["memory_pattern"] in ("uniform","binary","fibonacci"):
                relevant["research"]["memory_pattern"]="sparse_tensor_spacing"
            family=hashlib.sha256(json.dumps(relevant,sort_keys=True).encode()).hexdigest()[:16]
        call=lambda:super(RecallCampaign,self).command(arguments,log_name,gpu=gpu,job=job,timeout=timeout)
        return call() if gpu is None else self.caches.run(gpu,family,call)


def worker(campaign,args,cfg,name,mode,slot,*,output=None,weights=None,job=None):
    path=materialize(cfg,campaign.output/"configs"/(name+".yaml"))
    cmd=["-m","llm2.research.recall_worker","--config",str(path),"--mode",mode]
    if output:cmd.extend(["--output",str(output)])
    if weights:cmd.extend(["--weights",str(weights)])
    if args.engineering_smoke:cmd.append("--engineering-smoke")
    started=time.monotonic()
    campaign.command(cmd,name+".log",gpu=None if args.engineering_smoke else slot,job=job)
    value=json.loads((Path(output or cfg.train.output_dir)/"worker.json").read_text(encoding="utf-8"))
    if mode=="train" and value.get("evaluations"):
        print(f"Acerto de desenvolvimento {name}: {100*value['evaluations'][-1]['accuracy']:.2f}%",flush=True)
    if mode=="post":
        for point in value["heldout"]:
            text=f"{100*point['accuracy']:.2f}%" if point.get("accuracy") is not None else point.get("error","indisponível")
            print(f"Teste {name}, contexto{point['length']}: {text}",flush=True)
    return {**value,"command_wall_seconds":time.monotonic()-started}


def comparable(value):
    value=json.loads(json.dumps(value));value["train"].pop("output_dir");value["data"].pop("path")
    return value


def native_history(campaign,args):
    rows={};runtime=environment()
    for path in sorted(campaign.output.parent.glob("llm2-campaign-*/campaign.json")):
        if path.parent==campaign.output or path.is_symlink():continue
        try:old=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,ValueError):continue
        if old.get("cycle")!="recall_memory_v1" or old.get("protocol_id")!=PROTOCOL_ID:continue
        if old.get("engineering_smoke")!=args.engineering_smoke:continue
        if old["provenance"]["source_sha256"]!=campaign.manifest["provenance"]["source_sha256"]:
            raise ValueError("recall history source mismatch")
        if any(old["environment"].get(k)!=runtime.get(k) for k in ("python","packages","cuda_version","cudnn_version")):
            raise ValueError("recall history runtime mismatch")
        for j in old["runs"]+old.get("historical_runs",[]):
            if j["status"] not in ("trained","complete"):continue
            expected=configuration(j["variant"],"unused",j["seed"],smoke=args.engineering_smoke,device=args.device)
            if comparable(expected.to_dict())!=comparable(j["config_resolved"]):raise ValueError("recall history config mismatch")
            origin=Path(j.get("historical_campaign",path.parent))
            if origin.is_symlink() or not origin.resolve().is_relative_to(campaign.output.parent.resolve()):raise ValueError("historical origin outside working")
            folder=Path(j["checkpoint"])
            if folder.is_symlink() or not folder.resolve().is_relative_to(origin.resolve()):raise ValueError("checkpoint outside historical campaign")
            meta=read_checkpoint_metadata(folder)
            if (j["step"]!=expected.train.max_steps or meta["dataset_sha256"]!=TASK_ID or
                    meta["config"]!=j["config_resolved"] or meta["files"]["model.safetensors"]!=j["checkpoint_model_sha256"] or
                    meta["provenance"]["source_sha256"]!=campaign.manifest["provenance"]["source_sha256"]):
                raise ValueError("recall history checkpoint mismatch")
            if j["id"] in rows and rows[j["id"]]["checkpoint_model_sha256"]!=j["checkpoint_model_sha256"]:
                raise ValueError("conflicting same-seed recall history")
            if j["id"] in rows and rows[j["id"]]["status"]=="complete" and j.get("post",{}).get("status")!="complete":continue
            rows[j["id"]]={**j,"status":"complete" if j.get("post",{}).get("status")=="complete" else "trained",
                "historical_campaign":str(origin),"history_manifest_sha256":sha256_file(origin/"campaign.json")}
    return list(rows.values())


def forecast(job,cold,warm,contention=1.,*,smoke=False):
    if smoke:return 0.,0.
    train=1.35*(job["max_steps"]*cold["median_step_seconds"]*contention+cold["command_wall_seconds"]+4*cold["validation_seconds"]+45)
    multiplier=170 if job["variant"]=="transformer" else 34
    post=1.35*(warm["command_wall_seconds"]+multiplier*cold["validation_seconds"]+60)
    return train,post


def batches(rows):
    groups=[]
    for seed in SEEDS:
        for pair in PAIRS:
            group=[j for j in rows if j["seed"]==seed and j["variant"] in pair]
            if group:groups.append(group)
    return groups


def admit(groups,profiles,remaining,contention,*,smoke=False):
    chosen=[];total=0.
    for group in groups:
        costs=[forecast(j,*profiles[j["variant"]],contention,smoke=smoke) for j in group]
        cost=max(t for t,p in costs)+sum(p for t,p in costs)
        if total+cost>remaining:break
        for j,(t,p) in zip(group,costs):j.update(estimated_train_seconds=t,estimated_post_seconds=p)
        chosen.append(group);total+=cost
    return chosen,total


def report(campaign):
    evidence=[j for j in campaign.manifest.get("historical_runs",[])+campaign.manifest["runs"] if j["status"]=="complete"]
    rows=[]
    for j in evidence:
        for e in j.get("post",{}).get("heldout",[]):
            p=j.get("post",{}).get("profile",{})
            rows.append({"variant":j["variant"],"seed":j["seed"],"length":e["length"],"accuracy_percent":100*e["accuracy"] if e.get("accuracy") is not None else None,
                "correct":e.get("correct"),"queries":e.get("queries"),"query_nll":e.get("query_nll"),"error":e.get("error"),
                "parameters":j["parameters"],"state_bytes_per_sequence":j["state_bytes_per_sequence"],"profile_length":p.get("input_bytes"),
                "median_forward_ms":1000*p["median_forward_seconds"] if p else None,"peak_forward_vram_bytes":p.get("peak_forward_vram_bytes"),
                "gpu_joules_per_query":p.get("energy",{}).get("gpu_joules_per_query"),"physical_gpu_slot":j["gpu_slot"]})
    folder=campaign.output/"report";folder.mkdir(exist_ok=True)
    value={"protocol_id":PROTOCOL_ID,"task_sha256":TASK_ID,"status":campaign.manifest["status"],"rows":rows,
        "decision":"exploratory; no automatic superiority claim","target":{k:PROTOCOL[k] for k in ("target_accuracy","accuracy_margin","resource_factor")},
        "limitations":["procedural one-byte recall, not natural language or official MQAR","two seeds, fixed LR, not iso-compute",
            "profile is full forward at training length, not KV decoding or time-to-quality","GPU energy excludes host and compilation",
            "Fibonacci uses rescaled gap spacing; dense bank has larger read budget","long-context probes do not establish general reasoning or LLM capability"]}
    write_json(folder/"report.json",value)
    if rows:
        with (folder/"scores.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    body="".join("<tr>"+"".join(f"<td>{html.escape(str(r[k]))}</td>" for k in ("variant","seed","length","accuracy_percent","parameters","median_forward_ms"))+"</tr>" for r in rows)
    page='<meta charset="utf-8"><title>Recuperação: resultados</title><style>body{font:16px sans-serif;margin:30px}td,th{padding:9px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style>'
    page+='<h1>Recuperação associativa — triagem</h1><p>Menor custo e maior acurácia são desejáveis. Tempos são de forward completo em T1024; não se aplicam automaticamente às linhas de outros comprimentos. Uma tarefa sintética não demonstra superioridade geral como LLM.</p>'
    page+='<table><tr><th>Modelo</th><th>Seed</th><th>Comprimento</th><th>Acerto %</th><th>Parâmetros</th><th>Forward T1024 ms</th></tr>'+body+'</table>'
    (folder/"report.html").write_text(page,encoding="utf-8",newline="\n")
    for r in rows:
        if r["length"]==campaign.manifest["protocol"]["context"]:print(f"RESULTADO {r['variant']} seed{r['seed']}: {r['accuracy_percent']}% | {r['parameters']} parâmetros",flush=True)


def publish(campaign):
    campaign.save();report(campaign)
    files=[p for p in campaign.output.rglob("*") if p.is_file() and "compiler-cache" not in p.parts and
           p.suffix in (".json",".jsonl",".yaml",".log",".txt",".csv",".html",".zip") and
           (p.name!="source.zip" or p.parent==campaign.output)]
    result=archive_files(campaign.output.parent/(campaign.output.name+"-results.zip"),files,base=campaign.output)
    backup=archive_files(campaign.output.parent/(campaign.output.name+"-checkpoints.zip"),checkpoint_files(campaign,campaign.manifest["runs"]),base=campaign.output)
    verify_pair(result,backup)
    return result,backup


def execute(args):
    budget=SessionBudget(args.session_seconds,300,elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-args.started)
    campaign=RecallCampaign(ROOT,args.output,device=args.device,gpus=args.gpus,budget=budget)
    campaign.manifest.update(cycle="recall_memory_v1",protocol=PROTOCOL,protocol_id=PROTOCOL_ID,task_sha256=TASK_ID,dataset_sha256=TASK_ID,
        engineering_smoke=args.engineering_smoke,environment=environment(),historical_runs=[],
        limitations="procedural retrieval screen; different parameters and read budgets; no language or general efficiency claim")
    old={k:os.environ.get(k) for k in ("LLM2_CYCLE4","LLM2_COMPILER_CACHE_ROOT")}
    os.environ.update(LLM2_CYCLE4="1",LLM2_COMPILER_CACHE_ROOT=str(campaign.output/"compiler-cache"))
    error=None
    try:
        with storage_budget(campaign,smoke=args.engineering_smoke),heartbeat(campaign,"memória e recuperação"):
            if args.bootstrap_metadata:
                metadata=json.loads(Path(args.bootstrap_metadata).read_text(encoding="utf-8"))
                for name,digest in metadata["source_files_sha256"].items():
                    if sha256_file(ROOT/name)!=digest:raise ValueError("bootstrap source changed: "+name)
                if metadata["source_git_revision"]!=campaign.manifest["provenance"]["git"]["commit"]:raise ValueError("bootstrap revision mismatch")
                write_json(campaign.output/"bootstrap.json",metadata)
            save_source_snapshot(campaign.output/"source.zip",campaign.manifest["provenance"])
            if not args.engineering_smoke:
                campaign.command(["-m","pytest","-q","tests/test_recall_models.py","-m","not cuda"],"tests-recall-cpu.log")
                campaign.command(["-m","pytest","-q","tests/test_recall_models.py","-m","cuda"],"tests-recall-cuda.log",gpu=args.gpus[0])
            history=native_history(campaign,args);campaign.manifest["historical_runs"]=history
            # Completed training is never repeated merely because its post-evaluation hit a deadline.
            for j in history:
                if j["status"]=="trained":
                    budget.require(0 if args.engineering_smoke else 300)
                    name="post-recovered-"+j["id"];cfg=ExperimentConfig.from_dict(j["config_resolved"])
                    j["post"]=worker(campaign,args,cfg,name,"post",j["gpu_slot"],output=campaign.output/"post"/name,weights=j["checkpoint"])
                    j["status"]="complete";campaign.save()
            done={j["id"] for j in history}
            rows=[j for j in jobs(campaign.output,args.gpus,seeds=args.seeds,smoke=args.engineering_smoke,device=args.device) if j["id"] not in done]
            campaign.manifest["runs"]=rows;campaign.save()
            required=sum(j["parameters"]*48 for j in rows)+(0 if args.engineering_smoke else 4*1024**3)
            campaign.manifest["storage_estimate"]={"required_free_bytes":required}
            if shutil.disk_usage(campaign.output).free<required:raise OSError("insufficient disk space including cache and reserve")
            profiles={}
            for v in VARIANTS:
                job=next((j for j in rows if j["variant"]==v),None)
                if job is None:continue
                values=[]
                for label in ("cold","warm"):
                    name=f"pilot-{v}-{label}";cfg=configuration(v,campaign.output/"pilots"/name,job["seed"],smoke=args.engineering_smoke,device=args.device)
                    values.append(worker(campaign,args,cfg,name,"pilot",job["gpu_slot"]))
                profiles[v]=values
            contention=1.
            first_group=next(iter(batches(rows)),[])
            if len(first_group)==2:
                def concurrent(j):
                    name="concurrent-"+j["variant"];cfg=configuration(j["variant"],campaign.output/"pilots"/name,j["seed"],smoke=args.engineering_smoke,device=args.device)
                    return j,worker(campaign,args,cfg,name,"pilot",j["gpu_slot"])
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results=list(pool.map(concurrent,first_group))
                contention=max(1.,*(v["median_step_seconds"]/profiles[j["variant"]][1]["median_step_seconds"] for j,v in results))
                campaign.manifest["concurrent_pilot"]={j["id"]:v for j,v in results}
            selected,total=admit(batches(rows),profiles,budget.work_remaining,contention,smoke=args.engineering_smoke)
            campaign.manifest.update(pilot_profiles=profiles,contention_factor=contention,forecast_seconds=total,
                admitted_ids=[j["id"] for group in selected for j in group]);campaign.save()
            print(f"Previsão dos trabalhos admitidos:{total/60:.1f}min; {sum(map(len,selected))}/{len(rows)}treinos pendentes cabem. Horizontes fixos.",flush=True)
            for group in selected:
                budget.require(max(j["estimated_train_seconds"] for j in group)+sum(j["estimated_post_seconds"] for j in group))
                def train(j):
                    j["status"]="running";campaign.save();cfg=ExperimentConfig.from_dict(j["config_resolved"])
                    result=worker(campaign,args,cfg,j["id"],"train",j["gpu_slot"],job=j)
                    if result["step"]!=j["max_steps"] or result["task_sha256"]!=TASK_ID or result["source_sha256"]!=campaign.manifest["provenance"]["source_sha256"]:
                        raise ValueError("worker horizon/task/source mismatch")
                    j.update(result,status="trained");campaign.save()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(train,j) for j in group]
                    try:
                        for f in as_completed(futures):f.result()
                    except BaseException:campaign.abort();raise
                for j in group:
                    name="post-"+j["id"];cfg=ExperimentConfig.from_dict(j["config_resolved"])
                    j["post"]=worker(campaign,args,cfg,name,"post",j["gpu_slot"],output=campaign.output/"post"/name,weights=j["checkpoint"])
                    j["status"]="complete";campaign.save()
            evidence=[j for j in history+rows if j["status"]=="complete"]
            for seed in args.seeds:
                hashes={j["initial_parameters_sha256"] for j in evidence if j["seed"]==seed and j["variant"].startswith("memory_")}
                if len(hashes)>1:raise ValueError("memory initial weights differ between patterns")
            expected={f"{v}-seed-{s}" for s in args.seeds for v in VARIANTS}
            campaign.manifest["status"]="complete" if expected<={j["id"] for j in evidence} and all(j.get("post",{}).get("status")=="complete" for j in evidence) else "partial_planned"
    except SessionBudgetExceeded as exc:campaign.manifest.update(status="partial_budget",error=str(exc))
    except Exception as exc:error=exc;campaign.manifest.update(status="failed",error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort()
        try:discard_compiler_cache(campaign)
        except Exception as exc:campaign.manifest["cache_cleanup_error"]=str(exc)
        complete=[j for j in campaign.manifest.get("historical_runs",[])+campaign.manifest["runs"] if j["status"]=="complete"]
        campaign.manifest["pending_ids"]=[f"{v}-seed-{s}" for s in args.seeds for v in VARIANTS if f"{v}-seed-{s}" not in {j["id"] for j in complete}]
        campaign.manifest["session_budget"]=budget.snapshot()
        try:result,backup=publish(campaign)
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
        print(f"Status:{campaign.manifest['status']} | {result.name} | {backup.name}",flush=True)
    if error:raise RuntimeError("recall campaign failed; evidence exported") from error
    return campaign.manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);p.add_argument("--gpus",nargs=2,default=["0","1"])
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS));p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--engineering-smoke",action="store_true");p.add_argument("--bootstrap-metadata")
    p.add_argument("--session-seconds",type=float,default=7200);p.add_argument("--bootstrap-elapsed-seconds",type=float,default=0.)
    args=p.parse_args(argv);args.started=STARTED
    if not 300<args.session_seconds<=7200 or (args.device=="cpu")!=args.engineering_smoke or len(set(args.gpus))!=2:
        raise ValueError("two-hour bound/two slots required; CPU is engineering smoke only")
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<=set(SEEDS):raise ValueError("invalid seeds")
    if not args.engineering_smoke and (not args.bootstrap_metadata or not torch.cuda.is_available() or
            any(not s.isdecimal() or int(s)>=torch.cuda.device_count() or "T4" not in torch.cuda.get_device_name(int(s)) for s in args.gpus)):
        raise ValueError("science requires pinned bootstrap and two T4 GPUs")
    return execute(args)


if __name__=="__main__":main()
