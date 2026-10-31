"""Synthetic platform qualification. CPU evidence never selects a CUDA winner."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time
import traceback
import torch
from ..runtime import environment, source_provenance,setup_runtime
from ..data import write_json
from .factory import build_backbone
from .state_protocol import RULES, PROTOCOL_ID, configuration, statistics as model_statistics
from .resource_telemetry import host_memory


def qualify(rule,device="cpu",*,compiled=False):
    """Independent sequential oracle, backward, prefix causality and streaming."""
    cfg=configuration("recurrent",rule,"unused","unused",17,smoke=not compiled,device="cpu")
    cfg.model.dropout=0.;cfg.research.state_execution="sequential"
    torch.manual_seed(17);oracle=build_backbone(cfg).to(device)
    fast=copy.deepcopy(oracle)
    for cell in fast.cells:cell.execution="chunked";cell.chunk_size=32 if compiled else 4
    ids=torch.randint(0,cfg.model.vocab_size,(1 if compiled else 2,65 if compiled else 17),device=device)
    implementation=torch.compile(fast,backend="inductor",dynamic=False) if compiled else fast
    expected=oracle(ids).logits;actual=implementation(ids).logits
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
    expected.square().mean().backward();actual.square().mean().backward()
    for a,b in zip(oracle.parameters(),fast.parameters()):
        if a.grad is None or b.grad is None:raise ValueError("missing gradient")
        torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=2e-3)
        if not torch.isfinite(b.grad).all():raise ValueError("nonfinite gradient")
    changed=ids.clone();changed[:,7:]=(changed[:,7:]+1)%cfg.model.vocab_size
    torch.testing.assert_close(fast(ids).logits[:,:7],fast(changed).logits[:,:7],atol=2e-5,rtol=2e-4)
    first=fast(ids[:,:5]);second=fast(ids[:,5:],state=first.state)
    torch.testing.assert_close(torch.cat((first.logits,second.logits),1),actual,atol=2e-5,rtol=2e-4)
    return {"passed":True,"compiled":compiled,"max_forward_absolute_error":float((expected-actual).abs().max().detach().cpu()),
        "checks":["sequential/chunked forward and gradients","causality","streaming","finite gradients"]}


def benchmark(rule,slot,*,smoke=False,warmup=10,steps=30):
    device="cpu" if smoke else f"cuda:{slot}"
    correctness=qualify(rule,device,compiled=not smoke)
    cfg=configuration("recurrent",rule,"unused","unused",17,smoke=smoke,device="cpu" if smoke else "cuda")
    setup_runtime(cfg.train)
    model=build_backbone(cfg).to(device).train()
    # The small correctness case has a different shape. Do not generalize that
    # graph to symbolic B/T when the actual experiment uses fixed-size windows.
    measured=torch.compile(model,backend="inductor",dynamic=False) if not smoke else model
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004)
    ids=torch.randint(0,cfg.model.vocab_size,(cfg.data.batch_size,cfg.data.seq_len),device=device)
    targets=torch.roll(ids,1,1)
    if not smoke:torch.cuda.reset_peak_memory_stats(device)
    def sync():
        if not smoke:torch.cuda.synchronize(device)
    def step():
        sync();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        logits=measured(ids).logits.float()
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten())
        if not torch.isfinite(loss):raise ValueError("nonfinite benchmark loss")
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError("nonfinite benchmark gradient")
        optimizer.step();sync();return time.perf_counter()-started
    warm=[step() for _ in range(warmup)];samples=[step() for _ in range(steps)]
    row={"rule":rule,"slot":str(slot),"status":"passed","correctness":correctness,
        "gpu_name":None if smoke else torch.cuda.get_device_name(slot),
        "mean_step_seconds":statistics.mean(samples),"median_step_seconds":statistics.median(samples),
        "step_samples_seconds":samples,"warmup_seconds":sum(warm),"first_step_seconds":warm[0] if warm else None,
        "peak_vram_bytes":None if smoke else torch.cuda.max_memory_allocated(device),
        "host_memory":host_memory(),"gpu_energy_joules":None,"compile_policy":"static_shapes",
        "config":cfg.to_dict(),**model_statistics(cfg)}
    return row


def select(rows,*,engineering_smoke=False):
    if engineering_smoke:return None
    candidates=[]
    for rule in RULES:
        group=[r for r in rows if r.get("rule")==rule]
        if len(group)!=2 or {r.get("slot") for r in group}!={"0","1"}:continue
        if not all(r.get("status")=="passed" and r.get("correctness",{}).get("passed") is True
                   and r.get("timeout_enforced_by")=="parent_process"
                   and r.get("candidate_limit_seconds")==600.
                   and r.get("supervisor_elapsed_seconds",math.inf)<=602.
                   and r.get("compile_policy")=="static_shapes"
                   and "T4" in str(r.get("gpu_name"))
                   and math.isfinite(r.get("median_step_seconds",math.inf)) and r["median_step_seconds"]>0
                   and type(r.get("peak_vram_bytes")) is int and 0<r["peak_vram_bytes"]<13*1024**3 for r in group):continue
        candidates.append((statistics.mean(r["median_step_seconds"] for r in group),
                           max(r["state_bytes_per_sequence"] for r in group),rule))
    if not candidates:return None
    best=min(c[0] for c in candidates)
    return min((c for c in candidates if c[0]<=best*1.05),key=lambda c:(c[1],c[0],c[2]))[2]


def validate_report(value,*,allow_smoke=False):
    smoke=value.get("engineering_smoke")
    if (type(smoke) is not bool or smoke and not allow_smoke or value.get("protocol_id")!=PROTOCOL_ID
        or value.get("qualification_version")!=2
        or value.get("implementation_scope")!="common_shell_memory_rules"
        or value.get("source_sha256")!=source_provenance()["source_sha256"]):
        raise ValueError("platform report/source/scope mismatch; rerun qualification")
    current=environment()
    if not smoke and any(value.get("environment",{}).get(key)!=current.get(key)
                         for key in ("python","packages","cuda_version","cudnn_version")):
        raise ValueError("platform Torch environment differs")
    chosen=select(value.get("rows",[]),engineering_smoke=smoke)
    if value.get("selected_rule")!=chosen or chosen is None and not smoke:
        raise ValueError("no qualified platform selection")
    return chosen


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--execute",action="store_true")
    p.add_argument("--engineering-smoke",action="store_true")
    p.add_argument("--rule",choices=RULES);p.add_argument("--slot",type=int,choices=(0,1),default=0)
    a=p.parse_args(argv)
    if not a.execute:return {"status":"not_started","rules":RULES,"selected_rule":None}
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    if not a.engineering_smoke and (not torch.cuda.is_available() or torch.cuda.device_count()<(1 if a.rule else 2)):
        raise RuntimeError("scientific qualification requires two T4 GPUs; use engineering-smoke for CPU")
    torch.set_num_threads(2);rows=[]
    for slot in ((a.slot,) if a.rule else ((0,) if a.engineering_smoke else (0,1))):
        for rule in ((a.rule,) if a.rule else RULES):
            print(f"Qualificando {rule}, slot {slot}...",flush=True)
            try:row=benchmark(rule,slot,smoke=a.engineering_smoke,warmup=1 if a.engineering_smoke else 10,
                              steps=2 if a.engineering_smoke else 30)
            except Exception as exc:
                row={"rule":rule,"slot":str(slot),"status":"failed","error":str(exc),"traceback":traceback.format_exc()}
            rows.append(row);write_json(output/f"{rule}-slot-{slot}.json",row)
            if torch.cuda.is_available():torch.cuda.empty_cache()
    result={"protocol_id":PROTOCOL_ID,"implementation_scope":"common_shell_memory_rules",
        "engineering_smoke":a.engineering_smoke,"source_sha256":source_provenance()["source_sha256"],
        "environment":environment(),"rows":rows,"selected_rule":select(rows,engineering_smoke=a.engineering_smoke),
        "selection_scope":"operational rule implementation; not quality comparison or published-model reproduction"}
    write_json(output/"platforms.json",result)
    print(f"Seleção: {result['selected_rule']} (CPU nunca seleciona).",flush=True)
    return result


if __name__=="__main__":main()
