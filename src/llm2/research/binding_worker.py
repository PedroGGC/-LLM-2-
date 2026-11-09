"""One persistent worker: compile once, restore pilot state, train, evaluate/profile."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import numpy as np
import torch
from ..config import load_config
from ..data import write_json
from ..checkpoint import read_checkpoint_metadata
from ..runtime import attention_context,capture_rng,restore_rng
from ..trainer import Trainer
from .binding_data import generate,intervene,BindingStream,TASK_ID
from .binding_protocol import PROTOCOL_ID
from .recall_worker import sync,state_storage_bytes
from .resource_telemetry import EnergyMeter,energy_result,host_memory
from .state_readout_worker import initial_digest,retain_final


class BindingData:
    def __init__(self,length):
        self.tokenizer=SimpleNamespace(vocab_size=257)
        self.arrays={k:np.zeros(length+1,dtype=np.uint16) for k in ("train","val")}
        self.fingerprint=TASK_ID


@torch.no_grad()
def score(model,length,*,examples,seed,device,writes=32,queries=16,prefix_bytes=0):
    was=model.training;model.eval();rng=np.random.default_rng(seed)
    correct=changed_correct=paired=changes=count=0;loss=0.;rows=[];per_sequence=[]
    begin=time.monotonic()
    try:
        with attention_context(device.type=="cpu"):
            for _ in range(examples):
                batch=generate(rng,1,length,writes=min(writes,length//8),queries=min(queries,length//8),device=device)
                if prefix_bytes:
                    batch={k:torch.nn.functional.pad(v,(prefix_bytes,0),value=131 if k=='input_ids' else -100 if k=='targets' else 0) for k,v in batch.items()}
                alt=intervene(batch);mask=batch["targets"]!=-100
                logits=model(batch["input_ids"]).logits;pred=logits.argmax(-1)[mask]
                alternate=model(alt["input_ids"]).logits.argmax(-1)[mask]
                target=batch["targets"][mask];other=alt["targets"][mask]
                a=pred==target;b=alternate==other
                correct+=int(a.sum());changed_correct+=int(b.sum());paired+=int((a&b).sum())
                changes+=int((pred!=alternate).sum());count+=len(target)
                loss+=float(torch.nn.functional.cross_entropy(logits[mask].float(),target,reduction="sum"))
                per_sequence.append({"queries":len(target),"correct":int(a.sum()),"paired_correct":int((a&b).sum())})
                if not rows:
                    positions=mask.nonzero()[:,1]
                    for p,y,z,yp,zp in zip(positions.tolist(),target.tolist(),pred.tolist(),other.tolist(),alternate.tolist()):
                        rows.append({"position":p,"key":int(batch["input_ids"][0,p]),"answer":y,"prediction":z,
                                     "changed_key":int(alt["input_ids"][0,p]),"changed_answer":yp,"changed_prediction":zp})
                        if len(rows)>=8:break
        sync(device)
    finally:model.train(was)
    return {"length":length+prefix_bytes,"prefix_bytes":prefix_bytes,"examples":examples,"writes":min(writes,length//8),"queries":count,"correct":correct,
        "accuracy":correct/count,"counterfactual_accuracy":changed_correct/count,"paired_accuracy":paired/count,
        "prediction_change_fraction":changes/count,"query_nll":loss/count,"seed":seed,
        "examples_out":rows,"per_sequence":per_sequence,"wall_seconds":time.monotonic()-begin}


class BindingTrainer(Trainer):
    def __init__(self,cfg,*,smoke=False,**kwargs):
        self.smoke=smoke;self.evaluations=[]
        super().__init__(cfg,data=BindingData(cfg.data.seq_len),sampler_factory=lambda data,cfg:BindingStream(cfg),**kwargs)
    def validate(self,max_batches=None):
        point=score(self.model,self.config.data.seq_len,examples=2 if self.smoke else 64,seed=91131,device=self.device)
        cost=self._evaluation_cost()
        if cost is not None:self.counters["validation_flops_estimate"]+=2*cost*point["examples"]
        self._update_experiment_wall();self._update_peak_memory()
        self.evaluations.append({"step":self.counters["step"],"training_queries":self.counters["tokens_processed"],
            "input_bytes":self.counters["input_tokens_processed"],"training_wall_seconds":self.counters['train_wall_seconds'],
            "experiment_wall_seconds":self.counters['experiment_wall_seconds'],**point})
        write_json(self.output_dir/'binding-progress.json',self.evaluations[-1])
        print(f"BINDING step{self.counters['step']}: acerto{100*point['accuracy']:.2f}% | par contrafactual{100*point['paired_accuracy']:.2f}%",flush=True)
        return {"validation_loss":point["query_nll"],"validation_accuracy":point["accuracy"],"paired_accuracy":point["paired_accuracy"]}


def training_benchmark(trainer,*,samples=12,warmup=3):
    """Actual updates for timing, then exact in-memory restore; no training credit."""
    def cpu_copy(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
        if isinstance(value,list):return [cpu_copy(v) for v in value]
        return copy.deepcopy(value)
    saved=cpu_copy((trainer.model.state_dict(),trainer.objective.state_dict(),trainer.optimizer.state_dict(),
        trainer.scheduler.state_dict(),trainer.sampler.state_dict(),trainer.counters))
    rng=capture_rng();mode=trainer.model.training;before=initial_digest(trainer.model)
    def update():
        # Measurement only: do not alter Trainer's horizon or bypass its guards.
        # The binding objective is language loss only, with token-weighted accumulation.
        if trainer.config.train.precision!='fp32':raise ValueError('binding benchmark requires fp32')
        sync(trainer.device);started=time.monotonic();trainer.model.train();trainer.optimizer.zero_grad(set_to_none=True)
        batches=[trainer.sampler.next(trainer.device) for _ in range(trainer.config.train.grad_accum_steps)]
        total=sum(int((b['targets']!=-100).sum()) for b in batches)
        with attention_context(trainer.config.train.deterministic):
            for batch in batches:
                result=trainer.objective(trainer.train_model,batch)
                if not torch.isfinite(result.loss):raise FloatingPointError('non-finite measured loss')
                (result.loss*result.token_count/total).backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters,trainer.config.train.grad_clip,error_if_nonfinite=True)
            trainer.optimizer.step();trainer.scheduler.step();trainer.optimizer.zero_grad(set_to_none=True)
        sync(trainer.device);return time.monotonic()-started
    try:
        warm=[update() for _ in range(warmup)]
        timed=[update() for _ in range(samples)]
        peak=torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type=="cuda" else None
    finally:
        weights,objective,optimizer,scheduler,sampler,counters=saved
        trainer.model.load_state_dict(weights);trainer.objective.load_state_dict(objective)
        trainer.optimizer.load_state_dict(optimizer);trainer.scheduler.load_state_dict(scheduler)
        trainer.sampler.load_state_dict(sampler);trainer.counters=counters;trainer.optimizer.zero_grad(set_to_none=True)
        trainer.model.train(mode);restore_rng(rng)
    if initial_digest(trainer.model)!=before:raise RuntimeError("benchmark changed training weights")
    return {"warmup_seconds":sum(warm),"samples_seconds":timed,"median_step_seconds":statistics.median(timed),
            "peak_vram_bytes":peak,"weights_and_optimizer_restored":True,
            "warmup_scope":"updates including possible Inductor compilation; not a pure compiler timer",
            "scope":"actual training updates restored; not counted in horizon"}


@torch.no_grad()
def profile(trainer,length,*,smoke=False):
    m=trainer.model;was=m.training;m.eval();device=trainer.device
    batch=generate(np.random.default_rng(93317+length),1,length,writes=min(32,length//8),queries=min(16,length//8),device=device)
    samples=[]
    try:
        with attention_context(device.type=="cpu"):
            for _ in range(1 if smoke else 3):out=m(batch["input_ids"])
            state_bytes=state_storage_bytes(out.state);del out;sync(device)
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
            meter=EnergyMeter(device);start=meter.energy_mj();began=time.monotonic();power=[];n=0
            watts=meter.power_watts()
            if watts is not None:power.append((began,watts))
            try:
                while n<(2 if smoke else 30) or (not smoke and (start is not None or watts is not None) and time.monotonic()-began<3):
                    sync(device);t=time.monotonic();m(batch["input_ids"]);sync(device)
                    if n<(2 if smoke else 30):samples.append(time.monotonic()-t)
                    n+=1;watts=meter.power_watts()
                    if watts is not None:power.append((time.monotonic(),watts))
                energy=energy_result(start,meter.energy_mj(),power,n*int((batch["targets"]!=-100).sum()),time.monotonic()-began)
            finally:meter.close()
        energy['gpu_joules_per_query']=energy.pop('gpu_joules_per_training_target')
        energy['measured_queries']=energy.pop('measured_training_targets')
        energy['scope']='isolated GPU board, repeated full forward; host/setup/compilation excluded'
        return {"length":length,"batch":1,"median_forward_seconds":statistics.median(samples),"samples_seconds":samples,
            "returned_state_bytes":state_bytes,"peak_forward_vram_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "energy":energy,"host_memory":host_memory(),"scope":"eager full forward at this exact length; no decoding/KV or time-to-quality"}
    finally:m.train(was)


def diagnose(trainer):
    m=trainer.model;was=m.training;m.eval();m.zero_grad(set_to_none=True)
    batch=generate(np.random.default_rng(95519),1,trainer.config.data.seq_len,writes=min(32,trainer.config.data.seq_len//8),
        queries=min(16,trainer.config.data.seq_len//8),device=trainer.device)
    with attention_context(trainer.device.type=="cpu"):
        logits=m(batch["input_ids"]).logits;mask=batch["targets"]!=-100
        torch.nn.functional.cross_entropy(logits[mask].float(),batch["targets"][mask]).backward()
    grads={name:float(p.grad.float().norm()) if p.grad is not None else None for name,p in m.named_parameters()}
    gates=[]
    if hasattr(m,"operators"):
        with torch.no_grad():
            x,previous,local,tail=m._inputs(batch["input_ids"],None)
            for op in m.operators:
                w=op.write_gate(local).sigmoid()[0,:,0]
                ids=batch["input_ids"][0]
                gates.append({"values":float(w[(ids>=64)&(ids<128)].mean()),"keys":float(w[ids<64].mean()),
                              "distractors":float(w[ids>=131].mean()) if bool((ids>=131).any()) else None})
    m.zero_grad(set_to_none=True);m.train(was)
    return {"parameter_gradient_norms":grads,"write_gate_by_task_role":gates,
            "scope":"one diagnostic batch, no optimizer update; task parser only labels diagnostics, never model input"}


def run(config,*,smoke=False,work_deadline=None,permit=None):
    cfg=load_config(config);output=Path(cfg.train.output_dir);began=time.monotonic()
    with BindingTrainer(cfg,smoke=smoke) as trainer:
        initial=initial_digest(trainer.model)
        initial_scores=[score(trainer.model,cfg.data.seq_len,examples=2 if smoke else 16,seed=90013+w,
            writes=min(w,cfg.data.seq_len//8),device=trainer.device) for w in ((2,) if smoke else (4,32))]
        pilot=training_benchmark(trainer,samples=2 if smoke else 12,warmup=1 if smoke else 3)
        if trainer.device.type=='cuda':torch.cuda.reset_peak_memory_stats(trainer.device)
        estimate=0. if smoke else 1.35*(cfg.train.max_steps*pilot["median_step_seconds"]+4*64*initial_scores[-1]["wall_seconds"]/initial_scores[-1]["examples"]+240)
        result={"protocol_id":PROTOCOL_ID,"task_sha256":TASK_ID,"initial_parameters_sha256":initial,"initial_scores":initial_scores,
            "pilot":pilot,"forecast_remaining_seconds":estimate,"parameters":trainer.model_parameters,
            "state_bytes_per_sequence":trainer.model.state_bytes_per_sequence() if hasattr(trainer.model,"state_bytes_per_sequence") else None,
            "source_sha256":trainer.provenance["source_sha256"],"environment":trainer.environment_info,"flops_estimate":trainer.flops}
        write_json(output/"forecast.json",result)
        print(f"PILOTO pronto: {pilot['median_step_seconds']*1000:.2f}ms/passo; previsão restante{estimate/60:.1f}min; pesos iniciais restaurados.",flush=True)
        if work_deadline is not None and time.monotonic()+estimate>=work_deadline:
            result.update(status="deferred_budget",worker_wall_seconds=time.monotonic()-began)
            write_json(output/"worker.json",result);return result
        trainer.fit();meta=read_checkpoint_metadata(trainer.last_checkpoint)
        result.update(step=trainer.counters["step"],counters=trainer.counters,stop_reason=trainer.stop_reason,
            checkpoint=str(trainer.last_checkpoint),checkpoint_model_sha256=meta["files"]["model.safetensors"],evaluations=trainer.evaluations)
        if trainer.counters["step"]!=cfg.train.max_steps:raise RuntimeError("binding worker did not reach fixed horizon")
        retain_final(output,trainer.last_checkpoint)
        result["status"]="trained";write_json(output/"worker.json",result)
        lengths=[cfg.data.seq_len,cfg.model.max_seq_len] if smoke else [256,1024,4096,8192]
        final=[]
        for length in lengths:
            try:point=score(trainer.model,length,examples=2 if smoke else 64,seed=92231+length,device=trainer.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();point={"length":length,"error":"CUDA OOM","accuracy":None}
            final.append(point)
            print(f"FINAL contexto{length}: acerto{100*point['accuracy']:.2f}% | par{100*point['paired_accuracy']:.2f}%" if point.get("accuracy") is not None else f"FINAL contexto{length}: OOM",flush=True)
        capacity=score(trainer.model,cfg.data.seq_len if smoke else 1024,examples=2 if smoke else 64,seed=94417,
            writes=2 if smoke else 64,device=trainer.device)
        result.update(final=final,capacity=capacity,diagnostics=diagnose(trainer))
        if not smoke:
            result['alignment_probe']=score(trainer.model,1024,examples=64,seed=96611,device=trainer.device,prefix_bytes=3)
        write_json(output/"profile-ready.json",{"status":"ready","at":time.monotonic()})
        while permit is not None and not Path(permit).is_file():
            if work_deadline is not None and time.monotonic()>=work_deadline:raise TimeoutError("profile isolation deadline")
            time.sleep(.25)
        before=initial_digest(trainer.model)
        result["training_profile"]=training_benchmark(trainer,samples=2 if smoke else 30,warmup=1 if smoke else 3)
        result["profiles"]=[profile(trainer,length,smoke=smoke) for length in lengths if not next(e for e in final if e["length"]==length).get("error")]
        if initial_digest(trainer.model)!=before:raise RuntimeError("final profiling changed model")
        result.update(status="complete" if all("error" not in e for e in final) else "partial_evaluation",
            worker_wall_seconds=time.monotonic()-began,final_parameters_sha256=before)
        write_json(output/"worker.json",result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--mode",choices=['train'],default='train')
    p.add_argument("--engineering-smoke",action="store_true");p.add_argument("--work-deadline",type=float);p.add_argument("--profile-permit")
    a=p.parse_args(argv);run(a.config,smoke=a.engineering_smoke,work_deadline=a.work_deadline,permit=a.profile_permit)


if __name__=="__main__":main()
