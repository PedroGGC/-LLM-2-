"""Learned retrieval task using the existing Trainer/checkpoint contract."""
import argparse
from dataclasses import fields,is_dataclass
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from ..config import load_config
from ..checkpoint import read_checkpoint_metadata
from ..data import write_json
from ..runtime import attention_context
from ..trainer import Trainer
from .recall_data import RecallData,RecallStream,generate,TASK_ID
from .recall_protocol import PROTOCOL_ID
from .resource_telemetry import EnergyMeter,energy_result,host_memory
from .state_readout_worker import initial_digest,retain_final


def sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def state_storage_bytes(value,seen=None):
    if value is None:return None
    if isinstance(value,(int,float,bool,str)):return 0
    seen=set() if seen is None else seen
    if isinstance(value,torch.Tensor):
        storage=value.untyped_storage();key=(str(value.device),storage.data_ptr())
        if key in seen:return 0
        seen.add(key);return storage.nbytes()
    items=(getattr(value,f.name) for f in fields(value)) if is_dataclass(value) else value.values() if isinstance(value,dict) else value
    return sum(state_storage_bytes(item,seen) or 0 for item in items)


@torch.no_grad()
def score(model,length,*,examples,seed,device,writes=32,queries=16,smoke=False):
    was=model.training;model.eval();rng=np.random.default_rng(seed)
    correct=count=sequences=0;loss=0.;examples_out=[];buckets={str(n):{"correct":0,"queries":0} for n in (128,256,512,1024,2048,4096,8192)}
    began=time.monotonic()
    try:
        with attention_context(device.type=="cpu"):
            for _ in range(examples):
                batch=generate(rng,1,length,writes=min(writes,length//8),queries=min(queries,length//8),device=device)
                logits=model(batch["input_ids"]).logits;mask=batch["targets"]!=-100
                targets=batch["targets"][mask];pred=logits.argmax(-1)[mask]
                valid=pred==targets;dist=batch["distances"][mask]
                loss+=float(torch.nn.functional.cross_entropy(logits[mask].float(),targets,reduction="sum"))
                correct+=int(valid.sum());count+=len(targets);sequences+=int(valid.all())
                for bound,row in buckets.items():
                    low=0 if int(bound)==128 else int(bound)//2
                    chosen=(dist>low)&(dist<=int(bound));row["correct"]+=int(valid[chosen].sum());row["queries"]+=int(chosen.sum())
                if len(examples_out)<8:
                    positions=mask.nonzero()[:,1]
                    for p,a,b,delta in zip(positions.tolist(),targets.tolist(),pred.tolist(),dist.tolist()):
                        examples_out.append({"key":int(batch["input_ids"][0,p]),"query_position":p,"distance_bytes":delta,
                            "expected_byte":a,"predicted_byte":b,"correct":a==b})
                        if len(examples_out)>=8:break
        sync(device)
    finally:model.train(was)
    return {"length":length,"examples":examples,"queries":count,"correct":correct,"accuracy":correct/count,
        "sequence_accuracy":sequences/examples,"query_nll":loss/count,"distance_buckets":buckets,
        "sample_predictions":examples_out,"evaluation_wall_seconds":time.monotonic()-began,"seed":seed,
        "scope":"single-byte answers; full 257-class head; not language perplexity","writes_per_example":min(writes,length//8)}


class RecallTrainer(Trainer):
    def __init__(self,config,*,smoke=False,**kwargs):
        self.smoke=smoke;self.evaluations=[]
        super().__init__(config,data=RecallData(config.data.seq_len),sampler_factory=lambda data,cfg:RecallStream(cfg),**kwargs)
    def validate(self,max_batches=None):
        value=score(self.model,self.config.data.seq_len,examples=2 if self.smoke else 64,seed=91001,device=self.device)
        cost=self._evaluation_cost()
        if cost is not None:self.counters["validation_flops_estimate"]+=cost*value["examples"]
        self._update_experiment_wall();self._update_peak_memory()
        point={"step":self.counters["step"],"training_queries":self.counters["tokens_processed"],
            "training_input_bytes":self.counters["input_tokens_processed"],**value}
        self.evaluations.append(point)
        return {"validation_loss":value["query_nll"],"validation_accuracy":value["accuracy"],
            "validation_queries":value["queries"],"validation_scope":"fixed procedural development examples"}


def forward_profile(trainer,*,smoke=False):
    model=trainer.model;model.eval();device=trainer.device
    batch=generate(np.random.default_rng(93001),1,trainer.config.data.seq_len,
        writes=min(32,trainer.config.data.seq_len//8),queries=min(16,trainer.config.data.seq_len//8),device=device)
    ids=batch["input_ids"]
    samples=[]
    with torch.no_grad(),attention_context(device.type=="cpu"):
        for _ in range(1 if smoke else 3):sample=model(ids)
        state_bytes=state_storage_bytes(sample.state);del sample
        sync(device)
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        meter=EnergyMeter(device);start=meter.energy_mj();begin=time.monotonic();power=[]
        watts=meter.power_watts()
        if watts is not None:power.append((begin,watts))
        count=0
        while count<(2 if smoke else 30) or (not smoke and (start is not None or watts is not None) and time.monotonic()-begin<3):
            sync(device);t=time.monotonic();model(ids);sync(device)
            if count<(2 if smoke else 30):samples.append(time.monotonic()-t)
            count+=1;watts=meter.power_watts()
            if watts is not None:power.append((time.monotonic(),watts))
        energy=energy_result(start,meter.energy_mj(),power,count*int((batch["targets"]!=-100).sum()),time.monotonic()-begin)
        meter.close()
    energy["gpu_joules_per_query"]=energy.pop("gpu_joules_per_training_target")
    energy["measured_queries"]=energy.pop("measured_training_targets")
    energy["scope"]="isolated batch1 full forward; repeated fixed input; GPU board only; no host/compile/data generation"
    return {"median_forward_seconds":statistics.median(samples),"samples_seconds":samples,"batch":1,
        "input_bytes":ids.shape[1],"queries_per_forward":int((batch["targets"]!=-100).sum()),"energy":energy,
        "peak_forward_vram_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        "host_memory":host_memory(),"returned_state_storage_bytes":state_bytes,
        "scope":"prefill-like full forward, not autoregressive decoding or time to quality"}


def run(config,mode,*,smoke=False,output=None,weights=None):
    cfg=load_config(config)
    if weights:
        meta=read_checkpoint_metadata(weights)
        if meta["config"]!=cfg.to_dict() or meta["dataset_sha256"]!=TASK_ID:raise ValueError("measurement checkpoint/config/task mismatch")
    if output:cfg.train.output_dir=str(output)
    with RecallTrainer(cfg,smoke=smoke) as trainer:
        if weights:
            if meta["provenance"]["source_sha256"]!=trainer.provenance["source_sha256"]:raise ValueError("weights source mismatch")
            from safetensors.torch import load_model
            load_model(trainer.model,str(Path(weights)/"model.safetensors"))
        initial=initial_digest(trainer.model)
        if mode=="train":
            write_json(Path(cfg.train.output_dir)/"initialization.json",{"unique_parameters_sha256":initial,"start_step":0,"seed":cfg.train.seed})
            trainer.fit();meta=read_checkpoint_metadata(trainer.last_checkpoint)
            if trainer.counters["step"]==cfg.train.max_steps:retain_final(cfg.train.output_dir,trainer.last_checkpoint)
            value={"step":trainer.counters["step"],"stop_reason":trainer.stop_reason,"counters":trainer.counters,
                "checkpoint":str(trainer.last_checkpoint),"checkpoint_model_sha256":meta["files"]["model.safetensors"],
                "initial_parameters_sha256":initial,"evaluations":trainer.evaluations}
        elif mode=="pilot":
            warm=[trainer.train_step()["step_train_wall_seconds"] for _ in range(1 if smoke else 3)]
            samples=[trainer.train_step()["step_train_wall_seconds"] for _ in range(2 if smoke else 8)]
            begin=time.monotonic();trainer.validate()
            value={"warmup_seconds":sum(warm),"median_step_seconds":statistics.median(samples),"samples_seconds":samples,
                "validation_seconds":time.monotonic()-begin,"profile":forward_profile(trainer,smoke=smoke)}
        else:
            lengths=[cfg.data.seq_len,cfg.model.max_seq_len] if smoke else [1024,2048,4096,8192]
            values=[]
            for length in lengths:
                try:values.append(score(trainer.model,length,examples=2 if smoke else 128,seed=92001+length,device=trainer.device))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache();values.append({"length":length,"error":"CUDA OOM","accuracy":None})
            capacity=[score(trainer.model,cfg.data.seq_len,examples=2 if smoke else 64,seed=94001+w,device=trainer.device,
                            writes=min(w,cfg.data.seq_len//8)) for w in ((2,4) if smoke else (8,64))]
            value={"heldout":values,"capacity":capacity,"profile":forward_profile(trainer,smoke=smoke),
                "weights_checkpoint":str(weights),"heldout_used_for_training_or_early_stopping":False,
                "selection_scope":"exploratory final probes may guide next campaign; confirmation needs new generator seeds",
                "status":"complete" if all("error" not in v for v in values) else "partial_evaluation"}
        value.update(protocol_id=PROTOCOL_ID,task_sha256=TASK_ID,source_sha256=trainer.provenance["source_sha256"],
            environment=trainer.environment_info,parameters=trainer.model_parameters,
            state_bytes_per_sequence=trainer.model.state_bytes_per_sequence() if hasattr(trainer.model,"state_bytes_per_sequence") else None,
            peak_train_vram_bytes=trainer.counters["peak_vram_allocated_bytes"],flops_estimate=trainer.flops)
        write_json(Path(cfg.train.output_dir)/"worker.json",value)
        return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--mode",choices=("train","pilot","post"),required=True);p.add_argument("--output");p.add_argument("--weights")
    p.add_argument("--engineering-smoke",action="store_true");a=p.parse_args(argv)
    if (a.mode=="post")!=bool(a.weights):raise ValueError("only post measurement requires weights")
    run(a.config,a.mode,smoke=a.engineering_smoke,output=a.output,weights=a.weights)


if __name__=="__main__":main()
