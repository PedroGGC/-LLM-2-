"""Single-process study worker with document isolation and explicit evaluation scopes."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import torch
from ..checkpoint import read_checkpoint_metadata
from ..config import load_config
from ..data import write_json
from ..evaluation import evaluate
from ..trainer import Trainer
from .document_windows import sampler_factory,validation
from .resource_telemetry import EnergyMeter,energy_result,host_memory


class DocumentTrainer(Trainer):
    def __init__(self,*args,tuning=False,**kwargs):
        self.evaluations=[];self.tuning=tuning
        super().__init__(*args,sampler_factory=sampler_factory,**kwargs)

    def validate(self,max_batches=None):
        cfg=self.config;step=self.counters["step"]
        full=step==cfg.train.max_steps or (not self.tuning and step==cfg.train.max_steps//2)
        limit=None if full else max_batches
        started=time.monotonic()
        result=evaluate(self.model,self.objective,
            validation(self.data,cfg.data.seq_len,cfg.data.eval_batch_size,device=self.device,max_batches=limit),
            device=self.device,precision=cfg.train.precision,deterministic=cfg.train.deterministic)
        result.update(validation_scope="all_document_full_windows" if limit is None else "fixed_document_subset",
            validation_bits_per_byte=result["validation_loss"]/math.log(2),eos_included=False)
        cost=self._evaluation_cost()
        if cost is not None:self.counters["validation_flops_estimate"]+=cost*result["validation_sequences"]
        self._update_experiment_wall();self._update_peak_memory()
        self.evaluations.append({"step":step,"full":limit is None,"wall_seconds":time.monotonic()-started,
            "training_targets":self.counters["tokens_processed"],"train_wall_seconds":self.counters["train_wall_seconds"],**result})
        return result


def run(config,*,mode="train",smoke=False,output=None,weights=None,energy_seconds=0.):
    cfg=load_config(config)
    if weights and read_checkpoint_metadata(weights)["config"]!=cfg.to_dict():
        raise ValueError("measurement weights/config mismatch")
    if output is not None:cfg.train.output_dir=str(output)
    with DocumentTrainer(cfg,tuning=mode=="tune") as trainer:
        if weights:
            from safetensors.torch import load_model
            load_model(trainer.model,str(Path(weights)/"model.safetensors"))
        if mode=="pilot":
            warmup=1 if smoke else 10;count=2 if smoke else 30
            meter=EnergyMeter(trainer.device)
            warm=[trainer.train_step()["step_train_wall_seconds"] for _ in range(warmup)]
            start_energy=meter.energy_mj();begin=time.monotonic()
            watts=meter.power_watts();power=[]
            if watts is not None:power.append((begin,watts))
            samples=[trainer.train_step()["step_train_wall_seconds"] for _ in range(count)]
            energy_steps=count
            watts=meter.power_watts()
            if watts is not None:power.append((time.monotonic(),watts))
            while (not smoke and energy_seconds>0 and (start_energy is not None or watts is not None)
                   and time.monotonic()-begin<energy_seconds and trainer.counters["step"]<cfg.train.max_steps-1):
                trainer.train_step();energy_steps+=1
                watts=meter.power_watts()
                if watts is not None:power.append((time.monotonic(),watts))
            energy=energy_result(start_energy,meter.energy_mj(),power,energy_steps*cfg.data.batch_size*cfg.data.seq_len,
                                 time.monotonic()-begin);meter.close()
            # Full-validation timing is isolated and does not select by loss.
            validation_start=time.monotonic()
            evaluate(trainer.model,trainer.objective,
                validation(trainer.data,cfg.data.seq_len,cfg.data.eval_batch_size,device=trainer.device),
                device=trainer.device,precision="fp32",deterministic=cfg.train.deterministic)
            result={"mean_step_seconds":statistics.mean(samples),"median_step_seconds":statistics.median(samples),
                "samples_seconds":samples,"warmup_seconds":sum(warm),"first_step_seconds":warm[0],
                "full_validation_seconds":time.monotonic()-validation_start,"gpu_energy":energy,
                "host_memory":host_memory(),"peak_vram_bytes":trainer.counters["peak_vram_allocated_bytes"],
                "state_bytes_per_sequence":trainer.model.state_bytes_per_sequence() if hasattr(trainer.model,"state_bytes_per_sequence") else None,
                "parameters":trainer.model_parameters,"timing_scope":"isolated forward/backward/clip/optimizer; compile warmup separate",
                "flops_estimate":trainer.flops,"weights_checkpoint":weights,"environment":trainer.environment_info,
                "generation":None,"generation_limitation":"streaming state correctness covered; matched generation/KV benchmark pending",
                "measurement_clone":"fresh optimizer; not checkpoint resume"}
        else:
            trainer.fit()
            meta=read_checkpoint_metadata(trainer.last_checkpoint)
            result={"evaluations":trainer.evaluations,"checkpoint":str(trainer.last_checkpoint),
                "checkpoint_model_sha256":meta["files"]["model.safetensors"],"stop_reason":trainer.stop_reason,
                "step":trainer.counters["step"],"parameters":trainer.model_parameters,
                "source_sha256":trainer.provenance["source_sha256"],"dataset_sha256":trainer.data.fingerprint,
                "counters":trainer.counters,"flops_estimate":trainer.flops,
                "environment":trainer.environment_info,
                "sampler_policy":"uniform eligible starts within a document; independent reset each window",
                "test_evaluated":False,"official_test_evaluated":False}
        write_json(Path(cfg.train.output_dir)/"worker.json",result)
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--mode",choices=("train","pilot","tune"),default="train")
    p.add_argument("--output");p.add_argument("--weights")
    p.add_argument("--energy-seconds",type=float,default=0.)
    p.add_argument("--engineering-smoke",action="store_true");a=p.parse_args(argv)
    if not math.isfinite(a.energy_seconds) or not 0<=a.energy_seconds<=30:raise ValueError("energy window must be 0..30 seconds")
    return run(a.config,mode=a.mode,smoke=a.engineering_smoke,output=a.output,weights=a.weights,energy_seconds=a.energy_seconds)


if __name__=="__main__":main()
