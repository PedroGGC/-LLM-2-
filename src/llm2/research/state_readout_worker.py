"""Same DocumentTrainer; fresh initialization and final-only checkpoint retention."""
import argparse
import hashlib
import json
from pathlib import Path

from ..checkpoint import read_checkpoint_metadata
from ..config import load_config
from ..data import write_json
from .state_worker import DocumentTrainer,run as original_run


class ReadoutTrainer(DocumentTrainer):
    def validate(self,max_batches=None):
        import torch
        sums=[0.]*4;counts=[0]*4
        def collect(module,inputs,result):
            targets=inputs[1]["targets"];valid=targets!=-100
            phase=torch.arange(targets.shape[1],device=targets.device)%4
            for i in range(4):
                mask=valid & (phase==i)[None,:]
                sums[i]+=result.token_losses[mask].double().sum().item();counts[i]+=mask.sum().item()
        hook=self.objective.register_forward_hook(collect)
        try:result=super().validate(max_batches)
        finally:hook.remove()
        detail={"phase_nll_sum":sums,"phase_targets":counts,
                "phase_loss":[s/n if n else None for s,n in zip(sums,counts)]}
        self.evaluations[-1].update(detail);result.update(detail)
        return result


def initial_digest(model):
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        digest.update(name.encode());digest.update(str((tuple(value.shape),str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def retain_final(run,final):
    """Only redundant checkpoints in this fresh run; verify before unlinking files."""
    run,final=Path(run).resolve(),Path(final)
    if final.is_symlink() or final.resolve().parent!=run/"checkpoints":raise ValueError("final checkpoint outside run")
    final_meta=read_checkpoint_metadata(final);retired=[]
    for folder in sorted((run/"checkpoints").glob("step-*")):
        if folder==final:continue
        if (folder.is_symlink() or folder.resolve().parent!=run/"checkpoints" or folder.parent.is_symlink()
                or int(folder.name.removeprefix("step-"))>=int(final.name.removeprefix("step-"))):
            raise ValueError("unsafe checkpoint retirement")
        meta=read_checkpoint_metadata(folder)
        if any(meta[k]!=final_meta[k] for k in ("config","dataset_sha256","provenance","architecture_version")):
            raise ValueError("retired checkpoint identity mismatch")
        names={"metadata.json",*meta["files"]}
        if set(p.name for p in folder.iterdir())!=names or any((folder/n).is_symlink() for n in names):
            raise ValueError("unknown payload in intermediate checkpoint")
        retired.append({"checkpoint":folder.name,"metadata":meta})
    # Preserve all intermediate metadata before removing any tensor copy.
    write_json(run/"retired-checkpoints.json",retired)
    for item in retired:
        folder=run/"checkpoints"/item["checkpoint"]
        if folder.is_symlink() or folder.parent.is_symlink() or folder.resolve().parent!=run/"checkpoints":
            raise ValueError("retirement path changed")
        for name in ("metadata.json",*item["metadata"]["files"]):
            if (folder/name).is_symlink():raise ValueError("linked checkpoint file")
            (folder/name).unlink()
        folder.rmdir()
    return retired


def train(config):
    cfg=load_config(config)
    with ReadoutTrainer(cfg) as trainer:
        initialization=initial_digest(trainer.model)
        write_json(Path(cfg.train.output_dir)/"initialization.json",{"unique_parameters_sha256":initialization,
            "parameters":trainer.model_parameters,"seed":cfg.train.seed,"start_step":trainer.counters["step"]})
        trainer.fit()
        meta=read_checkpoint_metadata(trainer.last_checkpoint)
        result={"evaluations":trainer.evaluations,"checkpoint":str(trainer.last_checkpoint),
            "checkpoint_model_sha256":meta["files"]["model.safetensors"],"step":trainer.counters["step"],
            "stop_reason":trainer.stop_reason,"parameters":trainer.model_parameters,
            "source_sha256":trainer.provenance["source_sha256"],"dataset_sha256":trainer.data.fingerprint,
            "counters":trainer.counters,"flops_estimate":trainer.flops,"environment":trainer.environment_info,
            "sampler_policy":"uniform eligible starts within document; independent reset per window",
            "initial_parameters_sha256":initialization,"test_evaluated":False,"official_test_evaluated":False}
        if result["step"]==cfg.train.max_steps and result["stop_reason"]=="max_steps":
            result["retired_checkpoints"]=[r["checkpoint"] for r in retain_final(cfg.train.output_dir,trainer.last_checkpoint)]
        write_json(Path(cfg.train.output_dir)/"worker.json",result)
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--mode",choices=("train","pilot"),default="train")
    p.add_argument("--output");p.add_argument("--weights");p.add_argument("--energy-seconds",type=float,default=0.)
    p.add_argument("--engineering-smoke",action="store_true");a=p.parse_args(argv)
    if a.mode=="train":
        if a.output or a.weights:raise ValueError("main training must start fresh with its configured output")
        return train(a.config)
    if not 0<=a.energy_seconds<=30:raise ValueError("invalid energy measurement duration")
    return original_run(a.config,mode="pilot",smoke=a.engineering_smoke,output=a.output,
                        weights=a.weights,energy_seconds=a.energy_seconds)


if __name__=="__main__":main()
