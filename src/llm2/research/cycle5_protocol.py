"""Fixed trained depths at matched estimated compute; no new model mechanism."""
import hashlib
import json
from pathlib import Path
from ..config import ExperimentConfig,load_config
from .cycle4_protocol import ROOT
from .cycle2_protocol import configuration_stats
from .cycle3_protocol import DATA_HASHES

SEEDS=(221,222,223)
VARIANTS=("fine_k1","fine_k2","fine_k4","transformer_two")
PROTOCOL={"cycle":5,"seeds":SEEDS,"variants":VARIANTS,"dataset_files":DATA_HASHES,
    "context":256,"batch":32,"precision":"fp32","compiler":"inductor","dropout":.05,
    "compute_anchor":"fine_k2 at 6000 optimizer steps","rounding":"floor to multiple of 4",
    "primary":"fine_k2-minus-fine_k1","quality_margin":.005,"test_evaluated":False,
    "scope":"three-seed exploration at matched estimated FLOPs/near-equal parameters; tokens and wall time differ"}
PROTOCOL_ID=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def configuration(variant,data,output,seed,*,steps=6000,device="cuda",smoke=False):
    if variant not in VARIANTS or seed not in SEEDS:raise ValueError("unknown Cycle 5 variant/seed")
    if type(steps) is not int or steps<4 or steps%4:raise ValueError("fixed multiple-of-four horizon required")
    cfg=load_config(ROOT/"configs/experiments/cycle3/base.yaml")
    cfg.data.path=str(data);cfg.data.batch_size=2 if smoke else 32
    cfg.train.output_dir=str(output);cfg.train.seed=seed;cfg.train.device=device
    cfg.train.precision="fp32";cfg.train.compile=device=="cuda";cfg.train.deterministic=device=="cpu"
    cfg.train.cpu_threads=2;cfg.train.grad_accum_steps=2 if smoke else 1
    cfg.train.accumulation_weighting="tokens";cfg.train.console_progress=True
    cfg.train.max_steps=cfg.optimizer.schedule_steps=steps
    cfg.optimizer.warmup_steps=1 if smoke else 50
    cfg.train.checkpoint_interval=cfg.train.eval_interval=cfg.train.log_interval=steps//4
    cfg.train.eval_batches=8
    cfg.research.scale_conditioning=False;cfg.research.context_normalization="none"
    cfg.research.max_seconds=cfg.research.shutdown_reserve_seconds=0.
    cfg.model.n_layers=2;cfg.model.hidden_dim=1024
    if variant=="transformer_two":
        cfg.research.architecture="transformer";cfg.research.scale_execution="reference"
        cfg.research.compressed_factors=None;cfg.research.cache_context_projection=False
        cfg.research.paired_common_init=False;cfg.model.hidden_dim=384
    else:
        cfg.research.architecture="fractal";cfg.research.scale_execution="compressed_native"
        cfg.research.compressed_factors=[];cfg.research.cache_context_projection=True
        cfg.research.paired_common_init=True;cfg.research.iterations=int(variant.removeprefix("fine_k"))
    if smoke:
        cfg.model.d_model=32;cfg.model.n_heads=4;cfg.model.hidden_dim=64
        cfg.data.seq_len=cfg.model.max_seq_len=17
    return ExperimentConfig.from_dict(cfg.to_dict())


def ordered_jobs(data,output,gpus,*,device="cuda",smoke=False):
    reference_steps=16 if smoke else 6000
    stats={v:configuration_stats(configuration(v,data,Path(output)/"meta",SEEDS[0],steps=reference_steps,
        device=device,smoke=smoke)) for v in VARIANTS}
    budget=stats["fine_k2"]["step_flops_estimated"]*reference_steps
    jobs=[]
    for index,seed in enumerate(SEEDS):
        order=VARIANTS[index:]+VARIANTS[:index]
        for variant in order:
            steps=int(budget//stats[variant]["step_flops_estimated"])//4*4
            if steps<4:raise ValueError("minimum horizon exceeds compute budget")
            cfg=configuration(variant,data,Path(output)/"runs"/f"{variant}-seed-{seed}",seed,
                steps=steps,device=device,smoke=smoke)
            total=stats[variant]["step_flops_estimated"]*steps
            if total>budget:raise ValueError("estimated compute budget exceeded")
            jobs.append({"id":f"{variant}-seed-{seed}","variant":variant,"model":variant,"arm":variant,
                "seed":seed,"status":"pending","max_steps":steps,"directory":cfg.train.output_dir,
                "gpu_slot":str(gpus[(VARIANTS.index(variant)+index)%len(gpus)]),
                "config_resolved":cfg.to_dict(),"progress_interval":steps//4,
                "batch_size":cfg.data.batch_size,"grad_accum_steps":cfg.train.grad_accum_steps,
                "reference_flop_budget_estimate":budget,"planned_training_flops_estimate":total,
                "relative_flop_budget_shortfall":(budget-total)/budget,**stats[variant]})
    return jobs
