"""Fixed small-model binding campaign, distinct from campaign0030."""
import hashlib
import json
from pathlib import Path
from ..config import ExperimentConfig,ModelConfig,DataConfig,OptimizerConfig,TrainConfig,ResearchConfig
from .binding_data import TASK,TASK_ID

VARIANTS = ("transformer", "transition", "pyramid_shared", "pyramid_independent")
SEEDS = (311,312)
PROTOCOL = {"name":"transition-binding-v1", "task":TASK, "task_sha256":TASK_ID,
    "variants":VARIANTS, "seeds":SEEDS, "steps":3000, "context":256, "batch":16,
    "lr":.001, "warmup":50, "weight_decay":.01, "dropout":.05, "precision":"fp32",
    "periodic_steps":[750,1500,2250,3000], "dev_examples":64,
    "final_lengths":[256,1024,4096,8192], "final_examples":64,
    "dev_seed":91131, "final_seed":92231, "capacity_writes":64,
    "alignment_probe":{"base_length":1024,"prefix_bytes":3},
    "dimensions":{"memory":96,"hidden":128,"key":64}, "group":4,
    "target_accuracy":.95, "paired_target":.90, "accuracy_margin":.01, "resource_factor":2.,
    "comparison":"equal inputs/targets and fixed common recipe; different parameters, no iso-compute claim",
    "scope":"task-specific lag-one associative bias, not natural language or general LLM superiority"}
PROTOCOL_ID = hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def configuration(variant, output, seed, *, smoke=False, device="cuda"):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("unknown binding arm/seed")
    d,h,k = (16,24,8) if smoke else (96,128,64)
    model = ModelConfig(vocab_size=257,d_model=d,n_layers=1,n_heads=2 if smoke else 4,
                        hidden_dim=h,max_seq_len=64 if smoke else 8192,dropout=.05)
    research = ResearchConfig(architecture="transition_memory",scale_conditioning=False,
        memory_dim=k,associative_levels=2 if variant.startswith("pyramid") else 1,
        share_scales=variant!="pyramid_independent",state_factor=4)
    if variant == "transformer":
        research = ResearchConfig(architecture="transformer",scale_conditioning=False)
        if not smoke:model = ModelConfig(vocab_size=257,d_model=384,n_layers=2,n_heads=6,
            hidden_dim=384,max_seq_len=8192,dropout=.05)
    steps = 4 if smoke else 3000
    return ExperimentConfig(model=model,
        data=DataConfig(path=str(Path(output).resolve()/"procedural-data-not-materialized"),seq_len=32 if smoke else 256,
                        batch_size=2 if smoke else 16,eval_batch_size=1),
        optimizer=OptimizerConfig(lr=.001,weight_decay=.01,warmup_steps=1 if smoke else 50,schedule_steps=steps),
        train=TrainConfig(seed=seed,device=device,precision="fp32",deterministic=device=="cpu",compile=not smoke,
            max_steps=steps,grad_accum_steps=2 if smoke else 1,eval_interval=max(1,steps//4),
            checkpoint_interval=max(1,steps//2),log_interval=1 if smoke else 150,eval_batches=2,
            output_dir=str(Path(output).resolve()),cpu_threads=2,accumulation_weighting="tokens",console_progress=True),
        research=research)


def jobs(output,gpus,*,seeds=SEEDS,smoke=False,device="cuda"):
    result=[]
    for si,seed in enumerate(seeds):
        for vi,variant in enumerate(VARIANTS):
            name=f"{variant}-seed-{seed}";cfg=configuration(variant,Path(output)/"runs"/name,seed,smoke=smoke,device=device)
            result.append({"id":name,"model":variant,"variant":variant,"seed":seed,"directory":cfg.train.output_dir,
                "config_resolved":cfg.to_dict(),"gpu_slot":str(gpus[(vi+si)%2]),"status":"pending",
                "max_steps":cfg.train.max_steps,"batch_size":cfg.data.batch_size,
                "grad_accum_steps":cfg.train.grad_accum_steps,"checkpoint_interval":cfg.train.checkpoint_interval,
                "physical_gpu":None if device=="cpu" else int(gpus[(vi+si)%2])})
    return result
