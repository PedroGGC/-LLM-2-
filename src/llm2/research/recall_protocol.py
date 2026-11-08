"""Frozen scientific screen: learned vector cache; no WikiText download."""
import hashlib
import json
from pathlib import Path
from ..config import ExperimentConfig,ResearchConfig
from .state_protocol import configuration as state_config,statistics
from .recall_data import TASK,TASK_ID

VARIANTS=("transformer","recurrent","hierarchy","memory_all","memory_uniform","memory_binary","memory_fibonacci")
PAIRS=(("transformer","memory_all"),("recurrent","memory_fibonacci"),("memory_uniform","memory_binary"),("hierarchy",))
SEEDS=(271,272)
PROTOCOL={"name":"learned-vector-memory-recall-v1","task":TASK,"task_sha256":TASK_ID,"seeds":SEEDS,
    "variants":VARIANTS,"steps":3000,"context":1024,"batch":8,"lr":.0004,"dropout":.05,"precision":"fp32",
    "slots":2048,"reads":16,"memory_dim":96,"evaluation_lengths":[1024,2048,4096,8192],
    "evaluation_examples":128,"capacity_writes":[8,32,64],"validation_seed":91001,"test_seed":92001,
    "periodic_steps":[750,1500,2250,3000],"pattern":"rescaled gap fractions plus mandatory distinct integer offsets; not literal Fibonacci",
    "comparison":"equal training inputs/targets, not equal compute; dense bank reads more than sparse controls",
    "target_accuracy":.95,"accuracy_margin":.01,"resource_factor":2.,"claim":"two-seed exploratory task-specific screen; not language modeling or general LLM superiority"}
PROTOCOL_ID=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def configuration(variant,output,seed,*,smoke=False,device="cuda"):
    if variant not in VARIANTS or seed not in SEEDS:raise ValueError("unknown recall arm/seed")
    mapped="transformer_two" if variant=="transformer" else "recurrent" if variant=="recurrent" else "hierarchy_shared"
    c=state_config(mapped,"mlstm","procedural:"+TASK_ID,output,seed,smoke=smoke,device=device)
    # load_config resolves relative paths against its YAML file. Keep this inert
    # data placeholder absolute so metadata equals the declared configuration.
    c.data.path=str(Path(output).resolve()/"procedural-data-not-materialized")
    c.model.vocab_size=257;c.model.max_seq_len=64 if smoke else 8192
    c.data.seq_len=32 if smoke else 1024;c.data.eval_batch_size=1
    c.train.max_steps=c.optimizer.schedule_steps=4 if smoke else 3000
    c.train.eval_interval=max(1,c.train.max_steps//4);c.train.checkpoint_interval=max(1,c.train.max_steps//2)
    c.train.log_interval=1 if smoke else 150;c.train.eval_batches=2 if smoke else 8
    c.optimizer.warmup_steps=1 if smoke else 50
    if variant not in ("transformer","recurrent"):c.research.state_readout="immediate"
    if variant.startswith("memory_"):
        c.research.memory_pattern=variant.removeprefix("memory_")
        c.research.memory_dim=8 if smoke else 96
        c.research.memory_reads=4 if smoke else 16
        c.research.memory_slots=16 if smoke else 2048
    return ExperimentConfig.from_dict(c.to_dict())


def jobs(output,gpus,*,seeds=SEEDS,smoke=False,device="cuda"):
    rows=[]
    for si,seed in enumerate(seeds):
        if seed not in SEEDS:raise ValueError("seed outside frozen screen")
        for pair in PAIRS:
            for index,v in enumerate(pair):
                name=f"{v}-seed-{seed}";cfg=configuration(v,output/"runs"/name,seed,smoke=smoke,device=device)
                rows.append({"id":name,"variant":v,"model":v,"seed":seed,"gpu_slot":str(gpus[(index+si)%2]),
                    "directory":cfg.train.output_dir,"config_resolved":cfg.to_dict(),"status":"pending",
                    "batch_size":cfg.data.batch_size,"grad_accum_steps":cfg.train.grad_accum_steps,
                    "physical_gpu":None if device=="cpu" else int(gpus[(index+si)%2]),"checkpoint_interval":cfg.train.checkpoint_interval,
                    "max_steps":cfg.train.max_steps,"progress_interval":cfg.train.log_interval,**statistics(cfg)})
    return rows
