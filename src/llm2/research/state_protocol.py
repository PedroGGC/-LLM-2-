"""Independent protocol: rules in a common shell; no preselected family."""
import hashlib
import json
from pathlib import Path
import torch
from ..config import ExperimentConfig, load_config, ResearchConfig
from ..model import count_parameters
from .factory import build_backbone
from .cycle3_protocol import DATA_HASHES, verify_dataset

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (241,242,243)
VARIANTS = ("recurrent","hierarchy_independent","hierarchy_shared","transformer_two")
RULES = ("gated_delta","rwkv7","mlstm")
PROTOCOL = {"name":"recurrent-fractal-v1","seeds":SEEDS,"variants":VARIANTS,
    "implementation_scope":"common_shell_memory_rules","dataset_files":DATA_HASHES,
    "context":1024,"batch":8,"steps":6000,"learning_rates":[.0002,.0004],
    "tuning_seed":901,"tuning_steps":750,"tuning_split":"document-disjoint subset of train",
    "full_validation_fractions":[.5,1.],"subset_validation_fractions":[.25,.75],
    "sampling":"document-contained windows; reset per window",
    "evaluation":"document-contained full windows; EOS excluded from byte NLL; tails excluded",
    "precision":"fp32","compiler":"inductor","factor":4,"dropout":.05,
    "test_evaluated":False,"flops_estimate":None,
    "comparisons":"equal targets; wall time/parameters differ; not iso-compute",
    "quality_margin":.005,"noninferiority_margin":.002,"efficiency_margin":.10}
PROTOCOL_ID = hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def configuration(variant,rule,data,output,seed,*,lr=.0004,smoke=False,device="cuda",steps=None):
    if variant not in VARIANTS or rule not in RULES:
        raise ValueError("unknown state study variant/rule")
    cfg = load_config(ROOT/"configs/experiments/cycle3/base.yaml")
    cfg.research = ResearchConfig(architecture="transformer" if variant=="transformer_two" else (
        "state_recurrent" if variant=="recurrent" else "state_hierarchy"),
        state_rule=rule,share_scales=variant=="hierarchy_shared",scale_conditioning=False)
    cfg.data.path = str(data)
    cfg.data.seq_len = cfg.model.max_seq_len = 17 if smoke else 1024
    cfg.data.batch_size = cfg.data.eval_batch_size = 2 if smoke else 8
    cfg.model.n_layers = 2; cfg.model.dropout = .05
    cfg.model.d_model = 32 if smoke else 384; cfg.model.n_heads = 4 if smoke else 6
    cfg.model.hidden_dim = 64 if smoke else (128 if rule=="rwkv7" and variant!="transformer_two" else 384)
    cfg.train.device=device; cfg.train.precision="fp32";cfg.train.compile=device=="cuda"
    cfg.train.deterministic=device=="cpu";cfg.train.seed=seed;cfg.train.cpu_threads=2
    cfg.train.max_steps = cfg.optimizer.schedule_steps = steps or (4 if smoke else 6000)
    cfg.optimizer.warmup_steps=1 if smoke else min(50,cfg.train.max_steps//4)
    cfg.optimizer.lr=lr;cfg.train.output_dir=str(output)
    cfg.train.grad_accum_steps=2 if smoke else 1;cfg.train.accumulation_weighting="tokens"
    cfg.train.eval_interval=max(1,cfg.train.max_steps//4)
    cfg.train.checkpoint_interval=max(1,cfg.train.max_steps//2)
    cfg.train.eval_batches=8;cfg.train.log_interval=max(1,cfg.train.max_steps//20)
    cfg.train.console_progress=True
    return ExperimentConfig.from_dict(cfg.to_dict())


def statistics(cfg):
    with torch.device("meta"):model=build_backbone(cfg)
    count=count_parameters(model)
    if hasattr(model,"parameter_formula") and model.parameter_formula()!=count:
        raise ValueError("parameter formula disagrees with enumeration")
    return {"parameters":count,"weight_bytes":count*4,
        "state_bytes_per_sequence":model.state_bytes_per_sequence() if hasattr(model,"state_bytes_per_sequence") else None,
        "flops_estimate":None,"flops_estimator_available":not cfg.research.architecture.startswith("state_")}


def validate_parameter_budgets(rule,*,smoke=False):
    stats={v:statistics(configuration(v,rule,"data","unused",SEEDS[0],smoke=smoke,device="cpu")) for v in VARIANTS}
    anchor=stats["transformer_two"]["parameters"]
    if not smoke and any(abs(stats[v]["parameters"]/anchor-1)>.05 for v in VARIANTS[:2]):
        raise ValueError("unshared controls exceed the 5% parameter tolerance")
    if stats["hierarchy_shared"]["parameters"]>=stats["hierarchy_independent"]["parameters"]:
        raise ValueError("sharing did not reduce unique parameters")
    return stats


def ordered_jobs(rule,data,output,gpus,*,learning_rates=None,smoke=False,device="cuda",seeds=SEEDS):
    if not seeds or len(set(seeds))!=len(seeds) or not set(seeds).issubset(SEEDS):
        raise ValueError("select unique protocol seeds")
    stats=validate_parameter_budgets(rule,smoke=smoke);jobs=[]
    for seed in seeds:
        index=SEEDS.index(seed);order=VARIANTS[index:]+VARIANTS[:index]
        for variant in order:
            name=f"{variant}-seed-{seed}";folder=Path(output)/"runs"/name
            cfg=configuration(variant,rule,data,folder,seed,lr=(learning_rates or {}).get(variant,.0004),smoke=smoke,device=device)
            jobs.append({"id":name,"variant":variant,"model":variant,"seed":seed,"status":"pending",
                "directory":str(folder),"config_resolved":cfg.to_dict(),"max_steps":cfg.train.max_steps,
                "gpu_slot":str(gpus[(VARIANTS.index(variant)+index)%len(gpus)]),
                "progress_interval":cfg.train.log_interval,"registered_parameters":stats[variant]["parameters"],
                "batch_size":cfg.data.batch_size,"grad_accum_steps":cfg.train.grad_accum_steps,
                **stats[variant]})
    return jobs
