"""Predeclared five-arm training screen, independent of the historical Stage B."""
import hashlib
import json
from pathlib import Path

from ..config import ExperimentConfig
from .state_protocol import configuration as original_configuration, statistics
from .cycle3_protocol import DATA_HASHES

ROOT=Path(__file__).resolve().parents[3]
VARIANTS=("hierarchy_next","hierarchy_immediate","recurrent","fine_only","transformer_two")
PAIRS=(("hierarchy_next","hierarchy_immediate"),("recurrent","fine_only"),("transformer_two",))
PROTOCOL={"name":"state-readout-v1","seed":251,"variants":VARIANTS,"steps":6000,"context":1024,
    "batch":8,"precision":"fp32","compiler":"inductor","dropout":.05,"lr":.0004,
    "lr_policy":"fixed initial recipe from0027; not individually tuned for new arms",
    "factor":4,"compression":"mean of contextualized fine states","data_files":DATA_HASHES,
    "full_validation_steps":[3000,6000],"subset_validation_steps":[1500,4500],
    "test_evaluated":False,"comparisons":"equal targets, not equal time/FLOPs/parameters",
    "primary":["hierarchy_immediate-minus-hierarchy_next","fine_only-minus-recurrent"],
    "quality_margin":.005,"loss_margin":.002,"time_margin":.10,
    "conclusion":"one-seed exploratory screen; no confirmation or general efficiency claim"}
PROTOCOL_ID=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def configuration(variant,data,output,*,smoke=False,device="cuda"):
    if variant not in VARIANTS:raise ValueError("unknown readout variant")
    mapped="hierarchy_independent" if variant.startswith("hierarchy_") else "recurrent" if variant=="fine_only" else variant
    cfg=original_configuration(mapped,"mlstm",data,output,251,smoke=smoke,device=device)
    if variant=="fine_only":cfg.model.n_layers=1
    if variant=="hierarchy_immediate":cfg.research.state_readout="immediate"
    return ExperimentConfig.from_dict(cfg.to_dict())


def jobs(data,output,gpus,*,smoke=False,device="cuda",variants=VARIANTS):
    if not variants or len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS):
        raise ValueError("select unique known variants")
    rows=[]
    for pair in PAIRS:
        for index,variant in enumerate(pair):
            if variant not in variants:continue
            name=f"{variant}-seed-251";folder=Path(output)/"runs"/name
            cfg=configuration(variant,data,folder,smoke=smoke,device=device)
            rows.append({"id":name,"variant":variant,"model":variant,"seed":251,"status":"pending",
                "gpu_slot":str(gpus[index]),"directory":str(folder),"config_resolved":cfg.to_dict(),
                "max_steps":cfg.train.max_steps,"progress_interval":cfg.train.log_interval,
                "batch_size":cfg.data.batch_size,"grad_accum_steps":cfg.train.grad_accum_steps,**statistics(cfg)})
    return rows
