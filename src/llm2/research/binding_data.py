"""Fresh unique key/value maps with counterfactual query-key interventions."""
import copy
import hashlib
import json
import numpy as np
import torch

TASK = {"name": "byte-binding-v2", "vocab": 257, "keys": [0,63], "values": [64,127],
        "write": 128, "query": 129, "blank": 130, "distractors": [131,255],
        "record_width": 4, "queries": 16, "unique_values": True,
        "policy": "unique keys and values per sequence; writes first half; queries second half; new mapping per example",
        "training": {"length": 256, "writes": [[0,4],[200,8],[400,16],[600,32]]},
        "intervention": "cycle queried key to a different written key; require both corresponding answers",
        "targets": "only the answer at query-key position; answers never supplied at query"}
TASK_ID = hashlib.sha256(json.dumps(TASK, sort_keys=True).encode()).hexdigest()


def generate(rng, batch, length, *, writes=32, queries=16, device="cpu"):
    slots = length//4
    if length%4 or not 2 <= writes <= 64 or queries < 1 or slots//2 < max(writes,queries):
        raise ValueError("invalid binding dimensions")
    x = rng.integers(131,256,size=(batch,length),dtype=np.int64)
    y = np.full_like(x,-100); distances = np.zeros_like(x)
    for row in range(batch):
        keys = rng.choice(64,writes,replace=False)
        values = rng.choice(np.arange(64,128),writes,replace=False)
        positions = rng.choice(slots//2,writes,replace=False)
        for key,value,slot in zip(keys,values,positions):
            x[row,slot*4:slot*4+4] = [128,key,value,130]
        qslots = rng.choice(np.arange(slots//2,slots),queries,replace=False)
        which = rng.integers(0,writes,queries)
        for slot,index in zip(qslots,which):
            p = slot*4; x[row,p:p+4] = [129,keys[index],130,130]
            y[row,p+1] = values[index]; distances[row,p+1] = (slot-int(positions[index]))*4
    return {"input_ids": torch.from_numpy(x).to(device), "targets": torch.from_numpy(y).to(device),
            "distances": torch.from_numpy(distances).to(device)}


def intervene(batch):
    """Evaluation-only task parser; its output is never used to write model state."""
    result = {k:v.clone() for k,v in batch.items()}
    # CPU construction avoids per-token GPU synchronization in the benchmark.
    x = batch["input_ids"].cpu().numpy(); y = batch["targets"].cpu().numpy()
    changed = result["input_ids"].cpu().numpy().copy()
    targets = result["targets"].cpu().numpy().copy()
    dist = result["distances"].cpu().numpy().copy()
    for row in range(len(x)):
        table = {int(x[row,p+1]):(int(x[row,p+2]),int(p)) for p in np.flatnonzero(x[row]==128)}
        keys = list(table); successors = dict(zip(keys,keys[1:]+keys[:1]))
        for p in np.flatnonzero(y[row] != -100):
            key = successors[int(x[row,p])]; value, origin = table[key]
            changed[row,p] = key; targets[row,p] = value; dist[row,p] = p-1-origin
    device = batch["input_ids"].device
    return {"input_ids": torch.from_numpy(changed).to(device), "targets": torch.from_numpy(targets).to(device),
            "distances": torch.from_numpy(dist).to(device)}


class BindingStream:
    def __init__(self, cfg):
        self.length, self.batch = cfg.data.seq_len,cfg.data.batch_size
        self.accumulation = cfg.train.grad_accum_steps
        self.smoke = self.length < 256
        self.rng = np.random.default_rng(cfg.train.seed+1); self.draws = 0
    def next(self, device):
        step = self.draws//self.accumulation
        writes = 2 if self.smoke else next(w for threshold,w in reversed(TASK["training"]["writes"]) if step >= threshold)
        value = generate(self.rng,self.batch,self.length,writes=writes,queries=min(16,self.length//8),device=device)
        self.draws += 1; return value
    def state_dict(self):
        return {"task": TASK_ID, "length": self.length, "batch": self.batch, "accumulation": self.accumulation,
                "draws": self.draws, "rng": copy.deepcopy(self.rng.bit_generator.state)}
    def load_state_dict(self, state):
        if any(state[k] != self.state_dict()[k] for k in ("task","length","batch","accumulation")):
            raise ValueError("incompatible binding sampler")
        self.draws = state["draws"]; self.rng.bit_generator.state = copy.deepcopy(state["rng"])
