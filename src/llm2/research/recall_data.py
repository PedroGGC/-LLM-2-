"""Procedural byte IDs, not natural language; labels never enter query inputs."""
import copy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import torch

TASK={"name":"byte-associative-recall-v1","vocab":257,"keys":[0,63],"values":[64,127],
      "write":128,"query":129,"blank":130,"distractors":[131,255],"record_width":4,
      "writes":32,"queries":16,"policy":"unique random keys; independent random values per example; writes first half, queries second half",
      "targets":"one byte per query; no answers included in input","distance_policy":"uniform slot sampling, no preferred offsets"}
TASK_ID=hashlib.sha256(json.dumps(TASK,sort_keys=True).encode()).hexdigest()


class RecallData:
    """Trainer data contract without corpus files or misleading tokenized arrays."""
    def __init__(self,length):
        self.tokenizer=SimpleNamespace(vocab_size=257)
        self.arrays={k:np.zeros(length+1,dtype=np.uint16) for k in ("train","val")}
        self.fingerprint=TASK_ID


def generate(rng,batch,length,*,writes=32,queries=16,device="cpu"):
    slots=length//4
    if length%4 or writes>64 or min(writes,queries)<1 or slots//2<max(writes,queries):
        raise ValueError("invalid recall dimensions")
    x=rng.integers(131,256,size=(batch,length),dtype=np.int64)
    y=np.full_like(x,-100);dist=np.zeros_like(x)
    for row in range(batch):
        keys=rng.choice(64,writes,replace=False);vals=rng.integers(64,128,writes)
        positions=rng.choice(slots//2,writes,replace=False)
        for key,value,position in zip(keys,vals,positions):x[row,position*4:position*4+4]=[128,key,value,130]
        qslots=rng.choice(np.arange(slots//2,slots),queries,replace=False)
        which=rng.integers(0,writes,queries)
        for slot,index in zip(qslots,which):
            pos=slot*4
            x[row,pos:pos+4]=[129,keys[index],130,130]
            y[row,pos+1]=vals[index];dist[row,pos+1]=(slot-int(positions[index]))*4
    return {"input_ids":torch.from_numpy(x).to(device),"targets":torch.from_numpy(y).to(device),
            "distances":torch.from_numpy(dist).to(device)}


class RecallStream:
    def __init__(self,cfg):
        self.length,self.batch=cfg.data.seq_len,cfg.data.batch_size
        self.seed=cfg.train.seed+1;self.rng=np.random.default_rng(self.seed);self.examples=0
    def next(self,device):
        result=generate(self.rng,self.batch,self.length,writes=min(32,self.length//8),queries=min(16,self.length//8),device=device)
        self.examples+=self.batch;return result
    def state_dict(self):
        return {"task":TASK_ID,"length":self.length,"batch":self.batch,"examples":self.examples,
                "rng":copy.deepcopy(self.rng.bit_generator.state)}
    def load_state_dict(self,state):
        if any(state[k]!=self.state_dict()[k] for k in ("task","length","batch")):
            raise ValueError("incompatible recall sampler")
        self.rng.bit_generator.state=copy.deepcopy(state["rng"]);self.examples=state["examples"]
