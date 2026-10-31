"""New study only: document-contained windows and train-only tuning split."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..document_data import prepare_documents
from ..data import PreparedData, write_json


def spans(tokens,eos):
    ends=np.flatnonzero(tokens==eos)+1
    if not len(ends) or ends[-1]!=len(tokens):
        raise ValueError("document stream must end in EOS")
    starts=np.concatenate(([0],ends[:-1]))
    return list(zip(starts.tolist(),ends.tolist()))


class DocumentBatchStream:
    def __init__(self,tokens,eos,seq_len,batch_size,seed):
        self.tokens,self.seq_len,self.batch_size=tokens,seq_len,batch_size
        eligible=[(a,b-a-seq_len) for a,b in spans(tokens,eos) if b-a>seq_len]
        if not eligible:raise ValueError("no document contains a complete training window")
        self.starts=np.array([a for a,_ in eligible],dtype=np.int64)
        self.counts=np.array([n for _,n in eligible],dtype=np.int64)
        self.cumulative=self.counts.cumsum();self.batches=0
        self.layout_hash=hashlib.sha256(self.starts.tobytes()+self.counts.tobytes()).hexdigest()
        self.generator=torch.Generator().manual_seed(seed)

    def next(self,device="cpu"):
        draws=torch.randint(int(self.cumulative[-1]),(self.batch_size,),generator=self.generator).numpy()
        docs=np.searchsorted(self.cumulative,draws,side="right")
        before=np.where(docs>0,self.cumulative[np.maximum(docs-1,0)],0)
        starts=self.starts[docs]+draws-before
        packed=torch.from_numpy(np.stack([self.tokens[a:a+self.seq_len+1] for a in starts]).astype(np.int64))
        self.batches+=1
        return {"input_ids":packed[:,:-1].to(device),"targets":packed[:,1:].to(device)}

    def state_dict(self):
        return {"kind":"document_windows_v1","rng":self.generator.get_state(),"batches":self.batches,
            "layout_hash":self.layout_hash,"seq_len":self.seq_len,"batch_size":self.batch_size}

    def load_state_dict(self,value):
        expected=self.state_dict()
        if any(value.get(k)!=expected[k] for k in ("kind","layout_hash","seq_len","batch_size")):
            raise ValueError("incompatible document sampler")
        self.generator.set_state(value["rng"].cpu());self.batches=value["batches"]


def sampler_factory(data,cfg):
    return DocumentBatchStream(data.arrays["train"],data.tokenizer.eos_id,cfg.data.seq_len,
                               cfg.data.batch_size,cfg.train.seed+1)


def validation(data,seq_len,batch_size,*,device="cpu",max_batches=None):
    batch=[];emitted=0
    for begin,end in spans(data.arrays["val"],data.tokenizer.eos_id):
        for start in range(begin,end-seq_len,seq_len):
            batch.append(data.arrays["val"][start:start+seq_len+1])
            if len(batch)==batch_size:
                packed=torch.from_numpy(np.stack(batch).astype(np.int64)).to(device)
                targets=packed[:,1:].clone();targets[targets==data.tokenizer.eos_id]=-100
                yield {"input_ids":packed[:,:-1],"targets":targets}
                batch=[];emitted+=1
                if max_batches is not None and emitted>=max_batches:return
    if batch:
        packed=torch.from_numpy(np.stack(batch).astype(np.int64)).to(device)
        targets=packed[:,1:].clone();targets[targets==data.tokenizer.eos_id]=-100
        yield {"input_ids":packed[:,:-1],"targets":targets}


def tuning_dataset(source,output):
    """Deterministic 80/10/10 split of original TRAIN; no official test evaluation."""
    source=PreparedData(source);output=Path(output)
    if source.tokenizer.kind!="byte":raise ValueError("train-only tuning preparation requires byte tokenization")
    output.mkdir(parents=True,exist_ok=False)
    raw=output/"train_documents.jsonl";counts={"train":0,"val":0,"test":0}
    original=source.manifest["splits"]["train"]["document_hashes"]
    documents=spans(source.arrays["train"],source.tokenizer.eos_id)
    if len(documents)<10:raise ValueError("at least ten training documents required for tuning")
    hashes=[]
    with raw.open("x",encoding="utf-8",newline="\n") as stream:
        for index,(a,b) in enumerate(documents):
            content=bytes(source.arrays["train"][a:b-1].tolist())
            digest=hashlib.sha256(content).hexdigest();hashes.append(digest)
            split="val" if index%10==8 else "test" if index%10==9 else "train"
            counts[split]+=1
            stream.write(json.dumps({"id":digest,"text":content.decode("utf-8"),"split":split},ensure_ascii=False)+"\n")
    if sorted(hashes)!=sorted(original):raise ValueError("training document bytes disagree with source metadata")
    prepare_documents([raw],output/"prepared",kind="byte")
    write_json(output/"provenance.json",{"source_dataset_sha256":source.fingerprint,"source_split":"train",
        "document_counts":counts,"test_role":"internal unused train-derived split, never official test",
        "source_train_document_hashes":hashes})
    raw.unlink()  # derived staging text; source/prepared data remain immutable
    return output/"prepared"
