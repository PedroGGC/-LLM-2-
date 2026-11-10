"""Pinned, bounded TinyStories prefixes and document-isolated byte windows."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import uuid
from urllib.request import Request,urlopen
import numpy as np
import torch
from ..data import TextTokenizer,write_json,sha256_file

REVISION='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
PREFIXES={
 'TinyStories-train.txt':(32*1024**2,'ecd780925291f61ec90c35351e0ff44e80e6925dab8eb6f8565207ff73d41c23'),
 'TinyStories-valid.txt':(2*1024**2,'734193bdb018bb706a5af89c8b423d5ebcbc170c0a6601239baedd4b62450cb7')}
SPEC={'name':'tinystories-byte-doc-v1','repo':'roneneldan/TinyStories','revision':REVISION,
 'prefixes':PREFIXES,'train_bytes':16*1024**2,'cal_val_bytes':512*1024,'val_bytes':512*1024,'test_bytes':512*1024,
 'policy':'complete records ending in <|endoftext|>; strip CR/LF at record edges; hash order; validation priority; exact global dedup',
 'training':'one document per window; uniform valid starts; BOS256; real byte targets only',
 'evaluation':'every byte once including document tails; reset per window; test reserved'}
SPEC_ID=hashlib.sha256(json.dumps(SPEC,sort_keys=True).encode()).hexdigest()
# Content-only manifest identity, fixed after independent local preparation.
PINNED_FINGERPRINT='d61615e79496f14ef473cc10ad11e5deac6d2bda89be7046d738498ff16ba849'


def download_prefix(cache,name,*,budget=None,opener=urlopen):
    cache=Path(cache).resolve();cache.mkdir(parents=True,exist_ok=True)
    size,digest=PREFIXES[name]
    for path in sorted(cache.glob(digest+'*.blob')):
        if not path.is_symlink() and path.stat().st_size==size and sha256_file(path)==digest:return path
    target=cache/(digest+'-'+uuid.uuid4().hex[:8]+'.blob')
    temporary=cache/('.'+uuid.uuid4().hex+'.part')
    request=Request(f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/{name}',
                    headers={'Range':f'bytes=0-{size-1}','Accept-Encoding':'identity'})
    try:
        if budget:budget.require()
        with opener(request,timeout=60) as response,temporary.open('xb') as out:
            status=getattr(response,'status',200)
            if status not in (200,206):raise ValueError('unexpected prefix response status')
            if status==206 and not response.headers.get('Content-Range','').startswith('bytes 0-'):
                raise ValueError('server returned wrong prefix range')
            remaining=size
            while remaining:
                if budget:budget.require()
                block=response.read(min(1024**2,remaining))
                if not block:raise ValueError('truncated TinyStories prefix')
                out.write(block);remaining-=len(block)
        if temporary.stat().st_size!=size or sha256_file(temporary)!=digest:raise ValueError('TinyStories pinned prefix checksum mismatch')
        temporary.rename(target);return target
    finally:
        if temporary.exists():temporary.unlink()


def parse_prefix(raw):
    documents={};duplicates=0
    # The final unterminated record is deliberately excluded.
    for chunk in raw.split(b'<|endoftext|>')[:-1]:
        chunk=chunk.strip(b'\r\n')
        if not chunk.strip():continue
        chunk.decode('utf-8',errors='strict')
        key=hashlib.sha256(chunk).hexdigest()
        duplicates+=key in documents;documents[key]=chunk
    return documents,duplicates


def take(documents,budget):
    selected={};total=0
    for key in sorted(documents):
        if total>=budget:break
        selected[key]=documents[key];total+=len(documents[key])
    if total<budget:raise ValueError('pinned prefix lacks requested complete documents')
    return selected


def select(train_raw,valid_raw):
    train,dupt=parse_prefix(train_raw);valid,dupv=parse_prefix(valid_raw)
    overlap=set(train)&set(valid)
    for key in overlap:train.pop(key)
    val=take(valid,SPEC['val_bytes'])
    test=take({k:v for k,v in valid.items() if k not in val},SPEC['test_bytes'])
    cal=take(train,SPEC['cal_val_bytes'])
    selected=take({k:v for k,v in train.items() if k not in cal},SPEC['train_bytes'])
    return {'train':selected,'cal_val':cal,'val':val,'test':test},{'within_train_removed':dupt,'within_validation_removed':dupv,
        'cross_source_removed_from_train':len(overlap),'validation_prefix_has_no_official_test':'reserved test derived from official validation subset'}


def write_dataset(output,splits,*,spec=SPEC,selection=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('immutable story dataset already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    owners=set();manifest={'schema':'stories-doc-bytes-v1','spec':spec,'selection':selection or {},
        'vocab_size':257,'bos_id':256,'dtype':'uint8','splits':{},'files':{}}
    for split,docs in splits.items():
        if not docs:raise ValueError('empty story split')
        for key,raw in docs.items():
            if key!=hashlib.sha256(raw).hexdigest() or key in owners:raise ValueError('story duplicate/leakage or content mismatch')
            owners.add(key)
    with tempfile.TemporaryDirectory(prefix='.stories-',dir=output.parent) as staging:
        stage=Path(staging);TextTokenizer().save(stage/'tokenizer.json')
        for split,docs in splits.items():
            offset=0;rows=[]
            with (stage/(split+'.bin')).open('wb') as stream:
                for key in sorted(docs):
                    raw=docs[key];raw.decode('utf-8',errors='strict');stream.write(raw)
                    rows.append({'sha256':key,'offset':offset,'bytes':len(raw)});offset+=len(raw)
            manifest['splits'][split]={'bytes':offset,'documents':rows}
        manifest['files']={name:sha256_file(stage/name) for name in ['tokenizer.json',*(s+'.bin' for s in splits)]}
        write_json(stage/'manifest.json',manifest)
        # Copy into a unique empty directory, then atomic rename; staging is owned.
        stage.rename(output)
    return StoriesData(output,require_pinned=False)


class StoriesData:
    def __init__(self,root,*,require_pinned=True):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        self.fingerprint=sha256_file(self.root/'manifest.json');m=self.manifest
        if m['schema']!='stories-doc-bytes-v1' or m['vocab_size']!=257 or m['bos_id']!=256 or m['dtype']!='uint8':raise ValueError('invalid stories schema')
        if set(m['splits'])!={'train','cal_val','val','test'} or set(m['files'])!={'tokenizer.json','train.bin','cal_val.bin','val.bin','test.bin'}:
            raise ValueError('invalid story splits/files')
        if require_pinned and (m['spec']!=json.loads(json.dumps(SPEC)) or self.fingerprint!=PINNED_FINGERPRINT):
            raise ValueError('prepared stories do not match pinned content')
        self.tokenizer=TextTokenizer.load(self.root/'tokenizer.json');self.arrays={};self.docs={};owners=set()
        if self.tokenizer.vocab_size!=257 or self.tokenizer.kind!='byte':raise ValueError('stories tokenizer mismatch')
        for name,digest in m['files'].items():
            if (self.root/name).is_symlink() or sha256_file(self.root/name)!=digest:raise ValueError('story artifact checksum mismatch')
        for split,record in m['splits'].items():
            array=np.memmap(self.root/(split+'.bin'),dtype=np.uint8,mode='r')
            if len(array)!=record['bytes']:raise ValueError('stories length mismatch')
            offset=0
            for row in record['documents']:
                key=row['sha256'];length=row['bytes']
                if row['offset']!=offset or length<1 or key in owners or hashlib.sha256(array[offset:offset+length].tobytes()).hexdigest()!=key:
                    raise ValueError('story document leakage/boundary/content mismatch')
                owners.add(key);offset+=length
            if offset!=len(array):raise ValueError('story offsets do not cover all bytes')
            self.arrays[split]=array;self.docs[split]=record['documents']


def prepare(cache,*,budget=None):
    cache=Path(cache).resolve();cache.mkdir(parents=True,exist_ok=True)
    for path in sorted(cache.glob('prepared-'+SPEC_ID+'*')):
        if path.is_symlink():continue
        try:return StoriesData(path)
        except (OSError,ValueError,KeyError):pass # Preserve unrecognized/corrupt material; create fresh owned output.
    raw={name:download_prefix(cache/'prefixes',name,budget=budget).read_bytes() for name in PREFIXES}
    splits,selection=select(raw['TinyStories-train.txt'],raw['TinyStories-valid.txt'])
    data=write_dataset(cache/('prepared-'+SPEC_ID+'-'+uuid.uuid4().hex[:8]),splits,selection=selection)
    if PINNED_FINGERPRINT is not None and data.fingerprint!=PINNED_FINGERPRINT:raise ValueError('new preparation differs from pinned TinyStories content')
    return data


def window(data,split,document,start,length):
    row=data.docs[split][document];raw=data.arrays[split][row['offset']:row['offset']+row['bytes']]
    if not 0<=start<len(raw):raise ValueError('invalid story target offset')
    n=min(length,len(raw)-start);x=np.full(length,256,dtype=np.int64);y=np.full(length,-100,dtype=np.int64)
    x[0]=256 if start==0 else int(raw[start-1])
    if n>1:x[1:n]=raw[start:start+n-1]
    y[:n]=raw[start:start+n]
    return x,y


class StoryStream:
    def __init__(self,data,cfg,*,calibration=False):
        self.data,self.length,self.batch=data,cfg.data.seq_len,cfg.data.batch_size
        self.calibration=calibration;rows=data.docs['train'];limit=len(rows)
        if calibration:
            total=0;limit=0
            for row in rows:
                total+=row['bytes'];limit+=1
                if total>=1024**2:break
        self.rows=rows[:limit];self.weights=np.array([max(1,r['bytes']-self.length+1) for r in self.rows],dtype=np.int64)
        self.cumulative=self.weights.cumsum();self.rng=np.random.default_rng(cfg.train.seed+1);self.draws=0
    def next(self,device):
        numbers=self.rng.integers(0,int(self.cumulative[-1]),size=self.batch);doc=np.searchsorted(self.cumulative,numbers,side='right')
        starts=numbers-np.where(doc>0,self.cumulative[np.maximum(0,doc-1)],0)
        windows=[window(self.data,'train',int(d),int(s),self.length) for d,s in zip(doc,starts)]
        self.draws+=1
        return {'input_ids':torch.from_numpy(np.stack([x for x,y in windows])).to(device),
                'targets':torch.from_numpy(np.stack([y for x,y in windows])).to(device)}
    def state_dict(self):
        return {'dataset':self.data.fingerprint,'length':self.length,'batch':self.batch,'calibration':self.calibration,
                'draws':self.draws,'rng':self.rng.bit_generator.state}
    def load_state_dict(self,state):
        if any(state[k]!=self.state_dict()[k] for k in ('dataset','length','batch','calibration')):raise ValueError('incompatible story sampler')
        self.draws=state['draws'];self.rng.bit_generator.state=state['rng']


def evaluation_batches(data,split,length,batch,device):
    if split not in ('val','cal_val'):raise ValueError('reserved test cannot be evaluated in this campaign')
    current=[]
    for i,row in enumerate(data.docs[split]):
        for start in range(0,row['bytes'],length):
            current.append(window(data,split,i,start,length))
            if len(current)==batch:
                yield {'input_ids':torch.from_numpy(np.stack([x for x,y in current])).to(device),
                       'targets':torch.from_numpy(np.stack([y for x,y in current])).to(device)};current=[]
    if current:
        yield {'input_ids':torch.from_numpy(np.stack([x for x,y in current])).to(device),
               'targets':torch.from_numpy(np.stack([y for x,y in current])).to(device)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);a=p.parse_args();data=prepare(a.cache)
    print('DATA',data.root,'FINGERPRINT',data.fingerprint)
    print({k:(v['bytes'],len(v['documents'])) for k,v in data.manifest['splits'].items()})


if __name__=='__main__':main()
