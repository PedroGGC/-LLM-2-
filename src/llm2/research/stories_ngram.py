"""Train-only count baseline; diagnostic quality, not a size-matched neural model."""
import math
import time
import numpy as np
from .stories_data import evaluation_batches


def evaluate_ngram(data,length=256):
    started=time.monotonic();raw=np.asarray(data.arrays['train'],dtype=np.uint32)
    previous=np.roll(raw,1);previous2=np.roll(raw,2)
    for row in data.docs['train']:
        start=row['offset'];previous[start]=previous2[start]=256
        if row['bytes']>1:previous2[start+1]=256
    ids=(previous2*257+previous)*257+raw
    counts=np.bincount(ids.astype(np.int64),minlength=257**3).astype(np.uint32).reshape(257,257,257)
    del ids,raw,previous,previous2
    counts2=counts.sum(0,dtype=np.uint64);counts1=counts2.sum(0,dtype=np.uint64)
    denom3=counts.sum(2,dtype=np.uint64);denom2=counts2.sum(1,dtype=np.uint64)
    loss=0.;n=0
    for pack in evaluation_batches(data,'val',length,32,'cpu'):
        x=pack['input_ids'].numpy();y=pack['targets'].numpy();mask=y!=-100
        p2=np.concatenate((np.full((len(x),1),256,dtype=np.int64),x[:,:-1]),1)
        a,b,c=p2[mask],x[mask],y[mask]
        one=(counts1[c]+.5)/(counts1.sum()+257*.5)
        two=(counts2[b,c]+10*one)/(denom2[b]+10)
        three=(counts[a,b,c]+10*two)/(denom3[a,b]+10)
        loss-=float(np.log(three).sum());n+=len(c)
    if n!=data.manifest['splits']['val']['bytes']:raise ValueError('ngram coverage mismatch')
    return {'name':'counting_trigram','nll':loss/n,'bits_per_byte':loss/n/math.log(2),'validation_bytes':n,
        'training_bytes':int(counts1.sum()),'count_state_bytes':counts.nbytes+counts2.nbytes+counts1.nbytes+denom3.nbytes+denom2.nbytes,
        'nonzero_triples':int(np.count_nonzero(counts)),'wall_seconds':time.monotonic()-started,
        'scope':'train-only smoothed counts, fixed backoff10 and prior0.5; same document/window boundaries; not a neural parameter match'}
