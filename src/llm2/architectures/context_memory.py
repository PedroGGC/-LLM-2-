"""Causal gated associative state, contextual depth and shared multirate hierarchy.

Scalar decay per head; a GLA/retention-family adaptation, not delta rule or the
complete published models. No global softmax attention or DecoderBlock.
"""
import copy
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, SwiGLU, TransformerLM


def retention_sequential(q,k,v,log_decay,state=None):
    q,k,v,log_decay=(x.float() for x in (q,k,v,log_decay))
    memory=q.new_zeros(*q.shape[:2],q.shape[-1],v.shape[-1]) if state is None else state
    outputs=[]
    for t in range(q.shape[2]):
        memory=memory*log_decay[:,:,t].exp()[...,None,None]+k[:,:,t,:,None]*v[:,:,t,None,:]
        outputs.append((q[:,:,t,None,:]@memory).squeeze(-2))
    return torch.stack(outputs,2),memory


def retention_blocked(q,k,v,log_decay,state=None,*,block_size=32):
    """Local CxC products and stable affine prefix scan over block summaries.

    Hillis-Steele scan uses O(N log N) work for N blocks; streaming is constant
    state. All exponentials have nonpositive exponents for valid decays.
    """
    q,k,v,log_decay=(x.float() for x in (q,k,v,log_decay))
    b,h,t,d=q.shape;c=block_size;pad=(-t)%c
    def pack(x):return F.pad(x,(0,0,0,pad)).reshape(b,h,-1,c,x.shape[-1])
    qc,kc,vc=map(pack,(q,k,v));n=qc.shape[2]
    logs=F.pad(log_decay,(0,pad)).reshape(b,h,n,c).cumsum(-1)
    index=torch.arange(c,device=q.device);causal=index[:,None]>=index[None,:]
    relative=logs.unsqueeze(-1)-logs.unsqueeze(-2)
    decay=relative.masked_fill(~causal,0).exp().masked_fill(~causal,0)
    local=((qc@kc.transpose(-1,-2))*decay)@vc
    tail=(logs[...,-1:]-logs).exp()
    summary=kc.transpose(-1,-2)@(vc*tail[...,None])
    factors=logs[...,-1].exp()[...,None,None]
    inclusive=summary;stride=1
    while stride<n:
        inclusive=torch.cat((inclusive[:,:,:stride],inclusive[:,:,stride:]+factors[:,:,stride:]*inclusive[:,:,:-stride]),2)
        factors=torch.cat((factors[:,:,:stride],factors[:,:,stride:]*factors[:,:,:-stride]),2)
        stride*=2
    if state is not None:inclusive=inclusive+factors*state[:,:,None]
    initial=q.new_zeros(b,h,1,d,v.shape[-1]) if state is None else state[:,:,None]
    previous=torch.cat((initial,inclusive[:,:,:-1]),2)
    out=local+(qc@previous)*logs.exp()[...,None]
    return out.reshape(b,h,-1,v.shape[-1])[:,:,:t],inclusive[:,:,-1].clone()


@dataclass
class ContextCellState:
    memory: torch.Tensor
    tail: torch.Tensor


@dataclass
class ContextState:
    cells: tuple[ContextCellState,...]
    pending: torch.Tensor
    last_coarse: torch.Tensor


class ContextCell(nn.Module):
    def __init__(self,config,research):
        super().__init__();d=config.d_model;h=config.n_heads;k=research.memory_dim
        self.heads,self.key_dim,self.value_dim=h,k,d//h
        self.execution=research.state_execution;self.block=research.state_chunk_size
        self.eps=config.norm_eps
        self.input_norm=RMSNorm(d,self.eps);self.local_norm=RMSNorm(d,self.eps);self.ffn_norm=RMSNorm(d,self.eps)
        self.conv=nn.Conv1d(d,d,4,groups=d,bias=False)
        self.q=nn.Linear(d,h*k,bias=False);self.k=nn.Linear(d,h*k,bias=False);self.v=nn.Linear(d,d,bias=False)
        self.gates=nn.Linear(d,2*h);self.output_gate=nn.Linear(d,d,bias=False);self.output=nn.Linear(d,d,bias=False)
        self.ffn=SwiGLU(d,config.hidden_dim);self.dropout=nn.Dropout(config.dropout)
        self.apply(TransformerLM._init_weights);nn.init.normal_(self.conv.weight,std=.02)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            times=torch.logspace(math.log10(8),math.log10(512),h)
            self.gates.bias[:h].copy_(torch.log(torch.expm1(1/times)))
            self.gates.bias[h:].fill_(-1.5)

    def forward(self,x,state=None):
        b,t,d=x.shape;z=self.input_norm(x).float()
        tail=z.new_zeros(b,3,d) if state is None else state.tail
        joined=torch.cat((tail,z),1)
        local=self.local_norm(z+self.conv(joined.transpose(1,2)).transpose(1,2))
        def heads(value,width):return value.reshape(b,t,self.heads,width).transpose(1,2).float()
        q=F.normalize(heads(self.q(local),self.key_dim),dim=-1,eps=1e-6)
        k=F.normalize(heads(self.k(local),self.key_dim),dim=-1,eps=1e-6)
        v=heads(self.v(local),self.value_dim)
        forget,write=self.gates(local).float().chunk(2,-1)
        log_decay=-F.softplus(forget).transpose(1,2)
        k=k*write.sigmoid().transpose(1,2)[...,None]
        function=retention_blocked if self.execution=='chunked' else retention_sequential
        kwargs={'block_size':self.block} if self.execution=='chunked' else {}
        read,memory=function(q,k,v,log_decay,None if state is None else state.memory,**kwargs)
        read=read*torch.rsqrt(read.square().mean(-1,keepdim=True)+self.eps)
        read=read.transpose(1,2).reshape(b,t,d)*self.output_gate(local).float().sigmoid()
        hidden=x+self.dropout(self.output(read))
        hidden=hidden+self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden,ContextCellState(memory,joined[:,-3:].clone())


class ContextMemory(Backbone):
    architecture_version=1
    supports_hidden_only=True

    def __init__(self,config,research):
        super().__init__();self.config=config;self.research=research
        self.hierarchy=research.architecture=='context_hierarchy';self.depth=config.n_layers
        if self.hierarchy and self.depth!=2:raise ValueError('context hierarchy requires two levels')
        self.shared=research.share_weights
        self.embedding=nn.Embedding(config.vocab_size,config.d_model);self.embedding.apply(TransformerLM._init_weights)
        first=ContextCell(config,research)
        self.cells=nn.ModuleList([first] if self.shared else [first,*[copy.deepcopy(first) for _ in range(self.depth-1)]])
        self.norm=RMSNorm(config.d_model,config.norm_eps)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False);self.lm_head.weight=self.embedding.weight
        if self.hierarchy:self.merge=nn.Parameter(torch.full((config.d_model,),.1))

    def _validate_state(self,state,x):
        if not isinstance(state,ContextState) or len(state.cells)!=self.depth:raise ValueError('context state depth/type mismatch')
        if self.training and self.config.dropout:raise ValueError('streaming with active dropout unsupported')
        b,_,d=x.shape;h=self.config.n_heads;k=self.research.memory_dim
        expected=[(s.memory,(b,h,k,d//h)) for s in state.cells]+[(s.tail,(b,3,d)) for s in state.cells]
        expected.append((state.last_coarse,(b,d)))
        if state.pending.ndim!=3 or state.pending.shape[0]!=b or state.pending.shape[2]!=d or not 0<=state.pending.shape[1]<self.research.state_factor:
            raise ValueError('context pending group mismatch')
        expected.append((state.pending,tuple(state.pending.shape)))
        if not self.hierarchy and state.pending.shape[1]:raise ValueError('unexpected pending nonhierarchical state')
        if any(v.shape!=shape or v.dtype!=torch.float32 or v.device!=x.device for v,shape in expected):raise ValueError('context state shape/dtype/device mismatch')

    def forward(self,input_ids,*,state=None,return_logits=True):
        if input_ids.ndim!=2 or not 0<input_ids.shape[1]<=self.config.max_seq_len:raise ValueError('invalid context input')
        x=self.embedding(input_ids).float();b,t,d=x.shape
        if state is not None:self._validate_state(state,x)
        states=[]
        if not self.hierarchy:
            for i in range(self.depth):
                x,cell=self.cells[0 if self.shared else i](x,None if state is None else state.cells[i]);states.append(cell)
            pending=x.new_empty(b,0,d);last=x.new_zeros(b,d)
        else:
            fine,cell=self.cells[0](x,None if state is None else state.cells[0]);states.append(cell)
            old=fine[:,:0] if state is None else state.pending
            last=fine.new_zeros(b,d) if state is None else state.last_coarse
            joined=torch.cat((old,fine),1);factor=self.research.state_factor;count=joined.shape[1]//factor
            pending=joined[:,count*factor:].clone()
            coarse_state=None if state is None else state.cells[1]
            if count:
                pooled=joined[:,:count*factor].reshape(b,count,factor,d).mean(2)
                coarse,coarse_state=self.cells[0 if self.shared else 1](pooled,coarse_state)
            else:
                coarse=fine[:,:0]
                if coarse_state is None:
                    coarse_state=ContextCellState(fine.new_zeros(b,self.config.n_heads,self.research.memory_dim,d//self.config.n_heads),fine.new_zeros(b,3,d))
            states.append(coarse_state)
            index=(torch.arange(t,device=x.device)+old.shape[1]+1)//factor
            available=torch.cat((last[:,None],coarse),1)
            x=fine+self.merge*available[:,index]
            if count:last=coarse[:,-1].clone()
        hidden=self.norm(x)
        return ModelOutput(logits=self.lm_head(hidden) if return_logits else None,hidden_states=hidden,
            state=ContextState(tuple(states),pending,last))

    def parameter_formula(self):
        d=self.config.d_model;h=self.config.n_heads;k=self.research.memory_dim
        core=2*d*h*k+3*d*d+3*d*self.config.hidden_dim+7*d+2*h*d+2*h
        return self.config.vocab_size*d+d+len(self.cells)*core+(d if self.hierarchy else 0)
