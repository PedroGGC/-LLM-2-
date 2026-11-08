"""Causal vector cache on a two-rate recurrent hierarchy; no token self-attention."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .state_recurrent import SequenceState, StateRecurrentModel
from ..interfaces import ModelOutput
from ..model import RMSNorm, TransformerLM


def fractions(pattern, count):
    if pattern == "uniform": values = list(range(count))
    elif pattern == "binary": values = [2**i for i in range(count)]
    elif pattern == "fibonacci":
        values = [1, 2]
        while len(values) < count: values.append(values[-1]+values[-2])
    else: raise ValueError("sparse pattern required")
    return torch.tensor([(v-values[0])/(values[-1]-values[0]) for v in values], dtype=torch.float64)


def sparse_indices(completed, slots, reads, spacing):
    """Absolute zero-based bank positions; same cardinality and endpoints per row."""
    n = completed.clamp(max=slots)
    k = n.clamp(max=reads)
    i = torch.arange(reads, device=completed.device)
    offsets = 1+i + torch.floor(spacing.to(completed.device)*(n-k)[:,None]).long()
    valid = i[None,:] < k[:,None]
    return (completed[:,None]-offsets).clamp(min=0), valid


@dataclass
class EpisodicState:
    sequence: SequenceState
    bank: torch.Tensor


class EpisodicHierarchy(StateRecurrentModel):
    architecture_version = 3

    def __init__(self, config, research):
        super().__init__(config, research)
        self.architecture_version = 3
        if not self.hierarchy or research.memory_pattern == "none" or research.state_readout != "immediate":
            raise ValueError("episodic cache requires immediate hierarchy")
        d, m = config.d_model, research.memory_dim
        self.dense_memory = research.memory_pattern == "all"
        self.capacity, self.reads = research.memory_slots, research.memory_reads
        self.memory_norm = RMSNorm(d, config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.query = nn.Linear(d,m,bias=False); self.key = nn.Linear(d,m,bias=False)
        self.value = nn.Linear(d,m,bias=False); self.readout = nn.Linear(m,d,bias=False)
        self.read_gate = nn.Linear(d,1)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(8.)))
        for layer in (self.query,self.key,self.value,self.readout,self.read_gate):layer.apply(TransformerLM._init_weights)
        nn.init.constant_(self.read_gate.bias, -1.)
        self.register_buffer("spacing", fractions("uniform" if self.dense_memory else research.memory_pattern,self.reads),persistent=False)

    def forward(self, input_ids, *, state=None, return_logits=True):
        if input_ids.ndim != 2 or not 0<input_ids.shape[1]<=self.config.max_seq_len:
            raise ValueError("invalid input shape/length")
        b, t = input_ids.shape; d = self.config.d_model
        if state is not None:
            if not isinstance(state,EpisodicState) or self.training and self.config.dropout:
                raise ValueError("invalid episodic state or active streaming dropout")
            if (state.bank.ndim != 3 or state.bank.shape[0] != b or state.bank.shape[2] != d or
                    state.bank.shape[1]>self.capacity or state.bank.device!=input_ids.device or state.bank.dtype!=torch.float32):
                raise ValueError("incompatible episodic bank")
        seq = None if state is None else state.sequence
        memories = (None,None) if seq is None else seq.memories
        if len(memories)!=2:raise ValueError("two temporal states required")
        for memory in memories:
            if memory is not None:
                matrix = memory[0] if isinstance(memory,tuple) else memory
                if matrix.shape != (b,self.config.n_heads,d//self.config.n_heads,d//self.config.n_heads):
                    raise ValueError("incompatible temporal state")
        fine, first = self.cells[0](self.embedding(input_ids),memories[0])
        pending = fine[:,:0] if seq is None else seq.pending
        previous = torch.zeros_like(fine[:,0]) if seq is None else seq.last_coarse
        if pending is None or previous is None or pending.shape[0]!=b or pending.shape[1]>=self.factor:
            raise ValueError("invalid pending hierarchy group")
        offset = pending.shape[1]; joined = torch.cat((pending,fine),1)
        complete = joined.shape[1]//self.factor; second=memories[1]
        old = fine[:,:0] if state is None else state.bank
        if complete:
            groups=joined[:,:complete*self.factor].reshape(b,complete,self.factor,d).float().mean(2)
            coarse,second=self.cells[0 if self.shared else 1](groups,second)
            candidates=torch.cat((previous[:,None],coarse),1); last=coarse[:,-1].clone()
        else:coarse=fine[:,:0];candidates=previous[:,None];last=previous
        completed=(offset+torch.arange(t,device=input_ids.device)+1)//self.factor
        base=fine+candidates[:,completed]
        bank=torch.cat((old,coarse),1)
        count=completed+old.shape[1]
        # A sentinel makes prefixes with no completed group safe and exactly zero.
        encoded=self.memory_norm(torch.cat((fine.new_zeros(b,1,d),bank),1))
        q=F.normalize(self.query(self.memory_norm(fine)).float(),dim=-1,eps=1e-6)
        keys=F.normalize(self.key(encoded).float(),dim=-1,eps=1e-6);values=self.value(encoded).float()
        temperature=self.log_temperature.clamp(max=math.log(100.)).exp()
        if self.dense_memory:
            positions=torch.arange(bank.shape[1]+1,device=input_ids.device)
            valid=(positions[None,:]>0)&(positions[None,:]<=count[:,None])&(positions[None,:]>count[:,None]-self.capacity)
            scores=(q@keys.transpose(-2,-1))*temperature
            weights=scores.masked_fill(~valid,-1e4).softmax(-1)*valid
            read=weights@values
        else:
            indices,valid=sparse_indices(count,self.capacity,self.reads,self.spacing)
            indices=torch.where(valid,indices+1,0)
            chosen_k=keys[:,indices];chosen_v=values[:,indices]
            scores=(q[:,:,None,:]*chosen_k).sum(-1)*temperature
            weights=scores.masked_fill(~valid,-1e4).softmax(-1)*valid
            read=(weights[:,:,:,None]*chosen_v).sum(-2)
        mixed=base+self.read_gate(fine).sigmoid()*self.dropout(self.readout(read))
        next_seq=SequenceState((first,second),joined[:,complete*self.factor:].clone(),last)
        next_state=EpisodicState(next_seq,bank[:,-self.capacity:].clone())
        hidden=self.norm(mixed)
        return ModelOutput(logits=self.lm_head(hidden) if return_logits else None,hidden_states=hidden,state=next_state)

    def parameter_formula(self):
        d,m=self.config.d_model,self.research.memory_dim
        return super().parameter_formula()+4*d*m+2*d+2

    def state_bytes_per_sequence(self):
        return super().state_bytes_per_sequence()+4*self.capacity*self.config.d_model
