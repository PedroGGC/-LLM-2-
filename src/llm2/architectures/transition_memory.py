"""Learned lag-one associative memory, with optional shared two-rate partition.

Inductive bias: a current value writes to the previous position's learned address.
No token IDs, task markers, labels or write masks are interpreted here. This is
linear associative memory, not mLSTM or an assertion of a novel attention rule.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, SwiGLU, TransformerLM


@dataclass
class TransitionState:
    totals: tuple[torch.Tensor, ...]
    partials: tuple[torch.Tensor, ...]
    previous: torch.Tensor
    local_tail: torch.Tensor
    phase: int


def associative_reads(q, k, v, total, partial, phase, group=4):
    """Exact additive recurrence via local causal reads and exclusive chunk sums."""
    b, t, width = q.shape
    pad = (-(phase+t)) % group
    def pack(value):
        value = F.pad(value, (0, 0, phase, pad))
        return value.reshape(b, -1, group, value.shape[-1])
    qc, kc, vc = map(pack, (q.float(), k.float(), v.float()))
    delta = kc.transpose(-2, -1) @ vc
    # Only the first chunk includes unfinished writes from a prior streaming call.
    first = (torch.arange(delta.shape[1], device=q.device) == 0).to(q.dtype)
    delta = delta + first[None, :, None, None]*partial[:, None]
    inclusive = delta.cumsum(1)
    exclusive = torch.cat((torch.zeros_like(inclusive[:, :1]), inclusive[:, :-1]), 1)
    prefix = total[:, None]-partial[:, None] + exclusive
    coarse = qc @ prefix
    causal = torch.ones(group, group, dtype=torch.bool, device=q.device).tril()
    fine = ((qc @ kc.transpose(-2, -1)).masked_fill(~causal, 0)) @ vc
    fine = fine + first[None, :, None, None]*(qc @ partial[:, None])
    def unpack(value):
        return value.reshape(b, -1, value.shape[-1])[:, phase:phase+t]
    next_total = total-partial+inclusive[:, -1]
    next_partial = delta[:, -1].clone() if (phase+t) % group else torch.zeros_like(partial)
    return unpack(fine), unpack(coarse), next_total, next_partial


class TransitionOperator(nn.Module):
    def __init__(self, dim, key_dim):
        super().__init__()
        self.address = nn.Linear(dim, key_dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.write_gate = nn.Linear(dim, 1)
        self.readout = nn.Linear(dim, dim, bias=False)
        self.apply(TransformerLM._init_weights)
        nn.init.constant_(self.write_gate.bias, -2.)

    def encode(self, addresses, previous, local):
        q = F.normalize(self.address(addresses).float(), dim=-1, eps=1e-6)
        old = torch.cat((previous[:, None], addresses[:, :-1]), 1)
        k = F.normalize(self.address(old).float(), dim=-1, eps=1e-6)
        gate = self.write_gate(local).float().sigmoid()
        return q, k*gate, self.value(local).float()


def blocked_associative_reads(q,k,v,total,partial,phase,group=4,block=32):
    """Same logical group partition; matrix prefixes only per larger compute block."""
    b,t,width=q.shape
    pad=(-(phase+t))%block
    def pack(value):return F.pad(value.float(),(0,0,phase,pad)).reshape(b,-1,block,value.shape[-1])
    qc,kc,vc=map(pack,(q,k,v))
    first=(torch.arange(qc.shape[1],device=q.device)==0).to(q.dtype)
    delta=kc.transpose(-2,-1)@vc+first[None,:,None,None]*partial[:,None]
    inclusive=delta.cumsum(1)
    exclusive=torch.cat((torch.zeros_like(inclusive[:,:1]),inclusive[:,:-1]),1)
    prefix=total[:,None]-partial[:,None]+exclusive
    index=torch.arange(block,device=q.device);groups=index//group
    similarity=qc@kc.transpose(-2,-1)
    fine_mask=(groups[:,None]==groups[None,:])&(index[:,None]>=index[None,:])
    coarse_mask=groups[:,None]>groups[None,:]
    previous_read=qc@partial[:,None]
    first_fine=first[None,:,None,None]*(groups==0)[None,None,:,None]
    first_coarse=first[None,:,None,None]*(groups>0)[None,None,:,None]
    fine=similarity.masked_fill(~fine_mask,0)@vc+first_fine*previous_read
    coarse=qc@prefix+similarity.masked_fill(~coarse_mask,0)@vc+first_coarse*previous_read
    def unpack(value):return value.reshape(b,-1,value.shape[-1])[:,phase:phase+t]
    remainder=(phase+t)%group
    if remainder:
        start=phase+t-remainder
        flat_k=kc.reshape(b,-1,k.shape[-1]);flat_v=vc.reshape(b,-1,v.shape[-1])
        next_partial=flat_k[:,start:start+remainder].transpose(-2,-1)@flat_v[:,start:start+remainder]
        if start==0:next_partial=next_partial+partial
    else:next_partial=torch.zeros_like(partial)
    return unpack(fine),unpack(coarse),total-partial+inclusive[:,-1],next_partial


class TransitionMemory(Backbone):
    architecture_version = 1
    supports_hidden_only = True

    def __init__(self, config, research):
        super().__init__()
        if config.n_layers != 1:raise ValueError('transition_memory has one core; use associative_levels for its hierarchy')
        self.config, self.research = config, research
        self.levels, self.key_dim = research.associative_levels, research.memory_dim
        self.shared = research.share_scales or self.levels == 1
        self.group = research.state_factor
        self.architecture_version = 2 if research.associative_execution == "blocked" else 1
        d = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, d)
        self.embedding.apply(TransformerLM._init_weights)
        self.input_norm = RMSNorm(d, config.norm_eps)
        self.local_conv = nn.Conv1d(d, d, 3, groups=d, bias=False)
        nn.init.normal_(self.local_conv.weight, std=.02)
        self.local_norm = RMSNorm(d, config.norm_eps)
        self.local_output = SwiGLU(d, config.hidden_dim)
        self.local_output.apply(TransformerLM._init_weights)
        self.operators = nn.ModuleList([TransitionOperator(d, self.key_dim) for _ in range(1 if self.shared else 2)])
        if not self.shared:
            self.operators[1].load_state_dict(self.operators[0].state_dict())
        self.level_gate = nn.Linear(d, 2) if self.levels == 2 else None
        if self.level_gate is not None:
            nn.init.zeros_(self.level_gate.weight); nn.init.zeros_(self.level_gate.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(d, config.norm_eps)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _inputs(self, input_ids, state):
        b, t = input_ids.shape; d = self.config.d_model
        x = self.input_norm(self.embedding(input_ids)).float()
        if state is None:
            previous = x.new_zeros(b, d); tail = x.new_zeros(b, 2, d)
        else:
            previous, tail = state.previous, state.local_tail
        joined = torch.cat((tail, x), 1)
        local = self.local_norm(x + self.local_conv(joined.transpose(1, 2)).transpose(1, 2))
        return x, previous, local, joined[:, -2:].clone()

    def forward(self, input_ids, *, state=None, return_logits=True):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("invalid transition input shape/length")
        b = input_ids.shape[0]; d = self.config.d_model
        if state is not None:
            if not isinstance(state, TransitionState) or not 0 <= state.phase < self.group:
                raise ValueError("invalid transition state")
            if self.training and self.config.dropout:
                raise ValueError("streaming with active dropout unsupported")
            if len(state.totals) != len(self.operators) or len(state.partials) != len(self.operators):
                raise ValueError("transition state operator mismatch")
            expected = [(value, (b, self.key_dim, d)) for value in (*state.totals, *state.partials)]
            expected += [(state.previous, (b, d)), (state.local_tail, (b, 2, d))]
            if any(value.shape != shape or value.dtype != torch.float32 or value.device != input_ids.device for value, shape in expected):
                raise ValueError("transition state shape/dtype/device mismatch")
        x, previous, local, tail = self._inputs(input_ids, state)
        phase = 0 if state is None else state.phase
        totals = []; partials = []; reads = []
        for index, op in enumerate(self.operators):
            zero = x.new_zeros(b, self.key_dim, d)
            total = zero if state is None else state.totals[index]
            partial = zero if state is None else state.partials[index]
            q, k, v = op.encode(x, previous, local)
            if self.research.associative_execution == "blocked":
                fine,coarse,total,partial=blocked_associative_reads(q,k,v,total,partial,phase,self.group,self.research.associative_block_size)
            else:
                fine, coarse, total, partial = associative_reads(q, k, v, total, partial, phase, self.group)
            reads.append((fine, coarse)); totals.append(total); partials.append(partial)
        if self.levels == 1:
            memory = self.operators[0].readout(reads[0][0] + reads[0][1])
        else:
            gates = 2*self.level_gate(x).float().sigmoid()
            fine = self.operators[0].readout(reads[0][0])
            coarse = self.operators[0 if self.shared else 1].readout(reads[0 if self.shared else 1][1])
            memory = gates[..., :1]*fine + gates[..., 1:]*coarse
        hidden = self.norm(self.dropout(memory) + self.dropout(self.local_output(local)))
        next_state = TransitionState(tuple(totals), tuple(partials), x[:, -1].clone(), tail, (phase+input_ids.shape[1]) % self.group)
        return ModelOutput(logits=self.lm_head(hidden) if return_logits else None, hidden_states=hidden, state=next_state)

    def parameter_formula(self):
        d, h, k = self.config.d_model, self.config.hidden_dim, self.key_dim
        operator = d*k + 2*d*d + d+1
        return self.config.vocab_size*d + 3*d + 3*d + 3*d*h + len(self.operators)*operator + (2*d+2 if self.levels == 2 else 0)

    def state_bytes_per_sequence(self):
        d, k = self.config.d_model, self.key_dim
        return 4*(2*len(self.operators)*k*d + 3*d)
