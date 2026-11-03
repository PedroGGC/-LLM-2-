"""Non-Transformer temporal controls and explicitly shared two-rate hierarchy."""
from dataclasses import dataclass
import torch
from torch import nn
from ..interfaces import Backbone, ModelOutput
from ..model import RMSNorm, SwiGLU, TransformerLM
from .state_kernels import memory_rule


@dataclass
class SequenceState:
    memories: tuple
    pending: torch.Tensor | None = None
    last_coarse: torch.Tensor | None = None


class StateCell(nn.Module):
    def __init__(self, config, research):
        super().__init__()
        self.rule, self.execution, self.chunk_size = research.state_rule, research.state_execution, research.state_chunk_size
        self.heads, self.dim = config.n_heads, config.d_model
        self.norm = RMSNorm(self.dim, config.norm_eps)
        self.qkv = nn.Linear(self.dim,3*self.dim,bias=False)
        gate_dim = self.dim if self.rule == "rwkv7" else self.heads
        self.gates = nn.Linear(self.dim,2*gate_dim,bias=True)
        self.proj = nn.Linear(self.dim,self.dim,bias=False)
        self.ffn_norm = RMSNorm(self.dim, config.norm_eps)
        self.ffn = SwiGLU(self.dim,config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, state=None):
        batch, length, dim = x.shape
        value = self.norm(x)
        q,k,v = self.qkv(value).view(batch,length,3,self.heads,dim//self.heads).unbind(2)
        q,k,v = (a.transpose(1,2) for a in (q,k,v))
        width = dim//self.heads if self.rule == "rwkv7" else 1
        gates = self.gates(value).view(batch,length,2,self.heads,width).unbind(2)
        gates = tuple(g.transpose(1,2) for g in gates)
        result, state = memory_rule(self.rule,q,k,v,gates,state,execution=self.execution,chunk_size=self.chunk_size)
        x = x+self.dropout(self.proj(result.transpose(1,2).reshape(batch,length,dim).to(x.dtype)))
        return x+self.dropout(self.ffn(self.ffn_norm(x))), state


class StateRecurrentModel(Backbone):
    architecture_version = 1
    supports_hidden_only = True

    def __init__(self, config, research):
        super().__init__()
        self.config, self.research = config,research
        self.hierarchy = research.architecture == "state_hierarchy"
        if config.n_layers not in (1,2) or self.hierarchy and config.n_layers != 2:
            raise ValueError("state recurrent supports one/two cells; hierarchy requires two levels")
        self.shared = self.hierarchy and research.share_scales
        self.factor = research.state_factor
        self.memory_count = 2 if self.hierarchy else config.n_layers
        self.architecture_version = 2 if config.n_layers == 1 or research.state_readout == "immediate" else 1
        self.embedding = nn.Embedding(config.vocab_size,config.d_model)
        TransformerLM._init_weights(self.embedding)
        self.cells = nn.ModuleList()
        for _ in range(1 if self.shared else config.n_layers):
            cell=StateCell(config,research);cell.apply(TransformerLM._init_weights)
            nn.init.zeros_(cell.gates.bias)
            with torch.no_grad():
                cell.gates.bias[:cell.gates.out_features//2].fill_(-2. if cell.rule == "rwkv7" else 2.)
            self.cells.append(cell)
        # Same seed gives both hierarchies identical embedding/fine/coarse weights.
        # The independent core may diverge after its first optimizer update.
        if self.hierarchy and not self.shared:
            self.cells[1].load_state_dict(self.cells[0].state_dict())
        self.norm = RMSNorm(config.d_model,config.norm_eps)
        self.lm_head = nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, *, state=None, return_logits=True):
        if input_ids.ndim != 2 or not 0<input_ids.shape[1]<=self.config.max_seq_len:
            raise ValueError("input_ids must be [batch, 1..max_seq_len]")
        if state is not None and not isinstance(state,SequenceState):
            raise ValueError("state must be a SequenceState belonging to this document")
        if state is not None and self.training and self.config.dropout:
            raise ValueError("streaming with active dropout is unsupported; reset each training window")
        if state is not None and len(state.memories)!=self.memory_count:
            raise ValueError("state temporal memory count mismatch")
        if state is not None:
            expected=(input_ids.shape[0],self.config.n_heads,self.config.d_model//self.config.n_heads,
                      self.config.d_model//self.config.n_heads)
            for memory in state.memories:
                if memory is None:continue
                matrix=memory[0] if isinstance(memory,tuple) else memory
                if matrix.shape!=expected or matrix.device!=input_ids.device or matrix.dtype!=torch.float32:
                    raise ValueError("state batch/head/shape/device/dtype mismatch")
        memories = (None,)*self.memory_count if state is None else state.memories
        x = self.embedding(input_ids)
        fine, first = self.cells[0](x,memories[0])
        if not self.hierarchy:
            if self.memory_count == 1:
                value = fine
                next_state = SequenceState((first,))
            else:
                value, second = self.cells[1](fine,memories[1])
                next_state = SequenceState((first,second))
        else:
            pending = fine[:, :0] if state is None else state.pending
            previous = torch.zeros_like(fine[:,0]) if state is None else state.last_coarse
            if pending is None or previous is None or pending.shape[0]!=fine.shape[0]:
                raise ValueError("incompatible hierarchy state")
            offset = pending.shape[1]
            if offset>=self.factor:
                raise ValueError("invalid pending group")
            all_fine = torch.cat((pending,fine),1)
            complete = all_fine.shape[1]//self.factor
            second = memories[1]
            if complete:
                groups = all_fine[:,:complete*self.factor].reshape(fine.shape[0],complete,self.factor,fine.shape[-1]).float().mean(2).to(fine.dtype)
                coarse, second = self.cells[0 if self.shared else 1](groups,second)
                candidates = torch.cat((previous[:,None],coarse),1)
                last = coarse[:,-1].clone()
            else:
                candidates = previous[:,None]; last = previous
            advance = int(self.research.state_readout == "immediate")
            indices = (offset+torch.arange(fine.shape[1],device=fine.device)+advance)//self.factor
            value = fine+candidates[:,indices]
            # Compact cache storage: a short slice must not retain the entire prefix buffer.
            next_state = SequenceState((first,second),all_fine[:,complete*self.factor:].clone(),last)
        value = self.norm(value)
        return ModelOutput(logits=self.lm_head(value) if return_logits else None,
                           hidden_states=value,state=next_state)

    def parameter_formula(self):
        c,r = self.config,self.research
        d,h = c.d_model,c.hidden_dim
        gate_width = d if r.state_rule == "rwkv7" else c.n_heads
        cell = 4*d*d+3*d*h+2*d+2*gate_width*(d+1)
        return c.vocab_size*d+d+len(self.cells)*cell

    def state_bytes_per_sequence(self):
        c = self.config
        matrix = c.n_heads*(c.d_model//c.n_heads)**2
        extra = c.d_model+c.n_heads if self.research.state_rule == "mlstm" else 0
        # Maximum pending group and last coarse are execution state, not parameters.
        hierarchy = self.factor*c.d_model if self.hierarchy else 0
        return 4*(self.memory_count*(matrix+extra)+hierarchy)
