"""FP32 references for three published memory rules, not full published models.

Tensor layout B,H,T,D; memory layout B,H,V,K. Chunked DPLR uses a
triangular solve, not a Python loop over tokens. No dense sequence attention.
Sources and adaptations: docs/RECURRENT_CANDIDATES.md.
"""
import math
import torch
from torch.nn import functional as F


RULES = ("gated_delta", "rwkv7", "mlstm")


def dplr_sequential(q, k, v, log_w, a, b, state=None):
    """S_t=S_{t-1}diag(w_t)+(S_{t-1}a_t)b_t^T+v_t k_t^T."""
    q, k, v, log_w, a, b = (x.float() for x in (q, k, v, log_w, a, b))
    memory = v.new_zeros(*q.shape[:2], v.shape[-1], q.shape[-1]) if state is None else state
    outputs = []
    for t in range(q.shape[-2]):
        removed = torch.einsum("bhvk,bhk->bhv", memory, a[:, :, t])
        memory = (memory * log_w[:, :, t].exp().unsqueeze(-2)
                  + removed.unsqueeze(-1) * b[:, :, t].unsqueeze(-2)
                  + v[:, :, t].unsqueeze(-1) * k[:, :, t].unsqueeze(-2))
        outputs.append(torch.einsum("bhvk,bhk->bhv", memory, q[:, :, t]))
    return torch.stack(outputs, -2), memory


def dplr_chunked(q, k, v, log_w, a, b, state=None, *, chunk_size=32):
    """Exact DPLR recurrence via within-chunk causal triangular systems."""
    q, k, v, log_w, a, b = (x.float() for x in (q, k, v, log_w, a, b))
    memory = v.new_zeros(*q.shape[:2], v.shape[-1], q.shape[-1]) if state is None else state
    outputs = []
    for start in range(0, q.shape[-2], chunk_size):
        qc, kc, vc, wc, ac, bc = (x[:, :, start:start+chunk_size] for x in (q, k, v, log_w, a, b))
        n = qc.shape[-2]
        cumulative = wc.cumsum(-2)
        previous = cumulative - wc
        index = torch.arange(n, device=q.device)
        strict = index[:, None] > index[None, :]
        causal = index[:, None] >= index[None, :]
        # Invalid future differences must be masked BEFORE exp (no inf*0).
        decay_before = (previous.unsqueeze(-2)-cumulative.unsqueeze(-3))
        decay_before = torch.where(strict[..., None], decay_before, 0.).exp()
        rank = (ac.unsqueeze(-2)*bc.unsqueeze(-3)*decay_before).sum(-1).masked_fill(~strict, 0.)
        inputs = (ac.unsqueeze(-2)*kc.unsqueeze(-3)*decay_before).sum(-1).masked_fill(~strict, 0.)
        rhs = torch.einsum("bhvk,bhtk->bhtv", memory, ac*previous.exp()) + inputs @ vc
        eye = torch.eye(n, device=q.device, dtype=q.dtype)
        updates = torch.linalg.solve_triangular(eye-rank, rhs, upper=False, unitriangular=True)
        decay_after = cumulative.unsqueeze(-2)-cumulative.unsqueeze(-3)
        decay_after = torch.where(causal[..., None], decay_after, 0.).exp()
        reads_k = (qc.unsqueeze(-2)*kc.unsqueeze(-3)*decay_after).sum(-1).masked_fill(~causal, 0.)
        reads_b = (qc.unsqueeze(-2)*bc.unsqueeze(-3)*decay_after).sum(-1).masked_fill(~causal, 0.)
        outputs.append(torch.einsum("bhvk,bhtk->bhtv", memory, qc*cumulative.exp())
                       + reads_k @ vc + reads_b @ updates)
        tail = (cumulative[:, :, -1:]-cumulative).exp()
        memory = (memory*cumulative[:, :, -1].exp().unsqueeze(-2)
                  + torch.einsum("bhtv,bhtk->bhvk", vc, kc*tail)
                  + torch.einsum("bhtv,bhtk->bhvk", updates, bc*tail))
    return torch.cat(outputs, -2), memory


def mlstm_sequential(q, k, v, input_gate, forget_gate, state=None):
    """Stabilized covariance memory including its normalizer and log scale."""
    q, k, v, input_gate, forget_gate = (x.float() for x in (q, k, v, input_gate, forget_gate))
    k = k / math.sqrt(k.shape[-1])
    if state is None:
        c = v.new_zeros(*q.shape[:2], v.shape[-1], q.shape[-1])
        n = torch.zeros_like(q[:, :, 0]); m = torch.zeros_like(input_gate[:, :, 0])
    else:
        c, n, m = state
    outputs = []
    for t in range(q.shape[-2]):
        lf = F.logsigmoid(forget_gate[:, :, t])
        next_m = torch.maximum(torch.maximum(m+lf, input_gate[:, :, t]), torch.zeros_like(m))
        f = (m+lf-next_m).exp(); i = (input_gate[:, :, t]-next_m).exp()
        c = c*f.unsqueeze(-1) + (v[:, :, t]*i).unsqueeze(-1)*k[:, :, t].unsqueeze(-2)
        n = n*f+k[:, :, t]*i; m = next_m
        numerator = torch.einsum("bhvk,bhk->bhv", c, q[:, :, t])
        denominator = torch.maximum((n*q[:, :, t]).sum(-1, keepdim=True).abs(), (-m).exp())+1e-6
        outputs.append(numerator/denominator)
    return torch.stack(outputs, -2), (c, n, m)


def mlstm_chunked(q, k, v, input_gate, forget_gate, state=None, *, chunk_size=32):
    q, k, v, input_gate, forget_gate = (x.float() for x in (q, k, v, input_gate, forget_gate))
    k = k / math.sqrt(k.shape[-1])
    if state is None:
        c = v.new_zeros(*q.shape[:2], v.shape[-1], q.shape[-1])
        n = torch.zeros_like(q[:, :, 0]); m = torch.zeros_like(input_gate[:, :, 0])
    else:
        c, n, m = state
    outputs = []
    for start in range(0, q.shape[-2], chunk_size):
        qc, kc, vc, ic, fc = (x[:, :, start:start+chunk_size] for x in (q,k,v,input_gate,forget_gate))
        length = qc.shape[-2]
        cumulative = F.logsigmoid(fc).cumsum(-2)
        causal = torch.arange(length, device=q.device)[:, None] >= torch.arange(length, device=q.device)[None, :]
        logs = (cumulative-cumulative.transpose(-2,-1)+ic.transpose(-2,-1)).masked_fill(~causal, -torch.inf)
        scale = torch.maximum(torch.maximum(logs.amax(-1,keepdim=True), m.unsqueeze(-2)+cumulative),
                              torch.zeros_like(cumulative))
        weights = (logs-scale).exp()
        carry = (m.unsqueeze(-2)+cumulative-scale).exp()
        similarity = (qc @ kc.transpose(-2,-1))*weights
        numerator = torch.einsum("bhvk,bhtk->bhtv", c, qc)*carry + similarity @ vc
        normalizer = (qc*n.unsqueeze(-2)).sum(-1,keepdim=True)*carry+similarity.sum(-1,keepdim=True)
        outputs.append(numerator/(torch.maximum(normalizer.abs(), (-scale).exp())+1e-6))
        final_weights = weights[:, :, -1]
        c = c*carry[:, :, -1].unsqueeze(-1)+torch.einsum("bhtv,bhtk->bhvk", vc, kc*final_weights.unsqueeze(-1))
        n = n*carry[:, :, -1]+(kc*final_weights.unsqueeze(-1)).sum(-2)
        m = scale[:, :, -1].clone()
    return torch.cat(outputs, -2), (c,n,m)


def memory_rule(rule, q, k, v, gates, state=None, *, execution="chunked", chunk_size=32):
    if rule not in RULES or execution not in ("sequential", "chunked"):
        raise ValueError("unknown memory rule/execution")
    if rule == "mlstm":
        function = mlstm_chunked if execution == "chunked" else mlstm_sequential
        kwargs = {"chunk_size": chunk_size} if execution == "chunked" else {}
        return function(q,k,v,*gates,state,**kwargs)
    key = F.normalize(k.float(), dim=-1, eps=1e-6)
    query = F.normalize(q.float(), dim=-1, eps=1e-6)
    if rule == "gated_delta":
        log_w = F.logsigmoid(gates[0].float()).expand_as(key)
        beta = gates[1].float().sigmoid()
        a, b, values = -log_w.exp()*beta*key, key, beta*v.float()
    else:
        # Generalized RWKV-7 rule; full model token-shift/LoRA/value residuals omitted.
        log_w = -F.softplus(gates[0].float())
        a, b, values = -key, key*gates[1].float().sigmoid(), v.float()
    function = dplr_chunked if execution == "chunked" else dplr_sequential
    kwargs = {"chunk_size": chunk_size} if execution == "chunked" else {}
    return function(query,key,values,log_w,a,b,state,**kwargs)
