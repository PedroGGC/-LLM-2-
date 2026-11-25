"""FP32 references for three published memory rules, not full published models.

Tensor layout B,H,T,D; memory layout B,H,V,K. Chunked DPLR uses a
triangular solve, not a Python loop over tokens. No dense sequence attention.
Sources and adaptations: docs/RECURRENT_CANDIDATES.md.
"""
import math
import torch
from torch.nn import functional as F

from .triton_dplr import can_use_triton_dplr, dplr_inter_chunk_scan


RULES = ("gated_delta", "rwkv7", "mlstm")


def dplr_sequential(q, k, v, log_w, a, b, state=None, *, return_all_states=False):
    """S_t=S_{t-1}diag(w_t)+(S_{t-1}a_t)b_t^T+v_t k_t^T (supports GQA when H_q = H_kv * G)."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, log_w, a, b = (x.float() for x in (q, k, v, log_w, a, b))
        batch, q_heads, seq_len, k_dim = q.shape
        kv_heads = k.shape[1]
        g_mult = q_heads // kv_heads
        memory = v.new_zeros(batch, kv_heads, v.shape[-1], k_dim) if state is None else state.float()
        if seq_len == 1 and not return_all_states:
            # Fast single-step decode path: direct batched matmul (no einsum parsing or list/stack overhead).
            a0 = a[:, :, 0]
            b0 = b[:, :, 0]
            k0 = k[:, :, 0]
            v0 = v[:, :, 0]
            w0 = log_w[:, :, 0].exp().unsqueeze(-2)
            removed = torch.matmul(memory, a0.unsqueeze(-1))
            memory = memory * w0 + removed * b0.unsqueeze(-2) + v0.unsqueeze(-1) * k0.unsqueeze(-2)
            if g_mult == 1:
                out = torch.matmul(memory, q[:, :, 0].unsqueeze(-1)).transpose(-1, -2)
            else:
                qg_t = q[:, :, 0].reshape(batch, kv_heads, g_mult, k_dim).transpose(-1, -2)
                out = torch.matmul(memory, qg_t).transpose(-1, -2).reshape(batch, q_heads, 1, v.shape[-1])
            return out, memory
        outputs = []
        states = [] if return_all_states else None
        for t in range(seq_len):
            removed = torch.einsum("bhvk,bhk->bhv", memory, a[:, :, t])
            memory = (memory * log_w[:, :, t].exp().unsqueeze(-2)
                      + removed.unsqueeze(-1) * b[:, :, t].unsqueeze(-2)
                      + v[:, :, t].unsqueeze(-1) * k[:, :, t].unsqueeze(-2))
            if g_mult == 1:
                outputs.append(torch.einsum("bhvk,bhk->bhv", memory, q[:, :, t]))
            else:
                qg = q[:, :, t].reshape(batch, kv_heads, g_mult, k_dim)
                outputs.append(torch.einsum("bhvk,bhgk->bhgv", memory, qg).reshape(batch, q_heads, v.shape[-1]))
            if states is not None:
                states.append(memory)
        out = torch.stack(outputs, -2)
        if states is not None:
            return out, memory, torch.stack(states, 0)
        return out, memory


def dplr_chunked(q, k, v, log_w, a, b, state=None, *, chunk_size=32, use_triton=True):
    """Exact DPLR recurrence via batched within-chunk causal triangular systems (supports GQA H_q = H_kv * G)."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, log_w, a, b = (x.float() for x in (q, k, v, log_w, a, b))
        batch, q_heads, total_len, k_dim = q.shape
        kv_heads = k.shape[1]
        g_mult = q_heads // kv_heads
        v_dim = v.shape[-1]
        w_dim = log_w.shape[-1]
        if w_dim not in (1, k_dim):
            raise ValueError(f"Expected log_w trailing dim 1 or {k_dim}, got {w_dim}")
        memory = v.new_zeros(batch, kv_heads, v_dim, k_dim) if state is None else state.float()
        pad = (-total_len) % chunk_size
        if pad:
            q, k, v, log_w, a, b = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, log_w, a, b))
        n_chunks = q.shape[-2] // chunk_size
        kc, ac, bc = (x.reshape(batch, kv_heads, n_chunks, chunk_size, k_dim) for x in (k, a, b))
        wc = log_w.reshape(batch, kv_heads, n_chunks, chunk_size, w_dim)
        vc = v.reshape(batch, kv_heads, n_chunks, chunk_size, v_dim)

        cumulative = wc.cumsum(-2)
        previous = cumulative - wc
        index = torch.arange(chunk_size, device=q.device)
        strict = index[:, None] > index[None, :]
        causal = index[:, None] >= index[None, :]
        if w_dim == 1:
            cum_s = cumulative.squeeze(-1)
            prev_s = previous.squeeze(-1)
            # Invalid future differences must be masked BEFORE exp (no inf*0).
            decay_before = torch.where(strict, prev_s.unsqueeze(-1) - cum_s.unsqueeze(-2), 0.0).exp().masked_fill(~strict, 0.0)
            rank = torch.matmul(ac, bc.transpose(-1, -2)) * decay_before
            inputs = torch.matmul(ac, kc.transpose(-1, -2)) * decay_before
            decay_after = torch.where(causal, cum_s.unsqueeze(-1) - cum_s.unsqueeze(-2), 0.0).exp().masked_fill(~causal, 0.0)
        else:
            decay_before = previous.unsqueeze(-2) - cumulative.unsqueeze(-3)
            decay_before = torch.where(strict[..., None], decay_before, 0.0).exp()
            rank = (ac.unsqueeze(-2) * bc.unsqueeze(-3) * decay_before).sum(-1).masked_fill(~strict, 0.0)
            inputs = (ac.unsqueeze(-2) * kc.unsqueeze(-3) * decay_before).sum(-1).masked_fill(~strict, 0.0)
            decay_after = cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)
            decay_after = torch.where(causal[..., None], decay_after, 0.0).exp()

        eye = torch.eye(chunk_size, device=q.device, dtype=q.dtype)
        rhs_combined = torch.cat((inputs @ vc, ac * previous.exp()), dim=-1)
        solved = torch.linalg.solve_triangular(eye - rank, rhs_combined, upper=False, unitriangular=True)
        u_local, u_state = solved.split((v_dim, k_dim), dim=-1)

        tail = (cumulative[:, :, :, -1:] - cumulative).exp()
        bc_tail = bc * tail
        mem_decay = cumulative[:, :, :, -1].exp().unsqueeze(-2)
        mem_local = vc.transpose(-1, -2) @ (kc * tail) + u_local.transpose(-1, -2) @ bc_tail
        state_trans = u_state.transpose(-1, -2) @ bc_tail

        if g_mult == 1:
            qc = q.reshape(batch, kv_heads, n_chunks, chunk_size, k_dim)
            if w_dim == 1:
                reads_k = torch.matmul(qc, kc.transpose(-1, -2)) * decay_after
                reads_b = torch.matmul(qc, bc.transpose(-1, -2)) * decay_after
            else:
                reads_k = (qc.unsqueeze(-2) * kc.unsqueeze(-3) * decay_after).sum(-1).masked_fill(~causal, 0.0)
                reads_b = (qc.unsqueeze(-2) * bc.unsqueeze(-3) * decay_after).sum(-1).masked_fill(~causal, 0.0)
            kv_local = reads_k @ vc + reads_b @ u_local
            read_proj = qc * cumulative.exp() + reads_b @ u_state
            if use_triton and (q.is_cuda or can_use_triton_dplr(chunk_size, v_dim, k_dim, q.device)):
                out, memory = dplr_inter_chunk_scan(kv_local, read_proj, mem_decay, state_trans, mem_local, memory)
                return (out[:, :, :total_len] if pad else out), memory
            outputs = []
            for m in range(n_chunks):
                outputs.append(kv_local[:, :, m] + read_proj[:, :, m] @ memory.transpose(-1, -2))
                memory = memory * mem_decay[:, :, m] + mem_local[:, :, m] + memory @ state_trans[:, :, m]
            out = torch.cat(outputs, dim=-2)
            return (out[:, :, :total_len] if pad else out), memory

        # GQA-DeltaNet path: H_kv states updated once; all G query groups read from shared H_kv states.
        qc_g = q.reshape(batch, kv_heads, g_mult, n_chunks, chunk_size, k_dim)
        if w_dim == 1:
            reads_k = torch.einsum("bhgctd,bhcsd->bhgcts", qc_g, kc) * decay_after.unsqueeze(2)
            reads_b = torch.einsum("bhgctd,bhcsd->bhgcts", qc_g, bc) * decay_after.unsqueeze(2)
        else:
            reads_k = (qc_g.unsqueeze(-2) * kc.unsqueeze(2).unsqueeze(-3) * decay_after.unsqueeze(2)).sum(-1).masked_fill(~causal, 0.0)
            reads_b = (qc_g.unsqueeze(-2) * bc.unsqueeze(2).unsqueeze(-3) * decay_after.unsqueeze(2)).sum(-1).masked_fill(~causal, 0.0)
        kv_local = torch.einsum("bhgcts,bhcsv->bhgctv", reads_k, vc) + torch.einsum("bhgcts,bhcsv->bhgctv", reads_b, u_local)
        read_proj = qc_g * cumulative.exp().unsqueeze(2) + torch.einsum("bhgcts,bhcsd->bhgctd", reads_b, u_state)
        outputs = []
        for m in range(n_chunks):
            outputs.append(kv_local[:, :, :, m] + torch.einsum("bhgtd,bhvd->bhgtv", read_proj[:, :, :, m], memory))
            memory = memory * mem_decay[:, :, m] + mem_local[:, :, m] + memory @ state_trans[:, :, m]
        out = torch.cat(outputs, dim=-2).reshape(batch, q_heads, -1, v_dim)
        return (out[:, :, :total_len] if pad else out), memory


def mlstm_sequential(q, k, v, input_gate, forget_gate, state=None):
    """Stabilized covariance memory including its normalizer and log scale."""
    with torch.autocast(device_type=q.device.type, enabled=False):
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
    with torch.autocast(device_type=q.device.type, enabled=False):
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
        log_w = F.logsigmoid(gates[0].float())
        beta = gates[1].float().sigmoid()
        a, b, values = -log_w.exp()*beta*key, key, beta*v.float()
    else:
        # Generalized RWKV-7 rule; full model token-shift/LoRA/value residuals omitted.
        log_w = -F.softplus(gates[0].float())
        a, b, values = -key, key*gates[1].float().sigmoid(), v.float()
    function = dplr_chunked if execution == "chunked" else dplr_sequential
    kwargs = {"chunk_size": chunk_size} if execution == "chunked" else {}
    return function(query,key,values,log_w,a,b,state,**kwargs)
