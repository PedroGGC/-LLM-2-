"""Experimental Titans / TTT (Test-Time Training) Surprise-Gated Neural Long-Term Memory.

Implements a 2-layer MLP fast-weight memory state updated causally at test time via
surprise-gated momentum gradient descent (Behrouz et al., Titans/MIRAS, arXiv:2501.00663):
    ell(M; k_t, v_t) = 0.5 * || M(k_t) - v_t ||_2^2
    S_t = eta_t * S_{t-1} - theta_t * grad_M ell(M_{t-1}; k_t, v_t)
    M_t = (1 - alpha_t) * M_{t-1} + S_t

Kept isolated for L4/A100 Colab ablation before any integration into Lumina 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..model import RMSNorm


@dataclass
class TitansMemoryState:
    """Fixed O(1) fast-weight MLP parameters + momentum buffers per batch sequence."""

    w1: torch.Tensor  # (B, H, M, D)
    w2: torch.Tensor  # (B, H, D, M)
    s1: torch.Tensor  # (B, H, M, D)
    s2: torch.Tensor  # (B, H, D, M)


class TitansMemoryCell(nn.Module):
    """Causal multi-head surprise-gated 2-layer MLP Test-Time Training memory block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        *,
        mem_hidden_mult: int = 2,
        chunk_size: int = 16,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.heads = int(n_heads)
        self.head_dim = self.d_model // self.heads
        self.mem_dim = max(8, self.head_dim * int(mem_hidden_mult))
        self.chunk_size = max(1, int(chunk_size))

        self.norm = RMSNorm(self.d_model, eps=norm_eps)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        # Per-head data-dependent forgetting (alpha), momentum decay (eta), and surprise step (theta)
        self.alpha_proj = nn.Linear(self.d_model, self.heads, bias=True)
        self.eta_proj = nn.Linear(self.d_model, self.heads, bias=True)
        self.theta_proj = nn.Linear(self.d_model, self.heads, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Learnable meta-initial fast weights M_0 shared across batch
        self.init_w1 = nn.Parameter(torch.empty(self.heads, self.mem_dim, self.head_dim))
        self.init_w2 = nn.Parameter(torch.empty(self.heads, self.head_dim, self.mem_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.alpha_proj.weight, std=0.02)
        nn.init.constant_(self.alpha_proj.bias, -3.5)  # small initial forget ~0.03
        nn.init.normal_(self.eta_proj.weight, std=0.02)
        nn.init.constant_(self.eta_proj.bias, 1.5)  # strong momentum ~0.82
        nn.init.normal_(self.theta_proj.weight, std=0.02)
        nn.init.constant_(self.theta_proj.bias, -2.5)  # conservative inner LR ~0.07
        nn.init.normal_(self.init_w1, std=1.0 / math.sqrt(self.head_dim))
        nn.init.normal_(self.init_w2, std=0.02 / math.sqrt(self.mem_dim))

    def empty_state(self, batch_size: int, device: torch.device) -> TitansMemoryState:
        w1 = self.init_w1.float().unsqueeze(0).expand(batch_size, -1, -1, -1).clone().to(device)
        w2 = self.init_w2.float().unsqueeze(0).expand(batch_size, -1, -1, -1).clone().to(device)
        return TitansMemoryState(w1=w1, w2=w2, s1=torch.zeros_like(w1), s2=torch.zeros_like(w2))

    def state_bytes_per_seq(self) -> int:
        # 2 FP32 weight matrices + 2 FP32 momentum matrices per head
        return 4 * self.heads * self.mem_dim * self.head_dim * 4

    @staticmethod
    def _mlp_read_and_grad(
        w1: torch.Tensor, w2: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analytic forward & gradient of 0.5*||W2 @ SiLU(W1 @ k) - v||^2 w.r.t. (W1, W2) in FP32."""
        # k, v: (B, H, C, D); w1: (B, H, M, D); w2: (B, H, D, M)
        pre = torch.einsum("bhmd,bhcd->bhcm", w1, k)
        sig = torch.sigmoid(pre)
        act = pre * sig  # SiLU(pre)
        pred = torch.einsum("bhdm,bhcm->bhcd", w2, act)
        err = pred - v  # dL/dpred
        inv_c = 1.0 / float(k.shape[2])
        grad_w2 = torch.einsum("bhcd,bhcm->bhdm", err, act) * inv_c
        d_act = torch.einsum("bhcd,bhdm->bhcm", err, w2)
        d_silu = sig * (1.0 + pre * (1.0 - sig))
        d_pre = d_act * d_silu
        grad_w1 = torch.einsum("bhcm,bhcd->bhmd", d_pre, k) * inv_c
        return pred, grad_w1, grad_w2

    def forward(
        self,
        x: torch.Tensor,
        state: TitansMemoryState | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, TitansMemoryState]:
        """Causal chunked Titans update (chunk_size=1 gives exact tokenwise recurrence)."""
        b, t, d = x.shape
        c_size = self.chunk_size if chunk_size is None else max(1, int(chunk_size))
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        alpha_all = torch.sigmoid(self.alpha_proj(h).transpose(1, 2).float())  # (B, H, T)
        eta_all = torch.sigmoid(self.eta_proj(h).transpose(1, 2).float())
        theta_all = 0.25 * torch.sigmoid(self.theta_proj(h).transpose(1, 2).float())

        st = self.empty_state(b, x.device) if state is None else state
        w1, w2, s1, s2 = st.w1.float(), st.w2.float(), st.s1.float(), st.s2.float()

        outputs = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf = F.normalize(q.float(), dim=-1, eps=1e-6)
            kf = F.normalize(k.float(), dim=-1, eps=1e-6)
            vf = v.float()

            for start in range(0, t, c_size):
                end = min(start + c_size, t)
                qc = qf[:, :, start:end]
                kc = kf[:, :, start:end]
                vc = vf[:, :, start:end]

                # Causal read BEFORE writing current chunk into memory M (strict causality within chunk too)
                pre_q = torch.einsum("bhmd,bhcd->bhcm", w1, qc)
                y_c = torch.einsum("bhdm,bhcm->bhcd", w2, F.silu(pre_q))
                outputs.append(y_c)

                # Surprise-gated momentum update using analytic inner gradient on current chunk
                _, g1, g2 = self._mlp_read_and_grad(w1, w2, kc, vc)
                a_c = alpha_all[:, :, start:end].mean(dim=-1, keepdim=True).unsqueeze(-1)
                e_c = eta_all[:, :, start:end].mean(dim=-1, keepdim=True).unsqueeze(-1)
                th_c = theta_all[:, :, start:end].mean(dim=-1, keepdim=True).unsqueeze(-1)

                s1 = e_c * s1 - th_c * g1
                s2 = e_c * s2 - th_c * g2
                w1 = (1.0 - a_c) * w1 + s1
                w2 = (1.0 - a_c) * w2 + s2

            y = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().reshape(b, t, d).to(x.dtype)
        return self.out_proj(y).to(x.dtype), TitansMemoryState(w1.contiguous(), w2.contiguous(), s1.contiguous(), s2.contiguous())
