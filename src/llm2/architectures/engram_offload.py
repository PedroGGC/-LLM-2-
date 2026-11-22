"""#7 Engram GPU/CPU offload table (pinned host RAM with async PCIe transfer) and L4 2:4 semi-structured sparsity helper."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F


class EngramTableCPU(nn.Module):
    """Static N-gram / factual memory table resident in pinned host CPU RAM (0 bytes VRAM for table weights).

    Computes rolling bigram/trigram hashes of token IDs, performs O(1) CPU lookup into a large
    host table, transfers only the active token slices asynchronously over PCIe (`non_blocking=True`),
    and fuses them into the GPU residual stream via a small gated projection.
    """

    def __init__(
        self,
        num_entries: int,
        engram_dim: int,
        d_model: int,
        *,
        pin_memory: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if num_entries < 16 or engram_dim < 8:
            raise ValueError("EngramTableCPU requires num_entries >= 16 and engram_dim >= 8")
        self.num_entries = int(num_entries)
        self.engram_dim = int(engram_dim)
        self.d_model = int(d_model)
        # Keep the massive table off-GPU in host RAM (not registered as a GPU parameter/buffer).
        table = torch.randn(self.num_entries, self.engram_dim, dtype=torch.float32) * 0.02
        table = table.to(dtype=dtype)
        if pin_memory and torch.cuda.is_available():
            table = table.pin_memory()
        self.host_table = table
        self.proj = nn.Linear(self.engram_dim, self.d_model, bias=False)
        self.gate = nn.Linear(self.d_model, 1, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.gate.bias, -2.0)

    def host_table_bytes(self) -> int:
        return int(self.host_table.numel() * self.host_table.element_size())

    def gpu_param_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.parameters()))

    def hash_ngrams(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Deterministic causal bigram hash indices in [0, num_entries)."""
        ids_cpu = token_ids.detach().to(device="cpu", dtype=torch.int64)
        prev = F.pad(ids_cpu[:, :-1], (1, 0), value=0)
        hashed = ((ids_cpu * 1315423911) ^ (prev * 2654435761)) & 0x7FFFFFFF
        return hashed % self.num_entries

    def forward(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        idx_cpu = self.hash_ngrams(token_ids)
        # Gather active rows on CPU, then stream only the tiny [B, T, engram_dim] slice over PCIe.
        fetched_cpu = F.embedding(idx_cpu, self.host_table)
        proj_dtype = self.proj.weight.dtype
        fetched_gpu = fetched_cpu.to(device=hidden.device, dtype=proj_dtype, non_blocking=True)
        projected = self.proj(fetched_gpu)
        alpha = torch.sigmoid(self.gate(hidden.to(self.gate.weight.dtype)))
        return hidden + (alpha * projected).to(hidden.dtype)


def benchmark_semi_structured_24(
    in_features: int,
    out_features: int,
    m_rows: int,
    device: torch.device,
    *,
    iters: int = 100,
) -> dict:
    """Measure native 2:4 semi-structured Tensor Core speedup vs Dense BF16 on CUDA."""
    if device.type != "cuda":
        return {"supported": False, "reason": "non_cuda"}
    try:
        from torch.sparse import to_sparse_semi_structured
    except Exception as exc:
        return {"supported": False, "reason": str(exc)}

    torch.manual_seed(42)
    x = torch.randn(m_rows, in_features, device=device, dtype=torch.bfloat16)
    w = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16) * 0.02
    # Enforce exact 2:4 sparsity pattern (zero smallest 2 of every 4 contiguous weights)
    w_4 = w.view(out_features, in_features // 4, 4)
    topk_idx = w_4.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(w_4, dtype=torch.bool).scatter_(-1, topk_idx, True)
    w_pruned = (w_4 * mask).view(out_features, in_features).contiguous()

    try:
        w_sparse = to_sparse_semi_structured(w_pruned)
        # Warmup & check support on current GPU/cuSPARSELt
        with torch.inference_mode():
            for _ in range(15):
                _ = F.linear(x, w_pruned)
                _ = F.linear(x, w_sparse)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = F.linear(x, w_pruned)
            torch.cuda.synchronize(device)
            dense_us = ((time.perf_counter() - t0) / iters) * 1e6

            t1 = time.perf_counter()
            for _ in range(iters):
                _ = F.linear(x, w_sparse)
            torch.cuda.synchronize(device)
            sparse_us = ((time.perf_counter() - t1) / iters) * 1e6
        return {
            "supported": True,
            "shape": [m_rows, in_features, out_features],
            "dense_bf16_us": round(dense_us, 1),
            "sparse_24_us": round(sparse_us, 1),
            "speedup": round(dense_us / max(sparse_us, 1e-6), 2),
        }
    except Exception as exc:
        return {"supported": False, "shape": [m_rows, in_features, out_features], "reason": str(exc)}
