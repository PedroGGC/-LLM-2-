"""Fase D — MetaQueryBridge connecting Lumina V1 O(1) recurrent state + GRM visual anchors to Nano Lumina (DiT / VAE 16ch & 3D GIF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..architectures.fractal_delta import FractalDeltaMemory
from ..model import RMSNorm, SwiGLU
from ..research.decode_optimization import SegmentMemoryCache
from .vision_bridge import LuminaMultimodalWrapper


@dataclass
class MetaQueryOutput:
    seq_cond: torch.Tensor  # (B, Q, dit_cond_dim) for image or (B, F, Q, dit_cond_dim) for multi-frame GIF
    pooled_cond: torch.Tensor  # (B, dit_pooled_dim) for DiT AdaLN / timestep conditioning
    query_hidden: torch.Tensor  # (B, Q, d_model) raw hidden states from Lumina V1
    state: object  # FractalDeltaState (fixed O(1) bytes)


class _ConnectorBlock(nn.Module):
    """Pre-norm bidirectional query refinement block with FP32 RMSNorm and SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, norm_eps: float = 1e-6):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.attn_norm = RMSNorm(self.d_model, eps=norm_eps)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.ffn_norm = RMSNorm(self.d_model, eps=norm_eps)
        self.ffn = SwiGLU(self.d_model, self.d_model * 2)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, q_len, d = x.shape
        h = self.attn_norm(x)
        q, k, v = self.qkv(h).reshape(b, q_len, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.out_proj(attn.transpose(1, 2).contiguous().reshape(b, q_len, d))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MetaQueryBridge(nn.Module):
    """Learnable MetaQuery tokens + 2-layer connector bridging Lumina V1 to Nano Lumina DiT (static images & 3D GIFs)."""

    def __init__(
        self,
        backbone: FractalDeltaMemory,
        *,
        num_queries: int = 64,
        dit_cond_dim: int = 2048,
        dit_pooled_dim: int = 768,
        connector_layers: int = 2,
        max_frames: int = 32,
        grm_alpha: float = 0.25,
    ):
        super().__init__()
        if not isinstance(backbone, FractalDeltaMemory):
            raise ValueError("MetaQueryBridge requires a FractalDeltaMemory backbone")
        if num_queries < 1 or dit_cond_dim < 1 or dit_pooled_dim < 1 or max_frames < 1:
            raise ValueError("num_queries, dit_cond_dim, dit_pooled_dim, and max_frames must be positive")
        self.backbone = backbone
        d = backbone.config.d_model
        heads = backbone.config.n_heads
        eps = backbone.config.norm_eps

        self.num_queries = int(num_queries)
        self.dit_cond_dim = int(dit_cond_dim)
        self.dit_pooled_dim = int(dit_pooled_dim)
        self.max_frames = int(max_frames)
        self.grm_alpha = float(grm_alpha)

        self.meta_queries = nn.Parameter(torch.empty(1, self.num_queries, d))
        self.temporal_frame_emb = nn.Parameter(torch.zeros(self.max_frames, d))
        nn.init.normal_(self.meta_queries, std=0.02)
        nn.init.normal_(self.temporal_frame_emb, std=0.01)

        self.blocks = nn.ModuleList([_ConnectorBlock(d, heads, norm_eps=eps) for _ in range(connector_layers)])
        self.final_norm = RMSNorm(d, eps=eps)
        self.seq_proj = nn.Linear(d, self.dit_cond_dim, bias=False)
        self.pool_proj = nn.Linear(d, self.dit_pooled_dim, bias=False)
        nn.init.normal_(self.seq_proj.weight, std=0.02)
        nn.init.normal_(self.pool_proj.weight, std=0.02)

    def freeze_backbone_for_metaquery(self, wrapper: LuminaMultimodalWrapper | None = None) -> dict[str, int]:
        """Freeze Lumina V1 backbone (and vision wrapper if provided), leaving only MetaQuery + Connector trainable."""
        self.backbone.requires_grad_(False)
        if wrapper is not None:
            wrapper.requires_grad_(False)
        self.meta_queries.requires_grad_(True)
        self.temporal_frame_emb.requires_grad_(True)
        self.blocks.requires_grad_(True)
        self.final_norm.requires_grad_(True)
        self.seq_proj.requires_grad_(True)
        self.pool_proj.requires_grad_(True)

        all_params = list(self.parameters()) + (list(wrapper.parameters()) if wrapper is not None else [])
        trainable = sum(p.numel() for p in {p.data_ptr(): p for p in all_params if p.requires_grad}.values())
        frozen = sum(p.numel() for p in {p.data_ptr(): p for p in all_params if not p.requires_grad}.values())
        return {"trainable_params": trainable, "frozen_params": frozen}

    def forward(
        self,
        *,
        prompt_ids: torch.Tensor | None = None,
        images: Sequence[torch.Tensor] | None = None,
        wrapper: LuminaMultimodalWrapper | None = None,
        state=None,
        grm_cache: SegmentMemoryCache | None = None,
        num_frames: int = 1,
    ) -> MetaQueryOutput:
        """Run prompt (+ optional reference images) into O(1) state, then read out Q MetaQuery tokens for DiT/GIF generation."""
        if not 1 <= int(num_frames) <= self.max_frames:
            raise ValueError(f"num_frames must be in 1..{self.max_frames}")
        cur_state = state
        b = 1
        target_dtype = self.meta_queries.dtype

        if prompt_ids is not None:
            if prompt_ids.ndim != 2:
                raise ValueError("prompt_ids must have shape (B, T)")
            b = prompt_ids.shape[0]
            if images and wrapper is not None:
                if b != 1:
                    raise ValueError("multimodal prefill with images requires batch=1")
                _, _, cur_state = wrapper.prefill_multimodal(
                    images=images,
                    prompt_ids=prompt_ids,
                    grm_cache=grm_cache,
                )
            else:
                out_p = self.backbone(prompt_ids, state=cur_state, return_logits=False)
                cur_state = out_p.state
        elif cur_state is not None:
            b = cur_state.levels[0].memory.shape[0]

        queries = self.meta_queries.to(dtype=self.backbone.embedding.weight.dtype).expand(b, -1, -1)
        out_q = self.backbone(inputs_embeds=queries, state=cur_state, return_logits=False)
        q_hidden = out_q.hidden_states  # (B, Q, d_model)

        if grm_cache is not None and len(grm_cache._grm_segments) > 0 and self.grm_alpha > 0.0:
            grm_delta, _ = grm_cache.query_grm_readout(q_hidden, model=self.backbone, top_k=2, temperature=0.5)
            q_hidden = q_hidden + self.grm_alpha * grm_delta.to(dtype=q_hidden.dtype)

        h = q_hidden.to(dtype=target_dtype)
        for blk in self.blocks:
            h = blk(h)
        h_norm = self.final_norm(h)
        pooled = self.pool_proj(h_norm.mean(dim=1))

        if num_frames == 1:
            seq_cond = self.seq_proj(h_norm)  # (B, Q, dit_cond_dim)
        else:
            # Broadcast temporal frame embeddings across F frames -> (B, F, Q, dit_cond_dim) for VAE 3D / animated GIF DiT
            t_emb = self.temporal_frame_emb[:num_frames].to(dtype=h_norm.dtype)[None, :, None, :]  # (1, F, 1, d_model)
            h_frames = self.final_norm(h_norm.unsqueeze(1) + t_emb)  # (B, F, Q, d_model)
            seq_cond = self.seq_proj(h_frames)

        return MetaQueryOutput(
            seq_cond=seq_cond,
            pooled_cond=pooled,
            query_hidden=q_hidden,
            state=out_q.state,
        )


def compute_metaquery_alignment_loss(
    output: MetaQueryOutput,
    target_seq_cond: torch.Tensor,
    target_pooled_cond: torch.Tensor | None = None,
    *,
    cosine_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """FP32 alignment loss matching MetaQuery outputs to target DiT / Anima 2B conditioning latents."""
    pred_s = output.seq_cond.float()
    targ_s = target_seq_cond.detach().float()
    if pred_s.shape != targ_s.shape:
        raise ValueError(f"seq_cond shape mismatch: {pred_s.shape} != {targ_s.shape}")
    mse_s = F.mse_loss(pred_s, targ_s)
    cos_s = 1.0 - F.cosine_similarity(pred_s, targ_s, dim=-1, eps=1e-6).mean()
    total = mse_s + float(cosine_weight) * cos_s
    metrics = {"seq_mse": float(mse_s.detach().item()), "seq_cos_dist": float(cos_s.detach().item())}

    if target_pooled_cond is not None:
        pred_p = output.pooled_cond.float()
        targ_p = target_pooled_cond.detach().float()
        if pred_p.shape != targ_p.shape:
            raise ValueError(f"pooled_cond shape mismatch: {pred_p.shape} != {targ_p.shape}")
        mse_p = F.mse_loss(pred_p, targ_p)
        total = total + 0.5 * mse_p
        metrics["pooled_mse"] = float(mse_p.detach().item())

    metrics["total_loss"] = float(total.detach().item())
    return total, metrics
