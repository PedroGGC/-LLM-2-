"""Vision bridge for Lumina V1: frozen SigLIP 2 + 2x2/4x4 pixel-unshuffle + 2D pos + MLP mmproj + GRM <img_end> anchors."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..architectures.fractal_delta import FractalDecode, FractalDeltaMemory
from ..model import RMSNorm
from ..research.decode_optimization import SegmentMemoryCache


class MockVisionEncoder(nn.Module):
    """Lightweight deterministic vision encoder for unit tests (no downloads, preserves 2D grid)."""

    def __init__(self, vision_dim: int = 64, grid_size: int = 8, patch_size: int = 16):
        super().__init__()
        self.vision_dim = int(vision_dim)
        self.grid_size = int(grid_size)
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(3, self.vision_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns 2D patch grid (B, H_p, W_p, vision_dim)."""
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape (B, 3, H, W)")
        x = pixel_values.to(dtype=self.proj.weight.dtype)
        feat = self.proj(x)  # (B, D_vis, H_p, W_p)
        return feat.permute(0, 2, 3, 1).contiguous()


class SigLIPVisionEncoder(nn.Module):
    """Frozen HuggingFace SigLIP / SigLIP 2 vision encoder returning 2D patch grids (B, H_p, W_p, D_vis)."""

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        *,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.model_id = str(model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        full = AutoModel.from_pretrained(self.model_id, torch_dtype=dtype)
        self.vision_model = full.vision_model if hasattr(full, "vision_model") else full
        self.vision_model.eval()
        self.vision_model.requires_grad_(False)
        self.vision_dim = int(self.vision_model.config.hidden_size)
        self.patch_size = int(getattr(self.vision_model.config, "patch_size", 16))

    def preprocess_pil(self, images, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        out = self.processor(images=images, return_tensors="pt")
        pix = out["pixel_values"].to(device=device)
        if dtype is not None:
            pix = pix.to(dtype=dtype)
        return pix

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape (B, 3, H, W)")
        param = next(self.vision_model.parameters())
        pix = pixel_values.to(device=param.device, dtype=param.dtype)
        out = self.vision_model(pixel_values=pix)
        tokens = out.last_hidden_state  # (B, N, D_vis)
        b, n, d = tokens.shape
        h_p = pix.shape[2] // self.patch_size
        w_p = pix.shape[3] // self.patch_size
        if h_p * w_p != n:
            side = int(math.isqrt(n))
            if side * side != n:
                raise ValueError(f"Cannot reshape {n} vision tokens into 2D grid for {pix.shape}")
            h_p, w_p = side, side
        return tokens.reshape(b, h_p, w_p, d).contiguous()


class VisionSpatialProjector(nn.Module):
    """2x2 or 4x4 spatial pixel-unshuffle compressor + explicit 2D positional embedding + MLP mmproj + FP32 RMSNorm."""

    def __init__(
        self,
        vision_dim: int,
        d_model: int,
        *,
        downsample_factor: int = 2,
        max_grid: int = 64,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if downsample_factor not in (1, 2, 4):
            raise ValueError("downsample_factor must be 1, 2, or 4")
        self.vision_dim = int(vision_dim)
        self.d_model = int(d_model)
        self.downsample_factor = int(downsample_factor)
        self.max_grid = int(max_grid)

        in_dim = self.vision_dim * (self.downsample_factor ** 2)
        self.row_pos = nn.Parameter(torch.zeros(self.max_grid, self.d_model))
        self.col_pos = nn.Parameter(torch.zeros(self.max_grid, self.d_model))
        nn.init.normal_(self.row_pos, std=0.02)
        nn.init.normal_(self.col_pos, std=0.02)

        self.pre_norm = RMSNorm(in_dim, eps=norm_eps)
        self.fc1 = nn.Linear(in_dim, self.d_model * 2, bias=False)
        self.fc2 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_norm = RMSNorm(self.d_model, eps=norm_eps)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def _pixel_unshuffle_2d(self, grid: torch.Tensor) -> torch.Tensor:
        """Compress (B, H, W, D) -> (B, ceil(H/f), ceil(W/f), D*f^2) preserving exact 2D neighborhood."""
        b, h, w, d = grid.shape
        f = self.downsample_factor
        if f == 1:
            return grid
        pad_h = (-h) % f
        pad_w = (-w) % f
        if pad_h or pad_w:
            # Pad right/bottom edges cleanly in (B, D, H, W) layout
            x = grid.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            grid = x.permute(0, 2, 3, 1)
            h += pad_h
            w += pad_w
        gh, gw = h // f, w // f
        return grid.reshape(b, gh, f, gw, f, d).permute(0, 1, 3, 2, 4, 5).reshape(b, gh, gw, d * f * f)

    def _pos_2d(self, gh: int, gw: int, dtype: torch.dtype) -> torch.Tensor:
        if gh <= self.max_grid and gw <= self.max_grid:
            r = self.row_pos[:gh]
            c = self.col_pos[:gw]
        else:
            r = F.interpolate(
                self.row_pos.T.unsqueeze(0), size=gh, mode="linear", align_corners=False
            ).squeeze(0).T
            c = F.interpolate(
                self.col_pos.T.unsqueeze(0), size=gw, mode="linear", align_corners=False
            ).squeeze(0).T
        return (r[:, None, :] + c[None, :, :]).to(dtype=dtype)  # (gh, gw, d_model)

    def forward(self, patch_grid: torch.Tensor) -> torch.Tensor:
        if patch_grid.ndim != 4 or patch_grid.shape[-1] != self.vision_dim:
            raise ValueError(f"patch_grid must have shape (B, H_p, W_p, {self.vision_dim})")
        compressed = self._pixel_unshuffle_2d(patch_grid)  # (B, gh, gw, in_dim)
        b, gh, gw, _ = compressed.shape
        h = self.pre_norm(compressed)
        gate, up = self.fc1(h).chunk(2, dim=-1)
        proj = self.fc2(F.silu(gate) * up)
        proj = proj + self._pos_2d(gh, gw, proj.dtype).unsqueeze(0)
        proj = self.out_norm(proj)
        return proj.reshape(b, gh * gw, self.d_model).contiguous()


class LuminaMultimodalWrapper(nn.Module):
    """Decoupled VLM wrapper injecting SigLIP 2 + VisionSpatialProjector embeddings into FractalDeltaMemory."""

    def __init__(
        self,
        backbone: FractalDeltaMemory,
        vision_encoder: nn.Module,
        *,
        downsample_factor: int = 2,
    ):
        super().__init__()
        if not isinstance(backbone, FractalDeltaMemory):
            raise ValueError("LuminaMultimodalWrapper requires a FractalDeltaMemory backbone")
        self.backbone = backbone
        self.vision_encoder = vision_encoder
        self.vision_encoder.requires_grad_(False)
        d = backbone.config.d_model
        vision_dim = int(getattr(vision_encoder, "vision_dim"))
        self.projector = VisionSpatialProjector(
            vision_dim,
            d,
            downsample_factor=downsample_factor,
            norm_eps=backbone.config.norm_eps,
        )
        # Learnable boundary markers <img_start> and <img_end> without resizing the BPE vocabulary.
        self.img_start_emb = nn.Parameter(torch.empty(1, 1, d))
        self.img_end_emb = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.img_start_emb, std=0.02)
        nn.init.normal_(self.img_end_emb, std=0.02)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode (B, 3, H, W) images -> (B, 2 + N_vis, d_model) wrapped with <img_start> ... <img_end>."""
        with torch.no_grad():
            grid = self.vision_encoder(pixel_values)
        vis_tokens = self.projector(grid.to(dtype=self.projector.fc1.weight.dtype))
        b = vis_tokens.shape[0]
        start = self.img_start_emb.to(dtype=vis_tokens.dtype).expand(b, 1, -1)
        end = self.img_end_emb.to(dtype=vis_tokens.dtype).expand(b, 1, -1)
        return torch.cat((start, vis_tokens, end), dim=1)

    def build_multimodal_inputs(
        self,
        *,
        images: Sequence[torch.Tensor],
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[int, int]]]:
        """Interleave [img_0, ..., img_k, prompt_ids, response_ids] into (inputs_embeds, targets, image_spans).

        Visual tokens and prompt tokens receive target=-100 so CE loss supervises only the response.
        """
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape (B, T_prompt)")
        b = prompt_ids.shape[0]
        device = prompt_ids.device
        segments: list[torch.Tensor] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for img in images:
            vis = self.encode_images(img.to(device=device))
            if vis.shape[0] != b:
                raise ValueError("image batch size must match prompt_ids batch size")
            segments.append(vis)
            spans.append((cursor, cursor + vis.shape[1]))
            cursor += vis.shape[1]

        prompt_emb = self.backbone.embedding(prompt_ids)
        target_dtype = prompt_emb.dtype
        segments = [seg.to(dtype=target_dtype) for seg in segments]
        segments.append(prompt_emb)
        prompt_end = cursor + prompt_ids.shape[1]

        if response_ids is None:
            return torch.cat(segments, dim=1), None, spans

        if response_ids.ndim != 2 or response_ids.shape[0] != b:
            raise ValueError("response_ids must have shape (B, T_resp)")
        resp_emb = self.backbone.embedding(response_ids)
        full_embeds = torch.cat((*segments, resp_emb), dim=1)
        # Next-token shift: inputs = full[:-1], targets = [-100 ... -100, response_ids] aligned so last prompt token predicts response[0].
        inputs = full_embeds[:, :-1, :].contiguous()
        t_in = inputs.shape[1]
        targets = torch.full((b, t_in), -100, dtype=torch.long, device=device)
        resp_start = prompt_end - 1
        resp_len = response_ids.shape[1]
        if resp_start < 0 or resp_start + resp_len > t_in:
            raise ValueError("prompt_ids must contain at least 1 token to anchor response prediction")
        targets[:, resp_start : resp_start + resp_len] = response_ids
        return inputs, targets, spans

    def forward(
        self,
        *,
        images: Sequence[torch.Tensor],
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor | None = None,
        state=None,
        return_logits: bool = True,
    ):
        embeds, targets, spans = self.build_multimodal_inputs(
            images=images,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
        )
        out = self.backbone(inputs_embeds=embeds, state=state, return_logits=return_logits)
        return out, targets, spans

    @torch.no_grad()
    def prefill_multimodal(
        self,
        *,
        images: Sequence[torch.Tensor],
        prompt_ids: torch.Tensor,
        grm_cache: SegmentMemoryCache | None = None,
        image_tags: Sequence[str] | None = None,
    ):
        """Stream each image/GIF frame into O(1) FractalDeltaState, freezing <img_end> anchors in GRM."""
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("prefill_multimodal requires batch=1")
        device = prompt_ids.device
        dec = FractalDecode(self.backbone)
        state = None
        last_logits = None
        last_hidden = None

        for idx, img in enumerate(images):
            vis = self.encode_images(img.to(device=device)).to(dtype=self.backbone.embedding.weight.dtype)
            if state is None:
                out = self.backbone(inputs_embeds=vis, state=None, return_logits=False)
                state = out.state
                vis_hidden = out.hidden_states
            else:
                _, vis_hidden, state = dec.step_hidden(inputs_embeds=vis, state=state, return_all_states=False)
            if grm_cache is not None:
                tag = image_tags[idx] if (image_tags is not None and idx < len(image_tags)) else f"img_{idx}"
                grm_cache.record_segment(state, vis_hidden, tag=tag)

        prompt_emb = self.backbone.embedding(prompt_ids)
        if state is None:
            last_logits, last_hidden, state = dec.prefill_embeds(prompt_emb, capacity=self.backbone.config.max_seq_len)
        else:
            logits_p, hidden_p, state = dec.step_hidden(inputs_embeds=prompt_emb, state=state, return_all_states=False)
            last_logits = logits_p[:, -1:]
            last_hidden = hidden_p[:, -1:]
        return last_logits, last_hidden, state
