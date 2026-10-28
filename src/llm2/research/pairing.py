"""Explicit common-weight pairing protocol for Stage 3 recurrent and multiscale controls.

When `research.paired_common_init=True`, all shared-weight recurrent backbones
(`LatentRecursiveModel` and `FractalLatentModel`) constructed with the same
`(model_config, core_layers, seed)` receive bit-for-bit identical common parameters:
- `embedding.weight` (tied to `lm_head.weight` within each model)
- `norm.weight`
- `core.*` / `block.cores.0.*` (`state_norm`, `context_projection`, `DecoderBlock`s, `gate`)

Exclusive multiscale parameters in `FractalLatentModel` (`conditioning`, `interaction`,
`comp4`, `comp16`, `ncm`) are initialized deterministically in an isolated RNG scope so
they neither shift common weights nor alter the post-initialization training RNG state.
"""

from __future__ import annotations

import hashlib
import torch

from ..config import ResearchConfig


FLRM_EXCLUSIVE_SEED_OFFSET = 10007


def _tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.numpy().tobytes())
    return h.hexdigest()


def iter_common_named_parameters(model):
    """Yield `(canonical_name, parameter)` for all parameters shared between LRM and FLRM."""
    yield "embedding.weight", model.embedding.weight
    yield "norm.weight", model.norm.weight
    core = model.core if hasattr(model, "core") else model.block.cores[0]
    for name, param in core.named_parameters():
        yield f"core.{name}", param


def iter_common_named_buffers(model):
    """Yield `(canonical_name, buffer)` for all buffers in the shared recurrent core."""
    core = model.core if hasattr(model, "core") else model.block.cores[0]
    for name, buf in core.named_buffers():
        yield f"core.{name}", buf


def iter_exclusive_named_parameters(model):
    """Yield `(name, parameter)` for parameters present only in the multiscale FLRM package."""
    common_ids = {id(p) for _, p in iter_common_named_parameters(model)}
    for name, param in model.named_parameters():
        if id(param) not in common_ids:
            yield name, param


def compute_paired_init_provenance(model, *, seed: int) -> dict:
    """Compute SHA-256 hashes and parameter counts for common and exclusive tensors."""
    on_meta = model.embedding.weight.device.type == "meta"
    is_fractal = hasattr(model, "block")
    common_hashes = {}
    common_count = 0
    for cname, param in iter_common_named_parameters(model):
        common_count += param.numel()
        common_hashes[cname] = None if on_meta else _tensor_sha256(param)

    exclusive_hashes = {}
    exclusive_count = 0
    for ename, param in iter_exclusive_named_parameters(model):
        exclusive_count += param.numel()
        exclusive_hashes[ename] = None if on_meta else _tensor_sha256(param)

    combined = None
    if not on_meta:
        comb_hasher = hashlib.sha256()
        for cname in sorted(common_hashes):
            comb_hasher.update(f"{cname}:{common_hashes[cname]}\n".encode("utf-8"))
        combined = comb_hasher.hexdigest()

    excl_combined = None
    if not on_meta and exclusive_hashes:
        excl_hasher = hashlib.sha256()
        for ename in sorted(exclusive_hashes):
            excl_hasher.update(f"{ename}:{exclusive_hashes[ename]}\n".encode("utf-8"))
        excl_combined = excl_hasher.hexdigest()

    return {
        "protocol": "paired_common_lrm_reference_v1",
        "paired_common_init": True,
        "seed": int(seed),
        "flrm_exclusive_seed": int((seed + FLRM_EXCLUSIVE_SEED_OFFSET) % (2**32)) if is_fractal else None,
        "tied_embedding_lm_head": bool(model.lm_head.weight is model.embedding.weight),
        "common_parameter_tensor_count": int(len(common_hashes)),
        "common_parameter_count": int(common_count),
        "exclusive_parameter_tensor_count": int(len(exclusive_hashes)),
        "exclusive_parameter_count": int(exclusive_count),
        "total_backbone_parameter_count": int(common_count + exclusive_count),
        "common_combined_sha256": combined,
        "exclusive_combined_sha256": excl_combined,
        "common_parameter_sha256": common_hashes,
        "exclusive_parameter_sha256": exclusive_hashes,
    }


def build_paired_recurrent_backbone(config):
    """Construct `LatentRecursiveModel` or `FractalLatentModel` with explicit common-weight pairing."""
    from ..architectures.latent_recursive import LatentRecursiveModel

    research = config.research
    if research is None or research.architecture not in ("recursive", "fractal"):
        raise ValueError("paired_common_init requires a recursive or fractal architecture")
    if not research.share_weights:
        raise ValueError("paired_common_init requires share_weights=True")
    if research.architecture == "fractal" and not research.share_scales:
        raise ValueError("paired_common_init requires share_scales=True for fractal models")

    seed = int(config.train.seed)
    ref_research = ResearchConfig(
        architecture="recursive",
        iterations=1,
        core_layers=research.core_layers,
        share_weights=True,
    )

    # Check if we are inside a `torch.device("meta")` context.
    probe = torch.empty(0)
    if probe.device.type == "meta":
        if research.architecture == "recursive":
            model = LatentRecursiveModel(config.model, research)
        else:
            from ..architectures.fractal_recursive import FractalLatentModel
            model = FractalLatentModel(config.model, research)
        model.paired_init_provenance = compute_paired_init_provenance(model, seed=seed)
        return model

    # 1. Build canonical reference LRM in a deterministic seed scope and capture post-ref CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        ref_lrm = LatentRecursiveModel(config.model, ref_research)
        post_ref_cpu_rng = torch.get_rng_state()

    # 2. Construct target model in an isolated deterministic scope so exclusive modules never shift RNG.
    with torch.random.fork_rng(devices=[]):
        if research.architecture == "recursive":
            torch.manual_seed(seed)
            target_model = LatentRecursiveModel(config.model, research)
        else:
            from ..architectures.fractal_recursive import FractalLatentModel
            exclusive_seed = (seed + FLRM_EXCLUSIVE_SEED_OFFSET) % (2**32)
            torch.manual_seed(exclusive_seed)
            target_model = FractalLatentModel(config.model, research)

    # 3. Explicitly copy all common parameters and buffers without sharing storage (`data_ptr`).
    with torch.no_grad():
        ref_params = dict(iter_common_named_parameters(ref_lrm))
        dst_params = dict(iter_common_named_parameters(target_model))
        if set(ref_params.keys()) != set(dst_params.keys()):
            raise RuntimeError(
                f"common parameter keys mismatch: {sorted(ref_params.keys())} vs {sorted(dst_params.keys())}"
            )
        for cname, src_p in ref_params.items():
            dst_p = dst_params[cname]
            if src_p.shape != dst_p.shape or src_p.dtype != dst_p.dtype:
                raise RuntimeError(
                    f"common parameter {cname} shape/dtype mismatch: {src_p.shape}/{src_p.dtype} vs {dst_p.shape}/{dst_p.dtype}"
                )
            dst_p.copy_(src_p.detach().clone())
            if dst_p.data_ptr() == src_p.data_ptr():
                raise RuntimeError(f"common parameter {cname} unexpectedly shares storage with reference")

        ref_buffers = dict(iter_common_named_buffers(ref_lrm))
        dst_buffers = dict(iter_common_named_buffers(target_model))
        if set(ref_buffers.keys()) != set(dst_buffers.keys()):
            raise RuntimeError(
                f"common buffer keys mismatch: {sorted(ref_buffers.keys())} vs {sorted(dst_buffers.keys())}"
            )
        for bname, src_b in ref_buffers.items():
            dst_b = dst_buffers[bname]
            dst_b.copy_(src_b.detach().clone())

        # Ensure intra-model embedding/output tying remains intact.
        target_model.lm_head.weight = target_model.embedding.weight
        if target_model.lm_head.weight is not target_model.embedding.weight:
            raise RuntimeError("embedding/lm_head weight tying was broken")

    # 4. Advance global CPU RNG to the exact post-reference state so all paired models leave RNG identical.
    torch.set_rng_state(post_ref_cpu_rng)

    target_model.paired_init_provenance = compute_paired_init_provenance(target_model, seed=seed)
    return target_model
