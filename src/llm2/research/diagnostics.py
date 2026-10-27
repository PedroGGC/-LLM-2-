"""Optional, bounded representation diagnostics outside training and benchmark timing."""

import torch

from ..runtime import capture_rng, restore_rng, attention_context, autocast_context


def linear_cka(x, y):
    """Centered linear CKA; undefined for a constant representation."""
    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]
    x, y = x.double()-x.double().mean(0), y.double()-y.double().mean(0)
    a, b = x @ x.T, y @ y.T
    denominator = torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float((a*b).sum()/denominator) if denominator > 0 else None


def representation_stats(value):
    x = value.float()
    centered = x-x.mean(0)
    eigenvalues = torch.linalg.svdvals(centered.double()).square()
    total = eigenvalues.sum()
    if total > 0:
        probabilities = eigenvalues/total
        positive = probabilities[probabilities > 0]
        effective_rank = float((-(positive*positive.log()).sum()).exp())
        participation = float(total.square()/eigenvalues.square().sum())
    else:
        effective_rank = participation = 0.0
    return {"rms": float(x.square().mean().sqrt()),
            "mean_feature_std": float(x.std(0, unbiased=False).mean()),
            "effective_rank_entropy": effective_rank, "participation_rank": participation,
            "samples": x.shape[0], "dimension": x.shape[1]}


@torch.no_grad()
def diagnose_model(model, input_ids, *, max_samples=128, precision="fp32", deterministic=True):
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("diagnostics need at least two samples")
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    snapshots = []
    hook = None

    def capture(module, args, output):
        states = output if isinstance(output, tuple) else (output,)
        snapshots.append(tuple(state.detach().float().flatten(0, 1)[:max_samples].cpu().clone()
                               for state in states))

    # A fractal block returns all scales; a shared LRM core returns one state.
    core = getattr(model, "block", getattr(model, "core", None))
    if core is not None:
        hook = core.register_forward_hook(capture)
    try:
        model.eval()
        with attention_context(deterministic), autocast_context(input_ids.device, precision):
            kwargs = {"return_logits": False} if getattr(model, "supports_hidden_only", False) else {}
            output = model(input_ids, **kwargs)
        if not snapshots:
            snapshots.append((output.hidden_states.detach().float().flatten(0, 1)[:max_samples].cpu(),))
        final = snapshots[-1]
        if any(not torch.isfinite(value).all() for value in final):
            raise FloatingPointError("non-finite diagnostic states")
        history = []
        previous = None
        for iteration, values in enumerate(snapshots, 1):
            history.append({"iteration": iteration,
                            "rms_by_scale": [float(x.square().mean().sqrt()) for x in values],
                            "relative_update_by_scale": [
                                float(torch.linalg.vector_norm(x-old)/(torch.linalg.vector_norm(old)+1e-8))
                                for x, old in zip(values, previous)] if previous is not None else None})
            previous = values
        result = {"scales": [representation_stats(value) for value in final],
                  "linear_cka": [[linear_cka(x, y) for y in final] for x in final],
                  "iterations": history,
                  "gates": {name: parameter.sigmoid().detach().cpu().tolist()
                            for name, parameter in model.named_parameters()
                            if name.endswith("gate")},
                  "max_samples": max_samples, "device": str(input_ids.device), "precision": precision,
                  "limitations": "Bounded states from one fixed validation batch; raw recurrent states for LRM/FLRM, final hidden for Transformer. CKA/rank are descriptive, not proof of semantic specialization or quality. No training gradients or benchmark timing are included."}
        if hasattr(model, "block"):
            result["cross_scale_mixing"] = model.block.interaction.mixing.softmax(-1).cpu().tolist()
            if model.block.conditioning is not None:
                film = model.block.conditioning.film.float()
                result["film"] = {"gamma_gain_min": float((1+.1*film[:, 0].tanh()).min()),
                                  "gamma_gain_max": float((1+.1*film[:, 0].tanh()).max()),
                                  "beta_rms": float((.1*film[:, 1]).square().mean().sqrt())}
        return result
    finally:
        if hook is not None:
            hook.remove()
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
