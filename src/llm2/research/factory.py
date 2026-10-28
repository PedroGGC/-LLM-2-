"""One explicit construction path shared by CLI, trainer and inference."""

from ..model import TransformerLM
from ..objectives import CausalLanguageModeling


def build_backbone(config):
    research = config.research
    if research is None or research.architecture == "transformer":
        return TransformerLM(config.model)
    if getattr(research, "paired_common_init", False):
        if research.architecture not in ("recursive", "fractal"):
            raise ValueError("paired_common_init is only supported for recursive and fractal architectures")
        from .pairing import build_paired_recurrent_backbone
        return build_paired_recurrent_backbone(config)
    if research.architecture in ("recursive", "untied"):
        from ..architectures.latent_recursive import LatentRecursiveModel
        return LatentRecursiveModel(config.model, research)
    if research.architecture == "fractal":
        try:
            from ..architectures.fractal_recursive import FractalLatentModel
        except ImportError as exc:
            raise ValueError("fractal implementation is isolated on experiment/fractal-lrm") from exc
        return FractalLatentModel(config.model, research)
    raise ValueError("unsupported architecture")


def build_objective(config, model):
    r = config.research
    if r is not None and (r.jepa_weight or r.mtp_weight or r.intermediate_weight or r.lm_weight != 1):
        from .objectives import PredictiveObjective
        return PredictiveObjective(model, r)
    return CausalLanguageModeling()
