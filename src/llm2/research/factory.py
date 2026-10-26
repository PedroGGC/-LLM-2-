"""One explicit construction path shared by CLI, trainer and inference."""

from ..model import TransformerLM
from ..objectives import CausalLanguageModeling


def build_backbone(config):
    research = config.research
    if research is None or research.architecture == "transformer":
        return TransformerLM(config.model)
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
