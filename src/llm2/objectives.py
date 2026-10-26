"""Objectives are independent of architecture and optimization."""

from torch.nn import functional as F

from .interfaces import Objective, ObjectiveOutput


class CausalLanguageModeling(Objective):
    def forward(self, model, batch):
        output = model(batch["input_ids"])
        targets = batch["targets"]
        if output.logits is None:
            raise ValueError("causal language modeling requires vocabulary logits")
        if output.logits.shape[:-1] != targets.shape:
            raise ValueError("logits and shifted targets have incompatible shapes")
        # Dense packed data has no padding; every shifted target is supervised.
        loss = F.cross_entropy(output.logits.float().flatten(0, 1), targets.flatten())
        return ObjectiveOutput(loss=loss, token_count=targets.numel())

