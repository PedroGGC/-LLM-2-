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
        token_losses = F.cross_entropy(output.logits.float().flatten(0, 1), targets.flatten(),
                                       reduction="none", ignore_index=-100).view_as(targets)
        mask = targets != -100
        count = int(mask.sum())
        loss = token_losses.sum() / count if count else output.logits.sum() * 0
        return ObjectiveOutput(loss=loss, token_count=count, language_loss=loss,
                               token_losses=token_losses)
