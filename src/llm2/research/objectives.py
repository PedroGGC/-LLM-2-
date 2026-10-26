"""Predictive hidden-space and multi-offset auxiliaries, independent of textual generation."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from ..interfaces import Objective, ObjectiveOutput


class PredictiveObjective(Objective):
    def __init__(self, model, research):
        super().__init__()
        self.research = research
        dim, vocab = model.config.d_model, model.config.vocab_size
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))
        if research.jepa_weight:
            self.target = copy.deepcopy(model).requires_grad_(False).eval()
            self.predictor = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU(),
                                           nn.Linear(dim, dim, bias=False))
        self.mtp_heads = nn.ModuleList([nn.Linear(dim, vocab, bias=False)
                                       for _ in range(research.mtp_tokens)])

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "target"):
            self.target.eval()
        return self

    @torch.no_grad()
    def after_optimizer_step(self, model):
        if hasattr(self, "target") and self.research.use_ema:
            online = dict(model.named_parameters())
            for name, parameter in self.target.named_parameters():
                parameter.lerp_(online[name].detach(), 1 - self.research.ema_decay)
            buffers = dict(model.named_buffers())
            for name, value in self.target.named_buffers():
                value.copy_(buffers[name])
            self.ema_updates.add_(1)

    @staticmethod
    def _ce(logits, targets):
        if not (targets != -100).any():
            return logits.sum() * 0
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)

    def forward(self, model, batch):
        r = self.research
        output = model(batch["input_ids"])
        lm = self._ce(output.logits, batch["targets"])
        loss = r.lm_weight * lm
        metrics = {"lm_loss": float(lm.detach())}
        hidden = output.hidden_states
        if r.jepa_weight:
            with torch.no_grad():
                target = self.target(batch["input_ids"]).hidden_states[:, r.jepa_offset:].float()
                target = F.layer_norm(target, (target.shape[-1],))
            online = hidden[:, :-r.jepa_offset].float()
            predicted = self.predictor(online)
            predicted = F.layer_norm(predicted.float(), (predicted.shape[-1],))
            mask = batch["targets"][:, r.jepa_offset:] != -100
            predictive = F.mse_loss(predicted[mask], target.detach()[mask]) if mask.any() else predicted.sum()*0
            selected = online[mask]
            if selected.shape[0] == 0:
                selected = online.flatten(0, 1)[:1]
            std = torch.sqrt(selected.var(0, unbiased=False) + 1e-4)
            variance_penalty = F.relu(r.variance_target - std).mean()
            auxiliary = predictive + r.variance_weight * variance_penalty
            loss = loss + r.jepa_weight * auxiliary
            metrics.update(jepa_loss=float(predictive.detach()), variance_penalty=float(variance_penalty.detach()),
                           representation_std=float(std.mean().detach()),
                           collapse_fraction=float((std < 0.05).float().mean().detach()),
                           target_std=float(target.flatten(0, 1).std(0, unbiased=False).mean()),
                           ema_updates=int(self.ema_updates))
        if r.mtp_weight:
            losses = []
            for offset, head in enumerate(self.mtp_heads, start=1):
                # targets[t] = token[t+1]; offset 1 therefore predicts token[t+2].
                losses.append(self._ce(head(hidden[:, :-offset]), batch["targets"][:, offset:]))
            mtp = torch.stack(losses).mean()
            loss = loss + r.mtp_weight * mtp
            metrics["mtp_loss"] = float(mtp.detach())
        if r.intermediate_weight:
            if not output.intermediate_states:
                raise ValueError("intermediate supervision requires a recursive backbone")
            intermediate = torch.stack([self._ce(model.lm_head(h), batch["targets"])
                                        for h in output.intermediate_states]).mean()
            loss = loss + r.intermediate_weight * intermediate
            metrics["intermediate_loss"] = float(intermediate.detach())
        if output.state is not None and hasattr(output.state, "iterations"):
            metrics["internal_iterations"] = output.state.iterations
        return ObjectiveOutput(loss, int((batch["targets"] != -100).sum()), metrics)
