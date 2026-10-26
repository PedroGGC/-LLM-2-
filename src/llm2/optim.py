"""Optimizer and step-indexed LR schedule, independent of backbone and loss."""

import math

import torch


def build_optimizer(model, objective, config):
    seen, decay, no_decay = set(), [], []
    for module in (model, objective):
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, foreach=False)


def build_scheduler(optimizer, config):
    def multiplier(completed_steps):
        if completed_steps < config.warmup_steps:
            return (completed_steps + 1) / config.warmup_steps
        progress = min(1.0, (completed_steps - config.warmup_steps) /
                       (config.schedule_steps - config.warmup_steps))
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

