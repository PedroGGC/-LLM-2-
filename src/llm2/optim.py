"""Optimizer and step-indexed LR schedule, independent of backbone and loss."""

import math

import torch


import math


def _zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16() if grad.is_cuda else grad.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        x = a * x + (b * aa + c * (aa @ aa)) @ x
    if transposed:
        x = x.mT
    return x.to(dtype=grad.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Muon (NS-orthogonalized Nesterov) on 2D hidden matrices + decoupled AdamW elsewhere."""

    def __init__(self, param_groups, *, lr: float, betas: tuple[float, float], eps: float, momentum: float = 0.95, ns_steps: int = 5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0, momentum=momentum, ns_steps=ns_steps, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            if group.get("use_muon", False):
                momentum = group["momentum"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(p)
                        state["momentum_buffer"] = buf
                    buf.lerp_(g, 1.0 - momentum)
                    u = g.lerp(buf, momentum)
                    u = _zeropower_via_newtonschulz5(u, steps=ns_steps)
                    scale = 0.2 * math.sqrt(max(p.size(0), p.size(1)))
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(u, alpha=-lr * scale)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.float()
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"] += 1
                    step_idx = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.lerp_(g, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    step_size = lr / (1.0 - beta1**step_idx)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1.0 - beta2**step_idx)).add_(eps)
                    p.add_((exp_avg / denom).to(dtype=p.dtype), alpha=-step_size)
        return loss


def build_optimizer(model, objective, config):
    seen, muon_params, decay, no_decay = set(), [], [], []
    adam_only = set()
    for owner in (model, objective):
        for attr in ("embedding", "lm_head", "mtp_heads"):
            sub = getattr(owner, attr, None)
            if sub is not None:
                for p in sub.parameters():
                    adam_only.add(id(p))
    use_muon = getattr(config, "kind", "adamw") == "muon"
    for module in (model, objective):
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if use_muon and parameter.ndim == 2 and id(parameter) not in adam_only:
                muon_params.append(parameter)
            elif parameter.ndim >= 2:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
    if use_muon:
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "weight_decay": config.weight_decay, "use_muon": True})
        if decay:
            groups.append({"params": decay, "weight_decay": config.weight_decay, "use_muon": False})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0, "use_muon": False})
        return MuonAdamW(groups, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
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

