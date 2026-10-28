"""Causal Neural Context Memory (NCM) with learnable semantic prototype slots.

Positions query fixed global prototypes independently, preserving strict prefix causality.
"""

import math

import torch
from torch import nn


class NeuralContextMemory(nn.Module):
    r"""Lightweight causal prototype memory bank ($M$ slots in $\mathbb{R}^D$)."""

    def __init__(self, slots: int, dim: int, *, init_parameters: bool = True):
        super().__init__()
        if slots <= 0 or dim <= 0:
            raise ValueError("NCM slots and dim must be positive")
        self.slots = slots
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        self.memory = nn.Parameter(torch.zeros(slots, dim))
        if init_parameters:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.memory, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(z, self.memory.t()) * self.scale
        weights = torch.softmax(scores.float(), dim=-1).to(z.dtype)
        return torch.add(z, torch.matmul(weights, self.memory), alpha=0.1)
