"""Extension contracts only; experimental objectives/states are not implemented."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn


@dataclass
class ModelOutput:
    logits: Tensor | None = None
    hidden_states: Tensor | None = None
    state: Any = None
    intermediate_states: tuple[Tensor, ...] = ()


@dataclass
class ObjectiveOutput:
    loss: Tensor
    token_count: int
    metrics: dict[str, float] = field(default_factory=dict)


class Backbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, input_ids: Tensor, *, state: Any = None) -> ModelOutput:
        """State is opaque to the trainer; future backbones may define its semantics."""


class AttentionKernel(nn.Module, ABC):
    @abstractmethod
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Input/output [batch, heads, time, head_dim]; kernel owns its masking."""


class Objective(nn.Module, ABC):
    def after_optimizer_step(self, model: nn.Module):
        """Optional update of frozen targets; called once per completed optimizer step."""

    @abstractmethod
    def forward(self, model: nn.Module, batch: dict[str, Tensor]) -> ObjectiveOutput:
        """Own model invocation and loss; trainer does not assume autoregression."""

