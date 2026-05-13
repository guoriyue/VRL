"""Algorithm ABC — advantage computation and policy loss."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from vrl.algorithms.types import TrainStepMetrics

if TYPE_CHECKING:
    from vrl.algorithms.trajectory import AlgorithmInput


class Algorithm(ABC):
    """Base class for RL algorithms (GRPO, REINFORCE, etc.).

    CEA pipeline interface:
    - compute_advantages_from_tensors(rewards, group_ids)
    - compute_loss(inputs)
    """

    @abstractmethod
    def compute_advantages_from_tensors(
        self,
        rewards: Any,        # [B] tensor
        group_ids: Any,      # [B] tensor — prompt group assignment
    ) -> Any:                # [B] tensor of advantages
        """Compute per-sample advantages from reward tensors."""

    @abstractmethod
    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute loss from strict trajectory-native algorithm inputs."""
