"""Algorithm Protocol — advantage computation and policy loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vrl.algorithms.types import TrainStepMetrics

if TYPE_CHECKING:
    from vrl.algorithms.trajectory import AlgorithmInput


class Algorithm(Protocol):
    """Structural interface for RL algorithms (GRPO, REINFORCE, etc.).

    CEA pipeline interface:
    - compute_advantages_from_tensors(rewards, group_ids)
    - compute_loss(inputs)
    """

    def compute_advantages_from_tensors(
        self,
        rewards: Any,        # [B] tensor
        group_ids: Any,      # [B] tensor — prompt group assignment
    ) -> Any:                # [B] tensor of advantages
        """Compute per-sample advantages from reward tensors."""
        ...

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute loss from strict trajectory-native algorithm inputs."""
        ...
