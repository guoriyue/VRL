"""Algorithm adapters for trajectory-native training inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vrl.algorithms.types import TrainStepMetrics

if TYPE_CHECKING:
    from vrl.algorithms.base import Algorithm
    from vrl.rollouts.evaluators.types import TrajectorySignalBatch


@dataclass(slots=True)
class AlgorithmInput:
    """Unified algorithm-facing input derived from trajectory contracts."""

    signals: TrajectorySignalBatch | None = None
    rewards: Any | None = None
    group_ids: Any | None = None
    advantages: Any | None = None
    model: Any | None = None
    rollout_batch: Any | None = None
    timestep_index: int | None = None


class AlgorithmAdapter:
    """Dispatch strict AlgorithmInput to objective-specific native APIs."""

    def compute_advantages(self, algorithm: Algorithm, inputs: AlgorithmInput) -> Any:
        if inputs.advantages is not None:
            return inputs.advantages
        if inputs.rewards is None:
            raise RuntimeError("AlgorithmInput.rewards is required to compute advantages")
        group_ids = inputs.group_ids
        if group_ids is None and inputs.signals is not None:
            group_ids = inputs.signals.group_ids
        if group_ids is None:
            raise RuntimeError("AlgorithmInput.group_ids is required to compute advantages")
        return algorithm.compute_advantages_from_tensors(inputs.rewards, group_ids)

    def compute_loss(
        self,
        algorithm: Algorithm,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        if inputs.advantages is None:
            inputs = replace(
                inputs,
                advantages=self.compute_advantages(algorithm, inputs),
            )
        return algorithm.compute_loss(inputs)


__all__ = [
    "AlgorithmAdapter",
    "AlgorithmInput",
]
