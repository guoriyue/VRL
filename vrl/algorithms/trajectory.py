"""Algorithm adapters for trajectory-native training inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import TrajectorySignalBatch


@dataclass(slots=True)
class AlgorithmInput:
    """Unified algorithm-facing input derived from trajectory contracts."""

    signals: TrajectorySignalBatch | None = None
    rewards: Any | None = None
    group_ids: Any | None = None
    advantages: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signals is not None and not isinstance(self.signals, TrajectorySignalBatch):
            raise TypeError(
                "AlgorithmInput.signals must be a TrajectorySignalBatch",
            )


class AlgorithmAdapter:
    """Dispatch strict AlgorithmInput to objective-specific native APIs."""

    def compute_advantages(self, algorithm: Any, inputs: AlgorithmInput) -> Any:
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
        algorithm: Any,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        compute_loss = getattr(algorithm, "compute_loss", None)
        if not callable(compute_loss):
            raise TypeError(
                f"{type(algorithm).__name__} must expose compute_loss(AlgorithmInput)",
            )

        if inputs.advantages is None:
            inputs = AlgorithmInput(
                signals=inputs.signals,
                rewards=inputs.rewards,
                group_ids=inputs.group_ids,
                advantages=self.compute_advantages(algorithm, inputs),
                metadata=inputs.metadata,
            )
        return compute_loss(inputs)


__all__ = [
    "AlgorithmAdapter",
    "AlgorithmInput",
]
