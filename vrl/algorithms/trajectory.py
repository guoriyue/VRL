"""Algorithm adapters for trajectory-native training inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from vrl.algorithms.types import TrainStepMetrics

if TYPE_CHECKING:
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

    def __post_init__(self) -> None:
        # Lazy: keep vrl.algorithms torch-free at import time so config parsing
        # (algorithm.kind dispatch) never pulls the rollout evaluator stack.
        from vrl.rollouts.evaluators.types import TrajectorySignalBatch

        if self.signals is not None and not isinstance(self.signals, TrajectorySignalBatch):
            raise TypeError(
                "AlgorithmInput.signals must be a TrajectorySignalBatch",
            )


class AlgorithmAdapter:
    """Dispatch strict AlgorithmInput to objective-specific native APIs."""

    def validate_inputs(self, algorithm: Any, inputs: AlgorithmInput) -> None:
        """Fail fast — with available-vs-missing diagnostics — when the rollout
        payload lacks a tensor the algorithm's loss declares it needs.

        One declarative gate replacing the per-loss inline checks each objective
        used to hand-roll (NFT's per-key ``isinstance`` loop, GRPO's
        ``signals is None`` guard). An algorithm declares ``required_signal_keys``
        (``SegmentSignal`` fields, signal branch) and/or ``required_data_keys``
        (replay-tensor names, replay branch); this validates them against what
        actually arrived. Mirrors verl-omni's ``DiffusionLossFn.validate_inputs``.
        """
        required_signal_keys = tuple(getattr(algorithm, "required_signal_keys", ()))
        required_data_keys = tuple(getattr(algorithm, "required_data_keys", ()))
        algo = type(algorithm).__name__

        if required_signal_keys:
            if inputs.signals is None:
                raise RuntimeError(
                    f"{algo} requires evaluator signals {list(required_signal_keys)} "
                    "but AlgorithmInput.signals is None.",
                )
            signal = inputs.signals.primary
            available = sorted(
                f.name for f in fields(signal) if getattr(signal, f.name) is not None
            )
            missing = [key for key in required_signal_keys if key not in available]
            if missing:
                raise KeyError(
                    f"Algorithm `{algo}` is missing required signal fields. "
                    f"Missing signal keys: {missing}. "
                    f"Available signal keys: [{', '.join(available)}].",
                )

        if required_data_keys:
            batch = inputs.rollout_batch
            if batch is None:
                raise RuntimeError(
                    f"{algo} requires replay tensors {list(required_data_keys)} but "
                    "AlgorithmInput.rollout_batch is missing.",
                )
            import torch

            from vrl.trajectory import TrajectoryResolver

            replay_tensors = TrajectoryResolver.from_batch(batch).replay_tensor_dict()
            available = sorted(
                key for key, value in replay_tensors.items() if isinstance(value, torch.Tensor)
            )
            missing = [key for key in required_data_keys if key not in available]
            if missing:
                raise KeyError(
                    f"Algorithm `{algo}` is missing required replay tensors. "
                    f"Missing data keys: {missing}. "
                    f"Available data keys: [{', '.join(available)}].",
                )

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

        self.validate_inputs(algorithm, inputs)

        if inputs.advantages is None:
            inputs = AlgorithmInput(
                signals=inputs.signals,
                rewards=inputs.rewards,
                group_ids=inputs.group_ids,
                advantages=self.compute_advantages(algorithm, inputs),
                model=inputs.model,
                rollout_batch=inputs.rollout_batch,
                timestep_index=inputs.timestep_index,
            )
        return compute_loss(inputs)


__all__ = [
    "AlgorithmAdapter",
    "AlgorithmInput",
]
