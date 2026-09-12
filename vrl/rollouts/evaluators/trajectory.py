"""Build trajectory-native evaluator signals from rollout batches.

Shared by every concrete evaluator (denoise and token): resolving recorded
``old_log_prob``/mask facts from the trajectory, slicing per-step values when
the replay is step-granular, and moving them to the replay device happen here
instead of per evaluator. Evaluators compute the fresh forward-pass values;
TrajectorySignalBatch validates signal shapes when the batch is assembled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch
from vrl.trajectory.device import move_value_to_device
from vrl.trajectory.types import TrajectoryBatch, TrajectorySegment, TrajectoryTensor


@dataclass(slots=True)
class TrajectorySignalBuilder:
    """Resolve trajectory facts and package evaluator outputs into signals."""

    batch: RolloutBatch
    trajectory: TrajectoryBatch = field(init=False)

    def __post_init__(self) -> None:
        trajectory = self.batch.trajectory
        if not isinstance(trajectory, TrajectoryBatch):
            raise RuntimeError(
                "trajectory-native evaluator signals require batch.trajectory "
                "to be a TrajectoryBatch",
            )
        self.trajectory = trajectory

    def single_segment(
        self,
        *,
        segment_name: str,
        log_prob: Any,
        timestep_idx: int | None = None,
        old_log_prob: Any | None = None,
        mask: Any | None = None,
        ref_log_prob: Any | None = None,
        prev_sample_mean: Any | None = None,
        ref_prev_sample_mean: Any | None = None,
        old_prev_sample_mean: Any | None = None,
        std_dev_t: Any | None = None,
        dt: Any | None = None,
        sigma: Any | None = None,
        mask_key: str = "token_mask",
    ) -> TrajectorySignalBatch:
        """Build a one-segment ``TrajectorySignalBatch`` for concrete evaluators."""

        segment = self.segment_signal(
            segment_name=segment_name,
            log_prob=log_prob,
            timestep_idx=timestep_idx,
            old_log_prob=old_log_prob,
            mask=mask,
            ref_log_prob=ref_log_prob,
            prev_sample_mean=prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            old_prev_sample_mean=old_prev_sample_mean,
            std_dev_t=std_dev_t,
            dt=dt,
            sigma=sigma,
            mask_key=mask_key,
        )
        return TrajectorySignalBatch(
            segments={segment.name: segment},
            group_ids=self.group_ids,
            context=self.context,
            primary_segment=segment.name,
        )

    def segment_signal(
        self,
        *,
        segment_name: str,
        log_prob: Any,
        timestep_idx: int | None = None,
        old_log_prob: Any | None = None,
        mask: Any | None = None,
        ref_log_prob: Any | None = None,
        prev_sample_mean: Any | None = None,
        ref_prev_sample_mean: Any | None = None,
        old_prev_sample_mean: Any | None = None,
        std_dev_t: Any | None = None,
        dt: Any | None = None,
        sigma: Any | None = None,
        mask_key: str = "token_mask",
    ) -> SegmentSignal:
        """Build one signal segment from first-class trajectory facts."""

        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            raise RuntimeError(f"unknown trajectory segment {segment_name!r}")

        resolved_old = old_log_prob
        if resolved_old is None:
            resolved_old = self._select_denoise_step(
                segment.role_tensor("old_log_prob"),
                timestep_idx=timestep_idx,
            )
        resolved_mask = mask
        if resolved_mask is None:
            resolved_mask = self._mask_from_trajectory(
                segment,
                timestep_idx=timestep_idx,
                mask_key=mask_key,
            )
        resolved_old = move_value_to_device(
            resolved_old,
            getattr(log_prob, "device", None),
        )
        resolved_mask = move_value_to_device(
            resolved_mask,
            getattr(log_prob, "device", None),
        )

        # Shape theorems live on TrajectorySignalBatch.__post_init__ — every
        # production consumer of this signal constructs one, so validating here
        # too was a drifting duplicate of the same checks.
        return SegmentSignal(
            name=segment_name,
            distribution=segment.distribution,
            log_prob=log_prob,
            old_log_prob=resolved_old,
            mask=resolved_mask,
            ref_log_prob=ref_log_prob,
            prev_sample_mean=prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            old_prev_sample_mean=old_prev_sample_mean,
            std_dev_t=std_dev_t,
            dt=dt,
            sigma=sigma,
        )

    @property
    def group_ids(self) -> Any:
        return self.batch.group_ids

    @property
    def context(self) -> dict[str, Any]:
        return dict(self.trajectory.context)

    def _mask_from_trajectory(
        self,
        segment: TrajectorySegment,
        *,
        timestep_idx: int | None,
        mask_key: str,
    ) -> Any:
        tensor = segment.tensors.get(mask_key)
        if tensor is None or tensor.role != "mask":
            tensor = segment.role_tensor("mask")
        return self._select_denoise_step(tensor, timestep_idx=timestep_idx)

    def _select_denoise_step(
        self,
        tensor: TrajectoryTensor,
        *,
        timestep_idx: int | None,
    ) -> Any:
        """Select the declared denoise axis before moving recorded values."""

        if timestep_idx is None:
            return tensor.value
        step_dims = [
            dim
            for dim, name in enumerate(tensor.axes)
            if self.trajectory.axes[name].kind == "denoise_step"
        ]
        if not step_dims:
            return tensor.value
        if len(step_dims) != 1:
            raise ValueError(f"tensor {tensor.name!r} must have exactly one denoise_step axis")
        key = [slice(None)] * len(tensor.axes)
        key[step_dims[0]] = timestep_idx
        return tensor.value[tuple(key)]


__all__ = ["TrajectorySignalBuilder"]
