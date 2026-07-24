"""Build trajectory-native evaluator signals from rollout batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch
from vrl.trajectory import TrajectoryBatch, role_tensor
from vrl.trajectory.device import move_value_to_device


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
        mask_key: str = "token_mask",
    ) -> SegmentSignal:
        """Build one signal segment from first-class trajectory facts."""

        segment = self.trajectory.segments.get(segment_name)
        if segment is None:
            raise RuntimeError(f"unknown trajectory segment {segment_name!r}")

        resolved_old = old_log_prob
        if resolved_old is None:
            resolved_old = self._old_log_prob_from_trajectory(
                segment,
                log_prob=log_prob,
                timestep_idx=timestep_idx,
            )
        resolved_mask = mask
        if resolved_mask is None:
            resolved_mask = self._mask_from_trajectory(
                segment,
                log_prob=log_prob,
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

        self._validate_signal_shapes(
            log_prob=log_prob,
            old_log_prob=resolved_old,
            mask=resolved_mask,
            segment=segment_name,
        )
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
        )

    @property
    def group_ids(self) -> Any:
        return self.batch.group_ids

    @property
    def context(self) -> dict[str, Any]:
        return dict(self.trajectory.context)

    def _old_log_prob_from_trajectory(
        self,
        segment: Any,
        *,
        log_prob: Any,
        timestep_idx: int | None,
    ) -> Any:
        value = role_tensor(segment, "old_log_prob").value
        return self._select_loss_value_if_needed(
            value,
            log_prob,
            timestep_idx=timestep_idx,
        )

    def _mask_from_trajectory(
        self,
        segment: Any,
        *,
        log_prob: Any,
        timestep_idx: int | None,
        mask_key: str,
    ) -> Any:
        tensor = segment.tensors.get(mask_key)
        value = (
            tensor.value
            if tensor is not None and tensor.role == "mask"
            else role_tensor(segment, "mask").value
        )
        value = self._select_loss_value_if_needed(
            value,
            log_prob,
            timestep_idx=timestep_idx,
        )
        return value

    @classmethod
    def _select_loss_value_if_needed(
        cls,
        value: Any,
        log_prob: Any,
        *,
        timestep_idx: int | None,
    ) -> Any:
        if cls._same_shape(value, log_prob):
            return value
        if timestep_idx is None:
            return value
        value_shape = getattr(value, "shape", None)
        log_prob_shape = getattr(log_prob, "shape", None)
        if value_shape is None or log_prob_shape is None:
            return value
        if len(value_shape) == len(log_prob_shape) + 1 and int(value_shape[0]) == int(
            log_prob_shape[0]
        ):
            selected = value[:, timestep_idx]
            if cls._same_shape(selected, log_prob):
                return selected
        return value

    @classmethod
    def _validate_signal_shapes(
        cls,
        *,
        log_prob: Any,
        old_log_prob: Any,
        mask: Any,
        segment: str,
    ) -> None:
        if not cls._same_shape(log_prob, old_log_prob):
            raise RuntimeError(
                "trajectory signal shape mismatch for segment "
                f"{segment!r}: log_prob shape={cls._shape(log_prob)}, "
                f"old_log_prob shape={cls._shape(old_log_prob)}. Evaluators must pass "
                "step-level old_log_prob for per-step signals.",
            )
        if mask is not None and not cls._same_shape(log_prob, mask):
            raise RuntimeError(
                "trajectory signal mask shape mismatch for segment "
                f"{segment!r}: log_prob shape={cls._shape(log_prob)}, "
                f"mask shape={cls._shape(mask)}",
            )

    @staticmethod
    def _same_shape(left: Any, right: Any) -> bool:
        left_shape = getattr(left, "shape", None)
        right_shape = getattr(right, "shape", None)
        if left_shape is None or right_shape is None:
            return True
        return tuple(left_shape) == tuple(right_shape)

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...] | str:
        shape = getattr(value, "shape", None)
        if shape is None:
            return "<unknown>"
        return tuple(int(dim) for dim in shape)


__all__ = ["TrajectorySignalBuilder"]
