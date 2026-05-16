"""Build trajectory-native evaluator signals from rollout batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.engine.trajectory import TrainingView, TrajectoryBatch
from vrl.engine.trajectory.views import LossUnit
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch


@dataclass(slots=True)
class TrajectorySignalBuilder:
    """Resolve trajectory facts and package evaluator outputs into signals."""

    batch: Any
    trajectory: TrajectoryBatch | None = field(init=False)
    training_view: TrainingView | None = field(init=False)

    def __post_init__(self) -> None:
        self.trajectory = getattr(self.batch, "trajectory", None)
        self.training_view = getattr(self.batch, "training_view", None)

    def single_segment(
        self,
        *,
        segment_name: str | None,
        log_prob: Any,
        distribution: str,
        timestep_idx: int | None = None,
        old_log_prob: Any | None = None,
        mask: Any | None = None,
        ref_log_prob: Any | None = None,
        entropy: Any | None = None,
        prev_sample_mean: Any | None = None,
        ref_prev_sample_mean: Any | None = None,
        std_dev_t: Any | None = None,
        dt: Any | None = None,
        axis: str | None = None,
        aux: dict[str, Any] | None = None,
        mask_key: str = "token_mask",
    ) -> TrajectorySignalBatch:
        """Build a one-segment ``TrajectorySignalBatch`` for concrete evaluators."""

        segment = self.segment_signal(
            segment_name=segment_name,
            log_prob=log_prob,
            distribution=distribution,
            timestep_idx=timestep_idx,
            old_log_prob=old_log_prob,
            mask=mask,
            ref_log_prob=ref_log_prob,
            entropy=entropy,
            prev_sample_mean=prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            std_dev_t=std_dev_t,
            dt=dt,
            axis=axis,
            aux=aux,
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
        segment_name: str | None,
        log_prob: Any,
        distribution: str,
        timestep_idx: int | None = None,
        old_log_prob: Any | None = None,
        mask: Any | None = None,
        ref_log_prob: Any | None = None,
        entropy: Any | None = None,
        prev_sample_mean: Any | None = None,
        ref_prev_sample_mean: Any | None = None,
        std_dev_t: Any | None = None,
        dt: Any | None = None,
        axis: str | None = None,
        aux: dict[str, Any] | None = None,
        mask_key: str = "token_mask",
    ) -> SegmentSignal:
        """Build one signal segment from first-class trajectory facts."""

        name = self._primary_segment_name(segment_name)
        segment = self.trajectory.segments.get(name) if self.trajectory is not None else None
        unit = self._loss_unit(name)

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

        self._validate_signal_shapes(
            log_prob=log_prob,
            old_log_prob=resolved_old,
            mask=resolved_mask,
            segment=name,
        )
        resolved_axis = axis or (unit.axis if unit is not None else None)
        if resolved_axis is None:
            resolved_axis = self._axis_from_segment_or_signal(segment, resolved_old)

        return SegmentSignal(
            name=name,
            segment=name,
            axis=resolved_axis,
            axes=self._axes_from_value(resolved_old, axis=resolved_axis),
            distribution=segment.distribution if segment is not None else distribution,
            log_prob=log_prob,
            old_log_prob=resolved_old,
            mask=resolved_mask,
            ref_log_prob=ref_log_prob,
            entropy=entropy,
            prev_sample_mean=prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            std_dev_t=std_dev_t,
            dt=dt,
            aux=dict(aux or {}),
        )

    @property
    def group_ids(self) -> Any:
        if self.trajectory is not None:
            return self.trajectory.group_ids
        return getattr(self.batch, "group_ids", None)

    @property
    def context(self) -> dict[str, Any]:
        if self.trajectory is not None:
            return dict(self.trajectory.context)
        batch_context = getattr(self.batch, "context", None)
        return dict(batch_context) if isinstance(batch_context, dict) else {}

    def _primary_segment_name(self, fallback: str | None) -> str:
        if fallback:
            return fallback
        if self.training_view is not None and self.training_view.primary_segment is not None:
            return self.training_view.primary_segment
        if self.trajectory is not None:
            for name, segment in self.trajectory.segments.items():
                if segment.trainable:
                    return name
        return "default"

    def _loss_unit(self, segment_name: str) -> LossUnit | None:
        if self.training_view is None:
            return None
        for unit in self.training_view.loss_units:
            if unit.segment == segment_name:
                return unit
        return None

    def _old_log_prob_from_trajectory(
        self,
        segment: Any | None,
        *,
        log_prob: Any,
        timestep_idx: int | None,
    ) -> Any:
        if segment is None:
            raise RuntimeError(
                "trajectory-native evaluator signals require batch.trajectory with "
                "an old_log_prob tensor",
            )
        value = self._role_value(segment, "old_log_prob")
        return self._select_loss_value_if_needed(
            value,
            log_prob,
            timestep_idx=timestep_idx,
        )

    def _mask_from_trajectory(
        self,
        segment: Any | None,
        *,
        log_prob: Any,
        timestep_idx: int | None,
        mask_key: str,
    ) -> Any:
        if segment is None:
            return torch.ones_like(log_prob)
        tensor = segment.tensors.get(mask_key)
        value = (
            tensor.value
            if tensor is not None and tensor.role == "mask"
            else self._role_value(segment, "mask")
        )
        value = self._select_loss_value_if_needed(
            value,
            log_prob,
            timestep_idx=timestep_idx,
        )
        if self._same_shape(value, log_prob):
            return value
        return torch.ones_like(log_prob)

    @staticmethod
    def _role_value(segment: Any, role: str) -> Any:
        matches = [tensor for tensor in segment.tensors.values() if tensor.role == role]
        if len(matches) != 1:
            raise RuntimeError(
                f"segment {segment.name!r} requires exactly one role {role!r}, "
                f"found {len(matches)}",
            )
        return matches[0].value

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
        if (
            len(value_shape) == len(log_prob_shape) + 1
            and int(value_shape[0]) == int(log_prob_shape[0])
        ):
            selected = value[:, timestep_idx]
            if cls._same_shape(selected, log_prob):
                return selected
        if (
            len(value_shape) == 2
            and len(log_prob_shape) == 1
            and int(value_shape[0]) == int(log_prob_shape[0])
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

    @staticmethod
    def _axis_from_segment_or_signal(segment: Any | None, value: Any) -> str:
        if segment is not None:
            for tensor in segment.tensors.values():
                if tensor.role == "old_log_prob":
                    for axis in tensor.axes:
                        if axis != "sample":
                            return axis
        return "step" if getattr(value, "ndim", 1) <= 1 else "token"

    @staticmethod
    def _axes_from_value(value: Any, *, axis: str) -> tuple[str, ...]:
        ndim = int(getattr(value, "ndim", 1))
        if ndim <= 1:
            return ("sample",)
        return ("sample", axis)


__all__ = ["TrajectorySignalBuilder"]
