"""Multi-segment token log-probability evaluator for Janus-Pro-R1 rollouts."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import ReplayEvaluatorBase
from vrl.rollouts.evaluators.token.ref_pass import reference_model_context
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.evaluators.types import SegmentSignal, SignalRequest, TrajectorySignalBatch
from vrl.trajectory.types import TrajectorySegment


class MultiSegmentTokenLogProbEvaluator(ReplayEvaluatorBase):
    """Replay each enabled R1 segment without concatenating image/text tokens."""

    replay_granularity = "trajectory"

    def __init__(self, *, enabled_segments: Iterable[str]) -> None:
        # Freeze the configured selection once; iterators must survive repeated replay.
        self.enabled_segments = tuple(enabled_segments)

    def evaluate(
        self,
        model: ReplayModel,
        batch: RolloutBatch,
        timestep_idx: int = 0,
        ref_model: ReplayModel | None = None,
        signal_request: SignalRequest | None = None,
    ) -> TrajectorySignalBatch:
        del timestep_idx
        model, ref_model = self._require_models(model, ref_model)
        request = signal_request or SignalRequest()
        segments = self._segments_from_batch(batch)
        if not isinstance(segments, dict):
            raise RuntimeError(
                "MultiSegmentTokenLogProbEvaluator requires trajectory segments",
            )

        enabled_names = [name for name in self.enabled_segments if name in segments]
        if not enabled_names:
            raise RuntimeError("no enabled R1 segments to evaluate")

        replay_request = ReplayRequest(segment_names=tuple(enabled_names))
        current_output = model.replay_forward(batch, request=replay_request)
        ref_output = None
        if request.need_ref:
            with reference_model_context(model, ref_model) as reference:
                ref_output = reference.replay_forward(batch, request=replay_request)

        signal_builder = TrajectorySignalBuilder(batch)
        # R1 rollout scores every segment (text and image) with the same
        # sampling temperature; replay renormalizes with the recorded value
        # to keep old/new log-prob parity.
        temperature = float(signal_builder.context.get("temperature", 1.0))
        segment_signals: dict[str, SegmentSignal] = {}

        for name in enabled_names:
            segment = segments[name]
            new_lp = self._compute_segment_logprobs(
                current_output,
                name,
                segment,
                temperature,
            )
            ref_lp = None
            if ref_output is not None:
                with torch.no_grad():
                    ref_lp = self._compute_segment_logprobs(
                        ref_output,
                        name,
                        segment,
                        temperature,
                    )

            segment_signals[name] = signal_builder.segment_signal(
                segment_name=name,
                log_prob=new_lp,
                old_log_prob=self._segment_tensor(segment, "old_log_prob").detach(),
                mask=self._segment_tensor(segment, "mask").to(
                    dtype=new_lp.dtype,
                    device=new_lp.device,
                ),
                ref_log_prob=ref_lp,
                mask_key="token_mask",
            )

        primary_name = self._primary_segment_name(batch, enabled_names)
        return TrajectorySignalBatch(
            segments=segment_signals,
            group_ids=signal_builder.group_ids,
            context=signal_builder.context,
            primary_segment=primary_name,
        )

    def _compute_segment_logprobs(
        self,
        output: ReplayResult,
        name: str,
        segment: TrajectorySegment,
        temperature: float,
    ) -> torch.Tensor:
        result = output.require_segment(name)
        token_ids = result.values.get("token_ids")
        if token_ids is None:
            token_ids = self._segment_tensor(segment, "action")
        return result.logprobs(token_ids, temperature=temperature)

    @staticmethod
    def _segments_from_batch(batch: RolloutBatch) -> dict[str, TrajectorySegment] | None:
        trajectory = batch.trajectory
        if trajectory is None:
            return None
        segments = {
            name: segment
            for name, segment in trajectory.segments.items()
            if segment.distribution == "categorical"
        }
        return segments or None

    @staticmethod
    def _segment_tensor(segment: TrajectorySegment, role: str) -> torch.Tensor:
        value = segment.role_tensor(role).value
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"R1 segment {segment.name!r} role {role!r} must be a tensor")
        return value

    @staticmethod
    def _primary_segment_name(batch: RolloutBatch, enabled_names: list[str]) -> str:
        trajectory = getattr(batch, "trajectory", None)
        if trajectory is not None:
            primary = trajectory.primary_segment
            if isinstance(primary, str) and primary in enabled_names:
                return primary
        return enabled_names[0]


__all__ = ["MultiSegmentTokenLogProbEvaluator"]
