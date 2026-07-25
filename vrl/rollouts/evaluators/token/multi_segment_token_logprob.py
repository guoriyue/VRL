"""Multi-segment token log-probability evaluator for Janus-Pro-R1 rollouts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator, ReplayEvaluatorBase
from vrl.rollouts.evaluators.token.ref_pass import ref_forward
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.evaluators.types import SegmentSignal, SignalRequest, TrajectorySignalBatch
from vrl.trajectory import role_tensor


class MultiSegmentTokenLogProbEvaluator(ReplayEvaluatorBase, Evaluator):
    """Replay each enabled R1 segment without concatenating image/text tokens."""

    replay_granularity = "trajectory"

    def __init__(self, *, enabled_segments: Iterable[str]) -> None:
        self.enabled_segments = enabled_segments

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
            ref_output = ref_forward(
                model,
                ref_model,
                lambda m: m.replay_forward(batch, request=replay_request),
            )

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
                ref_lp = self._compute_segment_logprobs(
                    ref_output,
                    name,
                    segment,
                    temperature,
                )

            segment_signals[name] = signal_builder.segment_signal(
                segment_name=name,
                log_prob=new_lp,
                old_log_prob=self._segment_tensor(segment, "token_log_probs").detach(),
                mask=self._segment_tensor(segment, "token_mask").to(
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
        segment: dict[str, Any],
        temperature: float,
    ) -> torch.Tensor:
        result = output.require_segment(name)
        return self._extract_logprobs(result, segment, temperature)

    @staticmethod
    def _segments_from_batch(batch: RolloutBatch) -> dict[str, Any] | None:
        trajectory = getattr(batch, "trajectory", None)
        if trajectory is not None:
            segments: dict[str, Any] = {}
            for name, segment in trajectory.segments.items():
                if segment.distribution != "categorical":
                    continue
                segments[name] = MultiSegmentTokenLogProbEvaluator._trajectory_segment_payload(
                    segment,
                )
            if segments:
                return segments
        return None

    @staticmethod
    def _trajectory_segment_payload(segment: Any) -> dict[str, Any]:
        return {
            "name": segment.name,
            "token_ids": role_tensor(segment, "action").value,
            "token_log_probs": role_tensor(segment, "old_log_prob").value,
            "token_mask": role_tensor(segment, "mask").value,
        }

    @staticmethod
    def _extract_logprobs(
        result: ReplaySegmentResult,
        segment: dict[str, Any],
        temperature: float,
    ) -> torch.Tensor:
        # The payload-key knowledge (log_probs vs modality-named logits) lives on
        # ReplaySegmentResult itself; this evaluator only supplies the trajectory
        # token-id fallback.
        token_ids = result.values.get("token_ids")
        if token_ids is None:
            token_ids = MultiSegmentTokenLogProbEvaluator._segment_tensor(segment, "token_ids")
        return result.logprobs(token_ids, temperature=temperature)

    @staticmethod
    def _segment_tensor(segment: dict[str, Any], key: str) -> torch.Tensor:
        value = segment.get(key)
        if value is None:
            name = segment.get("name", "<unknown>")
            raise RuntimeError(f"R1 segment {name!r} is missing tensor field {key!r}")
        if not isinstance(value, torch.Tensor):
            name = segment.get("name", "<unknown>")
            raise RuntimeError(f"R1 segment {name!r} field {key!r} must be a tensor")
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
