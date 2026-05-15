"""Multi-segment token log-probability evaluator for Janus-Pro-R1 rollouts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    require_replay_model,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.replay_result import require_replay_segment, require_replay_value
from vrl.rollouts.evaluators.trajectory import segment_signal_from_batch
from vrl.rollouts.evaluators.types import SegmentSignal, SignalRequest, TrajectorySignalBatch


class MultiSegmentTokenLogProbEvaluator(Evaluator):
    """Replay each enabled R1 segment without concatenating image/text tokens."""

    def __init__(
        self,
        *,
        enabled_segments: Iterable[str] | Mapping[str, bool] | None = None,
        mask_key: str = "token_mask",
    ) -> None:
        self.enabled_segments = enabled_segments
        self.mask_key = mask_key

    def evaluate(
        self,
        model: ReplayModel,
        batch: RolloutBatch,
        timestep_idx: int = 0,
        ref_model: ReplayModel | None = None,
        signal_request: SignalRequest | None = None,
    ) -> TrajectorySignalBatch:
        del timestep_idx
        model = require_replay_model(model, owner="MultiSegmentTokenLogProbEvaluator.model")
        if ref_model is not None:
            ref_model = require_replay_model(
                ref_model,
                owner="MultiSegmentTokenLogProbEvaluator.ref_model",
            )
        request = signal_request or SignalRequest()
        segments = _segments_from_batch(batch)
        if not isinstance(segments, dict):
            raise RuntimeError(
                "MultiSegmentTokenLogProbEvaluator requires trajectory segments",
            )

        enabled_names = self._enabled_segment_names(segments)
        if not enabled_names:
            raise RuntimeError("no enabled R1 segments to evaluate")

        replay_request = ReplayRequest(segment_names=tuple(enabled_names))
        current_output = model.replay_forward(batch, request=replay_request)
        ref_output = None
        if request.need_ref:
            if ref_model is not None:
                ref_output = ref_model.replay_forward(batch, request=replay_request)
            else:
                with torch.no_grad(), model.disable_adapter():
                    ref_output = model.replay_forward(batch, request=replay_request)

        segment_signals: dict[str, SegmentSignal] = {}

        for name in enabled_names:
            segment = segments[name]
            new_lp = self._compute_segment_logprobs(current_output, name, segment)
            ref_lp = None
            if ref_output is not None:
                ref_lp = self._compute_segment_logprobs(ref_output, name, segment)

            mask = _segment_tensor(segment, self.mask_key, required=False)
            if mask is None:
                mask = torch.ones_like(new_lp)
            old_lp = _segment_tensor(segment, "token_log_probs")

            segment_signals[name] = segment_signal_from_batch(
                batch,
                segment_name=name,
                log_prob=new_lp,
                old_log_prob=old_lp.detach(),
                mask=mask.to(dtype=new_lp.dtype, device=new_lp.device),
                ref_log_prob=ref_lp,
                entropy=None,
                distribution="categorical",
                aux={
                    "segment_name": name,
                    "segment_modality": _segment_modality(name, segment),
                },
                mask_key=self.mask_key,
            )

        primary_name = _primary_segment_name(batch, enabled_names)
        return TrajectorySignalBatch(
            segments=segment_signals,
            group_ids=_group_ids_from_batch(batch),
            context={
                **_context_from_batch(batch),
                "segment_order": tuple(enabled_names),
                "primary_segment": primary_name,
            },
            primary_segment=primary_name,
        )

    def _enabled_segment_names(self, segments: dict[str, Any]) -> list[str]:
        if self.enabled_segments is None:
            return [
                name
                for name, segment in segments.items()
                if _segment_enabled(segment)
            ]
        if isinstance(self.enabled_segments, Mapping):
            return [
                name
                for name, enabled in self.enabled_segments.items()
                if enabled and name in segments and _segment_enabled(segments[name])
            ]
        return [
            name
            for name in self.enabled_segments
            if name in segments and _segment_enabled(segments[name])
        ]

    def _compute_segment_logprobs(
        self,
        output: ReplayResult,
        name: str,
        segment: dict[str, Any],
    ) -> torch.Tensor:
        result = require_replay_segment(output, name)
        return _extract_logprobs(result, segment)


def _segments_from_batch(batch: RolloutBatch) -> dict[str, Any] | None:
    trajectory = getattr(batch, "trajectory", None)
    if trajectory is not None:
        segments: dict[str, Any] = {}
        for name, segment in trajectory.segments.items():
            if segment.distribution != "categorical":
                continue
            segments[name] = _trajectory_segment_payload(segment)
        if segments:
            return segments
    return None


def _trajectory_segment_payload(segment: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": segment.name,
        "token_ids": _trajectory_role_value(segment, "action"),
        "token_log_probs": _trajectory_role_value(segment, "old_log_prob"),
        "token_mask": _trajectory_role_value(segment, "mask"),
        "visual": bool(segment.metadata.get("visual", segment.modality == "image")),
        "cfg": bool(segment.metadata.get("cfg", False)),
        "train": bool(segment.metadata.get("train", segment.trainable)),
        "modality": segment.modality,
    }
    for key in ("prompt_embeds", "attention_mask", "prompt_attention_mask", "prompt_input_ids"):
        tensor = segment.tensors.get(key)
        if tensor is not None:
            payload[key] = tensor.value
    return payload


def _trajectory_role_value(segment: Any, role: str) -> Any:
    matches = [tensor.value for tensor in segment.tensors.values() if tensor.role == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"trajectory segment {segment.name!r} requires exactly one {role!r} tensor",
        )
    return matches[0]


def _extract_logprobs(result: ReplaySegmentResult, segment: dict[str, Any]) -> torch.Tensor:
    values = result.values
    if "log_probs" in values:
        return require_replay_value(result, "log_probs").float()

    logits = values.get("logits")
    if logits is None:
        logits = values.get("image_logits")
    if logits is None:
        logits = values.get("text_logits")
    if logits is None:
        logits = require_replay_value(result, "logits")

    token_ids = values.get("token_ids")
    if token_ids is None:
        token_ids = _segment_tensor(segment, "token_ids")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def _segment_tensor(
    segment: dict[str, Any],
    key: str,
    *,
    required: bool = True,
) -> torch.Tensor | None:
    value = segment.get(key)
    if value is None:
        replay = segment.get("replay")
        if isinstance(replay, dict):
            value = replay.get(key)
    if value is None:
        if required:
            name = segment.get("name", "<unknown>")
            raise RuntimeError(f"R1 segment {name!r} is missing tensor field {key!r}")
        return None
    if not isinstance(value, torch.Tensor):
        name = segment.get("name", "<unknown>")
        raise RuntimeError(f"R1 segment {name!r} field {key!r} must be a tensor")
    return value


def _segment_modality(name: str, segment: dict[str, Any]) -> str:
    modality = segment.get("modality")
    if modality is not None:
        return str(modality)
    visual = segment.get("visual")
    if visual is not None:
        return "image" if bool(visual) else "text"
    if name.endswith("_text"):
        return "text"
    return "image"


def _segment_enabled(segment: dict[str, Any]) -> bool:
    return bool(segment.get("enabled", segment.get("train", True)))


def _primary_segment_name(batch: RolloutBatch, enabled_names: list[str]) -> str:
    training_view = getattr(batch, "training_view", None)
    primary = getattr(training_view, "primary_segment", None)
    if isinstance(primary, str) and primary in enabled_names:
        return primary
    trajectory = getattr(batch, "trajectory", None)
    if trajectory is not None:
        primary = trajectory.context.get("primary_segment")
        if isinstance(primary, str) and primary in enabled_names:
            return primary
    return enabled_names[0]


def _group_ids_from_batch(batch: RolloutBatch) -> Any:
    trajectory = getattr(batch, "trajectory", None)
    if trajectory is not None:
        return trajectory.group_ids
    return batch.group_ids


def _context_from_batch(batch: RolloutBatch) -> dict[str, Any]:
    trajectory = getattr(batch, "trajectory", None)
    if trajectory is not None:
        return dict(trajectory.context)
    return dict(batch.context)


__all__ = ["MultiSegmentTokenLogProbEvaluator"]
