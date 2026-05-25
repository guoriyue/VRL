"""Pose-structure scoring model built on whole-body keypoints.

Consumes DWPose whole-body pose results and scores anatomy structure from
keypoint geometry. Implements ``score_request`` so pose prediction is invoked
once across every artifact's images, then per-rollout scores are aggregated by
mean.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from vrl.rewards.models.pose.dwpose import (
    DWPoseModel,
    _onnx_providers,
)
from vrl.rewards.models.pose.geometry import (
    _REQUIRED_BODY_POINTS,
    _body_scale,
    _collapsed_hand_fraction,
    _coverage,
    _feet_missing_fraction,
    _joint_geometry_penalty,
    _limb_asymmetry_penalty,
    _people_from_result,
    _person_confidence,
    _present_count,
    _visible_hand_count,
)
from vrl.rewards.models.pose.hints import (
    _BOTH_HAND_HINTS,
    _FEET_HINTS,
    _HAND_HINTS,
    _constraint_texts,
    _contains_hint,
)

_LOGGER = logging.getLogger(__name__)

# Fixed geometry penalty weights — not tunable per-instance.
_MISSING_KEYPOINT_PENALTY = 0.35
_IMPOSSIBLE_ANGLE_PENALTY = 0.25
_ASYMMETRIC_LIMB_PENALTY = 0.15
_HAND_MISSING_PENALTY = 0.30
_COLLAPSED_HAND_PENALTY = 0.15
_MULTI_PERSON_PENALTY = 0.10
_MIN_JOINT_ANGLE_DEGREES = 18.0
_MAX_SEGMENT_RATIO = 4.0
_MAX_LIMB_ASYMMETRY_RATIO = 2.5
_MIN_HAND_SPREAD_RATIO = 0.035
_MIN_HAND_KEYPOINTS = 6


class PoseStructureRewardModel:
    """Batch RewardModel returning one configured structure score per artifact."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._score_name = str(cfg.get("score_name", "pose_structure"))
        pose_model = cfg.get("pose_model")
        detector = cfg.get("detector")
        if pose_model is not None and detector is not None:
            raise ValueError("provide either pose_model or detector, not both")
        self._pose_model: Any | None = pose_model
        if detector is not None:
            self._pose_model = _BatchPoseCallable(detector)
        detect_resolution = int(cfg.get("detect_resolution", 512))
        if detect_resolution <= 0:
            raise ValueError("detect_resolution must be > 0")
        self._pose_model_config = {
            "model_repo": str(cfg.get("model_repo", "yzd-v/DWPose")),
            "detector_file": str(cfg.get("detector_file", "yolox_l.onnx")),
            "pose_file": str(cfg.get("pose_file", "dw-ll_ucoco_384.onnx")),
            "cache_dir": str(cfg.get("cache_dir", "") or "") or None,
            "local_files_only": bool(cfg.get("local_files_only", False)),
            "device": str(cfg.get("device", "cuda")),
            "detect_resolution": detect_resolution,
        }
        min_keypoint_confidence = float(cfg.get("min_keypoint_confidence", 0.25))
        if not (0.0 <= min_keypoint_confidence <= 1.0):
            raise ValueError("min_keypoint_confidence must be in [0, 1]")
        self._min_keypoint_confidence = min_keypoint_confidence
        min_body_keypoints = int(cfg.get("min_body_keypoints", 8))
        if min_body_keypoints <= 0:
            raise ValueError("min_body_keypoints must be > 0")
        self._min_body_keypoints = min_body_keypoints
        require_hands = cfg.get("require_hands", "prompt")
        require_feet = cfg.get("require_feet", "prompt")
        for name, val in (("require_hands", require_hands), ("require_feet", require_feet)):
            normalized = val
            if isinstance(normalized, bool):
                normalized = "always" if normalized else "never"
            if str(normalized).strip().lower() not in {"always", "never", "prompt"}:
                raise ValueError(f"{name} must be one of: always, never, prompt")
        self._require_hands = (
            "always"
            if require_hands is True
            else "never"
            if require_hands is False
            else str(require_hands).strip().lower()
        )
        self._require_feet = (
            "always"
            if require_feet is True
            else "never"
            if require_feet is False
            else str(require_feet).strip().lower()
        )
        debug_dir = cfg.get("debug_dir")
        self._debug_dir = Path(debug_dir) if debug_dir else None
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        self.last_diagnostics: list[dict[str, Any]] = []

    def score_request(self, request: Any) -> list[dict[str, float]]:
        image_groups = [_extract_images(artifact.as_media()) for artifact in request.artifacts]
        flat_images: list[Any] = []
        flat_requirements: list[tuple[int, bool]] = []
        for artifact, group in zip(request.artifacts, image_groups, strict=True):
            requirements = self._requirements_from_artifact(artifact)
            for image in group:
                flat_images.append(image)
                flat_requirements.append(requirements)

        flat_results = self._detect(flat_images)

        flat_scores: list[float] = []
        diagnostics: list[dict[str, Any]] = []
        for result, requirements in zip(flat_results, flat_requirements, strict=True):
            score, diagnostic = self._score_pose_result(result, requirements)
            flat_scores.append(score)
            diagnostics.append(diagnostic)
        self.last_diagnostics = diagnostics
        if self._debug_dir is not None:
            self._write_debug_diagnostics(diagnostics)

        out: list[dict[str, float]] = []
        cursor = 0
        for group in image_groups:
            count = len(group)
            scores = flat_scores[cursor : cursor + count]
            cursor += count
            value = sum(scores) / len(scores) if scores else 0.0
            out.append({self._score_name: value})
        return out

    def _detect(self, flat_images: list[Any]) -> list[Any]:
        if not flat_images:
            return []
        if self._pose_model is None:
            self._pose_model = DWPoseModel(**self._pose_model_config)
        return _predict_pose_batch(self._pose_model, flat_images)

    def _score_pose_result(
        self,
        result: Any,
        requirements: tuple[int, bool],
    ) -> tuple[float, dict[str, Any]]:
        hand_count, require_feet = requirements
        diagnostic: dict[str, Any] = {
            "required_hands": hand_count,
            "required_feet": require_feet,
        }
        people = _people_from_result(result, min_score=self._min_keypoint_confidence)
        diagnostic["num_people"] = len(people)
        if isinstance(result, Mapping):
            provider = result.get("provider")
            if provider:
                diagnostic["provider"] = provider
            detector_boxes = result.get("detector_boxes")
            if detector_boxes is not None:
                with suppress(TypeError):
                    diagnostic["detector_box_count"] = len(detector_boxes)
        if not people:
            diagnostic["score"] = 0.0
            return 0.0, diagnostic

        person = max(people, key=_person_confidence)
        score = 1.0
        body_coverage = _coverage(person.body, _REQUIRED_BODY_POINTS)
        if _present_count(person.body) < self._min_body_keypoints:
            body_coverage = min(
                body_coverage, _present_count(person.body) / self._min_body_keypoints
            )
        missing_body_penalty = _MISSING_KEYPOINT_PENALTY * (1.0 - body_coverage)
        score -= missing_body_penalty
        diagnostic["body_coverage"] = body_coverage
        diagnostic["body_present_count"] = _present_count(person.body)
        diagnostic["missing_body_penalty"] = missing_body_penalty

        if require_feet:
            feet_missing_fraction = _feet_missing_fraction(person)
            feet_penalty = _MISSING_KEYPOINT_PENALTY * feet_missing_fraction
            score -= feet_penalty
            diagnostic["feet_missing_fraction"] = feet_missing_fraction
            diagnostic["feet_penalty"] = feet_penalty

        if hand_count > 0:
            min_hand_spread = _MIN_HAND_SPREAD_RATIO * max(_body_scale(person), 1e-6)
            visible_hands = _visible_hand_count(
                person,
                min_points=_MIN_HAND_KEYPOINTS,
                min_spread=min_hand_spread,
            )
            missing_hand_penalty = (
                _HAND_MISSING_PENALTY * max(0.0, hand_count - visible_hands) / hand_count
            )
            collapsed_hand_fraction = _collapsed_hand_fraction(
                person,
                min_points=_MIN_HAND_KEYPOINTS,
                min_spread=min_hand_spread,
            )
            collapsed_hand_penalty = _COLLAPSED_HAND_PENALTY * collapsed_hand_fraction
            score -= missing_hand_penalty
            score -= collapsed_hand_penalty
            diagnostic["visible_hands"] = visible_hands
            diagnostic["missing_hand_penalty"] = missing_hand_penalty
            diagnostic["collapsed_hand_fraction"] = collapsed_hand_fraction
            diagnostic["collapsed_hand_penalty"] = collapsed_hand_penalty

        joint_geometry_penalty = _IMPOSSIBLE_ANGLE_PENALTY * _joint_geometry_penalty(
            person,
            min_angle_degrees=_MIN_JOINT_ANGLE_DEGREES,
            max_segment_ratio=_MAX_SEGMENT_RATIO,
        )
        limb_asymmetry_penalty = _ASYMMETRIC_LIMB_PENALTY * _limb_asymmetry_penalty(
            person,
            max_ratio=_MAX_LIMB_ASYMMETRY_RATIO,
        )
        score -= joint_geometry_penalty
        score -= limb_asymmetry_penalty
        diagnostic["joint_geometry_penalty"] = joint_geometry_penalty
        diagnostic["limb_asymmetry_penalty"] = limb_asymmetry_penalty
        if len(people) > 1:
            score -= _MULTI_PERSON_PENALTY
            diagnostic["multi_person_penalty"] = _MULTI_PERSON_PENALTY
        final_score = max(0.0, min(1.0, score))
        diagnostic["score"] = final_score
        return final_score, diagnostic

    def _write_debug_diagnostics(self, diagnostics: list[dict[str, Any]]) -> None:
        if self._debug_dir is None:
            return
        import json

        path = self._debug_dir / "pose_structure_diagnostics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for diagnostic in diagnostics:
                f.write(json.dumps(diagnostic, sort_keys=True) + "\n")

    def _requirements_from_artifact(self, artifact: Any) -> tuple[int, bool]:
        prompt = str(getattr(artifact, "prompt", "") or "")
        constraints = _constraint_texts(getattr(artifact, "metadata", None))
        texts = constraints if constraints else (prompt,)

        if self._require_hands == "never":
            hand_count = 0
        elif self._require_hands == "always" or _contains_hint(texts, _BOTH_HAND_HINTS):
            hand_count = 2
        else:
            hand_count = 1 if _contains_hint(texts, _HAND_HINTS) else 0

        if self._require_feet == "never":
            feet_required = False
        elif self._require_feet == "always":
            feet_required = True
        else:
            feet_required = _contains_hint(texts, _FEET_HINTS)

        return hand_count, feet_required


def pose_structure_reward_model(
    worker_config: Mapping[str, Any],
) -> PoseStructureRewardModel:
    return PoseStructureRewardModel(worker_config)


class _BatchPoseCallable:
    def __init__(self, detector: Callable[[list[Any]], list[Any]]) -> None:
        self._detector = detector

    def predict_batch(self, images: list[Any]) -> list[Any]:
        return list(self._detector(images))


def _predict_pose_batch(pose_model: Any, images: list[Any]) -> list[Any]:
    predict_batch = getattr(pose_model, "predict_batch", None)
    if callable(predict_batch):
        return list(predict_batch(images))
    if not callable(pose_model):
        raise TypeError("pose_model must expose predict_batch(images) or be callable")
    return list(pose_model(images))


def _extract_images(output: Any) -> list[Any]:
    if output is None:
        return []

    from PIL import Image

    if isinstance(output, Image.Image):
        return [output.convert("RGB")]
    if isinstance(output, (list, tuple)):
        images: list[Any] = []
        for item in output:
            images.extend(_extract_images(item))
        return images

    try:
        import torch

        if torch.is_tensor(output):
            tensor = output.detach().cpu()
            if tensor.ndim == 4:
                return _extract_images(tensor[tensor.shape[0] // 2])
            if tensor.ndim == 3:
                if tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor.permute(1, 2, 0)
                arr = tensor.float().numpy()
                if arr.max(initial=0.0) <= 1.0:
                    arr = arr * 255.0
                return [Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")]
    except ImportError:
        pass
    return []


__all__ = [
    "PoseStructureRewardModel",
    "_onnx_providers",
    "pose_structure_reward_model",
]
