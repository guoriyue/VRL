"""NSFW safety penalty as a batch-capable RewardModel.

Returns a non-positive penalty (high NSFW probability lowers reward; low NSFW
never adds a bonus). Implements ``score_request`` so the classifier is called
once across all artifacts' images, then per-rollout penalties are aggregated —
identical to the original in-process NSFWSafetyReward behavior.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_DEFAULT_NSFW_LABELS = (
    "nsfw",
    "unsafe",
    "not safe",
    "not_safe",
    "explicit",
    "porn",
    "hentai",
)
_DEFAULT_SAFE_LABELS = ("sfw", "safe", "normal", "neutral")


class NSFWSafetyRewardModel:
    """Batch RewardModel returning ``{"nsfw_safety": penalty}`` per artifact."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._device = str(cfg.get("classifier_device") or cfg.get("device", "cuda"))
        self._model_name = str(cfg.get("model_name", "Falconsai/nsfw_image_detection"))
        self._threshold = _validate_probability(
            "threshold", cfg.get("threshold", 0.35), upper_open=True,
        )
        self._penalty_scale = _validate_lower_bounded(
            "penalty_scale", cfg.get("penalty_scale", 1.0), inclusive=True,
        )
        self._max_penalty = _validate_lower_bounded(
            "max_penalty", cfg.get("max_penalty", 1.0), inclusive=False,
        )
        self._image_sample_count = int(cfg.get("image_sample_count", 1))
        if self._image_sample_count <= 0:
            raise ValueError("image_sample_count must be > 0")
        self._aggregation = str(cfg.get("aggregation", "max"))
        if self._aggregation not in {"max", "mean"}:
            raise ValueError("aggregation must be 'max' or 'mean'")
        self._nsfw_labels = tuple(cfg.get("nsfw_labels") or _DEFAULT_NSFW_LABELS)
        self._safe_labels = tuple(cfg.get("safe_labels") or _DEFAULT_SAFE_LABELS)
        self._scorer: Callable[[list[Any]], Sequence[float]] | None = cfg.get("scorer")
        self._classifier: Any = None

    def score_request(self, request: Any) -> list[dict[str, float]]:
        image_groups = [
            _extract_images(artifact.as_media(), self._image_sample_count)
            for artifact in request.artifacts
        ]
        flat_images = [image for group in image_groups for image in group]
        flat_scores = self._score_images(flat_images)
        out: list[dict[str, float]] = []
        cursor = 0
        for group in image_groups:
            count = len(group)
            probs = flat_scores[cursor : cursor + count]
            cursor += count
            out.append({"nsfw_safety": self._penalty_from_probs(probs)})
        return out

    def __call__(self, *, artifact: Any, request: Any) -> dict[str, float]:
        images = _extract_images(artifact.as_media(), self._image_sample_count)
        return {"nsfw_safety": self._penalty_from_probs(self._score_images(images))}

    def _score_images(self, images: list[Any]) -> list[float]:
        if not images:
            return []
        if self._scorer is not None:
            scores = self._scorer(images)
            if len(scores) != len(images):
                raise ValueError(
                    "NSFW safety scorer returned wrong number of scores: "
                    f"got {len(scores)}, expected {len(images)}",
                )
            return [_clamp01(float(score)) for score in scores]

        self._ensure_loaded()
        try:
            raw_results = self._classifier(images, top_k=None)
        except TypeError:
            raw_results = self._classifier(images)
        return [
            self._probability_from_classifier_result(result)
            for result in _normalize_classifier_batch(raw_results, len(images))
        ]

    def _ensure_loaded(self) -> None:
        if self._classifier is not None:
            return
        from transformers import pipeline

        self._classifier = pipeline(
            "image-classification",
            model=self._model_name,
            device=_pipeline_device(self._device),
        )

    def _probability_from_classifier_result(self, result: Any) -> float:
        items = result if isinstance(result, list) else [result]
        nsfw_total = 0.0
        safe_total = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", ""))
            score = _clamp01(float(item.get("score", 0.0)))
            if _label_matches(label, self._nsfw_labels):
                nsfw_total += score
            elif _label_matches(label, self._safe_labels):
                safe_total += score
        if nsfw_total > 0.0:
            return _clamp01(nsfw_total)
        if safe_total > 0.0:
            return _clamp01(1.0 - safe_total)
        return 0.0

    def _penalty_from_probs(self, probs: Sequence[float]) -> float:
        if not probs:
            return 0.0
        probability = max(probs) if self._aggregation == "max" else sum(probs) / len(probs)
        excess = max(0.0, _clamp01(float(probability)) - self._threshold)
        if excess <= 0.0 or self._penalty_scale == 0.0:
            return 0.0
        normalized_excess = excess / (1.0 - self._threshold)
        penalty = min(self._max_penalty, self._penalty_scale * normalized_excess)
        return -float(penalty)


def nsfw_safety_reward_model(worker_config: Mapping[str, Any]) -> NSFWSafetyRewardModel:
    return NSFWSafetyRewardModel(worker_config)


def _extract_images(output: Any, max_images: int) -> list[Any]:
    images: list[Any] = []
    _append_images(output, images)
    if len(images) <= max_images:
        return images
    if max_images == 1:
        return [images[len(images) // 2]]
    step = (len(images) - 1) / float(max_images - 1)
    return [images[round(i * step)] for i in range(max_images)]


def _append_images(value: Any, images: list[Any]) -> None:
    if value is None:
        return

    from PIL import Image

    if isinstance(value, Image.Image):
        images.append(value.convert("RGB"))
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _append_images(item, images)
        return

    try:
        import torch

        if torch.is_tensor(value):
            _append_array_images(value.detach().cpu().numpy(), images)
            return
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            _append_array_images(value, images)
            return
    except ImportError:
        pass


def _append_array_images(array: Any, images: list[Any]) -> None:
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 5:
        if arr.shape[2] in (1, 3, 4):  # [B, T, C, H, W]
            arr = arr[:, arr.shape[1] // 2]
        elif arr.shape[1] in (1, 3, 4):  # [B, C, T, H, W]
            arr = arr[:, :, arr.shape[2] // 2]
        else:
            arr = arr.reshape((-1, *arr.shape[-3:]))

    if arr.ndim == 4:
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4:  # [C, T, H, W]
            image = _pil_from_array(arr[:, arr.shape[1] // 2])
            if image is not None:
                images.append(image)
            return
        for item in arr:
            image = _pil_from_array(item)
            if image is not None:
                images.append(image)
        return

    image = _pil_from_array(arr)
    if image is not None:
        images.append(image)


def _pil_from_array(array: Any) -> Any | None:
    import numpy as np
    from PIL import Image

    arr = np.asarray(array)
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] not in (1, 3, 4):
        return None

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr.astype("float32"), nan=0.0, posinf=1.0, neginf=0.0)
        if float(arr.min(initial=0.0)) < 0.0:
            arr = (arr + 1.0) * 0.5
        if float(arr.max(initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = arr.round().clip(0, 255).astype("uint8")
    elif arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype("uint8")

    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr).convert("RGB")


def _normalize_classifier_batch(raw_results: Any, expected_count: int) -> list[Any]:
    if expected_count == 1:
        if (
            isinstance(raw_results, list)
            and len(raw_results) == 1
            and not isinstance(raw_results[0], dict)
        ):
            return raw_results
        return [raw_results]
    if isinstance(raw_results, list) and len(raw_results) == expected_count:
        return raw_results
    raise ValueError(
        "NSFW classifier returned wrong number of results: "
        f"got {len(raw_results) if isinstance(raw_results, list) else type(raw_results).__name__}, "
        f"expected {expected_count}",
    )


def _label_matches(label: str, patterns: Sequence[str]) -> bool:
    normalized = label.strip().lower().replace("_", " ")
    for pattern in patterns:
        needle = str(pattern).strip().lower().replace("_", " ")
        if not needle:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", normalized):
            return True
    return False


def _pipeline_device(device: str) -> int:
    text = str(device).strip().lower()
    if text in {"", "cpu", "none"}:
        return -1
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        return int(suffix) if suffix else 0
    if text == "cuda":
        return 0
    return -1


def _validate_probability(name: str, value: float, *, upper_open: bool = False) -> float:
    out = float(value)
    upper_ok = out < 1.0 if upper_open else out <= 1.0
    if not (out >= 0.0 and upper_ok):
        relation = "<" if upper_open else "<="
        raise ValueError(f"{name} must satisfy 0.0 <= {name} {relation} 1.0")
    return out


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _validate_lower_bounded(name: str, value: float, *, inclusive: bool) -> float:
    out = float(value)
    below_bound = out < 0.0 if inclusive else out <= 0.0
    if not math.isfinite(out) or below_bound:
        bound = ">= 0" if inclusive else "> 0"
        raise ValueError(f"{name} must be finite and {bound}")
    return out


__all__ = ["NSFWSafetyRewardModel", "nsfw_safety_reward_model"]
