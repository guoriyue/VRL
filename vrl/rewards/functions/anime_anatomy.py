"""Anime anatomy classifier rewards for Anima RL.

These rewards are thin wrappers around anime-domain image classifiers. They do
not embed a default model checkpoint: the experiment must provide one once the
positive and hard-negative calibration sets exist.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout


class AnimeAnatomyPlausibilityReward(RewardFunction):
    """Reward generated anime images for plausible limb/body structure."""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "",
        aggregation: str = "mean",
        scorer: Callable[[list[Any]], Sequence[Any]] | None = None,
        positive_labels: Sequence[str] = (),
        negative_labels: Sequence[str] = (),
    ) -> None:
        self._classifier = _AnimeClassifierScorer(
            device=device,
            model_name=model_name,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            aggregation=aggregation,
            scorer=scorer,
        )

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        return self._classifier.score_outputs([rollout.trajectory.output for rollout in rollouts])


class AnimeHandQualityReward(RewardFunction):
    """Reward generated anime images for stable hands and fingers."""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "",
        aggregation: str = "mean",
        scorer: Callable[[list[Any]], Sequence[Any]] | None = None,
        positive_labels: Sequence[str] = (),
        negative_labels: Sequence[str] = (),
    ) -> None:
        self._classifier = _AnimeClassifierScorer(
            device=device,
            model_name=model_name,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            aggregation=aggregation,
            scorer=scorer,
        )

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        return self._classifier.score_outputs([rollout.trajectory.output for rollout in rollouts])


class _AnimeClassifierScorer:
    def __init__(
        self,
        *,
        device: str,
        model_name: str,
        positive_labels: Sequence[str],
        negative_labels: Sequence[str],
        aggregation: str,
        scorer: Callable[[list[Any]], Sequence[Any]] | None,
    ) -> None:
        if aggregation not in {"mean", "max"}:
            raise ValueError("aggregation must be 'mean' or 'max'")
        self._device = str(device)
        self._model_name = str(model_name or "").strip()
        self._positive_labels = tuple(_normalize_label(label) for label in positive_labels)
        self._negative_labels = tuple(_normalize_label(label) for label in negative_labels)
        self._aggregation = aggregation
        self._scorer = scorer
        self._pipeline: Any = None

    def score_outputs(self, outputs: list[Any]) -> list[float]:
        image_groups = [_extract_images(output) for output in outputs]
        flat_images = [image for group in image_groups for image in group]
        flat_scores = self._score_images(flat_images)
        out: list[float] = []
        cursor = 0
        for group in image_groups:
            count = len(group)
            scores = flat_scores[cursor : cursor + count]
            cursor += count
            if not scores:
                out.append(0.0)
            elif self._aggregation == "max":
                out.append(max(scores))
            else:
                out.append(sum(scores) / len(scores))
        return out

    def _score_images(self, images: list[Any]) -> list[float]:
        if not images:
            return []
        raw = (
            self._scorer(images)
            if self._scorer is not None
            else self._pipeline_score(images)
        )
        if len(raw) != len(images):
            raise ValueError(
                "anime anatomy classifier returned wrong number of scores: "
                f"got {len(raw)}, expected {len(images)}",
            )
        return [self._score_result(result) for result in raw]

    def _pipeline_score(self, images: list[Any]) -> Sequence[Any]:
        if not self._model_name:
            raise ValueError(
                "anime anatomy reward requires model_name or an injected scorer",
            )
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-classification",
                model=self._model_name,
                device=_pipeline_device(self._device),
            )
        try:
            return self._pipeline(images, top_k=None)
        except TypeError:
            return self._pipeline(images)

    def _score_result(self, result: Any) -> float:
        if isinstance(result, (int, float)):
            return _clamp01(float(result))
        labels = _label_scores(result)
        positive = _max_matching(labels, self._positive_labels)
        negative = _max_matching(labels, self._negative_labels)
        if positive is None and negative is None:
            return 0.0
        if positive is None:
            return _clamp01(1.0 - float(negative or 0.0))
        if negative is None:
            return _clamp01(float(positive))
        return _clamp01((float(positive) + (1.0 - float(negative))) / 2.0)


def _extract_images(output: Any) -> list[Any]:
    images: list[Any] = []
    _append_images(output, images)
    return images


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
            tensor = value.detach().cpu()
            if tensor.ndim == 4:
                # Use the middle frame/image for reward classifier scoring.
                _append_images(tensor[tensor.shape[0] // 2], images)
                return
            if tensor.ndim == 3:
                if tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor[:3].permute(1, 2, 0)
                array = (tensor.float().clamp(0, 1) * 255).to(torch.uint8).numpy()
                images.append(Image.fromarray(array).convert("RGB"))
                return
    except ImportError:
        pass

    images.append(value)


def _label_scores(result: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(result, Mapping):
        if "label" in result and "score" in result:
            out[_normalize_label(str(result["label"]))] = _clamp01(float(result["score"]))
            return out
        for key, value in result.items():
            try:
                out[_normalize_label(str(key))] = _clamp01(float(value))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        for item in result:
            out.update(_label_scores(item))
    return out


def _max_matching(scores: Mapping[str, float], labels: Sequence[str]) -> float | None:
    matched = [score for label, score in scores.items() if label in labels]
    return max(matched) if matched else None


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _pipeline_device(device: str) -> int:
    text = str(device)
    if text.startswith("cuda"):
        parts = text.split(":", 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
    return -1


__all__ = ["AnimeAnatomyPlausibilityReward", "AnimeHandQualityReward"]
