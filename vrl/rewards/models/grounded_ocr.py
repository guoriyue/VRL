"""Atomic OCR spelling reward with a fail-closed visual grounding guard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from vrl.rewards.models.codex_image_qa import CodexImageQARewardModel
from vrl.rewards.models.ocr import OCRRewardModel


@dataclass(frozen=True, slots=True)
class GroundedOcrConfig:
    """Closed top-level configuration for the two owners of this reward."""

    ocr: dict[str, Any]
    guard: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any) -> GroundedOcrConfig:
        if not isinstance(value, Mapping):
            raise TypeError("grounded_ocr configuration must be a mapping")
        expected = {field.name for field in fields(cls)}
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            raise ValueError(
                f"invalid grounded_ocr configuration fields: missing={missing} unknown={unknown}",
            )
        nested: dict[str, dict[str, Any]] = {}
        for name in sorted(expected):
            raw = value[name]
            if not isinstance(raw, Mapping):
                raise TypeError(f"grounded_ocr.{name} must be a mapping")
            nested[name] = dict(raw)
        return cls(**nested)


class GroundedOCRRewardModel:
    """Multiply duplicate-aware OCR spelling by a mirrored binary guard.

    This model owns one domain objective rather than exposing two compensating
    reward components. A grounding or image-integrity failure therefore makes
    the final reward exactly zero before GRPO computes a group baseline.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        config = GroundedOcrConfig.from_mapping(worker_config)
        self.ocr = OCRRewardModel(config.ocr)
        if self.ocr.scoring_policy.near_duplicate_min_similarity is None:
            raise ValueError(
                "grounded_ocr requires ocr.near_duplicate_min_similarity",
            )
        self.guard = CodexImageQARewardModel(config.guard)
        if self.guard.comparison_mode != "binary_guard":
            raise ValueError(
                "grounded_ocr requires guard.comparison_mode='binary_guard'",
            )

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        artifacts = list(artifacts)
        if not artifacts:
            return []
        spelling_scores = [self.ocr(artifact) for artifact in artifacts]
        guard_scores = self.guard.score_batch(artifacts)
        if len(guard_scores) != len(spelling_scores):
            raise RuntimeError(
                "grounded_ocr inner reward length mismatch: "
                f"ocr={len(spelling_scores)} guard={len(guard_scores)}",
            )

        combined: list[dict[str, float]] = []
        for spelling, guard in zip(spelling_scores, guard_scores, strict=True):
            guard_pass = float(guard["codex_image_qa"])
            if guard_pass not in {0.0, 1.0}:
                raise ValueError(
                    f"grounded_ocr binary guard must return exactly 0 or 1; got {guard_pass}",
                )
            spelling_score = float(spelling["ocr"])
            combined.append(
                {
                    "grounded_ocr": spelling_score * guard_pass,
                    "grounded_ocr_spelling": spelling_score,
                    "grounded_ocr_spelling_raw": float(spelling["ocr_raw"]),
                    "grounded_ocr_near_duplicate_count": float(
                        spelling["ocr_near_duplicate_count"],
                    ),
                    "grounded_ocr_guard_pass": guard_pass,
                    **{str(name): float(value) for name, value in guard.items()},
                },
            )
        return combined


__all__ = ["GroundedOCRRewardModel", "GroundedOcrConfig"]
