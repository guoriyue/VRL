"""Paddle-free OCR text comparison contracts shared by data and reward code."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any


class OcrTextSelection(StrEnum):
    """How detected OCR lines become candidates for target comparison."""

    ALL_TEXT = "all_text"
    BEST_COMPLETE_LINE = "best_complete_line"
    BEST_CONTIGUOUS_LINES = "best_contiguous_lines"


class OcrEngineProfile(StrEnum):
    """Pinned OCR model families whose outputs define reward semantics."""

    FLOW_GRPO_COMPAT = "flow_grpo_compat"
    PP_OCRV6_MEDIUM = "ppocrv6_medium"

    @classmethod
    def parse(cls, value: Any, *, what: str = "OCR engine_profile") -> OcrEngineProfile:
        if not isinstance(value, str):
            raise TypeError(f"{what} must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            choices = [profile.value for profile in cls]
            raise ValueError(f"{what} must be one of {choices}; got {value!r}") from exc


@dataclass(frozen=True, slots=True)
class OcrScoringPolicy:
    """Resolved OCR scoring semantics persisted with experiments and reports."""

    text_selection: OcrTextSelection = OcrTextSelection.ALL_TEXT
    substring_full_credit: bool = True
    exclusive_alphanumeric_lines: bool = False
    extra_line_min_confidence: float = 0.5
    near_duplicate_min_similarity: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text_selection, OcrTextSelection):
            raise TypeError("OCR text_selection must be an OcrTextSelection")
        if not isinstance(self.substring_full_credit, bool):
            raise TypeError("OCR substring_full_credit must be a bool")
        if not isinstance(self.exclusive_alphanumeric_lines, bool):
            raise TypeError("OCR exclusive_alphanumeric_lines must be a bool")
        threshold = self.extra_line_min_confidence
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("OCR extra_line_min_confidence must be finite and in [0, 1]")
        object.__setattr__(self, "extra_line_min_confidence", float(threshold))
        duplicate_threshold = self.near_duplicate_min_similarity
        if duplicate_threshold is not None:
            if (
                isinstance(duplicate_threshold, bool)
                or not isinstance(duplicate_threshold, (int, float))
                or not math.isfinite(duplicate_threshold)
                or not 0.0 <= duplicate_threshold <= 1.0
            ):
                raise ValueError(
                    "OCR near_duplicate_min_similarity must be null or finite and in [0, 1]",
                )
            object.__setattr__(
                self,
                "near_duplicate_min_similarity",
                float(duplicate_threshold),
            )
        if self.text_selection is not OcrTextSelection.ALL_TEXT and self.substring_full_credit:
            raise ValueError(
                f"{self.text_selection.value} requires substring_full_credit=false so a "
                "larger detected token cannot receive exact credit",
            )
        if self.exclusive_alphanumeric_lines and self.text_selection is OcrTextSelection.ALL_TEXT:
            raise ValueError(
                "exclusive_alphanumeric_lines requires a line-preserving text_selection",
            )
        if (
            self.near_duplicate_min_similarity is not None
            and self.text_selection is OcrTextSelection.ALL_TEXT
        ):
            raise ValueError(
                "near_duplicate_min_similarity requires a line-preserving text_selection",
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        what: str = "OCR scoring policy",
    ) -> OcrScoringPolicy:
        """Parse a fail-closed persisted or configuration policy."""

        if not isinstance(value, Mapping):
            raise TypeError(f"{what} must be a mapping")
        expected = {field.name for field in fields(cls)}
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            raise ValueError(f"invalid {what} fields: missing={missing} unknown={unknown}")
        raw_selection = value["text_selection"]
        if not isinstance(raw_selection, str):
            raise TypeError(f"{what} text_selection must be a string")
        try:
            text_selection = OcrTextSelection(raw_selection)
        except ValueError as exc:
            choices = [selection.value for selection in OcrTextSelection]
            raise ValueError(
                f"{what} text_selection must be one of {choices}; got {raw_selection!r}",
            ) from exc
        substring_full_credit = value["substring_full_credit"]
        if not isinstance(substring_full_credit, bool):
            raise TypeError(f"{what} substring_full_credit must be a bool")
        exclusive_alphanumeric_lines = value["exclusive_alphanumeric_lines"]
        if not isinstance(exclusive_alphanumeric_lines, bool):
            raise TypeError(f"{what} exclusive_alphanumeric_lines must be a bool")
        return cls(
            text_selection=text_selection,
            substring_full_credit=substring_full_credit,
            exclusive_alphanumeric_lines=exclusive_alphanumeric_lines,
            extra_line_min_confidence=value["extra_line_min_confidence"],
            near_duplicate_min_similarity=value["near_duplicate_min_similarity"],
        )

    def to_record(self) -> dict[str, str | bool | float | None]:
        """Serialize using the typed fields as the sole key source of truth."""

        return {
            "text_selection": self.text_selection.value,
            "substring_full_credit": self.substring_full_credit,
            "exclusive_alphanumeric_lines": self.exclusive_alphanumeric_lines,
            "extra_line_min_confidence": self.extra_line_min_confidence,
            "near_duplicate_min_similarity": self.near_duplicate_min_similarity,
        }


def normalize_ocr_text(text: str) -> str:
    """Match Flow-GRPO OCR targets by lowercasing and removing ASCII spaces.

    Punctuation remains significant. Keeping this narrow contract independent
    from the OCR model lets dataset derivation use the exact runtime comparison
    key without importing PaddleOCR or media dependencies.
    """

    return text.replace(" ", "").lower()


__all__ = [
    "OcrEngineProfile",
    "OcrScoringPolicy",
    "OcrTextSelection",
    "normalize_ocr_text",
]
