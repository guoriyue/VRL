"""Typed contracts owned by the opt-in inference-quality tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Literal

EVIDENCE_SCHEMA_VERSION = 1


class InferencePhase(str, Enum):
    """Artifact identity evaluated by the preflight test."""

    BASE = "base"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """One approved primary checkpoint identity for a family profile."""

    path: str
    revision: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ModelIdentity:
        _reject_unknown_fields(cls, raw, "model identity")
        path = str(raw.get("path") or "").strip()
        if not path:
            raise ValueError("model identity path must be non-empty")
        revision_raw = raw.get("revision")
        revision = None if revision_raw is None else str(revision_raw).strip() or None
        return cls(path=path, revision=revision)


@dataclass(frozen=True, slots=True)
class QualityProfile:
    """Family identity selector layered over a modality protocol."""

    extends: str
    model_identities: tuple[ModelIdentity, ...]
    protocol_overrides: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> QualityProfile:
        _reject_unknown_fields(cls, raw, "quality profile")
        identities_raw = raw.get("model_identities")
        if not isinstance(identities_raw, list) or not identities_raw:
            raise ValueError("quality profile model_identities must be a non-empty list")
        identities: list[ModelIdentity] = []
        for item in identities_raw:
            if not isinstance(item, Mapping):
                raise ValueError("quality profile model identity must be a mapping")
            identities.append(ModelIdentity.from_mapping(item))
        extends = str(raw.get("extends") or "").strip()
        if not extends or "/" in extends or "\\" in extends:
            raise ValueError("quality profile extends must name one sibling protocol resource")
        overrides = raw.get("protocol_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("quality profile protocol_overrides must be a mapping")
        return cls(
            extends=extends,
            model_identities=tuple(identities),
            protocol_overrides=dict(overrides),
        )


@dataclass(frozen=True, slots=True)
class QualityProtocol:
    """Versioned image/video assertions evaluated without importing trainers.

    Visual quality is deliberately NOT scored by pixel statistics; the media
    files are decoded, hashed, and surfaced in the report for human review.
    """

    name: str
    media_kind: Literal["image", "video"]
    min_samples: int
    min_native_similarity: float
    min_alignment_margin: float
    max_replay_abs_error: float
    requires_condition_delta: bool
    min_condition_delta: float
    required_segments: tuple[str, ...]
    required_corruptions: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> QualityProtocol:
        _reject_unknown_fields(cls, raw, "quality protocol")
        segments_raw = raw.get("required_segments", ())
        if not isinstance(segments_raw, (list, tuple)):
            raise ValueError("quality protocol required_segments must be a list")
        corruptions_raw = raw.get("required_corruptions", ())
        if not isinstance(corruptions_raw, (list, tuple)):
            raise ValueError("quality protocol required_corruptions must be a list")
        protocol = cls(
            name=str(raw.get("name", "")).strip(),
            media_kind=str(raw.get("media_kind", "")).strip(),  # type: ignore[arg-type]
            min_samples=int(raw.get("min_samples", 0)),
            min_native_similarity=float(raw.get("min_native_similarity", 0.0)),
            min_alignment_margin=float(raw.get("min_alignment_margin", 0.0)),
            max_replay_abs_error=float(raw.get("max_replay_abs_error", 0.0)),
            requires_condition_delta=bool(raw.get("requires_condition_delta", False)),
            min_condition_delta=float(raw.get("min_condition_delta", 0.0)),
            required_segments=tuple(str(value) for value in segments_raw),
            required_corruptions=tuple(str(value) for value in corruptions_raw),
        )
        protocol._validate()
        return protocol

    def _validate(self) -> None:
        if not self.name:
            raise ValueError("quality protocol name must be non-empty")
        if self.media_kind not in {"image", "video"}:
            raise ValueError("quality protocol media_kind must be image or video")
        if self.min_samples <= 0:
            raise ValueError("quality protocol min_samples must be > 0")
        unit_interval_fields = {
            "min_native_similarity": self.min_native_similarity,
            "min_condition_delta": self.min_condition_delta,
        }
        for field_name, value in unit_interval_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"quality protocol {field_name} must be in [0, 1]")
        if self.max_replay_abs_error < 0.0:
            raise ValueError("quality protocol max_replay_abs_error must be >= 0")
        if len(set(self.required_corruptions)) != len(self.required_corruptions):
            raise ValueError("quality protocol required_corruptions contains duplicates")


@dataclass(frozen=True, slots=True)
class QualityAssertion:
    """One independently inspectable preflight assertion."""

    name: str
    passed: bool
    # Display/provenance-only: the boolean drives pytest; detail explains it.
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _reject_unknown_fields(cls: type[Any], raw: Mapping[str, Any], label: str) -> None:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")
