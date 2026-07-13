"""Model-neutral assertions over native, production, and replay artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tests.quality.contracts import (
    EVIDENCE_SCHEMA_VERSION,
    InferencePhase,
    QualityAssertion,
    QualityProtocol,
)
from tests.quality.io import sha256_file
from vrl.utils.media import read_image_as_frames, read_video_frames


def evaluate_artifact_evidence(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    protocol: QualityProtocol,
    family: str,
    phase: InferencePhase,
    checkpoint_sha256: str | None,
    expected_provenance: Mapping[str, str],
) -> tuple[list[QualityAssertion], list[dict[str, str]], dict[str, str]]:
    """Evaluate fixed artifacts; this function never starts a trainer or model."""

    assertions: list[QualityAssertion] = []
    artifacts: list[dict[str, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        assertions.append(QualityAssertion(name=name, passed=bool(passed), detail=detail))

    check(
        "evidence_schema",
        manifest.get("schema_version") == EVIDENCE_SCHEMA_VERSION,
        f"expected={EVIDENCE_SCHEMA_VERSION} actual={manifest.get('schema_version')!r}",
    )
    check(
        "family_identity",
        manifest.get("family") == family,
        f"expected={family!r} actual={manifest.get('family')!r}",
    )
    check(
        "phase_identity",
        manifest.get("phase") == phase.value,
        f"expected={phase.value!r} actual={manifest.get('phase')!r}",
    )
    for name, expected in expected_provenance.items():
        actual = manifest.get(name)
        check(
            f"{name}_identity",
            actual == expected,
            f"expected={expected!r} actual={actual!r}",
        )
    model_identity = str(manifest.get("model_identity") or "").strip()

    scorer = manifest.get("scorer")
    scorer_identity = ""
    if isinstance(scorer, Mapping):
        scorer_name = str(scorer.get("name") or "").strip()
        scorer_revision = str(scorer.get("revision") or "").strip()
        scorer_identity = f"{scorer_name}@{scorer_revision}"
        scorer_ok = bool(scorer_name and scorer_revision)
    else:
        scorer_ok = False
    check("scorer_identity", scorer_ok, "scorer requires non-empty name and revision")

    if phase is InferencePhase.CHECKPOINT:
        actual_checkpoint = str(manifest.get("checkpoint_sha256") or "")
        check(
            "checkpoint_identity",
            bool(checkpoint_sha256) and actual_checkpoint == checkpoint_sha256,
            f"expected={checkpoint_sha256!r} actual={actual_checkpoint!r}",
        )

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        samples = []
    check(
        "sample_count",
        len(samples) >= protocol.min_samples,
        f"actual={len(samples)} minimum={protocol.min_samples}",
    )

    base_dir = manifest_path.parent
    for index, raw_sample in enumerate(samples):
        prefix = f"sample_{index}"
        if not isinstance(raw_sample, Mapping):
            check(f"{prefix}.shape", False, "sample must be an object")
            continue
        prompt = str(raw_sample.get("prompt") or "").strip()
        check(f"{prefix}.prompt", bool(prompt), "prompt must be non-empty")
        seed = raw_sample.get("seed")
        check(
            f"{prefix}.seed",
            isinstance(seed, int) and not isinstance(seed, bool),
            f"seed={seed!r}",
        )

        native_path = _resolve_artifact_path(base_dir, raw_sample.get("native"))
        production_path = _resolve_artifact_path(base_dir, raw_sample.get("production"))
        if native_path is None or production_path is None:
            check(f"{prefix}.artifacts", False, "native and production paths are required")
            continue
        try:
            native = _load_media(native_path, protocol.media_kind)
            production = _load_media(production_path, protocol.media_kind)
        except (OSError, ValueError, RuntimeError) as exc:
            check(f"{prefix}.decode", False, f"{type(exc).__name__}: {exc}")
            continue
        # Visual quality is judged by a human looking at these files, not by
        # pixel statistics; the report lists them so the reviewer can open them.
        for role, path in (("native", native_path), ("production", production_path)):
            artifacts.append(
                {
                    "sample": str(index),
                    "prompt": prompt,
                    "role": role,
                    "path": str(path),
                    "sha256": sha256_file(path),
                },
            )

        check(
            f"{prefix}.shape_match",
            native.shape == production.shape,
            f"native={tuple(native.shape)} production={tuple(production.shape)}",
        )
        native_similarity = (
            1.0 - float((native - production).abs().mean().item())
            if native.shape == production.shape
            else 0.0
        )
        check(
            f"{prefix}.native_similarity",
            native_similarity >= protocol.min_native_similarity,
            f"actual={native_similarity:.6f} minimum={protocol.min_native_similarity:.6f}",
        )

        replay_error = _finite_float(raw_sample.get("replay_max_abs_error"))
        check(
            f"{prefix}.replay_parity",
            replay_error is not None and replay_error <= protocol.max_replay_abs_error,
            f"actual={replay_error!r} maximum={protocol.max_replay_abs_error:.6f}",
        )
        matched = _finite_float(raw_sample.get("matched_alignment"))
        shuffled = _finite_float(raw_sample.get("shuffled_alignment"))
        margin = None if matched is None or shuffled is None else matched - shuffled
        check(
            f"{prefix}.alignment_margin",
            margin is not None and margin >= protocol.min_alignment_margin,
            f"actual={margin!r} minimum={protocol.min_alignment_margin:.6f}",
        )
        if protocol.requires_condition_delta:
            condition_delta = _finite_float(raw_sample.get("condition_delta"))
            check(
                f"{prefix}.condition_delta",
                condition_delta is not None and condition_delta >= protocol.min_condition_delta,
                f"actual={condition_delta!r} minimum={protocol.min_condition_delta:.6f}",
            )

        segments = raw_sample.get("segments", [])
        if not isinstance(segments, list):
            segments = []
        check(
            f"{prefix}.segments",
            tuple(str(value) for value in segments) == protocol.required_segments,
            f"expected={protocol.required_segments!r} actual={tuple(segments)!r}",
        )

        clean_score = _finite_float(raw_sample.get("clean_quality_score"))
        corruption_scores = raw_sample.get("corruption_scores")
        if not isinstance(corruption_scores, Mapping):
            corruption_scores = {}
        for corruption in protocol.required_corruptions:
            corrupt_score = _finite_float(corruption_scores.get(corruption))
            gap = (
                None
                if clean_score is None or corrupt_score is None
                else clean_score - corrupt_score
            )
            check(
                f"{prefix}.corruption.{corruption}",
                gap is not None and gap >= protocol.min_alignment_margin,
                f"clean={clean_score!r} corrupt={corrupt_score!r} gap={gap!r} "
                f"minimum={protocol.min_alignment_margin:.6f}",
            )

    return assertions, artifacts, {"model": model_identity, "scorer": scorer_identity}


def _resolve_artifact_path(base_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_media(path: Path, media_kind: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames = read_video_frames(path) if media_kind == "video" else read_image_as_frames(path)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"media must be [T,H,W,3], got {tuple(frames.shape)}")
    return frames.float()


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
