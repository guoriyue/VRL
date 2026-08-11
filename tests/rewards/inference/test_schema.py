"""Tests for the generic reward inference schema."""

from __future__ import annotations

import pytest

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
    validate_reward_results,
)


def _artifact(artifact_id: str) -> RewardInferenceArtifact:
    return RewardInferenceArtifact(
        artifact_id=artifact_id,
        path=f"/tmp/{artifact_id}.pt",
        prompt="prompt",
    )


def test_request_rejects_duplicate_artifact_ids() -> None:
    artifact = _artifact("x")
    with pytest.raises(ValueError, match="duplicate"):
        RewardInferenceRequest(request_id="req", artifacts=(artifact, artifact))


@pytest.mark.parametrize(
    ("size_bytes", "sha256", "message"),
    [
        (1, None, "must be set together"),
        (None, "0" * 64, "must be set together"),
        (-1, "0" * 64, "size_bytes must be >= 0"),
        (1, "not-a-digest", "lowercase hex SHA-256"),
    ],
)
def test_artifact_rejects_invalid_file_integrity(
    size_bytes: int | None,
    sha256: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RewardInferenceArtifact(
            artifact_id="artifact",
            path="/tmp/artifact.pt",
            size_bytes=size_bytes,
            sha256=sha256,
        )


def test_inmemory_artifact_rejects_file_integrity() -> None:
    with pytest.raises(ValueError, match="in-memory media cannot declare"):
        RewardInferenceArtifact(
            artifact_id="artifact",
            path="",
            media=object(),
            size_bytes=1,
            sha256="0" * 64,
        )


def test_validate_reward_results_orders_by_request_artifacts() -> None:
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(_artifact("a"), _artifact("b")),
    )
    results = [
        RewardInferenceResult(artifact_id="b", scores={"overall_reward": 2.0}),
        RewardInferenceResult(artifact_id="a", scores={"overall_reward": 1.0}),
    ]

    ordered = validate_reward_results(request, results)

    assert [result.artifact_id for result in ordered] == ["a", "b"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_result_rejects_nonfinite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="scores must be finite"):
        RewardInferenceResult(artifact_id="a", scores={"overall_reward": score})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_result_rejects_invalid_timing(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        RewardInferenceResult(
            artifact_id="a",
            scores={"overall_reward": 1.0},
            timing_ms={"inference_ms": value},
        )
