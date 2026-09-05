"""Tests for the generic reward inference schema."""

from __future__ import annotations

import pytest

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)


def _artifact(artifact_id: str) -> RewardInferenceArtifact:
    return RewardInferenceArtifact(
        artifact_id=artifact_id,
        sample_id=f"sample-{artifact_id}",
        path=f"/tmp/{artifact_id}.pt",
        prompt="prompt",
    )


def test_request_rejects_duplicate_artifact_ids() -> None:
    artifact = _artifact("x")
    with pytest.raises(ValueError, match="duplicate"):
        RewardInferenceRequest(request_id="req", artifacts=(artifact, artifact))


def test_artifact_requires_a_nonempty_sample_id() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        RewardInferenceArtifact(
            artifact_id="artifact",
            sample_id="",
            path="/tmp/artifact.pt",
        )


@pytest.mark.parametrize(
    ("size_bytes", "sha256", "message"),
    [
        (1, None, "must be set together"),
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
            sample_id="sample-0",
            path="/tmp/artifact.pt",
            size_bytes=size_bytes,
            sha256=sha256,
        )


def test_inmemory_artifact_rejects_file_integrity() -> None:
    with pytest.raises(ValueError, match="in-memory media cannot declare"):
        RewardInferenceArtifact(
            artifact_id="artifact",
            sample_id="sample-0",
            path="",
            media=object(),
            size_bytes=1,
            sha256="0" * 64,
        )


def test_request_validates_and_orders_results_by_its_artifacts() -> None:
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(_artifact("a"), _artifact("b")),
    )
    results = [
        RewardInferenceResult(artifact_id="b", scores={"overall_reward": 2.0}),
        RewardInferenceResult(artifact_id="a", scores={"overall_reward": 1.0}),
    ]

    ordered = request.validate_and_order_results(results)

    assert [result.artifact_id for result in ordered] == ["a", "b"]


def test_result_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="scores must be finite"):
        RewardInferenceResult(artifact_id="a", scores={"overall_reward": float("nan")})


def test_result_rejects_invalid_timing() -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        RewardInferenceResult(
            artifact_id="a",
            scores={"overall_reward": 1.0},
            timing_ms={"inference_ms": -1.0},
        )
