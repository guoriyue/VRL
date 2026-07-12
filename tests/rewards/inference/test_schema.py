"""Tests for generic reward inference schema."""

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
        media_type="video",
        prompt="prompt",
        source_request_id="source-request",
        sample_id=f"sample-{artifact_id}",
        group_id="group-0",
        trajectory_id=f"trajectory-{artifact_id}",
        policy_version=7,
    )


def test_select_score_sums_composite_key() -> None:
    """Checks select score sums composite key."""
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(),
        reward_name="reward",
        score_key="a+b",
    )
    assert request.select_score({"a": 1.0, "b": 2.5}) == pytest.approx(3.5)


def test_select_score_fails_on_missing_key() -> None:
    """Checks select score fails on missing key."""
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(),
        reward_name="reward",
        score_key="a+b",
    )
    with pytest.raises(KeyError, match="missing score keys"):
        request.select_score({"a": 1.0})


def test_request_rejects_duplicate_artifact_ids() -> None:
    """Checks request rejects duplicate artifact IDs."""
    artifact = _artifact("x")
    with pytest.raises(ValueError, match="duplicate"):
        RewardInferenceRequest(
            request_id="req",
            artifacts=(artifact, artifact),
            reward_name="reward",
            score_key="overall_reward",
        )


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
            media_type="video",
            size_bytes=size_bytes,
            sha256=sha256,
        )


def test_inmemory_artifact_rejects_file_integrity() -> None:
    with pytest.raises(ValueError, match="in-memory media cannot declare"):
        RewardInferenceArtifact(
            artifact_id="artifact",
            path="",
            media_type="image",
            media=object(),
            size_bytes=1,
            sha256="0" * 64,
        )


def test_validate_reward_results_orders_by_request_artifacts() -> None:
    """Checks validate reward results orders by request artifacts."""
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(_artifact("a"), _artifact("b")),
        reward_name="reward",
        score_key="overall_reward",
    )
    results = [
        RewardInferenceResult(
            artifact_id="b",
            scores={"overall_reward": 2.0},
            selected_score=2.0,
            reward_name="reward",
            score_key="overall_reward",
            source_request_id="source-request",
            sample_id="sample-b",
            group_id="group-0",
            trajectory_id="trajectory-b",
            policy_version=7,
        ),
        RewardInferenceResult(
            artifact_id="a",
            scores={"overall_reward": 1.0},
            selected_score=1.0,
            reward_name="reward",
            score_key="overall_reward",
            source_request_id="source-request",
            sample_id="sample-a",
            group_id="group-0",
            trajectory_id="trajectory-a",
            policy_version=7,
        ),
    ]

    ordered = validate_reward_results(request, results)

    assert [result.artifact_id for result in ordered] == ["a", "b"]


def test_validate_reward_results_rejects_lineage_mismatch() -> None:
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(_artifact("a"),),
        reward_name="reward",
        score_key="overall_reward",
        policy_version=7,
    )
    result = RewardInferenceResult(
        artifact_id="a",
        scores={"overall_reward": 1.0},
        selected_score=1.0,
        reward_name="reward",
        score_key="overall_reward",
        source_request_id="wrong-request",
        sample_id="sample-a",
        group_id="group-0",
        trajectory_id="trajectory-a",
        policy_version=7,
    )

    with pytest.raises(RuntimeError, match="source_request_id"):
        validate_reward_results(request, [result])


def test_result_rejects_selected_score_mismatch() -> None:
    """Checks result rejects selected score mismatch."""
    with pytest.raises(ValueError, match="selected_score mismatch"):
        RewardInferenceResult(
            artifact_id="a",
            scores={"overall_reward": 2.0},
            selected_score=1.0,
            reward_name="reward",
            score_key="overall_reward",
        )
