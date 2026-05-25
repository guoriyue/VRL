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
    )


def test_select_score_sums_composite_key() -> None:
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(),
        reward_name="reward",
        score_key="a+b",
    )
    assert request.select_score({"a": 1.0, "b": 2.5}) == pytest.approx(3.5)


def test_select_score_fails_on_missing_key() -> None:
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(),
        reward_name="reward",
        score_key="a+b",
    )
    with pytest.raises(KeyError, match="missing score keys"):
        request.select_score({"a": 1.0})


def test_request_rejects_duplicate_artifact_ids() -> None:
    artifact = _artifact("x")
    with pytest.raises(ValueError, match="duplicate"):
        RewardInferenceRequest(
            request_id="req",
            artifacts=(artifact, artifact),
            reward_name="reward",
            score_key="overall_reward",
        )


def test_validate_reward_results_orders_by_request_artifacts() -> None:
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
        ),
        RewardInferenceResult(
            artifact_id="a",
            scores={"overall_reward": 1.0},
            selected_score=1.0,
            reward_name="reward",
            score_key="overall_reward",
        ),
    ]

    ordered = validate_reward_results(request, results)

    assert [result.artifact_id for result in ordered] == ["a", "b"]


def test_result_rejects_selected_score_mismatch() -> None:
    with pytest.raises(ValueError, match="selected_score mismatch"):
        RewardInferenceResult(
            artifact_id="a",
            scores={"overall_reward": 2.0},
            selected_score=1.0,
            reward_name="reward",
            score_key="overall_reward",
        )
