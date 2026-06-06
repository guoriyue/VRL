"""Tests for the in-process LocalRewardRuntime transport."""

from __future__ import annotations

import pytest

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.runtime import LocalRewardRuntime


class _SumMediaModel:
    """Toy RewardModel: scores = sum of the in-memory media values."""

    def __call__(self, *, artifact, request):
        total = float(sum(artifact.as_media()))
        return {"overall": total, "extra": 1.0}


def _make_request(score_key: str = "overall") -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req-1",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a", path="", media_type="image", media=[1.0, 2.0],
            ),
            RewardInferenceArtifact(
                artifact_id="b", path="", media_type="image", media=[3.0],
            ),
        ),
        reward_name="fake",
        score_key=score_key,
    )


@pytest.mark.asyncio
async def test_local_runtime_scores_in_process_without_disk_or_ray() -> None:
    """Checks local runtime scores in process without disk or Ray."""
    runtime = LocalRewardRuntime(model=_SumMediaModel())
    results = await runtime.score_batch(_make_request())

    assert [r.artifact_id for r in results] == ["a", "b"]  # original order preserved
    assert results[0].selected_score == pytest.approx(3.0)
    assert results[1].selected_score == pytest.approx(3.0)
    assert all(r.worker_id == "local" for r in results)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_local_runtime_composite_score_key_sums_components() -> None:
    """Checks local runtime composite score key sums components."""
    runtime = LocalRewardRuntime(model=_SumMediaModel())
    results = await runtime.score_batch(_make_request(score_key="overall+extra"))

    assert results[0].selected_score == pytest.approx(4.0)  # 3.0 + 1.0
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_local_runtime_empty_request_returns_empty() -> None:
    """Checks local runtime empty request returns empty."""
    runtime = LocalRewardRuntime(model=_SumMediaModel())
    request = RewardInferenceRequest(
        request_id="req-empty",
        artifacts=(),
        reward_name="fake",
        score_key="overall",
    )
    assert await runtime.score_batch(request) == []
