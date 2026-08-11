"""Version and latency guards for video reward inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vrl.rewards.functions.kling_video_reward import KlingVideoReward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardSample


class _VersionedRuntime:
    async def score_batch(self, request):
        return [
            RewardInferenceResult(
                artifact_id=artifact.artifact_id,
                scores={"overall_reward": 4.0},
                reward_model_version="reward-v2",
                timing_ms={"queue_wait_ms": 2.0, "inference_ms": 5.0},
            )
            for artifact in request.artifacts
        ]

    async def shutdown(self) -> None:
        return None


def _sample() -> RewardSample:
    return RewardSample(
        prompt="prompt",
        output=torch.ones(1, 2, 2, 2),
        sample_id="sample-v",
    )


@pytest.mark.asyncio
async def test_video_reward_debug_records_versions_and_latency(tmp_path: Path) -> None:
    """Checks video reward debug records versions and latency."""
    reward = KlingVideoReward(
        reward_name="KlingTeam/VideoReward@main",
        score_key="overall_reward",
        artifact_format="tensor",  # codec-independent wiring test (no imageio dep)
        artifact_dir=str(tmp_path / "reward_artifacts"),
        debug_dir=str(tmp_path / "reward_debug"),
        inference_runtime=_VersionedRuntime(),
    )

    report = await reward.score_batch([_sample()])

    assert report.scores == pytest.approx([4.0])
    request_rows = [
        json.loads(line)
        for line in (tmp_path / "reward_debug" / "kling_video_reward_requests.jsonl")
        .read_text()
        .splitlines()
    ]
    result_rows = [
        json.loads(line)
        for line in (tmp_path / "reward_debug" / "kling_video_reward_results.jsonl")
        .read_text()
        .splitlines()
    ]

    assert request_rows[0]["score_key"] == "overall_reward"
    assert request_rows[0]["artifact_materialization_ms"] >= 0
    assert request_rows[0]["inference_total_ms"] >= 0
    assert request_rows[0]["total_reward_latency_ms"] >= 0
    assert result_rows[0]["reward_model_version"] == "reward-v2"
    assert result_rows[0]["timing_ms"]["queue_wait_ms"] == pytest.approx(2.0)
    assert result_rows[0]["timing_ms"]["inference_ms"] == pytest.approx(5.0)
