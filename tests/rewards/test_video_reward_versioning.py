"""Version and latency guards for video reward inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vrl.rewards.functions.video_reward import VideoReward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardRollout, RewardTrajectory


class _VersionedActorRuntime:
    num_workers = 1

    async def map(self, shards):
        return [
            [
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores={"overall_reward": 4.0},
                    selected_score=4.0,
                    reward_name=shard.reward_name,
                    score_key=shard.score_key,
                    policy_version=artifact.policy_version,
                    reward_model_version="reward-v2",
                    latency_ms=7.0,
                    queue_wait_ms=2.0,
                    inference_ms=5.0,
                    worker_id="reward-0",
                    metadata={"gpu_ids": [1], "node_ip": "127.0.0.1"},
                )
                for artifact in shard.artifacts
            ]
            for shard in shards
        ]

    async def shutdown(self) -> None:
        return None


def _rollout() -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(
            prompt="prompt",
            seed=0,
            steps=[],
            output=torch.ones(1, 2, 2, 2),
        ),
        metadata={"policy_version": 23, "sample_ids": ["sample-v"]},
    )


@pytest.mark.asyncio
async def test_video_reward_debug_records_versions_and_latency(tmp_path: Path) -> None:
    reward = VideoReward(
        inference_runtime="ray",
        reward_name="cosmos_reason1",
        score_key="overall_reward",
        artifact_dir=str(tmp_path / "reward_artifacts"),
        debug_dir=str(tmp_path / "reward_debug"),
        actor_runtime=_VersionedActorRuntime(),
    )

    scores = await reward.score_batch([_rollout()])

    assert scores == pytest.approx([4.0])
    assert reward.last_results[0].policy_version == 23
    assert reward.last_results[0].reward_model_version == "reward-v2"
    request_rows = [
        json.loads(line)
        for line in (tmp_path / "reward_debug" / "video_reward_requests.jsonl")
        .read_text()
        .splitlines()
    ]
    result_rows = [
        json.loads(line)
        for line in (tmp_path / "reward_debug" / "video_reward_results.jsonl")
        .read_text()
        .splitlines()
    ]

    assert request_rows[0]["policy_version"] == 23
    assert request_rows[0]["score_key"] == "overall_reward"
    assert request_rows[0]["artifact_materialization_ms"] >= 0
    assert request_rows[0]["inference_total_ms"] >= 0
    assert request_rows[0]["total_reward_latency_ms"] >= 0
    assert result_rows[0]["policy_version"] == 23
    assert result_rows[0]["score_key"] == "overall_reward"
    assert result_rows[0]["reward_model_version"] == "reward-v2"
    assert result_rows[0]["latency_ms"] == pytest.approx(7.0)
    assert result_rows[0]["queue_wait_ms"] == pytest.approx(2.0)
    assert result_rows[0]["inference_ms"] == pytest.approx(5.0)


def test_video_reward_rejects_async_scheduling_until_supported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scheduling"):
        VideoReward(
            inference_runtime="ray",
            reward_name="cosmos_reason1",
            score_key="overall_reward",
            artifact_dir=str(tmp_path / "reward_artifacts"),
            scheduling="async",
            actor_runtime=_VersionedActorRuntime(),
        )
