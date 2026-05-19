"""Tests for reward inference runtime helpers."""

from __future__ import annotations

import pytest

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    build_reward_inference_runtime,
    shard_reward_request,
)


def _request(count: int = 3) -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req",
        artifacts=tuple(
            RewardInferenceArtifact(
                artifact_id=f"a{i}",
                path=f"/tmp/a{i}.pt",
                media_type="video",
            )
            for i in range(count)
        ),
        reward_name="reward",
        score_key="overall_reward",
    )


def test_shard_reward_request_balances_artifacts() -> None:
    shards = shard_reward_request(_request(5), num_shards=2)

    assert [len(shard.artifacts) for shard in shards] == [3, 2]
    assert [artifact.artifact_id for shard in shards for artifact in shard.artifacts] == [
        "a0",
        "a2",
        "a4",
        "a1",
        "a3",
    ]


def test_runtime_factory_rejects_missing_worker_config() -> None:
    with pytest.raises(ValueError, match="worker_config"):
        build_reward_inference_runtime({"inference_runtime": "ray"})


def test_runtime_factory_rejects_non_ray_runtime() -> None:
    with pytest.raises(ValueError, match="must be 'ray'"):
        build_reward_inference_runtime({"inference_runtime": "local"})
