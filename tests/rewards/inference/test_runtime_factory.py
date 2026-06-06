"""Tests for reward inference runtime: request sharding, worker-config validation, and model-factory loading."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.kling_video_reward import KlingVideoReward
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    shard_reward_request,
)
from vrl.rewards.ray import RayRewardRuntime
from vrl.rewards.ray.worker import RewardModelWorker


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
    """Checks shard reward request balances artifacts."""
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
    """Checks runtime factory rejects missing worker config."""
    with pytest.raises(ValueError, match="worker_config"):
        RayRewardRuntime({"inference_runtime": "ray"})


def test_reward_name_without_worker_config_is_not_a_worker_loader() -> None:
    """Checks reward name without worker config is not a worker loader."""
    with pytest.raises(ValueError, match="worker_config"):
        RayRewardRuntime(
            {
                "inference_runtime": "ray",
                "reward_name": "KlingTeam/VideoReward@main",
                "score_key": "overall_reward",
            },
        )


def test_video_reward_derives_internal_model_factory_from_reward_name(tmp_path) -> None:
    """Checks video reward derives internal model factory from reward name."""
    reward = KlingVideoReward(
        reward_name="KlingTeam/VideoReward@main",
        score_key="overall_reward",
        artifact_dir=str(tmp_path),
        worker_config={"model_path": "", "dtype": "bfloat16"},
    )

    assert reward._actor_runtime.worker_config == {
        "model_path": "",
        "dtype": "bfloat16",
        "model_factory": "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel",
        "reward_model_name": "KlingTeam/VideoReward@main",
        "reward_model_version": "KlingTeam/VideoReward@main",
    }


def test_worker_loads_reward_model_via_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks worker loads reward model via factory."""
    class _FakeRewardModel:
        def __init__(self, worker_config):
            self.worker_config = worker_config

        def __call__(self, *, artifact, request):
            assert artifact.prompt == "prompt"
            assert request.score_key == "overall_reward"
            assert self.worker_config["reward_model_name"] == "KlingTeam/VideoReward@main"
            return {"overall_reward": 3.0, "motion_quality": 1.0}

    def _build(worker_config):
        return _FakeRewardModel(worker_config)

    monkeypatch.setattr(
        "vrl.rewards.ray.worker.import_from_path",
        lambda path: _build,
    )
    worker = RewardModelWorker(
        "reward-0",
        {
            "model_factory": "fake.module:build_model",
            "reward_model_name": "KlingTeam/VideoReward@main",
            "reward_model_version": "KlingTeam/VideoReward@main",
        },
    )
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path="/tmp/a0.mp4",
                media_type="video",
                prompt="prompt",
            ),
        ),
        reward_name="kling_video_reward",
        score_key="overall_reward",
    )

    results = worker.score_batch(request)

    assert results[0].selected_score == pytest.approx(3.0)
    assert results[0].reward_model_version == "KlingTeam/VideoReward@main"


def test_worker_requires_explicit_model_factory_even_with_reward_model_name() -> None:
    """Checks worker requires explicit model factory even with reward model name."""
    with pytest.raises(ValueError, match="model_factory"):
        RewardModelWorker("reward-0", {})
    with pytest.raises(ValueError, match="model_factory"):
        RewardModelWorker(
            "reward-0",
            {"reward_model_name": "KlingTeam/VideoReward@main"},
        )
