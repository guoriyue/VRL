"""Reward runtime resource and lifecycle tests."""

from __future__ import annotations

import asyncio

import pytest
from omegaconf import OmegaConf

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.ray.launcher import build_reward_ray_runtime
from vrl.scripts.common.factory import build_reward_from_cfg


def _request() -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path="/tmp/a0.pt",
                media_type="video",
                policy_version=11,
            ),
        ),
        reward_name="reward",
        score_key="overall_reward",
        policy_version=11,
    )


def test_reward_runtime_releases_actors_after_score_when_configured() -> None:
    ray = pytest.importorskip("ray")
    ray.shutdown()
    runtime = None
    try:
        runtime = build_reward_ray_runtime(
            {
                "inference_runtime": "ray",
                "worker_config": {
                    "scorer": "constant",
                    "scores": {"overall_reward": 3.0},
                    "reward_model_version": "lifecycle-v1",
                },
                "num_workers": 1,
                "cpus_per_worker": 0.5,
                "gpus_per_worker": 0.0,
                "release_after_score": True,
            },
            ray_init_kwargs={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": 1,
                "log_to_driver": False,
            },
        )

        results = asyncio.run(runtime.score_batch(_request()))

        assert results[0].selected_score == pytest.approx(3.0)
        assert runtime.actor_runtime._actor_group is None

        results = asyncio.run(runtime.score_batch(_request()))

        assert results[0].selected_score == pytest.approx(3.0)
        assert runtime.actor_runtime._actor_group is None
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()


def test_reward_config_receives_resolved_resource_plan() -> None:
    cfg = OmegaConf.create(
        {
            "distributed": {
                "backend": "ray",
                "resources": {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 0},
                    "rollout": {"num_gpus": 0, "gpus_per_worker": 0},
                    "reward": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
                },
                "rollout": {"release_after_collect": False},
                "reward": {
                    "release_after_score": True,
                    "cpus_per_worker": 0.5,
                    "max_inflight_batches": 1,
                    "placement_strategy": "STRICT_PACK",
                },
            },
            "reward": {
                "components": {"video_reward": 1.0},
                "kwargs": {
                    "video_reward": {
                        "inference_runtime": "ray",
                        "reward_name": "cosmos_reason1",
                        "score_key": "overall_reward",
                        "worker_config": {
                            "scorer": "constant",
                            "scores": {"overall_reward": 1.0},
                            "reward_model_version": "resource-v1",
                        },
                    },
                },
            },
        },
    )

    reward_fn = build_reward_from_cfg(
        cfg,
        built={
            "reward": (
                {"video_reward": 1.0},
                {
                    "video_reward": {
                        "inference_runtime": "ray",
                        "reward_name": "cosmos_reason1",
                        "score_key": "overall_reward",
                        "worker_config": {
                            "scorer": "constant",
                            "scores": {"overall_reward": 1.0},
                            "reward_model_version": "resource-v1",
                        },
                    },
                },
            ),
        },
        device="cpu",
    )

    video_reward = reward_fn.rewards[0][2]
    actor_runtime = video_reward.runtime.actor_runtime
    assert actor_runtime.num_workers == 1
    assert actor_runtime.gpus_per_worker == 1.0
    assert actor_runtime.expected_gpu_ids == (0,)
    assert actor_runtime.release_after_call is True
