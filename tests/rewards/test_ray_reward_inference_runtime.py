"""Ray-backed reward inference runtime tests."""

from __future__ import annotations

import asyncio

import pytest

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
)
from vrl.rewards.ray.launcher import build_reward_ray_runtime


def test_ray_reward_runtime_scores_artifacts_with_fake_worker() -> None:
    ray = pytest.importorskip("ray")
    ray.shutdown()
    runtime = None
    try:
        runtime = build_reward_ray_runtime(
            {
                "inference_runtime": "ray",
                "worker_config": {
                    "scorer": "constant",
                    "scores": {"overall_reward": 2.0},
                    "reward_model_version": "fake-v1",
                },
                "num_workers": 2,
                "cpus_per_worker": 0.5,
                "gpus_per_worker": 0.0,
            },
            ray_init_kwargs={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": 2,
                "log_to_driver": False,
            },
        )
        request = RewardInferenceRequest(
            request_id="req",
            artifacts=tuple(
                RewardInferenceArtifact(
                    artifact_id=f"a{i}",
                    path=f"/tmp/a{i}.pt",
                    media_type="video",
                    policy_version=7,
                )
                for i in range(3)
            ),
            reward_name="reward",
            score_key="overall_reward",
            policy_version=7,
        )

        results = asyncio.run(runtime.score_batch(request))

        assert [result.artifact_id for result in results] == ["a0", "a1", "a2"]
        assert [result.selected_score for result in results] == pytest.approx([2.0, 2.0, 2.0])
        assert {result.reward_model_version for result in results} == {"fake-v1"}
        assert {result.policy_version for result in results} == {7}
        assert all(result.latency_ms is not None for result in results)
        assert all(result.queue_wait_ms is not None for result in results)
        assert all(result.inference_ms is not None for result in results)
        assert all(result.metadata["worker"]["worker_id"].startswith("reward-") for result in results)
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()
