"""Reward runtime resource and lifecycle tests."""

from __future__ import annotations

import asyncio

import pytest
from omegaconf import OmegaConf

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.ray import RayRewardRuntime
from vrl.scripts.common.factory import build_reward_from_cfg

# Every test here spins up Ray (~seconds each) — slow by nature, run nightly not per-PR.
pytestmark = pytest.mark.slow_test

_CONSTANT_MODEL_FACTORY = (
    "tests.rewards.ray.test_resource_lifecycle:build_constant_reward_model"
)


def build_constant_reward_model(worker_config):
    """Test RewardModel factory: return fixed scores from worker_config."""

    scores = {str(k): float(v) for k, v in dict(worker_config["scores"]).items()}

    def _model(*, artifact, request):
        return dict(scores)

    return _model


class _EchoWorker:
    """Minimal RayActorMethodRuntime worker for owner-placement tests."""

    def __init__(self, worker_id, worker_config):
        self.worker_id = str(worker_id)

    def worker_metadata(self):
        return {"worker_id": self.worker_id, "node_ip": "local", "gpu_ids": []}

    def echo(self, payload):
        return (self.worker_id, payload)


def _cpu_owner(ray):
    from omegaconf import OmegaConf

    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.ray.resources import resolve_distributed_resources

    resolved = resolve_distributed_resources(
        OmegaConf.create(
            {
                "distributed": {
                    "resources": {
                        "visible_devices": [],
                        "trainer": {"num_gpus": 0},
                        "rollout": {"num_gpus": 0, "gpus_per_worker": 0, "num_workers": 1},
                    },
                    "rollout": {},
                    "reward": {},
                },
            },
        ),
    )
    owner = GlobalRayPlacementOwner(resolved, rollout_cpus_per_worker=0.5)
    owner.create()
    return owner


def test_actor_method_runtime_owner_placement_survives_release() -> None:
    """release_after_call drops actors but never the owner-managed PG."""
    ray = pytest.importorskip("ray")
    from vrl.ray.runtime import RayActorMethodRuntime

    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
    owner = _cpu_owner(ray)
    placement = owner.rollout_placement
    runtime = RayActorMethodRuntime(
        worker_cls=_EchoWorker,
        worker_config={},
        method_name="echo",
        worker_id_prefix="reward",
        num_workers=1,
        cpus_per_worker=0.5,
        gpus_per_worker=0.0,
        init_ray=False,
        release_after_call=True,
        placement_group=placement.placement_group,
        bundle_indices=placement.bundle_indices,
        validate_role="reward",
    )
    try:
        out = asyncio.run(runtime.map(["ping"]))
        assert out == [("reward-0", "ping")]
        # Actors released, but the owner-managed PG is untouched.
        assert runtime._actor_group is None
        assert owner._placement_group is not None
        # ...and a second call re-acquires actors into the same owner-managed PG.
        out2 = asyncio.run(runtime.map(["pong"]))
        assert out2 == [("reward-0", "pong")]
        assert owner._placement_group is not None
    finally:
        asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()


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
    """Checks reward runtime releases actors after score when configured."""
    ray = pytest.importorskip("ray")
    ray.shutdown()
    runtime = None
    try:
        runtime = RayRewardRuntime(
            {
                "execution": "pool",
                "worker_config": {
                    "model_factory": _CONSTANT_MODEL_FACTORY,
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
        assert runtime._actor._actor_group is None

        results = asyncio.run(runtime.score_batch(_request()))

        assert results[0].selected_score == pytest.approx(3.0)
        assert runtime._actor._actor_group is None
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()


def test_reward_config_receives_resolved_resource_plan() -> None:
    """Checks reward config receives resolved resource plan."""
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 0},
                    "rollout": {"devices": [0], "gpus_per_worker": 1, "num_workers": 1},
                    "reward": {
                        "num_gpus": 1,
                        "gpus_per_worker": 1,
                        "num_workers": 1,
                        "share_with_rollout": True,
                    },
                },
                # release_after_score derives to true for a shared reward pool.
                "reward": {
                    "cpus_per_worker": 0.5,
                    "max_inflight_batches": 1,
                    "placement_strategy": "STRICT_PACK",
                },
            },
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "KlingTeam/VideoReward@main",
                        "score_key": "overall_reward",
                        "worker_config": {
                            "model_factory": _CONSTANT_MODEL_FACTORY,
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
                {"kling_video_reward": 1.0},
                {
                    "kling_video_reward": {
                        "execution": "pool",
                        "reward_name": "KlingTeam/VideoReward@main",
                        "score_key": "overall_reward",
                        "worker_config": {
                            "model_factory": _CONSTANT_MODEL_FACTORY,
                            "scores": {"overall_reward": 1.0},
                            "reward_model_version": "resource-v1",
                        },
                    },
                },
            ),
        },
        device="cpu",
    )

    kling_reward = reward_fn.rewards[0][2]
    actor_runtime = kling_reward._actor_runtime
    assert actor_runtime.num_workers == 1
    assert actor_runtime.gpus_per_worker == 1.0
    assert actor_runtime.expected_gpu_ids == (0,)
    assert actor_runtime.release_after_call is True
