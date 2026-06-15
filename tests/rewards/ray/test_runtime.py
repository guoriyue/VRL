"""GPU-aware contract coverage for Ray-backed reward inference."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.ray import RayRewardRuntime

# Every test here spins up Ray (~seconds each) — slow by nature, run nightly not per-PR.
pytestmark = pytest.mark.slow_test


def build_tensor_mean_model(worker_config):
    """Test RewardModel factory: score = mean of the artifact tensor."""

    score_key = str(worker_config["score_key"])

    def _model(*, artifact, request):
        tensor = torch.load(Path(artifact.path), map_location="cpu").float()
        return {score_key: float(tensor.mean().item())}

    return _model


def build_constant_reward_model(worker_config):
    """Test RewardModel factory: return fixed scores independent of the artifact."""

    scores = {str(k): float(v) for k, v in dict(worker_config["scores"]).items()}

    def _model(*, artifact, request):
        return dict(scores)

    return _model


def _request(path: Path, *, score_key: str = "overall_reward") -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="gpu-runtime",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path=str(path),
                media_type="video",
                policy_version=17,
            ),
        ),
        reward_name="KlingTeam/VideoReward@main",
        score_key=score_key,
        policy_version=17,
    )


def test_ray_reward_runtime_uses_repo_owned_model_factory(tmp_path: Path) -> None:
    """Checks Ray reward runtime uses repo owned model factory."""
    ray = pytest.importorskip("ray")
    artifact = tmp_path / "artifact.pt"
    torch.save(torch.tensor([1.0, 3.0]), artifact)
    ray.shutdown()
    runtime = None
    try:
        runtime = RayRewardRuntime(
            {
                "execution": "pool",
                "worker_config": {
                    "model_factory": (
                        "tests.rewards.ray.test_runtime:build_tensor_mean_model"
                    ),
                    "score_key": "overall_reward",
                    "reward_model_version": "tensor-mean-v1",
                },
                "num_workers": 1,
                "cpus_per_worker": 0.5,
                "gpus_per_worker": 0.0,
            },
            ray_init_kwargs={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": 1,
                "log_to_driver": False,
            },
        )

        results = asyncio.run(runtime.score_batch(_request(artifact)))

        assert results[0].selected_score == pytest.approx(2.0)
        assert results[0].reward_model_version == "tensor-mean-v1"
        assert results[0].metadata["worker"]["worker_id"] == "reward-0"
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()


def test_ray_reward_runtime_fans_out_across_workers_with_timing() -> None:
    # Multi-worker fan-out: three artifacts scored across two workers must come
    # back in request order, with populated, non-negative timing breakdowns.
    """Checks Ray reward runtime fans out across workers with timing."""
    ray = pytest.importorskip("ray")
    ray.shutdown()
    runtime = None
    try:
        runtime = RayRewardRuntime(
            {
                "execution": "pool",
                "worker_config": {
                    "model_factory": (
                        "tests.rewards.ray.test_runtime:build_constant_reward_model"
                    ),
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
        for result in results:
            assert result.latency_ms is not None and result.latency_ms >= 0.0
            assert result.queue_wait_ms is not None and result.queue_wait_ms >= 0.0
            assert result.inference_ms is not None and result.inference_ms >= 0.0
            assert result.metadata["worker"]["worker_id"].startswith("reward-")
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()


@pytest.mark.gpu
def test_ray_reward_runtime_assigns_gpu_ids_for_tensor_model(tmp_path: Path) -> None:
    """Reward GPU actors schedule into the owner's placement group."""
    ray = pytest.importorskip("ray")
    from omegaconf import OmegaConf

    from vrl.ray.placement import GlobalRayPlacementOwner
    from vrl.ray.resources import resolve_distributed_resources

    artifact = tmp_path / "artifact.pt"
    torch.save(torch.ones(2, 2), artifact)
    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=2,
        num_gpus=1,
        log_to_driver=False,
    )
    resolved = resolve_distributed_resources(
        OmegaConf.create(
            {
                "distributed": {
                    "resources": {
                        "visible_devices": [0],
                        "trainer": {"num_gpus": 0},
                        "rollout": {"num_gpus": 0, "gpus_per_worker": 0, "num_workers": 1},
                        "reward": {
                            "devices": [0],
                            "num_gpus": 1,
                            "gpus_per_worker": 1,
                            "num_workers": 1,
                        },
                    },
                    "rollout": {},
                    "reward": {},
                },
            },
        ),
    )
    owner = GlobalRayPlacementOwner(resolved, rollout_cpus_per_worker=0.5)
    owner.create()
    runtime = None
    try:
        runtime = RayRewardRuntime(
            {
                "execution": "pool",
                "worker_config": {
                    "model_factory": (
                        "tests.rewards.ray.test_runtime:build_tensor_mean_model"
                    ),
                    "score_key": "overall_reward",
                    "reward_model_version": "gpu-runtime-v1",
                },
                "num_workers": 1,
                "cpus_per_worker": 0.5,
                "gpus_per_worker": 1.0,
                "placement": owner.reward_placement,
            },
            init_ray=False,
        )

        results = asyncio.run(runtime.score_batch(_request(artifact)))

        assert results[0].selected_score == pytest.approx(1.0)
        assert results[0].metadata["gpu_ids"] == [0]
        assert results[0].worker_id == "reward-0"
        assert results[0].policy_version == 17
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        owner.shutdown()
        ray.shutdown()
