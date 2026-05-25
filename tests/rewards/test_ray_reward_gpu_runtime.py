"""GPU-aware contract coverage for Ray-backed reward inference."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.ray import RayRewardRuntime


def build_tensor_mean_model(worker_config):
    """Test RewardModel factory: score = mean of the artifact tensor."""

    score_key = str(worker_config["score_key"])

    def _model(*, artifact, request):
        tensor = torch.load(Path(artifact.path), map_location="cpu").float()
        return {score_key: float(tensor.mean().item())}

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
    ray = pytest.importorskip("ray")
    artifact = tmp_path / "artifact.pt"
    torch.save(torch.tensor([1.0, 3.0]), artifact)
    ray.shutdown()
    runtime = None
    try:
        runtime = RayRewardRuntime(
            {
                "inference_runtime": "ray",
                "worker_config": {
                    "model_factory": (
                        "tests.rewards.test_ray_reward_gpu_runtime:build_tensor_mean_model"
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


def test_ray_reward_runtime_assigns_gpu_ids_for_tensor_model(tmp_path: Path) -> None:
    ray = pytest.importorskip("ray")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU for Ray reward GPU placement contract")

    artifact = tmp_path / "artifact.pt"
    torch.save(torch.ones(2, 2), artifact)
    ray.shutdown()
    runtime = None
    try:
        runtime = RayRewardRuntime(
            {
                "inference_runtime": "ray",
                "worker_config": {
                    "model_factory": (
                        "tests.rewards.test_ray_reward_gpu_runtime:build_tensor_mean_model"
                    ),
                    "score_key": "overall_reward",
                    "reward_model_version": "gpu-runtime-v1",
                },
                "num_workers": 1,
                "cpus_per_worker": 0.5,
                "gpus_per_worker": 1.0,
                "expected_gpu_ids": (0,),
                "placement_strategy": "STRICT_PACK",
            },
            ray_init_kwargs={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": 1,
                "num_gpus": 1,
                "log_to_driver": False,
            },
        )

        results = asyncio.run(runtime.score_batch(_request(artifact)))

        assert results[0].selected_score == pytest.approx(1.0)
        assert results[0].metadata["gpu_ids"] == [0]
        assert results[0].worker_id == "reward-0"
        assert results[0].policy_version == 17
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()
