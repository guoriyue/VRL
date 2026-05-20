"""GPU smoke coverage for Ray-backed reward inference."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.ray.launcher import build_reward_ray_runtime


def imported_score_fn(*, artifact_path: str, worker_config: dict[str, Any], **_: Any) -> dict[str, float]:
    tensor = torch.load(Path(artifact_path), map_location="cpu").float()
    return {str(worker_config["score_key"]): float(tensor.mean().item())}


def _request(path: Path, *, score_key: str = "overall_reward") -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="gpu-smoke",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path=str(path),
                media_type="video",
                policy_version=17,
            ),
        ),
        reward_name="cosmos_reason1",
        score_key=score_key,
        policy_version=17,
    )


def test_import_path_reward_scorer_is_repo_owned_runtime_contract(tmp_path: Path) -> None:
    ray = pytest.importorskip("ray")
    artifact = tmp_path / "artifact.pt"
    torch.save(torch.tensor([1.0, 3.0]), artifact)
    ray.shutdown()
    runtime = None
    try:
        runtime = build_reward_ray_runtime(
            {
                "inference_runtime": "ray",
                "worker_config": {
                    "scorer": "import_path",
                    "import_path": (
                        "tests.rewards.test_ray_reward_gpu_smoke:imported_score_fn"
                    ),
                    "score_key": "overall_reward",
                    "reward_model_version": "import-path-v1",
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
        assert results[0].reward_model_version == "import-path-v1"
        assert results[0].metadata["worker"]["worker_id"] == "reward-0"
    finally:
        if runtime is not None:
            asyncio.run(runtime.shutdown())
        ray.shutdown()

def test_ray_reward_runtime_assigns_gpu_ids_for_tensor_scorer(tmp_path: Path) -> None:
    ray = pytest.importorskip("ray")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU for Ray reward GPU placement smoke")

    artifact = tmp_path / "artifact.pt"
    torch.save(torch.ones(2, 2), artifact)
    ray.shutdown()
    runtime = None
    try:
        runtime = build_reward_ray_runtime(
            {
                "inference_runtime": "ray",
                "worker_config": {
                    "scorer": "tensor_mean",
                    "reward_model_version": "gpu-smoke-v1",
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
