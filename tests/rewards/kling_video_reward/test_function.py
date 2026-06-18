"""Tests for VideoReward as a thin inference-runtime adapter."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.validation import validate_reward_config
from vrl.rewards.functions.kling_video_reward import KlingVideoReward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardRollout, RewardTrajectory


class _FakeActorRuntime:
    num_workers = 1

    def __init__(self) -> None:
        self.requests = []

    async def map(self, shards):
        nested = []
        for shard in shards:
            self.requests.append(shard)
            out = []
            for artifact in shard.artifacts:
                scores = {"overall_reward": 1.5, "detail": 0.5}
                out.append(
                    RewardInferenceResult(
                        artifact_id=artifact.artifact_id,
                        scores=scores,
                        selected_score=shard.select_score(scores),
                        reward_name=shard.reward_name,
                        score_key=shard.score_key,
                        policy_version=artifact.policy_version,
                        reward_model_version="fake-test",
                        latency_ms=1.0,
                        worker_id="fake",
                    ),
                )
            nested.append(out)
        return nested

    async def shutdown(self) -> None:
        return None


class _EmptyActorRuntime:
    num_workers = 1

    async def map(self, shards):
        return [[] for _ in shards]

    async def shutdown(self) -> None:
        return None


def _rollout(output: torch.Tensor, *, policy_version: int = 3) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt="prompt", output=output),
        metadata={"policy_version": policy_version, "sample_ids": ["sample-a"]},
    )


def _video_reward_config(**video_kwargs: object):
    kwargs = {
        "execution": "pool",
        "reward_name": "kling_video_reward",
        "score_key": "overall_reward",
        "worker_config": {
            "reward_model_version": "unit-test",
        },
    }
    kwargs.update(video_kwargs)
    return OmegaConf.create(
        {
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {"kling_video_reward": kwargs},
            },
        },
    )


@pytest.mark.asyncio
async def test_video_reward_materializes_artifacts_and_returns_runtime_scores(
    tmp_path: Path,
) -> None:
    """Checks video reward materializes artifacts and returns runtime scores."""
    actor_runtime = _FakeActorRuntime()
    reward = KlingVideoReward(
        execution="pool",
        reward_name="kling_video_reward",
        score_key="overall_reward",
        media_type="video",
        # codec-independent wiring test: tensor avoids the imageio mp4 writer
        # (the production default is mp4; this test checks materialization wiring).
        artifact_format="tensor",
        artifact_dir=str(tmp_path / "artifacts"),
        debug_dir=str(tmp_path / "debug"),
        actor_runtime=actor_runtime,
    )

    scores = await reward.score_batch([_rollout(torch.ones(1, 2, 2, 2))])

    assert scores == pytest.approx([1.5])
    assert len(actor_runtime.requests) == 1
    request = actor_runtime.requests[0]
    assert request.reward_name == "kling_video_reward"
    assert request.artifacts[0].policy_version == 3
    assert Path(request.artifacts[0].path).exists()
    assert (tmp_path / "artifacts" / "manifest.jsonl").exists()
    assert (tmp_path / "debug" / "kling_video_reward_requests.jsonl").exists()
    assert (tmp_path / "debug" / "kling_video_reward_results.jsonl").exists()
    assert asdict(reward.last_results[0])["reward_model_version"] == "fake-test"


@pytest.mark.asyncio
async def test_video_reward_rejects_missing_runtime_results(tmp_path: Path) -> None:
    """Checks video reward rejects missing runtime results."""
    reward = KlingVideoReward(
        execution="pool",
        reward_name="kling_video_reward",
        score_key="overall_reward",
        artifact_format="tensor",  # codec-independent wiring test (no imageio dep)
        artifact_dir=str(tmp_path / "artifacts"),
        actor_runtime=_EmptyActorRuntime(),
    )

    with pytest.raises(RuntimeError, match="result/artifact mismatch"):
        await reward.score_batch([_rollout(torch.ones(1, 2, 2, 2))])


def test_video_reward_rejects_non_ray_runtime(tmp_path: Path) -> None:
    """Checks video reward rejects non Ray runtime."""
    with pytest.raises(ValueError, match="execution must be 'pool'"):
        KlingVideoReward(
            execution="inline",
            reward_name="kling_video_reward",
            score_key="overall_reward",
            artifact_dir=str(tmp_path),
            actor_runtime=_FakeActorRuntime(),
        )


def test_video_reward_config_accepts_ray_runtime() -> None:
    """Checks video reward config accepts Ray runtime."""
    validate_reward_config(_video_reward_config())


