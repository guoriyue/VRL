"""Tests for VideoReward as a thin inference-runtime adapter."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.validation import validate_reward_config
from vrl.rewards.inference import RewardInferenceResult, select_score
from vrl.rewards.types import RewardRollout, RewardTrajectory
from vrl.rewards.video_reward import VideoReward


class _FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    async def score_batch(self, request):
        self.requests.append(request)
        out = []
        for artifact in request.artifacts:
            scores = {"overall_reward": 1.5, "detail": 0.5}
            out.append(
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores=scores,
                    selected_score=select_score(scores, request.score_key),
                    reward_name=request.reward_name,
                    score_key=request.score_key,
                    policy_version=artifact.policy_version,
                    reward_model_version="fake-test",
                    latency_ms=1.0,
                    worker_id="fake",
                ),
            )
        return out

    async def shutdown(self) -> None:
        return None


class _EmptyRuntime:
    async def score_batch(self, request):
        del request
        return []

    async def shutdown(self) -> None:
        return None


def _rollout(output: torch.Tensor, *, policy_version: int = 3) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt="prompt", seed=0, steps=[], output=output),
        metadata={"policy_version": policy_version, "sample_ids": ["sample-a"]},
    )


def _video_reward_config(**video_kwargs: object):
    kwargs = {
        "inference_runtime": "ray",
        "reward_name": "dance_grpo",
        "score_key": "overall_reward",
    }
    kwargs.update(video_kwargs)
    return OmegaConf.create(
        {
            "reward": {
                "components": {"video_reward": 1.0},
                "kwargs": {"video_reward": kwargs},
            },
        },
    )


@pytest.mark.asyncio
async def test_video_reward_materializes_artifacts_and_returns_runtime_scores(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    reward = VideoReward(
        inference_runtime="ray",
        reward_name="dance_grpo",
        score_key="overall_reward",
        media_type="video",
        artifact_dir=str(tmp_path / "artifacts"),
        debug_dir=str(tmp_path / "debug"),
        runtime=runtime,
    )

    scores = await reward.score_batch([_rollout(torch.ones(1, 2, 2, 2))])

    assert scores == pytest.approx([1.5])
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.reward_name == "dance_grpo"
    assert request.artifacts[0].policy_version == 3
    assert Path(request.artifacts[0].path).exists()
    assert (tmp_path / "artifacts" / "manifest.jsonl").exists()
    assert (tmp_path / "debug" / "video_reward_requests.jsonl").exists()
    assert (tmp_path / "debug" / "video_reward_results.jsonl").exists()
    assert asdict(reward.last_results[0])["reward_model_version"] == "fake-test"


@pytest.mark.asyncio
async def test_video_reward_rejects_missing_runtime_results(tmp_path: Path) -> None:
    reward = VideoReward(
        inference_runtime="ray",
        reward_name="dance_grpo",
        score_key="overall_reward",
        artifact_dir=str(tmp_path / "artifacts"),
        runtime=_EmptyRuntime(),
    )

    with pytest.raises(RuntimeError, match="result/artifact mismatch"):
        await reward.score_batch([_rollout(torch.ones(1, 2, 2, 2))])


def test_video_reward_rejects_legacy_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend is no longer supported"):
        VideoReward(
            backend="stub",
            inference_runtime="ray",
            reward_name="dance_grpo",
            score_key="overall_reward",
            artifact_dir=str(tmp_path),
            runtime=_FakeRuntime(),
        )


def test_video_reward_config_rejects_legacy_backend() -> None:
    cfg = _video_reward_config(backend="stub")

    with pytest.raises(ValueError, match=r"video_reward\.backend is no longer supported"):
        validate_reward_config(cfg)


def test_video_reward_config_rejects_removed_endpoint_fields() -> None:
    cfg = _video_reward_config(enqueue_url="/removed")

    with pytest.raises(ValueError, match="external reward endpoint fields"):
        validate_reward_config(cfg)


def test_video_reward_config_accepts_ray_runtime() -> None:
    validate_reward_config(_video_reward_config())
