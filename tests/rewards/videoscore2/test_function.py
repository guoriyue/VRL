"""VideoScore2 reward as a disk-artifact inference-runtime adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.validation import validate_reward_config
from vrl.rewards.functions.videoscore2 import VideoScore2Reward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardRollout, RewardTrajectory

_FAKE_SCORES = {
    "visual_quality": 4.0,
    "text_alignment": 2.5,
    "physical_common_sense": 3.25,
    "overall": 3.25,
}


class _FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    async def score_batch(self, request):
        self.requests.append(request)
        out = []
        for artifact in request.artifacts:
            # select_score raises KeyError here when score_key names a key
            # the model never returns — the fail-fast under test.
            out.append(
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores=dict(_FAKE_SCORES),
                    selected_score=request.select_score(_FAKE_SCORES),
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


def _rollout(output: torch.Tensor, *, policy_version: int = 7) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt="a dancer spins, skirt billowing", output=output),
        metadata={"policy_version": policy_version, "sample_ids": ["sample-a"]},
    )


def _build_reward(tmp_path: Path, *, score_key: str = "physical_common_sense", **kwargs):
    return VideoScore2Reward(
        reward_name="videoscore2",
        score_key=score_key,
        media_type="video",
        # tensor avoids the imageio mp4 writer; this checks materialization wiring.
        artifact_format="tensor",
        artifact_dir=str(tmp_path / "artifacts"),
        debug_dir=str(tmp_path / "debug"),
        runtime=_FakeRuntime(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_materializes_artifacts_and_selects_score_key(tmp_path: Path) -> None:
    """Default score_key picks physical_common_sense and debug logs the public keys."""
    reward = _build_reward(tmp_path)
    scores = await reward.score_batch([_rollout(torch.ones(1, 2, 2, 2))])

    assert scores == pytest.approx([3.25])
    request = reward.runtime.requests[0]
    assert request.reward_name == "videoscore2"
    assert request.score_key == "physical_common_sense"
    assert request.artifacts[0].policy_version == 7
    assert Path(request.artifacts[0].path).exists()
    results_file = tmp_path / "debug" / "videoscore2_results.jsonl"
    assert (tmp_path / "debug" / "videoscore2_requests.jsonl").exists()
    body = results_file.read_text(encoding="utf-8")
    for key in ("visual_quality", "text_alignment", "physical_common_sense", "overall"):
        assert key in body


@pytest.mark.asyncio
async def test_alternate_score_key_selects_visual_quality(tmp_path: Path) -> None:
    """A different score_key selects the matching public axis."""
    reward = _build_reward(tmp_path, score_key="visual_quality")
    scores = await reward.score_batch([_rollout(torch.ones(1, 2, 2, 2))])
    assert scores == pytest.approx([4.0])


@pytest.mark.asyncio
async def test_missing_score_key_fails_fast(tmp_path: Path) -> None:
    """An unknown score_key raises rather than silently scoring zero."""
    reward = _build_reward(tmp_path, score_key="not_a_real_axis")
    with pytest.raises(KeyError, match="missing score keys"):
        await reward.score_batch([_rollout(torch.ones(1, 2, 2, 2))])


def test_rejects_removed_pool_execution(tmp_path: Path) -> None:
    """The removed pool execution fails loud with the migration hint."""
    with pytest.raises(ValueError, match="sleep_offload"):
        VideoScore2Reward(
            execution="pool",
            reward_name="videoscore2",
            score_key="physical_common_sense",
            artifact_dir=str(tmp_path),
        )


def test_config_accepts_videoscore2() -> None:
    """The shipped component shape validates."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"videoscore2": 1.0},
                "kwargs": {
                    "videoscore2": {
                        "execution": "pool",
                        "reward_name": "videoscore2",
                        "score_key": "physical_common_sense",
                        "worker_config": {"reward_model_name": "TIGER-Lab/VideoScore2@main"},
                    },
                },
            },
        },
    )
    validate_reward_config(cfg)
