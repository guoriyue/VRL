"""HPSv3 reward as a disk-artifact inference-runtime adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.validation import validate_reward_config
from vrl.rewards.functions.hpsv3 import HPSv3Reward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardSample

_FAKE_SCORES = {
    "top_frame_mean": 9.5,
    "frame_mean": 7.25,
    "frame_min": 1.5,
}


class _FakeRuntime:
    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = False

    def __init__(self) -> None:
        self.requests = []

    async def score_batch(self, request):
        self.requests.append(request)
        out = []
        for artifact in request.artifacts:
            out.append(
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores=dict(_FAKE_SCORES),
                    reward_model_version="fake-test",
                    timing_ms={"inference_ms": 1.0},
                ),
            )
        return out

    async def shutdown(self) -> None:
        return None


def _sample(output: torch.Tensor, *, policy_version: int = 7) -> RewardSample:
    return RewardSample(
        prompt="a red fox curled on mossy stones",
        output=output,
        sample_id="sample-a",
        metadata={"policy_version": policy_version},
    )


def _build_reward(tmp_path: Path, *, score_key: str = "top_frame_mean", **kwargs):
    return HPSv3Reward(
        reward_name="hpsv3",
        score_key=score_key,
        media_type="video",
        # tensor avoids the imageio mp4 writer; this checks materialization wiring.
        artifact_format="tensor",
        artifact_dir=str(tmp_path / "artifacts"),
        debug_dir=str(tmp_path / "debug"),
        retain_artifacts=True,
        scorer=_FakeRuntime(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_materializes_artifacts_and_selects_score_key(tmp_path: Path) -> None:
    """Default score_key picks top_frame_mean and debug logs the public keys."""
    reward = _build_reward(tmp_path)
    output = await reward.score_batch([_sample(torch.ones(1, 2, 2, 2))])

    assert output.scores == pytest.approx([9.5])
    request = reward.scorer.requests[0]
    assert len(request.artifacts) == 1
    assert Path(request.artifacts[0].path).exists()
    results_file = tmp_path / "debug" / "hpsv3_results.jsonl"
    assert (tmp_path / "debug" / "hpsv3_requests.jsonl").exists()
    body = results_file.read_text(encoding="utf-8")
    for key in ("top_frame_mean", "frame_mean", "frame_min"):
        assert key in body


@pytest.mark.asyncio
async def test_alternate_score_key_selects_frame_mean(tmp_path: Path) -> None:
    """A different score_key selects the matching public axis."""
    reward = _build_reward(tmp_path, score_key="frame_mean")
    output = await reward.score_batch([_sample(torch.ones(1, 2, 2, 2))])
    assert output.scores == pytest.approx([7.25])


@pytest.mark.asyncio
async def test_missing_score_key_fails_fast(tmp_path: Path) -> None:
    """An unknown score_key raises rather than silently scoring zero."""
    reward = _build_reward(tmp_path, score_key="not_a_real_axis")
    with pytest.raises(KeyError, match="missing score keys"):
        await reward.score_batch([_sample(torch.ones(1, 2, 2, 2))])


def test_config_accepts_hpsv3() -> None:
    """The shipped component shape validates."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"hpsv3": 1.0},
                "kwargs": {
                    "hpsv3": {
                        "reward_name": "hpsv3",
                        "score_key": "top_frame_mean",
                        "worker_config": {"reward_model_name": "MizzenAI/HPSv3@main"},
                    },
                },
            },
        },
    )
    validate_reward_config(cfg)
