"""Model-backed reward functions as thin in-memory inference-runtime adapters.

HPSv3, Kling VideoReward and UnifiedReward-2.0 are all zero-method subclasses of
``ModelRewardFunction``: they only pin a model factory, a debug basename and their
defaults. Every behavior asserted here -- media forwarding, ``score_key``
selection, the missing-key and result/artifact-mismatch failures -- lives in
``vrl/rewards/base.py`` and ``vrl/rewards/inference.py``, so one wrapper
(Kling) stands in for all of them.

Model-side parsing (UnifiedReward axis scales) and
the debug-row schema stay in their own modules; they are not shared behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import RewardConfig
from vrl.rewards.functions.kling_video_reward import KlingVideoReward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardSample

_FAKE_SCORES = {"overall_reward": 1.5, "detail": 0.5}


class _FakeRuntime:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = dict(scores)
        self.requests: list[Any] = []

    async def score_batch(self, request):
        self.requests.append(request)
        return [
            RewardInferenceResult(
                artifact_id=artifact.artifact_id,
                scores=dict(self.scores),
                reward_model_version="fake-test",
                timing_ms={"inference_ms": 1.0},
            )
            for artifact in request.artifacts
        ]

    async def shutdown(self) -> None:
        return None


class _EmptyRuntime(_FakeRuntime):
    """Returns no results at all, so every artifact is unaccounted for."""

    async def score_batch(self, request):
        self.requests.append(request)
        return []


def _sample() -> RewardSample:
    return RewardSample(
        prompt="a red fox curled on mossy stones",
        output=torch.ones(3, 2, 16, 16),
        sample_id="sample-a",
        metadata={"policy_version": 7},
    )


def _build_reward(tmp_path: Path, *, score_key: str, scorer: _FakeRuntime | None = None):
    return KlingVideoReward(
        reward_name="kling_video_reward",
        score_key=score_key,
        debug_dir=str(tmp_path / "debug"),
        scorer=scorer if scorer is not None else _FakeRuntime(_FAKE_SCORES),
    )


@pytest.mark.asyncio
async def test_forwards_media_and_selects_the_score_key(tmp_path: Path) -> None:
    """Default and alternate score keys select their axis; debug rows log the score key,
    every public score, the model version and the scorer's per-artifact timing."""
    reward = _build_reward(tmp_path, score_key="overall_reward")
    sample = _sample()
    output = await reward.score_batch([sample])

    assert output.scores == pytest.approx([_FAKE_SCORES["overall_reward"]])
    request = reward.scorer.requests[0]
    assert len(request.artifacts) == 1
    assert request.artifacts[0].path == ""
    assert request.artifacts[0].media is sample.output
    assert not list(tmp_path.rglob("*.mp4"))
    assert not list(tmp_path.rglob("*.pt"))
    requests_path = tmp_path / "debug" / "kling_video_reward_requests.jsonl"
    request_row = json.loads(requests_path.read_text(encoding="utf-8").splitlines()[0])
    assert request_row["score_key"] == "overall_reward"
    results_path = tmp_path / "debug" / "kling_video_reward_results.jsonl"
    body = results_path.read_text(encoding="utf-8")
    assert all(key in body for key in _FAKE_SCORES)
    first_row = json.loads(body.splitlines()[0])
    assert first_row["reward_model_version"] == "fake-test"
    assert first_row["timing_ms"]["inference_ms"] == pytest.approx(1.0)

    alternate = _build_reward(tmp_path / "alt", score_key="detail")
    output = await alternate.score_batch([_sample()])
    assert output.scores == pytest.approx([_FAKE_SCORES["detail"]])


@pytest.mark.asyncio
async def test_missing_score_key_fails_fast(tmp_path: Path) -> None:
    """An unknown score_key raises rather than silently scoring zero."""
    reward = _build_reward(tmp_path, score_key="not_a_real_axis")
    with pytest.raises(KeyError, match="missing score keys"):
        await reward.score_batch([_sample()])


@pytest.mark.asyncio
async def test_unmatched_runtime_results_fail(tmp_path: Path) -> None:
    """A runtime that drops results raises instead of scoring a partial batch."""
    reward = _build_reward(
        tmp_path, score_key="overall_reward", scorer=_EmptyRuntime(_FAKE_SCORES)
    )

    with pytest.raises(RuntimeError, match="result/artifact mismatch"):
        await reward.score_batch([_sample()])


def test_config_passes_the_shipped_component_shape_through_unvalidated() -> None:
    """``RewardConfig.kwargs`` is open by design (vrl/config/schema.py): a reward's kwargs
    contract belongs to the reward class at construction, not to the config layer. The
    nested ``worker_config`` must survive validation byte-for-byte; tightening this schema
    (a sub-model, ``extra="forbid"``) would silently break every reward's kwargs passthrough
    at once.
    """
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
    parsed = RewardConfig.from_cfg(cfg)

    assert parsed.kwargs["hpsv3"]["worker_config"] == {"reward_model_name": "MizzenAI/HPSv3@main"}


def test_as_media_normalizes_quantized_pt_files_to_unit_floats(tmp_path: Path) -> None:
    """Every reward model reads unit-range floats; the file may hold the wire's uint8."""
    from vrl.rewards.inference import RewardInferenceArtifact

    frames = torch.tensor([[[0, 128], [255, 64]]], dtype=torch.uint8)
    quantized = tmp_path / "q.pt"
    torch.save(frames, quantized)
    unit = tmp_path / "u.pt"
    torch.save(frames.float() / 255.0, unit)

    def _artifact(path: Path) -> RewardInferenceArtifact:
        return RewardInferenceArtifact(artifact_id="a", sample_id="s", path=str(path))

    loaded = _artifact(quantized).as_media()
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, frames.float() / 255.0)
    assert torch.equal(_artifact(unit).as_media(), loaded)
    in_memory = RewardInferenceArtifact(artifact_id="a", sample_id="s", path="", media=frames)
    assert in_memory.as_media() is frames
