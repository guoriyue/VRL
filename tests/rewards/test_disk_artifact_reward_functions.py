"""Disk-artifact reward functions as thin inference-runtime adapters.

HPSv3, VideoScore2, Kling VideoReward and UnifiedReward-2.0 are all
zero-method subclasses of ``DiskArtifactRewardFunction``: they only pin a
model factory, a debug basename and their defaults. Every behavior asserted
here -- artifact materialization, ``score_key`` selection, the missing-key and
result/artifact-mismatch failures, and post-scoring artifact release -- lives
in ``vrl/rewards/base.py`` and ``vrl/rewards/inference.py``. One parametrized
module pins that shared contract for all four wrappers, so a wrapper that
starts overriding the adapter shows up as a red test rather than as an
untested divergence.

Model-side parsing (VideoScore2 rubric text, UnifiedReward axis scales) and
the debug-row schema stay in their own modules; they are not shared behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import RewardConfig
from vrl.rewards.functions.hpsv3 import HPSv3Reward
from vrl.rewards.functions.kling_video_reward import KlingVideoReward
from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward
from vrl.rewards.functions.videoscore2 import VideoScore2Reward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardSample


@dataclass(frozen=True)
class _Case:
    reward_cls: type
    reward_name: str
    worker_config: dict[str, str]
    default_score_key: str
    alternate_score_key: str
    fake_scores: dict[str, float]


_CASES = [
    _Case(
        HPSv3Reward,
        "hpsv3",
        {"reward_model_name": "MizzenAI/HPSv3@main"},
        "top_frame_mean",
        "frame_mean",
        {"top_frame_mean": 9.5, "frame_mean": 7.25, "frame_min": 1.5},
    ),
    _Case(
        VideoScore2Reward,
        "videoscore2",
        {"reward_model_name": "TIGER-Lab/VideoScore2@main"},
        "physical_common_sense",
        "visual_quality",
        {
            "visual_quality": 4.0,
            "text_alignment": 2.5,
            "physical_common_sense": 3.25,
            "overall": 3.25,
        },
    ),
    _Case(
        KlingVideoReward,
        "kling_video_reward",
        {"reward_model_version": "unit-test"},
        "overall_reward",
        "detail",
        {"overall_reward": 1.5, "detail": 0.5},
    ),
    _Case(
        UnifiedRewardVideoReward,
        "unified_reward_video",
        {"reward_model_name": "CodeGoat24/UnifiedReward-2.0-qwen-7b@main"},
        "overall",
        "physics",
        {"alignment": 4.0, "physics": 2.0, "style": 3.0, "overall": 3.0},
    ),
]
_CASE_IDS = [case.reward_name for case in _CASES]


class _FakeRuntime:
    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = False

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
        output=torch.ones(1, 2, 2, 2),
        sample_id="sample-a",
        metadata={"policy_version": 7},
    )


def _build_reward(
    case: _Case,
    tmp_path: Path,
    *,
    score_key: str,
    retain_artifacts: bool = True,
    scorer: _FakeRuntime | None = None,
):
    return case.reward_cls(
        reward_name=case.reward_name,
        score_key=score_key,
        media_type="video",
        # tensor avoids the imageio mp4 writer; this checks materialization wiring.
        artifact_format="tensor",
        artifact_dir=str(tmp_path / "artifacts"),
        debug_dir=str(tmp_path / "debug"),
        retain_artifacts=retain_artifacts,
        scorer=scorer if scorer is not None else _FakeRuntime(case.fake_scores),
    )


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
@pytest.mark.asyncio
async def test_materializes_artifacts_and_selects_the_score_key(
    case: _Case, tmp_path: Path
) -> None:
    """Default and alternate score keys select their axis; debug logs every public key."""
    reward = _build_reward(case, tmp_path, score_key=case.default_score_key)
    output = await reward.score_batch([_sample()])

    assert output.scores == pytest.approx([case.fake_scores[case.default_score_key]])
    request = reward.scorer.requests[0]
    assert len(request.artifacts) == 1
    assert Path(request.artifacts[0].path).exists()
    assert (tmp_path / "debug" / f"{case.reward_name}_requests.jsonl").exists()
    results_path = tmp_path / "debug" / f"{case.reward_name}_results.jsonl"
    body = results_path.read_text(encoding="utf-8")
    assert all(key in body for key in case.fake_scores)
    first_row = json.loads(body.splitlines()[0])
    assert first_row["reward_model_version"] == "fake-test"

    alternate = _build_reward(case, tmp_path / "alt", score_key=case.alternate_score_key)
    output = await alternate.score_batch([_sample()])
    assert output.scores == pytest.approx([case.fake_scores[case.alternate_score_key]])


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
@pytest.mark.asyncio
async def test_missing_score_key_fails_fast(case: _Case, tmp_path: Path) -> None:
    """An unknown score_key raises rather than silently scoring zero."""
    reward = _build_reward(case, tmp_path, score_key="not_a_real_axis")
    with pytest.raises(KeyError, match="missing score keys"):
        await reward.score_batch([_sample()])


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
@pytest.mark.asyncio
async def test_releases_artifacts_after_success_by_default(case: _Case, tmp_path: Path) -> None:
    """Without retain_artifacts a scored run leaves nothing on disk."""
    reward = _build_reward(
        case, tmp_path, score_key=case.default_score_key, retain_artifacts=False
    )

    await reward.score_batch([_sample()])

    assert not Path(reward.scorer.requests[0].artifacts[0].path).exists()


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
@pytest.mark.asyncio
async def test_unmatched_runtime_results_fail_without_leaking_artifacts(
    case: _Case, tmp_path: Path
) -> None:
    """A runtime that drops results raises, and the failed run still releases its artifacts."""
    reward = _build_reward(
        case,
        tmp_path,
        score_key=case.default_score_key,
        retain_artifacts=False,
        scorer=_EmptyRuntime(case.fake_scores),
    )

    with pytest.raises(RuntimeError, match="result/artifact mismatch"):
        await reward.score_batch([_sample()])
    assert not list((tmp_path / "artifacts").glob("*.pt"))


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_config_passes_the_shipped_component_shape_through_unvalidated(case: _Case) -> None:
    """``RewardConfig.kwargs`` is open by design (vrl/config/schema.py): a reward's kwargs
    contract belongs to the reward class at construction, not to the config layer. The
    nested ``worker_config`` must survive validation byte-for-byte; tightening this schema
    (a sub-model, ``extra="forbid"``) would silently break every reward's kwargs passthrough
    at once.
    """
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {case.reward_name: 1.0},
                "kwargs": {
                    case.reward_name: {
                        "reward_name": case.reward_name,
                        "score_key": case.default_score_key,
                        "worker_config": dict(case.worker_config),
                    },
                },
            },
        },
    )
    parsed = RewardConfig.from_cfg(cfg)

    assert parsed.kwargs[case.reward_name]["worker_config"] == dict(case.worker_config)
