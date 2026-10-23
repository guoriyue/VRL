"""Tests for model-backed RewardFunction over the in-process transport."""

from __future__ import annotations

import pytest
import torch

from vrl.rewards.base import (
    ModelRewardFunction,
)
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.runtime import InProcessRewardScorer
from vrl.rewards.types import RewardSample
from vrl.utils.stats import recording


class _FakeReward(ModelRewardFunction):
    """A model reward whose scorer each test injects."""

    model_factory = "unused:factory"
    name = "fake"
    default_score_key = "fake"


class _FakeTorchReward(TorchRewardModel):
    """Toy torch reward model: score = mean of the in-memory media tensor."""

    def _load_module(self) -> torch.nn.Module:
        self.loaded_marker = True
        return torch.nn.Identity()

    def score_media(self, *, media, prompt):
        return {"fake": float(media.float().mean().item())}


def _sample(
    output: torch.Tensor,
    *,
    sample_id: str = "sample-0",
    policy_version: int = 2,
) -> RewardSample:
    return RewardSample(
        prompt="p",
        output=output,
        sample_id=sample_id,
        metadata={"policy_version": policy_version},
    )


def _reward_function_in_process() -> ModelRewardFunction:
    return _FakeReward(
        scorer=InProcessRewardScorer(
            model=_FakeTorchReward({"device": "cpu"}),
        ),
    )


@pytest.mark.asyncio
async def test_reward_function_in_process_scores_without_disk() -> None:
    """Checks reward function scoring in-process without disk artifacts."""
    reward = _reward_function_in_process()
    report = await reward.score_batch(
        [
            _sample(torch.full((1, 3, 2, 2), 0.5)),
            _sample(torch.ones(1, 3, 2, 2), sample_id="sample-1"),
        ],
    )

    assert report.scores == pytest.approx([0.5, 1.0])
    assert report.components["fake"] == pytest.approx([0.5, 1.0])
    await reward.shutdown()


@pytest.mark.asyncio
async def test_missing_axis_fails_instead_of_silently_losing_observations() -> None:
    class InconsistentModel:
        def __call__(self, artifact):
            if artifact.sample_id == "sample-0":
                return {"fake": 1.0, "quality": 0.5}
            return {"fake": 2.0}

    reward = _FakeReward(
        scorer=InProcessRewardScorer(model=InconsistentModel()),
    )
    try:
        with pytest.raises(ValueError, match="inconsistent score axes"):
            await reward.score_batch(
                [
                    _sample(torch.zeros(3, 2, 2)),
                    _sample(torch.ones(3, 2, 2), sample_id="sample-1"),
                ]
            )
    finally:
        await reward.shutdown()


def test_lazy_reward_models_defer_module_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights land in the runtime's CuMem build frame, never in ``__init__``."""
    from vrl.rewards.models.aesthetic import AestheticRewardModel
    from vrl.rewards.models.motion_dynamics import MotionDynamicsModel
    from vrl.rewards.models.pickscore import PickScoreRewardModel
    from vrl.rewards.models.target_dino_similarity import TargetDinoSimilarityModel

    models = [
        AestheticRewardModel({"device": "cuda:7"}),
        PickScoreRewardModel({"device": "cuda:7"}),
        MotionDynamicsModel({"device": "cuda:7"}),
        TargetDinoSimilarityModel({"device": "cuda:7"}),
    ]
    for model in models:
        built_on: list[str] = []
        module = object()

        def build(*, owner=model, result=module, devices=built_on):
            devices.append(owner.device)
            return result

        monkeypatch.setattr(model, "_load_module", build)
        assert model._module is None  # construction allocated nothing

        model.prepare_for_inference()

        assert built_on == ["cuda:7"]
        assert model._module is module

        model.prepare_for_inference()
        assert built_on == ["cuda:7"]  # one build per runtime lifetime


def test_constructor_rejects_bad_score_keys_and_stray_model_knobs() -> None:
    """Construction-time checks shared by every ModelRewardFunction (base.py)."""

    class _Runtime:
        async def score_batch(self, request):
            return []

        async def shutdown(self) -> None:
            return None

    class _ClosedReward(_FakeReward):
        score_keys = ("dense", "recall")
        worker_config_only = True

    with pytest.raises(ValueError, match="score_key must be non-empty"):
        _FakeReward(score_key="", scorer=_Runtime())
    with pytest.raises(ValueError, match="empty component"):
        _FakeReward(score_key="a++b", scorer=_Runtime())
    with pytest.raises(ValueError, match=r"score_key must be one of \['dense', 'recall'\]"):
        _ClosedReward(score_key="fake", scorer=_Runtime())
    with pytest.raises(TypeError, match="unknown_knob"):
        _ClosedReward(score_key="dense", scorer=_Runtime(), unknown_knob=True)


@pytest.mark.asyncio
async def test_each_call_folds_its_result_timings_into_the_bound_accumulator() -> None:
    """Per call: inference adds up over results, a stage keeps its slowest result;
    calls then add up. An unbound call scores the same and records nothing."""

    stamps = (
        {"inference_ms": 3.0, "service_inference_wall_ms": 5.0, "http_roundtrip_ms": 20.0},
        {"inference_ms": 4.0, "service_inference_wall_ms": 2.0, "http_roundtrip_ms": 30.0},
    )

    class StampedScorer:
        async def score_batch(self, request):
            return [
                RewardInferenceResult(artifact.artifact_id, {"fake": 1.0}, timing_ms=stamp)
                for artifact, stamp in zip(request.artifacts, stamps, strict=True)
            ]

    reward = _FakeReward(scorer=StampedScorer())
    samples = [_sample(torch.zeros(3, 2, 2)), _sample(torch.ones(3, 2, 2), sample_id="sample-1")]

    assert (await reward.score_batch(samples)).scores == (1.0, 1.0)
    with recording() as stats:
        await reward.score_batch(samples)
        await reward.score_batch(samples)

    phases = stats.phase_seconds
    assert phases["reward.inference_s"] == pytest.approx(2 * 0.007)
    assert phases["reward.service_inference_wall_s"] == pytest.approx(2 * 0.005)
    assert phases["reward.transport_roundtrip_s"] == pytest.approx(2 * 0.030)
    # Measured on the driver side of every call.
    for phase in ("reward.latency_s", "reward.artifact_materialization_s"):
        assert phase in phases
    assert phases["reward.latency_s"] >= phases["reward.inference_wall_s"]
    # No result stamped these stages.
    assert "reward.artifact_validation_s" not in phases
