"""Tests for model-backed RewardFunction over the in-process transport."""

from __future__ import annotations

import pytest
import torch

from vrl.rewards.base import (
    DiskArtifactRewardFunction,
    InferenceRewardFunction,
    RewardCleanupError,
    RewardFunction,
)
from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.runtime import InProcessRewardInferenceRuntime
from vrl.rewards.types import RewardSample
from vrl.utils.config import import_from_path


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


def _reward_function_in_process() -> InferenceRewardFunction:
    return InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        inference_runtime=InProcessRewardInferenceRuntime(
            model=_FakeTorchReward({"device": "cpu"}),
        ),
        artifact_builder=lambda samples: InferenceRewardFunction.build_inmemory_artifacts(
            samples,
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
    await reward.shutdown()


@pytest.mark.asyncio
async def test_reward_function_in_process_single_score() -> None:
    """Checks a single in-process reward score."""
    reward = _reward_function_in_process()
    assert await reward.score(_sample(torch.zeros(1, 3, 2, 2))) == pytest.approx(0.0)


def test_pickscore_reward_model_constructs_lazily() -> None:
    """Checks pickscore reward model constructs lazily."""
    from vrl.rewards.models.pickscore import PickScoreRewardModel

    model = PickScoreRewardModel(
        {"device": "cpu", "model_name": "x", "processor_name": "y"},
    )
    assert model.model_name == "x"
    assert model.processor_name == "y"
    assert model._module is None  # no heavy load at construction


@pytest.mark.parametrize(
    ("factory_path", "expected_type"),
    [
        (
            "vrl.rewards.models.aesthetic:AestheticRewardModel",
            "AestheticRewardModel",
        ),
        (
            "vrl.rewards.models.pickscore:PickScoreRewardModel",
            "PickScoreRewardModel",
        ),
    ],
)
def test_reward_model_factory_class_paths_construct(
    factory_path: str,
    expected_type: str,
) -> None:
    model = import_from_path(factory_path)({"device": "cpu"})

    assert type(model).__name__ == expected_type


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


def test_reward_artifact_transport_is_registry_visible() -> None:
    assert RewardFunction.artifact_transport == "in_memory"
    assert DiskArtifactRewardFunction.artifact_transport == "disk"


def test_inference_reward_defaults_to_inmemory_artifact_builder() -> None:
    class _Runtime:
        async def score_batch(self, request):
            return []

        async def shutdown(self) -> None:
            return None

    with pytest.raises(TypeError, match="inference_runtime"):
        InferenceRewardFunction(
            reward_name="fake",
            score_key="fake",
            artifact_builder=InferenceRewardFunction.build_inmemory_artifacts,
        )
    reward = InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        inference_runtime=_Runtime(),
    )
    assert reward._artifact_builder is InferenceRewardFunction.build_inmemory_artifacts
    with pytest.raises(TypeError, match="artifact_builder"):
        InferenceRewardFunction(
            reward_name="fake",
            score_key="fake",
            inference_runtime=_Runtime(),
            artifact_builder="not-callable",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("inference_runtime", [None, object()])
def test_inference_reward_rejects_invalid_runtime(inference_runtime: object) -> None:
    with pytest.raises(TypeError, match="inference_runtime"):
        InferenceRewardFunction(
            reward_name="fake",
            score_key="fake",
            inference_runtime=inference_runtime,  # type: ignore[arg-type]
            artifact_builder=InferenceRewardFunction.build_inmemory_artifacts,
        )


def test_inference_runtime_has_an_explicit_layer_name() -> None:
    reward = _reward_function_in_process()

    assert isinstance(reward.inference_runtime, InProcessRewardInferenceRuntime)
    assert not hasattr(reward, "runtime")


@pytest.mark.asyncio
async def test_reward_reports_operation_and_artifact_cleanup_failures() -> None:
    class _FailingRuntime:
        async def score_batch(self, request):
            raise RuntimeError("score failed")

        async def shutdown(self) -> None:
            return None

    def fail_cleanup(artifacts) -> None:
        assert artifacts
        raise OSError("cleanup failed")

    reward = InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        inference_runtime=_FailingRuntime(),
        artifact_builder=InferenceRewardFunction.build_inmemory_artifacts,
        artifact_finalizer=fail_cleanup,
    )

    with pytest.raises(RewardCleanupError) as error:
        await reward.score(_sample(torch.zeros(1, 3, 2, 2)))

    assert [str(item) for item in error.value.errors] == [
        "score failed",
        "cleanup failed",
    ]


@pytest.mark.asyncio
async def test_reward_retains_artifacts_when_remote_state_is_ambiguous() -> None:
    class _AmbiguousRuntime:
        async def score_batch(self, request):
            error = RuntimeError("remote state unknown")
            error.retain_reward_artifacts = True
            raise error

        async def shutdown(self) -> None:
            return None

    finalized = False
    retained = False

    def finalize(artifacts) -> None:
        nonlocal finalized
        finalized = True

    def retain(artifacts) -> None:
        nonlocal retained
        assert artifacts
        retained = True

    reward = InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        inference_runtime=_AmbiguousRuntime(),
        artifact_builder=InferenceRewardFunction.build_inmemory_artifacts,
        artifact_finalizer=finalize,
        artifact_retainer=retain,
    )

    with pytest.raises(RuntimeError, match="remote state unknown"):
        await reward.score(_sample(torch.zeros(1, 3, 2, 2)))

    assert retained is True
    assert finalized is False
