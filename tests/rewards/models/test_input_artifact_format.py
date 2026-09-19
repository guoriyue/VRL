"""Only path-consuming reward models opt into scorer-local materialization."""

from __future__ import annotations

from importlib import import_module
from typing import ClassVar, Literal

import pytest

from vrl.rewards.models.base import FileRewardModel


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("hpsv3", "HPSv3Model"),
        ("kling_video_reward", "KlingVideoRewardModel"),
        ("unified_reward_video", "UnifiedRewardVideoModel"),
        ("videocon_physics", "VideoConPhysicsModel"),
        ("robotics_video_reward", "RoboticsVideoRewardModel"),
    ],
)
def test_path_consuming_model_declares_mp4_without_loading(module_name, class_name) -> None:
    model_type = getattr(import_module(f"vrl.rewards.models.{module_name}"), class_name)
    # Capability discovery must not require downloading weights or building CUDA state.
    model = object.__new__(model_type)

    assert isinstance(model, FileRewardModel)
    assert model.input_artifact_format == "mp4"


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("motion_dynamics", "MotionDynamicsModel"),
        ("target_dino_similarity", "TargetDinoSimilarityModel"),
        ("idm_action_following", "ActionFollowingIDMModel"),
        ("ocr", "OCRRewardModel"),
        ("countgd", "CountGDModel"),
        ("wd_tagger", "WDTaggerRewardModel"),
        ("nsfw_safety", "NSFWSafetyRewardModel"),
        ("geneval_owl", "GenEvalOwlRewardModel"),
    ],
)
def test_media_decoding_model_does_not_request_local_file(module_name, class_name) -> None:
    model_type = getattr(import_module(f"vrl.rewards.models.{module_name}"), class_name)
    model = object.__new__(model_type)

    assert not isinstance(model, FileRewardModel)
    assert not hasattr(model, "input_artifact_format")


@pytest.mark.parametrize("artifact_format", ["mp4", "tensor"])
def test_external_factory_model_declares_file_capability_structurally(artifact_format) -> None:
    class ExternalModel:
        input_artifact_format: ClassVar[Literal["mp4", "tensor"]] = artifact_format

        def __call__(self, artifact):
            return {"score": float(bool(artifact.as_path()))}

    assert isinstance(ExternalModel(), FileRewardModel)
