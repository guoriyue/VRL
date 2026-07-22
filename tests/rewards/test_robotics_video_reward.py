"""Tests for the audited robotics video reward blend."""

from __future__ import annotations

from pathlib import Path

import pytest

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest


class _FakeKling:
    def __init__(self, config):
        self.config = config
        self.prepared = False
        self.moves = []

    def prepare_for_inference(self):
        self.prepared = True

    def move_to(self, device):
        self.moves.append(device)

    def __call__(self, *, artifact, request):
        del artifact, request
        return {
            "text_alignment": -1.0,
            "visual_quality": -2.0,
            "motion_quality": 3.0,
            "overall_reward": 0.0,
        }


class _FakeDino:
    def __init__(self, config):
        self.config = config
        self.prepared = False
        self.moves = []

    def prepare_for_inference(self):
        self.prepared = True

    def move_to(self, device):
        self.moves.append(device)

    def __call__(self, *, artifact, request):
        del artifact, request
        return {"target_dino_similarity": 0.8}


class _FakeMotion:
    def __init__(self, config):
        self.config = config
        self.prepared = False
        self.moves = []

    def prepare_for_inference(self):
        self.prepared = True

    def move_to(self, device):
        self.moves.append(device)

    def __call__(self, *, artifact, request):
        del artifact, request
        return {"motion_dynamics": 0.5}


def test_robotics_reward_uses_only_audited_kling_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vrl.rewards.models import robotics_video_reward as robotics

    monkeypatch.setattr(robotics, "KlingVideoRewardModel", _FakeKling)
    monkeypatch.setattr(robotics, "TargetDinoSimilarityModel", _FakeDino)
    monkeypatch.setattr(robotics, "MotionDynamicsModel", _FakeMotion)
    model = robotics.RoboticsVideoRewardModel(
        {
            "device": "cuda:0",
            "data_root": "/data/droid",
            "weights": {
                "target_dino_similarity": 1.0,
                "motion_dynamics": 0.2,
                "kling_text_alignment": 0.3,
            },
            "kling": {"local_files_only": True},
        },
    )
    artifact = RewardInferenceArtifact(
        artifact_id="a0",
        path=str(tmp_path / "a0.mp4"),
        media_type="video",
        prompt="Move the object into the tray",
        metadata={"target_video": "targets/a0.mp4"},
    )
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(artifact,),
        reward_name="robotics_video_reward",
        score_key="robotics_blend",
    )

    scores = model(artifact=artifact, request=request)

    assert scores["robotics_blend"] == pytest.approx(0.8 + 0.2 * 0.5 + 0.3 * -1.0)
    assert scores["kling_motion_quality"] == 3.0
    assert model.kling.config == {"local_files_only": True, "device": "cuda:0"}
    assert model.dino.config == {"device": "cuda:0", "data_root": "/data/droid"}
    assert model.motion.config == {"device": "cuda:0"}


def test_robotics_reward_prepares_every_lazy_child_in_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.rewards.models import robotics_video_reward as robotics

    monkeypatch.setattr(robotics, "KlingVideoRewardModel", _FakeKling)
    monkeypatch.setattr(robotics, "TargetDinoSimilarityModel", _FakeDino)
    monkeypatch.setattr(robotics, "MotionDynamicsModel", _FakeMotion)
    model = robotics.RoboticsVideoRewardModel({"device": "cuda:0"})

    model.prepare_for_inference()

    assert model.kling.prepared is True
    assert model.dino.prepared is True
    assert model.motion.prepared is True


def test_robotics_reward_moves_every_child_as_one_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.rewards.models import robotics_video_reward as robotics

    monkeypatch.setattr(robotics, "KlingVideoRewardModel", _FakeKling)
    monkeypatch.setattr(robotics, "TargetDinoSimilarityModel", _FakeDino)
    monkeypatch.setattr(robotics, "MotionDynamicsModel", _FakeMotion)
    model = robotics.RoboticsVideoRewardModel({"device": "cuda:0"})

    model.move_to("cpu")
    model.move_to("cuda:0")

    assert model.kling.moves == ["cpu", "cuda:0"]
    assert model.dino.moves == ["cpu", "cuda:0"]
    assert model.motion.moves == ["cpu", "cuda:0"]


def test_robotics_reward_weights_reject_unknown_or_non_finite_values() -> None:
    from vrl.rewards.models.robotics_video_reward import RoboticsRewardWeights

    with pytest.raises(ValueError, match="unsupported"):
        RoboticsRewardWeights.from_mapping({"unknown": 1.0})
    with pytest.raises(ValueError, match="finite and non-negative"):
        RoboticsRewardWeights.from_mapping({"motion_dynamics": float("nan")})
