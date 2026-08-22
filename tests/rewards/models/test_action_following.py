"""IDM action-following reward: architecture shapes, score math, contracts."""

from __future__ import annotations

import pytest
import torch

from vrl.rewards.models.action_following import (
    FramePairIDM,
    SpatialSoftmax,
    frame_pairs_from_clip,
    score_action_following,
)


def _tiny_idm(action_dim: int = 7) -> dict:
    model = FramePairIDM(action_dim=action_dim, width=16).eval()
    return {
        "model": model,
        "action_mean": torch.zeros(action_dim),
        "action_std": torch.ones(action_dim),
        "image_size": 32,
    }


def test_spatial_softmax_returns_bounded_keypoints() -> None:
    out = SpatialSoftmax()(torch.randn(2, 8, 5, 7))
    assert out.shape == (2, 16)
    assert out.abs().max() <= 1.0


def test_frame_pairs_shape_and_alignment() -> None:
    frames = torch.rand(33, 24, 24, 3)
    pairs = frame_pairs_from_clip(frames, image_size=32)
    assert pairs.shape == (32, 6, 32, 32)
    # pair i = (frame i, frame i+1): channels 0-2 from i, 3-5 from i+1.
    single = frame_pairs_from_clip(frames[:2], image_size=32)
    torch.testing.assert_close(single[0, :3], pairs[0, :3])


def test_frame_pairs_rejects_single_frame() -> None:
    with pytest.raises(ValueError, match="T>=2"):
        frame_pairs_from_clip(torch.rand(1, 8, 8, 3), image_size=8)


def test_score_perfect_agreement_is_one() -> None:
    idm = _tiny_idm()
    frames = torch.rand(5, 16, 16, 3)
    with torch.no_grad():
        target = idm["model"](frame_pairs_from_clip(frames, image_size=32))
    scores = score_action_following(frames, target, idm)
    assert scores["action_match"] == pytest.approx(1.0, abs=1e-5)
    assert scores["overall"] == scores["action_match"]
    assert 0.0 < scores["ee_pose_match"] <= 1.0
    assert 0.0 < scores["gripper_match"] <= 1.0


def test_score_decays_with_disagreement() -> None:
    idm = _tiny_idm()
    frames = torch.rand(5, 16, 16, 3)
    with torch.no_grad():
        target = idm["model"](frame_pairs_from_clip(frames, image_size=32))
    perfect = score_action_following(frames, target, idm)["action_match"]
    off = score_action_following(frames, target + 3.0, idm)["action_match"]
    assert off < perfect
    assert 0.0 <= off < 0.1


def test_score_uses_min_of_pairs_and_targets() -> None:
    idm = _tiny_idm()
    frames = torch.rand(4, 16, 16, 3)  # 3 pairs
    target = torch.zeros(10, 7)  # more targets than pairs
    scores = score_action_following(frames, target, idm)
    assert set(scores) == {"action_match", "ee_pose_match", "gripper_match", "overall"}


def test_checkpoint_roundtrip(tmp_path) -> None:
    from vrl.rewards.models.action_following import load_idm_checkpoint

    model = FramePairIDM(action_dim=7, width=16)
    path = tmp_path / "idm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "action_mean": torch.zeros(7),
            "action_std": torch.ones(7),
            "action_dim": 7,
            "width": 16,
            "image_size": 32,
        },
        path,
    )
    idm = load_idm_checkpoint(str(path))
    frames = torch.rand(3, 16, 16, 3)
    with torch.no_grad():
        a = idm["model"](frame_pairs_from_clip(frames, image_size=32))
        b = model(frame_pairs_from_clip(frames, image_size=32))
    torch.testing.assert_close(a, b)


def test_artifact_worker_requires_target_actions() -> None:
    from vrl.rewards.models.action_following import ActionFollowingIDMModel

    class _Artifact:
        artifact_id = "a0"

        def __init__(self) -> None:
            self.metadata: dict = {}

    worker = ActionFollowingIDMModel.__new__(ActionFollowingIDMModel)
    worker._idm = _tiny_idm()
    with pytest.raises(ValueError, match="target_actions"):
        worker(_Artifact())


def test_registry_exposes_idm_action_following() -> None:
    import vrl.rewards.functions.registry as reg

    reg._register_builtins()
    assert reg.get_reward("idm_action_following").__name__ == "ActionFollowingReward"
