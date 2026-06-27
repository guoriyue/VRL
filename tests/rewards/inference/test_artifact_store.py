"""Tests for video reward artifact materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vrl.rewards.artifacts import VideoRewardArtifactStore
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _rollout(output: torch.Tensor) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt="prompt", output=output),
        metadata={
            "policy_version": 4,
            "sample_ids": ["sample-x"],
            "fps": 8,
            "task_type": "video2world",
            "reference_image": "/tmp/reference.png",
            "source_repo": "lerobot/droid_100",
            "source_episode": "000001",
        },
    )


def test_video_artifact_store_writes_tensor_and_manifest(tmp_path: Path) -> None:
    """Checks video artifact store writes tensor and manifest."""
    store = VideoRewardArtifactStore(tmp_path, media_type="video")

    artifacts = store.materialize([_rollout(torch.ones(1, 2, 2, 2))])

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_id == "sample-x-0"
    assert artifact.policy_version == 4
    assert Path(artifact.path).is_absolute()
    assert Path(artifact.path).exists()
    assert torch.load(artifact.path).shape == (1, 2, 2, 2)
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["artifact_id"] == artifact.artifact_id
    assert rows[0]["metadata"]["artifact_format"] == "tensor"
    assert rows[0]["metadata"]["reference_image"] == "/tmp/reference.png"
    assert rows[0]["metadata"]["source_repo"] == "lerobot/droid_100"


def test_video_artifact_store_writes_mp4_for_reward_models(tmp_path: Path) -> None:
    """Checks video artifact store writes mp4 for reward models."""
    store = VideoRewardArtifactStore(tmp_path, media_type="video", artifact_format="mp4")

    artifacts = store.materialize([_rollout(torch.ones(3, 2, 4, 4))])

    artifact = artifacts[0]
    assert artifact.path.endswith(".mp4")
    assert Path(artifact.path).is_absolute()
    assert Path(artifact.path).exists()
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["path"].endswith(".mp4")
    assert rows[0]["metadata"]["artifact_format"] == "mp4"


def test_video_artifact_store_rejects_mp4_for_non_video_media_type(tmp_path: Path) -> None:
    """Checks mp4 artifacts require explicit video media type."""

    with pytest.raises(ValueError, match="artifact_format=mp4 requires media_type=video"):
        VideoRewardArtifactStore(tmp_path, media_type="image", artifact_format="mp4")


def test_video_artifact_store_rejects_bad_shape(tmp_path: Path) -> None:
    """Checks video artifact store rejects bad shape."""
    store = VideoRewardArtifactStore(tmp_path, media_type="video")

    with pytest.raises(ValueError, match="video reward artifact expects"):
        store.materialize([_rollout(torch.ones(2, 2))])


def test_video_artifact_store_rejects_unknown_artifact_format(tmp_path: Path) -> None:
    """An unknown artifact_format is rejected against the Literal-derived allow-list."""

    with pytest.raises(ValueError, match="artifact_format must be one of"):
        VideoRewardArtifactStore(tmp_path, media_type="video", artifact_format="webm")
