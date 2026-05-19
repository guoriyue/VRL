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
        trajectory=RewardTrajectory(prompt="prompt", seed=0, steps=[], output=output),
        metadata={"policy_version": 4, "sample_ids": ["sample-x"], "fps": 8},
    )


def test_video_artifact_store_writes_tensor_and_manifest(tmp_path: Path) -> None:
    store = VideoRewardArtifactStore(tmp_path, media_type="video")

    artifacts = store.materialize([_rollout(torch.ones(1, 2, 2, 2))])

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_id == "sample-x-0"
    assert artifact.policy_version == 4
    assert Path(artifact.path).exists()
    assert torch.load(artifact.path).shape == (1, 2, 2, 2)
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["artifact_id"] == artifact.artifact_id


def test_video_artifact_store_rejects_bad_shape(tmp_path: Path) -> None:
    store = VideoRewardArtifactStore(tmp_path, media_type="video")

    with pytest.raises(ValueError, match="video reward artifact expects"):
        store.materialize([_rollout(torch.ones(2, 2))])
