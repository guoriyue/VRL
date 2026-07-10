"""CPU preflight tests for the offline SFT target encoder."""

from __future__ import annotations

import pytest
import torch

from vrl.scripts.diffusion.encode_targets import (
    _resolve_target_videos,
    _video_at_sampling_geometry,
)
from vrl.trainers.data.prompts import PromptExample


def test_target_preflight_uses_artifact_root_and_target_identity(tmp_path) -> None:
    root = tmp_path / "artifacts"
    (root / "targets").mkdir(parents=True)
    (root / "targets/a.mp4").touch()
    (root / "targets/b.mp4").touch()
    examples = [
        PromptExample(prompt="same instruction", target_video="targets/a.mp4"),
        PromptExample(prompt="same instruction", target_video="targets/b.mp4"),
    ]

    targets = _resolve_target_videos(
        examples,
        data_root=root,
        allow_absolute=False,
    )

    assert [key for key, _ in targets] == ["targets/a.mp4", "targets/b.mp4"]
    assert [path for _, path in targets] == [
        str(root / "targets/a.mp4"),
        str(root / "targets/b.mp4"),
    ]


def test_target_preflight_rejects_duplicate_target_identity(tmp_path) -> None:
    target = tmp_path / "target.mp4"
    target.touch()
    examples = [
        PromptExample(prompt="first", target_video=str(target)),
        PromptExample(prompt="second", target_video=str(target)),
    ]

    with pytest.raises(ValueError, match="repeats target_video"):
        _resolve_target_videos(examples, data_root=None, allow_absolute=True)


def test_target_preflight_rejects_missing_file(tmp_path) -> None:
    example = PromptExample(prompt="missing", target_video="targets/missing.mp4")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_target_videos([example], data_root=tmp_path, allow_absolute=False)


def test_video_geometry_rejects_short_target(monkeypatch) -> None:
    monkeypatch.setattr(
        "vrl.utils.media.read_video_frames",
        lambda _path, *, num_frames: torch.zeros(num_frames - 1, 4, 4, 3),
    )

    with pytest.raises(ValueError, match="requires 3"):
        _video_at_sampling_geometry("short.mp4", height=4, width=4, num_frames=3)


def test_video_geometry_resizes_without_changing_frame_count(monkeypatch) -> None:
    monkeypatch.setattr(
        "vrl.utils.media.read_video_frames",
        lambda _path, *, num_frames: torch.zeros(num_frames, 2, 3, 3),
    )

    video = _video_at_sampling_geometry("target.mp4", height=4, width=6, num_frames=3)

    assert video.shape == (1, 3, 3, 4, 6)
