from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np

from vrl.scripts.eval import kling_reward_noise_gate as gate


def test_import_does_not_select_a_video_reader(monkeypatch) -> None:
    monkeypatch.delenv("FORCE_QWENVL_VIDEO_READER", raising=False)

    importlib.reload(gate)

    assert "FORCE_QWENVL_VIDEO_READER" not in os.environ


def test_scoring_uses_public_reward_artifacts(monkeypatch, tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "rollout.mp4").touch()
    scored = []

    class FakeModel:
        def __init__(self, _config) -> None:
            pass

        def __call__(self, artifact):
            scored.append(artifact)
            return {
                "visual_quality": 1.0,
                "motion_quality": 2.0,
                "text_alignment": 4.0,
                "overall_reward": 3.0,
            }

    monkeypatch.setattr(
        "vrl.rewards.models.kling_video_reward.KlingVideoRewardModel",
        FakeModel,
    )
    monkeypatch.setattr(gate.iio, "imread", lambda _path: np.zeros((2, 4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(gate.iio, "immeta", lambda _path: {"fps": 8.0})
    monkeypatch.setattr(gate.iio, "imwrite", lambda path, _frames, **_kwargs: Path(path).touch())
    args = argparse.Namespace(
        videos=str(videos),
        shard="0/1",
        n=1,
        out=str(tmp_path / "scores.json"),
    )

    gate._run_shard(args)

    assert len(scored) == len(gate._LEVELS) + 1
    assert all(artifact.prompt == gate._PROMPT for artifact in scored)
    assert all(Path(artifact.path).is_absolute() for artifact in scored)
    assert os.environ["FORCE_QWENVL_VIDEO_READER"] == "decord"


def test_combine_classifies_public_reward_keys(capsys, tmp_path: Path) -> None:
    levels = {}
    for index, sigma in enumerate(gate._LEVELS):
        levels[str(sigma)] = {
            "visual_quality": 1.0 - index * 0.1,
            "motion_quality": 1.0 + index * 0.1,
            "overall_reward": 1.0,
        }
    shard = tmp_path / "scores.json"
    shard.write_text(
        json.dumps(
            [
                {
                    "video": "rollout.mp4",
                    "levels": levels,
                    "repeat0": levels["0"],
                },
            ],
        ),
        encoding="utf-8",
    )

    gate._combine([str(shard)])

    output = capsys.readouterr().out
    assert "visual_quality" in output
    assert "MIS-WEIGHTED reward" in output
    assert "MQ is INVERTED and Overall is FLAT" in output
