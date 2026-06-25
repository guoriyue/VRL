"""PhyMotion external-wrapper: config validation, JSON parsing, subprocess call."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.phymotion import PhyMotionModel, _read_phymotion_scores


def test_requires_phymotion_cmd() -> None:
    with pytest.raises(ValueError, match=r"requires worker_config\.phymotion_cmd"):
        PhyMotionModel({})


def test_rejects_cmd_without_slots() -> None:
    with pytest.raises(ValueError, match="must contain both"):
        PhyMotionModel({"phymotion_cmd": "score.py --video {video}"})


def test_read_scores_falls_back_to_mean_overall(tmp_path: Path) -> None:
    out = tmp_path / "scores.json"
    out.write_text(
        json.dumps({"kinematic": 0.6, "contact": 0.9, "dynamic": 0.3}),
        encoding="utf-8",
    )
    scores = _read_phymotion_scores(str(out))
    assert scores["overall"] == pytest.approx((0.6 + 0.9 + 0.3) / 3.0)


def test_read_scores_prefers_upstream_overall(tmp_path: Path) -> None:
    out = tmp_path / "scores.json"
    out.write_text(
        json.dumps({"kinematic": 0.6, "contact": 0.9, "dynamic": 0.3, "overall": 0.5}),
        encoding="utf-8",
    )
    assert _read_phymotion_scores(str(out))["overall"] == pytest.approx(0.5)


def test_read_scores_missing_axis_raises(tmp_path: Path) -> None:
    out = tmp_path / "scores.json"
    out.write_text(json.dumps({"kinematic": 0.6}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing axes"):
        _read_phymotion_scores(str(out))


def test_full_call_runs_external_command(tmp_path: Path) -> None:
    """A fake external scorer writes the three axes; the wrapper reads them back."""
    scorer = tmp_path / "fake_phymotion.py"
    scorer.write_text(
        "import json, sys\n"
        "json.dump({'kinematic': 0.7, 'contact': 0.8, 'dynamic': 0.6}, "
        "open(sys.argv[1], 'w'))\n",
        encoding="utf-8",
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")  # path only needs to exist for as_path()

    model = PhyMotionModel(
        {"phymotion_cmd": f"{sys.executable} {scorer} {{output}} {{video}}"},
    )
    artifact = RewardInferenceArtifact(
        artifact_id="clip", path=str(video), media_type="video", prompt="dance",
    )
    request = RewardInferenceRequest(
        request_id="t", artifacts=(artifact,), reward_name="phymotion", score_key="overall",
    )
    scores = model(artifact=artifact, request=request)
    assert scores["kinematic"] == pytest.approx(0.7)
    assert scores["overall"] == pytest.approx((0.7 + 0.8 + 0.6) / 3.0)
