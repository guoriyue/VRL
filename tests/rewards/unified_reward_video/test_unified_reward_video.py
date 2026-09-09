"""UnifiedReward-2.0 judge output: axis parsing and rubric loading.

The wrapper's adapter behavior (artifact materialization, ``score_key``
selection, config shape) is shared with the other disk-artifact rewards and
lives in ``tests/rewards/test_disk_artifact_reward_functions.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vrl.rewards.models.unified_reward_video import _load_rubric, _parse_axis_scores


def test_parse_axis_scores_reads_floats() -> None:
    text = "Alignment Score (1-5): 2.4036\nPhysics Score (1-5): 3.0987\nStyle Score (1-5): 3.3889"
    scores = _parse_axis_scores(text)
    assert scores["alignment"] == pytest.approx(2.4036)
    assert scores["physics"] == pytest.approx(3.0987)
    assert scores["style"] == pytest.approx(3.3889)
    assert scores["overall"] == pytest.approx((2.4036 + 3.0987 + 3.3889) / 3.0)


def test_parse_axis_scores_normalizes_model_declared_scale() -> None:
    text = (
        "Alignment Score (1-5): 2.7907\n"
        "Physics Score (1.0-10.0): 4.8017\n"
        "Style Score (1-5): 2.7633"
    )
    scores = _parse_axis_scores(text)
    assert scores["alignment"] == pytest.approx(2.7907)
    assert scores["physics"] == pytest.approx(1.0 + 4.0 * (4.8017 - 1.0) / 9.0)
    assert scores["style"] == pytest.approx(2.7633)


@pytest.mark.parametrize(
    ("physics_line", "message"),
    [
        ("Physics Score (5-1): 3", "invalid 'physics' score range"),
        ("Physics Score (1-5): 7", "out-of-range 'physics' score"),
    ],
)
def test_parse_axis_scores_rejects_invalid_declared_scale(
    physics_line: str,
    message: str,
) -> None:
    text = f"Alignment Score (1-5): 4\n{physics_line}\nStyle Score (1-5): 3"
    with pytest.raises(ValueError, match=message):
        _parse_axis_scores(text)


def test_parse_axis_scores_missing_axis_raises() -> None:
    with pytest.raises(ValueError, match="missing 'style'"):
        _parse_axis_scores("Alignment Score (1-5): 4\nPhysics Score (1-5): 3")


def test_load_rubric_default_and_override(tmp_path: Path) -> None:
    assert "{prompt}" in _load_rubric("")
    rubric = tmp_path / "r.yaml"
    rubric.write_text(
        "problem_template: 'judge {prompt} on Style Score (1-5): Z'\n", encoding="utf-8"
    )
    assert "{prompt}" in _load_rubric(str(rubric))


def test_load_rubric_rejects_missing_prompt_slot(tmp_path: Path) -> None:
    rubric = tmp_path / "bad.yaml"
    rubric.write_text("problem_template: 'no slot here'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a"):
        _load_rubric(str(rubric))


@pytest.mark.parametrize("failure_at", [None, "get", "read", "convert"])
def test_frame_sampling_releases_capture_on_success_and_failure(monkeypatch, failure_at) -> None:
    import cv2
    import numpy as np

    from vrl.rewards.models.unified_reward_video import _sample_frames

    failure = RuntimeError("decoder failed")

    class Capture:
        releases = 0

        def get(self, key):
            if failure_at == "get":
                raise failure
            return 1

        def isOpened(self):
            return True

        def read(self):
            if failure_at == "read":
                raise failure
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            self.releases += 1

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    if failure_at == "convert":

        def broken_conversion(*args):
            raise failure

        monkeypatch.setattr(cv2, "cvtColor", broken_conversion)
    if failure_at:
        with pytest.raises(RuntimeError) as caught:
            _sample_frames("video.mp4", 1)
        assert caught.value is failure
    else:
        frames = _sample_frames("video.mp4", 1)
        assert len(frames) == 1
        assert frames[0].mode == "RGB"
    assert capture.releases == 1
