"""CPU tests for optional anime-probe metric handling."""

from __future__ import annotations

from vrl.scripts.eval.anime_probe_common import optional_finger_cv


def test_optional_finger_cv_preserves_zero() -> None:
    assert optional_finger_cv({"mean_finger_cv": 0.0}) == 0.0


def test_optional_finger_cv_keeps_missing_value_distinct() -> None:
    assert optional_finger_cv({}) is None
