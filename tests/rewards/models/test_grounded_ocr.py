"""Atomic composition tests for grounded OCR."""

from __future__ import annotations

import pytest

from vrl.rewards.models.grounded_ocr import GroundedOcrConfig, GroundedOCRRewardModel


class _SpellingModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def __call__(self, artifact):
        del artifact
        return {
            "ocr": self.score,
            "ocr_raw": 1.0,
            "ocr_near_duplicate_count": 1.0,
        }


class _GuardModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def score_batch(self, artifacts):
        return [
            {
                "codex_image_qa": self.score,
                "codex_image_qa_mirror_agreement": 1.0,
            }
            for _ in artifacts
        ]


@pytest.mark.parametrize(("guard", "expected"), [(1.0, 0.6), (0.0, 0.0)])
def test_grounded_ocr_hard_gate_cannot_be_compensated(
    guard: float,
    expected: float,
) -> None:
    model = GroundedOCRRewardModel.__new__(GroundedOCRRewardModel)
    model.ocr = _SpellingModel(0.6)
    model.guard = _GuardModel(guard)

    score = model.score_batch([object()])[0]

    assert score["grounded_ocr"] == pytest.approx(expected)
    assert score["grounded_ocr_spelling"] == pytest.approx(0.6)
    assert score["grounded_ocr_near_duplicate_count"] == pytest.approx(1.0)


def test_grounded_ocr_rejects_non_binary_guard_output() -> None:
    model = GroundedOCRRewardModel.__new__(GroundedOCRRewardModel)
    model.ocr = _SpellingModel(0.6)
    model.guard = _GuardModel(0.3)

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        model.score_batch([object()])


def test_grounded_ocr_top_level_config_is_closed() -> None:
    with pytest.raises(ValueError, match=r"unknown=\['weight'\]"):
        GroundedOcrConfig.from_mapping({"ocr": {}, "guard": {}, "weight": 1.0})
