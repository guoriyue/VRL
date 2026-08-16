"""Tests for the AnimeReward visual-quality reward model.

The Idefics2-8B head itself needs a pinned transformers-4.x environment, so
these cover the repo-side logic instead: media normalisation, the frame-window
fill that lets a still be scored by a video-trained head, and score scaling.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.rewards.models.animereward import AnimeRewardQualityModel

_MODEL_NAME = "/nonexistent/checkpoint-final"


class _FakeProcessor:
    """Records the frames it was handed so the window logic can be asserted."""

    def __init__(self) -> None:
        self.seen_frame_counts: list[int] = []
        self.seen_text: str = ""

    def apply_chat_template(self, messages: Any, add_generation_prompt: bool) -> str:
        assert add_generation_prompt is True
        content = messages[0]["content"]
        self.seen_text = content[-1]["text"]
        return "<chat>"

    def __call__(self, *, text: str, images: Any, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        self.seen_frame_counts.append(len(images))
        return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}


class _FakeHead(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self._logit = logit

    def forward(self, **_: Any) -> Any:
        class _Output:
            logits = torch.tensor([[self._logit]])

        return _Output()


def _model(logit: float = 70.0, **config: Any) -> tuple[AnimeRewardQualityModel, _FakeProcessor]:
    model = AnimeRewardQualityModel({"model_name": _MODEL_NAME, "device": "cpu", **config})
    processor = _FakeProcessor()
    model._processor = processor
    model._module = _FakeHead(logit)
    return model, processor


def test_requires_model_name() -> None:
    """A missing checkpoint path must fail at construction, not at first score."""
    with pytest.raises(ValueError, match="model_name"):
        AnimeRewardQualityModel({"device": "cpu"})


def test_still_is_held_across_the_whole_frame_window() -> None:
    """A single still fills the head's window instead of being zero-padded.

    The head was trained on sampled video frames; padding with blanks would put
    it out of distribution, so the still is repeated.
    """
    model, processor = _model()
    still = torch.zeros(3, 8, 8)

    model.score_media(media=still, prompt="1girl, solo")

    assert processor.seen_frame_counts == [8]


def test_longer_clip_is_subsampled_to_the_window() -> None:
    """More frames than the window are evenly subsampled, never truncated."""
    model, processor = _model(num_frames=4)
    clip = torch.zeros(10, 3, 8, 8)

    model.score_media(media=clip, prompt="1girl, solo")

    assert processor.seen_frame_counts == [4]


def test_score_is_scaled_into_unit_range() -> None:
    """The head regresses onto 0-100; the reward is reported in ~[0, 1]."""
    model, _ = _model(logit=73.5)

    scores = model.score_media(media=torch.zeros(3, 8, 8), prompt="1girl")

    assert scores == {"animereward_quality": pytest.approx(0.735)}


def test_prompt_is_carried_into_the_rubric() -> None:
    """The head is caption-conditioned, so the prompt must reach the query text."""
    model, processor = _model()

    model.score_media(media=torch.zeros(3, 8, 8), prompt="1girl, full body, blue hair")

    assert "1girl, full body, blue hair" in processor.seen_text


def test_empty_media_fails_loud() -> None:
    """An empty payload must raise rather than score an empty frame list."""
    model, _ = _model()

    with pytest.raises(ValueError, match="empty media"):
        model.score_media(media=torch.zeros(0, 3, 8, 8), prompt="1girl")


def test_unsupported_media_type_fails_loud() -> None:
    model, _ = _model()

    with pytest.raises(TypeError, match="cannot score media"):
        model.score_media(media="not-an-image", prompt="1girl")
