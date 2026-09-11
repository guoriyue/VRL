"""Explicit tensor layouts at the shared reward media boundary."""

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.rewards.models.media import pil_frames_from_media


@pytest.mark.parametrize("frames", [1, 3, 4, 10])
def test_video_frame_count_does_not_change_tensor_layout(frames) -> None:
    media = torch.zeros(3, frames, 8, 8)
    media[:, -1] = 1
    samples = pil_frames_from_media(media)
    assert len(samples) == 1
    assert len(samples[0]) == frames
    assert samples[0][-1].getextrema() == ((255, 255),) * 3
    if frames > 1:
        assert samples[0][0].getextrema() == ((0, 0),) * 3


def test_frame_major_tensor_is_not_guessed_as_video() -> None:
    with pytest.raises(ValueError, match="channel-first video"):
        pil_frames_from_media(torch.zeros(10, 3, 8, 8))


@pytest.mark.parametrize(
    "media",
    [
        [],
        (),
        Image.new("RGB", (0, 8)),
        [Image.new("RGB", (8, 0))],
        np.empty((0, 8, 8, 3)),
        np.empty((0, 8, 3)),
        torch.empty(3, 0, 8),
        torch.empty(3, 0, 8, 8),
        torch.empty(0, 3, 2, 8, 8),
    ],
)
def test_empty_media_is_rejected_at_shared_boundary(media) -> None:
    with pytest.raises(ValueError, match="empty media"):
        pil_frames_from_media(media)
