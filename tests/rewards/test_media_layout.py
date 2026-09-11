"""Explicit tensor layouts at the shared reward media boundary."""

import pytest
import torch

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
