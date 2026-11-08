from __future__ import annotations

import torch

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.media import artifact_middle_frame_image, decode_artifact_frames


def test_inline_uint8_video_reaches_reward_as_original_normalized_pixels() -> None:
    video = torch.tensor(
        [
            [[[0, 1], [2, 127]], [[128, 254], [255, 3]]],
            [[[4, 5], [6, 126]], [[129, 253], [252, 7]]],
            [[[8, 9], [10, 125]], [[130, 251], [250, 11]]],
        ],
        dtype=torch.uint8,
    )
    artifact = RewardInferenceArtifact(
        artifact_id="inline-uint8-video",
        sample_id="sample-inline-uint8-video",
        path="",
        media=video,
    )

    decoded = decode_artifact_frames(artifact)

    expected = video.permute(1, 2, 3, 0).float() / 255.0
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, expected, rtol=0.0, atol=0.0)


def test_inline_image_is_one_frame() -> None:
    """Text-to-image rollouts hand rewards a ``[C,H,W]`` tensor, not a video."""

    image = torch.zeros(3, 4, 5, dtype=torch.uint8)
    image[0] = 255
    artifact = RewardInferenceArtifact(
        artifact_id="inline-image",
        sample_id="sample-inline-image",
        path="",
        media=image,
    )

    decoded = decode_artifact_frames(artifact)

    assert decoded.shape == (1, 4, 5, 3)
    assert decoded[0, 0, 0].tolist() == [1.0, 0.0, 0.0]


def test_middle_frame_image_of_a_video_is_its_middle_frame() -> None:
    """Image-domain rewards score frame T // 2, not the conditioning first frame."""

    video = torch.arange(5, dtype=torch.uint8).mul(40).view(1, 5, 1, 1).expand(3, 5, 2, 2)
    artifact = RewardInferenceArtifact(
        artifact_id="inline-five-frame-video",
        sample_id="sample-inline-five-frame-video",
        path="",
        media=video.contiguous(),
    )

    image = artifact_middle_frame_image(artifact)

    assert image.getpixel((0, 0)) == (80, 80, 80)
