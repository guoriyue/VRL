from __future__ import annotations

import torch

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.media import decode_artifact_frames


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
        path="",
        media_type="video",
        media=video,
    )

    decoded = decode_artifact_frames(artifact)

    expected = video.permute(1, 2, 3, 0).float() / 255.0
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, expected, rtol=0.0, atol=0.0)
