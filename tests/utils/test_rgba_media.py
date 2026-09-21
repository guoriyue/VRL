"""RGBA survives output/archive IO; RGB reward views composite over white."""

import numpy as np
import torch
from PIL import Image

from vrl.rewards.artifacts import DiskRewardArtifactStore, InMemoryRewardArtifactStore
from vrl.rewards.models.media import artifact_middle_frame_image
from vrl.rewards.types import RewardSample
from vrl.utils.media import image_to_uint8_hwc, read_image_as_frames, write_png


def test_rgba_png_and_reward_transport_preserve_alpha_with_consistent_rgb_view(tmp_path) -> None:
    rgba = torch.zeros(4, 5, 6)
    rgba[0] = 1
    rgba[3, :, 2:4] = 0.5
    rgba[3, :, 4:] = 1
    path = tmp_path / "layer.png"
    write_png(rgba, path)
    with Image.open(path) as saved:
        assert saved.mode == "RGBA"
        assert saved.getpixel((0, 0)) == (255, 0, 0, 0)
        assert saved.getpixel((5, 0)) == (255, 0, 0, 255)
    # Compare the byte-identical wire representation (workers quantize before sending).
    wire = (rgba * 255).round().to(torch.uint8)
    expected = image_to_uint8_hwc(wire)
    np.testing.assert_array_equal(expected[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(expected[0, 5], [255, 0, 0])
    torch.testing.assert_close(
        read_image_as_frames(path)[0], torch.from_numpy(expected).float() / 255
    )
    sample = RewardSample(prompt="red layer", output=wire, sample_id="one")
    for store in (
        InMemoryRewardArtifactStore(),
        DiskRewardArtifactStore(tmp_path / "reward", media_type="image"),
    ):
        artifact = store.materialize([sample])[0]
        media = artifact.as_media()
        assert media.shape == (4, 5, 6)
        np.testing.assert_array_equal(np.asarray(artifact_middle_frame_image(artifact)), expected)
        store.release([artifact])
