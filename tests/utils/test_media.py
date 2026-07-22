from __future__ import annotations

import numpy as np
import torch

from vrl.utils.media import image_to_uint8_hwc, video_tensor_to_uint8_frames


def test_image_to_uint8_hwc_preserves_uint8_tensor_values() -> None:
    image = torch.tensor(
        [
            [[0, 1, 2], [127, 128, 255]],
            [[3, 4, 5], [129, 254, 253]],
            [[6, 7, 8], [130, 252, 251]],
        ],
        dtype=torch.uint8,
    )

    converted = image_to_uint8_hwc(image)

    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, image.permute(1, 2, 0).numpy())


def test_image_to_uint8_hwc_clips_integer_tensor_without_float_scaling() -> None:
    image = torch.tensor(
        [
            [[-1, 0, 1]],
            [[2, 254, 255]],
            [[256, 127, 128]],
        ],
        dtype=torch.int16,
    )

    converted = image_to_uint8_hwc(image)

    expected = np.array(
        [[[0, 2, 255], [0, 254, 127], [1, 255, 128]]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(converted, expected)


def test_video_tensor_to_uint8_frames_preserves_uint8_values() -> None:
    video = torch.tensor(
        [
            [[[0, 1]], [[2, 255]]],
            [[[3, 4]], [[5, 254]]],
            [[[6, 7]], [[8, 253]]],
        ],
        dtype=torch.uint8,
    )

    converted = video_tensor_to_uint8_frames(video)

    expected = video.permute(1, 2, 3, 0).numpy()
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, expected)


def test_video_tensor_to_uint8_frames_scales_unit_float_values() -> None:
    video = torch.tensor(
        [
            [[[0.0, 1.0]]],
            [[[0.5, 0.25]]],
            [[[1.0 / 255.0, 254.0 / 255.0]]],
        ],
        dtype=torch.float32,
    )

    converted = video_tensor_to_uint8_frames(video)

    expected = np.array([[[[0, 128, 1], [255, 64, 254]]]], dtype=np.uint8)
    np.testing.assert_array_equal(converted, expected)
