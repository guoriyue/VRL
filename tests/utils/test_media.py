from __future__ import annotations

import numpy as np
import pytest
import torch

from vrl.utils.media import image_to_uint8_hwc, sample_frames, video_tensor_to_uint8_frames


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 10.5, "2"])
def test_frame_sampling_rejects_invalid_count_before_noop(count) -> None:
    with pytest.raises(ValueError, match="num_frames"):
        sample_frames(torch.arange(4).reshape(4, 1), count)


def test_frame_sampling_preserves_order_and_noop_identity() -> None:
    frames = torch.arange(5).reshape(5, 1)
    assert torch.equal(sample_frames(frames, 3), frames[[0, 2, 4]])
    for count in (None, 5, 8):
        assert sample_frames(frames, count) is frames


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


@pytest.mark.parametrize("layout", ["chw", "hwc"])
def test_numpy_single_image_batch_preserves_pixels(layout: str) -> None:
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    source = image.transpose(2, 0, 1) if layout == "chw" else image
    np.testing.assert_array_equal(image_to_uint8_hwc(source[None]), image)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_image_conversion_rejects_multiple_images(as_tensor: bool) -> None:
    images = np.zeros((2, 3, 5, 6), dtype=np.uint8)
    source = torch.from_numpy(images) if as_tensor else images
    with pytest.raises(ValueError, match="expected one image"):
        image_to_uint8_hwc(source)
