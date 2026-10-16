"""Truth-free and RORD decomposition metrics on hand-built RGBA stacks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vrl.scripts.eval.layered_decompose_eval import background_recovery, truth_free_metrics

_H = _W = 100


def _rgba(color, box=None) -> np.ndarray:
    layer = np.zeros((_H, _W, 4), dtype=np.float32)
    top, left, bottom, right = box or (0, 0, _H, _W)
    layer[top:bottom, left:right, :3] = color
    layer[top:bottom, left:right, 3] = 1.0
    return layer


def test_complete_room_with_object_hides_without_holes() -> None:
    room = _rgba((0.2, 0.4, 0.6))
    chair = _rgba((0.9, 0.1, 0.1), (20, 20, 60, 60))
    empty = np.zeros_like(room)
    source = room[..., :3].copy()
    source[20:60, 20:60] = (0.9, 0.1, 0.1)

    metrics = truth_free_metrics(np.stack([room, chair, empty]), source)

    assert metrics == {
        "nonempty_layers": 2.0,
        "composite_coverage": 1.0,
        "reconstruction_quality": 1.0,
        "hide_without_holes": 1.0,
    }


def test_room_layer_with_a_hole_under_the_object_is_penalized() -> None:
    room = _rgba((0.2, 0.4, 0.6))
    room[20:60, 20:60, 3] = 0.0  # nothing painted behind the chair
    chair = _rgba((0.9, 0.1, 0.1), (20, 20, 60, 60))

    metrics = truth_free_metrics(np.stack([room, chair]), np.zeros((_H, _W, 3), np.float32))

    assert metrics["composite_coverage"] == 1.0
    assert metrics["hide_without_holes"] == 0.0


def test_background_recovery_is_one_for_exact_background_and_zero_for_the_photo(tmp_path) -> None:
    # Colors are multiples of 1/255 so the PNG round trip is exact.
    background = _rgba((51 / 255, 102 / 255, 153 / 255))
    photo = background[..., :3].copy()
    photo[20:60, 20:60] = (230 / 255, 25 / 255, 25 / 255)
    mask = np.zeros((_H, _W), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    Image.fromarray(np.round(background * 255).astype(np.uint8), "RGBA").save(
        tmp_path / "layer_0.png"
    )
    Image.fromarray(np.round(photo * 255).astype(np.uint8), "RGB").save(tmp_path / "source.png")
    Image.fromarray(mask, "L").save(tmp_path / "foreground_mask.png")

    exact = background_recovery(np.stack([background]), tmp_path)
    unchanged = np.concatenate([photo, np.ones((_H, _W, 1), np.float32)], axis=-1)
    copied = background_recovery(np.stack([unchanged]), tmp_path)

    assert abs(exact["background_recovery"] - 1.0) < 1e-6
    assert abs(copied["background_recovery"]) < 1e-6
