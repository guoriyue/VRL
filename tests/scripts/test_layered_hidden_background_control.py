"""Exact opaque recomposition does not imply a usable hidden background."""

import json

import numpy as np
from PIL import Image

from vrl.scripts.eval.layered_reward_calibration import prepare_controls


def test_opaque_control_reveals_missing_content_only_after_removal(tmp_path):
    source = Image.new("RGB", (32, 24), (180, 130, 80))
    foreground = np.zeros((24, 32, 4), dtype=np.uint8)
    foreground[5:19, 8:24] = (20, 90, 180, 255)
    layers = [
        Image.new("RGBA", source.size),
        source.convert("RGBA"),
        Image.fromarray(foreground),
    ]
    # The largest upper layer is the room, not the intended occluding object.
    prepare_controls(source, layers, tmp_path, foreground=2)
    control = tmp_path / "opaque_background_hole"
    stats = json.loads((control / "diagnostics.json").read_text())
    assert stats["rgb_mae_over_black"] == 0
    assert stats["composite_under_opaque_pixels"] == 0
    removed = np.asarray(Image.open(control / "remove_2.png"))
    assert np.all(removed[5:19, 8:24] == (0, 0, 0, 255))
    assert np.all(removed[0, 0] == (180, 130, 80, 255))
