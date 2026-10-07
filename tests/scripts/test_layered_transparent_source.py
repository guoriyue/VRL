"""Recursive layer review measures alpha and ignores invisible source RGB."""

import numpy as np
from PIL import Image

from vrl.scripts.eval.layered_candidate_views import prepare_views


def test_transparent_input_is_not_compared_as_an_opaque_rgb_image(tmp_path):
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[3:12, 4:11] = (80, 120, 200, 128)
    source = pixels.copy()
    source[..., :3][source[..., 3] == 0] = 255
    layers = [Image.fromarray(pixels)]
    exact = prepare_views(Image.fromarray(source), layers, tmp_path / "exact")
    assert exact["rgb_mae_over_black"] == 0
    assert exact["alpha_mae"] == 0
    opaque = pixels.copy()
    opaque[..., 3] = 255
    wrong = prepare_views(Image.fromarray(source), [Image.fromarray(opaque)], tmp_path / "wrong")
    assert wrong["alpha_mae"] > 0
