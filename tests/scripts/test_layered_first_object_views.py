"""The first recursive layer is editable even when it is not a background."""

import numpy as np
from PIL import Image

from vrl.scripts.eval.layered_candidate_views import composite, prepare_views


def test_first_object_can_be_hidden_and_moved_without_changing_second(tmp_path):
    cabinet = Image.new("RGBA", (24, 24))
    cabinet.paste((140, 80, 30, 255), (4, 10, 15, 22))
    lamp = Image.new("RGBA", cabinet.size)
    lamp.paste((230, 220, 150, 255), (8, 4, 11, 12))
    prepare_views(composite([cabinet, lamp]), [cabinet, lamp], tmp_path)
    assert np.array_equal(np.array(Image.open(tmp_path / "remove_0.png")), np.array(lamp))
    shifted = Image.new("RGBA", cabinet.size)
    shifted.paste(cabinet, (2, 0))
    expected = composite([shifted, lamp])
    assert np.array_equal(np.array(Image.open(tmp_path / "move_0.png")), np.array(expected))
