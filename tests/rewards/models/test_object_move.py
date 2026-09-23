"""object_move scoring over real images, with the three models replaced at their call seams.

The detector, DINOv2 and EfficientLoFTR are large downloads, so each test
feeds the score path what those models would return for a hand-built image:
detections from the known pasted positions, embeddings that are identical for
the same object crop, and correspondences computed from the known motion. The
arithmetic under test -- instance matching, direction, size, background
geometry, the composed score -- is the production code.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from vrl.rewards.models.object_move import ObjectMoveRewardModel

W = H = 200
BOX = (20.0, 80.0, 60.0, 120.0)  # the object in the source


def _scene(object_x: float | None, copy_x: float | None = None) -> Image.Image:
    image = Image.new("RGB", (W, H), (90, 140, 90))
    for x in (object_x, copy_x):
        if x is not None:
            image.paste((200, 30, 30), (int(x), 80, int(x) + 40, 120))
    return image


class _Stubbed(ObjectMoveRewardModel):
    """Model seams answer from the known layout of each test image."""

    def __init__(self, layouts: dict[int, list[float]], background_shift: float = 0.0) -> None:
        super().__init__({"device": "cpu"})
        self._layouts = layouts
        self._shift = background_shift

    def _detect(self, images, obj):
        return [
            [((x, 80.0, x + 40.0, 120.0), 0.9) for x in self._layouts[id(image)]]
            for image in images
        ]

    def _embed(self, images):
        # Every crop of the red object is the same object; nothing else is scored.
        return torch.nn.functional.normalize(torch.ones(len(images), 4), dim=-1)

    def _match(self, first, second):
        grid = torch.stack(
            torch.meshgrid(torch.arange(5.0, W, 10), torch.arange(5.0, H, 10), indexing="xy"), -1
        ).reshape(-1, 2)
        moved = grid.clone()
        if first is not second:
            moved[:, 0] += self._shift
        return grid, moved


def _score(edited_x, copy_x=None, direction="right", background_shift=0.0):
    source, edited = _scene(BOX[0]), _scene(edited_x, copy_x)
    layouts = {id(source): [BOX[0]], id(edited): [x for x in (edited_x, copy_x) if x is not None]}
    model = _Stubbed(layouts, background_shift)
    return model.score(source, edited, {"object": "block", "direction": direction})


def test_a_clean_move_in_the_instructed_direction_scores_high() -> None:
    out = _score(edited_x=90.0)
    assert out["object_move_count_ok"] == 1.0
    assert out["object_move_displacement"] == pytest.approx(0.35)
    assert out["object_move_background"] == pytest.approx(1.0)
    assert out["object_move"] == pytest.approx(1.0)


def test_unchanged_wrong_direction_duplicate_and_shift_all_score_zero() -> None:
    assert _score(edited_x=BOX[0])["object_move"] == 0.0
    assert _score(edited_x=90.0, direction="left")["object_move"] == 0.0
    duplicate = _score(edited_x=BOX[0], copy_x=90.0)
    assert duplicate["object_move_edit_count"] == 2.0 and duplicate["object_move"] == 0.0
    # The whole frame moved with the object: no background stayed in place.
    shifted = _score(edited_x=90.0, background_shift=70.0)
    assert shifted["object_move_background"] == 0.0 and shifted["object_move"] == 0.0


def test_a_lost_object_scores_zero_and_a_partial_move_scores_partially() -> None:
    assert _score(edited_x=None)["object_move"] == 0.0
    partial = _score(edited_x=45.0)  # an eighth of the image, half of full credit
    assert 0.0 < partial["object_move"] < 1.0
    assert partial["object_move_geometry"] == pytest.approx(0.5)


def test_spec_is_validated() -> None:
    model = _Stubbed({})
    with pytest.raises(ValueError, match="direction"):
        model.score(_scene(20.0), _scene(20.0), {"object": "block", "direction": "sideways"})
    with pytest.raises(ValueError, match="equal size"):
        model.score(_scene(20.0), Image.new("RGB", (10, 10)), {"object": "b", "direction": "left"})
