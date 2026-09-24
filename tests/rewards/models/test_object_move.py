"""object_move scoring over real images, with the two models replaced at their call seams.

The detector and DINOv2 are large downloads, so each test feeds the score
path what those models would return for a hand-built image: detections from
the known pasted positions, crop embeddings that are identical for the same
object, and patch tokens read off the pixels of a textured scene. The
arithmetic under test -- instance matching, landing, size, the phase-
correlation shift guard, the composed score -- is the production code.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from vrl.rewards.models.object_move import ObjectMoveRewardModel

W = H = 200
CELL = 20  # the stubbed patch grid; boxes sit on cell edges so a move leaves other cells untouched
BOX = (20.0, 80.0, 60.0, 120.0)  # the object in the source
# A table top to the right of it: the object rests on it when its bottom centre is inside.
SUPPORT = {"object": "block", "support_box": [120 / W, 60 / H, 180 / W, 130 / H]}


def _scene(
    object_x: float | None, copy_x: float | None = None, shift: int = 0, seed: int = 0
) -> Image.Image:
    """A textured room (seeded noise) with the red block pasted; ``shift`` slides the whole frame."""

    if seed < 0:  # a flat wall: nothing but the block for the shift guard to lock onto
        image = Image.new("RGB", (W, H), (100, 130, 160))
    else:
        noise = torch.Generator().manual_seed(seed)
        texture = torch.randint(40, 200, (H, W, 3), generator=noise).to(torch.uint8)
        image = Image.frombytes("RGB", (W, H), texture.numpy().tobytes())
    for x in (object_x, copy_x):
        if x is not None:
            image.paste((200, 30, 30), (int(x), 80, int(x) + 40, 120))
    if shift:
        shifted = Image.new("RGB", (W, H), (0, 0, 0))
        shifted.paste(image, (shift, 0))
        return shifted
    return image


class _Stubbed(ObjectMoveRewardModel):
    """Model seams answer from the known layout of each test image."""

    def __init__(self, layouts: dict[int, list[float]]) -> None:
        super().__init__({"device": "cpu"})
        self._layouts = layouts

    def _detect(self, images, obj):
        return [
            [((x, 80.0, x + 40.0, 120.0), 0.9) for x in self._layouts[id(image)]]
            for image in images
        ]

    def _embed(self, images):
        # Every crop of the red object is the same object; nothing else is scored.
        return torch.nn.functional.normalize(torch.ones(len(images), 4), dim=-1)

    def _patches(self, image):
        # Each cell's mean colour stands in for its DINOv2 token: identical
        # pixels match at cosine 1, another texture or a pasted block does not.
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(H, W, 3)
        cells = pixels.float().reshape(H // CELL, CELL, W // CELL, CELL, 3).mean((1, 3)) - 120.0
        return torch.nn.functional.normalize(cells, dim=-1)


def _score(edited_x, copy_x=None, shift=0, seed=0):
    source, edited = _scene(BOX[0]), _scene(edited_x, copy_x, shift, seed)
    layouts = {
        id(source): [BOX[0]],
        id(edited): [x + shift for x in (edited_x, copy_x) if x is not None],
    }
    model = _Stubbed(layouts)
    return model.score(source, edited, SUPPORT)


def test_setting_the_object_on_the_support_scores_high() -> None:
    out = _score(edited_x=120.0)
    assert out["object_move_count_ok"] == 1.0
    assert out["object_move_background"] == pytest.approx(1.0)
    assert out["object_move_geometry"] == pytest.approx(1.0)
    assert out["object_move"] == pytest.approx(1.0)


def test_unchanged_away_duplicate_and_shift_all_score_zero() -> None:
    assert _score(edited_x=BOX[0])["object_move"] == 0.0
    assert _score(edited_x=0.0)["object_move"] == 0.0  # moved away from the table
    duplicate = _score(edited_x=BOX[0], copy_x=120.0)
    assert duplicate["object_move_edit_count"] == 2.0 and duplicate["object_move"] == 0.0
    # The whole frame slid 20 px with the object: the shift guard zeroes the background.
    shifted = _score(edited_x=100.0, shift=20)
    assert shifted["object_move_background"] == 0.0 and shifted["object_move"] == 0.0


def test_a_big_object_moved_across_a_flat_wall_is_not_a_frame_shift() -> None:
    # With the block blanked, the correlation peak stays at zero; without it the
    # block's own 100 px move would read as the frame sliding.
    source, edited = _scene(BOX[0], seed=-1), _scene(120.0, seed=-1)
    out = _Stubbed({id(source): [BOX[0]], id(edited): [120.0]}).score(source, edited, SUPPORT)
    assert out["object_move_background"] == pytest.approx(1.0)
    assert out["object_move"] == pytest.approx(1.0)


def test_a_lost_object_scores_zero_and_a_partial_move_scores_partially() -> None:
    assert _score(edited_x=None)["object_move"] == 0.0
    # Bottom centre from x 40 to x 90: 50 of the 80 px gap to the table closed.
    partial = _score(edited_x=70.0)
    assert 0.0 < partial["object_move"] < 1.0
    assert partial["object_move_geometry"] == pytest.approx(50 / 80)


def test_spec_is_validated() -> None:
    model = _Stubbed({})
    with pytest.raises(ValueError, match="target_box, support_box"):
        model.score(_scene(20.0), _scene(20.0), {"object": "block", "direction": "left"})
    with pytest.raises(ValueError, match="equal size"):
        model.score(_scene(20.0), Image.new("RGB", (10, 10)), SUPPORT)


def test_target_box_mode_scores_overlap_with_the_drawn_target() -> None:
    source, hit, miss = _scene(BOX[0]), _scene(80.0), _scene(150.0)
    layouts = {id(source): [BOX[0]], id(hit): [80.0], id(miss): [150.0]}
    model = _Stubbed(layouts)
    target = {"object": "block", "target_box": [80 / W, 80 / H, 120 / W, 120 / H]}

    on_target = model.score(source, hit, target)
    off_target = model.score(source, miss, target)

    assert on_target["object_move_geometry"] == pytest.approx(1.0)
    assert on_target["object_move"] == pytest.approx(1.0)
    assert off_target["object_move"] == 0.0


def test_shaped_score_ranks_move_over_cleared_origin_over_copy_over_a_redrawn_scene() -> None:
    moved = _score(edited_x=130.0)["object_move_shaped"]
    unchanged = _score(edited_x=BOX[0])["object_move_shaped"]
    duplicate = _score(edited_x=BOX[0], copy_x=120.0)["object_move_shaped"]
    lost = _score(edited_x=None)["object_move_shaped"]
    redrawn = _score(edited_x=None, seed=1)["object_move_shaped"]  # another texture entirely

    assert moved == pytest.approx(1.0)
    # w = 0.2: a faithful failure keeps up to 0.2 / 1.2 for the scene, the full
    # amount only when the origin was cleared (lost), half when the original
    # is still there (no-op, copy).
    assert lost == pytest.approx(0.2 / 1.2)
    assert unchanged == duplicate == pytest.approx(0.1 / 1.2)
    assert redrawn < 0.05


class _SeenFromANewSide(_Stubbed):
    """Edit crops look only faintly like the source crop (cosine ``edit_cosine``)."""

    def __init__(self, layouts, edit_cosine, boxes=None):
        super().__init__(layouts)
        self._cos = edit_cosine
        self._boxes = boxes or {}

    def _detect(self, images, obj):
        return [self._boxes.get(id(image)) or super()._detect([image], obj)[0] for image in images]

    def _embed(self, images):
        # First crop is the source object; the rest are edit crops at the given cosine.
        source = torch.tensor([1.0, 0.0])
        edit = torch.tensor([self._cos, (1 - self._cos**2) ** 0.5])
        return torch.stack([source] + [edit] * (len(images) - 1))


def test_support_moves_count_a_reposed_object_but_not_a_furniture_sized_box() -> None:
    source, onto = _scene(BOX[0]), _scene(120.0)
    layouts = {id(source): [BOX[0]], id(onto): [120.0]}
    # Cosine 0.2 to its source crop: below the 0.5 instance floor, above noise.
    reposed = _SeenFromANewSide(layouts, edit_cosine=0.2).score(source, onto, SUPPORT)
    assert reposed["object_move_edit_count"] == 1.0 and reposed["object_move_geometry"] == 1.0
    # The whole table detected as the object (10x its area) is not a second copy.
    boxes = {id(onto): [((120.0, 80.0, 160.0, 120.0), 0.9), ((0.0, 0.0, 200.0, 80.0), 0.3)]}
    noisy = _SeenFromANewSide(layouts, edit_cosine=0.9, boxes=boxes).score(source, onto, SUPPORT)
    assert noisy["object_move_edit_count"] == 1.0


def test_support_moves_allow_a_depth_rescale_up_to_four_times() -> None:
    source, onto = _scene(BOX[0]), _scene(120.0)
    far = {id(onto): [((130.0, 100.0, 150.0, 120.0), 0.9)]}  # a quarter of the area
    out = _SeenFromANewSide({id(source): [BOX[0]]}, 0.9, far).score(source, onto, SUPPORT)
    assert out["object_move_geometry"] == pytest.approx(1.0)
