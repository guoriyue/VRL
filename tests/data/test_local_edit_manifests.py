"""Local-edit manifest building: the change box from a reference edit, the hint drawing, the split."""

from __future__ import annotations

from PIL import Image

from vrl.scripts.data.local_edit import (
    HINT_SUFFIX,
    change_box,
    draw_hint,
    split_heldout,
    square_crop,
)


def _scene(patch: tuple[int, int, int, int] | None = None, size: int = 256) -> Image.Image:
    image = Image.new("RGB", (size, size), (90, 140, 90))
    if patch:
        image.paste((200, 30, 30), patch)
    return image


def test_change_box_covers_the_edited_patch_and_nothing_else() -> None:
    box = change_box(_scene(), _scene((64, 96, 128, 160)))
    assert box is not None
    x0, y0, x1, y1 = box
    # The patch spans x 0.25-0.5, y 0.375-0.625; the box holds it with a small pad.
    assert x0 <= 0.25 <= 0.5 <= x1 and y0 <= 0.375 <= 0.625 <= y1
    assert (x1 - x0) * (y1 - y0) < 0.3


def test_change_box_rejects_no_change_and_whole_frame_change() -> None:
    assert change_box(_scene(), _scene()) is None
    assert change_box(_scene(), Image.new("RGB", (256, 256), (10, 10, 200))) is None


def test_square_crop_takes_the_centre_and_hint_draws_red_on_a_copy() -> None:
    wide = Image.new("RGB", (400, 200), (0, 0, 0))
    wide.paste((255, 255, 255), (100, 0, 300, 200))
    square = square_crop(wide)
    assert square.size == (200, 200) and square.getpixel((100, 100)) == (255, 255, 255)
    source = _scene()
    hinted = draw_hint(source, (0.25, 0.25, 0.75, 0.75))
    assert hinted.getpixel((64, 128)) == (255, 0, 0) and source.getpixel((64, 128)) == (
        90,
        140,
        90,
    )
    assert hinted.getpixel((128, 128)) == (90, 140, 90)


def test_split_holds_out_per_task_and_hint_rows_carry_the_suffix() -> None:
    rows = [
        {
            "prompt": "x" + (HINT_SUFFIX if i % 2 else ""),
            "metadata": {"local_edit": {"task": t, "hint": bool(i % 2)}},
        }
        for t in ("swap", "removal")
        for i in range(5)
    ]
    train, held = split_heldout(rows, per_task=2, seed=0)
    assert len(held) == 4 and len(train) == 6
    assert all(
        r["prompt"].endswith(HINT_SUFFIX) == r["metadata"]["local_edit"]["hint"] for r in rows
    )
