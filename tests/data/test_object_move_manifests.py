"""Feasibility rules and red-box reading for the object-move manifests."""

from __future__ import annotations

from PIL import Image, ImageDraw

from vrl.scripts.data.object_move import feasible_move, red_box


def test_move_goes_toward_the_free_side_and_needs_room() -> None:
    frame = (100.0, 100.0)
    assert feasible_move((60, 40, 90, 70), [], frame) == "left"
    assert feasible_move((10, 40, 40, 70), [], frame) == "right"
    # Too wide (>40% of the frame) or no side with 30% free.
    assert feasible_move((10, 40, 60, 70), [], frame) is None
    assert feasible_move((25, 40, 75, 70), [], frame) is None


def test_move_is_rejected_when_another_object_sits_where_it_would_land() -> None:
    frame = (100.0, 100.0)
    assert feasible_move((60, 40, 90, 70), [(25, 40, 55, 70)], frame) is None
    assert feasible_move((60, 40, 90, 70), [(0, 0, 10, 10)], frame) == "left"


def test_red_box_reads_the_drawn_outline_and_never_guesses() -> None:
    image = Image.new("RGB", (200, 100), (90, 120, 90))
    ImageDraw.Draw(image).rectangle((50, 20, 149, 79), outline=(255, 0, 0), width=3)
    assert red_box(image) == [0.25, 0.2, 0.75, 0.8]

    # A separate red object does not widen the box; one that touches the
    # outline merges with it, and the row is dropped rather than mislabelled.
    apart = image.copy()
    ImageDraw.Draw(apart).rectangle((170, 30, 195, 90), fill=(255, 0, 0))
    assert red_box(apart) == [0.25, 0.2, 0.75, 0.8]
    touching = image.copy()
    ImageDraw.Draw(touching).rectangle((140, 30, 190, 90), fill=(255, 0, 0))
    assert red_box(touching) is None

    filled = Image.new("RGB", (200, 100), (90, 120, 90))
    ImageDraw.Draw(filled).rectangle((50, 20, 149, 79), fill=(255, 0, 0))
    assert red_box(filled) is None
    assert red_box(Image.new("RGB", (50, 50), (90, 120, 90))) is None
