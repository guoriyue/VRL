"""Placement rules and red-box reading for the object-move manifests."""

from __future__ import annotations

from PIL import Image, ImageDraw

from vrl.scripts.data.object_move import landing_spot, red_box


def test_landing_spot_rests_on_the_support_and_needs_it_free() -> None:
    frame = (100.0, 100.0)
    cup = (10, 60, 20, 72)  # rests on the floor at the left
    table = (50, 50, 90, 90)
    land = landing_spot(cup, table, [], frame)
    # Bottom edge 40% down the table box, centred on it.
    assert land == (65, 54, 75, 66)
    # A person sitting where the cup would go: try the other spots, then give up.
    assert landing_spot(cup, table, [(66, 40, 90, 70)], frame) == (57, 54, 67, 66)
    assert landing_spot(cup, table, [(50, 40, 90, 70)], frame) is None


def test_landing_spot_rejects_impossible_or_trivial_moves() -> None:
    frame = (100.0, 100.0)
    table = (50, 50, 90, 90)
    # Already on the table, too small to see, too big to carry, or a tiny support.
    assert landing_spot((60, 50, 70, 60), table, [], frame) is None
    assert landing_spot((10, 60, 16, 66), table, [], frame) is None
    assert landing_spot((0, 0, 40, 40), table, [], frame) is None
    assert landing_spot((10, 60, 20, 72), (50, 50, 60, 60), [], frame) is None


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
