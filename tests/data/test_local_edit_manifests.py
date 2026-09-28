"""Local-edit manifest building: the change box from a reference edit, the hint drawing, the split."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

from PIL import Image

from vrl.rewards.models.local_edit import HINT_SUFFIX
from vrl.scripts.data import local_edit
from vrl.scripts.data.local_edit import (
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


def test_reference_retention_preserves_review_pair_without_changing_generation_input(
    tmp_path, monkeypatch
) -> None:
    source = _scene()
    edited = _scene((64, 96, 128, 160))

    def load_dataset(*args, data_files, **kwargs):
        task = data_files["train"][0].rsplit("/", 1)[-1]
        return iter(
            [
                {
                    "task": task,
                    "omni_edit_id": f"original-{task}",
                    "edited_prompt_list": ["Change the patch"],
                    "src_img": source,
                    "edited_img": edited,
                }
            ]
        )

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(
        local_edit,
        "task_shards",
        lambda _: {task: [f"fixture/{task}"] for task in local_edit.LOCAL_TASKS},
    )
    rows = local_edit.rows_from_omniedit(
        tmp_path / "retained",
        tmp_path,
        1,
        0,
        0.01,
        0.45,
        128,
        128,
        0,
        retain_reference_edits=True,
    )
    for row in rows:
        metadata = row["metadata"]["local_edit"]
        reference = tmp_path / metadata["reference_edit"]
        with Image.open(reference) as image:
            assert image.tobytes() == square_crop(edited, 128).tobytes()
        assert (
            metadata["reference_edit_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
        )
        assert metadata["upstream_id"] == f"original-{metadata['task']}"
        assert metadata["reference_edit_review_status"] == "unreviewed"
        assert row["reference_image"] == metadata["source_image"]
        assert row["reference_image"] != metadata["reference_edit"]

    # Opting out still preserves row identity without saving large reference assets.
    plain = local_edit.rows_from_omniedit(
        tmp_path / "plain", tmp_path, 1, 0, 0.01, 0.45, 128, 128, 0
    )
    assert not (tmp_path / "plain" / "reference_edits").exists()
    assert all("reference_edit" not in row["metadata"]["local_edit"] for row in plain)
    assert [row["metadata"]["local_edit"]["upstream_id"] for row in rows] == [
        row["metadata"]["local_edit"]["upstream_id"] for row in plain
    ]
