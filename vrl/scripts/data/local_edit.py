"""Local-edit manifests for Qwen-Image-2.1 GRPO (``local_edit`` reward).

The skill being trained is locality: carry out an instruction ("turn the
mushroom grey", "remove the person", "add a hat") and change nothing else.
Rows come from OmniEdit-Filtered-1.2M (MIT), whose every row has a source
image, an instruction and a reference edited image. The reference edit is not
a training target (RL needs none); it is used once, to locate the **change
box**: the bounding box of the cells where the reference edit differs from the
source. That box is the region the instruction is allowed to touch -- the
``local_edit`` reward scores everything outside it for staying put.

Two arms share the same rows, so training mixes them and evaluation compares
them: ``hint`` rows show the model the box (drawn in red on a copy of the
source, with "Edit only inside the red box, then remove the red box." added to
the instruction, the SpatialEdit convention the model already follows) and
plain rows do not. Only the local task families are taken; style and
environment edits change the whole frame by design.

Sources are centre-cropped to a square before anything else -- the trainer
generates square outputs, and a non-square reference would be resampled into
one, which no consistency measure can then read.

Binary images are written under ``data/external/local_edit`` (or
``VRL_DATA_ROOT``); manifests hold paths relative to that root.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections.abc import Sequence
from pathlib import Path

from vrl.utils.artifacts import default_data_root

LOCAL_TASKS = ("attribute_modification", "swap", "removal", "addition")
HINT_SUFFIX = " Edit only inside the red box, then remove the red box."


def square_crop(image, size: int | None = None):
    """Centre square of ``image``; ``size`` resizes the result (same crop for source and edit)."""

    from PIL import Image

    w, h = image.size
    side = min(w, h)
    out = image.crop(
        ((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)
    )
    return out.resize((size, size), Image.Resampling.LANCZOS) if size and side != size else out


def change_box(
    source, edited, cells: int = 32, threshold: float = 12.0, pad: float = 0.03
) -> tuple[float, float, float, float] | None:
    """Normalized bounding box of the cells where ``edited`` differs from ``source``.

    Both images are compared on a ``cells x cells`` grid of mean absolute pixel
    difference (0-255). A cell counts as changed above ``threshold``; the box
    around the changed cells, padded by ``pad`` of the frame, is returned. None
    when nothing changed or the change covers most of the frame (not a local
    edit).
    """

    import numpy as np

    size = (cells * 8, cells * 8)
    a = np.asarray(source.convert("RGB").resize(size), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").resize(size), dtype=np.float32)
    diff = np.abs(a - b).mean(-1).reshape(cells, 8, cells, 8).mean((1, 3))
    changed = diff > threshold
    if not changed.any() or changed.mean() > 0.6:
        return None
    rows = np.where(changed.any(1))[0]
    cols = np.where(changed.any(0))[0]
    x0, x1 = cols[0] / cells, (cols[-1] + 1) / cells
    y0, y1 = rows[0] / cells, (rows[-1] + 1) / cells
    return (max(x0 - pad, 0.0), max(y0 - pad, 0.0), min(x1 + pad, 1.0), min(y1 + pad, 1.0))


def draw_hint(image, box: Sequence[float]):
    """Copy of ``image`` with ``box`` (normalized) drawn as a red outline."""

    from PIL import ImageDraw

    out = image.copy()
    w, h = out.size
    width = max(2, round(min(w, h) * 0.006))
    ImageDraw.Draw(out).rectangle(
        [box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(255, 0, 0), width=width
    )
    return out


def rows_from_omniedit(
    out: Path,
    root: Path,
    per_task: int,
    hint_fraction: float,
    min_area: float,
    max_area: float,
    size: int,
    seed: int,
) -> list[dict]:
    """Stream OmniEdit until every local task has ``per_task`` rows; write images, return manifest rows."""

    from datasets import load_dataset
    from PIL import Image

    rng = random.Random(seed)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "hint").mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(LOCAL_TASKS, 0)
    rows: list[dict] = []
    for example in load_dataset("TIGER-Lab/OmniEdit-Filtered-1.2M", split="train", streaming=True):
        task = str(example.get("task"))
        prompts = example.get("edited_prompt_list") or []
        if task not in counts or counts[task] >= per_task or not prompts:
            continue
        source, edited = (
            v if isinstance(v, Image.Image) else Image.open(io.BytesIO(v["bytes"]))
            for v in (example["src_img"], example["edited_img"])
        )
        if min(source.size) < size or source.size != edited.size:
            continue
        source, edited = square_crop(source, size), square_crop(edited, size)
        box = change_box(source, edited)
        if box is None:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if not min_area <= area <= max_area:
            continue
        stem = f"{task}_{counts[task]:04d}"
        source.convert("RGB").save(out / "img" / f"{stem}.jpg", quality=94)
        hint = rng.random() < hint_fraction
        if hint:
            draw_hint(source.convert("RGB"), box).save(out / "hint" / f"{stem}.jpg", quality=94)
        rel = out.relative_to(root)
        rows.append(
            {
                "prompt": prompts[0].strip() + (HINT_SUFFIX if hint else ""),
                "reference_image": str(rel / ("hint" if hint else "img") / f"{stem}.jpg"),
                "metadata": {
                    "source": "OmniEdit-Filtered-1.2M/train",
                    "license": "MIT",
                    "local_edit": {
                        "task": task,
                        "box": [round(v, 4) for v in box],
                        "hint": hint,
                        "source_image": str(rel / "img" / f"{stem}.jpg"),
                    },
                },
            }
        )
        counts[task] += 1
        if all(v >= per_task for v in counts.values()):
            break
    return rows


def split_heldout(rows: list[dict], per_task: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out ``per_task`` rows per task (by row; each source appears once)."""

    rng = random.Random(seed)
    held: list[dict] = []
    train: list[dict] = []
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["metadata"]["local_edit"]["task"], []).append(row)
    for task_rows in by_task.values():
        rng.shuffle(task_rows)
        held.extend(task_rows[:per_task])
        train.extend(task_rows[per_task:])
    return train, held


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-task", type=int, default=400)
    parser.add_argument("--heldout-per-task", type=int, default=40)
    parser.add_argument("--hint-fraction", type=float, default=0.5)
    parser.add_argument("--min-area", type=float, default=0.01)
    parser.add_argument("--max-area", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests/local_edit"))
    args = parser.parse_args(argv)
    root = Path(default_data_root())
    out = root / "local_edit"
    rows = rows_from_omniedit(
        out,
        root,
        args.per_task,
        args.hint_fraction,
        args.min_area,
        args.max_area,
        args.size,
        args.seed,
    )
    train, held = split_heldout(rows, args.heldout_per_task, args.seed)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("heldout", held)):
        with open(args.manifest_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for row in part:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"train {len(train)} heldout {len(held)} (hint rows: {sum(r['metadata']['local_edit']['hint'] for r in rows)})"
    )


if __name__ == "__main__":
    main()
