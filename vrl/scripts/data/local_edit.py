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
``VRL_DATA_ROOT``); manifests hold paths relative to that root. The manifests
go to ``manifests/local_edit``: ``train.jsonl`` / ``heldout.jsonl`` and their
``*_plain_remove_add`` views (unhinted removal and addition rows), which
``dataset/local_edit`` and ``dataset/local_edit_plain_remove_add`` read.

    python -m vrl.scripts.data.local_edit
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections.abc import Sequence
from pathlib import Path

from vrl.rewards.models.local_edit import HINT_SUFFIX
from vrl.utils.paths import RootedPaths, repo_root

LOCAL_TASKS = ("attribute_modification", "swap", "removal", "addition")
OMNIEDIT = "TIGER-Lab/OmniEdit-Filtered-1.2M"
OMNIEDIT_SHARD = (
    "https://huggingface.co/datasets/"
    + OMNIEDIT
    + "/resolve/main/data/train-{:05d}-of-00571.parquet"
)
OMNIEDIT_SHARDS = 571
# The build that produced the committed manifests/local_edit/{train,heldout}.jsonl.
PER_TASK = 400
HELDOUT_PER_TASK = 40
HINT_FRACTION = 0.5
# Change-box area bounds, as a fraction of the frame.
MIN_AREA, MAX_AREA = 0.01, 0.45
# Shorter sources are skipped; the square crop keeps its native side (the
# trainer samples at 512 and evaluation resamples the reference to its own
# geometry).
MIN_SIDE = 768
SEED = 0
SHARDS_PER_TASK = 2
MANIFEST_DIR = repo_root() / "manifests" / "local_edit"
# The rows where the base model fails by doing nothing: the training view.
PLAIN_TASKS = ("removal", "addition")


def task_shards(shards_per_task: int, stride: int = 20) -> dict[str, list[str]]:
    """Shard files to stream per local task.

    The train split is grouped by task (additions in the first shards, then
    removals, swaps, attribute edits, environments, styles), so scanning it
    from the start would read a hundred shards before the first swap. Peek at
    one row every ``stride`` shards and take ``shards_per_task`` shards from
    where each task begins.
    """

    from datasets import load_dataset

    found: dict[str, list[str]] = {}
    for index in range(0, OMNIEDIT_SHARDS, stride):
        url = OMNIEDIT_SHARD.format(index)
        first = next(
            iter(
                load_dataset(
                    "parquet",
                    data_files={"train": [url]},
                    split="train",
                    streaming=True,
                    columns=["task"],
                )
            )
        )
        task = str(first["task"])
        if task in LOCAL_TASKS and task not in found:
            found[task] = [OMNIEDIT_SHARD.format(index + k) for k in range(shards_per_task)]
        if len(found) == len(LOCAL_TASKS):
            break
    missing = sorted(set(LOCAL_TASKS) - set(found))
    if missing:
        raise RuntimeError(f"no OmniEdit shard starts with task(s) {missing}")
    return found


def square_crop(image):
    """Centre square of ``image`` (the same crop for source and edit)."""

    w, h = image.size
    side = min(w, h)
    return image.crop(
        ((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)
    )


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


def rows_from_omniedit(root: Path) -> list[dict]:
    """Stream OmniEdit until every local task has ``PER_TASK`` rows; write images under
    ``root/local_edit``, return manifest rows."""

    from datasets import load_dataset
    from PIL import Image

    out = root / "local_edit"
    rng = random.Random(SEED)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "hint").mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(LOCAL_TASKS, 0)
    rows: list[dict] = []
    seen = 0
    shards = task_shards(SHARDS_PER_TASK)
    print(
        "shards per task:",
        {k: [u.rsplit("/", 1)[1] for u in v] for k, v in shards.items()},
        flush=True,
    )
    for wanted in LOCAL_TASKS:
        # One task's shards at a time, stopping as soon as that task is full.
        stream = load_dataset(
            "parquet", data_files={"train": shards[wanted]}, split="train", streaming=True
        )
        for example in stream:
            seen += 1
            if seen % 500 == 0:
                print(f"scanned {seen} rows, kept {counts}", flush=True)
            task = str(example.get("task"))
            prompts = example.get("edited_prompt_list") or []
            if task != wanted or not prompts:
                continue
            source, edited = (
                v if isinstance(v, Image.Image) else Image.open(io.BytesIO(v["bytes"]))
                for v in (example["src_img"], example["edited_img"])
            )
            if min(source.size) < MIN_SIDE:
                continue
            # The reference edit is stored at its own resolution (the source is the
            # 768-side copy); bring it onto the source's grid before differencing.
            if source.size != edited.size:
                if abs(source.size[0] / source.size[1] - edited.size[0] / edited.size[1]) > 0.1:
                    continue
                edited = edited.resize(source.size, Image.Resampling.LANCZOS)
            source, edited = square_crop(source), square_crop(edited)
            box = change_box(source, edited)
            if box is None:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if not MIN_AREA <= area <= MAX_AREA:
                continue
            stem = f"{task}_{counts[task]:04d}"
            source.convert("RGB").save(out / "img" / f"{stem}.jpg", quality=94)
            hint = rng.random() < HINT_FRACTION
            if hint:
                draw_hint(source.convert("RGB"), box).save(
                    out / "hint" / f"{stem}.jpg", quality=94
                )
            rel = out.relative_to(root)
            provenance = {}
            if example.get("omni_edit_id"):
                provenance["upstream_id"] = str(example["omni_edit_id"])
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
                            **provenance,
                        },
                    },
                }
            )
            counts[task] += 1
            if counts[task] >= PER_TASK:
                break
    short = {k: v for k, v in counts.items() if v < PER_TASK}
    if short:
        print(f"short of {PER_TASK} for {short}; raise SHARDS_PER_TASK", flush=True)
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
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    rows = rows_from_omniedit(RootedPaths.data_root().root)
    train, held = split_heldout(rows, HELDOUT_PER_TASK, SEED)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("heldout", held)):
        plain = [
            row
            for row in part
            if not row["metadata"]["local_edit"]["hint"]
            and row["metadata"]["local_edit"]["task"] in PLAIN_TASKS
        ]
        for path, view in (
            (MANIFEST_DIR / f"{name}.jsonl", part),
            (MANIFEST_DIR / f"{name}_plain_remove_add.jsonl", plain),
        ):
            with open(path, "w", encoding="utf-8") as handle:
                for row in view:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"train {len(train)} heldout {len(held)} (hint rows: {sum(r['metadata']['local_edit']['hint'] for r in rows)})"
    )


if __name__ == "__main__":
    main()
