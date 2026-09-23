"""Object-move edit manifests for Qwen-Image-2.1 GRPO (``object_move`` reward).

Three sources, each converted into prompt-manifest rows that carry the source
photo as ``reference_image`` and the reward's ``metadata.object_move`` spec:

- ``coco``: real photos from COCO 2017 with ground-truth boxes. A row is made
  only when the move is physically possible (the feasibility rules below), the
  photo is cropped to a square that keeps it possible, and the instruction
  names the category and a horizontal direction toward the free side.
- ``spatialedit``: SpatialEdit-500K ``object_moving`` shards. The destination is
  a red box drawn on the source ("Move X into the red box ..."); its coordinates
  are read from the red pixels and stored as ``target_box``.
- ``bench``: SpatialEdit-Bench ``move`` items (held out; ``bbox_gt`` is the
  target box).

Feasibility rules (the same rules select the offline reward-gate cases): one
non-crowd instance of the category (``person`` excluded: usually cropped or in
groups), box width <= 40% of the frame, area 3-25% of the frame, >= 30% of the
width free on one side, and no other annotated object where it would land.

Binary images are written under ``data/external/object_move`` (or
``VRL_DATA_ROOT``); manifests hold paths relative to that root.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import tarfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import default_data_root, emit, repo_root

COCO_URL = "http://images.cocodataset.org/{split}/{file}"
SPATIALEDIT_TRAIN = (
    "https://huggingface.co/datasets/EasonXiao-888/SpatialEdit-500K/resolve/main/"
    "object_moving/worker0-{shard:06d}.tar"
)
SPATIALEDIT_BENCH = "https://huggingface.co/datasets/EasonXiao-888/SpatialEdit-Bench/resolve/main/"
MAX_WIDTH, MIN_AREA, MAX_AREA, MIN_ROOM, SHIFT = 0.40, 0.03, 0.25, 0.30, 0.35
_MOVE_PHRASE = re.compile(r"^move (.+?) into the red box", re.IGNORECASE)


def _fetch(url: str, timeout: float = 120.0, attempts: int = 4) -> bytes:
    """Download ``url``, retrying transient network failures."""

    import time

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def feasible_move(
    box: Sequence[float], others: Sequence[Sequence[float]], size: tuple[float, float]
) -> str | None:
    """The horizontal direction a move of ``box`` can take in a ``size`` frame, if any."""

    width, height = size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w > MAX_WIDTH * width or not MIN_AREA <= w * h / (width * height) <= MAX_AREA:
        return None
    left_room, right_room = x0, width - x1
    direction = "left" if left_room >= right_room else "right"
    room = max(left_room, right_room)
    if room < MIN_ROOM * width:
        return None
    shift = min(SHIFT * width, room) * (-1 if direction == "left" else 1)
    land = (x0 + shift, y0, x1 + shift, y1)
    if any(_overlap(land, other) > 0.1 * w * h for other in others):
        return None
    return direction


def coco_rows(annotations: Path, split: str, out: Path, limit: int, skip: set[int]) -> list[dict]:
    """Square-cropped, feasibility-checked COCO rows (one per image, one category each)."""

    data = json.loads(annotations.read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}
    by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    rows: list[dict] = []
    out.mkdir(parents=True, exist_ok=True)
    for image_id in sorted(by_image):
        if len(rows) >= limit:
            break
        if image_id in skip:
            continue
        info = images[image_id]
        row = _coco_row(info, by_image[image_id], cats, split, out)
        if row is not None:
            rows.append(row)
    return rows


def _coco_row(info: dict, anns: list[dict], cats: dict, split: str, out: Path) -> dict | None:
    from PIL import Image

    width, height = info["width"], info["height"]
    side = min(width, height)
    for ann in anns:
        name = cats[ann["category_id"]]
        same = [a for a in anns if a["category_id"] == ann["category_id"]]
        if ann["iscrowd"] or name == "person" or len(same) != 1:
            continue
        x0, y0, x1, y1 = _xyxy(ann["bbox"])
        # Square crops that keep the object whole: centred on it, or flush to either edge.
        offsets = sorted(
            {
                min(max(0.0, (x0 + x1) / 2 - side / 2), width - side),
                0.0,
                float(width - side),
            }
        )
        for off_x in offsets if width > height else [0.0]:
            off_y = (
                0.0 if width >= height else min(max(0.0, (y0 + y1) / 2 - side / 2), height - side)
            )
            crop = (off_x, off_y, off_x + side, off_y + side)
            if not (crop[0] <= x0 and x1 <= crop[2] and crop[1] <= y0 and y1 <= crop[3]):
                continue
            box = (x0 - off_x, y0 - off_y, x1 - off_x, y1 - off_y)
            others = [
                (a[0] - off_x, a[1] - off_y, a[2] - off_x, a[3] - off_y)
                for a in (_xyxy(o["bbox"]) for o in anns if o is not ann)
            ]
            direction = feasible_move(box, others, (side, side))
            if direction is None:
                continue
            path = out / info["file_name"]
            if not path.exists():
                payload = _fetch(COCO_URL.format(split=split, file=info["file_name"]))
                image = Image.open(io.BytesIO(payload)).convert("RGB")
                image.crop(tuple(round(v) for v in crop)).save(path, quality=95)
            rel = path.relative_to(out.parents[1]).as_posix()
            return {
                "prompt": f"Move the {name} to the {direction} side of the image. "
                "Keep everything else unchanged.",
                "reference_image": rel,
                "metadata": {
                    "source": f"coco_{split}",
                    "coco_image_id": info["id"],
                    "object_move": {
                        "object": name,
                        "direction": direction,
                        "source_box": [v / side for v in box],
                    },
                },
            }
    return None


def red_box(image: Any) -> list[float] | None:
    """Normalized box of the red rectangle outline drawn on a SpatialEdit source.

    Red pixels are split into connected components and only outline-shaped ones
    count: each of the four sides of the component's box is mostly red while
    its interior is mostly not. Red things in the scene (a jersey, a red bin)
    fail that test even when they touch the drawn box.
    """

    import cv2
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    red = (pixels[..., 0] > 180) & (pixels[..., 1] < 70) & (pixels[..., 2] < 70)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    height, width = red.shape
    best: tuple[int, list[float]] | None = None
    for index in range(1, len(stats)):
        x, y, w, h = (int(v) for v in stats[index][:4])
        if h < 8 or w < 8:
            continue
        component = labels[y : y + h, x : x + w] == index
        sides = (
            component[:3].any(0),
            component[-3:].any(0),
            component[:, :3].any(1),
            component[:, -3:].any(1),
        )
        interior = component[4:-4, 4:-4]
        if min(float(side.mean()) for side in sides) < 0.8 or (
            interior.size and interior.mean() > 0.2
        ):
            continue
        box = [x / width, y / height, (x + w) / width, (y + h) / height]
        if best is None or h * w > best[0]:
            best = (h * w, box)
    return None if best is None else best[1]


def spatialedit_rows(shards: Sequence[int], out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for shard in shards:
        payload = _fetch(SPATIALEDIT_TRAIN.format(shard=shard), timeout=300)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            rows.extend(_spatialedit_shard_rows(archive, out))
    return rows


def _spatialedit_shard_rows(archive: tarfile.TarFile, out: Path) -> list[dict]:
    from PIL import Image

    rows: list[dict] = []
    members = {m.name: m for m in archive.getmembers()}
    for name in sorted(n for n in members if n.endswith(".json")):
        stem = name[: -len(".json")]
        if f"{stem}.0.jpg" not in members:
            continue
        meta = json.loads(archive.extractfile(members[name]).read())
        text = meta["conversations"][0]["value"].replace("<image>", "").strip()
        match = _MOVE_PHRASE.match(text)
        source = Image.open(io.BytesIO(archive.extractfile(members[f"{stem}.0.jpg"]).read()))
        target = red_box(source)
        if match is None or target is None:
            continue
        path = out / f"{stem}.jpg"
        source.convert("RGB").save(path, quality=95)
        rows.append(
            {
                "prompt": text,
                "reference_image": path.relative_to(out.parents[1]).as_posix(),
                "metadata": {
                    "source": "spatialedit_500k",
                    "object_move": {"object": match.group(1), "target_box": target},
                },
            }
        )
    return rows


def bench_rows(out: Path, limit: int) -> list[dict]:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(_fetch(SPATIALEDIT_BENCH + "SpatialEdit_Bench_Meta_File.json"))
    items = [item for item in meta if item["type"] == "move"][:limit]
    rows: list[dict] = []
    for item in items:
        path = out / item["image_path"].replace("/", "__")
        if not path.exists():
            path.write_bytes(_fetch(SPATIALEDIT_BENCH + "images/" + item["image_path"]))
        with Image.open(path) as image:
            width, height = image.size
        x0, y0, x1, y1 = (float(v) for v in json.loads(item["bbox_gt"]))
        rows.append(
            {
                "prompt": item["prompt"],
                "reference_image": path.relative_to(out.parents[1]).as_posix(),
                "metadata": {
                    "source": "spatialedit_bench",
                    "bench_image_id": item["image_id"],
                    "bench_edit_id": item["edit_id"],
                    "object_move": {
                        "object": item["object"],
                        "target_box": [x0 / width, y0 / height, x1 / width, y1 / height],
                    },
                },
            }
        )
    return rows


def _write(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coco-train-annotations", type=Path, required=True)
    parser.add_argument("--coco-val-annotations", type=Path, required=True)
    parser.add_argument("--coco-train", type=int, default=600)
    parser.add_argument("--coco-heldout", type=int, default=48)
    parser.add_argument(
        "--coco-heldout-skip",
        type=int,
        nargs="*",
        default=[],
        help="COCO val image ids already used by the reward gate",
    )
    parser.add_argument("--spatialedit-shards", type=int, default=10)
    parser.add_argument("--bench", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    root = (args.data_root or default_data_root()) / "object_move"
    manifests = repo_root() / "manifests" / "object_move"
    train = coco_rows(
        args.coco_train_annotations, "train2017", root / "coco_train2017", args.coco_train, set()
    )
    train += spatialedit_rows(range(args.spatialedit_shards), root / "spatialedit_500k")
    random.Random(args.seed).shuffle(train)
    heldout_coco = coco_rows(
        args.coco_val_annotations,
        "val2017",
        root / "coco_val2017",
        args.coco_heldout,
        set(args.coco_heldout_skip),
    )
    heldout_bench = bench_rows(root / "spatialedit_bench", args.bench)
    _write(manifests / "train.jsonl", train)
    _write(manifests / "heldout_coco.jsonl", heldout_coco)
    _write(manifests / "heldout_bench.jsonl", heldout_bench)
    emit(
        {
            "train": len(train),
            "train_coco": sum(r["metadata"]["source"] == "coco_train2017" for r in train),
            "train_spatialedit": sum(r["metadata"]["source"] == "spatialedit_500k" for r in train),
            "heldout_coco": len(heldout_coco),
            "heldout_bench": len(heldout_bench),
            "data_root": str(root),
        }
    )


if __name__ == "__main__":
    main()
