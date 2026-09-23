"""Object-move edit manifests for Qwen-Image-2.1 GRPO (``object_move`` reward).

Three sources, each converted into prompt-manifest rows that carry the source
photo as ``reference_image`` and the reward's ``metadata.object_move`` spec:

- ``coco``: real photos from COCO 2017 with ground-truth boxes. The instruction
  puts a thing a person could pick up onto a piece of furniture that is really
  in the photo ("Move the laptop from the table onto the chair"); a row is made
  only when that placement is possible (the rules below), and the photo is
  cropped to a square that keeps both in view.
- ``spatialedit``: SpatialEdit-500K ``object_moving`` shards. The destination is
  a red box drawn on the source ("Move X into the red box ..."); its coordinates
  are read from the red pixels and stored as ``target_box``.
- ``bench``: SpatialEdit-Bench ``move`` items (held out; ``bbox_gt`` is the
  target box).

Placement rules: the object is the only instance of a ``MOVABLE`` category,
0.8-12% of the frame and not in anyone's hands (no person box covers a third
of it); the support is the only instance of a ``SUPPORTS``
category, >= 8% of the frame, and the object is not already resting on it (its
bottom centre is >= 8% of the frame away from the support's box); an
object-sized spot on the support -- its bottom edge 40% of the way down the
support's box, where a table top or seat is -- is free of every other
annotated thing, people included. Plausibility beyond boxes (is that a seat
or a backrest?) is not decidable from annotations; a VLM check filters rows
afterwards (SPRINT_qwen21_object_move_edit_rl section 14).

Binary images are written under ``data/external/object_move`` (or
``VRL_DATA_ROOT``); manifests hold paths relative to that root.
"""

from __future__ import annotations

import argparse
import io
import json
import math
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
MOVABLE = frozenset(
    (
        "backpack",
        "umbrella",
        "handbag",
        "suitcase",
        "frisbee",
        "sports ball",
        "baseball glove",
        "skateboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "donut",
        "cake",
        "pizza",
        "potted plant",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "book",
        "vase",
        "scissors",
        "teddy bear",
        "cat",
        "dog",
        "hair drier",
        "toothbrush",
    )
)
SUPPORTS = frozenset(("dining table", "chair", "couch", "bed", "bench"))
MIN_AREA, MAX_AREA, MIN_SUPPORT_AREA, MIN_GAP, REST_DEPTH = 0.008, 0.12, 0.08, 0.08, 0.4
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


def _offset(box: Sequence[float], dx: float, dy: float) -> tuple[float, float, float, float]:
    return (box[0] - dx, box[1] - dy, box[2] - dx, box[3] - dy)


def _rest_point(box: Sequence[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, box[3])


def _inside(point: tuple[float, float], box: Sequence[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _gap(point: tuple[float, float], box: Sequence[float], size: tuple[float, float]) -> float:
    x, y = point
    return math.hypot(
        max(box[0] - x, 0.0, x - box[2]) / size[0], max(box[1] - y, 0.0, y - box[3]) / size[1]
    )


def landing_spot(
    box: Sequence[float],
    support: Sequence[float],
    blockers: Sequence[Sequence[float]],
    size: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """A free, object-sized spot resting on ``support``, or None when the move is impossible."""

    width, height = size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if not MIN_AREA <= w * h / (width * height) <= MAX_AREA:
        return None
    sx0, sy0, sx1, sy1 = support
    if (sx1 - sx0) * (sy1 - sy0) < MIN_SUPPORT_AREA * width * height:
        return None
    if _gap(_rest_point(box), support, size) < MIN_GAP:
        return None
    rest_y = sy0 + REST_DEPTH * (sy1 - sy0)
    for fraction in (0.5, 0.3, 0.7):
        rest_x = sx0 + fraction * (sx1 - sx0)
        land = (rest_x - w / 2, rest_y - h, rest_x + w / 2, rest_y)
        if land[0] < 0 or land[1] < 0 or land[2] > width or land[3] > height:
            continue
        if all(_overlap(land, other) <= 0.1 * w * h for other in blockers):
            return land
    return None


def coco_rows(annotations: Path, split: str, out: Path, limit: int, skip: set[int]) -> list[dict]:
    """Square-cropped, placement-checked COCO rows (one per image)."""

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
    counts: dict[str, int] = {}
    for ann in anns:
        counts[cats[ann["category_id"]]] = counts.get(cats[ann["category_id"]], 0) + 1
    unique = [a for a in anns if not a["iscrowd"] and counts[cats[a["category_id"]]] == 1]
    people = [_xyxy(a["bbox"]) for a in anns if cats[a["category_id"]] == "person"]
    movables = [
        a
        for a in unique
        if cats[a["category_id"]] in MOVABLE
        # Held or carried (a glass in a hand, a board under a skater): not free to set down.
        and all(_overlap(_xyxy(a["bbox"]), p) <= a["bbox"][2] * a["bbox"][3] / 3 for p in people)
    ]
    supports = [a for a in unique if cats[a["category_id"]] in SUPPORTS]
    for ann in movables:
        for sup in supports:
            obj_box, sup_box = _xyxy(ann["bbox"]), _xyxy(sup["bbox"])
            # Square crops holding the object and the whole support.
            lo_x, hi_x = min(obj_box[0], sup_box[0]), max(obj_box[2], sup_box[2])
            lo_y, hi_y = min(obj_box[1], sup_box[1]), max(obj_box[3], sup_box[3])
            if hi_x - lo_x > side or hi_y - lo_y > side:
                continue
            off_x = min(max(0.0, (lo_x + hi_x) / 2 - side / 2), width - side)
            off_y = min(max(0.0, (lo_y + hi_y) / 2 - side / 2), height - side)
            off_x = min(max(off_x, hi_x - side), lo_x)
            off_y = min(max(off_y, hi_y - side), lo_y)
            box, support = _offset(obj_box, off_x, off_y), _offset(sup_box, off_x, off_y)
            blockers = [
                _offset(_xyxy(o["bbox"]), off_x, off_y)
                for o in anns
                if o is not ann and o is not sup
            ]
            land = landing_spot(box, support, blockers, (side, side))
            if land is None:
                continue
            name, support_name = cats[ann["category_id"]], cats[sup["category_id"]]
            # Name where it rests now when that is another unique support.
            resting = next(
                (
                    cats[o["category_id"]]
                    for o in supports
                    if o is not sup
                    and _inside(_rest_point(box), _offset(_xyxy(o["bbox"]), off_x, off_y))
                ),
                None,
            )
            # The crop depends on the pair, so the file name names the pair.
            path = out / f"{info['id']:012d}_{ann['id']}_{sup['id']}.jpg"
            if not path.exists():
                payload = _fetch(COCO_URL.format(split=split, file=info["file_name"]))
                image = Image.open(io.BytesIO(payload)).convert("RGB")
                crop = (off_x, off_y, off_x + side, off_y + side)
                image.crop(tuple(round(v) for v in crop)).save(path, quality=95)
            origin = f" from the {resting}" if resting else ""
            return {
                "prompt": f"Move the {name}{origin} onto the {support_name}. "
                "Keep everything else unchanged.",
                "reference_image": path.relative_to(out.parents[1]).as_posix(),
                "metadata": {
                    "source": f"coco_{split}",
                    "coco_image_id": info["id"],
                    "object_move": {
                        "object": name,
                        "support": support_name,
                        "source_box": [v / side for v in box],
                        "support_box": [v / side for v in support],
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
        args.coco_train_annotations,
        "train2017",
        root / "coco_onto_train2017",
        args.coco_train,
        set(),
    )
    train += spatialedit_rows(range(args.spatialedit_shards), root / "spatialedit_500k")
    random.Random(args.seed).shuffle(train)
    heldout_coco = coco_rows(
        args.coco_val_annotations,
        "val2017",
        root / "coco_onto_val2017",
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
