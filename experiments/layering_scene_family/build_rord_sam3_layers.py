"""Split each RORD foreground into one layer per removed object with SAM 3 (text prompts).

Input: a ``build_rord_layers.py`` tree (source.png, layer_0.png = true background,
foreground_mask.png = RORD's coarse polygon). Per frame:

1. SAM 3 finds instances of the ``FOREGROUND`` words in the photo. An instance is kept
   when 85% of it lies near the polygon and at least 30% of it clearly changed between
   the two shots (unchanged things belong to the room). The same object found under two
   words (mop / broom), or a part of a better mask of the same word, is dropped.
2. SAM 3 finds ``ROOM`` furniture in the true background; inside furniture only
   clearly changed pixels can be foreground.
3. Order, back to front: of two overlapping objects the smaller is in front (a dog on a
   lap, a book in a hand); otherwise the lower bottom edge is nearer. Each pixel belongs
   to the front-most object; every layer keeps only its main connected body, counting
   pixels of objects in front as connecting material (legs cut off by a dog stay).

Writes the same tree with ``layer_1.png`` ... ``layer_K.png`` (hard alpha, photo colour)
and ``layers.json``. Edges are hard and cast shadows are not attached yet, so the
recomposition differs from the photo by roughly 0.3-3%; the per-frame error is recorded.
Run with an interpreter whose transformers provides ``Sam3Model``; GPU, no lock.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from clean_rord_fragments import grow
from fragment_stats import components
from PIL import Image
from refine_rord_sam import difference, iou
from transformers import Sam3Model, Sam3Processor

FOREGROUND = [
    "person",
    "dog",
    "cat",
    "bag",
    "book",
    "bottle",
    "cup",
    "jug",
    "ball",
    "mop",
    "broom",
    "paint roller",
    "tray",
    "box",
    "toy",
    "phone",
    "tablet",
    "umbrella",
]
ROOM = [
    "table",
    "chair",
    "sofa",
    "bed",
    "cabinet",
    "shelf",
    "desk",
    "stool",
    "pouf",
    "bench",
    "plant",
    "lamp",
    "door",
    "picture frame",
    "television",
    "rug",
]


def find(model, processor, image, words, threshold):
    found = []
    for start in range(0, len(words), 6):
        chunk = words[start : start + 6]
        inputs = processor(images=[image] * len(chunk), text=chunk, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        for word, result in zip(chunk, results, strict=True):
            for mask, score in zip(
                result["masks"].cpu().numpy(), result["scores"].cpu().tolist(), strict=True
            ):
                found.append({"label": word, "score": float(score), "mask": mask.astype(bool)})
    return found


def main_body(mask, image_pixels, front):
    """Connected pieces of at least 10% of the largest and 0.02% of the image; objects in front glue pieces together."""
    if not mask.any():
        return mask
    labels = components(mask | (grow(front, 2) & grow(mask, 12)))
    ids, counts = np.unique(labels[mask], return_counts=True)
    keep = ids[(counts >= 0.1 * counts.max()) & (counts >= 0.0002 * image_pixels)]
    return mask & np.isin(labels, keep)


def split(model, processor, photo_image, photo, background, polygon, threshold):
    reach = grow(polygon, 16)
    diff = difference(photo, background, polygon)
    kept = []
    for c in sorted(
        find(model, processor, photo_image, FOREGROUND, threshold), key=lambda c: -c["score"]
    ):
        m = c["mask"]
        if (
            not m.any()
            or (m & reach).sum() < 0.85 * m.sum()
            or ((diff > 0.15) & m).sum() < 0.3 * m.sum()
        ):
            continue
        if any(
            iou(m, o["mask"]) > 0.5
            or (o["label"] == c["label"] and (m & o["mask"]).sum() > 0.8 * m.sum())
            for o in kept
        ):
            continue
        kept.append(c)
    room = np.zeros(polygon.shape, bool)
    for c in find(
        model,
        processor,
        Image.fromarray((background * 255).round().astype(np.uint8)),
        ROOM,
        threshold,
    ):
        if c["mask"].sum() < 0.6 * polygon.size:
            room |= c["mask"]
    for c in kept:
        c["mask"] = c["mask"] & ~(room & (diff < 0.08))
        c["bottom"] = int(np.nonzero(c["mask"].any(1))[0].max()) if c["mask"].any() else 0
    order = sorted(kept, key=lambda c: c["bottom"])
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i]["mask"], order[j]["mask"]
                if (a & b).sum() > 0.02 * min(a.sum(), b.sum()) and a.sum() < b.sum():
                    order[i], order[j] = order[j], order[i]
                    swapped = True
    taken = np.zeros(polygon.shape, bool)
    layers = []
    for c in reversed(order):  # front first claims its pixels
        m = main_body(c["mask"] & ~taken, polygon.size, taken)
        if m.sum() < 0.0003 * polygon.size:
            continue
        taken |= m
        layers.append({"label": c["label"], "score": c["score"], "mask": m})
    return layers[::-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sam3",
        default=os.environ.get("SAM3_PATH", "facebook/sam3"),
        help="SAM 3 directory or hub id",
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument(
        "--names", nargs="*", default=None, help="split/name entries (default: every frame)"
    )
    args = parser.parse_args()
    local = Path(args.sam3).exists()
    model = Sam3Model.from_pretrained(args.sam3, local_files_only=local).eval().cuda()
    processor = Sam3Processor.from_pretrained(args.sam3, local_files_only=local)
    frames = (
        [args.src / n for n in args.names]
        if args.names
        else sorted(p.parent for p in args.src.glob("*/*/source.png"))
    )
    rows = []
    for scene in frames:
        photo_image = Image.open(scene / "source.png").convert("RGB")
        photo = np.asarray(photo_image, dtype=np.float32) / 255
        background = (
            np.asarray(Image.open(scene / "layer_0.png").convert("RGB"), dtype=np.float32) / 255
        )
        polygon = np.asarray(Image.open(scene / "foreground_mask.png").convert("L")) > 127
        layers = split(model, processor, photo_image, photo, background, polygon, args.threshold)
        out = args.out / scene.parent.name / scene.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("source.png", "layer_0.png", "foreground_mask.png"):
            shutil.copy(scene / name, out / name)
        composite = background.copy()
        rgb = (photo * 255).round().astype(np.uint8)
        for k, layer in enumerate(layers, 1):
            alpha = layer["mask"]
            Image.fromarray(np.dstack([rgb, alpha.astype(np.uint8) * 255]), "RGBA").save(
                out / f"layer_{k}.png"
            )
            composite = np.where(alpha[..., None], photo, composite)
        row = {
            "name": scene.name,
            "split": scene.parent.name,
            "layers": ["room"] + [layer["label"] for layer in layers],
            "scores": [round(layer["score"], 3) for layer in layers],
            "recompose_error": float(np.abs(composite - photo).mean()),
        }
        (out / "layers.json").write_text(json.dumps(row, indent=1) + "\n")
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sam3_card.json").write_text(
        json.dumps(
            {
                "source": str(args.src),
                "sam3": args.sam3,
                "threshold": args.threshold,
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
