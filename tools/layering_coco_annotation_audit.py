"""Recover filtered COCO thing masks without altering an active experiment.

COCO categories are not exhaustive object labels, and panoptic masks provide
visible support only. Crowd regions remain unscored. Expanded-mask evaluation
uses the existing generated images and reports inadequate layer budgets.
"""

import argparse
import copy
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def dense_masks(archive, annotation, categories):
    image = Image.open(
        io.BytesIO(archive.read("panoptic_val2017/" + annotation["file_name"]))
    ).convert("RGB")
    side = min(image.size)
    x, y = (image.width - side) // 2, (image.height - side) // 2
    rgb = np.asarray(
        image.crop((x, y, x + side, y + side)).resize((640, 640), Image.Resampling.NEAREST),
        dtype=np.uint32,
    )
    labels = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    segments = []
    for item in annotation["segments_info"]:
        category = categories[item["category_id"]]
        if not category["isthing"]:
            continue
        mask = labels == item["id"]
        area = int(mask.sum())
        if area:
            segments.append(
                {
                    "id": item["id"],
                    "name": category["name"],
                    "iscrowd": item["iscrowd"],
                    "area": area,
                    "mask": mask,
                }
            )
    return segments


def process(row, segments, out, *, minimum):
    old = np.load(row["metadata"]["instances"])["masks"].astype(bool)
    matches = []
    for mask in old:
        ious = [float((mask & s["mask"]).sum() / (mask | s["mask"]).sum()) for s in segments]
        best = int(np.argmax(ious))
        matches.append(
            {
                "segment_id": segments[best]["id"],
                "iou": ious[best],
                "iscrowd": segments[best]["iscrowd"],
            }
        )
    aligned = min(m["iou"] for m in matches) >= 0.99
    selected = [s for s in segments if not s["iscrowd"] and s["area"] >= minimum]
    legacy_ids = {m["segment_id"] for m in matches}
    added = [
        {k: v for k, v in s.items() if k != "mask"} for s in selected if s["id"] not in legacy_ids
    ]
    report = {
        "coco_id": row["metadata"]["coco_id"],
        "legacy_objects": len(old),
        "dense_objects": len(selected),
        "geometry_aligned": aligned,
        "minimum_legacy_iou": min(m["iou"] for m in matches),
        "legacy_matches": matches,
        "added": added,
        "counts_by_pixel_threshold": {
            str(t): sum(s["area"] >= t and not s["iscrowd"] for s in segments)
            for t in [1, 64, 256, 6144]
        },
        "current_layers": row.get("layers", row.get("request_overrides", {}).get("layers")),
    }
    report["slots_shortfall"] = max(0, len(selected) + 1 - report["current_layers"])
    if not aligned or not selected:
        return None, report
    path = out / "instances" / f"{row['metadata']['coco_id']:012d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        masks=np.stack([s["mask"] for s in selected]),
        names=np.array([s["name"] for s in selected]),
        segment_ids=np.array([s["id"] for s in selected]),
    )
    new = copy.deepcopy(row)
    new["metadata"].update(
        instances=str(path),
        objects=[s["name"] for s in selected],
        complete_layer_truth=False,
        annotation_version="coco-panoptic-visible-things-256px-v1",
        minimum_object_pixels=minimum,
    )
    if "request_overrides" in new:
        new["request_overrides"]["layers"] = max(report["current_layers"], len(selected) + 1)
    return new, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--training", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minimum-pixels", type=int, default=256)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.annotations / "panoptic_val2017.json").read_text())
    annotations = {a["image_id"]: a for a in meta["annotations"]}
    categories = {c["id"]: c for c in meta["categories"]}
    batches = {
        "heldout": [
            r
            for r in map(json.loads, args.heldout.read_text().splitlines())
            if r["id"].startswith("coco")
        ]
    }
    if args.training:
        batches["training"] = [
            r
            for r in map(json.loads, args.training.read_text().splitlines())
            if r["metadata"].get("source") == "coco_val2017"
        ]
    records = {}
    with zipfile.ZipFile(args.annotations / "panoptic_val2017.zip") as archive:
        for split, rows in batches.items():
            admitted, reports = [], []
            for row in rows:
                segments = dense_masks(
                    archive, annotations[row["metadata"]["coco_id"]], categories
                )
                new, report = process(row, segments, args.output, minimum=args.minimum_pixels)
                if "id" in row:
                    report["id"] = row["id"]
                reports.append(report)
                if new is not None and (
                    split == "heldout" or new["request_overrides"]["layers"] <= 11
                ):
                    admitted.append(new)
            (args.output / f"{split}_expanded.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in admitted)
            )
            records[split] = {
                "input_rows": len(rows),
                "admitted_rows": len(admitted),
                "unaligned_rows": sum(not r["geometry_aligned"] for r in reports),
                "legacy_objects": sum(r["legacy_objects"] for r in reports),
                "dense_objects": sum(r["dense_objects"] for r in reports),
                "added_objects": sum(len(r["added"]) for r in reports),
                "rows_with_insufficient_existing_slots": sum(
                    r["slots_shortfall"] > 0 for r in reports
                ),
                "rows": reports,
            }
    (args.output / "annotation_audit.json").write_text(
        json.dumps(
            {
                "minimum_pixels": args.minimum_pixels,
                "splits": records,
                "active_training_changed": False,
                "scope": "Supplementary visible thing-instance annotations from the official COCO archive. Not amodal truth, not exhaustive beyond COCO categories; crowds excluded. Primary experiment gates remain unchanged. Training candidates capped at 11 layers and not admitted to the running pilot.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "rows"} for k, v in records.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
