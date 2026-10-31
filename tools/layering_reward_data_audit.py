"""Inspect actual mask supervision and layer budgets used by the layering run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def scan_mask(path):
    with np.load(path) as data:
        masks = np.asarray(data["masks"]).astype(bool)
    areas = masks.sum(axis=(1, 2))
    nonempty = areas[areas > 0]
    return {
        "objects": len(nonempty),
        "empty_masks": int((areas == 0).sum()),
        "small_objects_under_64": int(((areas > 0) & (areas < 64)).sum()),
        "areas": nonempty.tolist(),
        "overlap_pixels": int((masks.sum(0) > 1).sum()),
        "labeled_fraction": float(masks.any(0).mean()),
        "shape": list(masks.shape),
    }


def scan_truth(path):
    root = Path(path)
    files = sorted(root.glob("layer_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    visible = []
    front = None
    for p in reversed(files[1:]):
        with Image.open(p) as im:
            solid = np.asarray(im.convert("RGBA"))[..., 3] >= 128
        if front is None:
            front = np.zeros_like(solid)
        visible.append(int((solid & ~front).sum()))
        front |= solid
    infer = root / "inferable_mask.png"
    if infer.exists():
        with Image.open(infer) as im:
            infer_pixels = int((np.asarray(im.convert("L")) > 127).sum())
    else:
        infer_pixels = 0
    return {
        "layers": len(files),
        "visible_objects": sum(x > 0 for x in visible),
        "small_objects_under_64": sum(0 < x < 64 for x in visible),
        "inferable_mask_exists": infer.exists(),
        "inferable_pixels": infer_pixels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    paths = sorted({r["metadata"]["instances"] for r in rows if r["metadata"].get("instances")})
    truths = sorted({r["metadata"]["truth_dir"] for r in rows if r["metadata"].get("truth_dir")})
    with ThreadPoolExecutor(max_workers=8) as pool:
        mask_stats = dict(zip(paths, pool.map(scan_mask, paths), strict=True))
    print("Scanned masks", len(paths), flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        truth_stats = dict(zip(truths, pool.map(scan_truth, truths), strict=True))
    budgets = []
    for i, row in enumerate(rows):
        m = row["metadata"]
        stats = mask_stats.get(m.get("instances")) or truth_stats[m["truth_dir"]]
        n = stats.get("objects", stats.get("visible_objects"))
        layers = row["request_overrides"]["layers"]
        if layers - 1 < n:
            budgets.append(
                {
                    "row": i,
                    "source": m["source"],
                    "image": row["reference_image"],
                    "foreground_slots": layers - 1,
                    "annotated_visible_objects": n,
                }
            )
    report = {
        "rows": len(rows),
        "sources": dict(Counter(r["metadata"]["source"] for r in rows)),
        "unique_instance_files": len(paths),
        "unique_truth_dirs": len(truths),
        "mask_stats": mask_stats,
        "truth_stats": truth_stats,
        "insufficient_layer_budgets": budgets,
        "mask_files_with_objects_under_64": sum(
            s["small_objects_under_64"] > 0 for s in mask_stats.values()
        ),
        "truth_dirs_with_visible_objects_under_64": sum(
            s["small_objects_under_64"] > 0 for s in truth_stats.values()
        ),
        "truth_dirs_without_inferable_pixels": sum(
            s["inferable_pixels"] == 0 for s in truth_stats.values()
        ),
        "mask_files_with_overlap": sum(s["overlap_pixels"] > 0 for s in mask_stats.values()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in report.items()
            if k not in ["mask_stats", "truth_stats", "insufficient_layer_budgets"]
        }
    )
    print("Insufficient budgets", len(budgets))


if __name__ == "__main__":
    main()
