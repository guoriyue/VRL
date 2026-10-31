"""Build source-disjoint, photo-derived occlusion references from reviewed cutouts.

These are controlled composites, not unmodified natural photographs or measured
physical ground truth. Never mix dataset benchmark assets into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vrl.rewards.models.layered_grouping import LayerStack

REVIEWED = {
    "train": {
        "00002781": "brown shoes",
        "00013620": "bowls",
        "00047170": "light bulbs",
        "00093378": "table lamps",
    },
    "test": {"00014511": "white shoes", "00062920": "easels", "00109515": "earrings"},
}


def place(image, center, size, *, floor_y=None):
    alpha = np.asarray(image)[..., 3]
    yy, xx = np.where(alpha > 16)
    crop = image.crop((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (640, 640))
    x, y = int(center[0] - crop.width / 2), int(center[1] - crop.height / 2)
    if floor_y is not None:
        y = floor_y - crop.height
    assert x >= 0 and y >= 0 and x + crop.width <= 640 and y + crop.height <= 640
    out.paste(crop, (x, y))
    return np.asarray(out, dtype=np.float32) / 255


def build_pair(foreground, target, rng, *, floor_y=None):
    size = int(rng.integers(290, 360))
    back = place(foreground[0], (280, 320), size, floor_y=floor_y)
    solid = back[..., 3] >= 0.5
    best = None
    for _ in range(240):
        center = (int(rng.integers(190, 450)), int(rng.integers(210, 430)))
        front = place(foreground[1], center, size, floor_y=floor_y)
        fraction = float((solid & (front[..., 3] >= 0.5)).sum() / solid.sum())
        error = abs(fraction - target)
        if best is None or error < best[0]:
            best = (error, front, fraction)
        if error < 0.015:
            break
    assert best[0] < 0.075, (target, best[2])
    return back, best[1], best[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not set(REVIEWED["train"]) & set(REVIEWED["test"])
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261025)
    records, hashes = [], {}
    train_rows, eval_rows = [], []
    for split, groups in REVIEWED.items():
        for source_id, description in groups.items():
            root = args.assets / "train_probe640" / source_id
            images = [Image.open(root / f"layer_{k}.png").convert("RGBA") for k in range(3)]
            for path in [root / "source.png", *sorted(root.glob("layer_*.png"))]:
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            # All selected backgrounds and individual layers were visually reviewed.
            # Fully recomposite the input: the generated background is a declared
            # reference, not an assertion about unseen physical content.
            targets = (
                [0.0, 0.25, 0.45, 0.60, 0.25, 0.45] if split == "train" else [0.0, 0.20, 0.40]
            )
            for variant, target in enumerate(targets):
                foreground = images[1:] if variant % 2 == 0 else images[:0:-1]
                back, front, hidden_fraction = build_pair(
                    foreground,
                    target,
                    rng,
                    floor_y=560 if source_id in {"00093378", "00062920"} else None,
                )
                rgba = np.stack([np.asarray(images[0], dtype=np.float32) / 255, back, front])
                truth = LayerStack.from_rgba(rgba)
                assert min(mask.sum() for mask in truth.regions[1:]) > 512
                scene_id = f"natural_{source_id}_{variant:02d}"
                scene = args.output / split / scene_id
                scene.mkdir(parents=True, exist_ok=True)
                for k, layer in enumerate(rgba):
                    Image.fromarray(np.rint(layer * 255).astype(np.uint8)).save(
                        scene / f"layer_{k}.png"
                    )
                Image.fromarray(np.rint(truth.composite() * 255).astype(np.uint8)).save(
                    scene / "source.png"
                )
                hidden = (rgba[1, ..., 3] >= 0.5) & ~truth.regions[1]
                Image.fromarray(hidden.astype(np.uint8) * 255).save(scene / "inferable_mask.png")
                np.savez_compressed(scene / "instances.npz", masks=truth.regions[1:])
                metadata = {
                    "source": "natural_controlled",
                    "source_id": source_id,
                    "objects": [description + " object 1", description + " object 2"],
                    "truth_dir": str(scene),
                    "instances": str(scene / "instances.npz"),
                    "complete_layer_truth": True,
                    "reference_images": [str(scene / "source.png")],
                    "controlled_composite": True,
                    "hidden_fraction": hidden_fraction,
                }
                row = {
                    "prompt": "a clean, well composed image",
                    "reference_image": str(scene / "source.png"),
                    "request_overrides": {"layers": 5},
                    "metadata": metadata,
                }
                if split == "train":
                    train_rows.append(row)
                else:
                    # Source-only conditioning prevents truth layers from entering the encoder.
                    input_dir = args.output / "eval_inputs" / scene_id
                    input_dir.mkdir(parents=True, exist_ok=True)
                    (input_dir / "source.png").write_bytes((scene / "source.png").read_bytes())
                    eval_rows.append(
                        {
                            "id": scene_id,
                            "scene_dir": str(input_dir),
                            "layers": 5,
                            "metadata": metadata,
                        }
                    )
                records.append(
                    {
                        "id": scene_id,
                        "split": split,
                        "source_id": source_id,
                        "description": description,
                        "target_hidden_fraction": target,
                        "actual_hidden_fraction": hidden_fraction,
                        "scene_dir": str(scene),
                    }
                )
    for name, rows in [("train.jsonl", train_rows), ("eval_rows.jsonl", eval_rows)]:
        (args.output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.output / "dataset_audit.json").write_text(
        json.dumps(
            {
                "revision": "2183b789899cb0ccff58a504f446cecf19c77103",
                "source_split": "train/train-000000.tar",
                "reviewer": "Assistant visual review of source, background, and each foreground; not independent human annotation.",
                "limitations": "Small controlled composites from generated/inpainted cutouts. Not a natural-photo generalization benchmark. Shadows and physical lighting are not re-rendered after moving objects.",
                "excluded": {
                    "00020864": "Two watch-strap parts: object granularity ambiguous.",
                    "00070159": "Unselected objects remain in background.",
                    "00106164": "Unselected pedestals remain in background.",
                    "00106349": "Toy/illustration domain excluded from this photo-derived pilot.",
                    "00022388": "Unselected tree/background content and cutout extent need review.",
                },
                "source_disjoint": True,
                "reviewed_source_ids": REVIEWED,
                "source_sha256": hashes,
                "train_scenes": len(train_rows),
                "test_scenes": len(eval_rows),
                "scenes": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(train_rows)} training and {len(eval_rows)} source-disjoint test scenes")


if __name__ == "__main__":
    main()
