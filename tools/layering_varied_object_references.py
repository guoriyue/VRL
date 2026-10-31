"""Build candidate controlled 2--5 object scenes from visually reviewed cutouts.

These collages have exact synthetic layer truth, not measured hidden real-world
content. They are separate from the running paired annotation experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.layering_natural_references import REVIEWED, place
from vrl.rewards.models.layered_grouping import LayerStack

SOURCES = {
    **REVIEWED["train"],
    "00022388": "birdhouse",
    "00070159": "pendant",
    "00106164": "portable lamp",
    "00106349": "toy diver",
}
ROOT = Path("/mnt/nvme/layering_varied_20261025")
ASSETS = Path("/mnt/nvme/layering_data/reveallayer/train_probe640")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--layout", choices=["mixed", "same_category"], default="mixed")
    parser.add_argument(
        "--sources", type=Path, help="Reviewed source IDs, names and layer indices"
    )
    parser.add_argument("--variants-per-count", type=int, default=12)
    args = parser.parse_args()
    output = args.output
    sources = (
        json.loads(args.sources.read_text())
        if args.sources
        else {key: {"name": name} for key, name in SOURCES.items()}
    )
    assert not set(sources) & set(REVIEWED["test"])
    if args.variants_per_count < 1:
        raise ValueError("variants-per-count must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261026)
    assets, hashes = [], {}
    for source_id, spec in sources.items():
        paths = sorted((ASSETS / source_id).glob("layer_*.png"))[1:]
        if "layers" in spec:
            paths = [ASSETS / source_id / f"layer_{index}.png" for index in spec["layers"]]
        if not paths:
            raise ValueError(f"No reviewed foreground layers for {source_id}")
        for path in paths:
            if path.name == "layer_0.png":
                raise ValueError("Background cannot be a foreground asset")
            assets.append((source_id, spec["name"], Image.open(path).convert("RGBA")))
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, records = [], []
    for count in range(2, 6):
        for variant in range(args.variants_per_count):
            selected = [assets[i] for i in rng.choice(len(assets), count, replace=False)]
            if args.layout == "same_category":
                source_id = list(sources)[variant % len(sources)]
                pool = [asset for asset in assets if asset[0] == source_id]
                selected = [
                    pool[i] for i in rng.choice(len(pool), count, replace=count > len(pool))
                ]
            base = np.ones((640, 640, 4), dtype=np.float32)
            base[..., :3] = (
                rng.uniform(0.65, 0.85)
                if args.layout == "same_category"
                else rng.uniform(0.25, 0.8, 3)
            )
            for _attempt in range(1000):
                layers = [base]
                for _, _, im in selected:
                    layers.append(
                        place(
                            im,
                            (rng.integers(160, 480), rng.integers(200, 440)),
                            int(rng.integers(170, 260)),
                            floor_y=520 if args.layout == "same_category" else None,
                        )
                    )
                # Quantize before deriving all truth so stored layers and masks agree.
                rgba = np.rint(np.stack(layers) * 255).astype(np.uint8)
                stack = LayerStack.from_rgba(rgba.astype(np.float32) / 255)
                full_area = (rgba[1:, ..., 3] >= 128).sum(axis=(1, 2))
                visible = stack.regions[1:].sum(axis=(1, 2))
                hidden = 1 - visible / full_area
                if visible.min() >= 512 and hidden.max() <= 0.7 and hidden.max() >= 0.1:
                    break
            else:
                raise RuntimeError("Could not construct a visible, partially occluded scene")
            scene_id = f"varied_{count}_{variant:02d}"
            scene = output / "scenes" / scene_id
            scene.mkdir(parents=True, exist_ok=True)
            for index, layer in enumerate(rgba):
                Image.fromarray(layer).save(scene / f"layer_{index}.png")
            Image.fromarray(np.rint(stack.composite() * 255).astype(np.uint8)).save(
                scene / "source.png"
            )
            hidden_pixels = np.any((rgba[1:, ..., 3] >= 128) & ~stack.regions[1:], axis=0)
            Image.fromarray(hidden_pixels.astype(np.uint8) * 255).save(
                scene / "inferable_mask.png"
            )
            np.savez_compressed(scene / "instances.npz", masks=stack.regions[1:])
            metadata = {
                "source": "controlled_varied_reviewed_cutouts",
                "source_ids": [s[0] for s in selected],
                "objects": [s[1] for s in selected],
                "instances": str(scene / "instances.npz"),
                "truth_dir": str(scene),
                "complete_layer_truth": True,
            }
            rows.append(
                {
                    "prompt": "a clean, well composed image",
                    "reference_image": str(scene / "source.png"),
                    "request_overrides": {"layers": count + 3},
                    "metadata": metadata,
                }
            )
            records.append(
                {
                    "id": scene_id,
                    "scene_dir": str(scene),
                    "objects": count,
                    "source_ids": metadata["source_ids"],
                    "hidden_fraction": hidden.tolist(),
                    "visible_pixels": visible.tolist(),
                }
            )
    (output / "candidate_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for page in range((len(rows) + 15) // 16):
        canvas = Image.new("RGB", (1280, 1360), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(rows[page * 16 : (page + 1) * 16]):
            im = Image.open(row["reference_image"]).resize((320, 320))
            x, y = (index % 4) * 320, (index // 4) * 340
            canvas.paste(im, (x, y))
            draw.text((x + 4, y + 322), Path(row["reference_image"]).parent.name, fill="black")
        canvas.save(output / f"review_{page}.png")
    (output / "dataset_audit.json").write_text(
        json.dumps(
            {
                "candidate_scenes": len(rows),
                "source_images": len(sources),
                "cutouts": len(assets),
                "active_training_changed": False,
                "layout": args.layout,
                "reviewed_sources": sources,
                "source_disjoint_from_existing_test": True,
                "sha256": hashes,
                "scenes": records,
                "limitations": "Controlled collages on flat backgrounds. Generated/inpainted source cutouts; no physical hidden truth, no rerendered lighting or shadows, and inconsistent physical scales. Toy assets deliberately broaden the controlled domain. Not a natural-photo benchmark. Candidate data requires reward audit and visual review before use.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Prepared {len(rows)} candidate scenes using {len(assets)} cutouts from {len(sources)} sources"
    )


if __name__ == "__main__":
    main()
