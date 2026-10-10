"""Turn indoor RORD triplets into 2-layer known-truth samples on real photographs.

RORD (MIT) films a static camera with and without a person/object: ``img`` has the
foreground, ``gt`` is the same frame without it, ``mask`` marks the foreground as a
coarse polygon (zero = foreground). Each sample is center-cropped to a square and
resized to 640: layer 0 = the clean background (real truth for the occluded
region), layer 1 = the photo inside the polygon. Indoor scenes only (scene id
second token starts with ``I``). Scenes are split by scene id into train and
heldout; a few frames per scene are taken, spread over the clip, keeping frames
whose foreground covers 3%-45% of the crop.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rord", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-per-scene", type=int, default=12)
    parser.add_argument("--heldout-scenes", type=int, default=8)
    parser.add_argument("--heldout-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    scenes = sorted(
        s.name for s in (args.rord / "img").iterdir() if s.name.split("_")[1].startswith("I")
    )
    rng.shuffle(scenes)
    split = {s: ("heldout" if k < args.heldout_scenes else "train") for k, s in enumerate(scenes)}
    rows = []
    for scene in sorted(scenes):
        frames = sorted((args.rord / "img" / scene).glob("*.jpg"))
        want = args.heldout_frames if split[scene] == "heldout" else args.frames_per_scene
        picks, step = [], max(1, len(frames) // (want * 3))
        for f in frames[::step]:
            img = Image.open(f).convert("RGB")
            w, h = img.size
            s = min(w, h)
            box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
            mask = (
                np.asarray(
                    Image.open(args.rord / "mask" / scene / f"{f.stem}.png")
                    .convert("L")
                    .crop(box)
                    .resize((640, 640), Image.NEAREST)
                )
                == 0
            )
            frac = float(mask.mean())
            if 0.03 <= frac <= 0.45:
                picks.append((f, box, frac))
        if len(picks) > want:
            idx = np.linspace(0, len(picks) - 1, want).round().astype(int)
            picks = [picks[i] for i in idx]
        for f, box, frac in picks:
            name = f"{scene}_{f.stem}"
            d = args.out / split[scene] / name
            d.mkdir(parents=True, exist_ok=True)
            img = Image.open(f).convert("RGB").crop(box).resize((640, 640), Image.LANCZOS)
            gt = (
                Image.open(args.rord / "gt" / scene / f.name)
                .convert("RGB")
                .crop(box)
                .resize((640, 640), Image.LANCZOS)
            )
            mask = (
                np.asarray(
                    Image.open(args.rord / "mask" / scene / f"{f.stem}.png")
                    .convert("L")
                    .crop(box)
                    .resize((640, 640), Image.NEAREST)
                )
                == 0
            )
            img.save(d / "source.png")
            Image.fromarray(
                np.dstack([np.asarray(gt), np.full((640, 640), 255, np.uint8)]), "RGBA"
            ).save(d / "layer_0.png")
            Image.fromarray(
                np.dstack([np.asarray(img), (mask * 255).astype(np.uint8)]), "RGBA"
            ).save(d / "layer_1.png")
            Image.fromarray((mask * 255).astype(np.uint8), "L").save(d / "foreground_mask.png")
            rows.append(
                {
                    "name": name,
                    "scene": scene,
                    "split": split[scene],
                    "frame": f.name,
                    "foreground_fraction": round(frac, 4),
                }
            )
    card = {
        "source": "RORD test split (MIT), indoor scenes, HF mirror JiaHuang01/RORD test_RORD.zip",
        "rows": rows,
        "train_scenes": sorted(s for s in scenes if split[s] == "train"),
        "heldout_scenes": sorted(s for s in scenes if split[s] == "heldout"),
        "layers": ["background (gt, opaque)", "foreground (photo inside coarse polygon)"],
        "caveat": "Foreground polygons are coarse; foreground-layer alpha is not a matte. The background layer is real ground truth.",
    }
    (args.out / "rord_card.json").write_text(json.dumps(card, indent=1) + "\n")
    print(
        json.dumps(
            {
                "train": sum(r["split"] == "train" for r in rows),
                "heldout": sum(r["split"] == "heldout" for r in rows),
                "scenes": len(scenes),
            }
        )
    )


if __name__ == "__main__":
    main()
