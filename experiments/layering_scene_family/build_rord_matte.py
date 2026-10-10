"""Replace the coarse RORD foreground polygon with a difference matte.

``build_rord_layers.py`` cuts the foreground layer with RORD's coarse polygon,
which drags large patches of floor and wall into the object layer; training on
it teaches object layers that carry room chunks. RORD photo and ``gt`` share a
static camera, so the foreground is where they differ: after a per-channel
gain/offset fit of ``gt`` to the photo outside the polygon (exposure drift),
pixels whose blurred difference exceeds ``--threshold`` inside the slightly
dilated polygon are foreground; the mask is opened, closed, hole-filled and
feathered. Layer 0 (true background) and ``foreground_mask.png`` (the polygon,
the evaluation region) are copied unchanged, so ``background_recovery`` stays
comparable. Frames whose matte keeps under ``--min-keep`` of the polygon are
dropped. Run with the layered_rl interpreter (CPU only).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def dilate(mask, radius):
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


def erode(mask, radius):
    return 1 - dilate(1 - mask, radius)


def fill_holes(mask):
    """Background = complement pixels reachable from the border at 1/4 scale; everything else is foreground."""
    small = (F.avg_pool2d(mask, 4) > 0.5).float()
    free = 1 - small
    reach = torch.zeros_like(small)
    reach[..., 0, :], reach[..., -1, :], reach[..., :, 0], reach[..., :, -1] = 1, 1, 1, 1
    reach = reach * free
    while True:
        grown = dilate(reach, 1) * free
        if torch.equal(grown, reach):
            break
        reach = grown
    holes = F.interpolate(1 - reach - small, scale_factor=4, mode="nearest").clamp(min=0)
    return torch.maximum(mask, holes)


def matte(photo, background, polygon, threshold):
    outside = ~polygon
    fitted = background.copy()
    for c in range(3):
        a = np.stack([background[..., c][outside], np.ones(outside.sum())], 1)
        gain, offset = np.linalg.lstsq(a, photo[..., c][outside], rcond=None)[0]
        fitted[..., c] = background[..., c] * gain + offset
    diff = torch.from_numpy(np.abs(photo - fitted)).permute(2, 0, 1)[None]
    diff = F.avg_pool2d(diff, 5, stride=1, padding=2).amax(1, keepdim=True)
    region = dilate(torch.from_numpy(polygon).float()[None, None], 8)
    mask = (diff > threshold).float() * region
    mask = dilate(erode(mask, 2), 2)
    mask = erode(dilate(mask, 6), 6)
    mask = fill_holes(mask)
    alpha = F.avg_pool2d(mask, 5, stride=1, padding=2)
    return alpha[0, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, required=True, help="rord_layers built by build_rord_layers.py"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--min-keep", type=float, default=0.15)
    args = parser.parse_args()
    rows = []
    for scene in sorted(args.src.glob("*/*/source.png")):
        scene = scene.parent
        photo = (
            np.asarray(Image.open(scene / "source.png").convert("RGB"), dtype=np.float32) / 255.0
        )
        background = (
            np.asarray(Image.open(scene / "layer_0.png").convert("RGB"), dtype=np.float32) / 255.0
        )
        polygon = np.asarray(Image.open(scene / "foreground_mask.png").convert("L")) > 127
        alpha = matte(photo, background, polygon, args.threshold)
        keep = float((alpha > 0.5).sum() / max(polygon.sum(), 1))
        row = {
            "name": scene.name,
            "split": scene.parent.name,
            "polygon_fraction": float(polygon.mean()),
            "matte_fraction": float((alpha > 0.5).mean()),
            "matte_over_polygon": keep,
            "kept": keep >= args.min_keep,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not row["kept"]:
            continue
        out = args.out / scene.parent.name / scene.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("source.png", "layer_0.png", "foreground_mask.png"):
            shutil.copy(scene / name, out / name)
        rgb = (photo * 255).round().astype(np.uint8)
        Image.fromarray(np.dstack([rgb, (alpha * 255).round().astype(np.uint8)]), "RGBA").save(
            out / "layer_1.png"
        )
    card = {
        "source": str(args.src),
        "threshold": args.threshold,
        "min_keep": args.min_keep,
        "rows": rows,
        "layers": ["background (gt, opaque)", "foreground (photo under the difference matte)"],
        "caveat": "Difference matte: includes cast shadows and any change between the two shots; the background layer is real ground truth.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rord_card.json").write_text(json.dumps(card, indent=1) + "\n")
    print(
        json.dumps(
            {
                split: sum(r["kept"] and r["split"] == split for r in rows)
                for split in ("train", "heldout")
            }
        )
    )


if __name__ == "__main__":
    main()
