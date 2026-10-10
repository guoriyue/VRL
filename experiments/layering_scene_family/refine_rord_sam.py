"""Refine RORD foreground layers with SAM 2.1 (shape) and ViTMatte (soft edges).

Input: a ``build_rord_layers.py`` tree (source.png, layer_0.png = true
background, foreground_mask.png = RORD's coarse polygon). Per frame:

1. Difference map between photo and background after a per-channel gain/offset
   fit outside the polygon (``build_rord_matte.matte`` style), the evidence of
   what actually changed between the two shots.
2. Each connected polygon component is one SAM prompt: its bounding box,
   positive points on strongly changed pixels, negative points on unchanged
   pixels inside the polygon (floor and wall the polygon over-covers).
   Of SAM's candidate masks the one agreeing best (IoU) with the changed
   pixels wins; SAM's own score is not trusted. Changed pixels the mask
   leaves uncovered (a second object in the same polygon) start up to two
   more rounds.
3. The union, plus strongly changed pixels SAM missed, is clipped to the
   dilated polygon and hole-filled; a narrow trimap around its boundary goes
   to ViTMatte for the soft alpha.

Writes the same tree layout with the refined ``layer_1.png`` and a
``refine_card.json`` with per-frame agreement (``iou_diff``) for filtering.
Run with the layered_rl interpreter; uses the GPU without the lock (small).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from build_rord_matte import fill_holes
from PIL import Image
from transformers import Sam2Model, Sam2Processor, VitMatteForImageMatting, VitMatteImageProcessor


def _local_or_hub(env, cache_glob, hub_id):
    """A model directory: ``$env``, else the local Hugging Face snapshot, else the hub id (downloaded)."""
    return os.environ.get(env) or str(next(Path.home().glob(cache_glob), hub_id))


SAM = _local_or_hub(
    "SAM2_PATH",
    ".cache/huggingface/hub/models--facebook--sam2.1-hiera-large/snapshots/*",
    "facebook/sam2.1-hiera-large",
)
MATTE = _local_or_hub(
    "VITMATTE_PATH",
    ".cache/huggingface/hub/models--hustvl--vitmatte-small-composition-1k/snapshots/*",
    "hustvl/vitmatte-small-composition-1k",
)


def dilate(mask, radius):
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


def erode(mask, radius):
    return 1 - dilate(1 - mask, radius)


def as_t(mask):
    return torch.from_numpy(mask.astype(np.float32))[None, None]


def difference(photo, background, polygon):
    outside = ~polygon
    fitted = background.copy()
    for c in range(3):
        a = np.stack([background[..., c][outside], np.ones(outside.sum())], 1)
        gain, offset = np.linalg.lstsq(a, photo[..., c][outside], rcond=None)[0]
        fitted[..., c] = background[..., c] * gain + offset
    diff = torch.from_numpy(np.abs(photo - fitted)).permute(2, 0, 1)[None]
    return F.avg_pool2d(diff, 5, stride=1, padding=2).amax(1)[0].numpy()


def components(mask, min_pixels):
    """4-connected components of a boolean mask, labelled at 1/4 scale and upsampled."""
    small = F.max_pool2d(as_t(mask), 4)[0, 0].numpy() > 0
    labels = np.zeros(small.shape, np.int32)
    count = 0
    for y, x in zip(*np.nonzero(small), strict=True):
        if labels[y, x]:
            continue
        count += 1
        stack = [(y, x)]
        labels[y, x] = count
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if (
                    0 <= ny < small.shape[0]
                    and 0 <= nx < small.shape[1]
                    and small[ny, nx]
                    and not labels[ny, nx]
                ):
                    labels[ny, nx] = count
                    stack.append((ny, nx))
    full = np.kron(labels, np.ones((4, 4), np.int32))[: mask.shape[0], : mask.shape[1]]
    out = [(full == k) & mask for k in range(1, count + 1)]
    return [c for c in out if c.sum() >= min_pixels]


def spread_points(candidates, count, rng):
    """Greedy farthest-point picks from a boolean candidate map, as (x, y)."""
    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        return []
    pts = np.stack([xs, ys], 1).astype(np.float32)
    picks = [pts[rng.integers(len(pts))]]
    while len(picks) < min(count, len(pts)):
        d = np.min(np.stack([((pts - p) ** 2).sum(1) for p in picks]), 0)
        picks.append(pts[int(d.argmax())])
    return [p.tolist() for p in picks]


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def refine(
    photo_pil,
    photo,
    background,
    polygon,
    sam,
    sam_processor,
    matte_model,
    matte_processor,
    rng,
    *,
    changed=0.07,
    strong=0.15,
    still=0.03,
    grow=15,
):
    diff = difference(photo, background, polygon)
    evidence = (diff > changed) & polygon
    mask = np.zeros(polygon.shape, bool)

    def best_mask(positive, negative, boxes, target, reach):
        """Run SAM once per prompt variant; keep the candidate agreeing best with ``target`` inside ``reach``."""
        candidates = []
        for box_source in boxes:
            kwargs = {
                "input_points": [[positive + negative]],
                "input_labels": [[[1] * len(positive) + [0] * len(negative)]],
            }
            if box_source is not None and box_source.any():
                ys, xs = np.nonzero(box_source)
                kwargs["input_boxes"] = [
                    [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]
                ]
            inputs = sam_processor(images=photo_pil, return_tensors="pt", **kwargs).to("cuda")
            with torch.no_grad():
                out = sam(**inputs, multimask_output=True)
            candidates += list(
                sam_processor.post_process_masks(
                    out.pred_masks.cpu(), inputs["original_sizes"].cpu()
                )[0][0].numpy()
                > 0.5
            )
        return max(candidates, key=lambda c: iou(c & reach, target)) & reach

    min_pixels = int(0.002 * polygon.size)
    for part in components(polygon, min_pixels=int(0.001 * polygon.size)):
        core = erode(as_t(part), 4)[0, 0].numpy() > 0.5
        local = (diff > strong) & part
        reach = dilate(as_t(part), 12)[0, 0].numpy() > 0.5
        positive = spread_points(core & (diff > strong), 6, rng)
        negative = spread_points(core & (diff < still), 3, rng)
        if not positive:
            continue
        # Three prompt variants (polygon box, changed-pixel box, points only); every candidate mask is
        # judged by agreement with the strongly changed pixels, so a textured background cannot win on
        # SAM's own score.
        found = best_mask(positive, negative, (part, local, None), local, reach)
        # One polygon can hold several objects (a person and a dog); SAM returns one. Changed pixels
        # left uncovered start another round on what remains, up to two more objects.
        for _ in range(2):
            left = (
                erode(
                    as_t(evidence & part & ~dilate(as_t(found), 3)[0, 0].numpy().astype(bool)), 3
                )[0, 0].numpy()
                > 0.5
            )
            if left.sum() < min_pixels:
                break
            points = spread_points(left, 4, rng)
            extra = best_mask(points, negative, (left, None), left, reach)
            if iou(extra, left) < 0.2:
                break
            found |= extra
        mask |= found
    # SAM can leave holes or drop a thin part (fingers, a translucent jug); strongly changed pixels fill
    # those in, but only by growing at most ``grow`` pixels out of SAM's mask, so textured background
    # that changed between the two shots (a counter top, a lit wall) cannot join as a large patch.
    # Shadows change the photo only weakly and stay out.
    reach = dilate(as_t(polygon), 12)
    certain = as_t(diff > strong) * reach
    grown = as_t(mask) * reach
    for _ in range(grow):
        grown = torch.maximum(grown, dilate(grown, 1) * certain)
    mask = fill_holes(grown)[0, 0].numpy() > 0.5
    if not mask.any():
        return np.zeros(polygon.shape, np.float32), 0.0
    sure = erode(as_t(mask), 2)[0, 0].numpy() > 0.5
    maybe = dilate(as_t(mask), 4)[0, 0].numpy() > 0.5
    trimap = np.where(sure, 255, np.where(maybe, 128, 0)).astype(np.uint8)
    inputs = matte_processor(
        images=photo_pil, trimaps=Image.fromarray(trimap, "L"), return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        alpha = (
            matte_model(**inputs)
            .alphas[0, 0, : polygon.shape[0], : polygon.shape[1]]
            .float()
            .cpu()
            .numpy()
        )
    alpha = np.where(sure, 1.0, np.where(maybe, alpha, 0.0)).astype(np.float32)
    return alpha, iou(alpha > 0.5, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.5,
        help="frames below this agreement with the changed pixels are dropped",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sam = Sam2Model.from_pretrained(SAM, local_files_only=Path(SAM).exists()).eval().cuda()
    sam_processor = Sam2Processor.from_pretrained(SAM, local_files_only=Path(SAM).exists())
    matte_model = (
        VitMatteForImageMatting.from_pretrained(MATTE, local_files_only=Path(MATTE).exists())
        .eval()
        .cuda()
    )
    matte_processor = VitMatteImageProcessor.from_pretrained(
        MATTE, local_files_only=Path(MATTE).exists()
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for source in sorted(args.src.glob("*/*/source.png"))[: args.limit]:
        scene = source.parent
        photo_pil = Image.open(source).convert("RGB")
        photo = np.asarray(photo_pil, dtype=np.float32) / 255.0
        background = (
            np.asarray(Image.open(scene / "layer_0.png").convert("RGB"), dtype=np.float32) / 255.0
        )
        polygon = np.asarray(Image.open(scene / "foreground_mask.png").convert("L")) > 127
        alpha, agreement = refine(
            photo_pil,
            photo,
            background,
            polygon,
            sam,
            sam_processor,
            matte_model,
            matte_processor,
            rng,
        )
        row = {
            "name": scene.name,
            "split": scene.parent.name,
            "polygon_fraction": float(polygon.mean()),
            "alpha_fraction": float((alpha > 0.5).mean()),
            "iou_diff": agreement,
            "kept": agreement >= args.min_iou,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not row["kept"]:
            continue
        out = args.out / scene.parent.name / scene.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("source.png", "layer_0.png", "foreground_mask.png"):
            shutil.copy(scene / name, out / name)
        Image.fromarray(
            np.dstack([np.asarray(photo_pil), (alpha * 255).round().astype(np.uint8)]), "RGBA"
        ).save(out / "layer_1.png")
    args.out.mkdir(parents=True, exist_ok=True)
    card = {
        "source": str(args.src),
        "models": [SAM, MATTE],
        "min_iou": args.min_iou,
        "rows": rows,
        "layers": ["background (gt, opaque)", "foreground (SAM 2.1 mask, ViTMatte edges)"],
        "caveat": "Foreground shape from SAM prompted by RORD's polygon and the photo/gt difference; shadows are excluded; the background layer is real ground truth.",
    }
    (args.out / "refine_card.json").write_text(json.dumps(card, indent=1) + "\n")
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
