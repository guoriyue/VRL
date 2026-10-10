"""Remove fragments and floating haze from RORD foreground layers (``refine_rord_sam.py`` output).

The foreground layer's main body is every connected piece (alpha > 0.1) of at least 10% of the
largest one. Every other piece is a fragment, kept only when SAM says it belongs to the body: one
SAM prompt with points on the nearest body piece and on the fragment must give a mask that covers
the fragment and matches body + fragment (``--join`` threshold); pieces under ``--min-piece``
pixels are dropped outright. Semi-transparent pixels more than 4 px from the opaque body are
cleared, alpha under 0.1 is zero, and edge colours are solved against the true background
(F = (photo - (1 - a) * background) / a), so the layer carries no background tint. The background
layer stays the true background: removed pixels are not pasted into it, and the per-frame
recomposition error is recorded instead. Run with the layered_rl interpreter (GPU, no lock).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fragment_stats import components, stats
from PIL import Image
from refine_rord_sam import SAM, Sam2Model, Sam2Processor, iou, spread_points


def grow(mask, radius):
    return (
        F.max_pool2d(
            torch.from_numpy(mask.astype(np.float32))[None, None], 2 * radius + 1, 1, radius
        )[0, 0].numpy()
        > 0
    )


def shrink(mask, radius):
    return ~grow(~mask, radius)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


class Joiner:
    """SAM with the image embedding computed once per frame."""

    def __init__(self, sam, processor):
        self.sam, self.processor, self.embedding, self.inputs = sam, processor, None, None

    def set_image(self, image):
        self.inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            self.embedding = self.sam.get_image_embeddings(self.inputs["pixel_values"])

    def score(self, body, piece, rng):
        """How well one SAM object through points on ``body`` and ``piece`` covers the piece and matches their union."""
        inner = shrink(piece, 1)
        points = spread_points(
            shrink(body, 3) if shrink(body, 3).any() else body, 3, rng
        ) + spread_points(inner if inner.any() else piece, 2, rng)
        union = body | piece
        prompt = self.processor(
            images=None,
            original_sizes=self.inputs["original_sizes"],
            input_points=[[points]],
            input_labels=[[[1] * len(points)]],
            input_boxes=[[bbox(union)]],
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = self.sam(
                image_embeddings=self.embedding,
                input_points=prompt["input_points"],
                input_labels=prompt["input_labels"],
                input_boxes=prompt["input_boxes"],
                multimask_output=True,
            )
        masks = (
            self.processor.post_process_masks(
                out.pred_masks.cpu(), self.inputs["original_sizes"].cpu()
            )[0][0].numpy()
            > 0.5
        )
        return max(float((m & piece).sum() / piece.sum()) * iou(m, union) for m in masks)


def clean(alpha, photo, background, joiner, rng, *, join, min_piece):
    """Decide on opaque pieces (alpha > 0.5, the same definition ``fragment_stats`` counts), then keep
    soft pixels only on the rim of what survived."""
    alpha = np.where(alpha < 0.1, 0.0, alpha)
    opaque = alpha > 0.5
    if not opaque.any():
        return np.zeros_like(alpha), photo, {"pieces": 0, "kept": 0, "dropped": 0}
    labels = components(opaque)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    body_ids = ids[counts >= 0.1 * counts.max()]
    keep = np.isin(labels, body_ids)
    body_parts = [labels == i for i in body_ids]
    pieces = [(i, c) for i, c in zip(ids, counts, strict=True) if i not in set(body_ids.tolist())]
    kept = dropped = 0
    if any(c >= min_piece for _, c in pieces):
        joiner.set_image(Image.fromarray((photo * 255).round().astype(np.uint8)))
    centres = [np.argwhere(b).mean(0) for b in body_parts]
    for i, c in pieces:
        piece = labels == i
        if c < min_piece:
            dropped += 1
            continue
        centre = np.argwhere(piece).mean(0)
        nearest = body_parts[int(np.argmin([np.abs(k - centre).sum() for k in centres]))]
        if joiner.score(nearest, piece, rng) >= join:
            keep |= piece
            kept += 1
        else:
            dropped += 1
    # soft pixels survive only within 4 px of a kept opaque pixel: no floating haze, no orphaned rims
    alpha = np.where(keep | ((alpha <= 0.5) & grow(keep, 4)), alpha, 0.0)
    soft = (alpha > 0) & (alpha < 1)
    colour = photo.copy()
    a = alpha[..., None]
    colour[soft] = np.clip((photo - (1 - a) * background) / np.maximum(a, 1e-3), 0, 1)[soft]
    return (
        alpha.astype(np.float32),
        colour,
        {"pieces": len(pieces), "kept": kept, "dropped": dropped},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--join", type=float, default=0.6)
    parser.add_argument("--min-piece", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--names", nargs="*", default=None, help="split/name entries to process (default: all)"
    )
    args = parser.parse_args()
    sam = Sam2Model.from_pretrained(SAM, local_files_only=Path(SAM).exists()).eval().cuda()
    joiner = Joiner(sam, Sam2Processor.from_pretrained(SAM, local_files_only=Path(SAM).exists()))
    rng = np.random.default_rng(0)
    frames = (
        [args.src / n for n in args.names]
        if args.names
        else sorted(p.parent for p in args.src.glob("*/*/layer_1.png"))
    )
    rows = []
    for scene in frames[: args.limit]:
        photo = np.asarray(Image.open(scene / "source.png").convert("RGB"), dtype=np.float32) / 255
        background = (
            np.asarray(Image.open(scene / "layer_0.png").convert("RGB"), dtype=np.float32) / 255
        )
        alpha = (
            np.asarray(Image.open(scene / "layer_1.png").convert("RGBA"))[..., 3].astype(
                np.float32
            )
            / 255
        )
        before = stats(alpha)
        new_alpha, colour, info = clean(
            alpha, photo, background, joiner, rng, join=args.join, min_piece=args.min_piece
        )
        a = new_alpha[..., None]
        error = np.abs(colour * a + background * (1 - a) - photo).mean(-1)
        after = stats(new_alpha)
        row = {
            "name": scene.name,
            "split": scene.parent.name,
            **info,
            "fragments_before": before["fragments"],
            "fragments_after": after["fragments"],
            "haze_before": before["haze_px"],
            "haze_after": after["haze_px"],
            "removed_px": int(((alpha > 0.1) & (new_alpha == 0)).sum()),
            "recompose_error": float(error.mean()),
            "recompose_error_removed": float(error[(alpha > 0.1) & (new_alpha == 0)].mean())
            if ((alpha > 0.1) & (new_alpha == 0)).any()
            else 0.0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        out = args.out / scene.parent.name / scene.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("source.png", "layer_0.png", "foreground_mask.png"):
            shutil.copy(scene / name, out / name)
        Image.fromarray(
            np.dstack(
                [
                    (colour * 255).round().astype(np.uint8),
                    (new_alpha * 255).round().astype(np.uint8),
                ]
            ),
            "RGBA",
        ).save(out / "layer_1.png")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "clean_card.json").write_text(
        json.dumps(
            {
                "source": str(args.src),
                "join": args.join,
                "min_piece": args.min_piece,
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
