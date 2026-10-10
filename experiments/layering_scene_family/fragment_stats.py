"""Count fragments in RORD foreground layers.

Per frame, on the foreground layer's alpha: the main body is every connected component (alpha > 0.5)
of at least 10% of the largest one; everything else above 0.5 is a fragment. Floating haze is
alpha in (0.02, 0.5) farther than 4 px from the main body. Prints per-tree summaries and writes
per-frame rows to ``<tree>/fragment_stats.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def components(mask):
    """8-connected component labels: min-label propagation with pointer jumping (converges in a few rounds)."""
    h, w = mask.shape
    big = h * w
    lab = np.where(mask, np.arange(big).reshape(h, w), big).astype(np.int64)
    while True:
        pad = np.pad(lab, 1, constant_values=big)
        neigh = np.min(
            [
                pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ],
            0,
        )
        flat = np.append(np.where(mask, np.minimum(lab, neigh), big).reshape(-1), big)
        for _ in range(8):  # pointer jumping: follow each label to its own label
            flat = flat[flat]
        new = flat[:-1].reshape(h, w)
        if np.array_equal(new, lab):
            break
        lab = new
    return np.where(mask, lab + 1, 0)


def stats(alpha):
    hard = alpha > 0.5
    n = alpha.size
    if not hard.any():
        return {"empty": True, "fragments": 0, "fragment_px": 0, "haze_px": 0}
    lab = components(hard)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    keep = ids[counts >= 0.1 * counts.max()]
    main = np.isin(lab, keep)
    frag = hard & ~main
    near = F.max_pool2d(torch.from_numpy(main).float()[None, None], 9, 1, 4)[0, 0].numpy() > 0
    haze = (alpha > 0.02) & (alpha <= 0.5) & ~near
    return {
        "empty": False,
        "fragments": int((counts < 0.1 * counts.max()).sum()),
        "fragment_px": int(frag.sum()),
        "fragment_share": float(frag.sum() / max(hard.sum(), 1)),
        "haze_px": int(haze.sum()),
        "main_px": int(main.sum()),
        "image_px": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trees", nargs="+", type=Path)
    parser.add_argument("--layer", default="layer_1.png")
    args = parser.parse_args()
    for tree in args.trees:
        rows = []
        for f in sorted(tree.glob(f"*/*/{args.layer}")):
            alpha = np.asarray(Image.open(f).convert("RGBA"))[..., 3].astype(np.float32) / 255
            rows.append({"name": f.parent.name, "split": f.parent.parent.name, **stats(alpha)})
        (tree / "fragment_stats.json").write_text(json.dumps(rows) + "\n")
        live = [r for r in rows if not r["empty"]]
        fr = np.array([r["fragments"] for r in live])
        hz = np.array([r["haze_px"] for r in live])
        fp = np.array([r["fragment_px"] for r in live])
        print(
            json.dumps(
                {
                    "tree": str(tree),
                    "frames": len(rows),
                    "empty": len(rows) - len(live),
                    "frames_with_fragments": f"{(fr > 0).mean():.1%}",
                    "fragments_per_frame_mean": round(float(fr.mean()), 2),
                    "fragments_p90": int(np.percentile(fr, 90)),
                    "fragment_px_mean": round(float(fp.mean()), 1),
                    "frames_with_haze_over_200px": f"{(hz > 200).mean():.1%}",
                    "haze_px_mean": round(float(hz.mean()), 1),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
