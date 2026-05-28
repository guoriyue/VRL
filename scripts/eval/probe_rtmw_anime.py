"""Probe RTMW whole-body pose on anime images — no vrl imports, standalone.

Usage
-----
# Install deps first (one-time):
#   pip install rtmlib opencv-python pillow

python scripts/eval/probe_rtmw_anime.py --images path/to/dir_or_glob/*.png
python scripts/eval/probe_rtmw_anime.py --images a.png b.png --out results/rtmw

What this shows
---------------
For each image:
  - Overlay of detected whole-body keypoints (body + hands + face)
  - Per-image numeric breakdown:
      body_coverage   : fraction of 17 body kps with conf > threshold
      hand_coverage   : fraction of 42 hand kps with conf > threshold
      mean_conf       : overall mean confidence
      bone_ratio_err  : crude limb length symmetry violation (0=perfect)
      visible_hands   : how many hands were detected at all

The goal is NOT to benchmark RTMW — it is to see with your own eyes how
well it fires on anime figures before committing to building a reward.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg in ("rtmlib", "cv2", "PIL", "numpy"):
        mod = "cv2" if pkg == "cv2" else pkg
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg if pkg != "PIL" else "Pillow")
    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing).replace('cv2', 'opencv-python')}")
        sys.exit(1)

_check_deps()

import cv2
import numpy as np
from PIL import Image
from rtmlib import Body, Wholebody, draw_skeleton

# ── keypoint index ranges (RTMW 133-kp convention) ───────────────────────────
BODY_KP_RANGE = range(0, 17)      # 17 COCO body keypoints
LHAND_KP_RANGE = range(91, 112)   # 21 left-hand keypoints
RHAND_KP_RANGE = range(112, 133)  # 21 right-hand keypoints
HAND_KP_RANGE = range(91, 133)    # both hands

CONF_THRESHOLD = 0.3


def _kp_coverage(kps: np.ndarray, scores: np.ndarray, kp_range: range) -> float:
    """Fraction of keypoints in range with conf > threshold."""
    if scores is None or len(scores) == 0:
        return 0.0
    idxs = list(kp_range)
    sub = scores[idxs]
    return float((sub > CONF_THRESHOLD).mean())


def _mean_conf(scores: np.ndarray, kp_range: range) -> float:
    if scores is None or len(scores) == 0:
        return 0.0
    return float(scores[list(kp_range)].mean())


def _bone_ratio_err(kps: np.ndarray, scores: np.ndarray) -> float:
    """Crude left/right limb length symmetry error ∈ [0, 1].

    Compares upper-arm and lower-arm lengths on both sides.
    Returns 0 if not enough keypoints are visible.
    """
    # COCO body kp indices: 5=Lshoulder 6=Rshoulder 7=Lelbow 8=Relbow 9=Lwrist 10=Rwrist
    pairs = [(5, 7, 6, 8), (7, 9, 8, 10)]  # (L_start, L_end, R_start, R_end)
    errs = []
    for ls, le, rs, re in pairs:
        if scores[ls] < CONF_THRESHOLD or scores[le] < CONF_THRESHOLD:
            continue
        if scores[rs] < CONF_THRESHOLD or scores[re] < CONF_THRESHOLD:
            continue
        l_len = np.linalg.norm(kps[ls] - kps[le]) + 1e-6
        r_len = np.linalg.norm(kps[rs] - kps[re]) + 1e-6
        errs.append(abs(l_len - r_len) / max(l_len, r_len))
    return float(np.mean(errs)) if errs else 0.0


def probe_image(model: Wholebody, img_bgr: np.ndarray) -> dict:
    """Run RTMW on one image, return metrics dict + annotated image."""
    keypoints, scores = model(img_bgr)  # shape [N_persons, 133, 2] / [N, 133]

    if keypoints is None or len(keypoints) == 0:
        annotated = img_bgr.copy()
        cv2.putText(annotated, "NO PERSON DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return {
            "n_persons": 0,
            "body_coverage": 0.0,
            "hand_coverage": 0.0,
            "mean_conf": 0.0,
            "bone_ratio_err": 0.0,
            "visible_hands": 0,
            "annotated": annotated,
        }

    # Use first (or highest-confidence) person
    idx = int(np.argmax(scores[:, :17].mean(axis=1)))
    kps = keypoints[idx]    # [133, 2]
    sc = scores[idx]        # [133]

    body_cov = _kp_coverage(kps, sc, BODY_KP_RANGE)
    hand_cov = _kp_coverage(kps, sc, HAND_KP_RANGE)
    mean_c = float(sc.mean())
    bone_err = _bone_ratio_err(kps, sc)

    lhand_visible = int(_mean_conf(sc, LHAND_KP_RANGE) > CONF_THRESHOLD)
    rhand_visible = int(_mean_conf(sc, RHAND_KP_RANGE) > CONF_THRESHOLD)

    annotated = draw_skeleton(img_bgr.copy(), keypoints, scores, kpt_thr=CONF_THRESHOLD)

    return {
        "n_persons": len(keypoints),
        "body_coverage": round(body_cov, 3),
        "hand_coverage": round(hand_cov, 3),
        "mean_conf": round(mean_c, 3),
        "bone_ratio_err": round(bone_err, 3),
        "visible_hands": lhand_visible + rhand_visible,
        "annotated": annotated,
    }


def _load_images(paths: list[str]) -> list[tuple[str, np.ndarray]]:
    from glob import glob
    out = []
    for p in paths:
        expanded = glob(p) if "*" in p else [p]
        for f in expanded:
            fp = Path(f)
            if not fp.exists():
                print(f"[warn] not found: {fp}")
                continue
            img = cv2.imread(str(fp))
            if img is None:
                print(f"[warn] cannot read: {fp}")
                continue
            out.append((fp.stem, img))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe RTMW on anime images")
    ap.add_argument("--images", nargs="+", required=True, help="Image files or glob")
    ap.add_argument("--out", default="results/rtmw_probe", help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--backend",
        default="onnxruntime",
        choices=["onnxruntime", "openvino"],
        help="rtmlib inference backend",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RTMW-x wholebody ({args.device}, {args.backend})…")
    print("(rtmlib will auto-download ONNX weights on first run, ~200 MB)\n")
    model = Wholebody(
        pose_model_type="rtmw-x",
        mode="performance",
        device=args.device,
        backend=args.backend,
    )

    images = _load_images(args.images)
    if not images:
        print("No valid images found.")
        sys.exit(1)

    rows = []
    for name, img in images:
        m = probe_image(model, img)
        rows.append((name, m))
        vis_path = out_dir / f"{name}_rtmw.jpg"
        cv2.imwrite(str(vis_path), m["annotated"])
        print(
            f"[{name}]  persons={m['n_persons']}  "
            f"body_cov={m['body_coverage']:.2f}  "
            f"hand_cov={m['hand_coverage']:.2f}  "
            f"mean_conf={m['mean_conf']:.2f}  "
            f"bone_err={m['bone_ratio_err']:.2f}  "
            f"hands_visible={m['visible_hands']}"
            f"  → {vis_path}"
        )

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n── SUMMARY ──────────────────────────────────────────────────────")
    body_covs = [r[1]["body_coverage"] for r in rows]
    hand_covs = [r[1]["hand_coverage"] for r in rows]
    print(f"  body_coverage  mean={np.mean(body_covs):.2f}  min={np.min(body_covs):.2f}  max={np.max(body_covs):.2f}")
    print(f"  hand_coverage  mean={np.mean(hand_covs):.2f}  min={np.min(hand_covs):.2f}  max={np.max(hand_covs):.2f}")
    print(f"\nAnnotated images saved to: {out_dir}/")
    print("\nInterpretation guide:")
    print("  body_coverage > 0.80  → RTMW reliably finds body on this image")
    print("  hand_coverage > 0.60  → hands are being detected")
    print("  hand_coverage < 0.30  → RTMW is mostly blind to hands here")
    print("  bone_ratio_err > 0.20 → limb asymmetry is visible (could be stylized or broken)")
    print("\nIf hand_coverage is consistently < 0.30 on your anime images,")
    print("RTMW will not be useful as a hand reward source without fine-tuning.")


if __name__ == "__main__":
    main()
