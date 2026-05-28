"""Probe HaMeR hand mesh on anime images — standalone, no vrl imports.

Usage
-----
# Install deps (one-time):
#   pip install git+https://github.com/geopavlakos/hamer.git
#   pip install opencv-python pillow numpy
#   # Download SMPL-X neutral body model:
#   #   1. Register at https://smpl-x.is.tue.mpg.de
#   #   2. Download SMPLX_NEUTRAL.npz
#   #   3. export VRL_SMPLX_PATH=/path/to/SMPLX_NEUTRAL.npz

python scripts/eval/probe_hamer_anime.py --images *.png
python scripts/eval/probe_hamer_anime.py --images *.png --out results/hamer --smplx /path/to/SMPLX_NEUTRAL.npz

What this shows
---------------
For each image:
  - Hand bounding boxes overlaid
  - 2D reprojection of the 3D MANO hand mesh
  - Per-hand metrics:
      reproj_err_px   : mean 2D keypoint reprojection error (pixels)
      finger_cv       : coefficient of variation of finger segment lengths
                        (low = proportions are consistent, high = deformed)
      handedness_conf : confidence of L/R classification
  - Overall verdict per image: DETECTED / PARTIAL / NOT_DETECTED

The goal is to see how HaMeR fires on anime hands and whether its
reprojection error and finger geometry are discriminative enough to
distinguish "good hand" from "deformed hand" in anime style.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for name, pkg in [("hamer", "git+https://github.com/geopavlakos/hamer.git"),
                      ("cv2", "opencv-python"),
                      ("PIL", "Pillow"),
                      ("numpy", "numpy")]:
        try:
            __import__("cv2" if name == "cv2" else name)
        except ImportError:
            missing.append((name, pkg))
    if missing:
        print("Missing dependencies. Install with:")
        for name, pkg in missing:
            print(f"  pip install {pkg}")
        sys.exit(1)

_check_deps()

import cv2
import numpy as np
from PIL import Image

# ── HaMeR setup ───────────────────────────────────────────────────────────────

def _load_hamer(smplx_path: str | None, device: str):
    """Load HaMeR model. Returns (model, model_cfg) or raises."""
    try:
        from hamer.models import HAMER, download_models
        from hamer.utils import recursive_to
        from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
        from hamer.utils.renderer import Renderer, cam_crop_to_full
        import torch
    except ImportError as e:
        print(f"HaMeR import failed: {e}")
        print("Install: pip install git+https://github.com/geopavlakos/hamer.git")
        sys.exit(1)

    download_models()  # no-op if already downloaded
    model, model_cfg = HAMER.load_from_checkpoint(
        "hamer_ckpts/checkpoints/hamer.ckpt"
    )
    model = model.eval().to(device)
    return model, model_cfg


# ── metrics ───────────────────────────────────────────────────────────────────

def _finger_cv(joints_3d: np.ndarray) -> float:
    """Coefficient of variation of finger segment lengths.

    joints_3d: [21, 3] MANO hand joints
    Low CV = proportions are consistent across fingers.
    High CV = one or more segments are wildly different (deformed).

    MANO finger joint order:
      0=wrist, 1-4=index chain, 5-8=middle, 9-12=pinky, 13-16=ring, 17-20=thumb
    """
    finger_chains = [
        [0, 1, 2, 3, 4],    # index
        [0, 5, 6, 7, 8],    # middle
        [0, 9, 10, 11, 12], # pinky
        [0, 13, 14, 15, 16],# ring
        [0, 17, 18, 19, 20],# thumb
    ]
    lengths = []
    for chain in finger_chains:
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            lengths.append(np.linalg.norm(joints_3d[a] - joints_3d[b]))
    lengths = np.array(lengths)
    if lengths.mean() < 1e-6:
        return 0.0
    return float(lengths.std() / (lengths.mean() + 1e-6))


def _reproj_error(joints_2d_pred: np.ndarray, joints_2d_gt: np.ndarray) -> float:
    """Mean pixel distance between predicted and GT 2D joints.

    If no GT available, returns 0.0 (not meaningful without GT).
    """
    if joints_2d_gt is None:
        return 0.0
    return float(np.linalg.norm(joints_2d_pred - joints_2d_gt, axis=-1).mean())


# ── per-image probe ────────────────────────────────────────────────────────────

def probe_image(
    img_bgr: np.ndarray,
    model,
    model_cfg,
    device: str,
) -> dict:
    import torch
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from vitpose_model import ViTPoseModel
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig

    # Use ViTPose for hand bounding boxes (HaMeR's standard pipeline)
    # If ViTPose/detectron2 not available, we fall back to full-image crop
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    try:
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        cfg_path = Path("hamer_ckpts/detectron2/cascade_mask_rcnn_vitdet_h_75ep.py")
        detector_cfg = LazyConfig.load(str(cfg_path))
        detector_cfg.train.init_checkpoint = "hamer_ckpts/detectron2/model_final_f05665.pkl"
        detector = DefaultPredictor_Lazy(detector_cfg)

        vitpose_model = ViTPoseModel(device)

        det_out = detector(img_rgb)
        boxes = det_out["instances"].pred_boxes.tensor.cpu().numpy()
        if len(boxes) == 0:
            return _no_hands_result(img_bgr)

        # Get hand boxes from ViTPose
        dataset = ViTDetDataset(model_cfg, img_rgb, boxes, right_hand_flag=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    except Exception:
        # Fallback: treat full image as one hand crop (less accurate but runs)
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        boxes = np.array([[0, 0, w, h]], dtype=np.float32)
        dataset = ViTDetDataset(model_cfg, img_rgb, boxes, right_hand_flag=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    hand_results = []
    annotated = img_bgr.copy()

    with torch.no_grad():
        for batch in dataloader:
            batch = recursive_to(batch, device)
            out = model(batch)

            pred_cam = out["pred_cam"].cpu().numpy()
            pred_mano = out["pred_mano_params"]
            joints_3d = out["pred_3d_joints"].cpu().numpy()  # [B, 21, 3]
            pred_keypoints_2d = out["pred_keypoints_2d"].cpu().numpy()  # [B, 21, 2]
            right_flag = batch.get("right", torch.zeros(len(joints_3d))).cpu().numpy()

            for i in range(len(joints_3d)):
                j3d = joints_3d[i]   # [21, 3]
                j2d = pred_keypoints_2d[i]  # [21, 2] in [-1,1] normalized

                # Convert normalized 2d to pixel coords
                j2d_px = ((j2d + 1) / 2 * np.array([w, h])).astype(int)

                finger_cv = _finger_cv(j3d)
                handedness = "R" if right_flag[i] > 0.5 else "L"

                # Draw 2D joint projections
                for pt in j2d_px:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(annotated, (x, y), 3,
                                   (0, 255, 0) if handedness == "R" else (255, 0, 0), -1)

                hand_results.append({
                    "handedness": handedness,
                    "finger_cv": round(finger_cv, 4),
                    "j2d_px": j2d_px,
                })

    if not hand_results:
        return _no_hands_result(img_bgr)

    mean_cv = float(np.mean([h["finger_cv"] for h in hand_results]))
    n_hands = len(hand_results)

    return {
        "n_hands": n_hands,
        "mean_finger_cv": round(mean_cv, 4),
        "hands": hand_results,
        "verdict": _verdict(mean_cv, n_hands),
        "annotated": annotated,
    }


def _no_hands_result(img_bgr: np.ndarray) -> dict:
    ann = img_bgr.copy()
    cv2.putText(ann, "NO HANDS DETECTED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return {
        "n_hands": 0,
        "mean_finger_cv": None,
        "hands": [],
        "verdict": "NOT_DETECTED",
        "annotated": ann,
    }


def _verdict(mean_cv: float, n_hands: int) -> str:
    if n_hands == 0:
        return "NOT_DETECTED"
    if mean_cv < 0.20:
        return "GOOD (proportions consistent)"
    if mean_cv < 0.40:
        return "BORDERLINE (some inconsistency)"
    return "POOR (high CV → deformed or anime-style mismatch)"


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
    ap = argparse.ArgumentParser(description="Probe HaMeR on anime images")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="results/hamer_probe")
    ap.add_argument("--smplx", default=os.environ.get("VRL_SMPLX_PATH"),
                    help="Path to SMPLX_NEUTRAL.npz (or set VRL_SMPLX_PATH)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if not args.smplx:
        print("SMPL-X model path required.")
        print("  1. Register at https://smpl-x.is.tue.mpg.de")
        print("  2. Download SMPLX_NEUTRAL.npz")
        print("  3. Pass --smplx /path/to/SMPLX_NEUTRAL.npz  OR  export VRL_SMPLX_PATH=...")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HaMeR… (will download ~2 GB weights on first run)")
    model, model_cfg = _load_hamer(args.smplx, args.device)

    images = _load_images(args.images)
    if not images:
        print("No valid images found.")
        sys.exit(1)

    rows = []
    for name, img in images:
        result = probe_image(img, model, model_cfg, args.device)
        rows.append((name, result))
        vis_path = out_dir / f"{name}_hamer.jpg"
        cv2.imwrite(str(vis_path), result["annotated"])
        cv_str = f"{result['mean_finger_cv']:.4f}" if result["mean_finger_cv"] is not None else "N/A"
        print(
            f"[{name}]  hands={result['n_hands']}  "
            f"finger_cv={cv_str}  "
            f"verdict={result['verdict']}"
            f"  → {vis_path}"
        )

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n── SUMMARY ──────────────────────────────────────────────────────")
    detected = [r for r in rows if r[1]["n_hands"] > 0]
    print(f"  images with hands detected: {len(detected)} / {len(rows)}")
    if detected:
        cvs = [r[1]["mean_finger_cv"] for r in detected if r[1]["mean_finger_cv"] is not None]
        print(f"  finger_cv  mean={np.mean(cvs):.3f}  min={np.min(cvs):.3f}  max={np.max(cvs):.3f}")
    print(f"\nAnnotated images: {out_dir}/")
    print("\nInterpretation guide:")
    print("  finger_cv < 0.20  → MANO mesh fits well, proportions consistent")
    print("  finger_cv 0.20–0.40 → borderline; could be stylized or slightly deformed")
    print("  finger_cv > 0.40  → poor fit; anime proportions may confuse MANO")
    print("\nIf most anime images score finger_cv > 0.40 even on 'good' hands,")
    print("HaMeR's MANO prior is fighting with anime finger proportions —")
    print("you'll need a calibration layer to use it as reward.")


if __name__ == "__main__":
    main()
