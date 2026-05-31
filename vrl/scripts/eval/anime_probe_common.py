"""Shared helpers for anime anatomy probe scripts.

This module intentionally has no ``vrl`` imports. The probe scripts are meant
to run as standalone diagnostics against generated anime images.
"""

from __future__ import annotations

import os
import sys
from glob import glob
from itertools import pairwise
from pathlib import Path
from typing import Any

CONF_THRESHOLD = 0.3

DEFAULT_PROBE_ROOT = Path("outputs/probes/anime_anatomy")
DEFAULT_REPORT_DIR = DEFAULT_PROBE_ROOT / "report"
DEFAULT_HAMER_CACHE_DIR = DEFAULT_PROBE_ROOT / "cache" / "hamer"

BODY_KP_RANGE = tuple(range(0, 17))
LHAND_KP_RANGE = tuple(range(91, 112))
RHAND_KP_RANGE = tuple(range(112, 133))
HAND_KP_RANGE = tuple(range(91, 133))

LIMB_SYMMETRY_PAIRS = (
    (5, 7, 6, 8),  # upper arms
    (7, 9, 8, 10),  # lower arms
    (11, 13, 12, 14),  # upper legs
    (13, 15, 14, 16),  # lower legs
)

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),  # index
    (0, 5, 6, 7, 8),  # middle
    (0, 9, 10, 11, 12),  # pinky
    (0, 13, 14, 15, 16),  # ring
    (0, 17, 18, 19, 20),  # thumb
)


def require_modules(modules: list[tuple[str, str]]) -> None:
    """Exit with install hints when optional probe dependencies are missing."""

    missing = []
    for module_name, install_name in modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(install_name)
    if not missing:
        return

    print("Missing dependencies. Install with:")
    for package in missing:
        print(f"  pip install {package}")
    sys.exit(1)


def require_rtmw_modules() -> None:
    require_modules(
        [
            ("rtmlib", "rtmlib"),
            ("cv2", "opencv-python"),
            ("PIL", "Pillow"),
            ("numpy", "numpy"),
        ],
    )


def require_hamer_modules() -> None:
    require_modules(
        [
            ("hamer", "git+https://github.com/geopavlakos/hamer.git"),
            ("cv2", "opencv-python"),
            ("numpy", "numpy"),
        ],
    )


def load_images(paths: list[str]) -> list[tuple[str, Any]]:
    """Load image files or globs as ``(stem, BGR ndarray)`` pairs."""

    import cv2

    images = []
    for raw_path in paths:
        expanded = glob(raw_path) if "*" in raw_path else [raw_path]
        for candidate in expanded:
            path = Path(candidate)
            if not path.exists():
                print(f"[warn] not found: {path}")
                continue
            image = cv2.imread(str(path))
            if image is None:
                print(f"[warn] cannot read: {path}")
                continue
            images.append((path.stem, image))
    return images


def load_rtmw(device: str, backend: str) -> Any:
    """Load RTMW whole-body model through rtmlib."""

    from rtmlib import Wholebody

    print(f"Loading RTMW-x wholebody ({device}, {backend})...")
    print("(rtmlib will auto-download ONNX weights on first run, ~300 MB)\n")
    return Wholebody(mode="performance", device=device, backend=backend)


def rtmw_metrics(model: Any, img_bgr: Any) -> dict[str, Any]:
    """Run RTMW on one image and return coverage metrics plus an overlay."""

    import cv2
    import numpy as np
    from rtmlib import draw_skeleton

    keypoints, scores = model(img_bgr)
    if keypoints is None or len(keypoints) == 0:
        annotated = img_bgr.copy()
        cv2.putText(
            annotated,
            "NO PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return {
            "n_persons": 0,
            "body_coverage": 0.0,
            "hand_coverage": 0.0,
            "mean_conf": 0.0,
            "bone_ratio_err": 0.0,
            "visible_hands": 0,
            "annotated": annotated,
        }

    idx = int(np.argmax(scores[:, list(BODY_KP_RANGE)].mean(axis=1)))
    kps = keypoints[idx]
    sc = scores[idx]

    body_cov = float((sc[list(BODY_KP_RANGE)] > CONF_THRESHOLD).mean())
    hand_cov = float((sc[list(HAND_KP_RANGE)] > CONF_THRESHOLD).mean())
    mean_conf = float(sc.mean())
    bone_err = _bone_ratio_err(kps, sc)
    lhand_visible = int(sc[list(LHAND_KP_RANGE)].mean() > CONF_THRESHOLD)
    rhand_visible = int(sc[list(RHAND_KP_RANGE)].mean() > CONF_THRESHOLD)
    annotated = draw_skeleton(img_bgr.copy(), keypoints, scores, kpt_thr=CONF_THRESHOLD)

    return {
        "n_persons": len(keypoints),
        "body_coverage": round(body_cov, 3),
        "hand_coverage": round(hand_cov, 3),
        "mean_conf": round(mean_conf, 3),
        "bone_ratio_err": round(bone_err, 3),
        "visible_hands": lhand_visible + rhand_visible,
        "annotated": annotated,
    }


def _bone_ratio_err(kps: Any, scores: Any) -> float:
    import numpy as np

    errors = []
    for left_start, left_end, right_start, right_end in LIMB_SYMMETRY_PAIRS:
        if (
            scores[left_start] < CONF_THRESHOLD
            or scores[left_end] < CONF_THRESHOLD
            or scores[right_start] < CONF_THRESHOLD
            or scores[right_end] < CONF_THRESHOLD
        ):
            continue
        left_len = np.linalg.norm(kps[left_start] - kps[left_end]) + 1e-6
        right_len = np.linalg.norm(kps[right_start] - kps[right_end]) + 1e-6
        errors.append(abs(left_len - right_len) / max(left_len, right_len))
    return float(np.mean(errors)) if errors else 0.0


def load_hamer_model(
    *,
    device: str,
    cache_dir: str | Path = DEFAULT_HAMER_CACHE_DIR,
) -> tuple[Any, Any]:
    """Load HaMeR using a repo-local output cache, not the repository root."""

    from hamer.models import HAMER, download_models

    cache_root = Path(cache_dir).expanduser().resolve()
    data_dir = cache_root / "_DATA"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Upstream extracts hamer_ckpts relative to cwd, so run it inside cache_root.
    cwd = Path.cwd()
    try:
        os.chdir(cache_root)
        download_models(folder=str(data_dir))
    finally:
        os.chdir(cwd)

    checkpoint = cache_root / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"HaMeR checkpoint missing after download: {checkpoint}")
    model, cfg = HAMER.load_from_checkpoint(str(checkpoint))
    return model.eval().to(device), cfg


def hamer_probe_image(
    img_bgr: Any,
    model: Any,
    model_cfg: Any,
    *,
    device: str,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run HaMeR on one image and return hand metrics plus an overlay."""

    import cv2
    import numpy as np
    import torch
    from hamer.utils import recursive_to

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    dataloader = _hamer_dataloader(
        img_rgb,
        model_cfg,
        image_size=(w, h),
        cache_dir=cache_dir,
    )

    hand_results = []
    annotated = img_bgr.copy()

    with torch.no_grad():
        for batch in dataloader:
            batch = recursive_to(batch, device)
            out = model(batch)
            joints_3d = out["pred_3d_joints"].cpu().numpy()
            pred_keypoints_2d = out["pred_keypoints_2d"].cpu().numpy()
            right_flag = batch.get("right", torch.zeros(len(joints_3d))).cpu().numpy()

            for i, j3d in enumerate(joints_3d):
                j2d_px = ((pred_keypoints_2d[i] + 1) / 2 * np.array([w, h])).astype(int)
                handedness = "R" if right_flag[i] > 0.5 else "L"
                color = (0, 255, 0) if handedness == "R" else (255, 0, 0)

                for x, y in j2d_px:
                    if 0 <= int(x) < w and 0 <= int(y) < h:
                        cv2.circle(annotated, (int(x), int(y)), 3, color, -1)

                hand_results.append(
                    {
                        "handedness": handedness,
                        "finger_cv": round(finger_cv(j3d), 4),
                        "j2d_px": j2d_px,
                    },
                )

    if not hand_results:
        return no_hands_result(img_bgr)

    mean_cv = float(np.mean([hand["finger_cv"] for hand in hand_results]))
    n_hands = len(hand_results)
    return {
        "n_hands": n_hands,
        "mean_finger_cv": round(mean_cv, 4),
        "hands": hand_results,
        "verdict": hamer_verdict(mean_cv, n_hands),
        "annotated": annotated,
    }


def _hamer_dataloader(
    img_rgb: Any,
    model_cfg: Any,
    *,
    image_size: tuple[int, int],
    cache_dir: str | Path | None,
) -> Any:
    import numpy as np
    import torch
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    w, h = image_size
    boxes = _detect_hands_or_full_image(img_rgb, image_size=image_size, cache_dir=cache_dir)
    dataset = ViTDetDataset(model_cfg, img_rgb, boxes, right_hand_flag=None)
    batch_size = 8 if not np.array_equal(boxes, np.array([[0, 0, w, h]], dtype=np.float32)) else 1
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _detect_hands_or_full_image(
    img_rgb: Any,
    *,
    image_size: tuple[int, int],
    cache_dir: str | Path | None,
) -> Any:
    import numpy as np

    w, h = image_size
    fallback = np.array([[0, 0, w, h]], dtype=np.float32)
    try:
        from detectron2.config import LazyConfig
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

        cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else Path.cwd()
        detector_dir = cache_root / "hamer_ckpts" / "detectron2"
        cfg_path = detector_dir / "cascade_mask_rcnn_vitdet_h_75ep.py"
        checkpoint_path = detector_dir / "model_final_f05665.pkl"
        if not cfg_path.exists() or not checkpoint_path.exists():
            return fallback

        detector_cfg = LazyConfig.load(str(cfg_path))
        detector_cfg.train.init_checkpoint = str(checkpoint_path)
        detector = DefaultPredictor_Lazy(detector_cfg)
        det_out = detector(img_rgb)
        boxes = det_out["instances"].pred_boxes.tensor.cpu().numpy()
        return boxes if len(boxes) else fallback
    except Exception:
        return fallback


def finger_cv(joints_3d: Any) -> float:
    """Coefficient of variation of MANO finger segment lengths."""

    import numpy as np

    lengths = [
        np.linalg.norm(joints_3d[a] - joints_3d[b])
        for chain in FINGER_CHAINS
        for a, b in pairwise(chain)
    ]
    arr = np.array(lengths)
    if arr.mean() < 1e-6:
        return 0.0
    return float(arr.std() / (arr.mean() + 1e-6))


def no_hands_result(img_bgr: Any) -> dict[str, Any]:
    import cv2

    annotated = img_bgr.copy()
    cv2.putText(
        annotated,
        "NO HANDS DETECTED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    return {
        "n_hands": 0,
        "mean_finger_cv": None,
        "hands": [],
        "verdict": "NOT_DETECTED",
        "annotated": annotated,
    }


def hamer_verdict(mean_cv: float, n_hands: int) -> str:
    if n_hands == 0:
        return "NOT_DETECTED"
    if mean_cv < 0.20:
        return "GOOD (proportions consistent)"
    if mean_cv < 0.40:
        return "BORDERLINE (some inconsistency)"
    return "POOR (high CV -> deformed or anime-style mismatch)"
