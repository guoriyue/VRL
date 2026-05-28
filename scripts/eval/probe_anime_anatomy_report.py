"""Anime anatomy probe — runs RTMW (+ optionally HaMeR), outputs HTML report.

This is the main entry point. RTMW is required. HaMeR is optional.

Setup (RTMW only, ~2 min):
    pip install rtmlib opencv-python pillow numpy

Setup (RTMW + HaMeR, ~10 min + SMPL-X file):
    pip install rtmlib opencv-python pillow numpy
    pip install git+https://github.com/geopavlakos/hamer.git
    # Register + download SMPLX_NEUTRAL.npz from https://smpl-x.is.tue.mpg.de
    export VRL_SMPLX_PATH=/path/to/SMPLX_NEUTRAL.npz

Usage:
    # Point at a directory of anime images:
    python scripts/eval/probe_anime_anatomy_report.py --images /path/to/anime/*.png

    # Optionally enable HaMeR:
    python scripts/eval/probe_anime_anatomy_report.py --images *.png --hamer

Output:
    results/anatomy_probe/report.html   ← open this in browser
    results/anatomy_probe/*.jpg         ← per-image annotated overlays

What the report shows
----------------------
Per image:
  - RTMW keypoint overlay (body + hands, color-coded confidence)
  - Numeric scores: body_coverage, hand_coverage, mean_conf, bone_ratio_err
  - Hand detected: yes/no, how many
  - If HaMeR enabled: finger_cv per hand, verdict

Summary table:
  - Mean/min/max of each metric across all images
  - "Reward signal discrimination" estimate: how much variance exists between
    images — low variance = model is not useful as discriminative reward

The key question: do GOOD anime images score higher than BAD ones?
To answer this properly, split your image set into:
  --good-dir  (images where body/hands look correct to human eye)
  --bad-dir   (images where body/hands are deformed)
and compare the distributions.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for name, pkg in [("rtmlib", "rtmlib"), ("cv2", "opencv-python"),
                      ("PIL", "Pillow"), ("numpy", "numpy")]:
        try:
            __import__("cv2" if name == "cv2" else name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing required dependencies:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        sys.exit(1)

_check_deps()

import cv2
import numpy as np
from glob import glob

CONF_THRESHOLD = 0.3
BODY_KP = range(0, 17)
HAND_KP = range(91, 133)
LHAND_KP = range(91, 112)
RHAND_KP = range(112, 133)


# ── RTMW ─────────────────────────────────────────────────────────────────────

def load_rtmw(device: str, backend: str):
    from rtmlib import Wholebody
    print(f"Loading RTMW-x ({device}, {backend})…")
    print("  (auto-downloading ONNX weights on first run, ~200 MB)\n")
    return Wholebody(pose_model_type="rtmw-x", mode="performance",
                     device=device, backend=backend)


def rtmw_metrics(model, img_bgr: np.ndarray) -> dict:
    from rtmlib import draw_skeleton
    kps, scores = model(img_bgr)

    if kps is None or len(kps) == 0:
        ann = img_bgr.copy()
        cv2.putText(ann, "NO PERSON", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return dict(n_persons=0, body_cov=0.0, hand_cov=0.0, mean_conf=0.0,
                    bone_err=0.0, visible_hands=0, annotated=ann)

    idx = int(np.argmax(scores[:, list(BODY_KP)].mean(axis=1)))
    kp, sc = kps[idx], scores[idx]

    body_cov = float((sc[list(BODY_KP)] > CONF_THRESHOLD).mean())
    hand_cov = float((sc[list(HAND_KP)] > CONF_THRESHOLD).mean())
    mean_c = float(sc.mean())

    # bone symmetry error
    pairs = [(5,7,6,8),(7,9,8,10),(11,13,12,14),(13,15,14,16)]
    errs = []
    for ls,le,rs,re in pairs:
        if sc[ls]>CONF_THRESHOLD and sc[le]>CONF_THRESHOLD and sc[rs]>CONF_THRESHOLD and sc[re]>CONF_THRESHOLD:
            ll = np.linalg.norm(kp[ls]-kp[le])+1e-6
            rl = np.linalg.norm(kp[rs]-kp[re])+1e-6
            errs.append(abs(ll-rl)/max(ll,rl))
    bone_err = float(np.mean(errs)) if errs else 0.0

    lh = int(sc[list(LHAND_KP)].mean() > CONF_THRESHOLD)
    rh = int(sc[list(RHAND_KP)].mean() > CONF_THRESHOLD)

    ann = draw_skeleton(img_bgr.copy(), kps, scores, kpt_thr=CONF_THRESHOLD)
    return dict(n_persons=len(kps), body_cov=round(body_cov,3),
                hand_cov=round(hand_cov,3), mean_conf=round(mean_c,3),
                bone_err=round(bone_err,3), visible_hands=lh+rh, annotated=ann)


# ── HaMeR (optional) ─────────────────────────────────────────────────────────

def load_hamer(device: str):
    try:
        from hamer.models import HAMER, download_models
        download_models()
        model, cfg = HAMER.load_from_checkpoint("hamer_ckpts/checkpoints/hamer.ckpt")
        return model.eval().to(device), cfg
    except Exception as e:
        print(f"[warn] HaMeR load failed: {e} — skipping hand mesh probe")
        return None, None


def hamer_metrics(img_bgr: np.ndarray, model, model_cfg, device: str) -> dict | None:
    if model is None:
        return None
    try:
        import torch
        from hamer.utils import recursive_to
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = np.array([[0, 0, w, h]], dtype=np.float32)  # full-image fallback
        dataset = ViTDetDataset(model_cfg, img_rgb, boxes, right_hand_flag=None)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        cvs = []
        with torch.no_grad():
            for batch in loader:
                batch = recursive_to(batch, device)
                out = model(batch)
                j3d = out["pred_3d_joints"].cpu().numpy()
                for j in j3d:
                    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
                    lengths = [np.linalg.norm(j[a]-j[b]) for ch in chains for a,b in zip(ch,ch[1:])]
                    arr = np.array(lengths)
                    cvs.append(float(arr.std()/(arr.mean()+1e-6)))
        if not cvs:
            return {"n_hands": 0, "mean_finger_cv": None}
        return {"n_hands": len(cvs), "mean_finger_cv": round(float(np.mean(cvs)), 4)}
    except Exception as e:
        return {"n_hands": 0, "mean_finger_cv": None, "error": str(e)}


# ── HTML report ───────────────────────────────────────────────────────────────

def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def _score_color(v: float, low: float, high: float) -> str:
    """Green if ≥ high, red if ≤ low, yellow in between."""
    if v >= high: return "#4caf50"
    if v <= low:  return "#f44336"
    return "#ff9800"


def build_report(rows: list[dict], out_dir: Path) -> Path:
    html_rows = ""
    for r in rows:
        m = r["rtmw"]
        hm = r.get("hamer") or {}
        b64 = _img_to_b64(m["annotated"])
        bc_col = _score_color(m["body_cov"], 0.4, 0.8)
        hc_col = _score_color(m["hand_cov"], 0.3, 0.6)
        be_col = _score_color(1 - m["bone_err"], 0.6, 0.85)
        cv_str = f"{hm['mean_finger_cv']:.3f}" if hm.get("mean_finger_cv") is not None else "—"
        cv_col = _score_color(1 - (hm.get("mean_finger_cv") or 0.5), 0.6, 0.85)
        html_rows += f"""
        <tr>
          <td><b>{r['name']}</b></td>
          <td><img src="data:image/jpeg;base64,{b64}" style="max-width:280px;border-radius:4px"></td>
          <td style="color:{bc_col};font-weight:bold">{m['body_cov']:.2f}</td>
          <td style="color:{hc_col};font-weight:bold">{m['hand_cov']:.2f}</td>
          <td>{m['mean_conf']:.2f}</td>
          <td style="color:{be_col}">{m['bone_err']:.2f}</td>
          <td>{m['visible_hands']}</td>
          <td style="color:{cv_col}">{cv_str}</td>
        </tr>"""

    # aggregate stats
    body_covs = [r["rtmw"]["body_cov"] for r in rows]
    hand_covs = [r["rtmw"]["hand_cov"] for r in rows]
    disc_body = float(np.std(body_covs))
    disc_hand = float(np.std(hand_covs))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Anime Anatomy Probe Report</title>
<style>
  body {{ font-family: monospace; background:#1a1a2e; color:#eee; padding:20px }}
  h1 {{ color:#a78bfa }}
  h2 {{ color:#7dd3fc; margin-top:30px }}
  table {{ border-collapse:collapse; width:100% }}
  th {{ background:#2d2d5e; padding:8px; text-align:left; color:#a78bfa }}
  td {{ padding:8px; border-bottom:1px solid #333; vertical-align:top }}
  tr:hover {{ background:#1e1e40 }}
  .stat {{ display:inline-block; background:#2d2d5e; padding:10px 20px; margin:5px; border-radius:8px }}
  .warn {{ color:#fbbf24 }}
  .good {{ color:#4caf50 }}
  .bad {{ color:#f44336 }}
</style>
</head>
<body>
<h1>Anime Anatomy Probe</h1>
<p>Models: <b>RTMW-x</b> (whole-body 2D keypoints) {'+ <b>HaMeR</b> (hand mesh)' if any(r.get('hamer') for r in rows) else ''}</p>
<p>Images: {len(rows)}</p>

<h2>Summary Statistics</h2>
<div>
  <div class="stat">body_coverage mean: <b>{np.mean(body_covs):.2f}</b></div>
  <div class="stat">hand_coverage mean: <b>{np.mean(hand_covs):.2f}</b></div>
  <div class="stat">body discrimination (std): <b class="{'good' if disc_body>0.15 else 'warn'}">{disc_body:.3f}</b></div>
  <div class="stat">hand discrimination (std): <b class="{'good' if disc_hand>0.15 else 'warn'}">{disc_hand:.3f}</b></div>
</div>

<h2>Interpretation</h2>
<ul>
  <li><b>body_coverage</b>: fraction of 17 body keypoints detected with conf &gt; 0.3
    <br>&nbsp;&nbsp;<span class="good">≥ 0.80</span> = body reliably found &nbsp;
    <span class="warn">0.40–0.80</span> = partial &nbsp;
    <span class="bad">&lt; 0.40</span> = mostly missed</li>
  <li><b>hand_coverage</b>: fraction of 42 hand keypoints detected
    <br>&nbsp;&nbsp;<span class="good">≥ 0.60</span> = hands visible &nbsp;
    <span class="bad">&lt; 0.30</span> = RTMW blind to hands here</li>
  <li><b>bone_err</b>: left/right limb asymmetry (0=symmetric, &gt;0.3=suspicious)</li>
  <li><b>finger_cv</b> (HaMeR): coefficient of variation of finger segment lengths
    <br>&nbsp;&nbsp;<span class="good">&lt; 0.20</span> = MANO fits well &nbsp;
    <span class="bad">&gt; 0.40</span> = anime proportions confuse MANO</li>
  <li><b>discrimination std</b>: how much the metric varies across your images.
    <span class="bad">std &lt; 0.10</span> = model gives almost the same score to every image → not useful as discriminative reward</li>
</ul>

<h2>Per-Image Results</h2>
<table>
  <tr>
    <th>Image</th><th>Overlay</th>
    <th>body_cov</th><th>hand_cov</th>
    <th>mean_conf</th><th>bone_err</th>
    <th>hands</th><th>finger_cv</th>
  </tr>
  {html_rows}
</table>

<h2>What to do with this</h2>
<ul>
  <li>If <b>hand_coverage is consistently &lt; 0.30</b>: RTMW hand keypoints will not be a useful reward signal without fine-tuning on anime data.</li>
  <li>If <b>body_coverage is consistently &gt; 0.70</b> and <b>std &gt; 0.15</b>: RTMW body metrics could discriminate quality — worth building the reward.</li>
  <li>If <b>finger_cv &gt; 0.40 even on visually correct hands</b>: HaMeR's MANO prior is too realistic-human-biased for anime hands. Need calibration layer.</li>
  <li>If discrimination std &lt; 0.10 for all metrics: these models may not be useful as rewards until fine-tuned on anime data.</li>
</ul>
</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


# ── main ──────────────────────────────────────────────────────────────────────

def _collect_images(paths: list[str]) -> list[tuple[str, np.ndarray]]:
    out = []
    for p in paths:
        for f in (glob(p) if "*" in p else [p]):
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
    ap = argparse.ArgumentParser(description="Anime anatomy probe — RTMW [+ HaMeR]")
    ap.add_argument("--images", nargs="+", required=True, help="Images or glob")
    ap.add_argument("--good-dir", help="Directory of 'good' images (optional, for discrimination test)")
    ap.add_argument("--bad-dir",  help="Directory of 'bad' images (optional, for discrimination test)")
    ap.add_argument("--out", default="results/anatomy_probe")
    ap.add_argument("--hamer", action="store_true", help="Also run HaMeR hand mesh")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--backend", default="onnxruntime", choices=["onnxruntime", "openvino"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rtmw = load_rtmw(args.device, args.backend)
    hamer_model, hamer_cfg = (load_hamer(args.device) if args.hamer else (None, None))

    # Collect images, optionally annotated as good/bad
    images: list[tuple[str, np.ndarray, str | None]] = []
    for name, img in _collect_images(args.images):
        images.append((name, img, None))
    if args.good_dir:
        for name, img in _collect_images([str(Path(args.good_dir) / "*")]):
            images.append((f"GOOD_{name}", img, "good"))
    if args.bad_dir:
        for name, img in _collect_images([str(Path(args.bad_dir) / "*")]):
            images.append((f"BAD_{name}", img, "bad"))

    if not images:
        print("No valid images found.")
        sys.exit(1)

    rows = []
    for name, img, label in images:
        print(f"  {name}…", end=" ", flush=True)
        m = rtmw_metrics(rtmw, img)
        hm = hamer_metrics(img, hamer_model, hamer_cfg, args.device) if args.hamer else None

        # save annotated
        vis_path = out_dir / f"{name}_overlay.jpg"
        cv2.imwrite(str(vis_path), m["annotated"])

        rows.append({"name": name, "label": label, "rtmw": m, "hamer": hm})
        cv_str = f"  finger_cv={hm['mean_finger_cv']:.3f}" if hm and hm.get("mean_finger_cv") else ""
        print(f"body={m['body_cov']:.2f} hand={m['hand_cov']:.2f} bone_err={m['bone_err']:.2f}{cv_str}")

    # discrimination test if good/bad dirs provided
    good_rows = [r for r in rows if r["label"] == "good"]
    bad_rows  = [r for r in rows if r["label"] == "bad"]
    if good_rows and bad_rows:
        print("\n── DISCRIMINATION TEST ──────────────────────────────────────────")
        for metric in ["body_cov", "hand_cov"]:
            good_vals = [r["rtmw"][metric] for r in good_rows]
            bad_vals  = [r["rtmw"][metric] for r in bad_rows]
            sep = np.mean(good_vals) - np.mean(bad_vals)
            print(f"  {metric}: good={np.mean(good_vals):.3f}  bad={np.mean(bad_vals):.3f}  Δ={sep:+.3f}", end="")
            print("  ✓ discriminative" if sep > 0.05 else "  ✗ NOT discriminative")

    report_path = build_report(rows, out_dir)
    print(f"\n── Report → {report_path}")
    print(f"   Open in browser: file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
