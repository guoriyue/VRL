"""Anime anatomy probe -- runs RTMW (+ optionally HaMeR), outputs HTML report.

This is the main entry point. RTMW is required. HaMeR is optional.

Setup (RTMW only, ~2 min):
    pip install rtmlib opencv-python pillow numpy

Setup (RTMW + HaMeR, first run downloads ~5.8 GB):
    pip install rtmlib opencv-python pillow numpy
    pip install git+https://github.com/geopavlakos/hamer.git

Usage:
    # Point at a directory of anime images:
    python -m vrl.scripts.eval.probe_anime_anatomy_report \
        --images /path/to/anime/*.png

    # Optionally enable HaMeR:
    python -m vrl.scripts.eval.probe_anime_anatomy_report --images *.png --hamer

Output:
    outputs/probes/anime_anatomy/report/report.html   <- open this in browser
    outputs/probes/anime_anatomy/report/*.jpg         <- per-image annotated overlays

What the report shows
----------------------
Per image:
  - RTMW keypoint overlay (body + hands, color-coded confidence)
  - Numeric scores: body_coverage, hand_coverage, mean_conf, bone_ratio_err
  - Hand detected: yes/no, how many
  - If HaMeR enabled: mean finger_cv

Summary table:
  - Mean/min/max of each metric across all images
  - "Reward signal discrimination" estimate: how much variance exists between
    images -- low variance = model is not useful as discriminative reward

The key question: do GOOD anime images score higher than BAD ones?
To answer this properly, split your image set into:
  --good-dir  (images where body/hands look correct to human eye)
  --bad-dir   (images where body/hands are deformed)
and compare the distributions.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from .anime_probe_common import (
    DEFAULT_HAMER_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    hamer_probe_image,
    load_hamer_model,
    load_images,
    load_rtmw,
    optional_finger_cv,
    require_hamer_modules,
    require_rtmw_modules,
    rtmw_metrics,
)

require_rtmw_modules()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def _score_color(v: float, low: float, high: float) -> str:
    """Green if high enough, red if low enough, yellow in between."""
    if v >= high:
        return "#4caf50"
    if v <= low:
        return "#f44336"
    return "#ff9800"


def build_report(rows: list[dict], out_dir: Path) -> Path:
    html_rows = ""
    for r in rows:
        m = r["rtmw"]
        hm = r.get("hamer") or {}
        b64 = _img_to_b64(m["annotated"])
        bc_col = _score_color(m["body_coverage"], 0.4, 0.8)
        hc_col = _score_color(m["hand_coverage"], 0.3, 0.6)
        be_col = _score_color(1 - m["bone_ratio_err"], 0.6, 0.85)
        finger_cv = optional_finger_cv(hm)
        cv_str = f"{finger_cv:.3f}" if finger_cv is not None else "-"
        cv_col = _score_color(1 - (0.5 if finger_cv is None else finger_cv), 0.6, 0.85)
        html_rows += f"""
        <tr>
          <td><b>{r["name"]}</b></td>
          <td><img src="data:image/jpeg;base64,{b64}" style="max-width:280px;border-radius:4px"></td>
          <td style="color:{bc_col};font-weight:bold">{m["body_coverage"]:.2f}</td>
          <td style="color:{hc_col};font-weight:bold">{m["hand_coverage"]:.2f}</td>
          <td>{m["mean_conf"]:.2f}</td>
          <td style="color:{be_col}">{m["bone_ratio_err"]:.2f}</td>
          <td>{m["visible_hands"]}</td>
          <td style="color:{cv_col}">{cv_str}</td>
        </tr>"""

    # aggregate stats
    body_covs = [r["rtmw"]["body_coverage"] for r in rows]
    hand_covs = [r["rtmw"]["hand_coverage"] for r in rows]
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
<p>Models: <b>RTMW-x</b> (whole-body 2D keypoints) {"+ <b>HaMeR</b> (hand mesh)" if any(r.get("hamer") for r in rows) else ""}</p>
<p>Images: {len(rows)}</p>

<h2>Summary Statistics</h2>
<div>
  <div class="stat">body_coverage mean: <b>{np.mean(body_covs):.2f}</b></div>
  <div class="stat">hand_coverage mean: <b>{np.mean(hand_covs):.2f}</b></div>
  <div class="stat">body discrimination (std): <b class="{"good" if disc_body > 0.15 else "warn"}">{disc_body:.3f}</b></div>
  <div class="stat">hand discrimination (std): <b class="{"good" if disc_hand > 0.15 else "warn"}">{disc_hand:.3f}</b></div>
</div>

<h2>Interpretation</h2>
<ul>
  <li><b>body_coverage</b>: fraction of 17 body keypoints detected with conf &gt; 0.3
    <br>&nbsp;&nbsp;<span class="good">&gt;= 0.80</span> = body reliably found &nbsp;
    <span class="warn">0.40-0.80</span> = partial &nbsp;
    <span class="bad">&lt; 0.40</span> = mostly missed</li>
  <li><b>hand_coverage</b>: fraction of 42 hand keypoints detected
    <br>&nbsp;&nbsp;<span class="good">&gt;= 0.60</span> = hands visible &nbsp;
    <span class="bad">&lt; 0.30</span> = RTMW blind to hands here</li>
  <li><b>bone_err</b>: left/right limb asymmetry (0=symmetric, &gt;0.3=suspicious)</li>
  <li><b>finger_cv</b> (HaMeR): coefficient of variation of finger segment lengths
    <br>&nbsp;&nbsp;<span class="good">&lt; 0.20</span> = MANO fits well &nbsp;
    <span class="bad">&gt; 0.40</span> = anime proportions confuse MANO</li>
  <li><b>discrimination std</b>: how much the metric varies across your images.
    <span class="bad">std &lt; 0.10</span> = model gives almost the same score to every image - not useful as discriminative reward</li>
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
  <li>If <b>body_coverage is consistently &gt; 0.70</b> and <b>std &gt; 0.15</b>: RTMW body metrics could discriminate quality - worth building the reward.</li>
  <li>If <b>finger_cv &gt; 0.40 even on visually correct hands</b>: HaMeR's MANO prior is too realistic-human-biased for anime hands. Need calibration layer.</li>
  <li>If discrimination std &lt; 0.10 for all metrics: these models may not be useful as rewards until fine-tuned on anime data.</li>
</ul>
</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Anime anatomy probe -- RTMW [+ HaMeR]")
    ap.add_argument("--images", nargs="+", required=True, help="Images or glob")
    ap.add_argument(
        "--good-dir", help="Directory of 'good' images (optional, for discrimination test)"
    )
    ap.add_argument(
        "--bad-dir", help="Directory of 'bad' images (optional, for discrimination test)"
    )
    ap.add_argument("--out", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_HAMER_CACHE_DIR))
    ap.add_argument("--hamer", action="store_true", help="Also run HaMeR hand mesh")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--backend", default="onnxruntime", choices=["onnxruntime", "openvino"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rtmw = load_rtmw(args.device, args.backend)
    hamer_model, hamer_cfg = None, None
    if args.hamer:
        require_hamer_modules()
        try:
            hamer_model, hamer_cfg = load_hamer_model(
                device=args.device,
                cache_dir=args.cache_dir,
            )
        except Exception as exc:
            print(f"[warn] HaMeR load failed: {exc} -- skipping hand mesh probe")
            args.hamer = False

    # Collect images, optionally annotated as good/bad
    images: list[tuple[str, np.ndarray, str | None]] = []
    for name, img in load_images(args.images):
        images.append((name, img, None))
    if args.good_dir:
        for name, img in load_images([str(Path(args.good_dir) / "*")]):
            images.append((f"GOOD_{name}", img, "good"))
    if args.bad_dir:
        for name, img in load_images([str(Path(args.bad_dir) / "*")]):
            images.append((f"BAD_{name}", img, "bad"))

    if not images:
        print("No valid images found.")
        raise SystemExit(1)

    rows = []
    for name, img, label in images:
        print(f"  {name}...", end=" ", flush=True)
        m = rtmw_metrics(rtmw, img)
        hm = (
            hamer_probe_image(
                img,
                hamer_model,
                hamer_cfg,
                device=args.device,
                cache_dir=args.cache_dir,
            )
            if args.hamer
            else None
        )

        # save annotated
        vis_path = out_dir / f"{name}_overlay.jpg"
        cv2.imwrite(str(vis_path), m["annotated"])

        rows.append({"name": name, "label": label, "rtmw": m, "hamer": hm})
        finger_cv = optional_finger_cv(hm or {})
        cv_str = f"  finger_cv={finger_cv:.3f}" if finger_cv is not None else ""
        print(
            f"body={m['body_coverage']:.2f} "
            f"hand={m['hand_coverage']:.2f} "
            f"bone_err={m['bone_ratio_err']:.2f}{cv_str}"
        )

    # discrimination test if good/bad dirs provided
    good_rows = [r for r in rows if r["label"] == "good"]
    bad_rows = [r for r in rows if r["label"] == "bad"]
    if good_rows and bad_rows:
        print("\n-- DISCRIMINATION TEST ------------------------------------------")
        for metric in ["body_coverage", "hand_coverage"]:
            good_vals = [r["rtmw"][metric] for r in good_rows]
            bad_vals = [r["rtmw"][metric] for r in bad_rows]
            sep = np.mean(good_vals) - np.mean(bad_vals)
            print(
                f"  {metric}: good={np.mean(good_vals):.3f}  bad={np.mean(bad_vals):.3f}  delta={sep:+.3f}",
                end="",
            )
            print("  PASS discriminative" if sep > 0.05 else "  FAIL NOT discriminative")

    report_path = build_report(rows, out_dir)
    print(f"\n-- Report -> {report_path}")
    print(f"   Open in browser: file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
