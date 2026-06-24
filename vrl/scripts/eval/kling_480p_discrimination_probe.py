"""#7 de-risk probe: can Kling VideoReward tell good vs bad at 480p_33f?

832x480 = 399360 px/frame > the reward model's max_frame_pixels budget (200704),
so it downscales — possibly losing the detail it needs to score quality. If a
clearly-bad video (heavy noise + shuffled/dropped frames) does NOT score
meaningfully below the matching real rollout (gap within the fixed-eval noise
+/-0.0339), then the flat training reward is the *reward model's* fault at this
resolution, and running more epochs / a longer sweep is pointless — fix
resolution/rescale instead.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.eval.kling_480p_discrimination_probe \
    --videos outputs/cosmos_ddp_paper/reward_artifacts --n 8
"""

from __future__ import annotations

import os

# qwen_vl_utils reads this at import time to pick the video backend; the installed
# torchvision lacks io.read_video, so force decord before importing the model.
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

import argparse
import statistics
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

_EVAL_NOISE = 0.0339  # fixed-eval stderr from the 2026-06-20 run (eval_metrics.csv)
# Same prompt for the good/bad pair: the videos share content, so text-alignment
# is held constant and the gap isolates the visual/motion-quality the reward must
# see at 480p.
_PROMPT = "A short video of a physical interaction between objects."


def _degrade(src: Path, dst: Path) -> None:
    """Make an unambiguously-bad video: heavy noise + shuffled + dropped frames."""
    frames = np.asarray(iio.imread(src))  # (T,H,W,C)
    rng = np.random.default_rng(0)
    # drop every other frame, then shuffle temporal order (destroys motion)
    kept = frames[::2]
    order = rng.permutation(len(kept))
    shuffled = kept[order]
    # heavy additive noise (destroys visual quality)
    noisy = shuffled.astype(np.float32) + rng.normal(0, 80, shuffled.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    iio.imwrite(dst, noisy, fps=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir of real rollout .mp4 (good)")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    goods = sorted(Path(args.videos).glob("*.mp4"))[: args.n]
    if not goods:
        raise SystemExit(f"no .mp4 under {args.videos}")

    model = KlingVideoRewardModel(
        {"reward_model_name": "KlingTeam/VideoReward@main", "device": "cuda:0", "dtype": "bfloat16"},
    )

    # VQ/MQ are prompt-independent (the degradation hits visual + motion quality);
    # TA/Overall fold in text-alignment, which the shared generic prompt confounds.
    # If VQ and MQ both fail to separate, the reward is blind to quality at 480p.
    dims = ["VQ", "MQ", "TA", "Overall"]

    def score(path: Path) -> dict[str, float]:
        # absolute path: qwen prepends file:// and decord rejects file://<relative>
        out = model._reward([str(path.resolve())], [_PROMPT], use_norm=True)
        return out[0]

    good: dict[str, list[float]] = {d: [] for d in dims}
    bad: dict[str, list[float]] = {d: [] for d in dims}
    with tempfile.TemporaryDirectory() as tmp:
        for i, g in enumerate(goods):
            bp = Path(tmp) / f"bad_{i}.mp4"
            _degrade(g, bp)
            gs, bs = score(g), score(bp)
            for d in dims:
                good[d].append(gs[d])
                bad[d].append(bs[d])
            print(f"  {g.name}: VQ {gs['VQ']:+.3f}/{bs['VQ']:+.3f}  MQ {gs['MQ']:+.3f}/{bs['MQ']:+.3f}"
                  f"  Overall {gs['Overall']:+.3f}/{bs['Overall']:+.3f}  (good/bad)", flush=True)

    print(f"\nn={len(goods)}   gap = mean(good) - mean(bad)   (bad = heavy noise + shuffled + dropped frames)")
    any_pass = False
    for d in dims:
        gm, bm = statistics.mean(good[d]), statistics.mean(bad[d])
        gap = gm - bm
        ok = gap > _EVAL_NOISE
        any_pass = any_pass or (ok and d in ("VQ", "MQ"))
        print(f"  {d:8s} good={gm:+.4f}  bad={bm:+.4f}  gap={gap:+.4f}  "
              f"{'separates' if ok else 'within noise'} (vs +/-{_EVAL_NOISE})")
    print(
        f"\nVERDICT: {'PASS — quality dims separate good/bad at 480p' if any_pass else 'FAIL — even VQ/MQ cannot tell a degraded clip from a real one at 480p; the flat training reward is the reward model at this resolution, NOT too-few-epochs. Fix resolution/rescale before any longer run.'}"
    )


if __name__ == "__main__":
    main()
