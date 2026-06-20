"""#7 follow-up: WHY is Kling reward flat at 480p_33f — resolution or OOD rollouts?

The first probe (kling_480p_discrimination_probe) showed real-vs-heavily-degraded
does not separate. But that degradation mixed noise + frame-shuffle + frame-drop +
an fps change, any of which could confound. This isolates a clean axis to tell two
root causes apart:

  H1 (reward blind at 480p): a pure gaussian-noise ladder on a real rollout does
      NOT lower the score -> the reward cannot see visual degradation at this
      resolution. Fix = raise resolution / rescale before scoring.
  H2 (rollouts uniformly OOD): the noise ladder DOES lower the score monotonically
      (reward works at 480p), but real rollouts cluster at one score -> there is no
      quality gradient among them for RL to climb. Fix is about the policy/reward
      scaling, not resolution.

Sharded for multi-GPU: --shard i/n scores every n-th video starting at i, writes a
JSON; a separate --combine pass merges shard JSONs and prints the verdict.

  # GPU 0 (node A) and GPU 0 (node B) in parallel:
  python -m vrl.scripts.eval.kling_reward_diagnosis_probe --videos DIR --shard 0/2 --out a.json
  python -m vrl.scripts.eval.kling_reward_diagnosis_probe --videos DIR --shard 1/2 --out b.json
  python -m vrl.scripts.eval.kling_reward_diagnosis_probe --combine a.json b.json
"""

from __future__ import annotations

import os

os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"  # torchvision lacks io.read_video

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np

_EVAL_NOISE = 0.0339
_PROMPT = "A short video of a physical interaction between objects."
_LEVELS = [0, 20, 40, 80, 160]  # gaussian noise sigma (0 = re-encoded original)
_DIMS = ["VQ", "MQ", "Overall"]


def _noised(frames: np.ndarray, sigma: float, fps: float, dst: Path) -> None:
    """Same frames + fps; only add gaussian noise (no shuffle/drop/fps confound)."""
    if sigma <= 0:
        out = frames
    else:
        rng = np.random.default_rng(int(sigma))
        out = np.clip(frames.astype(np.float32) + rng.normal(0, sigma, frames.shape), 0, 255).astype(
            np.uint8,
        )
    iio.imwrite(dst, out, fps=fps)


def _run_shard(args: argparse.Namespace) -> None:
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    i, n = (int(x) for x in args.shard.split("/"))
    allv = sorted(Path(args.videos).glob("*.mp4"))
    mine = [v for k, v in enumerate(allv) if k % n == i][: args.n]
    model = KlingVideoRewardModel(
        {"reward_model_name": "KlingTeam/VideoReward@main", "device": "cuda:0", "dtype": "bfloat16"},
    )

    def score(path: Path) -> dict[str, float]:
        return model._reward([str(path.resolve())], [_PROMPT], use_norm=True)[0]  # noqa: SLF001

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for v in mine:
            frames = np.asarray(iio.imread(v))
            meta = iio.immeta(v)
            fps = float(meta.get("fps", 8.0))
            rec = {"video": v.name, "levels": {}}
            for s in _LEVELS:
                p = Path(tmp) / f"{v.stem}_s{s}.mp4"
                _noised(frames, s, fps, p)
                rec["levels"][str(s)] = score(p)
            # determinism: re-score the clean re-encode a 2nd time
            p0 = Path(tmp) / f"{v.stem}_s0.mp4"
            rec["repeat0"] = score(p0)
            rows.append(rec)
            print(f"  [{args.shard}] {v.name} scored {len(_LEVELS)} levels", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"  [{args.shard}] wrote {len(rows)} rows -> {args.out}", flush=True)


def _combine(paths: list[str]) -> None:
    rows = []
    for p in paths:
        rows.extend(json.loads(Path(p).read_text()))
    print(f"combine: {len(rows)} videos x {len(_LEVELS)} noise levels\n")

    # determinism: max |score(level0) - repeat(level0)| across videos/dims
    det = max(
        abs(r["levels"]["0"][d] - r["repeat0"][d]) for r in rows for d in _DIMS
    )
    print(f"determinism (max |rescore - score| at sigma=0) = {det:.4f}  "
          f"({'deterministic — gaps below are real model behavior' if det < 1e-3 else 'NON-deterministic!'})\n")

    print(f"{'noise sigma':>11} | " + " | ".join(f"{d:>8}" for d in _DIMS))
    means = {d: [] for d in _DIMS}
    for s in _LEVELS:
        line = f"{s:>11} | "
        cells = []
        for d in _DIMS:
            vals = [r["levels"][str(s)][d] for r in rows]
            m = statistics.mean(vals)
            means[d].append(m)
            cells.append(f"{m:+8.4f}")
        print(line + " | ".join(cells))

    print()
    # classify each dim: noise should LOWER a faithful quality score.
    #   drop > +noise -> RESPONDS (correct sign); |drop| <= noise -> FLAT (blind);
    #   drop < -noise -> INVERTED (noise RAISES the score — gameable, worse than blind)
    cls = {}
    for d in _DIMS:
        drop = means[d][0] - means[d][-1]
        cls[d] = "RESPONDS" if drop > _EVAL_NOISE else ("INVERTED" if drop < -_EVAL_NOISE else "FLAT")
        tag = {"RESPONDS": "RESPONDS to noise (correct)",
               "FLAT": "FLAT (blind to noise)",
               "INVERTED": "INVERTED (noise RAISES it — gameable)"}[cls[d]]
        print(f"  {d:8s} clean(s0)={means[d][0]:+.4f}  noisiest(s{_LEVELS[-1]})={means[d][-1]:+.4f}  "
              f"drop={drop:+.4f}  {tag}")

    # cross-rollout spread at clean (is there a quality gradient among real rollouts?)
    print("\ncross-rollout spread at sigma=0 (std of real-rollout scores; vs eval noise "
          f"{_EVAL_NOISE}):")
    for d in _DIMS:
        vals = [r["levels"]["0"][d] for r in rows]
        print(f"  {d:8s} std={statistics.pstdev(vals):.4f}  range=[{min(vals):+.3f}, {max(vals):+.3f}]")

    print("\nVERDICT:")
    if cls["VQ"] == "RESPONDS" and cls.get("Overall") != "RESPONDS":
        print("  MIS-WEIGHTED reward (not pure resolution-blindness): VQ responds correctly to noise, "
              f"but MQ is {cls['MQ']} and Overall is {cls['Overall']} — the motion sub-score corrupts the "
              "trained 'overall_reward' target. Switch the training score_key to a correctly-signed dim "
              "(visual_quality) and/or restore frames toward 93f so MQ works; re-run this probe and require "
              "the trained key to RESPOND before any longer run.")
    elif cls["VQ"] == "FLAT" and cls["MQ"] == "FLAT":
        print("  H1 — reward BLIND at 480p_33f: a clean noise ladder does not move VQ/MQ. "
              "Fix resolution/rescale, not epochs.")
    else:
        print("  Reward responds to noise on the quality dims; if training is still flat, the real "
              "rollouts lack a quality gradient (see cross-rollout spread) — policy/reward-scaling, not resolution.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", help="dir of real rollout .mp4")
    ap.add_argument("--shard", help="i/n — score every n-th video starting at i")
    ap.add_argument("--n", type=int, default=999, help="cap videos in this shard")
    ap.add_argument("--out", help="shard JSON output path")
    ap.add_argument("--combine", nargs="+", help="merge shard JSONs and print verdict")
    args = ap.parse_args()
    if args.combine:
        _combine(args.combine)
    else:
        _run_shard(args)


if __name__ == "__main__":
    main()
