"""One-shot probe: measure KlingTeam/VideoReward per-video scoring latency.

NOT a long-term asset. Answers a single question — "how much wall-clock does the
Kling video reward forward cost per video, and what epoch fraction does that
imply" — so the reward_execution sprint priority can be decided on real numbers
instead of the doc's stale 93ms/0.8% figure. Delete after the answer is recorded.

Runs the real `KlingVideoRewardModel.__call__` (batch=1, the only path the model
implements) over synthetic mp4s at the real rollout shape (240p, 33 frames), with
warm-up excluded. The reward model internally resizes every frame to its training
budget (min_frame_pixels=200704 ~= 448^2), so input 240p vs 512p barely moves the
forward cost — synthetic input is representative of in-distribution scoring.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel


def _write_synthetic_mp4(path: Path, *, width: int, height: int, frames: int, fps: int) -> None:
    import imageio.v2 as imageio

    rng = np.random.default_rng(0)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=6)
    try:
        for _ in range(frames):
            frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reward/kling_video_reward.yaml")
    ap.add_argument("--videos", type=int, default=8, help="synthetic videos to score")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--width", type=int, default=426)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--videos-per-epoch", type=int, default=72,
                    help="for the epoch-fraction projection (doc baseline: 6 groups x 12)")
    ap.add_argument("--epoch-seconds", type=float, default=13.4 * 60,
                    help="reference epoch wall-clock for the fraction projection")
    ap.add_argument("--workdir", default="outputs/_kling_latency_probe")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    worker_config = dict(cfg["reward"]["kwargs"]["kling_video_reward"]["worker_config"])
    print(f"[probe] worker_config = {worker_config}")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    n = args.warmup + args.videos
    print(f"[probe] writing {n} synthetic mp4s "
          f"({args.width}x{args.height}, {args.frames}f @ {args.fps}fps)")
    paths = []
    for i in range(n):
        p = work / f"clip_{i:03d}.mp4"
        if not p.exists():
            _write_synthetic_mp4(p, width=args.width, height=args.height,
                                 frames=args.frames, fps=args.fps)
        paths.append(p)

    print("[probe] loading Kling VideoReward model (this loads ~4.8G weights)...")
    t0 = time.perf_counter()
    model = KlingVideoRewardModel(worker_config)
    print(f"[probe] model loaded in {time.perf_counter() - t0:.1f}s")

    def make_artifact(i: int, path: Path) -> RewardInferenceArtifact:
        return RewardInferenceArtifact(
            artifact_id=f"probe-{i}",
            path=str(path),
            media_type="video",
            prompt="A person walking across a sunlit room.",
        )

    request = RewardInferenceRequest(
        request_id="probe",
        artifacts=tuple(make_artifact(i, p) for i, p in enumerate(paths)),
        reward_name="kling_video_reward",
        score_key="overall_reward",
        score_aggregation="sum",
    )

    timings_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for i, (artifact, _path) in enumerate(zip(request.artifacts, paths, strict=True)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = model(artifact=artifact, request=request)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - started) * 1000.0
        tag = "warmup" if i < args.warmup else "timed "
        if i >= args.warmup:
            timings_ms.append(ms)
        print(f"[probe] {tag} clip {i:03d}: {ms:8.1f} ms  scores={ {k: round(v, 3) for k, v in scores.items()} }")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    mean = statistics.mean(timings_ms)
    median = statistics.median(timings_ms)
    p90 = sorted(timings_ms)[int(0.9 * (len(timings_ms) - 1))]

    per_epoch_s = mean / 1000.0 * args.videos_per_epoch
    frac = per_epoch_s / args.epoch_seconds * 100.0

    print("\n========== RESULT ==========")
    print(f"per-video (batch=1):  mean {mean:.1f} ms | median {median:.1f} ms | p90 {p90:.1f} ms")
    print(f"GPU peak alloc:       {peak_gb:.2f} GB")
    print(f"reward / epoch:       {args.videos_per_epoch} videos x {mean:.0f}ms = {per_epoch_s:.1f}s")
    print(f"epoch fraction:       {per_epoch_s:.1f}s / {args.epoch_seconds:.0f}s = {frac:.2f}%")
    print("============================")


if __name__ == "__main__":
    main()
