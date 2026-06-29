"""KILL-RISK discrimination probe for Future Reward candidates (SPRINT_future_reward S4).

The single gate every reward must pass before it is allowed into Video2World RL.
It is the harness that killed the pixel-L1 ``target_video_similarity`` reward: take
a real DROID target clip and build a battery of degenerate "generations" *from that
same clip*, then check that the reward ranks the true clip well above every cheap
reward-hack. A reward that cannot separate the exact clip from a blurred temporal
mean, a frozen frame, a shuffle, or a different episode is not a Future Reward.

Candidates built from each anchor clip (real future ``frames``):

* ``exact``           - the real clip (re-encoded)              -> must score highest
* ``perceptual_blur`` - heavily blurred temporal mean           -> the pixel-L1 optimum
* ``temporal_mean``   - per-pixel mean over time, repeated      -> static, detail-free
* ``static_frozen``   - first frame repeated                    -> zero motion
* ``frame_shuffle``   - frames randomly permuted                -> time-order destroyed
* ``reverse``         - frames time-reversed                    -> dynamics reversed
* ``wrong_clip``      - a *different* episode's clip, but scored against THIS clip's
                        target / actions                        -> decisive: real motion,
                                                                   wrong instruction
* ``random``          - uniform noise                           -> floor

PASS rules (per reward family):

* main signals (target_video_similarity / target_dino_similarity): ``exact`` is the
  max, and ``exact - max(perceptual_blur, static_frozen, temporal_mean)`` is >= 25%
  of the metric dynamic range. pixel-L1 fails this (~4%).
* target_dino_similarity also: blur and temporal_mean each >= 0.15 cosine below exact.
* motion_dynamics (quality guard): static_frozen collapses to the floor and exact is
  far above it. (Order is not its job, so shuffle/reverse may score high.)

Usage::

  python -m vrl.scripts.eval.future_reward_discrimination_probe \
    --reward target_dino_similarity \
    --manifest data/external/video_world/manifests/droid_targets_eval.jsonl \
    --out outputs/future_reward_probe_dino.jsonl --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

import torch

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.trainers.data.artifacts import default_data_root, resolve_artifact_path
from vrl.trainers.data.prompts import load_prompt_manifest
from vrl.utils.media import align_frame_counts, read_video_frames, write_mp4

# reward name -> (model module, model class, primary score key)
_REWARDS: dict[str, tuple[str, str, str]] = {
    "target_video_similarity": (
        "vrl.rewards.models.target_video_similarity",
        "TargetVideoSimilarityModel",
        "target_similarity",
    ),
    "target_dino_similarity": (
        "vrl.rewards.models.target_dino_similarity",
        "TargetDinoSimilarityModel",
        "target_dino_similarity",
    ),
    "motion_dynamics": (
        "vrl.rewards.models.motion_dynamics",
        "MotionDynamicsModel",
        "motion_dynamics",
    ),
}

_HACKS = ("perceptual_blur", "static_frozen", "temporal_mean")
# Candidates a *main* signal must rank below exact (wrong_clip included; random is
# the floor used to size the dynamic range).
_MAIN_RANKED = ("perceptual_blur", "static_frozen", "temporal_mean", "frame_shuffle", "reverse", "wrong_clip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", required=True, choices=sorted(_REWARDS))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-anchors", type=int, default=0, help="0 = all manifest rows")
    parser.add_argument(
        "--reward-kwargs",
        type=str,
        default="",
        help='JSON merged into the model worker_config, e.g. {"temporal_weight": 0.5}',
    )
    return parser


def _build_candidates(
    frames: torch.Tensor,
    other: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build the degenerate-generation battery from one real clip ``frames`` ([T,H,W,3])."""

    from torchvision.transforms.functional import gaussian_blur

    generator = torch.Generator().manual_seed(seed)
    count = frames.shape[0]
    mean_clip = frames.mean(dim=0, keepdim=True).repeat(count, 1, 1, 1)
    blurred_mean = (
        gaussian_blur(mean_clip.permute(0, 3, 1, 2), kernel_size=[31, 31], sigma=[9.0, 9.0])
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    candidates = {
        "exact": frames,
        "perceptual_blur": blurred_mean,
        "temporal_mean": mean_clip,
        "static_frozen": frames[0:1].repeat(count, 1, 1, 1),
        "frame_shuffle": frames.index_select(0, torch.randperm(count, generator=generator)),
        "reverse": frames.flip(0),
        "random": torch.rand(frames.shape, generator=generator),
    }
    if other.numel():
        aligned, _ = align_frame_counts(other, frames)
        candidates["wrong_clip"] = aligned
    return candidates


def _score_anchor(
    model: Any,
    score_key: str,
    candidates: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any],
    prompt: str,
    fps: float,
    tmp: Path,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, clip in candidates.items():
        path = tmp / f"{name}.mp4"
        write_mp4(clip.permute(3, 0, 1, 2), path, fps=fps)
        artifact = RewardInferenceArtifact(
            artifact_id=f"{name}",
            path=str(path),
            media_type="video",
            prompt=prompt,
            metadata=dict(metadata),
        )
        result = dict(model(artifact=artifact, request=None))
        scores[name] = float(result[score_key])
    return scores


def _aggregate(per_anchor: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = sorted({name for row in per_anchor for name in row})
    out: dict[str, dict[str, float]] = {}
    for name in names:
        values = [row[name] for row in per_anchor if name in row]
        out[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return out


def _verdict(reward: str, agg: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the per-family PASS rule, returning the decision plus the numbers behind it."""

    def mean(name: str) -> float:
        return agg[name]["mean"] if name in agg else float("nan")

    exact = mean("exact")
    floor = min(stat["mean"] for stat in agg.values())
    dynamic_range = exact - floor
    hack_max = max(mean(name) for name in _HACKS)
    gap = exact - hack_max
    gap_ratio = gap / dynamic_range if dynamic_range > 0 else 0.0
    exact_is_top = all(exact >= mean(name) for name in _MAIN_RANKED if name in agg)

    reasons: list[str] = []
    if reward in {"target_video_similarity", "target_dino_similarity"}:
        passed = exact_is_top and gap_ratio >= 0.25
        reasons.append(f"exact_is_top={exact_is_top}")
        reasons.append(f"gap_ratio={gap_ratio:.3f} (need >=0.25)")
        if reward == "target_dino_similarity":
            blur_drop = exact - mean("perceptual_blur")
            tmean_drop = exact - mean("temporal_mean")
            dino_ok = blur_drop >= 0.15 and tmean_drop >= 0.15
            passed = passed and dino_ok
            reasons.append(f"blur_drop={blur_drop:.3f}, tmean_drop={tmean_drop:.3f} (need >=0.15)")
    elif reward == "motion_dynamics":
        static = mean("static_frozen")
        static_floored = static <= 0.15
        exact_above = exact >= 2 * static and (exact - static) >= 0.25 * dynamic_range
        passed = static_floored and exact_above
        reasons.append(f"static_frozen={static:.3f} (need <=0.15, collapse to floor)")
        reasons.append(f"exact={exact:.3f} far above static={static:.3f}: {exact_above}")
    else:  # pragma: no cover - choices() guards this
        passed = False

    return {
        "passed": bool(passed),
        "exact": exact,
        "floor": floor,
        "dynamic_range": dynamic_range,
        "hack_max": hack_max,
        "gap": gap,
        "gap_ratio": gap_ratio,
        "exact_is_top": exact_is_top,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_prompt_manifest(args.manifest)
    if args.num_anchors:
        examples = examples[: args.num_anchors]
    if len(examples) < 2:
        raise ValueError("discrimination probe needs >=2 manifest rows (for wrong_clip)")

    worker_config: dict[str, Any] = {"data_root": str(data_root), "device": device}
    if args.reward_kwargs:
        worker_config.update(json.loads(args.reward_kwargs))
    module_path, class_name, score_key = _REWARDS[args.reward]
    model_cls = getattr(importlib.import_module(module_path), class_name)
    model = model_cls(worker_config)

    # Pre-resolve target paths + decode real clips once (also the wrong_clip source).
    clips: list[torch.Tensor] = []
    for example in examples:
        if not example.target_video:
            raise ValueError(f"manifest row {example.prompt!r} has no target_video")
        path = resolve_artifact_path(example.target_video, data_root=data_root, allow_absolute=True)
        frames = read_video_frames(path)
        if frames.shape[0] < 2:
            # 1-frame clips make every temporal candidate identical to exact (mean of
            # one frame is itself, flip/shuffle are no-ops), so gap collapses to 0 and a
            # good reward would falsely FAIL. The probe needs real motion to discriminate.
            raise ValueError(
                f"anchor {example.target_video!r} has {frames.shape[0]} frame(s); "
                "the discrimination probe needs >=2 frames per clip",
            )
        clips.append(frames)

    rows: list[dict[str, Any]] = []
    per_anchor: list[dict[str, float]] = []
    for index, example in enumerate(examples):
        other = clips[(index + 1) % len(clips)]
        candidates = _build_candidates(clips[index], other, seed=1234 + index)
        metadata = dict(example.metadata)
        metadata["target_video"] = example.target_video
        fps = float(example.metadata.get("source_fps", 15.0))
        with tempfile.TemporaryDirectory(prefix="future-reward-probe-") as tmp_name:
            scores = _score_anchor(
                model,
                score_key,
                candidates,
                metadata=metadata,
                prompt=example.prompt,
                fps=fps,
                tmp=Path(tmp_name),
            )
        per_anchor.append(scores)
        rows.append({"row_index": index, "prompt": example.prompt, "scores": scores})

    agg = _aggregate(per_anchor)
    verdict = _verdict(args.reward, agg)
    summary = {
        "reward": args.reward,
        "score_key": score_key,
        "anchors": len(examples),
        "device": device,
        "aggregate": agg,
        "verdict": verdict,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.write(json.dumps({"summary": summary}, sort_keys=True) + "\n")

    _print_report(summary, agg)


def _print_report(summary: dict[str, Any], agg: dict[str, dict[str, float]]) -> None:
    print(f"\n=== Future Reward discrimination probe: {summary['reward']} "
          f"(score_key={summary['score_key']}, anchors={summary['anchors']}) ===")
    for name in sorted(agg, key=lambda key: -agg[key]["mean"]):
        stat = agg[name]
        print(f"  {name:16s}  mean={stat['mean']:+.4f}  std={stat['std']:.4f}  n={stat['n']}")
    verdict = summary["verdict"]
    print(f"  dynamic_range={verdict['dynamic_range']:.4f}  gap={verdict['gap']:.4f}  "
          f"gap_ratio={verdict['gap_ratio']:.3f}")
    for reason in verdict["reasons"]:
        print(f"    - {reason}")
    print(f"  VERDICT: {'PASS' if verdict['passed'] else 'FAIL'}\n")


if __name__ == "__main__":
    main()
