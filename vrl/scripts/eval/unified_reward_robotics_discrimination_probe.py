"""Fail-closed robotics discrimination gate for UnifiedReward-2.0.

The gate scores real DROID target clips and deterministic reward-hack variants
with the same caption-conditioned ``unified_reward_video`` model used by
training. It records the raw ``alignment`` and ``physics`` axes plus the actual
training compound, ``alignment+physics``. A reward is safe for a long robotics
run only when real clips beat semantic-free, static, temporally broken, and
wrong-instruction candidates by explicit margins.

``build_discrimination_candidates`` below defines the adversarial candidate
battery shared by any future reward gate.

Example (reuse the UnifiedReward service already pinned to physical GPU 3)::

  python -m vrl.scripts.eval.unified_reward_robotics_discrimination_probe \
    --manifest outputs/droid/video_world/manifests/droid_eval.jsonl \
    --data-root outputs/droid \
    --out outputs/gates/unified_reward_robotics.json \
    --endpoint http://127.0.0.1:8300 --num-anchors 4

The process exits with status 2 on a discrimination failure so launch scripts
can use it as a hard prerequisite rather than relying on a human reading logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    sha256_file,
)
from vrl.trainers.data.prompts import PromptExample, load_prompt_manifest
from vrl.utils.artifacts import default_data_root, resolve_artifact_path
from vrl.utils.media import align_frame_counts, read_video_frames, write_mp4


@dataclass(frozen=True, slots=True)
class RoboticsRewardGatePolicy:
    """Acceptance policy on the public UnifiedReward 1-5 axis scale."""

    exact_axis_min: float = 3.0
    low_information_axis_max: float = 2.5
    exact_low_information_compound_gap: float = 1.0
    exact_adversary_compound_gap: float = 0.5
    wrong_clip_alignment_gap: float = 0.5
    frame_shuffle_physics_gap: float = 0.5
    anchor_pass_rate_min: float = 0.75


_AXES = ("alignment", "physics", "alignment+physics")
_LOW_INFORMATION = (
    "perceptual_blur",
    "temporal_mean",
    "static_frozen",
    "random",
    "color_blob",
)
_REQUIRED_CANDIDATES = (
    "exact",
    *_LOW_INFORMATION,
    "frame_shuffle",
    "reverse",
    "wrong_clip",
)


def build_discrimination_candidates(
    frames: torch.Tensor,
    other: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build the degenerate-generation battery from one real clip ``frames`` ([T,H,W,3])."""

    from torchvision.transforms.functional import gaussian_blur

    generator = torch.Generator().manual_seed(seed)
    count = frames.shape[0]
    height, width = frames.shape[1:3]
    mean_clip = frames.mean(dim=0, keepdim=True).repeat(count, 1, 1, 1)
    blurred_mean = (
        gaussian_blur(mean_clip.permute(0, 3, 1, 2), kernel_size=[31, 31], sigma=[9.0, 9.0])
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    blob_keyframes = torch.rand((2, 3, 4, 4), generator=generator)
    blob_keyframes = torch.nn.functional.interpolate(
        blob_keyframes,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    blend = torch.linspace(0.0, 1.0, count).reshape(count, 1, 1, 1)
    color_blob = (
        blob_keyframes[0].permute(1, 2, 0).unsqueeze(0) * (1.0 - blend)
        + blob_keyframes[1].permute(1, 2, 0).unsqueeze(0) * blend
    )
    candidates = {
        "exact": frames,
        "perceptual_blur": blurred_mean,
        "temporal_mean": mean_clip,
        "static_frozen": frames[0:1].repeat(count, 1, 1, 1),
        "frame_shuffle": frames.index_select(0, torch.randperm(count, generator=generator)),
        "reverse": frames.flip(0),
        "random": torch.rand(frames.shape, generator=generator),
        "color_blob": color_blob,
    }
    if other.numel():
        aligned, _ = align_frame_counts(other, frames)
        candidates["wrong_clip"] = aligned
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8300")
    parser.add_argument("--expected-model", default="unified-reward-robotics")
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Keep generated candidate MP4s for visual audit; default uses a temporary directory.",
    )
    return parser


def aggregate_axis_scores(
    anchors: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Aggregate each candidate/axis while retaining extrema for fail-closed gates."""

    candidates = sorted({candidate for anchor in anchors for candidate in anchor})
    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    for candidate in candidates:
        aggregate[candidate] = {}
        for axis in _AXES:
            values = [anchor[candidate][axis] for anchor in anchors if candidate in anchor]
            if not values:
                continue
            aggregate[candidate][axis] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
    return aggregate


def robotics_reward_verdict(
    anchors: list[dict[str, dict[str, float]]],
    aggregate: dict[str, dict[str, dict[str, float | int]]],
    *,
    policy: RoboticsRewardGatePolicy | None = None,
) -> dict[str, Any]:
    """Return a structured PASS/FAIL verdict for the robotics reward contract."""

    policy = policy or RoboticsRewardGatePolicy()
    missing = [name for name in _REQUIRED_CANDIDATES if name not in aggregate]
    if missing:
        raise ValueError(f"robotics reward gate missing candidates: {missing}")
    if not anchors:
        raise ValueError("robotics reward gate needs at least one scored anchor")

    def mean(candidate: str, axis: str) -> float:
        return float(aggregate[candidate][axis]["mean"])

    exact_alignment = mean("exact", "alignment")
    exact_physics = mean("exact", "physics")
    exact_compound = mean("exact", "alignment+physics")
    low_information_max = max(mean(name, "alignment+physics") for name in _LOW_INFORMATION)
    low_information_gap = exact_compound - low_information_max
    adversary_max = max(
        mean(name, "alignment+physics") for name in _REQUIRED_CANDIDATES if name != "exact"
    )
    adversary_gap = exact_compound - adversary_max
    wrong_clip_gap = exact_alignment - mean("wrong_clip", "alignment")
    frame_shuffle_gap = exact_physics - mean("frame_shuffle", "physics")

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add_check(
        "exact_alignment_floor",
        exact_alignment >= policy.exact_axis_min,
        f"mean={exact_alignment:.3f}, need >= {policy.exact_axis_min:.3f}",
    )
    add_check(
        "exact_physics_floor",
        exact_physics >= policy.exact_axis_min,
        f"mean={exact_physics:.3f}, need >= {policy.exact_axis_min:.3f}",
    )
    add_check(
        "exact_low_information_compound_gap",
        low_information_gap >= policy.exact_low_information_compound_gap,
        f"gap={low_information_gap:.3f}, need >= {policy.exact_low_information_compound_gap:.3f}",
    )
    add_check(
        "exact_adversary_compound_gap",
        adversary_gap >= policy.exact_adversary_compound_gap,
        f"gap={adversary_gap:.3f}, need >= {policy.exact_adversary_compound_gap:.3f}",
    )
    add_check(
        "wrong_clip_alignment_gap",
        wrong_clip_gap >= policy.wrong_clip_alignment_gap,
        f"gap={wrong_clip_gap:.3f}, need >= {policy.wrong_clip_alignment_gap:.3f}",
    )
    add_check(
        "frame_shuffle_physics_gap",
        frame_shuffle_gap >= policy.frame_shuffle_physics_gap,
        f"gap={frame_shuffle_gap:.3f}, need >= {policy.frame_shuffle_physics_gap:.3f}",
    )

    high_axes: list[str] = []
    for candidate in _LOW_INFORMATION:
        for axis in ("alignment", "physics"):
            observed = float(aggregate[candidate][axis]["max"])
            if observed > policy.low_information_axis_max:
                high_axes.append(f"{candidate}.{axis}={observed:.3f}")
    add_check(
        "low_information_axes_ceiling",
        not high_axes,
        (
            f"all per-anchor scores <= {policy.low_information_axis_max:.3f}"
            if not high_axes
            else "too high: " + ", ".join(high_axes)
        ),
    )

    anchor_passes: list[bool] = []
    for scores in anchors:
        exact = scores["exact"]
        low_max = max(scores[name]["alignment+physics"] for name in _LOW_INFORMATION)
        adversary_max = max(
            scores[name]["alignment+physics"] for name in _REQUIRED_CANDIDATES if name != "exact"
        )
        low_axes_ok = all(
            scores[name][axis] <= policy.low_information_axis_max
            for name in _LOW_INFORMATION
            for axis in ("alignment", "physics")
        )
        anchor_passes.append(
            exact["alignment"] >= policy.exact_axis_min
            and exact["physics"] >= policy.exact_axis_min
            and exact["alignment+physics"] - low_max >= policy.exact_low_information_compound_gap
            and exact["alignment+physics"] - adversary_max >= policy.exact_adversary_compound_gap
            and exact["alignment"] - scores["wrong_clip"]["alignment"]
            >= policy.wrong_clip_alignment_gap
            and exact["physics"] - scores["frame_shuffle"]["physics"]
            >= policy.frame_shuffle_physics_gap
            and low_axes_ok
        )
    anchor_pass_rate = sum(anchor_passes) / len(anchor_passes)
    add_check(
        "anchor_pass_rate",
        anchor_pass_rate >= policy.anchor_pass_rate_min,
        f"rate={anchor_pass_rate:.3f}, need >= {policy.anchor_pass_rate_min:.3f}",
    )

    return {
        "passed": all(check["passed"] for check in checks),
        "policy": asdict(policy),
        "metrics": {
            "exact_alignment": exact_alignment,
            "exact_physics": exact_physics,
            "exact_alignment+physics": exact_compound,
            "exact_low_information_compound_gap": low_information_gap,
            "exact_adversary_compound_gap": adversary_gap,
            "wrong_clip_alignment_gap": wrong_clip_gap,
            "frame_shuffle_physics_gap": frame_shuffle_gap,
            "anchor_pass_rate": anchor_pass_rate,
        },
        "checks": checks,
    }


def _validated_scores(scores: dict[str, Any]) -> dict[str, float]:
    selected: dict[str, float] = {}
    for axis in ("alignment", "physics"):
        if axis not in scores:
            raise KeyError(f"UnifiedReward result missing {axis!r}: {sorted(scores)}")
        value = float(scores[axis])
        if not math.isfinite(value) or not 1.0 <= value <= 5.0:
            raise ValueError(f"UnifiedReward {axis} must be finite and in [1, 5], got {value}")
        selected[axis] = value
    selected["alignment+physics"] = selected["alignment"] + selected["physics"]
    return selected


async def _score_anchor(
    scorer: Any,
    candidates: dict[str, torch.Tensor],
    *,
    anchor_index: int,
    prompt: str,
    fps: float,
    candidate_root: Path,
) -> dict[str, dict[str, float]]:
    anchor_dir = candidate_root / f"anchor_{anchor_index:03d}"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[RewardInferenceArtifact] = []
    candidate_by_artifact: dict[str, str] = {}
    for name, clip in candidates.items():
        path = (anchor_dir / f"{name}.mp4").resolve()
        write_mp4(clip.permute(3, 0, 1, 2), path, fps=fps)
        artifact_id = f"anchor-{anchor_index}:{name}"
        artifact = RewardInferenceArtifact(
            artifact_id=artifact_id,
            path=str(path),
            prompt=prompt,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        artifacts.append(artifact)
        candidate_by_artifact[artifact_id] = name

    request = RewardInferenceRequest(
        request_id=f"robotics-gate:{anchor_index}:{uuid.uuid4().hex}",
        artifacts=tuple(artifacts),
    )
    results = await scorer.score_batch(request)
    scores: dict[str, dict[str, float]] = {}
    for result in results:
        candidate = candidate_by_artifact.get(result.artifact_id)
        if candidate is None:
            raise ValueError(f"UnifiedReward returned unknown artifact {result.artifact_id!r}")
        candidate_scores = _validated_scores(result.scores)
        scores[candidate] = candidate_scores
    missing = sorted(set(candidates) - set(scores))
    if missing:
        raise ValueError(f"UnifiedReward returned no scores for candidates: {missing}")
    return scores


def _load_anchors(
    examples: list[PromptExample],
    *,
    data_root: Path,
) -> list[torch.Tensor]:
    clips: list[torch.Tensor] = []
    for example in examples:
        if not example.target_video:
            raise ValueError(f"manifest row {example.prompt!r} has no target_video")
        path = resolve_artifact_path(
            example.target_video,
            data_root=data_root,
            allow_absolute=True,
        )
        frames = read_video_frames(path)
        if frames.shape[0] < 2:
            raise ValueError(
                f"anchor {example.target_video!r} has {frames.shape[0]} frame(s); need >=2",
            )
        clips.append(frames)
    return clips


def _different_prompt_index(examples: list[PromptExample], index: int) -> int:
    prompt = examples[index].prompt.strip().casefold()
    for offset in range(1, len(examples)):
        other = (index + offset) % len(examples)
        if examples[other].prompt.strip().casefold() != prompt:
            return other
    raise ValueError("robotics reward gate needs at least two distinct prompts for wrong_clip")


async def _run_probe(
    scorer: Any,
    examples: list[PromptExample],
    clips: list[torch.Tensor],
    *,
    candidate_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scored_anchors: list[dict[str, dict[str, float]]] = []
    for index, example in enumerate(examples):
        wrong_index = _different_prompt_index(examples, index)
        candidates = build_discrimination_candidates(
            clips[index],
            clips[wrong_index],
            seed=20260721 + index,
        )
        scores = await _score_anchor(
            scorer,
            candidates,
            anchor_index=index,
            prompt=example.prompt,
            fps=float(example.metadata.get("source_fps", 15.0)),
            candidate_root=candidate_root,
        )
        scored_anchors.append(scores)
        rows.append(
            {
                "row_index": index,
                "prompt": example.prompt,
                "target_video": example.target_video,
                "wrong_clip_prompt": examples[wrong_index].prompt,
                "wrong_clip_target_video": examples[wrong_index].target_video,
                "scores": scores,
            },
        )
        print(f"scored anchor {index + 1}/{len(examples)}", flush=True)

    aggregate = aggregate_axis_scores(scored_anchors)
    verdict = robotics_reward_verdict(scored_anchors, aggregate)
    return rows, {"aggregate": aggregate, "verdict": verdict}


def _print_report(summary: dict[str, Any]) -> None:
    print("\n=== UnifiedReward robotics discrimination gate ===")
    for candidate, axes in summary["aggregate"].items():
        print(
            f"  {candidate:16s} alignment={axes['alignment']['mean']:.3f} "
            f"physics={axes['physics']['mean']:.3f} "
            f"alignment+physics={axes['alignment+physics']['mean']:.3f}",
        )
    print("  checks:")
    for check in summary["verdict"]["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"    {status:4s} {check['name']}: {check['detail']}")
    status = "PASS" if summary["verdict"]["passed"] else "FAIL"
    print(f"  VERDICT: {status}\n")


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_anchors < 2:
        raise ValueError("--num-anchors must be >=2 for wrong_clip discrimination")
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    examples = load_prompt_manifest(args.manifest)[: args.num_anchors]
    if len(examples) < 2:
        raise ValueError("robotics reward gate needs at least two manifest rows")
    clips = _load_anchors(examples, data_root=data_root)

    from vrl.rewards.service.client import HttpRewardScorer

    scorer = HttpRewardScorer(
        args.endpoint,
        expected_model=args.expected_model,
    )
    deployment = {
        "kind": "http",
        "endpoint": args.endpoint,
        "expected_model": args.expected_model,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        await scorer.ensure_ready()
        info = await scorer.info()
        deployment.update(
            {
                "model_name": info.model_name,
                "model_version": info.model_version,
                "capabilities": list(info.capabilities),
                "artifact_transport": info.artifact_transport,
            },
        )
        if args.candidate_dir:
            candidate_root = args.candidate_dir.expanduser().resolve()
            candidate_root.mkdir(parents=True, exist_ok=True)
            rows, summary = await _run_probe(
                scorer,
                examples,
                clips,
                candidate_root=candidate_root,
            )
            retained_candidates = str(candidate_root)
        else:
            with tempfile.TemporaryDirectory(
                prefix="unified-reward-robotics-gate-",
                dir=args.out.parent.resolve(),
            ) as tmp_name:
                rows, summary = await _run_probe(
                    scorer,
                    examples,
                    clips,
                    candidate_root=Path(tmp_name),
                )
            retained_candidates = None
    finally:
        await scorer.shutdown()

    return {
        "reward": "unified_reward_video",
        "score_key": "alignment+physics",
        "manifest": str(args.manifest.expanduser().resolve()),
        "data_root": str(data_root),
        "anchors": len(examples),
        "deployment": deployment,
        "candidate_dir": retained_candidates,
        "rows": rows,
        **summary,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = asyncio.run(_main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report)
    if not report["verdict"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
