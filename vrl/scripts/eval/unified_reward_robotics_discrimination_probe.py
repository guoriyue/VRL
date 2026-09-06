"""Fail-closed robotics discrimination gate for UnifiedReward-2.0.

The gate scores real DROID target clips and deterministic reward-hack variants
with the same caption-conditioned ``unified_reward_video`` model used by
training. It records the raw ``alignment`` and ``physics`` axes plus the actual
training compound, ``alignment+physics``. A reward is safe for a long robotics
run only when real clips beat semantic-free, static, temporally broken, and
wrong-instruction candidates by explicit margins.

The reward-domain module owns the adversarial candidate battery and acceptance
policy; this script only materializes media, calls the service, and writes the
report.

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
import tempfile
import uuid
from pathlib import Path
from typing import Any

import torch

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
)
from vrl.scripts.eval.robotics_discrimination import (
    aggregate_axis_scores,
    build_discrimination_candidates,
    require_robotics_scores,
    robotics_reward_verdict,
)
from vrl.trainers.data.prompts import PromptExample, load_prompt_manifest
from vrl.utils.artifacts import default_data_root, resolve_artifact_path, sha256_file
from vrl.utils.media import read_video_frames, write_mp4


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
            sample_id=artifact_id,
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
        candidate_scores = require_robotics_scores(result.scores)
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
                "generation_overlap_safe": info.generation_overlap_safe,
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
