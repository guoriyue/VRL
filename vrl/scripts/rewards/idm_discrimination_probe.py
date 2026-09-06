"""Score the reward-hacking candidate battery with a trained DROID IDM.

SPRINT_idm_action_following_reward Phase 3 gate. For every eval clip this
builds the deterministic candidate set (``build_discrimination_candidates``:
exact / perceptual_blur / temporal_mean / static_frozen / frame_shuffle /
reverse / random / color_blob / wrong_clip), scores each with the IDM
action-following reward, and prints per-candidate mean/std plus the sprint's
reference thresholds. The verdict is HUMAN-read by design (the sprint retired
the automatic PASS/FAIL gate); this script only lays out the evidence:

  - exact >= 0.7, and static_frozen / perceptual_blur / frame_shuffle /
    reverse / wrong_clip all <= 0.4;
  - exact above wrong_clip by > 2 sigma (the decisive test — wrong_clip's
    motion is real, only the commands differ);
  - static_frozen or perceptual_blur within 0.1 of exact means the IDM is
    reading global appearance, exactly the pixel-L1 failure — retrain.

Run:
    python -m vrl.scripts.rewards.idm_discrimination_probe \
        --checkpoint outputs/idm/droid_idm.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vrl.scripts.eval.robotics_discrimination import build_discrimination_candidates
from vrl.rewards.models.idm_action_following import load_idm_checkpoint, score_action_following
from vrl.utils.media import read_video_frames

_DEFAULT_MANIFEST = Path("data/external/video_world/manifests/droid_targets_eval.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-key", default="action_match")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    idm = load_idm_checkpoint(str(args.checkpoint), device=args.device)

    rows = [json.loads(line) for line in args.eval_manifest.read_text().splitlines()]
    rows = [row for row in rows if row.get("metadata", {}).get("target_actions")]
    if len(rows) < 2:
        raise SystemExit("need >= 2 eval rows (wrong_clip pairs each clip with another)")
    root = args.eval_manifest.parent.parent.parent

    clips = [read_video_frames(root / row["target_video"]) for row in rows]
    actions = [
        torch.as_tensor(row["metadata"]["target_actions"], dtype=torch.float32) for row in rows
    ]

    per_candidate: dict[str, list[float]] = {}
    for index, (frames, target) in enumerate(zip(clips, actions, strict=True)):
        other = clips[(index + 1) % len(clips)]
        candidates = build_discrimination_candidates(frames, other, seed=args.seed + index)
        for name, candidate_frames in candidates.items():
            score = score_action_following(candidate_frames, target, idm)[args.score_key]
            per_candidate.setdefault(name, []).append(score)

    exact = per_candidate.get("exact", [])
    print(f"clips={len(rows)}  checkpoint={args.checkpoint}  key={args.score_key}\n")
    print(f"{'candidate':>16} | {'mean':>6} | {'std':>6} | {'vs exact':>8}")
    print("-" * 48)
    exact_mean = statistics.mean(exact) if exact else float("nan")
    for name, scores in sorted(per_candidate.items(), key=lambda kv: -statistics.mean(kv[1])):
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{name:>16} | {mean:6.3f} | {std:6.3f} | {mean - exact_mean:+8.3f}")

    wrong = per_candidate.get("wrong_clip", [])
    if exact and wrong:
        pooled = statistics.stdev(exact + wrong)
        sigma_gap = (exact_mean - statistics.mean(wrong)) / max(pooled, 1e-6)
        print(
            f"\nexact - wrong_clip = {exact_mean - statistics.mean(wrong):+.3f} ({sigma_gap:.1f} sigma)"
        )
    print(
        "\nsprint reference: exact >= 0.7; static_frozen/perceptual_blur/frame_shuffle/"
        "reverse/wrong_clip <= 0.4; exact within 0.1 of static_frozen or perceptual_blur "
        "= appearance-reading IDM, retrain."
    )


if __name__ == "__main__":
    main()
