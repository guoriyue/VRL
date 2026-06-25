"""Bridge the vendored PhyMotion scorer to a one-video -> JSON contract.

Run this INSIDE the PhyMotion conda env (see third_party/PhyMotion/README.md for
its SMPL + MuJoCo setup), pointing ``reward.kwargs.phymotion.worker_config
.phymotion_cmd`` at it::

    phymotion_cmd: "python vrl/scripts/eval/phymotion_score.py --video {video} --output {output}"

It imports ``astrolabe.rewards.smpl_physics_score`` from the vendored submodule,
scores one video, and writes ``{kinematic, contact, dynamic, overall, ...}`` JSON
to ``--output`` — exactly what ``vrl.rewards.models.phymotion`` reads back.

The heavy PhyMotion / MuJoCo / SMPL imports are deferred into ``main`` so this
file never imports them at module load (it is not part of the normal VRL test or
import graph).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_VENDORED_PHYMOTION = Path(__file__).resolve().parents[3] / "third_party" / "PhyMotion"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to the video to score.")
    parser.add_argument("--output", required=True, help="Path to write the scores JSON.")
    parser.add_argument("--prompt", default="", help="Optional caption for the video.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for SMPL recovery and scoring.",
    )
    parser.add_argument(
        "--phymotion-root",
        default=str(_VENDORED_PHYMOTION),
        help="Path to the vendored PhyMotion checkout (default: third_party/PhyMotion).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.phymotion_root).expanduser().resolve()
    if not (root / "astrolabe").is_dir():
        raise SystemExit(
            f"PhyMotion checkout not found at {root}; run "
            "`git submodule update --init third_party/PhyMotion` and set up its env.",
        )
    sys.path.insert(0, str(root))

    # Deferred: pulls SMPL-X + MuJoCo + the PhyMotion scoring stack.
    from astrolabe.rewards import smpl_physics_score

    score_fn = smpl_physics_score(args.device)
    results = score_fn([args.video], [args.prompt], metadata=None)
    scores = results[0] if isinstance(results, (list, tuple)) else results

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({str(k): float(v) for k, v in dict(scores).items()}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "scores": {k: float(v) for k, v in dict(scores).items()}}))


if __name__ == "__main__":
    main()
