"""Diagnose editable layers against complete, known RGBA references.

This is a small-fixture reward candidate, not a semantic judge for real photos.
By default reference layer zero is treated as background. Object-only stacks
can explicitly include it in edit diagnostics. Transparent RGB is ignored.
No model pixels are repaired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vrl.rewards.models.layered_score import (  # noqa: F401 - re-exported for callers
    completeness_priority_reward,
    composite,
    constrained_fidelity_reward,
    coverage_fidelity_reward,
    match_layers,
    premultiplied,
    score_layers,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stacks = []
    for directory in (args.reference, args.candidate):
        paths = sorted(directory.glob("layer_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        if not paths:
            raise ValueError(f"No RGBA layers in {directory}")
        stacks.append([np.asarray(Image.open(path).convert("RGBA")) / 255 for path in paths])
    result = score_layers(*stacks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
