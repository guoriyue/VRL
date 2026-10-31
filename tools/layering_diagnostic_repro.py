"""Compare repeated original-budget 10-step outputs with the prior evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text())
    comparisons = []
    for arm in ("v3c_28", "v3e_44"):
        for scene in plan["scenes"]:
            old = args.previous / "eval" / arm / scene["id"] / "seed4401"
            if not old.exists():
                continue
            new = args.root / "eval" / arm / scene["id"] / f"layers{scene['layers']}_steps10"
            assert (new / "result.json").exists(), new
            rows = []
            for i in range(scene["layers"]):
                name = f"layer_{i}.png"
                left, right = old / name, new / name
                a, b = np.asarray(Image.open(left)), np.asarray(Image.open(right))
                assert a.shape == b.shape == (640, 640, 4)
                delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
                rows.append(
                    {
                        "layer": i,
                        "bytes_identical": left.read_bytes() == right.read_bytes(),
                        "pixels_identical": bool(np.array_equal(a, b)),
                        "rgba_mae_0_255": float(delta.mean()),
                        "max_channel_delta_0_255": float(delta.max()),
                        "old_sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
                        "new_sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
                    }
                )
            comparisons.append({"arm": arm, "scene": scene["id"], "layers": rows})
    output = {
        "comparisons": comparisons,
        "all_pixels_identical": all(
            r["pixels_identical"] for c in comparisons for r in c["layers"]
        ),
        "compared_outputs": len(comparisons),
    }
    assert len(comparisons) == 6
    (args.root / "reproducibility.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
