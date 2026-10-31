"""Gate full-layer integrity on real truth scenes before an RL pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = Path("data/external/coco/layered_val2017/train_mixed_v3_rewardfix.jsonl")
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    full = []
    for row in rows:
        meta = row["metadata"]
        meta["complete_layer_truth"] = meta["source"] == "layered_v3"
        if meta["complete_layer_truth"]:
            root = Path(meta["truth_dir"])
            info = json.loads((root / "render_manifest.json").read_text())
            assert len(info["layers_back_to_front"]) == row["request_overrides"]["layers"]
            assert len(list(root.glob("layer_*.png"))) == row["request_overrides"]["layers"]
            full.append(row)
    corrected = args.output / "train_integrity.jsonl"
    corrected.write_text("".join(json.dumps(r) + "\n" for r in rows))
    heldout = [
        json.loads(s)
        for s in Path("/mnt/nvme/layering/eval_inputs/all_rows.jsonl").read_text().splitlines()
    ]
    for row in heldout:
        row["metadata"]["complete_layer_truth"] = row["id"].startswith("rendered_")
    (args.output / "eval_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in heldout))
    # Compare genuine truth, not an arbitrary pixel drawing, under hidden edits.
    scenes = [
        {"id": f"train_{i}", "scene_dir": r["metadata"]["truth_dir"], "metadata": r["metadata"]}
        for i, r in enumerate(full[:12])
    ]
    scenes += [r for r in heldout if r["id"].startswith("rendered_")]
    scorer = LayeredGroupingRewardModel({"integrity_weight": 1})
    checks = []
    for row in scenes:
        ref = LayeredReference.from_metadata(
            {**row["metadata"], "reference_images": [str(Path(row["scene_dir"]) / "source.png")]},
            640,
            640,
        )
        truth = ref.truth
        selected = max(
            range(1, len(truth.rgba)), key=lambda k: np.count_nonzero(truth.rgba[k, ..., 3] == 1)
        )
        mask = truth.rgba[selected, ..., 3] == 1
        assert mask.any()
        duplicate = truth.rgba[selected].copy()
        duplicate[~mask] = 0
        copied = np.concatenate([truth.rgba[:selected], duplicate[None], truth.rgba[selected:]])
        residual = truth.rgba.copy()
        residual[0, mask, :3] = truth.rgba[selected, mask, :3]
        base = scorer.score(truth, ref)
        report = {"scene": row["id"], "correct": base}
        for name, rgba in [("hidden_duplicate", copied), ("background_residual", residual)]:
            stack = LayerStack.from_rgba(rgba)
            scores = scorer.score(stack, ref)
            assert np.max(np.abs(stack.composite() - truth.composite())) < 1e-6
            assert scores["layered_grouping"] < base["layered_grouping"] - 1e-6, (row["id"], name)
            report[name] = scores
        checks.append(report)
    report = {
        "complete_truth_rows": len(full),
        "partial_or_photo_rows": len(rows) - len(full),
        "counterexample_scenes": len(checks),
        "cases": checks,
        "all_hidden_edits_lower_reward": True,
        "train_manifest": str(corrected.resolve()),
        "manifest_sha256": hashlib.sha256(corrected.read_bytes()).hexdigest(),
        "scope": "12 training scenes plus 8 held-out rendered scenes. No claim of real-photo duplicate detection.",
    }
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "cases"})


if __name__ == "__main__":
    main()
