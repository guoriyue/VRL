"""Rescore paired layer-budget probes and flag similar foregrounds for review."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tools.layering_diagnostic_eval import diagnostics
from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def similar_layers(stack):
    pairs = []
    for a, b in itertools.combinations(range(1, len(stack.rgba)), 2):
        x, y = stack.rgba[a], stack.rgba[b]
        overlap = np.minimum(x[..., 3], y[..., 3])
        area = float(overlap.sum())
        if area < 64:
            continue
        iou = area / float(np.maximum(x[..., 3], y[..., 3]).sum())
        error = float((np.abs(x[..., :3] - y[..., :3]).mean(-1) * overlap).sum() / area)
        if iou > 0.8 and error < 0.06:
            pairs.append({"layers": [a, b], "alpha_iou": iou, "overlap_rgb_mae": error})
    return pairs


def hidden_nonempty_layers(stack):
    """Flag foregrounds with opacity but almost no alpha-over contribution."""
    transmission = np.ones(stack.rgba.shape[1:3], dtype=np.float32)
    hidden = []
    for k in range(len(stack.rgba) - 1, 0, -1):
        alpha = stack.rgba[k, ..., 3]
        mass = float(alpha.sum(dtype=np.float64))
        visible = float((alpha * transmission).sum(dtype=np.float64))
        if mass >= 64 and visible < 1:
            hidden.append({"layer": k, "alpha_mass": mass, "visible_alpha_mass": visible})
        transmission *= 1 - alpha
    return hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "stress_plan.json").read_text())
    candidate = next(iter(plan["adapters"]))
    scorer = LayeredGroupingRewardModel({"integrity_weight": 1})
    records = []
    for scene, extra in itertools.product(plan["scenes"], plan["extra_layers"]):
        layers = scene["layers"] + extra
        reference = LayeredReference.from_metadata(
            {
                **scene["metadata"],
                "reference_images": [str(Path(scene["scene_dir"]) / "source.png")],
            },
            640,
            640,
        )
        labeled = np.logical_or.reduce(reference.objects)
        for arm, root in [
            ("baseline_28", args.baseline),
            (candidate, args.run / "stress_eval" / candidate),
        ]:
            directory = root / scene["id"] / f"layers{layers}_steps10"
            assert len(list(directory.glob("layer_*.png"))) == layers
            stack = LayerStack.from_directory(directory, 640, 640)
            matches, _ = stack.match(reference.objects)
            complete = 0
            for i, mask in enumerate(reference.objects):
                k = matches.get(i)
                if k is None:
                    continue
                inter = int((mask & stack.regions[k]).sum())
                area = int((labeled & stack.regions[k]).sum())
                complete += inter / int(mask.sum()) >= 0.9 and inter / area >= 0.9
            records.append(
                {
                    "arm": arm,
                    "id": scene["id"],
                    "extra_layers": extra,
                    "requested_layers": layers,
                    "complete_pure_objects": int(complete),
                    "similar_foreground_pairs_for_review": similar_layers(stack),
                    "nearly_fully_hidden_nonempty_layers_for_review": hidden_nonempty_layers(
                        stack
                    ),
                    **scorer.score(stack, reference),
                    **diagnostics(stack, reference),
                }
            )
    (args.run / "stress_comparison.json").write_text(
        json.dumps(
            {
                "rows": records,
                "scope": "Four previously selected diagnostic cases, not a representative benchmark. Same seed across budgets does not preserve identical noise tensors because shapes differ. Similar full foreground pairs are review flags, not semantic duplicate ground truth; this check cannot detect object remnants embedded in a larger background layer.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Compared", len(records), "outputs")


if __name__ == "__main__":
    main()
