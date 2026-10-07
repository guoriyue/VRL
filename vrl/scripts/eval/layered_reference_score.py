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


def premultiplied(rgba):
    return np.concatenate((rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]), axis=-1)


def composite(layers):
    result = np.zeros_like(layers[0])
    for layer in layers:
        result = premultiplied(layer) + result * (1 - layer[..., 3:])
    return result


def match_layers(similarities):
    """Find an injective assignment using a reference-subset dynamic program.

    The small reference budget bounds exponential work; candidate count is linear.
    Missing references can remain unmatched, and empty surplus layers are allowed.
    """
    count, candidates = similarities.shape
    if not 1 <= count <= 8:
        raise ValueError("Known-layer fixtures require one to eight reference layers")
    states = {0: (0.0, [None] * count)}
    for candidate in range(candidates):
        updated = dict(states)
        for mask, (score, assignment) in states.items():
            for reference in range(count):
                if mask & (1 << reference):
                    continue
                next_mask = mask | (1 << reference)
                next_score = score + similarities[reference, candidate]
                if next_mask not in updated or next_score > updated[next_mask][0]:
                    next_assignment = assignment.copy()
                    next_assignment[reference] = candidate
                    updated[next_mask] = (next_score, next_assignment)
        states = updated
    return max(states.values(), key=lambda item: item[0])[1]


def score_layers(reference, candidate, *, include_first_layer_edits=False):
    """Return diagnostics, optionally editing the first object of an object-only stack.

    The default preserves historical background-first reward calculations.
    Changing the edit scope changes the reward and requires fresh calibration.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        reference.ndim != 4
        or candidate.ndim != 4
        or reference.shape[1:] != candidate.shape[1:]
        or reference.shape[-1] != 4
        or not len(candidate)
    ):
        raise ValueError("Expected nonempty layer stacks with identical H,W,4 dimensions")
    for stack in (reference, candidate):
        if not np.isfinite(stack).all() or stack.min() < 0 or stack.max() > 1:
            raise ValueError("RGBA samples must be finite and normalized to [0,1]")
    if any(layer[..., 3].sum() == 0 for layer in reference):
        raise ValueError("Reference layers must contain visible or amodal object support")
    ref_features = premultiplied(reference)
    pred_features = premultiplied(candidate)
    similarities = np.zeros((len(reference), len(candidate)))
    for i, ref in enumerate(reference):
        for j, pred in enumerate(candidate):
            union = np.maximum(ref[..., 3], pred[..., 3])
            iou = np.minimum(ref[..., 3], pred[..., 3]).sum() / union.sum()
            error = np.abs(ref_features[i] - pred_features[j]).mean(axis=-1)
            similarities[i, j] = iou * (1 - (error * union).sum() / union.sum())
    assignment = match_layers(similarities)
    blank = np.zeros_like(reference[0])
    rows = []
    hidden_errors = []
    edit_errors = []
    offset = max(1, reference.shape[2] // 8)
    for i, predicted_index in enumerate(assignment):
        predicted = blank if predicted_index is None else candidate[predicted_index]
        transmission = np.ones(reference.shape[1:3])
        for above in reference[i + 1 :]:
            transmission *= 1 - above[..., 3]
        hidden = reference[i, ..., 3] * (1 - transmission)
        visible = reference[i, ..., 3] * transmission
        residual = ref_features[i] - premultiplied(predicted)
        visible_error = None
        if visible.sum() > 0:
            visible_error = float((np.abs(residual).mean(axis=-1) * visible).sum() / visible.sum())
        # Compare texture to the reference, rather than rewarding edge energy:
        # noise and oversharpening must not earn a sharpness bonus. Both pixels
        # must be visible, so hidden content cannot contaminate this diagnostic.
        gradient_error_sum = 0.0
        gradient_mass = 0.0
        for axis in (0, 1):
            before = [slice(None), slice(None)]
            after = [slice(None), slice(None)]
            before[axis] = slice(None, -1)
            after[axis] = slice(1, None)
            support = np.minimum(visible[tuple(before)], visible[tuple(after)])
            difference = np.abs(np.diff(residual, axis=axis)).mean(axis=-1)
            gradient_error_sum += float((difference * support).sum())
            gradient_mass += float(support.sum())
        hidden_error = None
        if hidden.sum() > 0:
            difference = np.abs(ref_features[i] - premultiplied(predicted)).mean(axis=-1)
            hidden_error = float((difference * hidden).sum() / hidden.sum())
            hidden_errors.append(hidden_error)
        row = {
            "reference_layer": i,
            "candidate_layer": predicted_index,
            "alpha_iou": float(
                np.minimum(reference[i, ..., 3], predicted[..., 3]).sum()
                / np.maximum(reference[i, ..., 3], predicted[..., 3]).sum()
            ),
            "reference_alpha_recall": float(
                np.minimum(reference[i, ..., 3], predicted[..., 3]).sum()
                / reference[i, ..., 3].sum()
            ),
            "layer_quality": 0.0
            if predicted_index is None
            else float(similarities[i, predicted_index]),
            "hidden_rgba_mae": hidden_error,
            "visible_rgba_mae": visible_error,
            "visible_gradient_mae": gradient_error_sum / gradient_mass if gradient_mass else None,
        }
        if i > 0 or include_first_layer_edits:
            for operation in ("hide", "move"):
                truth_layers = reference.copy()
                predicted_layers = candidate.copy()
                truth_layers[i] = blank
                if operation == "move":
                    truth_layers[i, :, offset:] = reference[i, :, :-offset]
                if predicted_index is not None:
                    predicted_layers[predicted_index] = blank
                    if operation == "move":
                        predicted_layers[predicted_index, :, offset:] = predicted[:, :-offset]
                region = np.maximum(reference[i, ..., 3], predicted[..., 3])
                if operation == "move":
                    shifted_support = np.maximum(
                        reference[i, :, :-offset, 3], predicted[:, :-offset, 3]
                    )
                    region[:, offset:] = np.maximum(region[:, offset:], shifted_support)
                difference = np.abs(composite(truth_layers) - composite(predicted_layers)).mean(
                    axis=-1
                )
                error = float((difference * region).sum() / region.sum())
                row[f"{operation}_rgba_mae"] = error
                edit_errors.append(error)
        rows.append(row)
    used = {index for index in assignment if index is not None}
    extra_mass = sum(candidate[j, ..., 3].sum() for j in range(len(candidate)) if j not in used)
    total_mass = candidate[..., 3].sum()
    surplus_fraction = float(extra_mass / total_mass) if total_mass else 0.0
    reconstruction_error = float(np.abs(composite(reference) - composite(candidate)).mean())
    components = {
        "mean_layer_quality": float(np.mean([row["layer_quality"] for row in rows])),
        "worst_layer_quality": min(row["layer_quality"] for row in rows),
        "hidden_quality": 1 - float(np.mean(hidden_errors)) if hidden_errors else 1.0,
        "edit_quality": 1 - float(np.mean(edit_errors)) if edit_errors else 1.0,
        "nonredundancy": 1 - surplus_fraction,
        "reconstruction_quality": 1 - reconstruction_error,
    }
    visible_errors = [
        row["visible_rgba_mae"] for row in rows if row["visible_rgba_mae"] is not None
    ]
    detail_errors = [
        row["visible_gradient_mae"] for row in rows if row["visible_gradient_mae"] is not None
    ]
    # Keep the v2 diagnostic for historical comparisons. Its minimum has a
    # plateau: an already better object can deteriorate without changing it.
    # This proposal responds to all objects while an absent object still zeros
    # the product. It requires counterexample and candidate calibration before use.
    reward_components = {
        "all_object_quality": float(np.prod([row["layer_quality"] for row in rows])),
        "hidden_quality": components["hidden_quality"],
        "edit_quality": components["edit_quality"],
        "nonredundancy": components["nonredundancy"],
        "reconstruction_quality": components["reconstruction_quality"],
        "visible_quality": 1 - float(np.mean(visible_errors)) if visible_errors else 1.0,
        # Signed gradient residuals can have magnitude up to two.
        "visible_detail_quality": 1 - float(np.mean(detail_errors)) / 2 if detail_errors else 1.0,
    }
    return {
        "diagnostic_version": 2,
        "visible_diagnostics_version": 1,
        "assignment": assignment,
        "objects": rows,
        "components": components,
        "diagnostic_score": min(components.values()),
        "reward_proposal": {
            "version": 3,
            "value": float(np.prod(list(reward_components.values()))),
            "components": reward_components,
            "validated_for_training": False,
        },
        "reconstruction_rgba_mae": reconstruction_error,
        "scope": "Known-reference diagnostic only; candidate ranking needs visual calibration before RL use.",
    }


def constrained_fidelity_reward(score, *, min_alpha_iou, min_alpha_recall, max_surplus_fraction):
    """Propose fidelity learning only after explicit known-reference geometry gates.

    Thresholds are experiment inputs, not universal semantic validity criteria.
    Failed geometry cannot be compensated by accurate remaining object colors.
    Existing v2/v3 scores are preserved; this proposal is not auto-approved for RL.
    """
    thresholds = {
        "min_alpha_iou": min_alpha_iou,
        "min_alpha_recall": min_alpha_recall,
        "max_surplus_fraction": max_surplus_fraction,
    }
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("Geometry thresholds must be finite and in [0,1]")
    violations = {
        "alpha_iou": max(0.0, min_alpha_iou - min(row["alpha_iou"] for row in score["objects"])),
        "alpha_recall": max(
            0.0, min_alpha_recall - min(row["reference_alpha_recall"] for row in score["objects"])
        ),
        "surplus": max(0.0, 1 - score["components"]["nonredundancy"] - max_surplus_fraction),
    }
    passed = not any(violations.values())
    components = score["reward_proposal"]["components"]
    fidelity = float(
        np.mean(
            [
                components[name]
                for name in (
                    "visible_quality",
                    "visible_detail_quality",
                    "hidden_quality",
                    "edit_quality",
                    "reconstruction_quality",
                )
            ]
        )
    )
    return {
        "version": "constrained_fidelity_v1",
        "value": fidelity if passed else -1.0 - sum(violations.values()),
        "geometry_gate_passed": passed,
        "violations": violations,
        "thresholds": thresholds,
        "fidelity": fidelity,
        "validated_for_training": False,
        "scope": "Known-reference proposal; numerical geometry gates do not prove semantic completeness.",
    }


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
