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


def score_layers(
    reference,
    candidate,
    *,
    include_first_layer_edits=False,
    min_match_quality=0.0,
    include_source_preservation=False,
):
    """Return diagnostics, optionally editing the first object of an object-only stack.

    The default preserves historical background-first reward calculations.
    Changing the edit scope changes the reward and requires fresh calibration.
    A positive match floor lets weak geometric/color overlaps remain unmatched.
    It is an experiment setting, not a semantic object detector.
    Source preservation optionally measures visible reconstruction relative to
    each object's deletion contrast; it does not certify hidden completeness.
    """
    if not np.isfinite(min_match_quality) or not 0 <= min_match_quality < 1:
        raise ValueError("Minimum match quality must be finite and in [0,1)")
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
    assignment = match_layers(similarities - min_match_quality)
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
    predicted_composite = composite(candidate)
    reconstruction_error = float(np.abs(composite(reference) - predicted_composite).mean())
    # Complete amodal references define where no object may exist. Composited
    # alpha counts spill from every candidate, without dilution by empty layers.
    exterior = reference[..., 3].max(axis=0) == 0
    exterior_pixels = int(exterior.sum())
    exterior_alpha = (
        float(predicted_composite[..., 3][exterior].mean()) if exterior_pixels else 0.0
    )
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
    result = {
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
        "exterior_alpha_mean": exterior_alpha,
        "exterior_pixel_count": exterior_pixels,
        "scope": "Known-reference diagnostic only; candidate ranking needs visual calibration before RL use.",
    }
    if min_match_quality:
        result["min_match_quality"] = min_match_quality
    if include_source_preservation:
        truth_composite = composite(reference)
        residual = np.abs(truth_composite - predicted_composite).mean(axis=-1)
        source_rows = []
        normalized_errors = []
        for i, layer in enumerate(reference):
            visible = layer[..., 3] * np.prod(1 - reference[i + 1 :, ..., 3], axis=0)
            mass = float(visible.sum())
            deleted = reference.copy()
            deleted[i] = 0
            deletion_residual = np.abs(truth_composite - composite(deleted)).mean(axis=-1)
            error = float((residual * visible).sum() / mass) if mass else None
            deletion_error = float((deletion_residual * visible).sum() / mass) if mass else None
            normalized = error / deletion_error if mass and deletion_error > 1e-12 else None
            if normalized is not None:
                normalized_errors.append(normalized)
            source_rows.append(
                {
                    "reference_layer": i,
                    "visible_mass": mass,
                    "source_rgba_mae": error,
                    "reference_deletion_rgba_mae": deletion_error,
                    "deletion_normalized_error": normalized,
                }
            )
        result["source_preservation"] = {
            "objects": source_rows,
            "informative_objects": len(normalized_errors),
            "quality": 1 / (1 + max(normalized_errors)) if normalized_errors else 1.0,
            "scope": "Visible reference contrast only; fully occluded or zero-contrast objects provide no source-preservation evidence. Hidden completion is scored separately.",
        }
    return result


def coverage_fidelity_reward(score, *, part_quality=None, part_weight=0.25):
    """Keep credit within incomplete candidates after rejecting weak matches.

    Matched-reference count takes priority over a bounded fidelity term. Unlike
    a product over object qualities, an absent object does not erase all credit
    for repairing other layers. The match floor must be calibrated separately;
    counting geometric matches does not establish semantic completeness.
    Optional source preservation prevents small-object deletion being diluted
    by full-canvas reconstruction. Optional part quality shares the bounded
    within-coverage budget; it cannot compensate for losing a matched object.
    Without part quality, historical calculations are unchanged. None of these
    proposals is automatically approved for training.
    """
    if not 0 < score.get("min_match_quality", 0) < 1:
        raise ValueError("Coverage reward requires an explicitly positive match floor")
    if not np.isfinite(part_weight) or not 0 <= part_weight <= 1:
        raise ValueError("Part weight must be finite and in [0,1]")
    if part_quality is not None and (not np.isfinite(part_quality) or not 0 <= part_quality <= 1):
        raise ValueError("Part quality must be finite and in [0,1]")
    objects = score["objects"]
    matched = sum(row["candidate_layer"] is not None for row in objects)
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
    within_coverage = (
        score["components"]["mean_layer_quality"] * fidelity * components["nonredundancy"]
    )
    source_preservation = score.get("source_preservation")
    if source_preservation is not None:
        within_coverage *= source_preservation["quality"]
    result = {
        "version": "coverage_fidelity_source_v2"
        if source_preservation is not None
        else "coverage_fidelity_v1",
        "value": (matched + within_coverage) / (len(objects) + 1),
        "matched_references": matched,
        "reference_count": len(objects),
        "within_coverage_quality": within_coverage,
        "min_match_quality": score["min_match_quality"],
        "validated_for_training": False,
        "scope": "Known-reference proposal; a calibrated geometric match is not semantic proof.",
    }
    if part_quality is not None:
        combined = (1 - part_weight) * within_coverage + part_weight * part_quality
        result.update(
            version="coverage_bounded_parts_v1",
            value=(matched + combined) / (len(objects) + 1),
            part_quality=float(part_quality),
            part_weight=float(part_weight),
            combined_within_coverage_quality=float(combined),
        )
    return result


def completeness_priority_reward(
    score, *, min_alpha_iou, min_alpha_recall, max_normalized_source_error
):
    """Prioritize visible retention, then complete supports, then partial matches.

    The bounded quality term still distinguishes incomplete candidates in the
    same priority tier. Thresholds are explicit experiment inputs; alpha recall
    tolerances are not anatomical completeness or pixel-exact guarantees.
    """
    if "source_preservation" not in score:
        raise ValueError("Completeness priority requires source-preservation diagnostics")
    if not all(np.isfinite(x) and 0 < x <= 1 for x in (min_alpha_iou, min_alpha_recall)):
        raise ValueError("Completeness thresholds must be finite and in (0,1]")
    if not np.isfinite(max_normalized_source_error) or max_normalized_source_error <= 0:
        raise ValueError("Source-error threshold must be finite and positive")
    partial = coverage_fidelity_reward(score)
    source = score["source_preservation"]
    preserved = sum(
        row["deletion_normalized_error"] is not None
        and row["deletion_normalized_error"] < max_normalized_source_error
        for row in source["objects"]
    )
    complete = sum(
        row["candidate_layer"] is not None
        and row["alpha_iou"] >= min_alpha_iou
        and row["reference_alpha_recall"] >= min_alpha_recall
        for row in score["objects"]
    )
    base = partial["reference_count"] + 1
    value = (
        preserved * base**2
        + complete * base
        + partial["matched_references"]
        + partial["within_coverage_quality"]
    ) / ((source["informative_objects"] + 1) * base**2)
    return {
        "version": "completeness_priority_v1",
        "value": value,
        "preserved_visible_references": preserved,
        "complete_support_references": complete,
        "matched_references": partial["matched_references"],
        "within_coverage_quality": partial["within_coverage_quality"],
        "thresholds": {
            "min_alpha_iou": min_alpha_iou,
            "min_alpha_recall": min_alpha_recall,
            "max_normalized_source_error": max_normalized_source_error,
        },
        "validated_for_training": False,
        "scope": "Known-reference priority proposal; geometric support and source contrast do not prove semantic completeness.",
    }


def constrained_fidelity_reward(
    score, *, min_alpha_iou, min_alpha_recall, max_surplus_fraction, max_exterior_alpha=None
):
    """Propose fidelity learning only after explicit known-reference geometry gates.

    Thresholds are experiment inputs, not universal semantic validity criteria.
    Failed geometry cannot be compensated by accurate remaining object colors.
    Existing v2/v3 scores are preserved; this proposal is not auto-approved for RL.
    An optional exterior-alpha cap prevents fidelity gains compensating for
    excess spill outside all complete reference objects. It does not constrain
    visible RGB errors, and is vacuous for references covering the whole canvas.
    """
    thresholds = {
        "min_alpha_iou": min_alpha_iou,
        "min_alpha_recall": min_alpha_recall,
        "max_surplus_fraction": max_surplus_fraction,
    }
    if max_exterior_alpha is not None:
        thresholds["max_exterior_alpha"] = max_exterior_alpha
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("Geometry thresholds must be finite and in [0,1]")
    violations = {
        "alpha_iou": max(0.0, min_alpha_iou - min(row["alpha_iou"] for row in score["objects"])),
        "alpha_recall": max(
            0.0, min_alpha_recall - min(row["reference_alpha_recall"] for row in score["objects"])
        ),
        "surplus": max(0.0, 1 - score["components"]["nonredundancy"] - max_surplus_fraction),
    }
    geometry_passed = not any(violations.values())
    if max_exterior_alpha is not None:
        exterior_alpha = score["exterior_alpha_mean"]
        if not np.isfinite(exterior_alpha) or not 0 <= exterior_alpha <= 1:
            raise ValueError("Exterior alpha must be finite and in [0,1]")
        violations["exterior_alpha"] = max(0.0, exterior_alpha - max_exterior_alpha)
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
    result = {
        "version": "constrained_fidelity_v1"
        if max_exterior_alpha is None
        else "constrained_fidelity_exterior_v1",
        "value": fidelity if passed else -1.0 - sum(violations.values()),
        "geometry_gate_passed": geometry_passed,
        "violations": violations,
        "thresholds": thresholds,
        "fidelity": fidelity,
        "validated_for_training": False,
        "scope": "Known-reference proposal; numerical geometry gates do not prove semantic completeness.",
    }
    if max_exterior_alpha is not None:
        result["constraints_passed"] = passed
        result["exterior_pixel_count"] = score["exterior_pixel_count"]
    return result


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
