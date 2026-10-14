"""Score a layer decomposition against known complete RGBA layers.

Both stacks are ``[N, H, W, 4]`` RGBA in ``[0, 1]``, back to front: the truth
(a rendered scene whose hidden content is known; layer 0 is the background) and
the candidate a model produced. Truth layers are matched to candidate layers
injectively by alpha IoU times color agreement, then measured on these axes,
each 1 for the truth itself:

* ``layered_object_match``: product of the per-object match qualities, so a
  merged or missing object zeroes it;
* ``layered_visible_retention``: one minus the error on visible truth pixels;
* ``layered_hidden_completion``: one minus the error on truth pixels hidden
  behind objects in front;
* ``layered_edit_holes``: one minus the composite error after hiding or moving
  each object, truth versus candidate;
* ``layered_nonredundancy``: share of candidate alpha inside matched layers;
* ``layered_reconstruction``: one minus the recomposed-image error;
* ``layered_composite_coverage``: share of the truth's opaque pixels the
  candidate composite covers opaquely;
* ``layered_inferable_completion``: the hidden-pixel error restricted to an
  ``inferable`` mask (deducible by continuation or symmetry), against truth;
* ``layered_unknowable_plausibility``: on hidden pixels inside an
  ``unknowable`` mask no truth is consulted: the matched layer must cover them
  with colors near its own visible colors.

These are hand-set evaluation metrics (the matching rule, the 0.5 alpha cuts
and the color-gap slope were never checked against human judgment). They rank
arms on one dataset; they are not a training reward.
"""

from __future__ import annotations

import numpy as np

SCORE_KEYS = (
    "layered_object_match",
    "layered_visible_retention",
    "layered_hidden_completion",
    "layered_edit_holes",
    "layered_nonredundancy",
    "layered_reconstruction",
    "layered_composite_coverage",
    "layered_inferable_completion",
    "layered_unknowable_plausibility",
)


def premultiplied(rgba):
    return np.concatenate((rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]), axis=-1)


def composite(layers):
    result = np.zeros_like(layers[0])
    for layer in layers:
        result = premultiplied(layer) + result * (1 - layer[..., 3:])
    return result


def match_layers(similarities):
    """Injective truth-to-candidate assignment maximizing total similarity.

    A dynamic program over subsets of the (at most eight) truth layers; candidate
    count is linear. A truth layer may stay unmatched and surplus candidates may
    stay unused.
    """
    count, candidates = similarities.shape
    if not 1 <= count <= 8:
        raise ValueError("Known-layer scenes need one to eight truth layers")
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


def score_layers(reference, candidate, *, inferable=None, unknowable=None):
    """Match ``candidate`` to the truth ``reference`` and score every axis.

    ``inferable`` / ``unknowable`` are optional ``[H, W]`` masks of hidden
    pixels; without one its axis is 1. Returns ``assignment`` (candidate index
    per truth layer, or None), per-object ``objects`` rows and the ``axes``.
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
        raise ValueError("Truth layers must contain visible or amodal object support")
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
    visible_errors, hidden_errors, edit_errors = [], [], []
    inferable_error = inferable_mass = plausibility = unknowable_mass = 0.0
    offset = max(1, reference.shape[2] // 8)
    for i, predicted_index in enumerate(assignment):
        predicted = blank if predicted_index is None else candidate[predicted_index]
        transmission = np.ones(reference.shape[1:3])
        for above in reference[i + 1 :]:
            transmission *= 1 - above[..., 3]
        hidden = reference[i, ..., 3] * (1 - transmission)
        visible = reference[i, ..., 3] * transmission
        residual = np.abs(ref_features[i] - premultiplied(predicted)).mean(axis=-1)
        row = {
            "reference_layer": i,
            "candidate_layer": predicted_index,
            "layer_quality": 0.0
            if predicted_index is None
            else float(similarities[i, predicted_index]),
            "visible_rgba_mae": None,
            "hidden_rgba_mae": None,
        }
        if visible.sum() > 0:
            row["visible_rgba_mae"] = float((residual * visible).sum() / visible.sum())
            visible_errors.append(row["visible_rgba_mae"])
        if hidden.sum() > 0:
            row["hidden_rgba_mae"] = float((residual * hidden).sum() / hidden.sum())
            hidden_errors.append(row["hidden_rgba_mae"])
        # The background (layer 0) is never hidden, moved or region-scored.
        if i > 0:
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
                row[f"{operation}_rgba_mae"] = float((difference * region).sum() / region.sum())
                edit_errors.append(row[f"{operation}_rgba_mae"])
            if inferable is not None:
                weight = hidden * inferable
                inferable_error += float((residual * weight).sum())
                inferable_mass += float(weight.sum())
            if unknowable is not None:
                weight = hidden * unknowable
                mass = float(weight.sum())
                if mass > 0:
                    alpha = float((predicted[..., 3] * weight).sum() / mass)
                    visible_px = (visible > 0.5) & (predicted[..., 3] > 0.5)
                    region_px = (weight > 0.5) & (predicted[..., 3] > 0.5)
                    if visible_px.any() and region_px.any():
                        gap = float(
                            np.abs(
                                predicted[region_px][:, :3].mean(axis=0)
                                - predicted[visible_px][:, :3].mean(axis=0)
                            ).mean()
                        )
                        continuity = max(0.0, 1.0 - 2.0 * gap)
                    else:
                        continuity = 0.0 if region_px.any() else 1.0
                    plausibility += alpha * continuity * mass
                    unknowable_mass += mass
        rows.append(row)
    used = {index for index in assignment if index is not None}
    extra_mass = sum(candidate[j, ..., 3].sum() for j in range(len(candidate)) if j not in used)
    total_mass = candidate[..., 3].sum()
    truth_composite = composite(reference)
    predicted_composite = composite(candidate)
    truth_opaque = truth_composite[..., 3] >= 0.99
    covered = predicted_composite[..., 3] >= 0.99
    axes = {
        "layered_object_match": float(np.prod([row["layer_quality"] for row in rows])),
        "layered_visible_retention": 1 - float(np.mean(visible_errors)) if visible_errors else 1.0,
        "layered_hidden_completion": 1 - float(np.mean(hidden_errors)) if hidden_errors else 1.0,
        "layered_edit_holes": 1 - float(np.mean(edit_errors)) if edit_errors else 1.0,
        "layered_nonredundancy": 1 - float(extra_mass / total_mass) if total_mass else 1.0,
        "layered_reconstruction": 1 - float(np.abs(truth_composite - predicted_composite).mean()),
        "layered_composite_coverage": float((covered & truth_opaque).sum() / truth_opaque.sum())
        if truth_opaque.any()
        else 1.0,
        "layered_inferable_completion": 1.0 - inferable_error / inferable_mass
        if inferable_mass
        else 1.0,
        "layered_unknowable_plausibility": plausibility / unknowable_mass
        if unknowable_mass
        else 1.0,
    }
    return {"assignment": assignment, "objects": rows, "axes": axes}
