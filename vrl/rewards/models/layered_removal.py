"""Object-removal diagnostics for partially annotated layer references.

A reference background may contain unselected objects. Score the composite after
removing matched target layers, not the candidate's background layer alone.
Callers must establish that truth layer zero really removes every annotated target;
COCO pasted-occluder training references do not satisfy that contract.
"""

from __future__ import annotations

import numpy as np

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack


def _edit_quality(predicted, expected, region):
    area = float(region.sum(dtype=np.float64))
    if not area:
        return 1.0
    error = np.abs(predicted - expected).mean(-1)
    removal_error = float((error * region).sum(dtype=np.float64) / area)
    outside = 1.0 - region
    outside_area = float(outside.sum(dtype=np.float64))
    if not outside_area:
        return 1.0 - removal_error
    preservation_error = float((error * outside).sum(dtype=np.float64) / outside_area)
    return 1.0 - 0.5 * (removal_error + preservation_error)


def target_removal_quality(layers: LayerStack, reference: LayeredReference) -> float:
    """Target removal and preservation of the surrounding scene.

    Unannotated objects may remain in the background or occupy separate layers.
    Copies exposed by removing the matched targets become visible. Copies still
    covered by retained objects may remain undetected.
    This is a reference-based pixel diagnostic, not a semantic duplicate detector.
    """
    if reference.truth is None:
        raise ValueError("Target removal requires foreground layers and a removal background")
    matches, _ = layers.match(reference.objects)
    retained = layers.rgba.copy()
    for index in matches.values():
        retained[index, ..., 3] = 0
    predicted = LayerStack.from_rgba(retained).composite()
    region = reference.truth.rgba[1:, ..., 3].max(axis=0)
    expected = reference.truth.rgba[0, ..., :3]
    return _edit_quality(predicted, expected, region)


def single_object_removal_quality(layers: LayerStack, reference: LayeredReference) -> list[dict]:
    """Delete each visible truth object separately, preserving the other targets.

    Removing all targets together cannot distinguish a correct decomposition
    from putting all of them in one merged layer. These individual edits can.
    As with the all-target edit, unexposed hidden copies may escape detection.
    """
    truth = reference.truth
    if truth is None:
        raise ValueError("Object removal requires foreground truth layers")
    targets = [k for k in range(1, len(truth.rgba)) if truth.regions[k].any()]
    matches, _ = layers.match([truth.regions[k] for k in targets])
    records = []
    for i, target in enumerate(targets):
        retained = layers.rgba.copy()
        match = matches.get(i)
        if match is not None:
            retained[match, ..., 3] = 0
        expected = truth.rgba.copy()
        expected[target, ..., 3] = 0
        records.append(
            {
                "truth_layer": target,
                "candidate_layer": match,
                "quality": _edit_quality(
                    LayerStack.from_rgba(retained).composite(),
                    LayerStack.from_rgba(expected).composite(),
                    truth.rgba[target, ..., 3],
                ),
            }
        )
    return records
