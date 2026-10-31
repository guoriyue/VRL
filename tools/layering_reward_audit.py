"""Audit grouping reward against controlled errors and saved diagnostic outputs.

This is an offline measurement tool; it does not change the production reward.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def maximum_assignment_sum(utility):
    """Exact subset DP for this small audit (at most nine annotated objects)."""
    assert utility.shape[0] <= 16
    states = {0: 0.0}
    for column in utility.T:
        updated = dict(states)
        for mask, value in states.items():
            for row, score in enumerate(column):
                if score > 0 and not mask & (1 << row):
                    key = mask | (1 << row)
                    updated[key] = max(updated.get(key, 0.0), value + float(score))
        states = updated
    return max(states.values())


def object_assignment_score(stack, reference):
    """Exploratory macro coverage^2 * purity with optimal one-to-one assignment.

    Only annotated pixels count. This cannot repair incomplete annotations.
    Every layer with annotated overlap is eligible, including tiny true objects.
    """
    objects = reference.objects
    labeled = np.logical_or.reduce(objects)
    layers = [k for k in range(1, len(stack.regions)) if (stack.regions[k] & labeled).any()]
    if not layers:
        return 0.0
    overlap = np.array(
        [[float((mask & stack.regions[k]).sum()) for k in layers] for mask in objects]
    )
    coverage = overlap / np.array([mask.sum() for mask in objects])[:, None]
    purity = overlap / np.array([(stack.regions[k] & labeled).sum() for k in layers])[None, :]
    utility = coverage**2 * purity
    return maximum_assignment_sum(utility) / len(objects)


def measured(stack, reference):
    return {
        **LayeredGroupingRewardModel({}).score(stack, reference),
        "candidate_assignment": object_assignment_score(stack, reference),
    }


def from_masks(masks, source):
    background = np.concatenate(
        [source.copy(), np.ones((*source.shape[:2], 1), dtype=np.float32)], -1
    )
    rgba = [background]
    for mask in masks:
        layer = np.zeros_like(background)
        layer[..., :3] = source
        layer[..., 3] = mask
        rgba.append(layer)
    return LayerStack.from_rgba(np.stack(rgba))


def synthetic():
    h = w = 64
    a = np.zeros((h, w), bool)
    a[4:24, 4:24] = True
    b = np.zeros_like(a)
    b[32:52, 32:52] = True
    void = np.zeros_like(a)
    void[4:24, 30:50] = True
    source = np.full((h, w, 3), 0.4, dtype=np.float32)
    source[a] = [0.8, 0.2, 0.2]
    source[b] = [0.2, 0.2, 0.8]
    half = a.copy()
    half[:, 14:] = False
    major = a.copy()
    major[:, 16:] = False
    minor = a & ~major
    ref = LayeredReference(source, [a, b])
    cases = {
        "perfect": [a, b],
        "extra_empty": [a, np.zeros_like(a), b],
        "layer_permutation": [b, a],
        "split_half": [half, a & ~half, b],
        "split_60_40": [major, minor, b],
        "split_with_void_padding": [major, minor | void, b],
        "fragment_in_background": [major, b],
        "object_in_background": [b],
        "merged": [a | b],
        "unannotated_left_in_background": [a, b],
        "unannotated_separated": [a, b, void],
    }
    results = {k: measured(from_masks(m, source), ref) for k, m in cases.items()}
    # A smaller true object demonstrates the fixed 64-pixel visibility cutoff.
    small = np.zeros_like(a)
    small[27:31, 27:35] = True
    small_ref = LayeredReference(source, [a, small])
    results["tiny_object_perfect"] = measured(from_masks([a, small], source), small_ref)
    results["tiny_object_missing"] = measured(from_masks([a], source), small_ref)
    # Same defect on different object sizes: quantify area-weighting sensitivity.
    unequal = LayeredReference(source, [a, small])
    results["large_object_missing"] = measured(from_masks([small], source), unequal)
    # Greedy IoU matching is not the optimum of the weighted reward it returns.
    a2 = np.zeros_like(a)
    a2.flat[:400] = True
    b2 = np.zeros_like(a)
    b2.flat[400:560] = True
    l1 = np.zeros_like(a)
    l1.flat[:240] = True
    l1.flat[400:560] = True
    l2 = a2 & ~l1
    stack = from_masks([l1, l2], source)
    ref2 = LayeredReference(source, [a2, b2])
    _, greedy = stack.match(ref2.objects)
    overlap = np.array([[240.0, 160.0], [160.0, 0.0]])
    sizes_o = overlap.sum(1)
    sizes_l = overlap.sum(0)
    iou = overlap / (sizes_o[:, None] + sizes_l[None, :] - overlap)
    weighted = (np.sqrt(sizes_o[:, None]) + np.sqrt(sizes_l[None, :])) * iou
    optimal = float(
        maximum_assignment_sum(weighted) / (np.sqrt(sizes_o).sum() + np.sqrt(sizes_l).sum())
    )
    checks = {
        "perfect_beats_split_merge_missing": all(
            results["perfect"]["layered_grouping"] > results[k]["layered_grouping"]
            for k in ["split_half", "merged", "object_in_background"]
        ),
        "empty_layers_neutral": abs(
            results["perfect"]["layered_grouping"] - results["extra_empty"]["layered_grouping"]
        )
        < 1e-8,
        "permutation_neutral": abs(
            results["perfect"]["layered_grouping"]
            - results["layer_permutation"]["layered_grouping"]
        )
        < 1e-8,
        "void_padding_cannot_raise_reward": results["split_with_void_padding"]["layered_grouping"]
        <= results["split_60_40"]["layered_grouping"] + 1e-8,
        "separating_unannotated_object_is_rewarded": results["unannotated_separated"][
            "layered_grouping"
        ]
        > results["unannotated_left_in_background"]["layered_grouping"] + 1e-8,
        "recovering_tiny_object_is_rewarded": results["tiny_object_perfect"]["layered_grouping"]
        > results["tiny_object_missing"]["layered_grouping"] + 1e-8,
        "greedy_maximizes_own_weighted_objective": abs(greedy - optimal) < 1e-8,
    }
    assert all(abs(r["fidelity"] - 1) < 1e-7 for r in results.values())
    assert (
        checks["perfect_beats_split_merge_missing"]
        and checks["empty_layers_neutral"]
        and checks["permutation_neutral"]
    )
    return {
        "scores": results,
        "acceptance_checks": checks,
        "matching_counterexample": {
            "overlap_pixels": overlap.tolist(),
            "current_grouping": greedy,
            "optimal_same_weighted_objective": optimal,
        },
        "candidate_checks": {
            "void_padding_neutral": results["split_60_40"]["candidate_assignment"]
            == results["split_with_void_padding"]["candidate_assignment"],
            "tiny_recovery_rewarded": results["tiny_object_perfect"]["candidate_assignment"]
            > results["tiny_object_missing"]["candidate_assignment"],
            "perfect_beats_split_merge_missing": all(
                results["perfect"]["candidate_assignment"] > results[k]["candidate_assignment"]
                for k in ["split_half", "merged", "object_in_background"]
            ),
        },
    }


def real_outputs(root):
    report = json.loads((root / "summary.json").read_text())
    scenes = {s["id"]: s for s in report["plan"]["scenes"]}
    refs = {}
    for s in scenes.values():
        refs[s["id"]] = LayeredReference.from_metadata(
            {**s["metadata"], "reference_images": [str(Path(s["scene_dir"]) / "source.png")]},
            640,
            640,
        )
    rows = []
    counterfactuals = []
    for row in report["rows"]:
        ref = refs[row["id"]]
        path = (
            root
            / "eval"
            / row["arm"]
            / row["id"]
            / f"layers{row['requested_layers']}_steps{row['sampling_steps']}"
        )
        stack = LayerStack.from_directory(path, 640, 640)
        scores = measured(stack, ref)
        assert abs(scores["layered_grouping"] - row["layered_grouping"]) < 1e-7
        rows.append({**row, **scores})
        if row["extra_layers"] or row["sampling_steps"] != 10:
            continue
        # A diagnostic oracle on visible masks, preserving the original composite.
        # This deliberately uses labels: it is a counterfactual, not a model result.
        source = stack.composite()
        disjoint = []
        seen = np.zeros(source.shape[:2], bool)
        for mask in ref.objects:
            disjoint.append(mask & ~seen)
            seen |= mask
        repaired = from_masks(disjoint, source)
        same_fidelity = abs(measured(repaired, ref)["fidelity"] - scores["fidelity"]) < 1e-7
        counterfactuals.append(
            {
                "id": row["id"],
                "arm": row["arm"],
                "original": scores,
                "visible_mask_oracle": measured(repaired, ref),
                "composite_fidelity_unchanged": same_fidelity,
                "completion_note": "Visible masks only; hidden completion is not repaired and may regress.",
            }
        )
    pairs = []
    for a, b in itertools.combinations(rows, 2):
        if a["id"] != b["id"]:
            continue
        # A strict count is a diagnostic proxy, not a complete human quality label.
        for better, worse in [(a, b), (b, a)]:
            count_better = (
                better["objects_with_iou_at_least_0_5"] > worse["objects_with_iou_at_least_0_5"]
            )
            no_more_bg = (
                better["object_background_share"] <= worse["object_background_share"] + 1e-8
            )
            no_more_merge = (
                better["layers_with_majority_of_multiple_objects"]
                <= worse["layers_with_majority_of_multiple_objects"]
            )
            if (
                count_better
                and no_more_bg
                and no_more_merge
                and better["layered_grouping"] < worse["layered_grouping"] - 1e-8
            ):
                keys = [
                    "arm",
                    "extra_layers",
                    "sampling_steps",
                    "grouping",
                    "completion",
                    "fidelity",
                    "layered_grouping",
                    "candidate_assignment",
                    "objects_with_iou_at_least_0_5",
                    "object_background_share",
                    "layers_with_majority_of_multiple_objects",
                ]
                pairs.append(
                    {
                        "id": a["id"],
                        "better_on_three_visible_proxies": {k: better[k] for k in keys},
                        "higher_reward": {k: worse[k] for k in keys},
                    }
                )
    return {
        "rows": rows,
        "visible_mask_counterfactuals": counterfactuals,
        "proxy_ranking_disagreements": pairs,
        "note": "Candidates and proxies are diagnostic only, not independently validated human preferences.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "synthetic": synthetic(),
        "real_outputs": real_outputs(args.diagnostics),
        "production_reward_unchanged": True,
    }
    (args.output / "reward_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "checks": result["synthetic"]["acceptance_checks"],
                "candidate_checks": result["synthetic"]["candidate_checks"],
                "rescored_outputs": len(result["real_outputs"]["rows"]),
                "proxy_ranking_disagreements": len(
                    result["real_outputs"]["proxy_ranking_disagreements"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
