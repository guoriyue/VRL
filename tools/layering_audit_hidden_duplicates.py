"""Check whether invisible object copies change the deployed layering reward."""

import json
from pathlib import Path

import numpy as np

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def main():
    height = width = 64
    rgba = np.zeros((3, height, width, 4), dtype=np.float32)
    rgba[0, ..., :3] = 0.4
    rgba[0, ..., 3] = 1
    rgba[1, 8:40, 8:40] = [0.8, 0.2, 0.2, 1]
    rgba[2, 24:56, 24:56] = [0.2, 0.2, 0.8, 1]
    correct = LayerStack.from_rgba(rgba)
    objects = list(correct.regions[1:])
    hidden = (rgba[1, ..., 3] > 0.5) & ~correct.regions[1]
    duplicated = LayerStack.from_rgba(np.concatenate([rgba[:1], rgba[1:2], rgba[1:]], axis=0))
    background_copy = rgba.copy()
    mask = rgba[1, ..., 3] > 0.5
    background_copy[0, mask, :3] = rgba[1, mask, :3]
    cases = {
        "correct_occlusion_and_completion": correct,
        "extra_fully_hidden_object_copy": duplicated,
        "object_also_painted_in_background": LayerStack.from_rgba(background_copy),
    }
    model = LayeredGroupingRewardModel({})
    results = {}
    for name, stack in cases.items():
        results[name] = {
            "without_hidden_truth": model.score(
                stack, LayeredReference(correct.composite(), objects)
            ),
            "with_hidden_truth": model.score(
                stack, LayeredReference(correct.composite(), objects, correct, hidden)
            ),
            "composite_max_error": float(np.abs(stack.composite() - correct.composite()).max()),
            "total_foreground_alpha_pixels": int((stack.rgba[1:, ..., 3] > 0.5).sum()),
        }
    for result in results.values():
        assert result["composite_max_error"] == 0
        assert result["without_hidden_truth"]["layered_grouping"] == 2.0
        assert result["with_hidden_truth"]["layered_grouping"] == 2.8
    report = {
        "cases": results,
        "finding": "Both extra-layer and background object copies are invisible to the current reward, even with completion truth.",
        "scope": "Constructed opaque-copy counterexamples; not a count of failures in generated outputs.",
        "implication": "Full layers and background need content validation, including unmatched invisible layers; visible masks and recomposition alone are insufficient.",
        "production_reward_changed": False,
    }
    output = Path("outputs/layered_grpo/reward_fix_20261024/hidden_duplicate_audit.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
