"""Shadow-test a visible partition score; never used by the running trainer.

For each annotated object, sum coverage squared times annotated-pixel purity
over foreground layers. Background earns no credit. Unlike normalization by
the number/size of predicted layers, removing an isolated pure fragment cannot
raise this score. Mixed fragments still involve precision/recall tradeoffs.
This cannot detect unannotated objects or invisible duplicates.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.layering_reward_audit import from_masks
from vrl.rewards.models.layered_grouping import LayeredReference


def visible_partition(stack, reference):
    if not reference.objects:
        return 0.0
    labeled = np.logical_or.reduce(reference.objects)
    objects = np.asarray([m.sum() for m in reference.objects], dtype=np.float64)
    layers = np.asarray([(m & labeled).sum() for m in stack.regions[1:]], dtype=np.float64)
    overlap = np.asarray(
        [[(m & layer).sum() for layer in stack.regions[1:]] for m in reference.objects],
        dtype=np.float64,
    )
    coverage = overlap / objects[:, None]
    purity = np.divide(overlap, layers[None], out=np.zeros_like(overlap), where=layers[None] > 0)
    quality = (coverage**2 * purity).sum(axis=1)
    return float(np.average(quality, weights=np.sqrt(objects)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = np.zeros((64, 64), bool)
    a[4:24, 4:24] = True
    b = np.zeros_like(a)
    b[32:52, 32:52] = True
    major = a.copy()
    major[:, 16:] = False
    minor = a & ~major
    void = np.zeros_like(a)
    void[4:24, 30:50] = True
    source = np.full((64, 64, 3), 0.4, np.float32)
    reference = LayeredReference(source, [a, b])
    cases = {
        "perfect": [a, b],
        "split_60_40": [major, minor, b],
        "fragment_in_background": [major, b],
        "merged": [a | b],
        "whole_object_in_background": [b],
        "void_padding": [major, minor | void, b],
        "extra_empty": [a, b, np.zeros_like(a)],
    }
    results = {}
    for name, masks in cases.items():
        stack = from_masks(masks, source)
        results[name] = {
            "current_grouping": stack.match(reference.objects)[1],
            "shadow_visible_partition": visible_partition(stack, reference),
        }
    def proposed(name):
        return results[name]["shadow_visible_partition"]

    assert proposed("perfect") > proposed("split_60_40") > proposed("fragment_in_background")
    assert proposed("split_60_40") == proposed("void_padding")
    assert proposed("perfect") == proposed("extra_empty")
    assert proposed("perfect") > proposed("merged")
    report = {
        "controlled_cases": results,
        "active_training_reward": False,
        "limitation": "Exploratory metric only. No representative human preference validation; incomplete annotations, hidden copies, and mixed-fragment tradeoffs remain. Do not interpret a shadow score gain as model improvement.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
