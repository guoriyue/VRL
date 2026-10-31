"""Quantify inherited Adam moments and a zero-new-gradient counterfactual.

This is a diagnostic, not an optimizer reset experiment or a causal explanation
of evaluation quality. The counterfactual uses float64 AdamW equations, so it
does not reproduce every float32 fused-kernel rounding operation.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    old = torch.load(
        args.baseline / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=False
    )
    new = torch.load(
        args.candidate / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=False
    )
    a, b = old["trainer"]["optimizer"], new["trainer"]["optimizer"]
    (group,) = b["param_groups"]
    assert group["decoupled_weight_decay"] and not group["amsgrad"] and not group["maximize"]
    beta1, beta2 = group["betas"]
    lr, eps, wd = (group[k] for k in ("lr", "eps", "weight_decay"))
    sums = dict.fromkeys(
        [
            "inherited_m2",
            "new_contribution_m2",
            "total_m2",
            "actual_delta2",
            "carry_delta2",
            "residual_delta2",
            "delta_dot",
        ],
        0.0,
    )
    steps = set()
    for name, state in b["state"].items():
        previous = a["state"][name]
        start, end = int(previous["step"]), int(state["step"])
        if end == start:
            continue
        steps.add(end - start)
        root, key = name.split(".", 1)
        x = old["model"]["owned_state"][root][key].double()
        target = new["model"]["owned_state"][root][key].double()
        m, v = previous["exp_avg"].double(), previous["exp_avg_sq"].double()
        inherited = m * beta1 ** (end - start)
        contribution = state["exp_avg"].double() - inherited
        sums["inherited_m2"] += float(inherited.square().sum())
        sums["new_contribution_m2"] += float(contribution.square().sum())
        sums["total_m2"] += float(state["exp_avg"].double().square().sum())
        predicted = x.clone()
        for step in range(start + 1, end + 1):
            m *= beta1
            v *= beta2
            predicted *= 1 - lr * wd
            predicted -= lr * (m / (1 - beta1**step)) / ((v / (1 - beta2**step)).sqrt() + eps)
        actual, carry = target - x, predicted - x
        sums["actual_delta2"] += float(actual.square().sum())
        sums["carry_delta2"] += float(carry.square().sum())
        sums["residual_delta2"] += float((actual - carry).square().sum())
        sums["delta_dot"] += float((actual * carry).sum())
    result = {
        "update_counts": sorted(steps),
        "new_moment_contribution_over_inherited_norm": (
            sums["new_contribution_m2"] / sums["inherited_m2"]
        )
        ** 0.5,
        "zero_gradient_update_over_actual_norm": (sums["carry_delta2"] / sums["actual_delta2"])
        ** 0.5,
        "zero_gradient_actual_update_cosine": sums["delta_dot"]
        / (sums["carry_delta2"] * sums["actual_delta2"]) ** 0.5,
        "residual_over_actual_update_norm": (sums["residual_delta2"] / sums["actual_delta2"])
        ** 0.5,
        "scope": __doc__,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
