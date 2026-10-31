"""Verify an exported adapter against owned checkpoint weights and report updates."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def owned_state(payload):
    roots = payload["model"]["owned_state"]
    if len(roots) != 1:
        raise ValueError("Expected exactly one owned model root")
    return next(iter(roots.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = torch.load(
        args.checkpoint / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=False
    )
    baseline = torch.load(
        args.baseline / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=False
    )
    state, original = owned_state(payload), owned_state(baseline)
    exported = load_file(args.adapter / "adapter_model.safetensors", device="cpu")
    if set(state) != set(original):
        raise ValueError("Checkpoint and baseline tensor names differ")
    mapped = {
        key.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."): value
        for key, value in state.items()
    }
    if set(mapped) != set(exported):
        raise ValueError("Export tensor names differ from owned weights")
    mismatches = [key for key, value in mapped.items() if not torch.equal(value, exported[key])]
    finite = all(bool(torch.isfinite(value).all()) for value in state.values())
    if mismatches or not finite:
        raise ValueError(
            f"Export mismatch or nonfinite weights: {mismatches[:5]}, finite={finite}"
        )
    delta_squared = baseline_squared = 0.0
    changed = 0
    for key, value in state.items():
        before = original[key].double()
        delta = value.double() - before
        changed += int(bool(delta.any()))
        delta_squared += float(delta.square().sum())
        baseline_squared += float(before.square().sum())
    optimizer = payload["trainer"]["optimizer"]
    steps = [float(state["step"]) for state in optimizer["state"].values()]
    report = {
        "checkpoint": str(args.checkpoint),
        "baseline": str(args.baseline),
        "adapter": str(args.adapter),
        "owned_export_exact": True,
        "weights_finite": finite,
        "tensors": len(state),
        "changed_tensors": changed,
        "delta_l2": math.sqrt(delta_squared),
        "relative_delta_l2": math.sqrt(delta_squared / baseline_squared)
        if baseline_squared
        else None,
        "adam_step_min": min(steps) if steps else None,
        "adam_step_max": max(steps) if steps else None,
        "optimizer_groups": [
            {key: group.get(key) for key in ("lr", "eps", "betas", "weight_decay")}
            for group in optimizer["param_groups"]
        ],
        "scope": "Exact live-weight export and numerical update evidence; not model-quality evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
