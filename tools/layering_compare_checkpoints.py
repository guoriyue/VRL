"""Compare live adapter tensors and Adam state across experiment checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_checkpoint(path: Path) -> dict:
    file = path / "checkpoint.pt" if path.is_dir() else path
    return torch.load(file, map_location="cpu", mmap=True, weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline = load_checkpoint(args.baseline)
    baseline_roots = baseline["model"]["owned_state"]
    baseline_steps = [
        float(state["step"])
        for state in baseline["trainer"]["optimizer"]["state"].values()
        if float(state["step"]) > 0
    ]
    results = []
    for path in args.checkpoints:
        candidate = load_checkpoint(path)
        roots = candidate["model"]["owned_state"]
        if roots.keys() != baseline_roots.keys():
            raise ValueError(f"checkpoint root mismatch: {path}")
        changed = count = elements = 0
        squared_delta = squared_baseline = max_abs_delta = 0.0
        for root, state in roots.items():
            reference = baseline_roots[root]
            if state.keys() != reference.keys():
                raise ValueError(f"checkpoint tensor-key mismatch: {path}/{root}")
            for name, tensor in state.items():
                previous = reference[name]
                if tensor.shape != previous.shape or tensor.dtype != previous.dtype:
                    raise ValueError(f"checkpoint tensor schema mismatch: {path}/{root}/{name}")
                if not torch.isfinite(tensor).all() or not torch.isfinite(previous).all():
                    raise ValueError(f"nonfinite checkpoint tensor: {path}/{root}/{name}")
                delta = tensor.double() - previous.double()
                peak = float(delta.abs().max())
                changed += peak != 0.0
                count += 1
                elements += tensor.numel()
                max_abs_delta = max(max_abs_delta, peak)
                squared_delta += float(delta.square().sum())
                squared_baseline += float(previous.double().square().sum())
        optimizer = candidate["trainer"]["optimizer"]
        steps = [float(state["step"]) for state in optimizer["state"].values()]
        active_steps = [step for step in steps if step > 0]
        results.append(
            {
                "checkpoint": str(path.resolve()),
                "progress": candidate["progress"],
                "live_tensor_count": count,
                "live_parameter_count": elements,
                "changed_live_tensors": changed,
                "all_live_tensors_finite": True,
                "max_abs_delta": max_abs_delta,
                "delta_l2": squared_delta**0.5,
                "relative_delta_l2": (squared_delta / squared_baseline) ** 0.5
                if squared_baseline
                else None,
                "adam_groups": [
                    {key: group[key] for key in ("lr", "eps", "betas", "weight_decay")}
                    for group in optimizer["param_groups"]
                ],
                "adam_active_step_range": [min(active_steps), max(active_steps)]
                if active_steps
                else [],
                "adam_dormant_states": sum(step == 0 for step in steps),
                "ema_num_updates": candidate["trainer"]["ema"]["num_updates"],
            }
        )
    report = {
        "baseline": str(args.baseline.resolve()),
        "baseline_progress": baseline["progress"],
        "baseline_adam_active_step_range": [min(baseline_steps), max(baseline_steps)]
        if baseline_steps
        else [],
        "scope": "Raw model.owned_state tensors; the exported EMA adapter is not compared.",
        "comparisons": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
