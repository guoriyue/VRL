"""Report from saved checkpoints whether training updates moved the weights.

An update can be numerically valid and still do nothing: an Adam eps far above
the square root of the second moment shrinks every step to a sliver of the
learning rate, while replay parity, loss and gradient norm all look normal.
This report reads only ``checkpoint.pt`` files (no model, no GPU) and answers
two questions after a run:

* Per checkpoint, from the saved Adam moments: how large the next step is in
  units of the learning rate (``next_step_to_lr``, RMS of
  ``m_hat / (sqrt(v_hat) + eps)``), how much eps shrinks it
  (``eps_damping``, that RMS over the same RMS without eps), and the fraction
  of elements whose ``sqrt(v_hat)`` sits below eps (``eps_dominated``).
* Between consecutive checkpoints of one series: how far the trained
  parameters moved (``relative_change`` = ||delta|| / ||before||) and the net
  displacement RMS per optimizer step in units of the later checkpoint's
  learning rate (``net_update_to_lr``). Net displacement is not the per-step
  update size: steps whose signs alternate cancel, so a healthy run with noisy
  gradients reads up to sqrt(steps) below ``next_step_to_lr``; a stalled run
  reads near zero on both. FP32 master values are compared when the checkpoint
  holds them, so sub-ULP updates of a low-precision model still show.

Each run directory is one series of its complete ``checkpoint-*`` in
global-step order; checkpoint directories named directly form one more series
in the order given. Optimizer steps between checkpoints come from Adam's own
step count (global steps can count micro-steps or skipped updates). Moments of
optimizers other than torch Adam/AdamW (8-bit bitsandbytes, Adafactor) are
reported as unsupported; the parameter movement is still reported. Run
``python -m vrl.scripts.eval.update_report --help``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    TrainingCheckpoint,
    is_complete_checkpoint,
    read_checkpoint_meta,
)
from vrl.utils.json_files import write_json
from vrl.utils.paths import user_path

# Prefix components DDP and torch.compile add to ``named_parameters()`` (the
# optimizer manifest) but checkpoint export peels from the owned state.
_WRAPPER_COMPONENTS = frozenset({"module", "_orig_mod"})


def _series(paths: Iterable[str | Path]) -> list[list[Path]]:
    """One series per run directory, plus one for directly named checkpoints."""

    runs: list[list[Path]] = []
    named: list[Path] = []
    for raw in paths:
        path = user_path(raw)
        if path.is_file() or (path / TRAINING_CHECKPOINT_NAME).is_file():
            named.append(path)
            continue
        found = [c for c in path.glob("checkpoint-*") if is_complete_checkpoint(c)]
        if not found:
            raise FileNotFoundError(f"{path}: no checkpoint.pt and no complete checkpoint-*")
        runs.append(sorted(found, key=lambda c: (read_checkpoint_meta(c)["global_step"], c)))
    return runs + ([named] if named else [])


def _global_step(checkpoint: TrainingCheckpoint) -> int:
    progress = checkpoint.progress
    if "global_step" in progress:
        return int(progress["global_step"])
    return int(checkpoint.trainer_state.get("global_step", 0))


def _optimizer_state(checkpoint: TrainingCheckpoint) -> Mapping[str, Any] | None:
    state = checkpoint.trainer_state.get("optimizer")
    return state if isinstance(state, Mapping) and state.get("param_groups") else None


def _parameters(checkpoint: TrainingCheckpoint) -> dict[str, torch.Tensor]:
    """Trained values by name, the FP32 masters where the optimizer keeps them."""

    values = {
        f"{root}.{name}": tensor
        for root, state in checkpoint.checkpoint_state.items()
        for name, tensor in state.items()
        if torch.is_tensor(tensor)
    }
    optimizer = _optimizer_state(checkpoint) or {}
    masters = optimizer.get("fp32_master_weights")
    manifest = checkpoint.trainer_state.get("optimizer_parameter_manifest")
    if not isinstance(masters, Mapping) or not isinstance(manifest, list):
        return values
    for entry, master in zip(manifest, masters["parameters"], strict=True):
        if master is None:
            continue
        name = entry["name"]
        if name not in values:
            root, _, rest = name.partition(".")
            peeled = [part for part in rest.split(".") if part not in _WRAPPER_COMPONENTS]
            name = ".".join([root, *peeled])
        if name not in values:
            raise ValueError(f"FP32 master {entry['name']!r} matches no owned-state tensor")
        values[name] = master
    return values


def adam_moment_stats(optimizer: Mapping[str, Any]) -> dict[str, Any]:
    """Size of Adam's next step from the saved moments, streamed per tensor."""

    groups = optimizer["param_groups"]
    state = optimizer["state"]
    elements = 0
    eps_dominated = 0
    step_squared = 0.0
    undamped_squared = 0.0
    adam_step = 0
    for group in groups:
        if "betas" not in group:
            return {"supported": False, "reason": f"param group keys {sorted(group)}"}
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        for key in group["params"]:
            moments = state.get(key)
            if not moments:
                continue
            if "exp_avg" not in moments or "exp_avg_sq" not in moments:
                return {"supported": False, "reason": f"state keys {sorted(map(str, moments))}"}
            step = int(float(moments["step"]))
            adam_step = max(adam_step, step)
            m_hat = moments["exp_avg"].double() / (1 - beta1**step)
            v_hat = moments["exp_avg_sq"].double() / (1 - beta2**step)
            seen = v_hat > 0
            root_v = v_hat[seen].sqrt()
            m_seen = m_hat[seen]
            elements += int(root_v.numel())
            eps_dominated += int((root_v < eps).sum())
            step_squared += float((m_seen / (root_v + eps)).square().sum())
            undamped_squared += float((m_seen / root_v).square().sum())
    next_step_to_lr = math.sqrt(step_squared / elements) if elements else 0.0
    undamped = math.sqrt(undamped_squared / elements) if elements else 0.0
    return {
        "supported": True,
        "adam_step": adam_step,
        "eps": [float(group["eps"]) for group in groups],
        "elements": elements,
        "eps_dominated": eps_dominated / elements if elements else 0.0,
        "next_step_to_lr": next_step_to_lr,
        "eps_damping": next_step_to_lr / undamped if undamped else 0.0,
    }


def parameter_change(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    steps: int,
    lr: float | None,
) -> dict[str, Any]:
    """How far the parameters shared by two checkpoints moved."""

    shared = sorted(before.keys() & after.keys())
    elements = 0
    changed = 0
    delta_squared = 0.0
    before_squared = 0.0
    for name in shared:
        old = before[name].double()
        new = after[name].double()
        if old.shape != new.shape:
            raise ValueError(f"{name}: shape {tuple(old.shape)} -> {tuple(new.shape)}")
        elements += old.numel()
        changed += int((new != old).sum())
        delta_squared += float((new - old).square().sum())
        before_squared += float(old.square().sum())
    net_rms = math.sqrt(delta_squared / elements) / steps if elements and steps > 0 else None
    return {
        "optimizer_steps": steps,
        "tensors": len(shared),
        "unmatched_tensors": len(before.keys() ^ after.keys()),
        "changed_fraction": changed / elements if elements else None,
        "relative_change": math.sqrt(delta_squared / before_squared) if before_squared else None,
        "net_update_rms_per_step": net_rms,
        "net_update_to_lr": net_rms / lr if net_rms is not None and lr else None,
    }


def _series_report(paths: Sequence[Path]) -> dict[str, Any]:
    """Load checkpoints one at a time; only the previous one's parameters stay alive."""

    checkpoints: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for path in paths:
        checkpoint = TrainingCheckpoint.load(path)
        optimizer = _optimizer_state(checkpoint)
        moments = adam_moment_stats(optimizer) if optimizer is not None else None
        current = {
            "checkpoint": str(checkpoint.checkpoint_dir),
            "global_step": _global_step(checkpoint),
            "lr": float(optimizer["param_groups"][0]["lr"]) if optimizer is not None else None,
            "adam": moments,
        }
        checkpoints.append(current)
        parameters = _parameters(checkpoint)
        del checkpoint
        if previous is not None:
            adam_steps = [
                entry["adam"]["adam_step"]
                for entry in (previous["entry"], current)
                if entry["adam"] is not None and entry["adam"]["supported"]
            ]
            steps = (
                adam_steps[1] - adam_steps[0]
                if len(adam_steps) == 2
                else current["global_step"] - previous["entry"]["global_step"]
            )
            intervals.append(
                {
                    "from": previous["entry"]["checkpoint"],
                    "to": current["checkpoint"],
                    **parameter_change(
                        previous["parameters"], parameters, steps=steps, lr=current["lr"]
                    ),
                },
            )
        previous = {"entry": current, "parameters": parameters}
    return {"checkpoints": checkpoints, "intervals": intervals}


def build_report(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Moment stats per checkpoint and parameter change between neighbours, per series."""

    return {"series": [_series_report(series) for series in _series(paths)]}


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def _print_report(report: Mapping[str, Any]) -> None:
    for index, series in enumerate(report["series"]):
        if index:
            print()
        for entry in series["checkpoints"]:
            adam = entry["adam"]
            if adam is None:
                detail = "no optimizer state"
            elif not adam["supported"]:
                detail = f"unsupported optimizer state ({adam['reason']})"
            else:
                detail = (
                    f"adam_step={adam['adam_step']} eps={adam['eps'][0]:.3g} "
                    f"next_step_to_lr={adam['next_step_to_lr']:.3g} "
                    f"eps_damping={adam['eps_damping']:.3g} "
                    f"eps_dominated={adam['eps_dominated']:.3f}"
                )
            print(
                f"step {entry['global_step']:>6}  {entry['checkpoint']}  "
                f"lr={_number(entry['lr'])} {detail}",
            )
        for interval in series["intervals"]:
            print(
                f"  {Path(interval['from']).name} -> {Path(interval['to']).name}  "
                f"optimizer_steps={interval['optimizer_steps']} "
                f"relative_change={_number(interval['relative_change'])} "
                f"net_update_to_lr={_number(interval['net_update_to_lr'])} "
                f"changed_fraction={_number(interval['changed_fraction'])} "
                f"unmatched_tensors={interval['unmatched_tensors']}",
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="checkpoint directories or run directories")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args(argv)
    report = build_report(args.paths)
    _print_report(report)
    if args.json is not None:
        write_json(args.json, report, allow_nan=False)


if __name__ == "__main__":
    main()
