"""Prepare an explicit Adam-reset control without changing policy or RNG state."""

import argparse
import json
from pathlib import Path

import torch

from tools.layering_expand_checkpoint import digest
from vrl.trainers.checkpointing import TrainingCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original = TrainingCheckpoint.load(args.source)
    payload = dict(original.payload)
    trainer = dict(payload["trainer"])
    optimizer = trainer["optimizer"]
    if set(optimizer) != {"state", "param_groups"}:
        raise ValueError("Only ordinary Adam state is supported; master weights must be preserved")
    reset = {}
    for name, state in optimizer["state"].items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Unexpected optimizer state for {name}")
        reset[name] = {key: torch.zeros_like(value) for key, value in state.items()}
    trainer["optimizer"] = {**optimizer, "state": reset}
    payload["trainer"] = trainer
    unchanged = {key: digest(value) for key, value in original.payload.items() if key != "trainer"}
    unchanged_trainer = digest(
        {k: v for k, v in original.payload["trainer"].items() if k != "optimizer"}
    )
    args.destination.mkdir(parents=True, exist_ok=False)
    target = args.destination / "checkpoint.pt"
    torch.save(payload, target)
    meta = dict(original.meta)
    meta["checkpoint_file_bytes"] = target.stat().st_size
    (args.destination / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    restored = TrainingCheckpoint.load(args.destination).payload
    assert unchanged == {key: digest(restored[key]) for key in unchanged}
    assert unchanged_trainer == digest(
        {k: v for k, v in restored["trainer"].items() if k != "optimizer"}
    )
    assert digest(restored["trainer"]["optimizer"]["param_groups"]) == digest(
        optimizer["param_groups"]
    )
    assert all(
        not bool(value.any())
        for state in restored["trainer"]["optimizer"]["state"].values()
        for value in state.values()
    )
    receipt = {
        "source": str(args.source),
        "reset_adam_states": len(reset),
        "unchanged_payload_sha256": unchanged,
        "unchanged_trainer_except_optimizer_sha256": unchanged_trainer,
        "optimizer_hyperparameters_unchanged": True,
        "framework_load_validation_passed": True,
        "scope": "Adam moments and Adam bias-correction step reset to zero. Model, EMA, trainer progress and per-rank RNG preserved. Prepared control only; not used by running training.",
    }
    (args.destination / "reset_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
