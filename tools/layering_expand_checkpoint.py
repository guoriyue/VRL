"""Make an explicit four-rank continuation from the experiment's single-rank checkpoint.

Model, Adam tensors, EMA and progress are unchanged. Adam's positional keys
become parameter names, as FSDP's checkpoint loader requires. Rank zero retains its CPU RNGs;
new ranks receive independent seeded process RNGs. All ranks retain the source
prompt generator because the distributed sampler shards one common ordering.
This is a topology-changing continuation, not a bitwise replay of the old run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from vrl.trainers.checkpointing import validate_rng_state


def digest(value) -> str:
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(
                item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(item, dict):
            for key, child in item.items():
                h.update(repr(key).encode())
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    source_file = args.source / "checkpoint.pt"
    payload = torch.load(source_file, map_location="cpu", mmap=True, weights_only=False)
    original = payload["rng"]
    if "by_rank" in original:
        raise ValueError("expected the single-process ckpt-28 RNG tree")
    original_trainer_digest = digest(payload["trainer"])
    optimizer = payload["trainer"]["optimizer"]
    manifest = payload["trainer"]["optimizer_parameter_manifest"]
    ids = [key for group in optimizer["param_groups"] for key in group["params"]]
    names = [entry["name"] for entry in manifest]
    if len(ids) != len(names) or not all(isinstance(key, int) for key in ids):
        raise ValueError("expected a positional Adam checkpoint matching its parameter manifest")
    by_id = dict(zip(ids, names, strict=True))
    payload["trainer"]["optimizer"] = {
        **optimizer,
        "state": {by_id[key]: value for key, value in optimizer["state"].items()},
        "param_groups": [
            {**group, "params": [by_id[key] for key in group["params"]]}
            for group in optimizer["param_groups"]
        ],
    }
    # Prove every state entry and every hyperparameter survived the key change.
    renamed = payload["trainer"]["optimizer"]
    for key in optimizer["state"]:
        name = by_id[key]
        if digest(optimizer["state"][key]) != digest(renamed["state"][name]):
            raise RuntimeError(f"Adam state changed for {name}")
    # Adam creates slots lazily. The last block's text-output adapter has never
    # received a gradient; DCP's strict FQN loader nevertheless needs its slots.
    # A zero-step, zero-moment slot is equivalent to the original absent slot
    # on its first future update. Existing slots are never changed.
    dormant = []
    template = next(iter(optimizer["state"].values()))
    if set(template) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("expected ordinary Adam moments for dormant-slot materialization")
    for item in manifest:
        name = item["name"]
        if name not in renamed["state"]:
            renamed["state"][name] = {
                "step": torch.zeros_like(template["step"]),
                "exp_avg": torch.zeros(item["shape"], dtype=template["exp_avg"].dtype),
                "exp_avg_sq": torch.zeros(item["shape"], dtype=template["exp_avg_sq"].dtype),
            }
            dormant.append(name)
    before = {key: digest(payload[key]) for key in ("model", "trainer", "progress")}
    states = []
    for rank in range(4):
        state = copy.deepcopy(original)
        if rank:
            seed = args.seed + rank
            state["torch"] = torch.Generator().manual_seed(seed).get_state()
            state["python_random"] = random.Random(seed).getstate()
            state["numpy"] = np.random.RandomState(seed).get_state()
            cuda_state = torch.Generator(device=f"cuda:{rank}").manual_seed(seed).get_state()
        else:
            cuda_state = original["cuda"][0]
        # The online entrypoint isolates each rank to one visible CUDA device.
        state["cuda"] = [cuda_state.clone()]
        states.append(state)
    payload["rng"] = {"world_size": 4, "by_rank": states}
    for rank in range(4):
        validate_rng_state(
            payload["rng"], rank=rank, world_size=4, generator_names=("prompt_generator",)
        )
    target = args.destination / "checkpoint.pt"
    torch.save(payload, target)
    restored = torch.load(target, map_location="cpu", mmap=True, weights_only=False)
    after = {key: digest(restored[key]) for key in before}
    if before != after:
        raise RuntimeError("model, optimizer/EMA or progress changed during conversion")
    meta = json.loads((args.source / "checkpoint_meta.json").read_text())
    meta["checkpoint_file_bytes"] = target.stat().st_size
    (args.destination / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    receipt = {
        "source": str(source_file.resolve()),
        "source_sha256": hashlib.file_digest(source_file.open("rb"), "sha256").hexdigest(),
        "world_size_before": 1,
        "world_size_after": 4,
        "new_rank_seed_base": args.seed,
        "source_trainer_sha256": original_trainer_digest,
        "optimizer_key_conversion": "positional IDs to manifest parameter names; all moment tensors and hyperparameters preserved",
        "dormant_adam_slots_materialized_at_step_zero": dormant,
        "unchanged_payload_sha256": before,
        "scope": "Exact model/Adam/EMA/progress preservation; explicit new RNG streams for the new topology.",
    }
    (args.destination / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
