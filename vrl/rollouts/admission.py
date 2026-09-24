"""Post-advantage sample admission and immutable-attempt audit records.

Algorithms still own normalization, including cross-rank statistics. Admission
consumes their prepared advantages and preserves the existing exact-zero mask.
Selection is not proof of an optimizer update, task difficulty or reward validity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from vrl.algorithms.advantages import nonzero_advantage_mask
from vrl.rollouts.batch import RolloutBatch


def _finite_values(value: Any) -> list[float | None]:
    values = value.detach().cpu().double().reshape(-1).tolist()
    return [number if math.isfinite(number) else None for number in values]


def select_advantage_rows(
    batches: list[RolloutBatch],
    advantages: list[Any],
    *,
    drop_zero_advantage: bool,
) -> tuple[list[RolloutBatch], list[Any], list[dict[str, Any]]]:
    """Apply the historical row mask without changing any advantage or group statistic."""
    import torch

    selected_batches, selected_advantages, records = [], [], []
    for batch, advantage in zip(batches, advantages, strict=True):
        count = batch.rewards.shape[0]
        mask = (
            nonzero_advantage_mask(advantage)
            if drop_zero_advantage
            else torch.ones(count, dtype=torch.bool, device=advantage.device)
        )
        if mask.shape != (count,):
            raise ValueError("admission advantage mask must match rollout sample count")
        groups = batch.group_ids.detach().cpu()
        mask_cpu = mask.detach().cpu()
        trajectory = batch.trajectory
        rows = trajectory.sample_rows if trajectory is not None else None
        if rows is not None and len(rows) != count:
            raise ValueError("admission trajectory rows must match reward samples")
        metadata = dict(batch.context.get("reward_metadata", {}))
        version = metadata.pop("rollout_policy_version", None)
        for group in dict.fromkeys(groups.tolist()):
            positions = (groups == group).nonzero(as_tuple=True)[0]
            group_mask = mask_cpu[positions]
            kept = int(group_mask.sum().item())
            group_rows = [rows[index] for index in positions.tolist()] if rows is not None else []
            prompts = {row.prompt for row in group_rows}
            if len(prompts) > 1:
                raise ValueError("one admission group contains multiple prompt texts")
            prompt = next(iter(prompts), None)
            explicit_id = metadata.get("prompt_id", metadata.get("id", metadata.get("task_id")))
            # Source/target metadata participates, so identical text with different
            # references does not silently become one history bucket. This is an
            # input identity, not attestation of the referenced file bytes.
            identity = json.dumps(
                {"prompt": prompt, "metadata": metadata},
                sort_keys=True,
                default=str,
                allow_nan=False,
            )
            prompt_key = (
                hashlib.sha256(identity.encode()).hexdigest() if prompt is not None else None
            )
            reward_values = _finite_values(batch.rewards[positions.to(batch.rewards.device)])
            finite_rewards = [value for value in reward_values if value is not None]
            components = {}
            for name, values in batch.extras.get("reward_components", {}).items():
                tensor = torch.as_tensor(values)
                if tensor.shape != (count,):
                    raise ValueError("admission reward component must match rollout sample count")
                components[name] = _finite_values(tensor[positions.to(tensor.device)])
            if kept == len(positions):
                decision, reason = "keep", "nonzero_advantage"
            elif kept == 0:
                decision, reason = "drop", "zero_advantage"
            else:
                decision, reason = "partial", "some_zero_advantages"
            if not drop_zero_advantage:
                reason = "filter_disabled"
            records.append(
                {
                    "group_id": int(group),
                    "request_id": trajectory.request_id if trajectory is not None else None,
                    "sample_ids": [row.sample_id for row in group_rows]
                    if rows is not None
                    else None,
                    "prompt_id": explicit_id,
                    "prompt_key": prompt_key,
                    "prompt": prompt,
                    # Preserve the same normalized inputs used for the key,
                    # including source/task IDs needed to inspect dropped groups.
                    "input_metadata": json.loads(identity)["metadata"],
                    "identity_status": "recorded" if rows is not None else "missing_trajectory",
                    "rollout_policy_version": version,
                    "rewards": reward_values,
                    "reward_components": components,
                    "reward_min": min(finite_rewards) if finite_rewards else None,
                    "reward_max": max(finite_rewards) if finite_rewards else None,
                    "advantages": _finite_values(advantage[positions.to(advantage.device)]),
                    "advantage_shape": list(advantage[positions.to(advantage.device)].shape),
                    "selected_rows": group_mask.tolist(),
                    "selected_count": kept,
                    "sample_count": len(positions),
                    "decision": decision,
                    "reason": reason,
                    "failure_attribution": "undetermined",
                }
            )
        if not drop_zero_advantage:
            selected_batches.append(batch)
            selected_advantages.append(advantage)
        elif bool(mask.any()):
            if not bool(mask.all()):
                batch = batch.select(mask)
                advantage = advantage[mask.to(advantage.device)]
            if batch.rewards.shape[0] > 0:
                selected_batches.append(batch)
                selected_advantages.append(advantage)
    return selected_batches, selected_advantages, records


class AdmissionLedger:
    """One writer per process attempt; never append across a crashed/resumed attempt."""

    def __init__(self, output_dir: str | Path, *, rank: int) -> None:
        self.attempt_id = uuid.uuid4().hex
        self.rank = rank
        self.path = Path(output_dir) / "admission" / f"{self.attempt_id}.rank-{rank:05d}.jsonl"
        self._collection = 0

    def record(
        self, decisions: list[dict[str, Any]], *, trainer_step: int, global_step: int
    ) -> None:
        lines = [
            json.dumps(
                {
                    "schema": "vrl.rollout-admission.v1",
                    "attempt_id": self.attempt_id,
                    "rank": self.rank,
                    "collection": self._collection,
                    "trainer_step": trainer_step,
                    "global_step": global_step,
                    "stage": "post_advantage_selection",
                    "optimizer_applied": None,
                    **decision,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for decision in decisions
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x" if self._collection == 0 else "a", encoding="utf-8") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        self._collection += 1
