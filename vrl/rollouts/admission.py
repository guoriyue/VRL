"""Post-advantage sample admission with a per-attempt audit ledger.

Algorithms own normalization, including cross-rank statistics. Admission
consumes their prepared advantages, applies the existing exact-zero row mask,
and records every group's decision so a dropped prompt can be inspected later.
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


class AdmissionLedger:
    """Select trainable rows and append one audit record per prompt group.

    One file per process attempt; a resumed attempt never appends to a crashed
    one's file.
    """

    def __init__(self, output_dir: str | Path, *, rank: int) -> None:
        self.attempt_id = uuid.uuid4().hex
        self.rank = rank
        self.path = Path(output_dir) / "admission" / f"{self.attempt_id}.rank-{rank:05d}.jsonl"
        self._collection = 0

    def admit(
        self,
        batches: list[RolloutBatch],
        advantages: list[Any],
        *,
        drop_zero_advantage: bool,
        trainer_step: int,
        global_step: int,
    ) -> tuple[list[RolloutBatch], list[Any]]:
        """Apply the zero-advantage row mask and record each group's decision."""

        import torch

        selected_batches, selected_advantages, records = [], [], []
        for batch, advantage in zip(batches, advantages, strict=True):
            count = batch.rewards.shape[0]
            mask = (
                nonzero_advantage_mask(advantage)
                if drop_zero_advantage
                else torch.ones(count, dtype=torch.bool, device=advantage.device)
            )
            groups = batch.group_ids.detach().cpu()
            mask_cpu = mask.detach().cpu()
            trajectory = batch.trajectory
            rows = trajectory.sample_rows if trajectory is not None else None
            metadata = dict(batch.context.get("reward_metadata", {}))
            version = metadata.pop("rollout_policy_version", None)
            for group in dict.fromkeys(groups.tolist()):
                positions = (groups == group).nonzero(as_tuple=True)[0]
                group_mask = mask_cpu[positions]
                kept = int(group_mask.sum().item())
                group_rows = [rows[i] for i in positions.tolist()] if rows is not None else []
                prompt = group_rows[0].prompt if group_rows else None
                # Source/target metadata participates, so identical text with
                # different references does not become one history bucket.
                identity = json.dumps(
                    {"prompt": prompt, "metadata": metadata},
                    sort_keys=True,
                    default=str,
                    allow_nan=False,
                )
                rewards = _finite_values(batch.rewards[positions.to(batch.rewards.device)])
                finite = [value for value in rewards if value is not None]
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
                        "sample_ids": [row.sample_id for row in group_rows] if rows else None,
                        "prompt_id": metadata.get(
                            "prompt_id", metadata.get("id", metadata.get("task_id"))
                        ),
                        "prompt_key": hashlib.sha256(identity.encode()).hexdigest()
                        if prompt is not None
                        else None,
                        "prompt": prompt,
                        "input_metadata": json.loads(identity)["metadata"],
                        "rollout_policy_version": version,
                        "rewards": rewards,
                        "reward_components": {
                            name: _finite_values(
                                torch.as_tensor(values)[
                                    positions.to(torch.as_tensor(values).device)
                                ]
                            )
                            for name, values in batch.extras.get("reward_components", {}).items()
                        },
                        "reward_min": min(finite) if finite else None,
                        "reward_max": max(finite) if finite else None,
                        "advantages": _finite_values(advantage[positions.to(advantage.device)]),
                        "selected_rows": group_mask.tolist(),
                        "selected_count": kept,
                        "sample_count": len(positions),
                        "decision": decision,
                        "reason": reason,
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
        self._record(records, trainer_step=trainer_step, global_step=global_step)
        return selected_batches, selected_advantages

    def _record(
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


__all__ = ["AdmissionLedger"]
