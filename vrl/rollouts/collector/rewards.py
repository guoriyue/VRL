"""Shared reward scoring for rollout collectors."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.rewards.types import RewardRollout, RewardTrajectory


@dataclass(frozen=True, slots=True)
class RewardScoringInput:
    """Batch-aligned reward scorer input built from one engine GenerationOutput."""

    outputs: Any
    prompts: Sequence[str]
    metadata: Mapping[str, Any]
    device: Any
    expected_count: int | None = None
    batch_size: int = field(init=False)

    def __post_init__(self) -> None:
        batch_size = self._outputs_batch_size(self.outputs)
        if self.expected_count is not None and batch_size != self.expected_count:
            raise ValueError(
                "reward output/sample batch mismatch: "
                f"outputs={batch_size}, samples={self.expected_count}",
            )
        if len(self.prompts) != batch_size:
            raise ValueError(
                "reward prompt/output batch mismatch: "
                f"prompts={len(self.prompts)}, outputs={batch_size}",
            )
        object.__setattr__(self, "batch_size", batch_size)

    @staticmethod
    def _outputs_batch_size(outputs: Any) -> int:
        shape = getattr(outputs, "shape", None)
        if shape is not None:
            if len(shape) == 0:
                raise ValueError("reward outputs must have a batch dimension")
            return int(shape[0])
        try:
            return len(outputs)
        except TypeError as exc:
            raise TypeError(
                "reward outputs must expose shape[0] or len()",
            ) from exc


class RewardScorer:
    """Score decoded rollout outputs without knowing RolloutBatch layout."""

    def __init__(self, reward_fn: Any | None) -> None:
        self.reward_fn = reward_fn

    async def score(
        self,
        request: RewardScoringInput,
    ) -> torch.Tensor:
        if self.reward_fn is None:
            return torch.zeros(request.batch_size, device=request.device)

        rollouts = [
            RewardRollout(
                request=None,
                trajectory=RewardTrajectory(
                    prompt=request.prompts[i],
                    output=request.outputs[i],
                ),
                metadata=dict(request.metadata),
            )
            for i in range(request.batch_size)
        ]

        batch_fn = getattr(self.reward_fn, "score_batch", None)
        if batch_fn is not None:
            raw = batch_fn(rollouts)
            if inspect.isawaitable(raw):
                raw = await raw
        else:
            raw = []
            for rollout in rollouts:
                value = self.reward_fn.score(rollout)
                if inspect.isawaitable(value):
                    value = await value
                raw.append(value)

        scores = [float(score) for score in raw]
        if len(scores) != request.batch_size:
            raise ValueError(
                "reward function returned wrong number of scores: "
                f"scores={len(scores)}, expected={request.batch_size}",
            )
        return torch.tensor(scores, device=request.device, dtype=torch.float32)


__all__ = ["RewardScorer", "RewardScoringInput"]
