"""Shared reward scoring for rollout collectors."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.rewards.base import RewardFunction
from vrl.rewards.inference import RewardMemoryReleaseProof
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


@dataclass(frozen=True, slots=True)
class RewardScoreBatch:
    """One aligned scoring result owned by the current collector call."""

    scores: list[torch.Tensor]
    components: dict[str, list[float]]
    timing_ms: dict[str, float]


class RewardScorer:
    """Score decoded rollout outputs without knowing RolloutBatch layout."""

    def __init__(self, reward_fn: Any | None) -> None:
        self.reward_fn = reward_fn
        self._shutdown_complete = False
        self._score_lock = asyncio.Lock()

    async def shutdown(self) -> None:
        """Release the owned reward function exactly once after success."""

        if self._shutdown_complete:
            return
        reward_fn = self.reward_fn
        if reward_fn is not None:
            shutdown = getattr(reward_fn, "shutdown", None)
            if callable(shutdown):
                result = shutdown()
                if inspect.isawaitable(result):
                    await result
        self._shutdown_complete = True

    async def score_many(
        self,
        requests: list[RewardScoringInput],
        *,
        require_memory_release: bool = False,
    ) -> RewardScoreBatch:
        """Score several prompt groups through one reward call.

        Rollout metadata/prompts are per-sample, so groups concatenate into a
        single score_batch call — model-backed rewards then pay one actor
        lifecycle (and one inference request) per call instead of one per
        group. Scores split back by each request's batch size.
        """

        async with self._score_lock:
            return await self._score_many_locked(
                requests,
                require_memory_release=require_memory_release,
            )

    async def _score_many_locked(
        self,
        requests: list[RewardScoringInput],
        *,
        require_memory_release: bool,
    ) -> RewardScoreBatch:
        if not requests:
            return RewardScoreBatch(scores=[], components={}, timing_ms={})
        if self.reward_fn is None:
            if require_memory_release:
                raise RuntimeError(
                    "shared reward topology requires a reward memory release proof, "
                    "but no reward function is configured",
                )
            scores = [
                torch.zeros(request.batch_size, device=request.device) for request in requests
            ]
            return RewardScoreBatch(scores=scores, components={}, timing_ms={})

        rollouts = [
            RewardRollout(
                request=None,
                trajectory=RewardTrajectory(
                    prompt=request.prompts[i],
                    output=request.outputs[i],
                ),
                metadata=dict(request.metadata),
            )
            for request in requests
            for i in range(request.batch_size)
        ]

        components: dict[str, list[float]] = {}
        timing_ms: dict[str, float] = {}
        report_fn = getattr(self.reward_fn, "score_batch_report", None)
        if callable(report_fn):
            report = report_fn(rollouts)
            if inspect.isawaitable(report):
                report = await report
            raw = report.scores
            components = {
                str(name): [float(value) for value in values]
                for name, values in dict(report.components).items()
            }
            timing_ms = {str(key): float(value) for key, value in dict(report.timing_ms).items()}
        else:
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
        if len(scores) != len(rollouts):
            raise ValueError(
                "reward function returned wrong number of scores: "
                f"scores={len(scores)}, expected={len(rollouts)}",
            )
        for name, values in components.items():
            if len(values) != len(rollouts):
                raise ValueError(
                    "reward component returned wrong number of scores: "
                    f"component={name!r}, scores={len(values)}, "
                    f"expected={len(rollouts)}",
                )
        if require_memory_release:
            await self.park_memory(required=True)

        split: list[torch.Tensor] = []
        offset = 0
        for request in requests:
            chunk = scores[offset : offset + request.batch_size]
            offset += request.batch_size
            split.append(
                torch.tensor(chunk, device=request.device, dtype=torch.float32),
            )
        return RewardScoreBatch(
            scores=split,
            components=components,
            timing_ms=timing_ms,
        )

    async def park_memory(
        self,
        *,
        required: bool,
    ) -> tuple[RewardMemoryReleaseProof, ...]:
        """Actively park reward owners and validate a fresh proof result."""

        if self.reward_fn is None:
            if required:
                raise RuntimeError(
                    "shared reward topology requires a reward memory release proof, "
                    "but no reward function is configured",
                )
            return ()
        if not isinstance(self.reward_fn, RewardFunction):
            if required:
                raise RuntimeError(
                    "shared reward topology requires release proof support from "
                    f"{type(self.reward_fn).__name__}",
                )
            return ()
        # Actively invoke the idempotent release operation. Reading cached proof
        # would let an earlier request authorize the current handoff.
        proofs = tuple(await self.reward_fn.park_memory())
        if required and not proofs:
            raise RuntimeError("reward function returned an empty memory release proof")
        for proof in proofs:
            if not proof.released:
                raise RuntimeError("reward function did not release GPU memory")
        return proofs


__all__ = ["RewardScoreBatch", "RewardScorer", "RewardScoringInput"]
