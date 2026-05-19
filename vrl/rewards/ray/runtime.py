"""Ray-backed reward inference runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.ray.runtime import RayActorMethodRuntime
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    shard_reward_request,
    validate_reward_results,
)


@dataclass(slots=True)
class RewardInferenceActorRuntime:
    """Reward request/result adapter over a generic Ray actor-method runtime."""

    actor_runtime: RayActorMethodRuntime

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        if not request.artifacts:
            return []
        shards = shard_reward_request(
            request,
            num_shards=self.actor_runtime.num_workers,
        )
        nested = await self.actor_runtime.map(shards)
        results = [result for shard_results in nested for result in shard_results]
        return validate_reward_results(request, results)

    async def shutdown(self) -> None:
        await self.actor_runtime.shutdown()


__all__ = ["RewardInferenceActorRuntime"]
