"""Video-level reward entry point for world-model RL."""

from __future__ import annotations

from typing import Any

import torch

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout


class VideoReward(RewardFunction):
    """Video/image reward placeholder until the Ray reward runtime is wired."""

    def __init__(
        self,
        *,
        backend: str,
        reward_name: str,
        score_key: str,
        device: str = "cuda",
        media_type: str = "video",
        timeout_s: float = 60.0,
        debug_dir: str = "",
        stub_scale: float = 0.01,
        **kwargs: Any,
    ) -> None:
        del device, timeout_s, debug_dir, kwargs
        self.backend = str(backend)
        self.reward_name = str(reward_name)
        self.score_key = str(score_key)
        self.media_type = str(media_type)
        self.stub_scale = float(stub_scale)
        if self.backend != "stub":
            raise NotImplementedError(
                "video_reward only supports backend='stub' until the Ray reward "
                f"inference runtime is implemented; got backend={self.backend!r}.",
            )

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        if not rollouts:
            return []
        return self._stub_scores(rollouts)

    def _stub_scores(self, rollouts: list[RewardRollout]) -> list[float]:
        scores: list[float] = []
        for idx, rollout in enumerate(rollouts):
            output = rollout.trajectory.output
            if isinstance(output, torch.Tensor):
                base = float(output.detach().float().mean().item())
            else:
                base = 0.0
            scores.append(base + self.stub_scale * (idx + 1))
        return scores


__all__ = ["VideoReward"]
