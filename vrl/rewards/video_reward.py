"""Video-level reward entry point for world-model RL."""

from __future__ import annotations

from typing import Any

import torch

from vrl.algorithms.types import Rollout
from vrl.rewards.base import RewardFunction
from vrl.rewards.remote_video import RemoteVideoRewardClient, RemoteVideoRewardConfig


class VideoReward(RewardFunction):
    """Video/image reward with explicit stub, local, or remote backend."""

    def __init__(
        self,
        *,
        backend: str,
        reward_name: str,
        score_key: str,
        device: str = "cuda",
        enqueue_url: str = "",
        fetch_url: str = "",
        token: str = "",
        media_type: str = "video",
        timeout_s: float = 60.0,
        poll_interval_s: float = 1.0,
        max_wait_s: float = 600.0,
        debug_dir: str = "",
        stub_scale: float = 0.01,
        **kwargs: Any,
    ) -> None:
        del device
        self.backend = str(backend)
        self.reward_name = str(reward_name)
        self.score_key = str(score_key)
        self.media_type = str(media_type)
        self.stub_scale = float(stub_scale)
        if self.backend not in {"stub", "remote", "local"}:
            raise ValueError("video_reward.backend must be one of: stub, remote, local")
        if self.backend == "remote":
            self.remote = RemoteVideoRewardClient(
                RemoteVideoRewardConfig(
                    enqueue_url=enqueue_url,
                    fetch_url=fetch_url,
                    token=token,
                    reward_name=self.reward_name,
                    score_key=self.score_key,
                    media_type=self.media_type,
                    timeout_s=timeout_s,
                    poll_interval_s=poll_interval_s,
                    max_wait_s=max_wait_s,
                    debug_dir=debug_dir,
                    extra_metadata=dict(kwargs.get("extra_metadata", {}) or {}),
                ),
            )
        else:
            self.remote = None
        if self.backend == "local":
            raise NotImplementedError(
                "video_reward backend='local' is reserved for an explicit local "
                "dance_grpo/cosmos_reason1 wrapper; use backend='stub' or 'remote'.",
            )

    async def score(self, rollout: Rollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        if not rollouts:
            return []
        if self.backend == "stub":
            return self._stub_scores(rollouts)
        assert self.remote is not None
        media = _stack_outputs([rollout.trajectory.output for rollout in rollouts])
        prompts = [rollout.trajectory.prompt for rollout in rollouts]
        metadata = dict(rollouts[0].metadata or {})
        scores, _raw = await self.remote.score_batch(
            media=media,
            prompts=prompts,
            metadata=metadata,
        )
        return scores

    def _stub_scores(self, rollouts: list[Rollout]) -> list[float]:
        scores: list[float] = []
        for idx, rollout in enumerate(rollouts):
            output = rollout.trajectory.output
            if isinstance(output, torch.Tensor):
                base = float(output.detach().float().mean().item())
            else:
                base = 0.0
            scores.append(base + self.stub_scale * (idx + 1))
        return scores


def _stack_outputs(outputs: list[Any]) -> Any:
    if all(isinstance(output, torch.Tensor) for output in outputs):
        tensors = [output for output in outputs]
        if tensors[0].ndim in {3, 4}:
            return torch.stack(tensors, dim=0)
        if tensors[0].ndim in {4, 5} and tensors[0].shape[0] == 1:
            return torch.cat(tensors, dim=0)
    return outputs


__all__ = ["VideoReward"]
