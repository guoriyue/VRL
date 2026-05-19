"""Video-level reward entry point for world-model RL."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.rewards.artifacts import VideoRewardArtifactStore
from vrl.rewards.base import RewardFunction
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceRuntime,
    build_reward_inference_runtime,
    validate_reward_results,
)
from vrl.rewards.types import RewardRollout


class VideoReward(RewardFunction):
    """Video/image reward adapter over the generic reward inference runtime."""

    def __init__(
        self,
        *,
        inference_runtime: str = "ray",
        reward_name: str,
        score_key: str,
        media_type: str = "video",
        artifact_dir: str = "outputs/reward_artifacts",
        debug_dir: str = "",
        timeout_s: float = 60.0,
        max_inflight_batches: int = 1,
        backend: str | None = None,
        runtime: RewardInferenceRuntime | None = None,
        **kwargs: Any,
    ) -> None:
        del timeout_s
        if backend is not None:
            raise ValueError(
                "reward.kwargs.video_reward.backend is no longer supported; "
                "use inference_runtime='ray'",
            )
        self.inference_runtime = str(inference_runtime)
        self.reward_name = str(reward_name)
        self.score_key = str(score_key)
        self.media_type = str(media_type)
        self.artifact_store = VideoRewardArtifactStore(
            artifact_dir,
            media_type=self.media_type,
        )
        self.debug_dir = str(debug_dir)
        self.last_results: list[Any] = []
        if runtime is not None:
            self.runtime = runtime
        else:
            runtime_cfg = {
                **dict(kwargs),
                "inference_runtime": self.inference_runtime,
                "max_inflight_batches": max_inflight_batches,
            }
            self.runtime = build_reward_inference_runtime(runtime_cfg)

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        if not rollouts:
            return []
        artifacts = self.artifact_store.materialize(rollouts)
        policy_version = artifacts[0].policy_version if artifacts else None
        request = RewardInferenceRequest(
            request_id=f"video-reward-{uuid.uuid4().hex}",
            artifacts=tuple(artifacts),
            reward_name=self.reward_name,
            score_key=self.score_key,
            policy_version=policy_version,
            metadata={"media_type": self.media_type},
        )
        results = validate_reward_results(request, await self.runtime.score_batch(request))
        self.last_results = list(results)
        self._write_debug(request, results)
        return [float(result.selected_score) for result in results]

    def _write_debug(self, request: RewardInferenceRequest, results: list[Any]) -> None:
        if not self.debug_dir:
            return
        debug_path = Path(self.debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        request_row = {
            "request_id": request.request_id,
            "artifact_ids": [artifact.artifact_id for artifact in request.artifacts],
            "reward_name": request.reward_name,
            "score_key": request.score_key,
            "policy_version": request.policy_version,
        }
        with (debug_path / "video_reward_requests.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request_row, sort_keys=True) + "\n")
        with (debug_path / "video_reward_results.jsonl").open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


__all__ = ["VideoReward"]
