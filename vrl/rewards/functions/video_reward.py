"""Video-level reward entry point for world-model RL.

VideoReward is the only driver-side concept: it inherits RewardFunction (so it
composes inside MultiReward), materializes rollouts into artifacts, runs them
through a Ray actor pool of RewardModelWorkers, and validates the results.
The worker side (RewardModelWorker hosting a RewardModel like Kling) runs in a
separate process because the reward model needs its own GPU.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.ray.runtime import RayActorMethodRuntime
from vrl.rewards.artifacts import VideoRewardArtifactStore
from vrl.rewards.base import RewardFunction
from vrl.rewards.inference import RewardInferenceRequest
from vrl.rewards.ray.runtime import build_reward_actor_runtime, score_reward_request
from vrl.rewards.types import RewardRollout

_KLING_VIDEO_REWARD_MODEL = (
    "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"
)


class VideoReward(RewardFunction):
    """RewardFunction whose model runs in a Ray actor pool on a separate GPU."""

    def __init__(
        self,
        *,
        inference_runtime: str = "ray",
        reward_name: str,
        score_key: str,
        media_type: str = "video",
        artifact_dir: str = "outputs/reward_artifacts",
        artifact_format: str = "tensor",
        debug_dir: str = "",
        timeout_s: float = 60.0,
        max_inflight_batches: int = 1,
        scheduling: str = "sync",
        backend: str | None = None,
        actor_runtime: RayActorMethodRuntime | None = None,
        **kwargs: Any,
    ) -> None:
        del timeout_s
        if backend is not None:
            raise ValueError(
                "reward.kwargs.video_reward.backend is no longer supported; "
                "use inference_runtime='ray'",
            )
        if str(scheduling) != "sync":
            raise ValueError(
                "reward.kwargs.video_reward.scheduling currently supports only 'sync'",
            )
        self.inference_runtime = str(inference_runtime)
        if self.inference_runtime != "ray":
            raise ValueError("reward.kwargs.video_reward.inference_runtime must be 'ray'")
        self.reward_name = str(reward_name)
        self.score_key = str(score_key)
        self.media_type = str(media_type)
        self.artifact_store = VideoRewardArtifactStore(
            artifact_dir,
            media_type=self.media_type,
            artifact_format=str(artifact_format),
        )
        self.debug_dir = str(debug_dir)
        self.last_results: list[Any] = []
        if actor_runtime is not None:
            self._actor_runtime = actor_runtime
        else:
            runtime_cfg = {
                **dict(kwargs),
                "max_inflight_batches": max_inflight_batches,
            }
            worker_config = runtime_cfg.get("worker_config")
            if isinstance(worker_config, dict):
                runtime_cfg["worker_config"] = _normalize_worker_config(
                    worker_config,
                    reward_name=self.reward_name,
                )
            self._actor_runtime = build_reward_actor_runtime(runtime_cfg)

    async def score(self, rollout: RewardRollout) -> float:
        return (await self.score_batch([rollout]))[0]

    async def shutdown(self) -> None:
        await self._actor_runtime.shutdown()

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        if not rollouts:
            return []
        total_started = time.perf_counter()
        materialize_started = time.perf_counter()
        artifacts = self.artifact_store.materialize(rollouts)
        materialization_ms = (time.perf_counter() - materialize_started) * 1000.0
        policy_version = artifacts[0].policy_version if artifacts else None
        request = RewardInferenceRequest(
            request_id=f"video-reward-{uuid.uuid4().hex}",
            artifacts=tuple(artifacts),
            reward_name=self.reward_name,
            score_key=self.score_key,
            policy_version=policy_version,
            metadata={
                "media_type": self.media_type,
                "artifact_materialization_ms": materialization_ms,
            },
        )
        inference_started = time.perf_counter()
        results = await score_reward_request(self._actor_runtime, request)
        inference_total_ms = (time.perf_counter() - inference_started) * 1000.0
        total_latency_ms = (time.perf_counter() - total_started) * 1000.0
        self.last_results = list(results)
        self._write_debug(
            request,
            results,
            artifact_materialization_ms=materialization_ms,
            inference_total_ms=inference_total_ms,
            total_reward_latency_ms=total_latency_ms,
        )
        return [float(result.selected_score) for result in results]

    def _write_debug(
        self,
        request: RewardInferenceRequest,
        results: list[Any],
        *,
        artifact_materialization_ms: float,
        inference_total_ms: float,
        total_reward_latency_ms: float,
    ) -> None:
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
            "artifact_materialization_ms": artifact_materialization_ms,
            "inference_total_ms": inference_total_ms,
            "total_reward_latency_ms": total_reward_latency_ms,
        }
        with (debug_path / "video_reward_requests.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request_row, sort_keys=True) + "\n")
        with (debug_path / "video_reward_results.jsonl").open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def _normalize_worker_config(
    worker_config: dict[str, Any],
    *,
    reward_name: str,
) -> dict[str, Any]:
    config = dict(worker_config)
    if str(config.get("model_factory", "")).strip():
        return config

    reward_model_name = str(
        config.get("reward_model_name")
        or config.get("model_name")
        or reward_name
        or "",
    ).strip()
    model_path = str(config.get("model_path", "")).strip()
    if not reward_model_name and not model_path:
        return config

    if reward_model_name:
        config["reward_model_name"] = reward_model_name
    config["model_factory"] = _KLING_VIDEO_REWARD_MODEL
    if not str(config.get("reward_model_version", "")).strip():
        config["reward_model_version"] = reward_model_name or model_path
    return config


__all__ = ["VideoReward"]
