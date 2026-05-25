"""Video-level reward entry point for world-model RL.

``VideoReward`` is a ``RewardFunction`` specialized for the Ray transport: it
materializes rollouts into media artifacts on disk, then scores them through a
Ray actor pool of reward workers (the model needs its own GPU). It keeps a
ray-only contract; cheap in-process rewards use ``RewardFunction`` directly with
the local transport.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.artifacts import VideoRewardArtifactStore
from vrl.rewards.base import RewardFunction
from vrl.rewards.ray.runtime import RayRewardRuntime

_KLING_VIDEO_REWARD_MODEL = (
    "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"
)


class VideoReward(RewardFunction):
    """Reward whose model runs in a Ray actor pool on a separate GPU."""

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
        actor_runtime: Any | None = None,
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
        self.media_type = str(media_type)
        self.artifact_store = VideoRewardArtifactStore(
            artifact_dir,
            media_type=self.media_type,
            artifact_format=str(artifact_format),
        )

        if actor_runtime is not None:
            runtime = RayRewardRuntime(actor_runtime=actor_runtime)
        else:
            runtime_cfg = {
                **dict(kwargs),
                "max_inflight_batches": max_inflight_batches,
            }
            worker_config = runtime_cfg.get("worker_config")
            if isinstance(worker_config, dict):
                runtime_cfg["worker_config"] = _normalize_worker_config(
                    worker_config,
                    reward_name=str(reward_name),
                )
            runtime = RayRewardRuntime(runtime_cfg)

        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            runtime=runtime,
            artifact_builder=self.artifact_store.materialize,
            request_metadata={"media_type": self.media_type},
            debug_dir=debug_dir,
            request_prefix="video-reward",
            debug_basename="video_reward",
        )

    @property
    def _actor_runtime(self) -> Any:
        """Underlying Ray actor pool (back-compat accessor for resource checks)."""

        return self.runtime._actor


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
