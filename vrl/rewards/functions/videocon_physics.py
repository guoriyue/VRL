"""VideoCon-Physics reward function (Ray-actor transport).

Mirrors :class:`vrl.rewards.functions.video_reward.VideoReward` — same artifact
store, same Ray runtime, same ``_init_reward_model`` machinery. Only the model
factory differs: ``VideoConPhysicsModel`` loads the vendored mPLUG-Owl-Video
backbone with the VideoCon-Physics checkpoint and returns ``physical_commonsense``,
``semantic_adherence``, and ``overall`` sub-scores per artifact.

A ``score_key`` of ``physical_commonsense`` (the default in
``configs/reward/videocon_physics.yaml``) gives GRPO a pure physics signal.
Switch to ``overall`` to also reward caption faithfulness.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.artifacts import VideoRewardArtifactStore
from vrl.rewards.base import RewardFunction
from vrl.rewards.ray.runtime import RayRewardRuntime

_VIDEOCON_PHYSICS_MODEL = (
    "vrl.rewards.models.videocon_physics:VideoConPhysicsModel"
)


class VideoConPhysicsReward(RewardFunction):
    """Reward whose VideoCon-Physics actor runs in a Ray pool on a separate GPU."""

    def __init__(
        self,
        *,
        inference_runtime: str = "ray",
        reward_name: str = "videophysics/videocon_physics@main",
        score_key: str = "physical_commonsense",
        media_type: str = "video",
        artifact_dir: str = "outputs/reward_artifacts",
        artifact_format: str = "mp4",
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
                "reward.kwargs.videocon_physics.backend is no longer supported; "
                "use inference_runtime='ray'",
            )
        if str(scheduling) != "sync":
            raise ValueError(
                "reward.kwargs.videocon_physics.scheduling currently supports only 'sync'",
            )
        self.inference_runtime = str(inference_runtime)
        if self.inference_runtime != "ray":
            raise ValueError(
                "reward.kwargs.videocon_physics.inference_runtime must be 'ray'",
            )
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
            request_prefix="videocon-physics",
            debug_basename="videocon_physics",
        )

    @property
    def _actor_runtime(self) -> Any:
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
    config["model_factory"] = _VIDEOCON_PHYSICS_MODEL
    if not str(config.get("reward_model_version", "")).strip():
        config["reward_model_version"] = reward_model_name or model_path
    return config


__all__ = ["VideoConPhysicsReward"]
