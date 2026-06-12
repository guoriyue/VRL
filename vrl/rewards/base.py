"""RewardFunction base class for async rollout scoring."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
    RewardInferenceRuntime,
)
from vrl.rewards.types import RewardRollout

ArtifactBuilder = Callable[[list[RewardRollout]], list[RewardInferenceArtifact]]


class RewardFunction:
    """Base class for rollout rewards.

    Subclasses can either override ``score`` / ``score_batch`` directly, or pass
    an inference runtime plus artifact builder to reuse the standard model-backed
    scoring path.
    """

    @staticmethod
    def build_inmemory_artifacts(
        rollouts: list[RewardRollout],
        *,
        media_type: str = "image",
    ) -> list[RewardInferenceArtifact]:
        """Build reward artifacts that carry media in-memory (no disk write)."""

        artifacts: list[RewardInferenceArtifact] = []
        for index, rollout in enumerate(rollouts):
            metadata = dict(rollout.metadata or {})
            policy_version = metadata.get("policy_version")
            artifacts.append(
                RewardInferenceArtifact(
                    artifact_id=f"local-{index}",
                    path="",
                    media_type=media_type,
                    media=rollout.trajectory.output,
                    prompt=str(rollout.trajectory.prompt),
                    sample_id=f"sample-{index}",
                    policy_version=None if policy_version is None else int(policy_version),
                    metadata=metadata,
                ),
            )
        return artifacts

    def __init__(
        self,
        *,
        reward_name: str = "",
        score_key: str = "",
        runtime: RewardInferenceRuntime | None = None,
        artifact_builder: ArtifactBuilder | None = None,
        request_metadata: Mapping[str, Any] | None = None,
        debug_dir: str = "",
        request_prefix: str = "reward",
        debug_basename: str = "reward",
    ) -> None:
        self.reward_name = str(reward_name)
        self.score_key = str(score_key)
        self.runtime = runtime
        self._artifact_builder = artifact_builder
        self._request_metadata = dict(request_metadata or {})
        self.debug_dir = str(debug_dir)
        self._request_prefix = request_prefix
        self._debug_basename = debug_basename
        self.last_results: list[RewardInferenceResult] = []

    async def score(self, rollout: RewardRollout) -> float:
        """Score a single rollout."""
        if self._uses_inference_runtime():
            return (await self.score_batch([rollout]))[0]
        raise NotImplementedError(f"{type(self).__name__}.score is not implemented")

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        """Score a batch of rollouts (default: sequential)."""
        if self._uses_inference_runtime():
            return await self._score_with_inference_runtime(rollouts)
        return [await self.score(r) for r in rollouts]

    async def shutdown(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            await runtime.shutdown()

    def _uses_inference_runtime(self) -> bool:
        return (
            getattr(self, "runtime", None) is not None
            and getattr(self, "_artifact_builder", None) is not None
        )

    def _init_reward_model(
        self,
        *,
        reward_name: str,
        score_key: str,
        model_factory: str,
        worker_config: Mapping[str, Any],
        inference_runtime: str,
        media_type: str = "image",
    ) -> None:
        """Initialize a RewardFunction backed by a RewardModel factory."""

        from vrl.rewards.runtime import make_reward_runtime

        RewardFunction.__init__(
            self,
            reward_name=reward_name,
            score_key=score_key,
            runtime=make_reward_runtime(
                inference_runtime,
                model_factory=model_factory,
                worker_config=worker_config,
            ),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts,
                media_type=media_type,
            ),
        )

    def _init_disk_artifact_reward(
        self,
        *,
        model_factory: str,
        config_key: str,
        request_prefix: str,
        debug_basename: str,
        inference_runtime: str = "ray",
        reward_name: str | None = None,
        score_key: str | None = None,
        media_type: str = "video",
        artifact_dir: str = "outputs/reward_artifacts",
        artifact_format: str | None = None,
        default_reward_name: str = "",
        default_score_key: str = "",
        default_artifact_format: str = "tensor",
        debug_dir: str = "",
        max_inflight_batches: int = 1,
        scheduling: str = "sync",
        actor_runtime: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a reward whose media is materialized to disk, scored by a Ray pool.

        Sibling to :meth:`_init_reward_model`: same idea (configure ``self`` as a
        ``RewardFunction``), but the heavyweight path — media is written to disk
        via ``VideoRewardArtifactStore`` and scored by a Ray actor pool on its own
        GPU, instead of passed in-memory to a local model. ``model_factory`` /
        ``config_key`` / ``request_prefix`` / ``debug_basename`` and the
        ``default_*`` values are the only per-reward differences; everything else
        is shared wiring, so no concrete reward copies this body.
        """

        from vrl.rewards.artifacts import VideoRewardArtifactStore
        from vrl.rewards.ray.runtime import RayRewardRuntime

        if str(scheduling) != "sync":
            raise ValueError(
                f"reward.kwargs.{config_key}.scheduling currently supports only 'sync'",
            )
        self.inference_runtime = str(inference_runtime)
        if self.inference_runtime != "ray":
            raise ValueError(f"reward.kwargs.{config_key}.inference_runtime must be 'ray'")

        resolved_reward_name = str(
            reward_name if reward_name is not None else default_reward_name,
        )
        resolved_score_key = str(score_key if score_key is not None else default_score_key)
        resolved_format = str(
            artifact_format if artifact_format is not None else default_artifact_format,
        )

        self.media_type = str(media_type)
        self.artifact_store = VideoRewardArtifactStore(
            artifact_dir,
            media_type=self.media_type,
            artifact_format=resolved_format,
        )

        if actor_runtime is not None:
            runtime: Any = RayRewardRuntime(actor_runtime=actor_runtime)
        else:
            runtime_cfg = {**dict(kwargs), "max_inflight_batches": max_inflight_batches}
            worker_config = runtime_cfg.get("worker_config")
            if isinstance(worker_config, dict):
                worker_config = dict(worker_config)
                has_model_factory = bool(
                    str(worker_config.get("model_factory", "")).strip(),
                )
                reward_model_name = str(
                    worker_config.get("reward_model_name")
                    or worker_config.get("model_name")
                    or resolved_reward_name
                    or "",
                ).strip()
                model_path = str(worker_config.get("model_path", "")).strip()
                # YAML names the public model; workers need the private loader.
                if not has_model_factory and (reward_model_name or model_path):
                    if reward_model_name:
                        worker_config["reward_model_name"] = reward_model_name
                    worker_config["model_factory"] = model_factory
                    if not str(worker_config.get("reward_model_version", "")).strip():
                        worker_config["reward_model_version"] = (
                            reward_model_name or model_path
                        )
                runtime_cfg["worker_config"] = worker_config
            runtime = RayRewardRuntime(runtime_cfg)

        RewardFunction.__init__(
            self,
            reward_name=resolved_reward_name,
            score_key=resolved_score_key,
            runtime=runtime,
            artifact_builder=self.artifact_store.materialize,
            request_metadata={"media_type": self.media_type},
            debug_dir=debug_dir,
            request_prefix=request_prefix,
            debug_basename=debug_basename,
        )

    @property
    def _actor_runtime(self) -> Any:
        """The Ray actor pool backing this reward, or ``None`` for local runtimes.

        Generic over any Ray-backed reward (the trainer's resource layer uses it
        to release the reward GPUs after scoring); not specific to one reward type.
        """

        return getattr(self.runtime, "_actor", None)

    async def _score_with_inference_runtime(
        self,
        rollouts: list[RewardRollout],
    ) -> list[float]:
        if not rollouts:
            return []

        runtime = self.runtime
        artifact_builder = self._artifact_builder
        if runtime is None or artifact_builder is None:
            raise RuntimeError("RewardFunction inference runtime is not configured")

        total_started = time.perf_counter()
        materialize_started = time.perf_counter()
        artifacts = artifact_builder(rollouts)
        materialization_ms = (time.perf_counter() - materialize_started) * 1000.0
        policy_version = artifacts[0].policy_version if artifacts else None
        request = RewardInferenceRequest(
            request_id=f"{self._request_prefix}-{uuid.uuid4().hex}",
            artifacts=tuple(artifacts),
            reward_name=self.reward_name,
            score_key=self.score_key,
            policy_version=policy_version,
            metadata={
                **self._request_metadata,
                "artifact_materialization_ms": materialization_ms,
            },
        )
        inference_started = time.perf_counter()
        results = await runtime.score_batch(request)
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
        results: list[RewardInferenceResult],
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
        requests_file = debug_path / f"{self._debug_basename}_requests.jsonl"
        results_file = debug_path / f"{self._debug_basename}_results.jsonl"
        with requests_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request_row, sort_keys=True) + "\n")
        with results_file.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")

__all__ = ["ArtifactBuilder", "RewardFunction"]
