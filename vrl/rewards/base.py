"""RewardFunction base class for async rollout scoring."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from vrl.rewards.inference import (
    MediaType,
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
        media_type: MediaType = "image",
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
        self.last_timing_ms: dict[str, float] = {}

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
        execution: Literal["inline"],
        media_type: MediaType = "image",
    ) -> None:
        """Initialize a RewardFunction backed by a RewardModel factory."""

        from vrl.rewards.runtime import make_reward_runtime

        RewardFunction.__init__(
            self,
            reward_name=reward_name,
            score_key=score_key,
            runtime=make_reward_runtime(
                execution,
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
        execution: Literal["inline"] = "inline",
        reward_name: str | None = None,
        score_key: str | None = None,
        media_type: MediaType = "video",
        artifact_dir: str = "outputs/reward_artifacts",
        artifact_format: str | None = None,
        default_reward_name: str = "",
        default_score_key: str = "",
        default_artifact_format: str = "tensor",
        debug_dir: str = "",
        device: str | None = None,
        sleep_offload: bool = False,
        worker_config: Mapping[str, Any] | None = None,
        runtime: Any | None = None,
    ) -> None:
        """Initialize a reward whose media is materialized to disk before scoring.

        Sibling to :meth:`_init_reward_model`: same idea (configure ``self`` as a
        ``RewardFunction``), but the heavyweight path — media is written to disk
        via ``VideoRewardArtifactStore`` and scored in-process, instead of passed
        in-memory. ``sleep_offload`` parks the model on CPU between scores (the
        rollout/trainer own the GPU then), mirroring the rollout lease's
        sleep/wake. ``model_factory`` / ``config_key`` / ``request_prefix`` /
        ``debug_basename`` and the ``default_*`` values are the only per-reward
        differences; everything else is shared wiring, so no concrete reward
        copies this body. ``runtime`` injects a ready ``RewardInferenceRuntime``
        (tests); it wins over the factory-built one.
        """

        from vrl.rewards.artifacts import VideoRewardArtifactStore
        from vrl.rewards.runtime import LocalRewardRuntime

        if str(execution) != "inline":
            raise ValueError(
                f"reward.kwargs.{config_key}.execution={execution!r} is no longer "
                "supported: the Ray reward pool was removed and rewards score "
                "in-process. Drop the key (inline is the default); for a "
                "heavyweight model on a shared GPU set "
                f"reward.kwargs.{config_key}.sleep_offload=true instead.",
            )

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

        if runtime is None:
            worker_cfg = dict(worker_config or {})
            has_model_factory = bool(
                str(worker_cfg.get("model_factory", "")).strip(),
            )
            # Normalize the model-id key ONCE here so the disk loaders
            # (kling/videocon) read only worker_config["reward_model_name"].
            # Precedence: an explicit worker_config.reward_model_name wins;
            # otherwise fold worker_config.model_name (deprecated alias) or a
            # top-level reward_name that looks like a HF repo (contains "/")
            # — a bare reward_name stays a logical tag, not a model id.
            reward_name_repo = (
                resolved_reward_name if "/" in resolved_reward_name else ""
            )
            reward_model_name = str(
                worker_cfg.get("reward_model_name")
                or worker_cfg.get("model_name")  # deprecated alias
                or reward_name_repo
                or "",
            ).strip()
            model_path = str(worker_cfg.get("model_path", "")).strip()
            # YAML names the public model; the loader needs the private factory.
            if not has_model_factory and (reward_model_name or model_path):
                if reward_model_name:
                    worker_cfg["reward_model_name"] = reward_model_name
                worker_cfg["model_factory"] = model_factory
                if not str(worker_cfg.get("reward_model_version", "")).strip():
                    worker_cfg["reward_model_version"] = (
                        reward_model_name or model_path
                    )
            # An explicit worker_config.device wins; otherwise the resolved
            # reward device from the caller (MultiReward.from_dict / factory).
            if device is not None and not str(worker_cfg.get("device", "")).strip():
                worker_cfg["device"] = str(device)
            if sleep_offload:
                worker_cfg["sleep_offload"] = True
            runtime = LocalRewardRuntime(worker_cfg)

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

    async def _score_with_inference_runtime(
        self,
        rollouts: list[RewardRollout],
    ) -> list[float]:
        if not rollouts:
            self.last_results = []
            self.last_timing_ms = {}
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
        # Contract enforcement lives at this seam, not inside each runtime, so
        # every RewardInferenceRuntime (including injected test fakes) gets the
        # same result/artifact mismatch guard and request-order re-sort.
        from vrl.rewards.inference import validate_reward_results

        results = validate_reward_results(
            request,
            await runtime.score_batch(request),
        )
        inference_total_ms = (time.perf_counter() - inference_started) * 1000.0
        total_latency_ms = (time.perf_counter() - total_started) * 1000.0
        self.last_results = list(results)
        self.last_timing_ms = {
            "latency_ms": total_latency_ms,
            "queue_wait_ms": _max_result_timing(results, "queue_wait_ms"),
            "inference_ms": _sum_result_timing(results, "inference_ms")
            if results
            else inference_total_ms,
        }
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


_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".webp"})


def decode_artifact_frames(artifact: Any, num_frames: int | None = None) -> Any:
    """Decode a reward artifact's generated media to a ``[T,H,W,3]`` float frame stack.

    Shared by the frame-comparison rewards (target_dino_similarity,
    motion_dynamics): an artifact carries either a materialized file path (the Ray
    disk path / probe mp4) or an in-memory tensor (the inline collector path), and
    both must yield the same ``[T,H,W,3]`` float tensor in ``[0,1]``. Lives in the
    rewards base (not utils/media) because it depends on the artifact contract.

    In-memory video media is channel-first ``[C,T,H,W]`` / ``[1,C,T,H,W]`` (the layout
    the collector emits and ``video_tensor_to_uint8_frames`` enforces, raising loudly on
    a wrong channel count); images are ``[C,H,W]``. The on-disk path (mp4/png) is the
    common case in practice (the probe and the Ray pool both materialize to disk).
    """

    import torch

    from vrl.utils.media import (
        frames_thwc_to_float,
        image_to_uint8_hwc,
        read_image_as_frames,
        read_video_frames,
        sample_frames,
        video_tensor_to_uint8_frames,
    )

    path = str(getattr(artifact, "path", "") or "")
    if path and not path.endswith(".pt"):
        if Path(path).suffix.lower() in _IMAGE_SUFFIXES:
            return read_image_as_frames(path)
        return read_video_frames(path, num_frames)
    media = artifact.as_media()
    if isinstance(media, torch.Tensor):
        if media.ndim in {4, 5}:
            frames = torch.from_numpy(video_tensor_to_uint8_frames(media))
            return sample_frames(frames_thwc_to_float(frames), num_frames)
        if media.ndim == 3:
            image = torch.from_numpy(image_to_uint8_hwc(media))
            return frames_thwc_to_float(image.unsqueeze(0))
    raise TypeError(
        f"reward artifact expected image/video tensor or media path, got {type(media)}",
    )


def _max_result_timing(results: list[RewardInferenceResult], field: str) -> float:
    values = [
        float(value)
        for result in results
        if (value := getattr(result, field, None)) is not None
    ]
    return max(values, default=0.0)


def _sum_result_timing(results: list[RewardInferenceResult], field: str) -> float:
    return sum(
        float(value)
        for result in results
        if (value := getattr(result, field, None)) is not None
    )


__all__ = ["ArtifactBuilder", "RewardFunction", "decode_artifact_frames"]
