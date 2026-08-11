"""RewardFunction base class for async generated-sample scoring.

The plugin layer between ``RewardFunctionRuntime`` above (runtime.py) and the
``RewardScorer`` transports below (inference.py): concrete rewards in
vrl/rewards/functions subclass one of the bases here. The class ladder encodes
capabilities the registry and runtime probe, not taxonomy:
``InferenceRewardFunction`` owns the artifact-build / score / validate /
finalize seam so every transport (including injected fakes) passes the same
result-identity guard; ``CumemRewardFunction`` declares that all model CUDA
state is built in the tagged pool, enabling verified memory parking;
``DiskArtifactRewardFunction`` tells registry preflight that media must be
materialized to disk before scoring.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar, Literal

from vrl.rewards.inference import (
    ArtifactRetainingError,
    MediaType,
    MemoryParkingScorer,
    RemoteReadyScorer,
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
    RewardScorer,
    validate_reward_results,
)
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

ArtifactBuilder = Callable[[list[RewardSample]], list[RewardInferenceArtifact]]
# Runs once after scoring reaches a terminal state: either deletes the call's
# materializations or transfers their ownership to the debug/output dir.
ArtifactFinalizer = Callable[[list[RewardInferenceArtifact]], None]
ArtifactTransport = Literal["in_memory", "disk"]


def resolve_reward_component_device(
    *,
    resolved_device: str,
    overrides: list[tuple[str, Any]],
) -> str:
    """Apply a component CPU downgrade without weakening GPU ownership."""

    resolved = str(resolved_device or "").strip().lower()
    configured = [
        (key, str(value).strip().lower()) for key, value in overrides if str(value or "").strip()
    ]
    distinct = {value for _, value in configured}
    if len(distinct) > 1:
        raise ValueError(
            f"reward component device overrides disagree: {configured}",
        )
    effective = configured[0][1] if configured else resolved
    key = configured[0][0] if configured else "resolved device"
    if resolved.startswith("cuda"):
        if effective.startswith("cuda") and effective != resolved:
            raise ValueError(
                f"reward {key}={effective!r} conflicts with the distributed-resources "
                f"CUDA device {resolved_device!r}. Remove the component override; "
                "distributed.resources owns the CUDA ordinal.",
            )
        # A component may explicitly downgrade from its GPU ownership ceiling to
        # CPU. It then creates no CuMem owner and can coexist with one GPU reward.
    elif effective.startswith("cuda"):
        raise ValueError(
            f"reward {key}={effective!r} requests CUDA, but distributed resources "
            f"resolved {resolved_device!r}. CPU resources cannot launch a CUDA reward.",
        )
    return effective


def reward_worker_config_with_device(
    worker_config: Mapping[str, Any] | None,
    *,
    device: str,
) -> dict[str, Any]:
    """Copy ``worker_config``, applying the resolved role device as a ceiling."""

    cfg = dict(worker_config or {})
    if device:
        configured_device = str(cfg.get("device", "")).strip()
        cfg["device"] = resolve_reward_component_device(
            resolved_device=str(device),
            overrides=[("worker_config.device", configured_device)],
        )
    return cfg


class RewardCleanupError(RuntimeError):
    """One reward operation accumulated multiple release/teardown failures."""

    def __init__(self, message: str, errors: list[BaseException]) -> None:
        self.errors = tuple(errors)
        details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{message}: {details}")


class RewardFunction:
    """Base class for pure scoring functions and reward composition."""

    # None is fail-closed. A specialized base class declares this capability only
    # when all model-owned CUDA state is built in the tagged runtime pool.
    memory_parking_residual_bytes_limit: ClassVar[int | None] = None
    # Most reward constructors expose the selected device as ``device``;
    # exceptional schemas (for example NSFW's classifier_device) override it.
    device_config_key: ClassVar[str] = "device"
    # Registry preflight reads this before constructing heavyweight models.
    artifact_transport: ClassVar[ArtifactTransport] = "in_memory"

    @classmethod
    def resolve_execution_device(
        cls,
        *,
        device: str,
        kwargs: Mapping[str, Any],
    ) -> str:
        """Return the concrete child device under the resource ownership ceiling."""

        configured_devices: list[tuple[str, Any]] = [
            (cls.device_config_key, kwargs.get(cls.device_config_key)),
        ]
        worker_config = kwargs.get("worker_config")
        if isinstance(worker_config, Mapping):
            configured_devices.append(
                ("worker_config.device", worker_config.get("device")),
            )
        return resolve_reward_component_device(
            resolved_device=device,
            overrides=configured_devices,
        )

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether this scorer yields while scoring runs elsewhere."""

        return False

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether out-of-plan reward accelerator work has been isolated."""

        return True

    async def preflight(self) -> None:
        """Validate dependencies before scoring begins."""

        return None

    async def activate(self) -> None:
        """Pre-warm this reward at a GPU handoff; CPU/remote rewards need none."""

        return None

    async def park_memory(self) -> bool:
        """Release reward-owned accelerator memory and report whether an owner parked."""

        return False

    async def score(self, sample: RewardSample) -> float:
        """Score one generated sample; family-specific scalar extension hook."""

        raise NotImplementedError(f"{type(self).__name__}.score is not implemented")

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Score ordered samples through the batch/runtime hook."""

        return RewardOutput(scores=tuple([await self.score(sample) for sample in samples]))

    async def shutdown(self) -> None:
        """Release reward-owned resources, when applicable."""

        return None


class InferenceRewardFunction(RewardFunction):
    """Reward function backed by one inference runtime and artifact builder."""

    @staticmethod
    def build_inmemory_artifacts(
        samples: list[RewardSample],
    ) -> list[RewardInferenceArtifact]:
        """Build reward artifacts that carry media in-memory (no disk write)."""

        artifacts: list[RewardInferenceArtifact] = []
        for sample in samples:
            metadata = dict(sample.metadata or {})
            artifacts.append(
                RewardInferenceArtifact(
                    artifact_id=f"{sample.sample_id}:in-memory",
                    path="",
                    media=sample.output,
                    prompt=str(sample.prompt),
                    metadata=metadata,
                ),
            )
        return artifacts

    def __init__(
        self,
        *,
        reward_name: str,
        score_key: str,
        scorer: RewardScorer,
        artifact_builder: ArtifactBuilder | None = None,
        artifact_finalizer: ArtifactFinalizer | None = None,
        artifact_retainer: ArtifactFinalizer | None = None,
        debug_dir: str = "",
        request_prefix: str = "reward",
        debug_basename: str = "reward",
    ) -> None:
        normalized_reward_name = str(reward_name).strip()
        if not normalized_reward_name:
            raise ValueError("reward_name must be non-empty")
        normalized_score_key = str(score_key).strip()
        if not normalized_score_key:
            raise ValueError("score_key must be non-empty")
        if not isinstance(scorer, RewardScorer):
            raise TypeError(
                "scorer must implement the complete RewardScorer protocol "
                "(score_batch/shutdown plus the two capability flags)",
            )
        if artifact_builder is None:
            # In-memory media is the default transport; disk rewards inject the
            # artifact-store builder explicitly.
            artifact_builder = InferenceRewardFunction.build_inmemory_artifacts
        if not callable(artifact_builder):
            raise TypeError("artifact_builder must be callable")
        self.reward_name = normalized_reward_name
        self.score_key = normalized_score_key
        self._selected_score_keys = tuple(
            part.strip() for part in normalized_score_key.split("+") if part.strip()
        )
        self.scorer = scorer
        self._artifact_builder = artifact_builder
        self._artifact_finalizer = artifact_finalizer
        self._artifact_retainer = artifact_retainer
        self.debug_dir = str(debug_dir)
        self._request_prefix = request_prefix
        self._debug_basename = debug_basename
        self._inference_started = False

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether this scorer yields while inference runs elsewhere."""

        return bool(self.scorer.scoring_is_nonblocking)

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether out-of-plan reward accelerator work has been isolated."""

        return bool(self.scorer.external_accelerator_isolation_verified)

    async def preflight(self) -> None:
        """Fail before training starts when a remote scoring dependency is broken.

        In-process runtimes have nothing to check here (their model loads
        lazily on the reward device). Remote runtimes expose ``ensure_ready``
        so an unreachable, not-ready, or wrong-model service is reported at
        startup instead of after the first generation batch completes.
        """

        scorer = self.scorer
        if isinstance(scorer, RemoteReadyScorer):
            await scorer.ensure_ready()

    async def activate(self) -> None:
        """Build or wake a parking-capable in-process model at a GPU handoff.

        The inverse of :meth:`park_memory`. Activation marks the model as
        started so a handoff that never scores still releases its GPU lease.
        """

        scorer = self.scorer
        if not isinstance(scorer, MemoryParkingScorer):
            return
        await scorer.activate()
        self._inference_started = True

    async def park_memory(self) -> bool:
        """Park this reward runtime when its model has been activated."""

        scorer = self.scorer
        if not isinstance(scorer, MemoryParkingScorer):
            return False
        if not scorer.requires_memory_parking:
            return False
        if not self._inference_started:
            # This component never activated its model (for example an earlier
            # sibling failed). There is no GPU lease to release.
            return False
        await scorer.park_memory()
        return True

    async def score(self, sample: RewardSample) -> float:
        """Score one sample through the batch inference path."""

        return (await self.score_batch((sample,))).scores[0]

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Score ordered samples through the configured inference runtime."""

        return await self._score_with_scorer(list(samples))

    def _select_score(self, scores: Mapping[str, Any]) -> float:
        missing = [key for key in self._selected_score_keys if key not in scores]
        if missing:
            raise KeyError(
                "reward inference result missing score keys: "
                f"missing={missing}, requested={self.score_key!r}, "
                f"available={sorted(scores)}",
            )
        value = float(sum(float(scores[key]) for key in self._selected_score_keys))
        if not math.isfinite(value):
            raise ValueError(
                f"reward score_key={self.score_key!r} selected non-finite score: {value}",
            )
        return value

    async def shutdown(self) -> None:
        await self.scorer.shutdown()

    def _init_reward_model(
        self,
        *,
        reward_name: str,
        score_key: str,
        model_factory: str,
        worker_config: Mapping[str, Any],
    ) -> None:
        """Initialize a RewardFunction backed by a RewardModel factory."""

        from vrl.rewards.runtime import build_reward_scorer

        InferenceRewardFunction.__init__(
            self,
            reward_name=reward_name,
            score_key=score_key,
            scorer=build_reward_scorer(
                {**dict(worker_config), "model_factory": str(model_factory)},
            ),
        )

    def _init_disk_artifact_reward(
        self,
        *,
        model_factory: str,
        request_prefix: str,
        debug_basename: str,
        artifact_format: str,
        reward_name: str,
        score_key: str,
        media_type: MediaType = "video",
        artifact_dir: str = "outputs/reward_artifacts",
        debug_dir: str = "",
        device: str | None = None,
        sleep_offload: bool = False,
        memory_parking_residual_bytes_limit: int = 0,
        retain_artifacts: bool = False,
        worker_config: Mapping[str, Any] | None = None,
        scorer: RewardScorer | None = None,
    ) -> None:
        """Initialize a reward whose media is materialized to disk before scoring.

        Sibling to :meth:`_init_reward_model`: same idea (configure ``self`` as a
        ``RewardFunction``), but the heavyweight path — media is written to disk
        via ``VideoRewardArtifactStore`` and scored through the selected inference runtime,
        instead of passed in-memory. ``sleep_offload`` releases an in-process
        model's physical GPU pages between scores while its contents stay in
        pinned host RAM (the rollout/trainer own the GPU then), mirroring the
        rollout lease's sleep/wake.
        ``model_factory`` / ``request_prefix`` / ``debug_basename``
        are the only per-reward differences (concrete rewards set their own
        ``reward_name`` / ``score_key`` / ``artifact_format`` defaults before
        delegating); everything else is shared wiring, so no concrete reward
        copies this body. ``scorer`` injects a ready
        ``RewardScorer`` (tests); it wins over the factory-built one.
        Disk files belong to this reward call and are deleted after terminal success or failure; explicit
        ``retain_artifacts`` or an ambiguous remote state transfers them to the
        debug/output owner instead.
        """

        from vrl.rewards.artifacts import VideoRewardArtifactStore
        from vrl.rewards.runtime import build_reward_scorer

        self.media_type = str(media_type)
        self.artifact_store = VideoRewardArtifactStore(
            artifact_dir,
            media_type=self.media_type,
            artifact_format=str(artifact_format),
        )

        if scorer is None:
            worker_cfg = dict(worker_config or {})
            has_model_factory = bool(
                str(worker_cfg.get("model_factory", "")).strip(),
            )
            # Normalize the model-id key ONCE here so the disk loaders
            # (kling/videocon) read only worker_config["reward_model_name"].
            # Precedence: an explicit worker_config.reward_model_name wins;
            # otherwise fold a top-level reward_name that looks like a HF repo
            # (contains "/") — a bare reward_name stays a logical tag, not a
            # model id.
            reward_name_repo = reward_name if "/" in reward_name else ""
            reward_model_name = str(
                worker_cfg.get("reward_model_name") or reward_name_repo or "",
            ).strip()
            model_path = str(worker_cfg.get("model_path", "")).strip()
            # YAML names the public model; the loader needs the private factory.
            if not has_model_factory:
                # A missing injected inference runtime is the in-process path:
                # HTTP components inject their ready client in MultiReward before they
                # reach this constructor. Every local disk reward therefore
                # needs its concrete factory even when it is a composite model
                # rather than one Hugging Face repository.
                worker_cfg["model_factory"] = model_factory
            if reward_model_name or model_path:
                if reward_model_name:
                    worker_cfg["reward_model_name"] = reward_model_name
                if not str(worker_cfg.get("reward_model_version", "")).strip():
                    worker_cfg["reward_model_version"] = reward_model_name or model_path
            # Resource resolution is the device source of truth. A nested model
            # override would split lifecycle ownership from real CUDA execution,
            # so reject it even when this helper is called outside MultiReward.
            if device is not None:
                configured_device = str(worker_cfg.get("device", "")).strip()
                worker_cfg["device"] = resolve_reward_component_device(
                    resolved_device=str(device),
                    overrides=[("worker_config.device", configured_device)],
                )
            if sleep_offload:
                worker_cfg["sleep_offload"] = True
                worker_cfg["memory_parking_residual_bytes_limit"] = int(
                    memory_parking_residual_bytes_limit,
                )
            scorer = build_reward_scorer(worker_cfg)

        InferenceRewardFunction.__init__(
            self,
            reward_name=str(reward_name),
            score_key=str(score_key),
            scorer=scorer,
            artifact_builder=self.artifact_store.materialize,
            artifact_finalizer=(
                self.artifact_store.retain if retain_artifacts else self.artifact_store.release
            ),
            artifact_retainer=self.artifact_store.retain,
            debug_dir=debug_dir,
            request_prefix=request_prefix,
            debug_basename=debug_basename,
        )

    async def _score_with_scorer(
        self,
        samples: list[RewardSample],
    ) -> RewardOutput:
        if not samples:
            return RewardOutput(scores=())

        scorer = self.scorer
        artifact_builder = self._artifact_builder

        total_started = time.perf_counter()
        materialize_started = time.perf_counter()
        artifacts = artifact_builder(samples)
        materialization_ms = (time.perf_counter() - materialize_started) * 1000.0
        operation_error: BaseException | None = None
        output: RewardOutput | None = None
        request_id: str | None = None
        try:
            if len(artifacts) != len(samples):
                raise ValueError(
                    "reward artifact builder returned wrong number of artifacts: "
                    f"artifacts={len(artifacts)}, samples={len(samples)}",
                )
            request_id = f"{self._request_prefix}-{uuid.uuid4().hex}"
            request = RewardInferenceRequest(
                request_id=request_id,
                artifacts=tuple(artifacts),
            )
            inference_started = time.perf_counter()
            # Contract enforcement lives at this seam, not inside each runtime,
            # so every runtime (including injected fakes) gets the same result
            # identity guard and request-order re-sort.
            self._inference_started = True
            raw_results = await scorer.score_batch(request)
            results = validate_reward_results(request, raw_results)
            inference_total_ms = (time.perf_counter() - inference_started) * 1000.0
            total_latency_ms = (time.perf_counter() - total_started) * 1000.0
            self._write_debug(
                request,
                results,
                artifact_materialization_ms=materialization_ms,
                inference_total_ms=inference_total_ms,
                total_reward_latency_ms=total_latency_ms,
            )
            output = RewardOutput(
                scores=tuple(self._select_score(result.scores) for result in results),
                timing_ms={
                    "latency_ms": total_latency_ms,
                    "queue_wait_ms": _max_result_timing(results, "queue_wait_ms"),
                    "inference_ms": _sum_result_timing(
                        results,
                        "inference_ms",
                        default=inference_total_ms,
                    ),
                    "artifact_materialization_ms": materialization_ms,
                    "artifact_validation_ms": _max_result_timing(
                        results,
                        "service_artifact_validation_ms",
                    ),
                    "service_inference_wall_ms": _max_result_timing(
                        results,
                        "service_inference_wall_ms",
                    ),
                    "transport_roundtrip_ms": _max_result_timing(
                        results,
                        "http_roundtrip_ms",
                    ),
                },
            )
        except BaseException as error:
            operation_error = error

        cleanup_error: BaseException | None = None
        artifact_finalizer = self._artifact_finalizer
        retain_for_remote = operation_error is not None and (
            isinstance(operation_error, asyncio.CancelledError)
            or (
                isinstance(operation_error, ArtifactRetainingError)
                and operation_error.retain_reward_artifacts
            )
        )
        if retain_for_remote:
            artifact_finalizer = self._artifact_retainer
            logger.warning(
                "reward inference did not confirm terminal state; retaining %d "
                "artifact(s) for request_id=%s",
                len(artifacts),
                request_id,
            )
        if artifact_finalizer is not None:
            try:
                artifact_finalizer(artifacts)
            except BaseException as error:
                cleanup_error = error
        if operation_error is not None and cleanup_error is not None:
            raise RewardCleanupError(
                "reward operation and artifact cleanup both failed",
                [operation_error, cleanup_error],
            )
        if operation_error is not None:
            raise operation_error
        if cleanup_error is not None:
            raise cleanup_error
        assert output is not None
        return output

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
            "reward_name": self.reward_name,
            "score_key": self.score_key,
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


class CumemRewardFunction(InferenceRewardFunction):
    """Reward whose model allocations support verified tagged-pool parking."""

    memory_parking_residual_bytes_limit: ClassVar[int] = CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT


class DiskArtifactRewardFunction(CumemRewardFunction):
    """Registry-visible base for rewards that require disk materialization."""

    artifact_transport: ClassVar[ArtifactTransport] = "disk"


def _max_result_timing(
    results: list[RewardInferenceResult],
    key: str,
) -> float:
    return max(
        (float(result.timing_ms[key]) for result in results if key in result.timing_ms),
        default=0.0,
    )


def _sum_result_timing(
    results: list[RewardInferenceResult],
    key: str,
    *,
    default: float,
) -> float:
    values = [float(result.timing_ms[key]) for result in results if key in result.timing_ms]
    return sum(values) if values else float(default)


__all__ = [
    "ArtifactBuilder",
    "ArtifactFinalizer",
    "ArtifactTransport",
    "CumemRewardFunction",
    "DiskArtifactRewardFunction",
    "InferenceRewardFunction",
    "RewardCleanupError",
    "RewardFunction",
    "resolve_reward_component_device",
]
