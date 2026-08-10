"""RewardFunction base class for async generated-sample scoring."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

from vrl.rewards.inference import (
    MediaType,
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
    RewardInferenceRuntime,
    RewardMemoryParkingCapability,
    RewardMemoryParkingRuntime,
    RewardMemoryReleaseProof,
)
from vrl.rewards.types import RewardSample
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


@dataclass(frozen=True, slots=True)
class RewardBatchReport:
    """One scoring call's scores plus the observations produced by that call.

    Observations travel with the return value instead of mutable ``last_*``
    attributes: continuous rollout can score future groups while the trainer
    consumes an older group, so cached instance state would attach metrics to
    whichever call finished last instead of to the batch being trained.
    """

    scores: list[float]
    components: dict[str, list[float]] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)


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
    memory_parking: ClassVar[RewardMemoryParkingCapability | None] = None
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

    async def park_memory(self) -> tuple[RewardMemoryReleaseProof, ...]:
        """Release reward-owned accelerator memory, when applicable."""

        return ()

    async def score(self, sample: RewardSample) -> float:
        """Score a single generated sample."""

        raise NotImplementedError(f"{type(self).__name__}.score is not implemented")

    async def score_batch(self, samples: list[RewardSample]) -> list[float]:
        """Score a batch of generated samples sequentially by default."""

        return [await self.score(sample) for sample in samples]

    async def score_batch_report(self, samples: list[RewardSample]) -> RewardBatchReport:
        """Score a batch and return the observations from this exact call."""

        return RewardBatchReport(scores=await self.score_batch(samples))

    async def shutdown(self) -> None:
        """Release reward-owned resources, when applicable."""

        return None


class InferenceRewardFunction(RewardFunction):
    """Reward function backed by one inference runtime and artifact builder."""

    @staticmethod
    def build_inmemory_artifacts(
        samples: list[RewardSample],
        *,
        media_type: MediaType = "image",
    ) -> list[RewardInferenceArtifact]:
        """Build reward artifacts that carry media in-memory (no disk write)."""

        artifacts: list[RewardInferenceArtifact] = []
        for sample in samples:
            metadata = dict(sample.metadata or {})
            artifacts.append(
                RewardInferenceArtifact(
                    artifact_id=(f"{sample.source_request_id}:{sample.sample_id}:in-memory"),
                    path="",
                    media_type=media_type,
                    media=sample.output,
                    prompt=str(sample.prompt),
                    source_request_id=sample.source_request_id,
                    sample_id=sample.sample_id,
                    group_id=sample.group_id,
                    trajectory_id=sample.trajectory_id,
                    policy_version=sample.policy_version,
                    metadata=metadata,
                ),
            )
        return artifacts

    def __init__(
        self,
        *,
        reward_name: str,
        score_key: str,
        inference_runtime: RewardInferenceRuntime,
        artifact_builder: ArtifactBuilder,
        artifact_finalizer: ArtifactFinalizer | None = None,
        artifact_retainer: ArtifactFinalizer | None = None,
        request_metadata: Mapping[str, Any] | None = None,
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
        if inference_runtime is None:
            raise TypeError("inference_runtime must be a RewardInferenceRuntime")
        for method_name in ("score_batch", "shutdown"):
            if not callable(getattr(inference_runtime, method_name, None)):
                raise TypeError(
                    "inference_runtime must provide async score_batch() and shutdown()",
                )
        if not callable(artifact_builder):
            raise TypeError("artifact_builder must be callable")
        self.reward_name = normalized_reward_name
        self.score_key = normalized_score_key
        self.inference_runtime = inference_runtime
        self._artifact_builder = artifact_builder
        self._artifact_finalizer = artifact_finalizer
        self._artifact_retainer = artifact_retainer
        self._request_metadata = dict(request_metadata or {})
        self.debug_dir = str(debug_dir)
        self._request_prefix = request_prefix
        self._debug_basename = debug_basename
        self._last_reward_request_id: str | None = None

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether this scorer yields while inference runs elsewhere."""

        return bool(getattr(self.inference_runtime, "scoring_is_nonblocking", False))

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether out-of-plan reward accelerator work has been isolated."""

        return bool(
            getattr(
                self.inference_runtime,
                "external_accelerator_isolation_verified",
                False,
            ),
        )

    async def preflight(self) -> None:
        """Fail before training starts when a remote scoring dependency is broken.

        In-process runtimes have nothing to check here (their model loads
        lazily on the reward device). Remote runtimes expose ``ensure_ready``
        so an unreachable, not-ready, or wrong-model service is reported at
        startup instead of after the first generation batch completes.
        """

        ensure_ready = getattr(self.inference_runtime, "ensure_ready", None)
        if callable(ensure_ready):
            await ensure_ready()

    async def park_memory(self) -> tuple[RewardMemoryReleaseProof, ...]:
        """Park this reward runtime, retrying the runtime's current request."""

        inference_runtime = self.inference_runtime
        if not isinstance(inference_runtime, RewardMemoryParkingRuntime):
            return ()
        if not inference_runtime.requires_memory_parking:
            return ()
        request_id = self._last_reward_request_id
        if request_id is None:
            # This component never activated its model (for example an earlier
            # sibling failed). There is no GPU lease to release.
            return ()
        proof = await inference_runtime.park_memory()
        proof.validate(request_id=request_id)
        return (proof,)

    async def score(self, sample: RewardSample) -> float:
        """Score a single generated sample."""

        return (await self.score_batch([sample]))[0]

    async def score_batch(self, samples: list[RewardSample]) -> list[float]:
        """Score a batch through the configured inference runtime."""

        return (await self._score_with_inference_runtime(samples)).scores

    async def score_batch_report(self, samples: list[RewardSample]) -> RewardBatchReport:
        """Score a batch and return the observations from this exact call."""

        return await self._score_with_inference_runtime(samples)

    async def shutdown(self) -> None:
        await self.inference_runtime.shutdown()

    def _init_reward_model(
        self,
        *,
        reward_name: str,
        score_key: str,
        model_factory: str,
        worker_config: Mapping[str, Any],
    ) -> None:
        """Initialize a RewardFunction backed by a RewardModel factory."""

        from vrl.rewards.runtime import build_reward_inference_runtime

        InferenceRewardFunction.__init__(
            self,
            reward_name=reward_name,
            score_key=score_key,
            inference_runtime=build_reward_inference_runtime(
                {**dict(worker_config), "model_factory": str(model_factory)},
            ),
            artifact_builder=lambda samples: InferenceRewardFunction.build_inmemory_artifacts(
                samples,
                media_type="image",
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
        inference_runtime: RewardInferenceRuntime | None = None,
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
        copies this body. ``inference_runtime`` injects a ready
        ``RewardInferenceRuntime`` (tests); it wins over the factory-built one.
        Disk files belong to this reward call and are deleted after terminal success or failure; explicit
        ``retain_artifacts`` or an ambiguous remote state transfers them to the
        debug/output owner instead.
        """

        from vrl.rewards.artifacts import VideoRewardArtifactStore
        from vrl.rewards.runtime import build_reward_inference_runtime

        self.media_type = str(media_type)
        self.artifact_store = VideoRewardArtifactStore(
            artifact_dir,
            media_type=self.media_type,
            artifact_format=str(artifact_format),
        )

        if inference_runtime is None:
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
            inference_runtime = build_reward_inference_runtime(worker_cfg)

        InferenceRewardFunction.__init__(
            self,
            reward_name=str(reward_name),
            score_key=str(score_key),
            inference_runtime=inference_runtime,
            artifact_builder=self.artifact_store.materialize,
            artifact_finalizer=(
                self.artifact_store.retain if retain_artifacts else self.artifact_store.release
            ),
            artifact_retainer=self.artifact_store.retain,
            request_metadata={"media_type": self.media_type},
            debug_dir=debug_dir,
            request_prefix=request_prefix,
            debug_basename=debug_basename,
        )

    async def _score_with_inference_runtime(
        self,
        samples: list[RewardSample],
    ) -> RewardBatchReport:
        if not samples:
            return RewardBatchReport(scores=[])

        inference_runtime = self.inference_runtime
        artifact_builder = self._artifact_builder

        total_started = time.perf_counter()
        materialize_started = time.perf_counter()
        artifacts = artifact_builder(samples)
        materialization_ms = (time.perf_counter() - materialize_started) * 1000.0
        operation_error: BaseException | None = None
        report: RewardBatchReport | None = None
        try:
            if len(artifacts) != len(samples):
                raise ValueError(
                    "reward artifact builder returned wrong number of artifacts: "
                    f"artifacts={len(artifacts)}, samples={len(samples)}",
                )
            correlated_artifacts: list[RewardInferenceArtifact] = []
            for artifact, sample in zip(artifacts, samples, strict=True):
                expected_lineage = {
                    "source_request_id": sample.source_request_id,
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "trajectory_id": sample.trajectory_id,
                    "policy_version": sample.policy_version,
                }
                for field_name, expected in expected_lineage.items():
                    actual = getattr(artifact, field_name)
                    if actual is not None and actual != expected:
                        raise ValueError(
                            "reward artifact lineage mismatch: "
                            f"artifact_id={artifact.artifact_id!r}, "
                            f"field={field_name}, expected={expected!r}, "
                            f"actual={actual!r}",
                        )
                correlated_artifacts.append(replace(artifact, **expected_lineage))
            artifacts = correlated_artifacts

            policy_versions = {artifact.policy_version for artifact in artifacts}
            policy_version = next(iter(policy_versions)) if len(policy_versions) == 1 else None
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
            self._last_reward_request_id = request.request_id
            inference_started = time.perf_counter()
            # Contract enforcement lives at this seam, not inside each runtime,
            # so every runtime (including injected fakes) gets the same result
            # identity guard and request-order re-sort.
            from vrl.rewards.inference import validate_reward_results

            score_error: BaseException | None = None
            raw_results: list[RewardInferenceResult] | None = None
            try:
                raw_results = await inference_runtime.score_batch(request)
            except BaseException as error:
                score_error = error
            park_error: BaseException | None = None
            try:
                await self.park_memory()
            except BaseException as error:
                park_error = error
            if score_error is not None and park_error is not None:
                raise RewardCleanupError(
                    "reward scoring and memory parking both failed",
                    [score_error, park_error],
                )
            if score_error is not None:
                raise score_error
            if park_error is not None:
                raise park_error
            assert raw_results is not None
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
            report = RewardBatchReport(
                scores=[float(result.selected_score) for result in results],
                timing_ms={
                    "latency_ms": total_latency_ms,
                    "queue_wait_ms": max(
                        (
                            float(result.queue_wait_ms)
                            for result in results
                            if result.queue_wait_ms is not None
                        ),
                        default=0.0,
                    ),
                    "inference_ms": (
                        sum(
                            float(result.inference_ms)
                            for result in results
                            if result.inference_ms is not None
                        )
                        if results
                        else inference_total_ms
                    ),
                    "artifact_materialization_ms": materialization_ms,
                    "artifact_validation_ms": _max_result_metadata_timing(
                        results,
                        "service_artifact_validation_ms",
                    ),
                    "service_inference_wall_ms": _max_result_metadata_timing(
                        results,
                        "service_inference_wall_ms",
                    ),
                    "transport_roundtrip_ms": _max_result_metadata_timing(
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
            or bool(getattr(operation_error, "retain_reward_artifacts", False))
        )
        if retain_for_remote:
            artifact_finalizer = self._artifact_retainer
            logger.warning(
                "reward inference did not confirm terminal state; retaining %d "
                "artifact(s) for request_id=%s",
                len(artifacts),
                self._last_reward_request_id,
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
        assert report is not None
        return report

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
            "source_request_ids": [artifact.source_request_id for artifact in request.artifacts],
            "sample_ids": [artifact.sample_id for artifact in request.artifacts],
            "group_ids": [artifact.group_id for artifact in request.artifacts],
            "trajectory_ids": [artifact.trajectory_id for artifact in request.artifacts],
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


class CumemRewardFunction(InferenceRewardFunction):
    """Reward whose model allocations support verified tagged-pool parking."""

    memory_parking: ClassVar[RewardMemoryParkingCapability] = RewardMemoryParkingCapability(
        residual_bytes_limit=CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT,
    )


class DiskArtifactRewardFunction(CumemRewardFunction):
    """Registry-visible base for rewards that require disk materialization."""

    artifact_transport: ClassVar[ArtifactTransport] = "disk"


def _max_result_metadata_timing(
    results: list[RewardInferenceResult],
    key: str,
) -> float:
    return max(
        (float(result.metadata[key]) for result in results if key in result.metadata),
        default=0.0,
    )


__all__ = [
    "ArtifactBuilder",
    "ArtifactFinalizer",
    "ArtifactTransport",
    "CumemRewardFunction",
    "DiskArtifactRewardFunction",
    "InferenceRewardFunction",
    "RewardBatchReport",
    "RewardCleanupError",
    "RewardFunction",
    "resolve_reward_component_device",
]
