"""RewardFunction base class for async generated-sample scoring.

The plugin layer between ``RewardFunctionRuntime`` above (runtime.py) and the
``RewardScorer`` transports below (inference.py): concrete rewards in
vrl/rewards/functions subclass one of the bases here. The class ladder encodes
capabilities the registry and runtime probe, not taxonomy:
``InferenceRewardFunction`` owns the materialize / score / validate /
release-or-retain seam so every transport (including injected fakes) passes
the same result-identity guard; ``CumemRewardFunction`` declares that all
model CUDA state is built in the tagged pool, enabling verified memory
parking; ``ModelRewardFunction`` resolves a model factory and scoring transport
while forwarding media in memory by default. Explicit archives are separate
experiment outputs; file-only models receive scorer-local temporary files.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.artifacts import (
    DiskRewardArtifactStore,
    InMemoryRewardArtifactStore,
    MediaType,
    RewardArtifactStore,
)
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.protocols import (
    ArtifactRetainingError,
    MemoryParkingScorer,
    RemoteReadyScorer,
    RewardScorer,
)
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.logging import init_logger

if TYPE_CHECKING:
    from vrl.rewards.ray import RayRewardPlacement

logger = init_logger(__name__)


class RewardCleanupError(RuntimeError):
    """One reward operation accumulated multiple release/teardown failures."""

    def __init__(self, message: str, errors: list[BaseException]) -> None:
        self.errors = tuple(errors)
        details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{message}: {details}")


class RewardFunction:
    """Base class for pure scoring functions and reward composition."""

    # Most reward constructors expose the selected device as ``device``;
    # exceptional schemas (for example NSFW's classifier_device) override it.
    device_config_key: ClassVar[str] = "device"

    @classmethod
    def resolve_execution_device(
        cls,
        *,
        device: str,
        kwargs: Mapping[str, Any],
    ) -> str:
        """Return the concrete execution device under the resource ceiling.

        The device policy in one place: distributed resources own placement,
        so a component override (``cls.device_config_key`` or
        ``worker_config.device``) may only downgrade from the resolved GPU to
        CPU — never pick a different CUDA ordinal, and never request CUDA on
        a CPU resource plan. A CPU downgrade creates no CuMem owner and can
        coexist with the one GPU reward. Reward classes with a fixed device
        override this hook (OCR forces CPU for its CPU-only engine).
        """

        candidates: list[tuple[str, Any]] = [
            (cls.device_config_key, kwargs.get(cls.device_config_key)),
        ]
        worker_config = kwargs.get("worker_config")
        if isinstance(worker_config, Mapping):
            candidates.append(
                ("worker_config.device", worker_config.get("device")),
            )
        resolved = str(device or "").strip().lower()
        configured = [
            (key, str(value).strip().lower())
            for key, value in candidates
            if str(value or "").strip()
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
                    f"reward {key}={effective!r} conflicts with the "
                    f"distributed-resources CUDA device {device!r}. Remove the "
                    "component override; distributed.resources owns the CUDA "
                    "ordinal.",
                )
            # A component may explicitly downgrade from its GPU ownership
            # ceiling to CPU. It then creates no CuMem owner and can coexist
            # with one GPU reward.
        elif effective.startswith("cuda"):
            raise ValueError(
                f"reward {key}={effective!r} requests CUDA, but distributed "
                f"resources resolved {device!r}. CPU resources cannot launch "
                "a CUDA reward.",
            )
        return effective

    @classmethod
    def worker_config_with_device(
        cls,
        worker_config: Mapping[str, Any] | None,
        *,
        device: str,
    ) -> dict[str, Any]:
        """Copy ``worker_config``, stamping the resolved device as a ceiling.

        The same policy as :meth:`resolve_execution_device`, applied at the
        config-bag boundary: constructors that feed a ``worker_config`` dict
        to a model must stamp the device through the ceiling check, not
        assign it directly.
        """

        cfg = dict(worker_config or {})
        if device:
            cfg["device"] = cls.resolve_execution_device(
                device=str(device),
                kwargs={"worker_config": cfg},
            )
        return cfg

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
    """Reward function backed by one scorer transport and one artifact store."""

    def __init__(
        self,
        *,
        reward_name: str,
        score_key: str,
        scorer: RewardScorer,
        artifact_store: RewardArtifactStore | None = None,
        archive_store: RewardArtifactStore | None = None,
        retain_artifacts: bool = False,
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
        if artifact_store is None:
            # Media stays in memory unless a caller explicitly injects a store.
            artifact_store = InMemoryRewardArtifactStore()
        self.reward_name = normalized_reward_name
        self.score_key = normalized_score_key
        selected_score_keys = tuple(part.strip() for part in normalized_score_key.split("+"))
        if not all(selected_score_keys):
            raise ValueError(
                f"score_key {normalized_score_key!r} contains an empty component; "
                'use "a+b" to sum score keys',
            )
        self._selected_score_keys = selected_score_keys
        self.scorer = scorer
        self.artifact_store = artifact_store
        self._archive_store = archive_store
        self._retain_artifacts = bool(retain_artifacts)
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
        """Materialize, score, validate, and finalize one ordered sample batch."""

        samples = list(samples)
        if not samples:
            return RewardOutput(scores=())

        if self._archive_store is not None:
            # Explicit experiment output is independent of scorer transport;
            # never send these driver-local paths to a remote worker.
            archived = self._archive_store.materialize(samples)
            self._archive_store.retain(archived)
        scorer = self.scorer
        total_started = time.perf_counter()
        materialize_started = time.perf_counter()
        artifacts = self.artifact_store.materialize(samples)
        materialization_ms = (time.perf_counter() - materialize_started) * 1000.0
        operation_error: BaseException | None = None
        output: RewardOutput | None = None
        request_id: str | None = None
        try:
            if len(artifacts) != len(samples):
                raise ValueError(
                    "reward artifact store returned wrong number of artifacts: "
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
            results = request.validate_and_order_results(raw_results)
            inference_total_ms = (time.perf_counter() - inference_started) * 1000.0
            total_latency_ms = (time.perf_counter() - total_started) * 1000.0
            self._write_debug(
                request,
                results,
                artifact_materialization_ms=materialization_ms,
                inference_total_ms=inference_total_ms,
                total_reward_latency_ms=total_latency_ms,
            )
            score_keys = set(results[0].scores)
            if any(set(result.scores) != score_keys for result in results):
                raise ValueError("reward model returned inconsistent score axes within a batch")
            output = RewardOutput(
                scores=tuple(self._select_score(result.scores) for result in results),
                components={
                    key: tuple(result.scores[key] for result in results)
                    for key in results[0].scores
                },
                timing_ms=_result_timing_summary(
                    results,
                    materialization_ms=materialization_ms,
                    inference_total_ms=inference_total_ms,
                    total_latency_ms=total_latency_ms,
                ),
            )
        except BaseException as error:
            operation_error = error

        cleanup_error: BaseException | None = None
        retain_for_remote = operation_error is not None and (
            isinstance(operation_error, asyncio.CancelledError)
            or (
                isinstance(operation_error, ArtifactRetainingError)
                and operation_error.retain_reward_artifacts
            )
        )
        if retain_for_remote:
            logger.warning(
                "reward inference did not confirm terminal state; retaining %d "
                "artifact(s) for request_id=%s",
                len(artifacts),
                request_id,
            )
        finalize = (
            self.artifact_store.retain
            if retain_for_remote or self._retain_artifacts
            else self.artifact_store.release
        )
        try:
            finalize(artifacts)
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
    """Reward whose model allocations are built in the tagged CuMem pool."""


class ModelRewardFunction(CumemRewardFunction):
    """Model-backed reward with shared factory and inference-transport wiring.

    Subclasses declare the model factory, score defaults, and debug identity;
    the registry uses this boundary to admit remote model inference. Samples
    remain in memory (including boxed references) by default. File-only models
    declare their input format on the model itself, and the scoring process
    owns any temporary encoding and cleanup.

    ``scorer`` injects a ready transport in place of factory construction.
    ``artifact_store`` is an explicit alternative input store, while
    ``archive_dir`` independently retains experiment output; an archive is
    never selected merely because scoring is remote. ``sleep_offload`` lets
    the scorer release its model's GPU memory between scoring phases.
    """

    # Concrete model rewards differ in these declarations, so each subclass is
    # a declaration block instead of a forwarding __init__. model_factory /
    # name / default_score_key have no class default on purpose — a subclass
    # that forgets them fails loudly at construction (AttributeError).
    model_factory: ClassVar[str]
    # The registry component name. It is also the request-id prefix, the debug
    # sidecar basename and the default ``reward_name``; a reward whose
    # ``reward_name`` is a hub model id sets ``default_reward_name`` instead.
    name: ClassVar[str]
    default_reward_name: ClassVar[str | None] = None
    default_score_key: ClassVar[str]
    # Explicit archive encoding only; scorer input format belongs to its model.
    default_artifact_format: ClassVar[str] = "mp4"
    default_media_type: ClassVar[MediaType] = "video"
    # In-process transport only: build the model in the constructor so config
    # validation fails at construction and tests can reach ``self._model``
    # (e.g. to inject a fake engine). Skipped under sleep_offload, whose pooled
    # model must be factory-built by the runtime itself.
    eager_model: ClassVar[bool] = False
    # The score keys this reward can select; empty means any non-empty key.
    score_keys: ClassVar[tuple[str, ...]] = ()
    # Model knobs arrive as top-level keywords (``reward.kwargs.<name>.dtype``)
    # and travel to the model in ``worker_config``. A reward whose model
    # vocabulary is nested under ``worker_config:`` in YAML sets this so a stray
    # top-level keyword is a typo, not a silent model knob.
    worker_config_only: ClassVar[bool] = False

    def __init__(
        self,
        *,
        reward_name: str | None = None,
        score_key: str | None = None,
        archive_dir: str = "",
        debug_dir: str = "",
        device: str | None = None,
        sleep_offload: bool = False,
        worker_config: Mapping[str, Any] | None = None,
        scorer: RewardScorer | None = None,
        artifact_store: RewardArtifactStore | None = None,
        inference: RewardInferenceConfig | None = None,
        ray_placement: RayRewardPlacement | None = None,
        **model_kwargs: Any,
    ) -> None:
        """Transport keywords are the explicit parameters; every other keyword
        is a model knob and joins ``worker_config``. Subclasses therefore
        declare only what they score (class attributes) and keep an
        ``__init__`` only for a check the base cannot express."""

        # Deferred: runtime.py imports this module (cycle guard).
        from vrl.rewards.runtime import build_reward_scorer

        removed = {
            "artifact_dir",
            "artifact_format",
            "media_type",
            "retain_artifacts",
        } & model_kwargs.keys()
        if removed:
            raise TypeError(
                f"reward transport options {sorted(removed)} were removed; "
                "scorers own their input format; use archive_dir only for explicit experiment output"
            )
        if model_kwargs and self.worker_config_only:
            raise TypeError(
                f"{type(self).__name__} takes model knobs under worker_config; "
                f"unexpected keyword(s) {sorted(model_kwargs)}",
            )
        worker_config = {**dict(worker_config or {}), **model_kwargs}
        model_factory = self.model_factory
        if reward_name is None:
            reward_name = self.default_reward_name or self.name
        score_key = self.default_score_key if score_key is None else score_key
        if self.score_keys and score_key not in self.score_keys:
            raise ValueError(
                f"{reward_name} score_key must be one of {list(self.score_keys)}, "
                f"got {score_key!r}",
            )
        in_process = scorer is None and (inference is None or inference.kind == "in_process")
        archive_store = (
            DiskRewardArtifactStore(
                archive_dir,
                media_type=self.default_media_type,
                artifact_format=self.default_artifact_format,
            )
            if archive_dir
            else None
        )

        if scorer is None:
            worker_cfg = dict(worker_config or {})
            has_model_factory = bool(
                str(worker_cfg.get("model_factory", "")).strip(),
            )
            # Normalize the model-id key ONCE here so the model loaders
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
                # A missing injected scorer is the in-process path: HTTP
                # components inject their ready client in MultiReward before
                # they reach this constructor. Every model-backed reward
                # therefore needs its concrete factory even when it is a
                # composite model rather than one Hugging Face repository.
                worker_cfg["model_factory"] = model_factory
            if reward_model_name or model_path:
                if reward_model_name:
                    worker_cfg["reward_model_name"] = reward_model_name
                if not str(worker_cfg.get("reward_model_version", "")).strip():
                    worker_cfg["reward_model_version"] = reward_model_name or model_path
            # Resource resolution is the device source of truth. A nested model
            # override would split lifecycle ownership from real CUDA execution,
            # so apply the shared ceiling even when this constructor runs
            # outside MultiReward.
            if device is not None:
                worker_cfg = self.worker_config_with_device(
                    worker_cfg,
                    device=str(device),
                )
            if sleep_offload:
                worker_cfg["sleep_offload"] = True
            # ``inference`` selects in-process (None/default) or a placed Ray
            # actor that receives this same worker_cfg. External
            # HTTP components never reach here: the registry injects their
            # ready client as ``scorer``.
            if in_process and self.eager_model and not worker_cfg.get("sleep_offload"):
                from vrl.rewards.runtime import InProcessRewardScorer
                from vrl.utils.config import import_from_path

                self._model = import_from_path(str(worker_cfg["model_factory"]))(worker_cfg)
                scorer = InProcessRewardScorer(model=self._model)
            else:
                scorer = build_reward_scorer(
                    worker_cfg,
                    inference=inference,
                    ray_placement=ray_placement,
                )

        super().__init__(
            reward_name=str(reward_name),
            score_key=str(score_key),
            scorer=scorer,
            artifact_store=artifact_store,
            archive_store=archive_store,
            debug_dir=debug_dir,
            request_prefix=self.name,
            debug_basename=self.name,
        )


def _result_timing_summary(
    results: list[RewardInferenceResult],
    *,
    materialization_ms: float,
    inference_total_ms: float,
    total_latency_ms: float,
) -> dict[str, float]:
    """Aggregate per-result transport timings into one reward-call summary.

    Stage-parallel phases report their slowest member (max over results);
    per-artifact inference cost is additive (sum), falling back to this
    call's measured inference wall time when a transport reports no
    per-result values.
    """

    def stage_max(key: str) -> float:
        return max(
            (float(result.timing_ms[key]) for result in results if key in result.timing_ms),
            default=0.0,
        )

    inference_values = [
        float(result.timing_ms["inference_ms"])
        for result in results
        if "inference_ms" in result.timing_ms
    ]
    return {
        "latency_ms": total_latency_ms,
        "queue_wait_ms": stage_max("queue_wait_ms"),
        "inference_ms": (sum(inference_values) if inference_values else float(inference_total_ms)),
        "artifact_materialization_ms": materialization_ms,
        "artifact_validation_ms": stage_max("service_artifact_validation_ms"),
        "service_inference_wall_ms": stage_max("service_inference_wall_ms"),
        "transport_roundtrip_ms": stage_max("http_roundtrip_ms"),
    }


__all__ = [
    "CumemRewardFunction",
    "InferenceRewardFunction",
    "ModelRewardFunction",
    "RewardCleanupError",
    "RewardFunction",
]
