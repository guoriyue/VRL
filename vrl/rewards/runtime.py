"""Reward runtime implementations and inference transport wiring.

The implementation layer behind the vrl/rewards/protocols contract:

- ``RewardFunctionRuntime``: the concrete ``RewardRuntime`` the collector
  drives — the reward dual of ``RayGenerationRuntime``. It owns the lifecycle
  FSM, the operation lock, and the score deadline so every wrapped
  ``RewardFunction`` gets identical admission and teardown semantics.
- ``InProcessRewardScorer``: the local ``RewardScorer`` transport (twin of the
  remote ``HttpRewardScorer``), also used inside remote scoring processes.
  It resolves media references, owns temporary files required by a model,
  and owns CUDA memory parking.
- ``build_reward_scorer``: the single transport-selection point, keyed by the
  typed inference deployment config.

This module imports the CUDA parking utilities; the contract modules
(types/protocols/inference) deliberately stay free of them.
"""

from __future__ import annotations

import asyncio
import random
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from vrl.config.reward_inference import (
    RewardInferenceConfig,
)
from vrl.models.parking import CumemBroken, CumemPool, ParkingSession
from vrl.rewards.base import RewardFunction
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.launch_contract import RewardRuntimeLaunchContract
from vrl.rewards.protocols import RewardScorer
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.config import import_from_path
from vrl.utils.cuda_memory import release_cuda_memory_for_parking
from vrl.utils.deadline import OperationDeadline
from vrl.utils.lifecycle import RuntimeLifecycle, RuntimePhase

if TYPE_CHECKING:
    from vrl.rewards.ray import RayRewardPlacement

# Matches the HTTP reward client's default request timeout so the two
# transports share one notion of "scoring took too long".
_DEFAULT_SCORE_TIMEOUT_S = 1800.0


class RewardFunctionRuntime:
    """Expose a reward function through the collector-facing runtime contract."""

    def __init__(self, reward_function: RewardFunction) -> None:
        self._reward_function = reward_function
        self._score_timeout_s = _DEFAULT_SCORE_TIMEOUT_S
        self._operation_lock = asyncio.Lock()
        # Same terminal FSM as the generation runtime: RUNNING accepts work,
        # SHUTTING_DOWN closes admission (retryable teardown), TERMINATED is
        # published only after a successful shutdown.
        self.lifecycle = RuntimeLifecycle(owner="reward runtime")

    async def preflight(self) -> None:
        """Validate the wrapped reward function before scoring begins."""

        async with self._operation_lock:
            self.lifecycle.require_running("preflight")
            await self._reward_function.preflight()

    async def activate(self) -> None:
        """Pre-warm reward model ownership at a GPU handoff."""

        async with self._operation_lock:
            self.lifecycle.require_running("activate")
            await self._reward_function.activate()

    async def score(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Score ordered samples while serializing function and memory ownership."""

        async with self._operation_lock:
            self.lifecycle.require_running("score")
            normalized = tuple(samples)
            if not normalized:
                raise ValueError("reward runtime requires at least one sample")
            if not all(isinstance(sample, RewardSample) for sample in normalized):
                raise TypeError("reward runtime samples must contain RewardSample values")
            sample_ids = [sample.sample_id for sample in normalized]
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("reward runtime sample_id values must be unique")
            # The deadline preempts every awaitable transport (HTTP
            # service round-trips, overlapped async scoring) and raises
            # the shared terminal OperationTimeout. A component that
            # blocks the event loop in synchronous model code cannot be
            # preempted in-process — like a launched CUDA kernel, its
            # bound is process supervision, not this timer.
            deadline = OperationDeadline(
                "reward.score",
                self._score_timeout_s,
                context=f"samples={len(normalized)}",
            )
            try:
                output = await asyncio.wait_for(
                    self._reward_function.score_batch(normalized),
                    timeout=deadline.remaining_s(),
                )
            except TimeoutError as cause:
                raise deadline.timeout_error() from cause
            if not isinstance(output, RewardOutput):
                raise TypeError("reward function score_batch() must return RewardOutput")
            return output

    async def park_memory(self) -> None:
        """Park the function's device memory under the operation lock.

        The collector calls this wherever the placement plan makes the reward
        yield its GPU: after scoring, and again at the phase-final handoff,
        where it retries a park that failed after scoring. Whether anything is
        held is the function's own parking state, so a park after a failed
        activation or a second park at the handoff reaches nothing.
        """

        async with self._operation_lock:
            self.lifecycle.require_running("park_memory")
            await self._reward_function.park_memory()

    async def shutdown(self) -> None:
        """Release the wrapped function exactly once after successful teardown.

        A failed teardown leaves the lifecycle SHUTTING_DOWN (admission stays
        closed, root cause retained) and the next call retries the release.
        """

        async with self._operation_lock:
            if self.lifecycle.phase is RuntimePhase.TERMINATED:
                return
            self.lifecycle.begin_shutdown()
            try:
                await self._reward_function.shutdown()
            except BaseException as error:
                self.lifecycle.fail(error)
                raise
            self.lifecycle.finish_shutdown()


class RewardParking:
    """The reward runtime's parking owner: one ``ParkingSession`` on the configured device.

    ``worker_config.sleep_offload`` opts a heavyweight model into the same
    sleep/wake semantics the rollout lease uses: the model holds no GPU memory
    between scores (the rollout/trainer own the card then) and comes back only
    for scoring; the caller's step ordering already guarantees the card is
    free at that point. Small in-memory rewards (CLIP-class) leave the knob
    off and stay resident, and only get the terminal device-cache release.

    Default parking is ``cumem``: the model is built inside a CuMem pool on
    the configured device, so process-wide ``sleep`` releases physical pages
    while preserving this reward's contents in pinned host RAM. vLLM's
    CuMemAllocator is required and missing it fails loud rather than
    degrading to a CPU round trip, which measured 6.2x slower per
    wake/score/sleep cycle and only appears to park.
    ``distributed.resources.reward.parking.mechanism: reload`` selects the
    session's ``reload`` mechanism instead: the model is dropped at each handoff and rebuilt from its
    factory at the next activation, paying checkpoint reload latency instead
    of pinned host RAM. CuMem tags do not isolate sleep operations, so
    the shared-topology preflight permits at most one configured GPU reward
    component per process; zero-weight observation-only scorers still execute
    and therefore count.
    """

    def __init__(self, launch: RewardRuntimeLaunchContract) -> None:
        self._launch = launch
        self._session: ParkingSession | None = None
        self._broken_reason: str | None = None

    def _require_healthy(self) -> None:
        if self._broken_reason is not None:
            raise CumemBroken(
                "reward CuMem state is unrecoverable; terminate the owning process "
                f"before retrying: {self._broken_reason}"
            )

    @property
    def required(self) -> bool:
        """Whether topology/config requires this runtime to release GPU pages."""

        return self._launch.sleep_offload

    @property
    def pool(self) -> CumemPool | None:
        return None if self._session is None else self._session.pool

    def build(self, factory: Callable[[], Any]) -> Any:
        """Build the model where its parking mechanism can reach it."""

        self._require_healthy()
        if not self.required:
            return factory()
        # Build inside the pool so every CUDA allocation the factory makes
        # (from_pretrained, .to(device), buffers) is tagged and sleep/wake can
        # release/restore it wholesale. PyTorch's pool scope captures the
        # current device, not .to() targets.
        session = ParkingSession("reward runtime", required=True, device=self._launch.device)
        try:
            model = session.build(factory, mechanism=self._launch.parking_mechanism)
        except CumemBroken as error:
            # A failed pool close is just as terminal as a partial sleep/wake.
            # Retain the session and avoid touching the corrupted allocator.
            self._session = session
            self._broken_reason = str(error)
            raise
        except BaseException as load_error:
            # Commit neither half of a failed model/pool build. The session
            # already closed its pool; drop traceback-held locals of the
            # half-built candidate, then release the device cache so a retry
            # starts clean.
            traceback.clear_frames(load_error.__traceback__)
            try:
                self._release_device_cache()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "reward model preparation and parking cleanup both failed: "
                    f"load={load_error!r}; cleanup={cleanup_error!r}",
                ) from cleanup_error
            raise
        self._session = session
        return model

    def restore(self) -> Any:
        """Wake or rebuild a parked model and return it; None without a session."""

        self._require_healthy()
        if self._session is None:
            return None
        try:
            return self._session.restore()
        except CumemBroken as error:
            self._broken_reason = str(error)
            raise

    def park(self) -> None:
        """Park the session; partial CuMem failures require process termination."""

        self._require_healthy()
        if not self.required:
            raise RuntimeError(
                "reward runtime was not configured for complete memory parking",
            )
        if self._session is None:
            raise RuntimeError(
                "reward runtime cannot park memory before its model is built",
            )
        try:
            self._session.park()
        except CumemBroken as error:
            self._broken_reason = str(error)
            raise
        self._release_device_cache()

    @contextmanager
    def release_scope(self) -> Iterator[None]:
        """Let the owner drop its model, then release every device and host page."""

        self._require_healthy()
        session = self._session
        if session is None:
            yield
        else:
            try:
                with session.release_scope():
                    yield
            except CumemBroken as error:
                self._broken_reason = str(error)
                raise
        # Dedicated CUDA rewards use torch's caching allocator rather than a
        # CuMem pool. Dropping the model alone leaves those physical pages
        # reserved in this long-lived driver process, so terminal cleanup must
        # release the configured device cache for every runtime.
        self._release_device_cache()
        self._session = None

    def _release_device_cache(self) -> None:
        release_cuda_memory_for_parking(self._launch.device)


class InProcessRewardScorer:
    """``RewardScorer`` that runs a ``RewardModel`` in this process.

    Its GPU residency between scores belongs to :class:`RewardParking`.
    """

    def __init__(
        self,
        worker_config: Mapping[str, Any] | None = None,
        *,
        model: Any | None = None,
        media_temp_dir: str | None = None,
    ) -> None:
        # Typed runtime contract; the verbatim bag still feeds the factory.
        self._launch = RewardRuntimeLaunchContract.from_mapping(worker_config)
        self._parking = RewardParking(self._launch)
        if model is not None and self._launch.sleep_offload:
            raise ValueError(
                "sleep_offload requires the runtime to build the model itself "
                "(worker_config.model_factory) so it can capture the pre-load "
                "GPU baseline and validate a complete parking backend.",
            )
        self._model = model
        self._media_temp_dir = media_temp_dir

    @property
    def requires_memory_parking(self) -> bool:
        return self._parking.required

    async def activate(self) -> None:
        """Build or wake the model so scoring starts resident.

        The handoff twin of :meth:`park_memory`; ``score_batch`` performs the
        same ensure/wake lazily, so activation only moves the load latency out
        of the measured scoring phase.
        """

        self._resident_model()

    async def park_memory(self) -> None:
        """Park pages; retry only while the parking owner remains healthy.

        The scorer drops its reference first: the parking session holds the
        model, and ``reload`` can free it only if nothing else does.
        """

        self._model = None
        self._parking.park()

    def _resident_model(self) -> Any:
        """The model ready to score: built on first use, woken or rebuilt after a park."""

        if self._model is None:
            self._model = self._parking.restore()
        if self._model is None:
            if not self._launch.model_factory:
                raise ValueError(
                    "InProcessRewardScorer requires worker_config.model_factory "
                    "(import path to a RewardModel factory) or an explicit model",
                )
            self._model = self._parking.build(self._build_model)
        return self._model

    def _build_model(self) -> Any:
        """Construct and prepare the model, leaving every RNG stream untouched.

        Synchronous cold construction, and each ``reload`` rebuild, must not
        change the trainer's resume streams: every RNG the build may touch is
        restored after.
        """

        import numpy as np
        import torch

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        try:
            with torch.random.fork_rng(devices=devices):
                factory = import_from_path(self._launch.model_factory)
                model = factory(self._launch.component_config)
                prepare = getattr(model, "prepare_for_inference", None)
                if callable(prepare):
                    prepare()
                return model
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        if not request.artifacts:
            return []
        model = self._resident_model()
        # CuMem's model-building scope is one-shot. Execution uses the normal
        # allocator; park_memory's physical baseline gate rejects any lazy
        # long-lived CUDA allocation that survives scoring.
        return self._score_artifacts(model, request)

    def _score_artifacts(
        self,
        model: Any,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        """Resolve media and adapt file-only models inside the scoring process."""

        from vrl.rewards.models.base import FileRewardModel

        request = request.resolve_media()
        if not isinstance(model, FileRewardModel):
            return self._infer(model, request)
        if all(artifact.path for artifact in request.artifacts):
            return self._infer(model, request)
        with TemporaryDirectory(prefix="vrl-reward-", dir=self._media_temp_dir) as directory:
            artifacts = []
            for index, artifact in enumerate(request.artifacts):
                if artifact.path:
                    artifacts.append(artifact)
                    continue
                media = artifact.as_media()
                if model.input_artifact_format == "mp4":
                    from vrl.utils.media import write_mp4

                    path = Path(directory) / f"{index}.mp4"
                    fps = artifact.metadata.get("video_fps", artifact.metadata.get("fps", 8.0))
                    write_mp4(media, path, fps=8.0 if fps is None else float(fps))
                elif model.input_artifact_format == "tensor":
                    import torch

                    path = Path(directory) / f"{index}.pt"
                    torch.save(media.detach().cpu(), path)
                else:
                    raise ValueError(
                        f"unsupported scorer input format {model.input_artifact_format!r}"
                    )
                artifacts.append(replace(artifact, path=str(path), media=None))
            return self._infer(model, replace(request, artifacts=tuple(artifacts)))

    def _infer(self, model: Any, request: RewardInferenceRequest) -> list[RewardInferenceResult]:
        """Run the reward model over the request's artifacts and build results.

        A model may expose ``score_results(artifacts)`` for structured evidence,
        or the existing ``score_batch(artifacts)`` hook; otherwise ``__call__``
        is looped. Results may be numeric mappings or RewardInferenceResult.
        ``model`` stays Any because the factory is unvalidated plugin input.
        """

        reward_model_version = self._launch.reward_model_version

        def build_result(
            artifact: RewardInferenceArtifact,
            raw_scores: Mapping[str, Any] | RewardInferenceResult,
            inference_ms: float,
        ) -> RewardInferenceResult:
            diagnostics, timings, version = {}, {}, reward_model_version or None
            if isinstance(raw_scores, RewardInferenceResult):
                if raw_scores.artifact_id != artifact.artifact_id:
                    raise ValueError("reward model result artifact identity mismatch")
                if version and raw_scores.reward_model_version not in {None, version}:
                    raise ValueError("reward model result revision differs from its deployment")
                diagnostics, timings = raw_scores.diagnostics, raw_scores.timing_ms
                version = version or raw_scores.reward_model_version
                raw_scores = raw_scores.scores
            return RewardInferenceResult(
                artifact_id=artifact.artifact_id,
                scores=raw_scores,
                reward_model_version=version,
                timing_ms={**timings, "inference_ms": inference_ms},
                diagnostics=diagnostics,
            )

        batch_score = getattr(model, "score_results", None)
        if not callable(batch_score):
            batch_score = getattr(model, "score_batch", None)
        if callable(batch_score):
            started = time.perf_counter()
            score_maps = list(batch_score(request.artifacts))
            if len(score_maps) != len(request.artifacts):
                raise ValueError(
                    "RewardModel.score_batch returned wrong number of score maps: "
                    f"got {len(score_maps)}, expected {len(request.artifacts)}",
                )
            per_artifact_ms = (time.perf_counter() - started) * 1000.0 / len(score_maps)
            return [
                build_result(artifact, raw_scores, inference_ms=per_artifact_ms)
                for artifact, raw_scores in zip(request.artifacts, score_maps, strict=True)
            ]

        results: list[RewardInferenceResult] = []
        for artifact in request.artifacts:
            started = time.perf_counter()
            raw_scores = model(artifact)
            inference_ms = (time.perf_counter() - started) * 1000.0
            results.append(build_result(artifact, raw_scores, inference_ms))
        return results

    async def shutdown(self) -> None:
        with self._parking.release_scope():
            self._model = None


def build_reward_scorer(
    worker_config: Mapping[str, Any] | None = None,
    *,
    inference: Mapping[str, Any] | RewardInferenceConfig | None = None,
    ray_placement: RayRewardPlacement | None = None,
) -> RewardScorer:
    """Build the runtime selected by the typed inference deployment config.

    Internal actors receive the same model configuration as direct evaluation;
    external HTTP services own their model configuration and receive only media.
    """

    if worker_config is not None and not isinstance(worker_config, Mapping):
        raise TypeError("reward worker_config must be a mapping or None")
    cfg = {} if worker_config is None else dict(worker_config)
    if "service_url" in cfg:
        raise ValueError(
            "worker_config.service_url was removed; configure "
            "reward.inference.<component>.kind=http and its endpoint",
        )
    if inference is None:
        # Direct construction (evaluation scripts, tests): no deployment was
        # resolved from YAML, so the model scores in this process.
        return InProcessRewardScorer(cfg)
    deployment = RewardInferenceConfig.from_mapping(
        inference,
        context="reward inference",
    )
    if deployment.kind == "in_process":
        return InProcessRewardScorer(cfg)
    if deployment.kind == "ray":
        from vrl.rewards.ray import RayRewardScorer

        return RayRewardScorer(
            cfg,
            placement=ray_placement,
            timeout_s=deployment.timeout_s,
        )
    if cfg:
        raise ValueError(
            "HTTP reward runtime cannot consume local worker_config; model and "
            "device configuration belong to the external service",
        )
    from vrl.rewards.service.client import HttpRewardScorer

    return HttpRewardScorer.from_section(deployment)


__all__ = [
    "InProcessRewardScorer",
    "RewardFunctionRuntime",
    "RewardParking",
    "build_reward_scorer",
]
