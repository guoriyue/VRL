"""RewardFunction base class for async generated-sample scoring.

The plugin layer between ``RewardFunctionRuntime`` above (runtime.py) and the
``RewardScorer`` transports below (inference.py): concrete rewards in
vrl/rewards/functions subclass ``ModelRewardFunction``. It resolves a model
factory and scoring transport, forwards media in memory, and owns the
materialize / score / validate seam, so every transport (including injected
fakes) passes the same result-identity guard; its scorer owns memory parking.
``MultiReward`` combines components on the plain ``RewardFunction`` base.
File-only models receive scorer-local temporary files.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.protocols import (
    MemoryParkingScorer,
    RemoteReadyScorer,
    RewardScorer,
)
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.logging import init_logger
from vrl.utils.stats import absorb, recording, timed

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

    async def preflight(self) -> None:
        """Validate dependencies before scoring begins."""

        return None

    async def activate(self) -> None:
        """Pre-warm this reward at a GPU handoff; CPU/remote rewards need none."""

        return None

    async def park_memory(self) -> None:
        """Release reward-owned accelerator memory; a no-op when none is held."""

        return None

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Score ordered samples; every concrete reward implements this."""

        raise NotImplementedError(f"{type(self).__name__}.score_batch is not implemented")

    async def shutdown(self) -> None:
        """Release reward-owned resources, when applicable."""

        return None


class ModelRewardFunction(RewardFunction):
    """Model-backed reward with shared factory and inference-transport wiring.

    Subclasses declare the model factory, score defaults, and debug identity;
    the registry uses this boundary to admit remote model inference. Samples
    remain in memory (including boxed references) by default. File-only models
    declare their input format on the model itself, and the scoring process
    owns any temporary encoding and cleanup.

    ``scorer`` injects a ready transport in place of factory construction.
    ``worker_config``'s
    ``sleep_offload``, which the registry sets for a GPU component of a
    shared reward, lets the scorer release its model's GPU memory between
    scoring phases.
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
    # In-process transport only: build the model in the constructor so config
    # validation fails at construction. Skipped under sleep_offload, whose pooled
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
        debug_dir: str = "",
        device: str | None = None,
        worker_config: Mapping[str, Any] | None = None,
        scorer: RewardScorer | None = None,
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
            "archive_dir",
            "artifact_dir",
            "artifact_format",
            "media_type",
            "retain_artifacts",
        } & model_kwargs.keys()
        if removed:
            raise TypeError(
                f"reward transport options {sorted(removed)} were removed; "
                "scorers own their input format and rewards write no media archive"
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
            # ``inference`` selects in-process (None/default) or a placed Ray
            # actor that receives this same worker_cfg. External
            # HTTP components never reach here: the registry injects their
            # ready client as ``scorer``.
            if in_process and self.eager_model and not worker_cfg.get("sleep_offload"):
                from vrl.rewards.runtime import InProcessRewardScorer
                from vrl.utils.config import import_from_path

                model = import_from_path(str(worker_cfg["model_factory"]))(worker_cfg)
                scorer = InProcessRewardScorer(model=model)
            else:
                scorer = build_reward_scorer(
                    worker_cfg,
                    inference=inference,
                    ray_placement=ray_placement,
                )

        normalized_reward_name = str(reward_name).strip()
        if not normalized_reward_name:
            raise ValueError("reward_name must be non-empty")
        normalized_score_key = str(score_key).strip()
        if not normalized_score_key:
            raise ValueError("score_key must be non-empty")
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
        self.debug_dir = str(debug_dir)
        # Prefixes request ids and names the debug sidecar files.
        self._name = self.name
        # Parked whenever the scorer holds no device memory: before activate
        # or score builds its model, and again after each park.
        self._parked = True

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
        self._parked = False

    async def park_memory(self) -> None:
        """Park the scorer's model; a no-op while it holds no device memory."""

        scorer = self.scorer
        if not isinstance(scorer, MemoryParkingScorer) or not scorer.requires_memory_parking:
            return
        if self._parked:
            # Never activated (an earlier sibling failed), or already parked:
            # there is no GPU lease to release.
            return
        await scorer.park_memory()
        self._parked = True

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Materialize, score and validate one ordered sample batch."""

        samples = list(samples)
        if not samples:
            return RewardOutput(scores=())

        scorer = self.scorer
        # This call's timings, folded into the caller's accumulator once below.
        with recording() as call, timed("reward.latency_s"):
            with timed("reward.artifact_materialization_s"):
                # Media rides the request in memory (tensors or boxed
                # references); the driver writes no files.
                artifacts = [
                    RewardInferenceArtifact(
                        artifact_id=f"{sample.sample_id}:in-memory",
                        sample_id=sample.sample_id,
                        path="",
                        media=sample.output,
                        prompt=str(sample.prompt),
                        metadata=dict(sample.metadata or {}),
                    )
                    for sample in samples
                ]
            request_id = f"{self._name}-{uuid.uuid4().hex}"
            request = RewardInferenceRequest(
                request_id=request_id,
                artifacts=tuple(artifacts),
            )
            with timed("reward.inference_wall_s"):
                # Contract enforcement lives at this seam, not inside each
                # runtime, so every runtime (including injected fakes) gets the
                # same result identity guard and request-order re-sort.
                self._parked = False
                raw_results = await scorer.score_batch(request)
                results = request.validate_and_order_results(raw_results)
        self._write_debug(request, results, call.phase_seconds)
        score_keys = set(results[0].scores)
        if any(set(result.scores) != score_keys for result in results):
            raise ValueError("reward model returned inconsistent score axes within a batch")

        # Each result's own timings (the record a transport stamps) fold into
        # this call: per-artifact inference cost is additive, falling back to
        # the measured scorer wall time when a transport reports none, and
        # stage-parallel phases report their slowest member.
        inference_ms = [
            float(result.timing_ms["inference_ms"])
            for result in results
            if "inference_ms" in result.timing_ms
        ]
        call.add_phase(
            "reward.inference_s",
            sum(inference_ms) / 1000.0
            if inference_ms
            else call.phase_seconds["reward.inference_wall_s"],
        )
        for key, phase in (
            ("service_artifact_validation_ms", "reward.artifact_validation_s"),
            ("service_inference_wall_ms", "reward.service_inference_wall_s"),
            ("http_roundtrip_ms", "reward.transport_roundtrip_s"),
        ):
            values = [
                float(result.timing_ms[key]) for result in results if key in result.timing_ms
            ]
            if values:
                call.add_phase(phase, max(values) / 1000.0)
        absorb(call)
        return RewardOutput(
            scores=tuple(self._select_score(result.scores) for result in results),
            components={
                key: tuple(result.scores[key] for result in results) for key in results[0].scores
            },
        )

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
        phase_seconds: Mapping[str, float],
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
            "artifact_materialization_ms": phase_seconds["reward.artifact_materialization_s"]
            * 1000.0,
            "inference_total_ms": phase_seconds["reward.inference_wall_s"] * 1000.0,
            "total_reward_latency_ms": phase_seconds["reward.latency_s"] * 1000.0,
        }
        requests_file = debug_path / f"{self._name}_requests.jsonl"
        results_file = debug_path / f"{self._name}_results.jsonl"
        with requests_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request_row, sort_keys=True) + "\n")
        with results_file.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


__all__ = [
    "ModelRewardFunction",
    "RewardCleanupError",
    "RewardFunction",
]
