"""Generation worker core independent of the Ray actor wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from vrl.generation.capabilities import (
    FamilyCapability,
    family_capability_from_value,
)
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import GenerationChunkExecutor
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.interfaces import require_runtime_model
from vrl.utils.config import import_from_path
from vrl.utils.cuda_memory import CumemPool, release_cuda_memory
from vrl.utils.logging import init_logger
from vrl.utils.profiling import TorchProfilerConfig

logger = init_logger(__name__)

class GenerationWorkerCore:
    """Own one generation executor and execute plan-aware chunks."""

    def __init__(
        self,
        worker_id: str,
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
        *,
        metadata_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.launch_contract = self._normalize_launch_contract(launch_contract)
        self.family = self.launch_contract.family
        self.executor: GenerationChunkExecutor | None = None
        self._policy_version: int | None = self.launch_contract.policy_version
        # Flipped on once a model that supports versioned trainable-state slots
        # receives its first weight install; from then on execute_chunk activates
        # the slot for each request's stamped version instead of comparing against
        # one global version (which is what makes a non-draining sync safe).
        self._uses_versioned_slots = False
        self._profiler_config = self._profiler_config_from_contract(self.launch_contract)
        self._profiler_output_dir = str(
            self.launch_contract.extra.get("profiler_output_dir", "outputs/"),
        )
        self._profiler_step = 0
        self.capability = self._capability_from_contract(self.launch_contract)
        self._metadata_provider = metadata_provider or self._fallback_metadata
        # Set to a CumemPool when load_policy pools the model for sleep-offload
        # (extra["sleep_offload"]); sleep()/wake() then release/restore GPU physical
        # pages through it instead of the naive to(cpu)/to(gpu) round trip. vLLM's
        # own "weights" tag = the offloaded (not discarded) slice of a sleeping
        # engine, keeping semantics identical to verl-omni's level-1.
        self._cumem: CumemPool | None = None

    def load_policy(self) -> None:
        """Build the family executor from the serialized launch contract."""

        if self.executor is not None:
            return
        from vrl.utils.cuda_memory import cap_cuda_memory_fraction
        from vrl.utils.memory import log_host_memory

        # Cap the allocator before the model loads so a colocated worker leaves the
        # trainer its share of the shared GPU (no-op when unset / dedicated GPU).
        cap_cuda_memory_fraction(self.launch_contract.extra.get("gpu_memory_fraction"))
        log_host_memory(f"generation_worker:{self.worker_id}:before_load_policy", log=logger)
        self.executor = self._build_executor_maybe_pooled()
        self.capability = self._merge_loaded_capability(self.executor)
        log_host_memory(f"generation_worker:{self.worker_id}:after_load_policy", log=logger)

    def _build_executor_maybe_pooled(self) -> GenerationChunkExecutor:
        """Build the executor, pooling the model in CuMemAllocator when requested.

        A sleep-offload worker (``extra["sleep_offload"]``, set by the lease for the
        colocated-trainer timeshare) allocates the whole model inside vLLM's
        CuMemAllocator pool, so ``sleep``/``wake`` release and restore the GPU
        *physical* pages via CUDA virtual memory — virtual addresses stay mapped, so
        wake needs no cudaMalloc and incurs no fragmentation (the naive
        ``to(cpu)``+``empty_cache``+``to(gpu)`` path pays both). Falls back to a
        plain build when offload is not requested or cumem is unavailable (CPU box,
        vLLM not installed), so non-colocated and AR workers are unaffected.
        """

        if not self.launch_contract.extra.get("sleep_offload"):
            return self._build_executor()
        pool = CumemPool.try_create(tag="weights")
        if pool is None:
            return self._build_executor()
        with pool.building():
            executor = self._build_executor()
        self._cumem = pool
        return executor

    def release_policy(self) -> None:
        """Drop loaded model state so the worker releases CUDA memory before exit."""

        self.executor = None
        release_cuda_memory(gc_collect=True, ipc_collect=True)

    def sleep(self) -> None:
        """Offload the loaded model to host RAM, freeing the GPU without discarding it.

        Level-1 offload-and-restore (cf. vLLM ``sleep_level=1``): unlike
        ``release_policy`` (level-2 discard → ``wake``/``load_policy`` cold-reloads
        the frozen VAE / text-encoders from disk), this keeps the executor and its
        loaded model alive, parking *all* weights on CPU so a colocated trainer
        reclaims the GPU for its step. ``wake`` then restores them from host RAM
        without re-reading the checkpoint. ``nn.Module.to`` moves only registered
        submodules (the transformer), so the diffusers pipeline's unregistered
        frozen components ride the same ``move_frozen_components`` hook the
        in-process driver-offload path uses. No-op when nothing is loaded.
        """

        if self.executor is None:
            return
        if self._cumem is not None:
            # cumem owns the whole pooled model (transformer + frozen components);
            # one call offloads every tagged allocation to host RAM and releases the
            # physical pages, no per-module .to() or empty_cache needed.
            self._cumem.sleep()
            return
        model = getattr(self.executor, "model", None)
        if model is None:
            return
        # Remember where to restore to: capture before the CPU move so a custom
        # ``device`` property that does not track ``nn.Module.to`` still reports the
        # GPU the model lived on.
        self._sleep_device = self._executor_device(self.executor)
        move = getattr(model, "to", None)
        if callable(move):
            move("cpu")
        move_frozen = getattr(model, "move_frozen_components", None)
        if callable(move_frozen):
            move_frozen("cpu")
        release_cuda_memory(gc_collect=True, ipc_collect=True)

    def wake(self) -> None:
        """Restore a slept model from host RAM onto its GPU (no disk reload).

        The counterpart to ``sleep``. If the executor was evicted instead of slept
        (e.g. a hard release happened in between), rebuild it via ``load_policy`` so
        a wake is always safe to call.
        """

        model = getattr(self.executor, "model", None)
        if model is None:
            self.load_policy()
            return
        if self._cumem is not None:
            # Restore the pooled model's physical pages from host RAM at the same
            # virtual addresses — the tensors stay valid cuda tensors throughout.
            self._cumem.wake()
            return
        device = getattr(self, "_sleep_device", None) or self._executor_device(self.executor)
        move = getattr(model, "to", None)
        if callable(move):
            move(device)
        move_frozen = getattr(model, "move_frozen_components", None)
        if callable(move_frozen):
            move_frozen(device)

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        """Update generation weights, then record the active policy version.

        When the model supports versioned trainable-state slots, install the new
        version as a retained slot WITHOUT overwriting the slots older in-flight
        requests still depend on (the non-draining-sync path); ``execute_chunk``
        then activates the right slot per request. Otherwise keep the single
        in-place overwrite (the draining-barrier path).
        """

        self.load_policy()
        policy_obj = getattr(self.executor, "model", None)
        if bool(getattr(policy_obj, "supports_versioned_trainable_state", False)):
            model = require_runtime_model(
                policy_obj,
                owner=f"{type(self.executor).__name__}.model",
            )
            model.install_trainable_state(int(policy_version), state_ref)
            self._uses_versioned_slots = True
        elif state_ref is not None:
            model = require_runtime_model(
                policy_obj,
                owner=f"{type(self.executor).__name__}.model",
            )
            model.load_trainable_state(state_ref)
        # The single scalar now means "current submit version": current_policy_version()
        # is what the producer reads to stamp NEW requests, so it must track the
        # latest installed version even in slot mode. Per-chunk results take their
        # version from request.policy_version, not this field.
        self._policy_version = int(policy_version)

    def current_policy_version(self) -> int | None:
        return self._policy_version

    def supports_versioned_trainable_state(self) -> bool:
        """Whether the loaded model can retain versioned trainable-state slots.

        Drives the runtime's non-draining-weight-sync capability. Requires the
        model to be built, so callers query it after at least one weight sync.
        """

        self.load_policy()
        model = getattr(self.executor, "model", None)
        return bool(getattr(model, "supports_versioned_trainable_state", False))

    def worker_metadata(self, *, runtime_debug: bool = False) -> dict[str, Any]:
        try:
            metadata = dict(self._metadata_provider())
        except Exception:
            metadata = self._fallback_metadata()
        metadata.setdefault("worker_id", self.worker_id)
        metadata.setdefault("node_ip", "unknown")
        metadata.setdefault("gpu_ids", [])
        metadata["policy_version"] = self._policy_version
        if runtime_debug:
            metadata["runtime_debug"] = True
            metadata["executor_type"] = type(self.executor).__name__ if self.executor else None
            policy = getattr(self.executor, "model", None) if self.executor is not None else None
            if policy is not None:
                from vrl.utils.model_diagnostics import (
                    parameter_state_summary,
                    trainable_state_digest,
                )

                metadata["trainable_state"] = trainable_state_digest(policy)
                metadata["parameter_state"] = parameter_state_summary(policy)
                metadata["policy_type"] = type(policy).__name__
        return metadata

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        self.load_policy()
        request = envelope.request
        chunk = envelope.chunk
        runtime_debug = bool(request.metadata.get("_runtime_debug"))
        expected_version = request.policy_version
        model = getattr(self.executor, "model", None)
        if self._uses_versioned_slots and expected_version is not None:
            # Slot mode: serve the request from the slot for ITS stamped version,
            # not the worker's latest. A missing slot means the request outlived
            # the retention window — a typed stale-slot result, not a mismatch.
            if not model.has_trainable_state(expected_version):
                return ChunkExecutionResult(
                    request_id=request.request_id,
                    worker_id=self.worker_id,
                    chunk=chunk,
                    output=None,
                    metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                    plan_id=envelope.plan_id,
                    stage_id=envelope.stage_id,
                    stage_name=envelope.stage_name,
                    profiler_label=envelope.profiler_label,
                    chunk_key=envelope.chunk_key,
                    policy_version=expected_version,
                    error=(
                        "trainable-state slot evicted for "
                        f"policy_version={expected_version}"
                    ),
                    stale_slot=True,
                )
        elif expected_version is not None and self._policy_version != expected_version:
            return ChunkExecutionResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                plan_id=envelope.plan_id,
                stage_id=envelope.stage_id,
                stage_name=envelope.stage_name,
                profiler_label=envelope.profiler_label,
                chunk_key=envelope.chunk_key,
                policy_version=self._policy_version,
                error=(
                    "policy_version mismatch: "
                    f"expected={expected_version}, actual={self._policy_version}"
                ),
            )
        # In slot mode the result must carry the REQUEST's version (the executor
        # asserts result.policy_version == request.policy_version), which is how an
        # old in-flight request keeps passing after the worker installs a newer slot.
        result_version = expected_version if expected_version is not None else self._policy_version
        try:
            assert self.executor is not None
            if self._uses_versioned_slots and expected_version is not None:
                model.activate_trainable_state(expected_version)
            output = self._profile_forward_chunk(envelope)
            return ChunkExecutionResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=self._to_cpu(output),
                metrics=self._chunk_metrics(
                    envelope,
                    runtime_debug=runtime_debug,
                    plan_aware_chunk=True,
                    chunk_output=output,
                ),
                plan_id=envelope.plan_id,
                stage_id=envelope.stage_id,
                stage_name=envelope.stage_name,
                profiler_label=envelope.profiler_label,
                chunk_key=envelope.chunk_key,
                policy_version=result_version,
            )
        except Exception as exc:
            return ChunkExecutionResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                plan_id=envelope.plan_id,
                stage_id=envelope.stage_id,
                stage_name=envelope.stage_name,
                profiler_label=envelope.profiler_label,
                chunk_key=envelope.chunk_key,
                policy_version=result_version,
                error=str(exc),
            )

    def execute_request_pipelined(
        self,
        request: Any,
        engine_plan: Any,
        sample_rows: Any,
    ) -> Any:
        """Run ALL of a request's chunks through the executor's software pipeline
        (``forward_plan_pipelined``) on THIS worker, so chunk N+1's denoise overlaps
        chunk N's GPU->CPU copy + host packing — hiding the per-chunk worker
        boundary that per-chunk dispatch leaves serial. Single-worker only (all the
        request's chunks must be here). Returns the gathered ``GenerationOutput``.

        Version safety mirrors ``execute_chunk`` but at the REQUEST level (every
        chunk shares ``request.policy_version``): slot mode serves the request from
        its stamped version's slot (stale slot -> ``StaleSlotDiscard`` so the
        producer counts a graceful discard, never an off-policy train); non-slot
        mode rejects a version mismatch. The version is checked + activated ONCE
        here, not per chunk.
        """

        from vrl.generation.execution.types import StaleSlotDiscard

        self.load_policy()
        expected_version = request.policy_version
        model = getattr(self.executor, "model", None)
        if self._uses_versioned_slots and expected_version is not None:
            if not model.has_trainable_state(expected_version):
                raise StaleSlotDiscard(
                    f"trainable-state slot evicted for policy_version={expected_version}",
                )
            model.activate_trainable_state(expected_version)
        elif expected_version is not None and self._policy_version != expected_version:
            raise RuntimeError(
                "policy_version mismatch: "
                f"expected={expected_version}, actual={self._policy_version}",
            )
        assert self.executor is not None
        forward_plan_pipelined = getattr(self.executor, "forward_plan_pipelined", None)
        if not callable(forward_plan_pipelined):
            raise TypeError(
                f"{type(self.executor).__name__} must implement forward_plan_pipelined(...) "
                "for per-request pipelined execution",
            )
        return forward_plan_pipelined(request, sample_rows, engine_plan)

    def _profile_forward_chunk(
        self,
        envelope: ChunkExecutionEnvelope,
    ) -> Any:
        from vrl.utils.profiling import capture_torch_trace, profile_range

        assert self.executor is not None
        request = envelope.request
        chunk = envelope.chunk
        step = self._profiler_step
        self._profiler_step += 1
        worker_name = (
            f"{self.worker_id}_{self.family}_{self.launch_contract.task}_"
            f"policy{self._policy_version}_chunk{chunk.prompt_index}_{chunk.sample_start}"
        )
        try:
            device = self._executor_device(self.executor)
            event_name = envelope.profiler_label or "engine.forward_chunk"
            forward_chunk_plan = getattr(self.executor, "forward_chunk_plan", None)
            if not callable(forward_chunk_plan):
                raise TypeError(
                    f"{type(self.executor).__name__} must implement "
                    "forward_chunk_plan(...) for distributed chunk execution",
                )
            if envelope.execution_stage is None:
                raise RuntimeError("chunk execution requires an EnginePlan execution stage")
            with capture_torch_trace(
                self._profiler_config,
                output_dir=self._profiler_output_dir,
                step=step,
                device=device,
                worker_name=worker_name,
                trace_subdir=f"generation/{self.worker_id}",
            ), profile_range(event_name):
                return forward_chunk_plan(
                    request,
                    chunk,
                    envelope.execution_stage,
                )
        except Exception:
            logger.exception("generation chunk execution failed")
            raise

    def _chunk_metrics(
        self,
        envelope: ChunkExecutionEnvelope,
        *,
        runtime_debug: bool,
        plan_aware_chunk: bool | None = None,
        chunk_output: Any | None = None,
    ) -> dict[str, Any]:
        metrics = self.worker_metadata(runtime_debug=runtime_debug)
        metrics.update(
            {
                "plan_id": envelope.plan_id,
                "engine_plan_id": envelope.plan_id,
                "stage_id": envelope.stage_id,
                "stage_name": envelope.stage_name,
                "profiler_label": envelope.profiler_label,
                "chunk_key": envelope.chunk_key,
            },
        )
        if plan_aware_chunk is not None:
            metrics["plan_aware_chunk"] = plan_aware_chunk
        if runtime_debug and chunk_output is not None:
            metrics.update(_chunk_output_debug_metrics(chunk_output))
        return metrics

    @staticmethod
    def _normalize_launch_contract(
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
    ) -> GenerationRuntimeLaunchContract:
        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
        if contract.family is None:
            raise ValueError("GenerationRuntimeLaunchContract.family is required")
        return contract

    def _build_executor(self) -> GenerationChunkExecutor:
        launch_contract = self.launch_contract
        builder_path = launch_contract.runtime_builder
        executor_path = launch_contract.executor_cls
        if builder_path is None or executor_path is None:
            raise ValueError(
                "GenerationRuntimeLaunchContract requires runtime_builder and "
                "executor_cls import paths",
            )

        from vrl.models.interfaces.runtime import RuntimeBuildSpec

        build_runtime_bundle = import_from_path(str(builder_path))
        executor_cls = import_from_path(str(executor_path))
        spec = RuntimeBuildSpec(
            **self._normalize_runtime_build_payload(
                launch_contract.model_build_payload(),
            ),
        )
        bundle = build_runtime_bundle(spec)
        model = require_runtime_model(bundle.model, owner="RuntimeBundle.model")
        # Family- and scheme-agnostic backstop: if precision.rollout asks for a
        # quantized rollout (fp8/fp4/...) but this family's builder forgot to swap,
        # the model would silently run bf16 — fail loudly instead.
        from vrl.models.loader import assert_rollout_quantization_applied

        assert_rollout_quantization_applied(model, spec)
        built = executor_cls(model, **dict(launch_contract.executor_kwargs))
        if getattr(bundle, "runtime_caps", None) is not None:
            built.runtime_caps = dict(bundle.runtime_caps)
        return _require_chunked_executor(built)

    @staticmethod
    def _profiler_config_from_contract(
        launch_contract: GenerationRuntimeLaunchContract,
    ) -> TorchProfilerConfig:
        raw = launch_contract.extra.get("torch_profiler", {})
        if isinstance(raw, Mapping):
            return TorchProfilerConfig(**dict(raw))
        return TorchProfilerConfig()

    @staticmethod
    def _capability_from_contract(
        launch_contract: GenerationRuntimeLaunchContract,
    ) -> FamilyCapability:
        capability = family_capability_from_value(
            launch_contract.extra.get("family_capability"),
        )
        if capability is not None:
            return capability
        raise ValueError(
            "GenerationRuntimeLaunchContract.extra['family_capability'] is required "
            "for distributed generation",
        )

    def _merge_loaded_capability(
        self,
        executor: GenerationChunkExecutor,
    ) -> FamilyCapability:
        runtime_caps = getattr(executor, "runtime_caps", None)
        merged = self.capability.with_runtime_caps(
            runtime_caps if isinstance(runtime_caps, Mapping) else None,
        )
        declared = self._declared_executor_capability(executor)
        if declared is None:
            return merged
        if declared.family != merged.family or declared.task != merged.task:
            raise ValueError(
                "executor capability does not match launch contract: "
                f"{declared.family}/{declared.task} != {merged.family}/{merged.task}",
            )
        if declared.trajectory_kind != merged.trajectory_kind:
            raise ValueError(
                "executor trajectory capability does not match launch contract: "
                f"{declared.trajectory_kind} != {merged.trajectory_kind}",
            )
        return declared.with_runtime_caps(
            runtime_caps if isinstance(runtime_caps, Mapping) else None,
        )

    @staticmethod
    def _declared_executor_capability(
        executor: GenerationChunkExecutor,
    ) -> FamilyCapability | None:
        method = getattr(executor, "capability", None)
        if callable(method):
            return family_capability_from_value(method())
        for attr_name in ("family_capability", "capability_metadata"):
            value = getattr(executor, attr_name, None)
            if value is not None:
                return family_capability_from_value(value)
        return None

    @staticmethod
    def _executor_device(executor: Any) -> Any:
        policy = getattr(executor, "model", None)
        device = getattr(policy, "device", None)
        if device is not None:
            return device
        try:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"

    @classmethod
    def _normalize_runtime_build_payload(
        cls,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(payload)
        device = normalized.get("device")
        if isinstance(device, str):
            import torch

            normalized["device"] = torch.device(device)
        dtype = normalized.get("dtype")
        if isinstance(dtype, str):
            normalized["dtype"] = resolve_torch_dtype(dtype)
        frozen_dtype = normalized.get("frozen_dtype")
        if isinstance(frozen_dtype, str):
            normalized["frozen_dtype"] = resolve_torch_dtype(frozen_dtype)
        return normalized

    @classmethod
    def _to_cpu(cls, value: Any) -> Any:
        """Move a chunk output to CPU with pinned, queued copies.

        Per-tensor ``.cpu()`` synchronizes the stream once per tensor (~376ms
        of cudaStreamSynchronize per chunk measured on the wire-diet profile);
        pinned-buffer ``non_blocking`` copies queue every transfer and the
        device synchronizes once before the payload is handed to Ray.
        """

        # Lazy import: this module stays torch-free at import time, and the
        # trajectory package (the walker's home) pulls torch transitively.
        from vrl.trajectory.device import map_tensor_tree

        pending = {"cuda_copies": False}

        def _pinned_copy(leaf: Any) -> Any:
            tensor = leaf.detach()
            if getattr(tensor, "is_cuda", False):
                import torch

                host = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host.copy_(tensor, non_blocking=True)
                pending["cuda_copies"] = True
                return host
            return tensor.cpu()

        copied = map_tensor_tree(value, _pinned_copy, is_leaf=cls._is_tensor)
        if pending["cuda_copies"]:
            import torch

            torch.cuda.synchronize()
        return copied

    @staticmethod
    def _is_tensor(value: Any) -> bool:
        return hasattr(value, "detach") and hasattr(value, "cpu")

    def _fallback_metadata(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "node_ip": "unknown", "gpu_ids": []}


def _require_chunked_executor(executor: Any) -> GenerationChunkExecutor:
    forward_chunk_plan = getattr(executor, "forward_chunk_plan", None)
    gather_chunks = getattr(executor, "gather_chunks", None)
    if not callable(forward_chunk_plan) or not callable(gather_chunks):
        raise TypeError(
            f"{type(executor).__name__} does not implement "
            "forward_chunk_plan(...) and gather_chunks(...)",
        )
    return executor


def _chunk_output_debug_metrics(output: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    stage_durations = getattr(output, "stage_durations", None)
    if isinstance(stage_durations, Mapping):
        metrics["stage_durations_s"] = {
            str(key): float(value) for key, value in stage_durations.items()
        }

    engine_counters = getattr(output, "engine_counters", None)
    if isinstance(engine_counters, Mapping):
        metrics["engine_counters"] = _debug_metric_value(dict(engine_counters))

    peak_memory_mb = getattr(output, "peak_memory_mb", None)
    if peak_memory_mb is not None:
        metrics["peak_memory_mb"] = float(peak_memory_mb)

    return metrics


def _debug_metric_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _debug_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_metric_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return repr(value)


__all__ = ["GenerationWorkerCore"]
