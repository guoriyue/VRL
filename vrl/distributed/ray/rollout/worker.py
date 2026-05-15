"""Ray rollout worker for chunk-level generation execution."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

from vrl.distributed.ray.dependencies import current_gpu_ids, current_node_ip
from vrl.distributed.ray.module_loading import import_from_path
from vrl.distributed.ray.rollout.types import RayChunkExecutionEnvelope, RayChunkResult
from vrl.engine.core.capabilities import (
    FamilyCapability,
    family_capability_from_value,
)
from vrl.engine.core.protocols import ChunkedFamilyPipelineExecutor
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.core.types import GenerationRequest
from vrl.engine.execution.gather import require_chunked_executor
from vrl.engine.execution.microbatching import MicroBatchPlan
from vrl.models.interfaces import require_runtime_model
from vrl.trainers.core.types import TorchProfilerConfig

logger = logging.getLogger(__name__)


class RayRolloutWorker:
    """Own one rollout executor inside a Ray actor process."""

    def __init__(
        self,
        worker_id: str,
        runtime_spec: GenerationRuntimeSpec | Mapping[str, Any],
    ) -> None:
        self.worker_id = worker_id
        self.runtime_spec = _normalize_runtime_spec(runtime_spec)
        self.family = self.runtime_spec.family
        self.executor: ChunkedFamilyPipelineExecutor | None = None
        self._policy_version: int | None = self.runtime_spec.policy_version
        self._profiler_config = _profiler_config_from_spec(self.runtime_spec)
        self._profiler_output_dir = str(
            self.runtime_spec.extra.get("profiler_output_dir", "outputs/"),
        )
        self._profiler_step = 0
        self.capability = _capability_from_spec(self.runtime_spec)

    def load_policy(self) -> None:
        """Build the family executor from the serialized runtime spec."""

        if self.executor is not None:
            return
        from vrl.trainers.memory import log_host_memory

        log_host_memory(f"ray_worker:{self.worker_id}:before_load_policy", log=logger)
        self.executor = _build_executor(self.runtime_spec)
        self.capability = _merge_loaded_capability(self.capability, self.executor)
        log_host_memory(f"ray_worker:{self.worker_id}:after_load_policy", log=logger)

    def release_policy(self) -> None:
        """Drop loaded model state so the actor releases CUDA memory before exit."""

        self.executor = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        """Update rollout weights, then record the active policy version."""

        self.load_policy()
        policy_obj = getattr(self.executor, "model", None)
        if state_ref is not None:
            model = require_runtime_model(
                policy_obj,
                owner=f"{type(self.executor).__name__}.model",
            )
            model.load_trainable_state(state_ref)
        self._policy_version = int(policy_version)

    def current_policy_version(self) -> int | None:
        return self._policy_version

    def worker_metadata(self, *, runtime_debug: bool = False) -> dict[str, Any]:
        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "local"
            gpu_ids = []
        metadata = {
            "worker_id": self.worker_id,
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
            "policy_version": self._policy_version,
        }
        if runtime_debug:
            metadata["runtime_debug"] = True
            metadata["executor_type"] = type(self.executor).__name__ if self.executor else None
            policy = getattr(self.executor, "model", None) if self.executor is not None else None
            if policy is not None:
                from vrl.trainers.online.diagnostics import (
                    parameter_state_summary,
                    trainable_state_digest,
                )

                metadata["trainable_state"] = trainable_state_digest(policy)
                metadata["parameter_state"] = parameter_state_summary(policy)
                metadata["policy_type"] = type(policy).__name__
        return metadata

    def execute_chunk(
        self,
        request_or_envelope: GenerationRequest | RayChunkExecutionEnvelope,
        chunk: MicroBatchPlan | None = None,
        profiler_label: str | None = None,
    ) -> RayChunkResult:
        self.load_policy()
        envelope = _normalize_execution_envelope(
            request_or_envelope,
            chunk,
            profiler_label,
            self.capability,
        )
        request = envelope.request
        chunk = envelope.chunk
        runtime_debug = bool(request.metadata.get("_runtime_debug"))
        expected_version = request.policy_version
        if expected_version is not None and self._policy_version != expected_version:
            return RayChunkResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                plan_id=envelope.plan_id,
                unit_id=envelope.unit_id,
                unit_name=envelope.unit_name,
                profiler_label=envelope.profiler_label,
                chunk_key=envelope.chunk_key,
                policy_version=self._policy_version,
                error=(
                    "policy_version mismatch: "
                    f"expected={expected_version}, actual={self._policy_version}"
                ),
            )
        try:
            assert self.executor is not None
            output = self._profile_forward_chunk(envelope)
            return RayChunkResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=_to_cpu(output),
                metrics=self._chunk_metrics(
                    envelope,
                    runtime_debug=runtime_debug,
                    plan_aware_chunk=True,
                ),
                plan_id=envelope.plan_id,
                unit_id=envelope.unit_id,
                unit_name=envelope.unit_name,
                profiler_label=envelope.profiler_label,
                chunk_key=envelope.chunk_key,
                policy_version=self._policy_version,
            )
        except Exception as exc:
            return RayChunkResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                plan_id=envelope.plan_id,
                unit_id=envelope.unit_id,
                unit_name=envelope.unit_name,
                profiler_label=envelope.profiler_label,
                chunk_key=envelope.chunk_key,
                policy_version=self._policy_version,
                error=str(exc),
            )

    def _profile_forward_chunk(
        self,
        envelope: RayChunkExecutionEnvelope,
    ) -> Any:
        from vrl.trainers.profiling import record_function, torch_profiler_step

        assert self.executor is not None
        request = envelope.request
        chunk = envelope.chunk
        step = self._profiler_step
        self._profiler_step += 1
        worker_name = (
            f"{self.worker_id}_{self.family}_{self.runtime_spec.task}_"
            f"policy{self._policy_version}_chunk{chunk.prompt_index}_{chunk.sample_start}"
        )
        try:
            device = _executor_device(self.executor)
            event_name = envelope.profiler_label or _default_chunk_profiler_label(
                self.capability,
            )
            forward_chunk_plan = getattr(self.executor, "forward_chunk_plan", None)
            if not callable(forward_chunk_plan):
                raise TypeError(
                    f"{type(self.executor).__name__} must implement "
                    "forward_chunk_plan(...) for Ray chunk execution",
                )
            if envelope.execution_unit is None:
                raise RuntimeError("Ray chunk execution requires an EnginePlan execution unit")
            with torch_profiler_step(
                self._profiler_config,
                output_dir=self._profiler_output_dir,
                step=step,
                device=device,
                worker_name=worker_name,
                trace_subdir=f"rollout/{self.worker_id}",
            ), record_function(event_name):
                return forward_chunk_plan(
                    request,
                    chunk,
                    envelope.execution_unit,
                    envelope.plan_summary,
                )
        except Exception:
            logger.exception("Ray rollout profiler-wrapped chunk failed")
            raise

    def _chunk_metrics(
        self,
        envelope: RayChunkExecutionEnvelope,
        *,
        runtime_debug: bool,
        plan_aware_chunk: bool | None = None,
    ) -> dict[str, Any]:
        metrics = self.worker_metadata(runtime_debug=runtime_debug)
        metrics.update(
            {
                "plan_id": envelope.plan_id,
                "engine_plan_id": envelope.plan_id,
                "unit_id": envelope.unit_id,
                "unit_name": envelope.unit_name,
                "profiler_label": envelope.profiler_label,
                "chunk_key": envelope.chunk_key,
                "capability": dict(envelope.capability_summary),
            }
        )
        if plan_aware_chunk is not None:
            metrics["plan_aware_chunk"] = plan_aware_chunk
        return metrics


def _normalize_runtime_spec(
    runtime_spec: GenerationRuntimeSpec | Mapping[str, Any],
) -> GenerationRuntimeSpec:
    spec = GenerationRuntimeSpec.from_value(runtime_spec)
    if spec.family is None:
        raise ValueError("GenerationRuntimeSpec.family is required")
    return spec


def _build_executor(runtime_spec: GenerationRuntimeSpec) -> ChunkedFamilyPipelineExecutor:
    builder_path = runtime_spec.runtime_builder
    executor_path = runtime_spec.executor_cls
    if builder_path is None or executor_path is None:
        raise ValueError(
            "GenerationRuntimeSpec requires runtime_builder and executor_cls import paths",
        )

    from vrl.models.interfaces.runtime import RuntimeBuildSpec

    build_runtime_bundle = import_from_path(str(builder_path))
    executor_cls = import_from_path(str(executor_path))
    bundle = build_runtime_bundle(
        RuntimeBuildSpec(
            **_normalize_runtime_build_spec_payload(
                runtime_spec.build_spec_payload(),
            )
        ),
    )
    model = require_runtime_model(bundle.model, owner="RuntimeBundle.model")
    built = executor_cls(model, **dict(runtime_spec.executor_kwargs))
    if getattr(bundle, "runtime_caps", None) is not None:
        built.runtime_caps = dict(bundle.runtime_caps)
    return require_chunked_executor(built)


def _profiler_config_from_spec(runtime_spec: GenerationRuntimeSpec) -> TorchProfilerConfig:
    raw = runtime_spec.extra.get("torch_profiler", {})
    if isinstance(raw, Mapping):
        return TorchProfilerConfig(**dict(raw))
    return TorchProfilerConfig()


def _capability_from_spec(runtime_spec: GenerationRuntimeSpec) -> FamilyCapability:
    capability = family_capability_from_value(runtime_spec.extra.get("family_capability"))
    if capability is not None:
        return capability
    raise ValueError(
        "GenerationRuntimeSpec.extra['family_capability'] is required for Ray rollout",
    )


def _normalize_execution_envelope(
    request_or_envelope: GenerationRequest | RayChunkExecutionEnvelope,
    chunk: MicroBatchPlan | None,
    profiler_label: str | None,
    capability: FamilyCapability,
) -> RayChunkExecutionEnvelope:
    if isinstance(request_or_envelope, RayChunkExecutionEnvelope):
        return request_or_envelope
    del chunk, profiler_label, capability
    raise TypeError(
        "RayRolloutWorker.execute_chunk requires RayChunkExecutionEnvelope; "
        "request+chunk execution is no longer supported",
    )


def _merge_loaded_capability(
    capability: FamilyCapability,
    executor: ChunkedFamilyPipelineExecutor,
) -> FamilyCapability:
    runtime_caps = getattr(executor, "runtime_caps", None)
    merged = capability.with_runtime_caps(runtime_caps if isinstance(runtime_caps, Mapping) else None)
    declared = _declared_executor_capability(executor)
    if declared is None:
        return merged
    if declared.family != merged.family or declared.task != merged.task:
        raise ValueError(
            "executor capability does not match runtime spec: "
            f"{declared.family}/{declared.task} != {merged.family}/{merged.task}",
        )
    if declared.trajectory_kind != merged.trajectory_kind:
        raise ValueError(
            "executor trajectory capability does not match runtime spec: "
            f"{declared.trajectory_kind} != {merged.trajectory_kind}",
        )
    return declared.with_runtime_caps(runtime_caps if isinstance(runtime_caps, Mapping) else None)


def _declared_executor_capability(
    executor: ChunkedFamilyPipelineExecutor,
) -> FamilyCapability | None:
    method = getattr(executor, "capability", None)
    if callable(method):
        return family_capability_from_value(method())
    for attr_name in ("family_capability", "capability_metadata"):
        value = getattr(executor, attr_name, None)
        if value is not None:
            return family_capability_from_value(value)
    return None


def _default_chunk_profiler_label(capability: FamilyCapability) -> str:
    del capability
    return "engine.forward_chunk"


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


def _normalize_runtime_build_spec_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    device = normalized.get("device")
    if isinstance(device, str):
        import torch

        normalized["device"] = torch.device(device)
    dtype = normalized.get("dtype")
    if isinstance(dtype, str):
        normalized["dtype"] = _torch_dtype_from_string(dtype)
    return normalized


def _torch_dtype_from_string(value: str) -> Any:
    import torch

    key = value.removeprefix("torch.").lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype string in runtime_spec: {value!r}") from exc


def _to_cpu(value: Any) -> Any:
    if _is_tensor(value):
        return value.detach().cpu()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload = {
            field.name: _to_cpu(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
        return type(value)(**payload)
    if isinstance(value, dict):
        return {key: _to_cpu(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_cpu(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(inner) for inner in value)
    return value


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu")


__all__ = ["RayRolloutWorker"]
