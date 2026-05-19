"""Generation worker core independent of the Ray actor wrapper."""

from __future__ import annotations

import dataclasses
import importlib
import logging
from collections.abc import Callable, Mapping
from typing import Any

from vrl.generation.capabilities import (
    FamilyCapability,
    family_capability_from_value,
)
from vrl.generation.execution.distributed.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkedFamilyPipelineExecutor
from vrl.models.interfaces import require_runtime_model
from vrl.utils.cuda_memory import release_cuda_memory
from vrl.utils.profiling import TorchProfilerConfig

logger = logging.getLogger(__name__)


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
        self.executor: ChunkedFamilyPipelineExecutor | None = None
        self._policy_version: int | None = self.launch_contract.policy_version
        self._profiler_config = self._profiler_config_from_contract(self.launch_contract)
        self._profiler_output_dir = str(
            self.launch_contract.extra.get("profiler_output_dir", "outputs/"),
        )
        self._profiler_step = 0
        self.capability = self._capability_from_contract(self.launch_contract)
        self._metadata_provider = metadata_provider or self._local_metadata

    def load_policy(self) -> None:
        """Build the family executor from the serialized launch contract."""

        if self.executor is not None:
            return
        from vrl.utils.memory import log_host_memory

        log_host_memory(f"generation_worker:{self.worker_id}:before_load_policy", log=logger)
        self.executor = self._build_executor()
        self.capability = self._merge_loaded_capability(self.executor)
        log_host_memory(f"generation_worker:{self.worker_id}:after_load_policy", log=logger)

    def release_policy(self) -> None:
        """Drop loaded model state so the worker releases CUDA memory before exit."""

        self.executor = None
        release_cuda_memory(gc_collect=True, ipc_collect=True)

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        """Update generation weights, then record the active policy version."""

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
            metadata = dict(self._metadata_provider())
        except Exception:
            metadata = self._local_metadata()
        metadata.setdefault("worker_id", self.worker_id)
        metadata.setdefault("node_ip", "local")
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
        if expected_version is not None and self._policy_version != expected_version:
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
        try:
            assert self.executor is not None
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
                ),
                plan_id=envelope.plan_id,
                stage_id=envelope.stage_id,
                stage_name=envelope.stage_name,
                profiler_label=envelope.profiler_label,
                chunk_key=envelope.chunk_key,
                policy_version=self._policy_version,
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
                policy_version=self._policy_version,
                error=str(exc),
            )

    def _profile_forward_chunk(
        self,
        envelope: ChunkExecutionEnvelope,
    ) -> Any:
        from vrl.utils.profiling import record_function, torch_profiler_step

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
            with torch_profiler_step(
                self._profiler_config,
                output_dir=self._profiler_output_dir,
                step=step,
                device=device,
                worker_name=worker_name,
                trace_subdir=f"generation/{self.worker_id}",
            ), record_function(event_name):
                return forward_chunk_plan(
                    request,
                    chunk,
                    envelope.execution_stage,
                    envelope.plan_summary,
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
                "capability": dict(envelope.capability_summary),
            },
        )
        if plan_aware_chunk is not None:
            metrics["plan_aware_chunk"] = plan_aware_chunk
        return metrics

    @staticmethod
    def _normalize_launch_contract(
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
    ) -> GenerationRuntimeLaunchContract:
        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
        if contract.family is None:
            raise ValueError("GenerationRuntimeLaunchContract.family is required")
        return contract

    def _build_executor(self) -> ChunkedFamilyPipelineExecutor:
        launch_contract = self.launch_contract
        builder_path = launch_contract.runtime_builder
        executor_path = launch_contract.executor_cls
        if builder_path is None or executor_path is None:
            raise ValueError(
                "GenerationRuntimeLaunchContract requires runtime_builder and "
                "executor_cls import paths",
            )

        from vrl.models.interfaces.runtime import RuntimeBuildSpec

        build_runtime_bundle = _import_from_path(str(builder_path))
        executor_cls = _import_from_path(str(executor_path))
        bundle = build_runtime_bundle(
            RuntimeBuildSpec(
                **self._normalize_runtime_build_payload(
                    launch_contract.model_build_payload(),
                ),
            ),
        )
        model = require_runtime_model(bundle.model, owner="RuntimeBundle.model")
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
        executor: ChunkedFamilyPipelineExecutor,
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
            normalized["dtype"] = cls._torch_dtype_from_string(dtype)
        return normalized

    @staticmethod
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
            raise ValueError(
                f"unsupported torch dtype string in launch_contract: {value!r}",
            ) from exc

    @classmethod
    def _to_cpu(cls, value: Any) -> Any:
        if cls._is_tensor(value):
            return value.detach().cpu()
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            payload = {
                field.name: cls._to_cpu(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
            return type(value)(**payload)
        if isinstance(value, dict):
            return {key: cls._to_cpu(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [cls._to_cpu(inner) for inner in value]
        if isinstance(value, tuple):
            return tuple(cls._to_cpu(inner) for inner in value)
        return value

    @staticmethod
    def _is_tensor(value: Any) -> bool:
        return hasattr(value, "detach") and hasattr(value, "cpu")

    def _local_metadata(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "node_ip": "local", "gpu_ids": []}


def _require_chunked_executor(executor: Any) -> ChunkedFamilyPipelineExecutor:
    forward_chunk_plan = getattr(executor, "forward_chunk_plan", None)
    gather_chunks = getattr(executor, "gather_chunks", None)
    if not callable(forward_chunk_plan) or not callable(gather_chunks):
        raise TypeError(
            f"{type(executor).__name__} does not implement "
            "forward_chunk_plan(...) and gather_chunks(...)",
        )
    return executor


def _import_from_path(path: str) -> Any:
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid import path: {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


__all__ = ["GenerationWorkerCore"]
