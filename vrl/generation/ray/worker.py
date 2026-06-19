"""Thin Ray actor wrapper for generation worker execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.ray.dependencies import current_gpu_ids, current_node_ip


class RayGenerationWorker:
    """Ray actor adapter around ``GenerationWorkerCore``."""

    def __init__(
        self,
        worker_id: str,
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
    ) -> None:
        self.core = GenerationWorkerCore(
            worker_id,
            launch_contract,
            metadata_provider=self._ray_metadata,
        )

    @property
    def worker_id(self) -> str:
        return self.core.worker_id

    @property
    def executor(self) -> Any:
        return self.core.executor

    def load_policy(self) -> None:
        self.core.load_policy()

    def release_policy(self) -> None:
        self.core.release_policy()

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        self.core.update_weights(state_ref, policy_version)

    def current_policy_version(self) -> int | None:
        return self.core.current_policy_version()

    def supports_versioned_trainable_state(self) -> bool:
        return self.core.supports_versioned_trainable_state()

    def worker_metadata(self, *, runtime_debug: bool = False) -> dict[str, Any]:
        return self.core.worker_metadata(runtime_debug=runtime_debug)

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        return self.core.execute_chunk(envelope)

    def _ray_metadata(self) -> dict[str, Any]:
        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "unknown"
            gpu_ids = []
        return {"worker_id": self.worker_id, "node_ip": node_ip, "gpu_ids": gpu_ids}

__all__ = ["RayGenerationWorker"]
