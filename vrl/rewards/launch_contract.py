"""Serializable reward worker launch contract.

The reward twin of ``GenerationRuntimeLaunchContract``
(vrl/generation/launch_contract.py): the typed, closed key set the runtime
itself branches on, parsed once from a reward worker config. It lives apart
from runtime.py and the service because both sides of the process boundary
(in-process runtime, standalone service launch parsing) share only this
contract. The verbatim mapping still reaches the model factory as its open
plugin bag (models read their own keys — model names, thresholds, debug dirs
— which are genuinely unvalidated user input).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RewardWorkerLaunchContract:
    """Runtime-owned keys parsed once from a reward worker config."""

    model_factory: str
    device: str
    sleep_offload: bool
    memory_parking_residual_bytes_limit: int
    reward_model_name: str
    reward_model_version: str
    worker_config: Mapping[str, Any]

    @classmethod
    def from_worker_config(
        cls,
        worker_config: Mapping[str, Any] | None,
    ) -> RewardWorkerLaunchContract:
        cfg = dict(worker_config or {})
        residual_limit = int(cfg.get("memory_parking_residual_bytes_limit", 0))
        if residual_limit < 0:
            raise ValueError("reward memory parking residual limit must be >= 0")
        return cls(
            model_factory=str(cfg.get("model_factory", "")).strip(),
            device=str(cfg.get("device", "")),
            sleep_offload=bool(cfg.get("sleep_offload", False)),
            memory_parking_residual_bytes_limit=residual_limit,
            reward_model_name=str(cfg.get("reward_model_name", "")).strip(),
            reward_model_version=str(cfg.get("reward_model_version", "")).strip(),
            worker_config=cfg,
        )


__all__ = ["RewardWorkerLaunchContract"]
