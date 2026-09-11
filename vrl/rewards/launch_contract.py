"""Serializable reward runtime launch contract.

The reward twin of ``GenerationRuntimeLaunchContract``
(vrl/generation/launch_contract.py): the typed, closed key set the runtime
itself branches on, parsed once from a reward component config. It lives apart
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

from vrl.utils.config import require_exact_int


@dataclass(frozen=True, slots=True)
class RewardRuntimeLaunchContract:
    """Runtime-owned keys parsed once from a reward component config."""

    model_factory: str
    device: str
    sleep_offload: bool
    memory_parking_residual_bytes_limit: int
    reward_model_name: str
    reward_model_version: str
    component_config: Mapping[str, Any]

    @classmethod
    def from_component_config(
        cls,
        component_config: Mapping[str, Any] | None,
    ) -> RewardRuntimeLaunchContract:
        cfg = dict(component_config or {})
        residual_limit = require_exact_int(
            cfg.get("memory_parking_residual_bytes_limit", 0),
            path="reward memory_parking_residual_bytes_limit",
            minimum=0,
        )
        sleep_offload = cfg.get("sleep_offload", False)
        if not isinstance(sleep_offload, bool):
            raise ValueError("reward sleep_offload must be a boolean")
        return cls(
            model_factory=str(cfg.get("model_factory", "")).strip(),
            device=str(cfg.get("device", "")),
            sleep_offload=sleep_offload,
            memory_parking_residual_bytes_limit=residual_limit,
            reward_model_name=str(cfg.get("reward_model_name", "")).strip(),
            reward_model_version=str(cfg.get("reward_model_version", "")).strip(),
            component_config=cfg,
        )


__all__ = ["RewardRuntimeLaunchContract"]
