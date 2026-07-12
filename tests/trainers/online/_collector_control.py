"""Uniform rollout-control fake for trainer unit tests."""

from __future__ import annotations


class _RuntimeControl:
    current_policy_version = None
    requires_driver_model_offload = False

    def is_colocated(self) -> bool:
        return False


class CollectorControlFake:
    """Supply the lifecycle protocol while tests specialize collection only."""

    runtime = _RuntimeControl()
    requires_runtime_offload_before_reward = False
    requires_driver_model_offload_for_reward = False

    async def activate_runtime(self) -> None:
        return None

    async def offload_runtime_memory(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


__all__ = ["CollectorControlFake"]
