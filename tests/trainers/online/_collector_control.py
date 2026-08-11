"""Uniform rollout-control fake for trainer unit tests."""

from __future__ import annotations


class _RuntimeControl:
    current_policy_version = None
    requires_driver_model_offload = False


class CollectorControlFake:
    """Supply the lifecycle protocol while tests specialize collection only."""

    generation_runtime = _RuntimeControl()
    requires_generation_offload_before_reward = False
    requires_driver_model_offload_for_reward = False
    supports_continuous_reward_execution = True

    async def activate_generation_runtime(self) -> None:
        return None

    async def offload_generation_runtime_memory(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


__all__ = ["CollectorControlFake"]
