"""Uniform rollout-control fake for trainer unit tests."""

from __future__ import annotations

from tests.rollouts.collector._helpers import PromptCollectionFake


class _RuntimeControl:
    current_policy_version = None


class CollectorControlFake(PromptCollectionFake):
    """Supply the lifecycle protocol while tests specialize collection only."""

    generation_runtime = _RuntimeControl()
    reward_isolation_verified = True

    async def activate_generation_runtime(self) -> None:
        return None

    async def offload_generation_runtime_memory(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


__all__ = ["CollectorControlFake"]
