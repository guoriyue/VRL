"""Rollout backend contract."""

from __future__ import annotations

from typing import Protocol

from vrl.engine.core.types import GenerationRequest, OutputBatch


class RolloutBackend(Protocol):
    """Generation backend consumed by rollout collectors."""

    async def generate(self, request: GenerationRequest) -> OutputBatch: ...


__all__ = ["RolloutBackend"]
