"""Distributed execution backends for VRL."""

from __future__ import annotations

from vrl.distributed.resources import (
    DistributedResourceConfig,
    ResolvedDistributedResources,
    RoleResourceConfig,
    RolloutResourceConfig,
    format_distributed_resource_plan,
    resolve_distributed_resources,
    trainer_torch_device,
)

__all__ = [
    "DistributedResourceConfig",
    "ResolvedDistributedResources",
    "RoleResourceConfig",
    "RolloutResourceConfig",
    "format_distributed_resource_plan",
    "resolve_distributed_resources",
    "trainer_torch_device",
]
