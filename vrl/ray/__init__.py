"""Shared Ray substrate for domain runtimes."""

from vrl.ray.actor_group import RayActorGroup, RayActorHandle
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs
from vrl.ray.dependencies import current_gpu_ids, current_node_ip, require_ray
from vrl.ray.resources import (
    DistributedResourceConfig,
    ResolvedDistributedResources,
    RewardResourceConfig,
    RoleResourceConfig,
    RolloutResourceConfig,
    format_distributed_resource_plan,
    resolve_distributed_resources,
    trainer_torch_device,
)

__all__ = [
    "DistributedResourceConfig",
    "RayActorGroup",
    "RayActorHandle",
    "RayActorJob",
    "ResolvedDistributedResources",
    "RewardResourceConfig",
    "RoleResourceConfig",
    "RolloutResourceConfig",
    "current_gpu_ids",
    "current_node_ip",
    "format_distributed_resource_plan",
    "require_ray",
    "resolve_distributed_resources",
    "run_actor_jobs",
    "trainer_torch_device",
]
