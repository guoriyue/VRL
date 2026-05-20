"""Shared Ray substrate for domain runtimes."""

from vrl.ray.actor_group import RayActorGroup, RayActorHandle
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs
from vrl.ray.dependencies import current_gpu_ids, current_node_ip, import_from_path, require_ray
from vrl.ray.resources import (
    DistributedResourceConfig,
    ResolvedDistributedResources,
    RewardResourceConfig,
    RoleResourceConfig,
    RolloutResourceConfig,
    format_distributed_resource_plan,
    resolve_distributed_resources,
    reward_runtime_resource_kwargs,
    trainer_torch_device,
)
from vrl.ray.runtime import RayActorMethodRuntime

__all__ = [
    "DistributedResourceConfig",
    "RayActorGroup",
    "RayActorHandle",
    "RayActorJob",
    "RayActorMethodRuntime",
    "ResolvedDistributedResources",
    "RewardResourceConfig",
    "RoleResourceConfig",
    "RolloutResourceConfig",
    "current_gpu_ids",
    "current_node_ip",
    "format_distributed_resource_plan",
    "import_from_path",
    "require_ray",
    "resolve_distributed_resources",
    "reward_runtime_resource_kwargs",
    "run_actor_jobs",
    "trainer_torch_device",
]
