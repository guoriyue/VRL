"""Shared Ray substrate for domain runtimes."""

from vrl.ray.actor_group import RayActorGroup, RayActorHandle
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs
from vrl.ray.dependencies import current_gpu_ids, current_node_ip, import_from_path, require_ray
from vrl.ray.runtime import RayActorMethodRuntime

__all__ = [
    "RayActorGroup",
    "RayActorHandle",
    "RayActorJob",
    "RayActorMethodRuntime",
    "current_gpu_ids",
    "current_node_ip",
    "import_from_path",
    "require_ray",
    "run_actor_jobs",
]
