"""Run-owned Ray reward actors, sharing the in-process model/parking backend.

The driver owns admission and deadlines; a serial actor owns model execution.
CUDA placement comes from the run's resource plan, never from an arbitrary
``worker_config.device`` ordinal. A shared trainer GPU uses explicit node-local
visibility under the existing memory lease, not a second GPU reservation.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.ray.dependencies import current_gpu_ids, current_node_ip, require_ray
from vrl.ray.operation_deadline import (
    RayCallDeadline,
    RayOperationCancelled,
    RayOperationTimeout,
    cancel_ray_refs,
)
from vrl.ray.placement import RolePlacement, actor_scheduling_strategy, require_actor_gpu_ids
from vrl.rewards.inference import RewardInferenceRequest, RewardInferenceResult
from vrl.rewards.launch_contract import RewardRuntimeLaunchContract
from vrl.runtime_errors import TerminalRuntimeError
from vrl.utils.config import import_from_path
from vrl.utils.deadline import require_timeout
from vrl.utils.lifecycle import RuntimeLifecycle, RuntimePhase


@dataclass(frozen=True, slots=True)
class RayRewardPlacement:
    """Resolved rank-local placement supplied by the run's resource owner.

    ``shared_gpu_id`` is a physical node-local visibility ordinal, not the
    actor's torch ordinal (always zero). Shared placement requires ``node_id``
    even when a placement group also constrains scheduling.
    """

    placement: RolePlacement | None = None
    shared_gpu_id: int | None = None
    node_id: str | None = None
    gpu_fraction: float = 1.0


class RayRewardError(TerminalRuntimeError):
    """Remote state is uncertain; preserve artifacts until actor shutdown."""

    retain_reward_artifacts = True


class RayRewardTimeout(RayOperationTimeout):
    retain_reward_artifacts = True


class RayRewardCancelled(RayOperationCancelled):
    retain_reward_artifacts = True


def _remove_actor_media_directory(path: str, token: str) -> None:
    """Remove only this actor's UUID-named root, on the node that created it."""

    root = Path(path)
    if not root.is_absolute() or root.name != f"vrl-reward-actor-{uuid.UUID(token).hex}":
        raise ValueError("refusing cleanup outside the owned reward actor media directory")
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(root)
    if root.exists():
        raise RuntimeError(f"reward actor media directory still exists after cleanup: {root}")


class _RewardActor:
    """Serial actor: synchronous model code never blocks the driver's loop."""

    def __init__(
        self,
        worker_config: dict[str, Any],
        *,
        cuda: bool,
        shared_gpu_id: int | None,
        media_directory_token: str,
    ) -> None:
        from vrl.rewards.runtime import InProcessRewardScorer

        assigned = current_gpu_ids()
        if cuda:
            if shared_gpu_id is None:
                if len(assigned) != 1:
                    raise RuntimeError(
                        f"reward actor requires exactly one assigned GPU: {assigned}"
                    )
            elif assigned or os.environ.get("CUDA_VISIBLE_DEVICES") != str(shared_gpu_id):
                raise RuntimeError("shared reward actor did not receive its scoped GPU visibility")
        elif assigned:
            raise RuntimeError("CPU reward actor unexpectedly owns Ray GPU resources")
        config = dict(worker_config)
        config["device"] = "cuda:0" if cuda else "cpu"
        factory = str(config.get("model_factory", "")).strip()
        if not factory or not callable(import_from_path(factory)):
            raise ValueError("Ray reward model_factory must resolve to a callable")
        self._media_directory_token = media_directory_token
        self._media_directory = Path(tempfile.gettempdir()) / (
            f"vrl-reward-actor-{uuid.UUID(media_directory_token).hex}"
        )
        # Publish the exact path before files exist. Startup cancellation
        # therefore cannot strand an unknown actor directory.
        self._media_directory_created = False
        self._runtime = InProcessRewardScorer(config, media_temp_dir=str(self._media_directory))
        self._loop = asyncio.new_event_loop()
        self._failed: str | None = None
        self._closed = False

    def ready(self) -> dict[str, Any]:
        if self._closed or self._failed is not None:
            raise RuntimeError(f"reward actor is unavailable: {self._failed or 'closed'}")
        ray = require_ray()
        return {
            "worker_id": "reward",
            "node_ip": current_node_ip(),
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "gpu_ids": tuple(current_gpu_ids()),
            "pid": os.getpid(),
            "media_directory": str(self._media_directory),
        }

    def _run(self, method: str, *args: Any) -> Any:
        self.ready()
        try:
            if method == "score_batch" and not self._media_directory_created:
                self._media_directory.mkdir(mode=0o700)
                self._media_directory_created = True
            return self._loop.run_until_complete(getattr(self._runtime, method)(*args))
        except BaseException as error:
            # Do not retain tracebacks and their model/tensor references.
            self._failed = f"{type(error).__name__}: {error}"
            raise

    def score_batch(self, request: RewardInferenceRequest) -> list[RewardInferenceResult]:
        return request.validate_and_order_results(self._run("score_batch", request))

    def activate(self) -> None:
        self._run("activate")

    def park_memory(self) -> None:
        self._run("park_memory")

    def shutdown(self) -> None:
        if self._closed:
            return
        self._loop.run_until_complete(self._runtime.shutdown())
        _remove_actor_media_directory(str(self._media_directory), self._media_directory_token)
        self._closed = True
        self._loop.close()


class RayRewardScorer:
    """One supervised, lazily launched actor per reward component.

    Failure closes admission permanently. Actor-task cancellation alone cannot
    interrupt a synchronous CUDA call: timeout/cancellation also kills the
    actor, retaining its handle until shutdown confirms death. This scorer
    never owns the Ray cluster or the run-level placement group.
    """

    scoring_is_nonblocking = True
    external_accelerator_isolation_verified = True

    def __init__(
        self,
        worker_config: Mapping[str, Any] | None = None,
        *,
        placement: RayRewardPlacement | None = None,
        cpus_per_worker: float = 0.0,
        timeout_s: float = 1800.0,
        startup_timeout_s: float = 600.0,
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        self._launch = RewardRuntimeLaunchContract.from_component_config(worker_config)
        self._cuda = self._launch.device.startswith("cuda")
        if self._launch.device not in ("", "cpu") and not self._cuda:
            raise ValueError("Ray rewards support only CPU or planner-owned CUDA devices")
        self._placement = placement if self._cuda else None
        if self._cuda:
            if self._placement is None:
                raise ValueError("CUDA Ray rewards require a resolved RayRewardPlacement")
            role = self._placement.placement
            shared = self._placement.shared_gpu_id
            fraction = float(self._placement.gpu_fraction)
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError("reward GPU fraction must be finite and in (0, 1]")
            if shared is not None:
                if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
                    raise ValueError("shared reward GPU must be a non-negative physical ordinal")
                if not self._placement.node_id:
                    raise ValueError("shared Ray rewards require an explicit node_id")
                if not self._launch.sleep_offload:
                    raise ValueError("shared Ray rewards require sleep_offload for the GPU lease")
            elif role is None:
                raise ValueError("dedicated CUDA Ray rewards require a reserved placement bundle")
            if role is not None and len(role.bundle_indices) != 1:
                raise ValueError(
                    "a reward scorer requires exactly one rank-local placement bundle"
                )
        self._cpus = float(cpus_per_worker)
        if not math.isfinite(self._cpus) or self._cpus < 0:
            raise ValueError("cpus_per_worker must be finite and >= 0")
        self._timeout = require_timeout(timeout_s)
        self._startup_timeout = require_timeout(startup_timeout_s)
        self._shutdown_timeout = require_timeout(shutdown_timeout_s)
        self.lifecycle = RuntimeLifecycle(owner="Ray reward scorer")
        self._operation_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._actor: Any | None = None
        self._active_ref: Any | None = None
        self._ready = False
        self._kill_requested = False
        self._media_directory_token = uuid.uuid4().hex
        self._actor_directory: tuple[str, str] | None = None

    @property
    def requires_memory_parking(self) -> bool:
        return self._launch.sleep_offload

    def _actor_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_cpus": self._cpus,
            "num_gpus": int(self._cuda),
            "max_restarts": 0,
            "max_task_retries": 0,
        }
        placement = self._placement
        if placement is None:
            # Explicitly prevent GPU discovery even when a parent process
            # disabled Ray's default visibility masking for its own actors.
            options["runtime_env"] = {"env_vars": {"CUDA_VISIBLE_DEVICES": ""}}
            return options
        options["num_gpus"] = placement.gpu_fraction
        if placement.placement is not None:
            role = placement.placement
            options["scheduling_strategy"] = actor_scheduling_strategy(
                role.placement_group,
                bundle_index=role.bundle_indices[0],
            )
        elif placement.node_id is not None:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=placement.node_id,
                soft=False,
            )
        if placement.shared_gpu_id is not None:
            options["num_gpus"] = 0
            options["runtime_env"] = {
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "CUDA_VISIBLE_DEVICES": str(placement.shared_gpu_id),
                },
            }
        return options

    async def _start(self) -> None:
        if self._ready:
            return
        ray = require_ray()
        if not ray.is_initialized():
            raise RuntimeError("Ray reward scorer requires the run owner to initialize Ray")
        self._actor = (
            ray.remote(_RewardActor)
            .options(**self._actor_options())
            .remote(
                dict(self._launch.component_config),
                cuda=self._cuda,
                shared_gpu_id=(None if self._placement is None else self._placement.shared_gpu_id),
                media_directory_token=self._media_directory_token,
            )
        )
        metadata = await self._call("ready", timeout_s=self._startup_timeout)
        self._actor_directory = (metadata["node_id"], metadata["media_directory"])
        placement = self._placement
        if placement is not None:
            try:
                if placement.node_id is not None and metadata["node_id"] != placement.node_id:
                    raise RuntimeError("reward actor landed outside its resolved node")
                if placement.shared_gpu_id is None:
                    require_actor_gpu_ids(
                        [metadata],
                        expected_gpu_ids=placement.placement.expected_gpu_ids,
                        role="reward",
                    )
            except BaseException as error:
                self._fail(error)
                raise
        self._ready = True

    def _fail(self, error: BaseException) -> None:
        self.lifecycle.fail(error)
        ray = require_ray()
        if self._active_ref is not None:
            cancel_ray_refs(ray, [self._active_ref], root_error=error)
        if self._actor is not None:
            try:
                ray.kill(self._actor, no_restart=True)
                self._kill_requested = True
            except Exception as kill_error:
                error.add_note(f"reward actor kill failed; shutdown must retry: {kill_error!r}")

    async def _call(self, method: str, *args: Any, timeout_s: float) -> Any:
        deadline = RayCallDeadline(f"reward.{method}", timeout_s)
        try:
            ref = getattr(self._actor, method).remote(*args)
            self._active_ref = ref
            return await asyncio.wait_for(
                asyncio.wrap_future(ref.future()),
                timeout=deadline.remaining_s(),
            )
        except TimeoutError as cause:
            error = RayRewardTimeout(deadline.operation, deadline.timeout_s)
            self._fail(error)
            raise error from cause
        except asyncio.CancelledError as cause:
            error = RayRewardCancelled(deadline.operation)
            self._fail(error)
            raise error from cause
        except BaseException as cause:
            error = RayRewardError(f"Ray reward {method} failed: {cause}")
            self._fail(error)
            raise error from cause
        finally:
            self._active_ref = None

    async def ensure_ready(self) -> None:
        async with self._operation_lock:
            self.lifecycle.require_running("ensure_ready")
            await self._start()

    async def _execute(self, method: str, *args: Any) -> Any:
        async with self._operation_lock:
            self.lifecycle.require_running(method)
            await self._start()
            self.lifecycle.require_running(method)
            return await self._call(method, *args, timeout_s=self._timeout)

    async def activate(self) -> None:
        await self._execute("activate")

    async def park_memory(self) -> None:
        if not self.requires_memory_parking:
            raise RuntimeError("reward runtime was not configured for complete memory parking")
        await self._execute("park_memory")

    async def score_batch(self, request: RewardInferenceRequest) -> list[RewardInferenceResult]:
        if not isinstance(request, RewardInferenceRequest):
            raise TypeError("Ray reward score_batch requires RewardInferenceRequest")
        results = await self._execute("score_batch", request)
        return request.validate_and_order_results(results)

    async def shutdown(self) -> None:
        self.lifecycle.begin_shutdown()
        async with self._shutdown_lock:
            if self.lifecycle.phase is RuntimePhase.TERMINATED:
                return
            actor = self._actor
            if actor is None:
                await self._cleanup_actor_directory()
                self.lifecycle.finish_shutdown()
                return
            ray = require_ray()
            # A queued shutdown RPC cannot stop a blocked synchronous model.
            # Only attempt graceful cleanup when no operation is in flight.
            if self._active_ref is None and not self._kill_requested:
                try:
                    await self._call("shutdown", timeout_s=self._shutdown_timeout)
                except RayRewardError:
                    pass  # Hard process teardown below remains authoritative.
                except (RayRewardTimeout, RayRewardCancelled):
                    pass
            try:
                ray.kill(actor, no_restart=True)
                self._kill_requested = True
                # kill() submits termination; keep ownership until an RPC
                # confirms actor death, rather than assuming submission freed GPU.
                ref = actor.ready.remote()
                try:
                    await asyncio.wait_for(
                        asyncio.wrap_future(ref.future()),
                        timeout=self._shutdown_timeout,
                    )
                except ray.exceptions.ActorDiedError:
                    pass
                else:
                    raise RuntimeError("reward actor responded after termination was requested")
            except BaseException as error:
                self.lifecycle.fail(error)
                raise
            self._actor = None
            self._ready = False
            await self._cleanup_actor_directory()
            self.lifecycle.finish_shutdown()

    async def _cleanup_actor_directory(self) -> None:
        """Retry exact-root cleanup even after the actor has been confirmed dead."""

        if self._actor_directory is None:
            return
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ray = require_ray()
        node_id, path = self._actor_directory
        if not ray.is_initialized():
            error = RayRewardError(
                f"cannot clean reward media {path!r} on node {node_id!r}: "
                "the run owner shut down Ray before reward cleanup completed",
            )
            self.lifecycle.fail(error)
            raise error
        ref = (
            ray.remote(num_cpus=0, num_gpus=0)(_remove_actor_media_directory)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": ""}},
            )
            .remote(path, self._media_directory_token)
        )
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(ref.future()),
                timeout=self._shutdown_timeout,
            )
        except BaseException as cause:
            error = RayRewardError(
                f"reward actor terminated but media cleanup on node {node_id!r} "
                f"for {path!r} failed; retry shutdown when that node is reachable",
            )
            cancel_ray_refs(ray, [ref], root_error=error)
            self.lifecycle.fail(error)
            raise error from cause
        self._actor_directory = None


__all__ = ["RayRewardPlacement", "RayRewardScorer"]
