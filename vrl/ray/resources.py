"""Role-level resource resolution for distributed VRL runs.

Resolves the ``distributed.resources`` config into concrete per-role CUDA
ordinals (``ResolvedDistributedResources``) and the topology-derived release
plan (``RayLifecyclePlan``).
Deliberately Ray-free: resolution runs before ``ray.init()`` (see
``vrl/scripts/common/online.py``), so everything here is static arithmetic
over config and visible GPUs; live-cluster checks belong to
``vrl.ray.placement.cross_node_preflight``. Consumers: the online launcher and
trainer read the resolved plan, ``vrl.ray.placement`` derives the run-level
bundle layout from it, and the rollout collector reads only ``lifecycle`` to
schedule GPU handoffs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.utils.config import to_builtin_deep

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


@dataclass(frozen=True, slots=True)
class RoleResourceConfig:
    """GPU ownership request for one execution role."""

    # Which ``distributed.resources.<role>`` block this request was parsed from.
    # A ClassVar, not a field: it is fixed by the subclass, and dataclasses.fields
    # (the source of truth for the accepted YAML keys, vrl/config/schema.py) must
    # not start advertising it as a settable key.
    role: ClassVar[str] = "trainer"

    num_gpus: int | str | None = "auto"
    devices: list[int] | str = "auto"

    @property
    def key_prefix(self) -> str:
        """Public config path of this role's block, for error messages."""

        return f"distributed.resources.{self.role}"


@dataclass(frozen=True, slots=True)
class RolloutResourceConfig(RoleResourceConfig):
    """GPU ownership request for rollout engines.

    The only role that runs engine replicas: the trainer is the driver process
    itself, and rewards score in-process (see :class:`RewardResourceConfig`),
    so the engine-count arithmetic lives here.
    """

    role: ClassVar[str] = "rollout"

    # Rollout GPU pool source (public key: distributed.resources.rollout.gpu_pool).
    # Mirrors reward.gpu_pool so all roles share one "which pool do I borrow" grammar:
    #   "auto"      derive from topology: a dedicated spare GPU when one exists,
    #               else share the trainer GPU (single-GPU colocated fallback).
    #   "trainer"   share the trainer GPU pool (colocated); shared roles always
    #               hand the GPU over between phases.
    #   "dedicated" require a dedicated spare rollout GPU; error if none exists.
    # Sharing needs no separate consent flag: it is declared either by a pool
    # word ("trainer") or by hand-pinning intersecting ``devices`` sets, and the
    # resolved sharing plan is announced in the startup resource receipt.
    gpu_pool: Literal["auto", "trainer", "dedicated"] = "auto"

    # Engine replica count (the data-parallel degree). A GPU fleet grants
    # ``gpus_per_engine`` GPUs to each engine, so the count derives from the
    # resolved GPU set; the knob carries information only for the two
    # GPU-less shapes: ``0`` = no fleet at all (offline recipes), explicit
    # ``N`` with ``num_gpus: 0`` = a CPU engine fleet (defaults to 1).
    num_engines: int | str = "auto"

    # Ranks (GPUs) per engine (public key:
    # distributed.resources.rollout.gpus_per_engine; slime's
    # --rollout-num-gpus-per-engine). 1 = single-GPU engines. > 1 groups
    # consecutive rollout GPUs into one sequence-parallel engine and requires
    # the family capability supports_multi_gpu_engine — validated at launch
    # preflight so an unsupporting family fails loud instead of silently
    # computing every batch N times.
    gpus_per_engine: int = 1

    def requests_cpu_fleet(self) -> bool:
        """True when the config explicitly asks for GPU-less rollout engines."""

        parsed_num_gpus = _parse_num_gpus(
            self.num_gpus,
            field_name=f"{self.key_prefix}.num_gpus",
        )
        return parsed_num_gpus == 0

    def requested_gpu_count(self, *, available_count: int) -> int:
        """GPU count this role requests from a pool of ``available_count`` GPUs."""

        parsed_num_gpus = _parse_num_gpus(
            self.num_gpus,
            field_name=f"{self.key_prefix}.num_gpus",
        )
        if parsed_num_gpus != "auto" and parsed_num_gpus is not None:
            count = int(parsed_num_gpus)
            if count < 0:
                raise ValueError(f"{self.key_prefix}.num_gpus must be >= 0")
            return count

        parsed_engines = _parse_num_engines(
            self.num_engines,
            field_name=f"{self.key_prefix}.num_engines",
        )
        if parsed_engines != "auto":
            # gpus_per_engine GPUs per engine; num_engines: 0 declares no fleet.
            return int(parsed_engines) * self.parsed_gpus_per_engine()
        return int(available_count)

    def parsed_gpus_per_engine(self) -> int:
        gpus_per_engine = int(self.gpus_per_engine)
        if gpus_per_engine < 1:
            raise ValueError(f"{self.key_prefix}.gpus_per_engine must be >= 1")
        return gpus_per_engine

    def resolve_num_engines(self, *, resolved_gpu_count: int) -> int:
        """Engine replica count for the GPUs this role actually resolved to.

        ``resolved_gpu_count`` is the length of the resolved device tuple, not
        the ``num_gpus`` request field. A GPU fleet groups its GPUs into
        engines of ``gpus_per_engine`` ranks; with zero resolved GPUs the
        fleet is either explicitly CPU (``num_gpus: 0`` -> ``num_engines``
        engines, default 1) or absent.
        """

        requested = _parse_num_engines(
            self.num_engines,
            field_name=f"{self.key_prefix}.num_engines",
        )
        gpus_per_engine = self.parsed_gpus_per_engine()
        if resolved_gpu_count > 0:
            if resolved_gpu_count % gpus_per_engine:
                raise ValueError(
                    f"{self.key_prefix}: {resolved_gpu_count} rollout GPU(s) are "
                    f"not divisible into engines of {gpus_per_engine} rank(s); "
                    "adjust the GPU count or gpus_per_engine",
                )
            derived = resolved_gpu_count // gpus_per_engine
            if requested == "auto":
                return derived
            if int(requested) != derived:
                raise ValueError(
                    f"{self.key_prefix}.num_engines must equal rollout GPUs / "
                    f"gpus_per_engine: {int(requested)} != "
                    f"{resolved_gpu_count} / {gpus_per_engine}",
                )
            return int(requested)

        if requested == "auto":
            return 1 if self.requests_cpu_fleet() else 0
        return int(requested)


@dataclass(frozen=True, slots=True)
class RewardResourceConfig:
    """Compute ownership for in-process reward inference.

    Rewards score in the driver process — there is no reward worker fleet, so
    the request is one question, not worker arithmetic: which compute does the
    reward model own?

    ``device`` (public key: distributed.resources.reward.device):
      "trainer"  follow the trainer's device; no reservation (default, and the
                 meaning of an absent reward block)
      "cpu"      score on CPU; no GPU reservation
      "gpu"      reserve exactly one GPU, chosen from ``gpu_pool``
    ``gpu_pool`` (device: gpu only):
      "auto"      a dedicated spare GPU when one exists, else share rollout's
      "rollout"   always share the rollout GPU pool
      "dedicated" require a dedicated spare reward GPU; error if none exists
    ``devices`` (device: gpu only): pin the reservation to one explicit CUDA
    ordinal instead of pool selection.
    """

    role: ClassVar[str] = "reward"

    device: Literal["trainer", "cpu", "gpu"] = "trainer"
    devices: list[int] | str = "auto"
    gpu_pool: Literal["auto", "rollout", "dedicated"] = "auto"

    @property
    def key_prefix(self) -> str:
        """Public config path of this role's block, for error messages."""

        return f"distributed.resources.{self.role}"


@dataclass(frozen=True, slots=True)
class DistributedResourceConfig:
    """Top-level role resource request."""

    visible_devices: list[int] | str = "auto"
    trainer: RoleResourceConfig = field(default_factory=RoleResourceConfig)
    rollout: RolloutResourceConfig = field(default_factory=RolloutResourceConfig)
    reward: RewardResourceConfig = field(default_factory=RewardResourceConfig)
    cross_node: bool = False


@dataclass(frozen=True, slots=True)
class RayLifecyclePlan:
    """Single topology-derived answer to "which role yields its GPU when".

    Stores the run's three pairwise GPU-sharing facts — the only independent
    bits in the trainer/rollout/reward topology triangle. Every handoff flag
    and lease mode is a phase-ordered *view* over them, derived by property,
    so no stored flag can drift from the sharing fact that implies it. Built
    by :func:`ResolvedDistributedResources`; the launcher, collector, and
    reward runtime read this one declarative plan instead of re-deriving
    releases from raw device sets.
    """

    trainer_and_rollout_share_gpu: bool
    rollout_and_reward_share_gpu: bool
    trainer_and_reward_share_gpu: bool

    @property
    def rollout_mode(self) -> Literal["resident", "on_demand"]:
        """``resident`` keeps serving across phases (rollout owns its GPU);
        ``on_demand`` yields a contested GPU at a handoff and reactivates on
        next use (workers park in host RAM — no process destruction)."""

        contested = self.trainer_and_rollout_share_gpu or self.rollout_and_reward_share_gpu
        return "on_demand" if contested else "resident"

    @property
    def release_rollout_before_train(self) -> bool:
        return self.trainer_and_rollout_share_gpu

    @property
    def release_rollout_before_reward(self) -> bool:
        return self.rollout_and_reward_share_gpu

    @property
    def release_trainer_before_reward(self) -> bool:
        return self.trainer_and_reward_share_gpu

    @property
    def release_reward_after_score(self) -> bool:
        """A reward on anyone else's card must prove it parked after scoring."""

        return self.rollout_and_reward_share_gpu or self.trainer_and_reward_share_gpu


@dataclass(frozen=True, slots=True)
class ResolvedDistributedResources:
    """Concrete resource plan consumed by trainer and Ray role launchers."""

    # display/provenance-only: the full visible GPU pool, printed by
    # format_distributed_resource_plan. No behavioral consumer reads it; it
    # records which GPUs the machine exposed (may exceed the role union's spare).
    visible_devices: tuple[int, ...]
    trainer_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]
    reward_devices: tuple[int, ...]
    # distributed.resources.reward.device == "cpu": an explicit CPU reward
    # reservation, distinct from "no reservation, follow the trainer device".
    reward_runs_on_cpu: bool
    rollout_num_engines: int
    # Ranks (GPUs) per rollout engine; consumed by the launcher's engine
    # grouping and BundleLayout's per-engine bundle groups.
    rollout_gpus_per_engine: int
    cross_node: bool
    # Named view over release decisions: lease mode per role plus the per-boundary
    # handoff. The launcher/collector/reward read this instead of re-deriving from
    # device sets.
    lifecycle: RayLifecyclePlan

    @property
    def rollout_num_gpus(self) -> int:
        return len(self.rollout_devices)

    @property
    def colocated(self) -> bool:
        return bool(set(self.trainer_devices) & set(self.rollout_devices))

    @property
    def requires_trainer_reservation(self) -> bool:
        return (
            bool(self.trainer_devices)
            and bool(self.rollout_devices)
            and not self.colocated
            and self.rollout_num_engines > 0
            and not self.cross_node
        )

    @property
    def trainer_torch_device(self) -> str:
        """Torch device string the single-process trainer should use."""

        devices = tuple(self.trainer_devices)
        if not devices:
            return "cpu"
        return f"cuda:{int(devices[0])}"

    def reward_torch_device(self, *, trainer_device: Any | None = None) -> str:
        """Device for the local, in-process reward runtime.

        A resolved reward GPU is a local reservation, not a Ray worker
        placement; resolution already enforced "exactly one, never under
        cross_node". When no reward GPU is reserved, reward inference follows
        the trainer device. ``trainer_device`` lets torchrun callers provide
        their rank-local device instead of the resolver's rank-agnostic
        trainer ordinal.
        """

        devices = tuple(self.reward_devices)
        if devices:
            return f"cuda:{self._local_torch_ordinal(int(devices[0]))}"
        if self.reward_runs_on_cpu:
            return "cpu"
        if trainer_device is not None:
            return str(trainer_device)
        return self.trainer_torch_device

    def plan_device_ordinal(self, torch_ordinal: int) -> int:
        """Translate a process-local torch ordinal back into plan space.

        Inverse of ``_local_torch_ordinal``, for callers that must compare an
        execution device against Ray-side state (placement bundles report
        physical ids from ``ray.get_gpu_ids()`` regardless of the process's
        CUDA mask). Passthrough whenever the process view and the plan already
        share one space.
        """

        import torch

        visible = [int(device) for device in self.visible_devices]
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == len(visible)
            and 0 <= torch_ordinal < len(visible)
        ):
            return visible[torch_ordinal]
        return torch_ordinal

    def _local_torch_ordinal(self, plan_ordinal: int) -> int:
        """Translate a plan-space CUDA ordinal into this process's torch ordinal.

        Rank-local torchrun launches keep the resource plan in Ray's physical
        ordinal space but narrow the process to CUDA_VISIBLE_DEVICES=<physical>
        (see train.py ``_narrow_rank_local_cuda_visibility``), so torch
        enumerates exactly ``visible_devices`` in order and the torch ordinal
        is the id's *position*, not its value — returning the raw physical id
        raised "CUDA error: invalid device ordinal" on every non-zero rank of
        the hpsv3 fsdp 4-rank smoke (2026-08-16; rank0 survived only because
        physical 0 == logical 0). Translate only when the process view
        provably matches the plan (device_count == len(visible_devices)); a
        wider process view (bare launch, subset plan without a mask) keeps
        plan ordinals valid as-is, and an identity pool translates to itself.
        """

        import torch

        visible = [int(device) for device in self.visible_devices]
        if plan_ordinal not in visible:
            return plan_ordinal
        if torch.cuda.is_available() and torch.cuda.device_count() == len(visible):
            return visible.index(plan_ordinal)
        return plan_ordinal

    @classmethod
    def from_root(
        cls,
        root: RootConfig,
        *,
        reward_inference: dict[str, RewardInferenceConfig] | None = None,
    ) -> ResolvedDistributedResources:
        """Build the resource plan from the root: role-level requests to concrete CUDA ordinals.

        This is the single source of truth for trainer/rollout/reward GPU
        ownership. It intentionally does static ownership checks only; memory
        pressure is still a runtime concern.

        ``reward_inference`` is the already-resolved per-component deployment map
        (``BuiltConfigs.reward.inference_configs``); training scripts pass it so the
        reward inference is resolved once at config-build time. When omitted (e.g.
        isolated resource tests) it is resolved from ``root.reward``.
        """

        distributed = root.distributed
        config = (
            distributed.resources
            if distributed is not None and distributed.resources is not None
            else DistributedResourceConfig()
        )
        if reward_inference is None:
            if root.reward is None:
                reward_inference = {}
            else:
                from vrl.config.builders import RewardRuntimeConfig

                reward_inference = RewardRuntimeConfig.from_cfg(root.reward).inference_configs
        local_reward_configured = any(
            inference.kind == "in_process" for inference in reward_inference.values()
        )
        if reward_inference and not local_reward_configured:
            # External services own their accelerator and process placement. Ignore
            # inherited reward presets here instead of creating a phantom local GPU
            # or CPU bundle; the HTTP runtime is a driver-side client.
            config = replace(config, reward=RewardResourceConfig())
        training = None if distributed is None else distributed.training
        training_strategy = "single_process" if training is None else str(training.strategy)
        training_world_size = (
            1 if training is None else int(training.num_nodes) * int(training.gpus_per_node)
        )
        if config.cross_node:
            visible_devices = _resolve_cross_node_visible_devices(config)
        else:
            parsed_visible_devices = _parse_devices(config.visible_devices)
            visible_devices = (
                _auto_visible_cuda_devices()
                if parsed_visible_devices == "auto"
                else tuple(
                    _dedupe_ints(
                        parsed_visible_devices,
                        field_name="distributed.resources.visible_devices",
                    ),
                )
            )

        # fsdp has two topologies. ASYMMETRIC: one resolver owns the whole training
        # world (trainer = num_nodes*gpus_per_node GPUs, rollout/reward on separate
        # cards) — the single-node multi-GPU model. SYMMETRIC COLOCATED: one torchrun
        # rank per node, each owning its 1 local GPU with a colocated rollout — the same
        # per-rank-local model ddp uses (SPRINT_symmetric_colocated_ddp), signaled by
        # rollout.gpu_pool=trainer. Only asymmetric fsdp sizes the trainer to the whole
        # world; symmetric fsdp follows the per-rank single-GPU rule like ddp.
        fsdp_symmetric_colocated = (
            training_strategy == "fsdp" and config.rollout.gpu_pool == "trainer"
        )
        fsdp_asymmetric = training_strategy == "fsdp" and not fsdp_symmetric_colocated
        trainer_default_auto = (
            training_world_size if fsdp_asymmetric else (1 if visible_devices else 0)
        )
        trainer_devices = _resolve_role_devices(
            visible_devices=visible_devices,
            role_config=config.trainer,
            default_auto_count=trainer_default_auto,
        )
        _validate_trainer_device_count(
            training_strategy,
            trainer_devices,
            training_world_size,
            symmetric_colocated=fsdp_symmetric_colocated,
        )

        rollout_gpus_per_engine = config.rollout.parsed_gpus_per_engine()
        if rollout_gpus_per_engine > 1:
            if config.cross_node:
                raise ValueError(
                    "distributed.resources.rollout.gpus_per_engine > 1 is not "
                    "supported with cross_node=true: an engine's ranks must share "
                    "one node for its process group and PACK placement",
                )
            if config.rollout.requests_cpu_fleet():
                raise ValueError(
                    "distributed.resources.rollout.gpus_per_engine > 1 requires a "
                    "GPU fleet; a CPU engine (num_gpus: 0) has no ranks to group",
                )
        rollout_devices = _resolve_rollout_devices(
            visible_devices=visible_devices,
            trainer_devices=trainer_devices,
            rollout_config=config.rollout,
        )
        rollout_num_gpus = len(rollout_devices)
        rollout_num_engines = config.rollout.resolve_num_engines(
            resolved_gpu_count=rollout_num_gpus,
        )

        # Sharing is consent: an intersection can only arise from hand-pinned
        # ``devices`` sets, a sharing pool word, or the auto spare-first-else-share
        # fallback. All are declarations; the startup receipt announces the plan.
        colocated = bool(set(trainer_devices) & set(rollout_devices))

        reward_mode = config.reward.device
        if config.cross_node and reward_mode == "gpu":
            raise ValueError(
                "distributed.resources.cross_node=true cannot reserve a local reward "
                "GPU: cross-node device ids are Ray budget tokens, not driver-local "
                "CUDA ordinals. Use reward.device=trainer to score on the trainer "
                "device, or an HTTP reward service.",
            )
        reward_devices = _resolve_reward_devices(
            visible_devices=visible_devices,
            trainer_devices=trainer_devices,
            rollout_devices=rollout_devices,
            reward_config=config.reward,
        )

        # Asymmetric fsdp owns the whole training world with rollout/reward on separate
        # cards, so the trainer set must be disjoint from both regardless of any
        # declared sharing. Symmetric colocated fsdp (rollout.gpu_pool=trainer) is the
        # opposite by design — each rank's rollout shares its trainer GPU, exactly like
        # ddp — so the disjoint rule does not apply to it.
        if fsdp_asymmetric:
            _validate_fsdp_trainer_disjoint(trainer_devices, rollout_devices, reward_devices)

        reward_runs_on_cpu = reward_mode == "cpu"
        # Rewards execute in-process. With no GPU reservation of their own, an active
        # reward follows the trainer's rank-local device instead of disappearing from
        # the topology — which is what makes trainer/reward sharing visible to the
        # lifecycle plan below.
        reward_uses_trainer_device = bool(
            local_reward_configured and reward_mode == "trainer" and trainer_devices
        )
        reward_execution_devices = (
            tuple(trainer_devices) if reward_uses_trainer_device else tuple(reward_devices)
        )
        reward_shared_with_rollout = bool(set(reward_execution_devices) & set(rollout_devices))
        reward_shared_with_trainer = bool(set(reward_execution_devices) & set(trainer_devices))

        # Release scheduling is derived entirely from the resolved GPU topology:
        # the plan stores the three sharing facts; handoffs and lease modes are
        # its derived views.
        lifecycle = RayLifecyclePlan(
            trainer_and_rollout_share_gpu=colocated,
            rollout_and_reward_share_gpu=reward_shared_with_rollout,
            trainer_and_reward_share_gpu=reward_shared_with_trainer,
        )
        return cls(
            visible_devices=visible_devices,
            trainer_devices=trainer_devices,
            rollout_devices=rollout_devices,
            reward_devices=reward_devices,
            reward_runs_on_cpu=reward_runs_on_cpu,
            rollout_num_engines=rollout_num_engines,
            rollout_gpus_per_engine=rollout_gpus_per_engine,
            cross_node=config.cross_node,
            lifecycle=lifecycle,
        )


_MISSING = object()


def _validate_trainer_device_count(
    strategy: str,
    trainer_devices: tuple[int, ...],
    world_size: int,
    *,
    symmetric_colocated: bool = False,
) -> None:
    """Trainer GPU-count rule, gated by ``distributed.training.strategy``.

    ``single_process`` is one process on one device (0 or 1 GPU).
    ASYMMETRIC ``fsdp`` runs one torchrun process per GPU under a single resolver,
    so the trainer device set must cover the whole training world
    (``num_nodes * gpus_per_node``). SYMMETRIC COLOCATED ``fsdp`` (and ``ddp``)
    resolve per-rank-local: each rank owns its 1 GPU, so they follow the
    single-device rule below, not the world-covering one.
    """

    if strategy == "fsdp" and not symmetric_colocated:
        if len(trainer_devices) != world_size:
            raise ValueError(
                "distributed.training.strategy=fsdp: trainer must own "
                f"num_nodes*gpus_per_node={world_size} GPUs, got "
                f"{list(trainer_devices)} (count {len(trainer_devices)}). Set "
                f"distributed.resources.trainer.num_gpus={world_size} (or matching "
                "devices).",
            )
        return
    if len(trainer_devices) > 1:
        raise ValueError(
            "distributed.resources.trainer.devices currently supports only "
            f"0 or 1 GPU for the single-process trainer, got {trainer_devices}. "
            "Set distributed.training.strategy=fsdp for multi-GPU training.",
        )


def _validate_fsdp_trainer_disjoint(
    trainer_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    reward_devices: tuple[int, ...],
) -> None:
    """fsdp trainer GPUs must not overlap rollout/reward (sharing cannot be declared)."""

    overlap_rollout = sorted(set(trainer_devices) & set(rollout_devices))
    overlap_reward = sorted(set(trainer_devices) & set(reward_devices))
    if overlap_rollout or overlap_reward:
        raise ValueError(
            "distributed.training.strategy=fsdp requires trainer GPUs disjoint from "
            "rollout and reward (each FSDP rank owns its GPU; rollout actors and "
            "the in-process reward model need separate reserved devices). "
            f"trainer={list(trainer_devices)} rollout={list(rollout_devices)} "
            f"reward={list(reward_devices)} "
            f"(overlap rollout={overlap_rollout}, reward={overlap_reward}).",
        )


def format_distributed_resource_plan(resolved: ResolvedDistributedResources) -> str:
    """Format a compact resource plan for logs and errors."""

    parts = [
        f"visible={list(resolved.visible_devices)}",
        f"trainer={list(resolved.trainer_devices)}",
        f"rollout={list(resolved.rollout_devices)}",
        f"reward={list(resolved.reward_devices)}",
        f"engines={resolved.rollout_num_engines}",
        f"gpus_per_engine={resolved.rollout_gpus_per_engine}",
        "reward_mode="
        + (
            "gpu"
            if resolved.reward_devices
            else ("cpu" if resolved.reward_runs_on_cpu else "trainer")
        ),
        f"colocated={resolved.colocated}",
        f"cross_node={resolved.cross_node}",
        f"trainer_reservation={resolved.requires_trainer_reservation}",
        # Reading the plan at a glance: rollout lease mode + which boundaries
        # release. resident=stays active, on_demand=parks at the handoff.
        f"lifecycle=rollout:{resolved.lifecycle.rollout_mode}",
        "handoff="
        f"before_train:{resolved.lifecycle.release_rollout_before_train}"
        f",before_reward:{resolved.lifecycle.release_rollout_before_reward}"
        f",trainer_before_reward:"
        f"{resolved.lifecycle.release_trainer_before_reward}"
        f",reward_after_score:{resolved.lifecycle.release_reward_after_score}",
    ]
    return "Distributed resources: " + " ".join(parts)


def _resolve_cross_node_visible_devices(
    config: DistributedResourceConfig,
) -> tuple[int, ...]:
    """Synthesise a visible-device budget for cross-node runs.

    Resolution runs before ``ray.init()`` (see ``vrl/scripts/common/online.py``),
    so the live Ray cluster cannot be queried here. Under ``cross_node`` we instead
    size the visible pool from explicit per-role GPU counts. The resulting ordinals
    are budget tokens only: trainer keeps the head-local ordinal, rollout ordinals
    are never used as real remote device ids (placement is by Ray + node, and
    ``require_actor_gpu_ids`` switches to a node-aware check).
    """

    explicit = _parse_devices(config.visible_devices)
    if explicit != "auto":
        return tuple(
            _dedupe_ints(explicit, field_name="distributed.resources.visible_devices"),
        )

    # Reward never reserves a local GPU under cross_node (rejected during
    # resolution), so only trainer and rollout contribute budget tokens.
    total = _explicit_role_gpu_count(config.trainer) + _explicit_role_gpu_count(config.rollout)
    return tuple(range(total))


def _explicit_role_gpu_count(role_config: RoleResourceConfig) -> int:
    """Return an explicit integer GPU count for a role under ``cross_node``."""

    devices = _parse_devices(role_config.devices)
    if devices != "auto":
        return len(_dedupe_ints(devices, field_name=f"{role_config.key_prefix}.devices"))

    num_gpus = _parse_num_gpus(
        role_config.num_gpus,
        field_name=f"{role_config.key_prefix}.num_gpus",
    )
    if num_gpus == "auto" or num_gpus is None:
        raise ValueError(
            "distributed.resources.cross_node=true requires an explicit integer "
            f"{role_config.key_prefix}.num_gpus (got 'auto'/null): the Ray "
            "cluster is not queryable at resolution time, so the GPU budget must be "
            "declared up front.",
        )
    if int(num_gpus) < 0:
        raise ValueError(f"{role_config.key_prefix}.num_gpus must be >= 0")
    return int(num_gpus)


def _explicit_role_devices(
    role_config: RoleResourceConfig,
    *,
    visible_devices: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Resolve an explicitly configured ``devices`` list, or None when auto.

    Shared by every role: dedupe and require a subset of the visible pool.
    Pinned ``devices`` are the authoritative count: a ``num_gpus`` merged in
    from a lower preset layer (e.g. the colocated single-GPU base) is the
    weaker count-only request and is superseded, not cross-checked — a
    mismatch error here would point at a line the experiment never wrote.
    """

    prefix = role_config.key_prefix
    explicit_devices = _parse_devices(role_config.devices)
    if explicit_devices == "auto":
        return None
    devices = tuple(_dedupe_ints(explicit_devices, field_name=f"{prefix}.devices"))
    _validate_subset(devices, visible_devices, field_name=f"{prefix}.devices")
    return devices


def _resolve_role_devices(
    *,
    visible_devices: tuple[int, ...],
    role_config: RoleResourceConfig,
    default_auto_count: int,
) -> tuple[int, ...]:
    prefix = role_config.key_prefix
    devices = _explicit_role_devices(role_config, visible_devices=visible_devices)
    if devices is not None:
        return devices

    num_gpus = _parse_num_gpus(role_config.num_gpus, field_name=f"{prefix}.num_gpus")
    count = default_auto_count if num_gpus == "auto" or num_gpus is None else int(num_gpus)
    if count < 0:
        raise ValueError(f"{prefix}.num_gpus must be >= 0")
    if count > len(visible_devices):
        raise ValueError(
            f"{prefix}.num_gpus={count} exceeds visible devices {list(visible_devices)}",
        )
    return tuple(visible_devices[:count])


def _resolve_rollout_devices(
    *,
    visible_devices: tuple[int, ...],
    trainer_devices: tuple[int, ...],
    rollout_config: RolloutResourceConfig,
) -> tuple[int, ...]:
    gpu_pool = rollout_config.gpu_pool
    devices = _explicit_role_devices(rollout_config, visible_devices=visible_devices)
    if devices is not None:
        trainer_pool = set(trainer_devices)
        if gpu_pool == "trainer" and not set(devices).issubset(trainer_pool):
            outside = sorted(set(devices) - trainer_pool)
            raise ValueError(
                "distributed.resources.rollout.gpu_pool=trainer requires every "
                "rollout device to belong to the trainer pool, but "
                f"rollout.devices={list(devices)} includes {outside} outside "
                f"trainer={list(trainer_devices)}. Devices disjoint from trainer "
                "or mixing trainer and spare GPUs are invalid; drop the explicit "
                "devices (auto pins onto the trainer pool) or use "
                "gpu_pool=dedicated.",
            )
        trainer_overlap = sorted(set(devices) & trainer_pool)
        if gpu_pool == "dedicated" and trainer_overlap:
            raise ValueError(
                "distributed.resources.rollout.gpu_pool=dedicated requires rollout "
                "devices disjoint from the trainer pool, but "
                f"rollout={list(devices)} trainer={list(trainer_devices)} "
                f"overlap={trainer_overlap}",
            )
        return devices

    if gpu_pool == "trainer":
        # Borrow the trainer GPU(s): pin rollout onto them even when spare GPUs exist
        # ("share the trainer card", not "find a free one").
        requested = rollout_config.requested_gpu_count(available_count=len(trainer_devices))
        if requested == 0:
            return ()
        if requested > len(trainer_devices):
            raise ValueError(
                "distributed.resources.rollout.gpu_pool=trainer shares the trainer "
                f"GPU(s), but rollout needs {requested} GPU(s) and trainer owns "
                f"{list(trainer_devices)}. A shared rollout cannot exceed the trainer "
                "pool; use gpu_pool=dedicated (or expose more trainer GPUs).",
            )
        return tuple(trainer_devices[:requested])

    # auto / dedicated: a pool disjoint from the trainer GPU(s). `dedicated` forbids
    # the share fallback (a spare GPU is required); `auto` is spare-first-else-share.
    excluded = set(trainer_devices)
    pool = tuple(device for device in visible_devices if device not in excluded)
    requested = rollout_config.requested_gpu_count(available_count=len(pool))
    if requested == 0:
        return ()
    if requested <= len(pool):
        return tuple(pool[:requested])
    if gpu_pool == "dedicated":
        raise ValueError(
            "distributed.resources.rollout.gpu_pool=dedicated requires spare "
            f"rollout GPUs: requested={requested}, available={len(pool)}, "
            f"trainer={list(trainer_devices)}, visible={list(visible_devices)}. "
            "Expose more GPUs, or drop gpu_pool=dedicated to allow time-sharing "
            "the trainer GPU.",
        )
    fallback = tuple(device for device in visible_devices if device in excluded)
    combined = pool + fallback
    if requested > len(combined):
        raise ValueError(
            "Not enough visible GPUs for rollout even with overlap allowed: "
            f"requested={requested}, visible={list(visible_devices)}",
        )
    return tuple(combined[:requested])


def _resolve_reward_devices(
    *,
    visible_devices: tuple[int, ...],
    trainer_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    reward_config: RewardResourceConfig,
) -> tuple[int, ...]:
    """Resolve the in-process reward reservation: () or exactly one GPU."""

    prefix = reward_config.key_prefix
    explicit = _parse_devices(reward_config.devices)
    if reward_config.device != "gpu":
        if explicit != "auto" and explicit:
            raise ValueError(
                f"{prefix}.devices={explicit} requires {prefix}.device=gpu; "
                f"device={reward_config.device!r} reserves no GPU",
            )
        return ()

    def _validated(devices: tuple[int, ...]) -> tuple[int, ...]:
        _validate_reward_overlap(
            devices=devices,
            trainer_devices=trainer_devices,
            rollout_devices=rollout_devices,
            reward_config=reward_config,
        )
        return devices

    if explicit != "auto":
        devices = tuple(_dedupe_ints(explicit, field_name=f"{prefix}.devices"))
        if len(devices) != 1:
            raise ValueError(
                f"{prefix}.device=gpu reserves exactly one GPU for the "
                f"in-process reward model, got {prefix}.devices={list(devices)}. "
                "Multi-GPU reward inference requires a remote runtime boundary.",
            )
        _validate_subset(devices, visible_devices, field_name=f"{prefix}.devices")
        return _validated(devices)

    pool_source = reward_config.gpu_pool
    if pool_source == "auto":
        # Auto placement: prefer a dedicated spare GPU when one exists;
        # otherwise fall back to sharing the rollout pool. Removes the footgun
        # where a spelled-out "rollout" kept forcing shared single-GPU churn
        # even on machines with spare GPUs.
        spare_excluded = set(trainer_devices) | set(rollout_devices)
        spare_pool = tuple(device for device in visible_devices if device not in spare_excluded)
        if spare_pool:
            return _validated((spare_pool[0],))
        pool_source = "rollout"

    if pool_source == "rollout":
        if not rollout_devices:
            raise ValueError(
                "Not enough rollout GPUs for reward shared inference pool: "
                f"requested=1, rollout={list(rollout_devices)}",
            )
        return _validated((rollout_devices[0],))

    # Explicit dedicated pool: a spare GPU is required. Auto placement may fall
    # back to sharing, but a declared dedicated pool never does.
    excluded = set(trainer_devices) | set(rollout_devices)
    pool = tuple(device for device in visible_devices if device not in excluded)
    if not pool:
        raise ValueError(
            "Not enough non-overlapping reward GPUs: requested=1, available=0, "
            f"trainer={list(trainer_devices)}, rollout={list(rollout_devices)}, "
            f"visible={list(visible_devices)}. Set "
            "distributed.resources.reward.gpu_pool=rollout for a shared "
            "inference pool (release is derived automatically), or expose a "
            "separate reward GPU.",
        )
    return _validated((pool[0],))


def _validate_reward_overlap(
    *,
    devices: tuple[int, ...],
    trainer_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    reward_config: RewardResourceConfig,
) -> None:
    device_set = set(devices)
    rollout_pool = set(rollout_devices)
    trainer_pool = set(trainer_devices)

    if reward_config.gpu_pool == "rollout" and not device_set.issubset(rollout_pool):
        outside = sorted(device_set - rollout_pool)
        raise ValueError(
            "distributed.resources.reward.gpu_pool=rollout requires every reward "
            f"device to belong to rollout={list(rollout_devices)}, but "
            f"reward={list(devices)} includes {outside} outside that pool",
        )

    rollout_overlap = sorted(device_set & rollout_pool)
    trainer_overlap = sorted(device_set & trainer_pool)
    if reward_config.gpu_pool == "dedicated" and (rollout_overlap or trainer_overlap):
        raise ValueError(
            "distributed.resources.reward.gpu_pool=dedicated requires reward "
            "devices disjoint from both trainer and rollout pools: "
            f"reward={list(devices)} trainer={list(trainer_devices)} "
            f"rollout={list(rollout_devices)}",
        )


def _parse_devices(value: Any) -> list[int] | str:
    value = to_builtin_deep(value)
    if _is_auto(value):
        return "auto"
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid device list: {value!r}") from exc
        return _parse_devices(parsed)
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(f"invalid device list: {value!r}")


def _parse_num_gpus(value: Any, *, field_name: str) -> int | str | None:
    value = to_builtin_deep(value)
    if _is_auto(value):
        return "auto"
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be int, auto, or null") from exc
    return parsed


def _parse_num_engines(value: Any, *, field_name: str) -> int | str:
    value = to_builtin_deep(value)
    if _is_auto(value):
        return "auto"
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be int or auto") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _dedupe_ints(values: list[int], *, field_name: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item < 0:
            raise ValueError(f"{field_name} cannot contain negative device ids")
        if item in seen:
            raise ValueError(f"{field_name} contains duplicate device id {item}")
        seen.add(item)
        out.append(item)
    return out


def _validate_subset(
    devices: tuple[int, ...],
    visible_devices: tuple[int, ...],
    *,
    field_name: str,
) -> None:
    missing = sorted(set(devices) - set(visible_devices))
    if missing:
        raise ValueError(
            f"{field_name} contains devices outside distributed.resources.visible_devices: "
            f"{missing} not in {list(visible_devices)}",
        )


def _auto_visible_cuda_devices() -> tuple[int, ...]:
    try:
        import torch
    except Exception:
        return ()
    try:
        if not torch.cuda.is_available():
            return ()
        return tuple(range(int(torch.cuda.device_count())))
    except Exception:
        return ()


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


__all__ = [
    "DistributedResourceConfig",
    "RayLifecyclePlan",
    "ResolvedDistributedResources",
    "format_distributed_resource_plan",
]
