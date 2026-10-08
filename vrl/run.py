"""THE single composition seam for training-run config resolution.

Every training entrypoint used to hand-wire the same choreography:
``build_configs`` -> family registry lookup -> distributed resource resolution
-> trainer device (and, for online recipes, run/generation/collector
projections). That choreography lives here, once. New entrypoints call
``ResolvedRun.from_built`` (or ``resolve_online_run``) and read fields off the
returned aggregate -- never re-wire the chain inline.

Model materialization is the same story: the hand-wired chain
``resolve_model_build`` -> ``resolve_checkpoint_model_identity`` -> bundle
build -> identity recheck lives here as ``resolve_model`` +
``ResolvedModel.materialize``.
The two stages are deliberately separate: the online recipe runs its
checkpoint-compatibility preflight between identity resolution and heavy
bundle construction, so the seam must not fuse them.

Ray worker composition follows the same ownership rule:
``ResolvedOnlineRun.ray_launch_inputs`` projects one resolved online run plus
its replay-model identity into a typed, serializable launch payload. The Ray
launcher consumes that payload and topology only; it never reinterprets trainer
schedule or model YAML.

The composers deliberately call their dependencies through the source modules
(``builders.build_configs``, ``registry.get_model_family_entry``,
``ray_resources.ResolvedDistributedResources``,
``checkpoint_identity.resolve_checkpoint_model_identity``) instead of
importing the bare names. Attribute lookup happens at call time, so tests that
stub a seam at its owning module (the established pattern in the recipe test
suites) reach the composer without patching this module too.

Deliberately NOT absorbed here: ``TrainingCheckpoint.load_for_resume`` does
checkpoint file I/O, not config resolution -- it stays in the recipes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from vrl.config import builders
from vrl.config.builders import BuiltConfigs, RewardRuntimeConfig
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RootConfig
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.models import checkpoint_identity
from vrl.models.dtypes import dtype_to_wire_name
from vrl.models.families.registry import ModelFamilyEntry
from vrl.models.interfaces import ModelBuild, RuntimeBundle
from vrl.ray import resources as ray_resources
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    """Controller-owned epoch, checkpoint cadence, and trainer RNG policy."""

    total_epochs: int
    save_freq: int = 50
    seed: int = 0
    deterministic: bool = False

    def __post_init__(self) -> None:
        # RootConfig types these fields as StrictInt/StrictBool; only the
        # ranges the schema does not express are checked here.
        if self.total_epochs < 0:
            raise ValueError("trainer.total_epochs must be >= 0")
        if self.save_freq < 0:
            raise ValueError("trainer.save_freq must be >= 0")
        if not -(2**63) <= self.seed < 2**64:
            raise ValueError("trainer.seed must fit torch's signed/unsigned 64-bit seed range")

    @classmethod
    def from_root(cls, root: RootConfig) -> OnlineRunConfig:
        trainer = root.trainer
        total_epochs = None if trainer is None else trainer.total_epochs
        if total_epochs is None:
            raise ValueError("config missing required key: trainer.total_epochs")
        values: dict[str, Any] = {"total_epochs": total_epochs}
        if trainer.save_freq is not None:
            values["save_freq"] = trainer.save_freq
        if trainer.seed is not None:
            values["seed"] = trainer.seed
        if trainer.deterministic is not None:
            values["deterministic"] = trainer.deterministic
        return cls(**values)

    def initialize_process_rng(self, *, rank: int = 0) -> None:
        """Seed model construction identically, then rank-local training streams.

        This config owns trainer RNG policy. It does not seed remote rollout or
        external reward processes. Resume restores checkpoint RNG after this call.
        """
        import os
        import random

        import numpy as np

        require_int(rank, path="training rank", minimum=0)
        if self.deterministic:
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace is None:
                if torch.cuda.is_initialized():
                    raise RuntimeError(
                        "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching deterministic training"
                    )
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            elif workspace not in {":4096:8", ":16:8"}:
                raise ValueError(
                    "deterministic training requires a supported cuBLAS workspace config"
                )
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seed = (self.seed + rank) % 2**64
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """Core resolution shared by every training entrypoint: the built configs
    and the distributed resource plan; the family and trainer device read off them."""

    built: BuiltConfigs
    resources: ResolvedDistributedResources

    @classmethod
    def from_built(cls, built: BuiltConfigs) -> ResolvedRun:
        return cls(
            built=built,
            resources=ray_resources.ResolvedDistributedResources.from_root(
                built.root,
                reward=built.reward,
            ),
        )

    @property
    def family(self) -> ModelFamilyEntry:
        return self.built.family

    @property
    def device(self) -> torch.device:
        return torch.device(self.resources.trainer_torch_device)


@dataclass(frozen=True, slots=True)
class ResolvedReward:
    """The reward engine of one run: its config, execution device and parking need.

    The reward twin of ``RayGenerationLaunchInputs``. Device and parking are
    read once off the resource plan (``from_plan``); the preflight parking check,
    the Ray actor placement and the function build all read these fields rather
    than re-deriving them.
    """

    config: RewardRuntimeConfig
    device: str
    memory_parking_required: bool

    @classmethod
    def from_plan(
        cls,
        config: RewardRuntimeConfig,
        resources: ResolvedDistributedResources,
        *,
        trainer_device: torch.device | str | None = None,
    ) -> ResolvedReward:
        """Read the execution device and parking need off the resource plan.

        The resource topology is the execution-device source of truth, so no
        caller-chosen device can contradict it. ``trainer_device`` lets torchrun
        callers supply their rank-local device for trainer-shared rewards.
        """

        return cls(
            config=config,
            device=resources.reward_torch_device(
                trainer_device=None if trainer_device is None else str(trainer_device),
            ),
            # The plan's reward devices are the reward's own GPU fact: an
            # all-HTTP reward reserves none (its services own their
            # accelerators), so the plan never parks it.
            memory_parking_required=bool(resources.lifecycle.offload_reward),
        )

    def validate_parking(self) -> None:
        """Reject a GPU-sharing reward whose components cannot park, before any build."""

        if not self.memory_parking_required:
            return
        from vrl.rewards.functions.registry import validate_reward_memory_parking_components

        validate_reward_memory_parking_components(
            tuple(self.config.weights),
            device=self.device,
            reward_kwargs=self.config.kwargs,
            inference_configs=self.config.inference_configs,
        )

    def actor_placement(self, owner: Any) -> Any:
        """Bind the reward's Ray actors to the run's acquired resource ownership.

        A GPU-sharing reward borrows the trainer's physical card; any other
        reward takes the reservation the run-level placement group made for it.
        """

        from vrl.rewards.ray import RayRewardPlacement

        if self.device.startswith("cuda") and self.memory_parking_required:
            import os

            from vrl.ray.dependencies import require_ray

            # CUDA ordinals in a driver mask are not Ray's physical GPU IDs.
            device_index = torch.device(self.device).index
            ordinal = torch.cuda.current_device() if device_index is None else device_index
            mask = os.environ.get("CUDA_VISIBLE_DEVICES")
            physical_gpu = int(mask.split(",")[ordinal]) if mask else ordinal
            return RayRewardPlacement(
                shared_gpu_id=physical_gpu,
                node_id=str(require_ray().get_runtime_context().get_node_id()),
            )
        return RayRewardPlacement(placement=owner.reward_placement)

    def build_function(self, *, ray_placement: Any = None) -> Any:
        """Build the reward function on this device under this parking policy.

        ``MultiReward.from_dict`` validates the components and constructs. A
        GPU-sharing reward parks its model memory completely; a dedicated
        reward stays resident; HTTP components own their deployment.
        """

        self.config.require_online_training()
        from vrl.rewards.functions.registry import MultiReward

        return MultiReward.from_dict(
            self.config.weights,
            device=self.device,
            reward_kwargs=self.config.kwargs,
            memory_parking_required=self.memory_parking_required,
            inference_configs=self.config.inference_configs,
            ray_placement=ray_placement,
        )


@dataclass(frozen=True, slots=True)
class ResolvedOnlineRun(ResolvedRun):
    """Online-recipe resolution: the shared core plus run/generation/collector."""

    run: OnlineRunConfig
    generation: RayGenerationConfig
    collector: RolloutCollectorConfig

    def reward_inputs(
        self,
        *,
        trainer_device: torch.device | str | None = None,
    ) -> ResolvedReward:
        """Project the resolved reward-engine construction inputs.

        ``trainer_device`` lets torchrun callers supply their rank-local device
        for trainer-shared rewards; dedicated reward GPUs come from the
        resource plan itself.
        """

        reward = self.built.reward
        if reward is None:
            raise ValueError("online recipe requires a reward section")
        return ResolvedReward.from_plan(reward, self.resources, trainer_device=trainer_device)

    def ray_launch_inputs(
        self,
        replay_model: ResolvedModel,
    ) -> RayGenerationLaunchInputs:
        """Build the serializable Ray worker contract for this online run."""

        if not isinstance(replay_model, ResolvedModel):
            raise TypeError(
                f"replay_model must be a ResolvedModel, got {type(replay_model).__name__}",
            )
        if replay_model.entry.family != self.family.family:
            raise ValueError(
                "replay model family does not match the resolved online run: "
                f"{replay_model.entry.family!r} != {self.family.family!r}",
            )
        replay_model.build.require_replay()
        trainer = self.built.trainer
        if trainer is None:
            raise ValueError("online generation launch requires a trainer config")

        generation = self.generation
        runtime_device = torch.device(
            "cuda" if generation.resources.rollout_devices else "cpu",
        )
        build = self.family.resolve_model_build(
            self.built.root,
            runtime_device,
            precision=self.built.precision,
        )
        rollout_model_identity = checkpoint_identity.resolve_checkpoint_model_identity(build)
        if rollout_model_identity != replay_model.identity:
            raise ValueError(
                "rollout model identity does not match the driver replay model "
                "identity before Ray worker launch: "
                f"replay={replay_model.identity!r}, rollout={rollout_model_identity!r}",
            )

        model_build_payload = asdict(build)
        # Family is the top-level launch identity and is restored worker-side.
        model_build_payload.pop("family", None)
        model_build_payload["device"] = str(model_build_payload["device"])
        model_build_payload["parameter_dtype"] = dtype_to_wire_name(
            model_build_payload["parameter_dtype"],
        )
        rollout = model_build_payload.get("rollout")
        if rollout is not None:
            rollout["prompt_encoder_dtype"] = dtype_to_wire_name(
                rollout["prompt_encoder_dtype"],
            )
            # Absence is the wire representation of the universal no-offload
            # default. Only a selected residency mode needs to cross Ray.
            if rollout.get("pipeline_offload_mode") == "none":
                rollout.pop("pipeline_offload_mode")

        return RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=self.family.family,
                model_build=model_build_payload,
                expected_model_identity=replay_model.identity,
                executor_kwargs=self.family.executor_kwargs(self.built.root),
                policy_version=0,
                torch_profiler={}
                if generation.torch_profiler is None
                else asdict(generation.torch_profiler),
                versioned_weight_sync=trainer.versioned_weight_sync,
                # A rollout that hands its GPUs to another role between phases
                # parks its workers' memory at the handoff (CuMem sleep).
                sleep_offload=(
                    self.resources.lifecycle.rollout_mode == "on_demand"
                    and bool(self.resources.rollout_devices)
                ),
            ),
            gatherer=self.family.new_gatherer(),
        )


def resolve_online_run(cfg: DictConfig) -> ResolvedOnlineRun:
    """Resolve everything the online recipe reads before heavy construction.

    The reward deployment is checked before any resource resolution; the
    shared core is ``ResolvedRun.from_built``; the online projections (run
    cadence, Ray generation, collector) are added on top of it.
    """

    built = builders.build_configs(cfg)
    if built.reward is not None:
        built.reward.require_online_training()
    core = ResolvedRun.from_built(built)
    return ResolvedOnlineRun(
        built=built,
        resources=core.resources,
        run=OnlineRunConfig.from_root(built.root),
        generation=RayGenerationConfig.from_root(built.root, resources=core.resources),
        collector=RolloutCollectorConfig.from_root(built.root),
    )


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Cheap model projection: family build inputs plus checkpoint identity."""

    entry: ModelFamilyEntry
    build: ModelBuild
    identity: dict[str, Any]

    def materialize(self, *, context: str, materialize_weights: bool = True) -> RuntimeBundle:
        """Build the resolved role and re-verify its checkpoint identity.

        ``materialize_weights`` is the training strategy's answer to whether
        this process needs real weights (replay only): a sharded strategy loads
        them on its primary rank and fills the other ranks' skeletons itself.
        """

        if self.build.rollout is None:
            bundle = self.entry.build_replay(
                self.build,
                materialize_weights=materialize_weights,
            )
        else:
            bundle = self.entry.build_rollout(self.build)
        loaded_identity = checkpoint_identity.resolve_checkpoint_model_identity(self.build)
        if loaded_identity != self.identity:
            raise RuntimeError(
                f"model checkpoint source changed during {context}; "
                f"before={self.identity!r}, after={loaded_identity!r}",
            )
        return bundle


def resolve_model(
    entry: ModelFamilyEntry,
    root: RootConfig,
    device: torch.device,
    *,
    precision: PrecisionPolicy,
    for_rollout: bool,
    precision_role: Literal["training", "rollout"] | None = None,
) -> ResolvedModel:
    """Project validated config into a ``ModelBuild`` and resolve its identity.

    This is the cheap stage of model materialization. Callers run their own
    preflights (checkpoint-compatibility, prompt validation) against the
    returned identity before paying for ``materialize``.
    """

    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        for_rollout=for_rollout,
        precision_role=precision_role,
    )
    identity = checkpoint_identity.resolve_checkpoint_model_identity(build)
    return ResolvedModel(entry=entry, build=build, identity=identity)


__all__ = [
    "OnlineRunConfig",
    "ResolvedModel",
    "ResolvedOnlineRun",
    "ResolvedReward",
    "ResolvedRun",
    "resolve_model",
    "resolve_online_run",
]
