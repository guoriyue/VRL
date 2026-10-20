"""Common runner skeleton for online training recipes."""

from __future__ import annotations

import gc
import inspect
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from vrl.config.builders import BuiltConfigs
from vrl.generation.ray.config import RolloutWorkerConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.models.interfaces import require_runtime_model
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import GlobalRayPlacementOwner, cross_node_preflight
from vrl.ray.resources import ResourcePlan
from vrl.rewards import RewardRuntime
from vrl.rewards.ray import RayRewardPlacement
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.collector import RolloutCollector
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.core import OwnedCollection
from vrl.rollouts.orchestration import RolloutSchedule
from vrl.run import OnlineRunConfig
from vrl.scripts.common.factory import AlgorithmEvaluatorPair
from vrl.trainers.activation_checkpointing import (
    enable_transformer_gradient_checkpointing,
)
from vrl.trainers.checkpointing import (
    AdapterExport,
    TrainingCheckpoint,
    build_adapter_exports,
    capture_rng_state,
    restore_rng_state,
    save_resolved_config,
    save_training_checkpoint,
    validate_rng_state,
)
from vrl.trainers.data.prompt_sampler import PromptBatchSampler
from vrl.trainers.data.prompts import PromptExample, load_prompt_examples_from_config
from vrl.trainers.distributed import (
    DistributedTrainingContext,
    TrainerProcessGroup,
    run_on_primary_rank,
)
from vrl.trainers.metrics_io import OnlineMetricsCSV
from vrl.trainers.online.config import OnlineBatchConfig
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.strategy import Strategy, build_strategy
from vrl.trainers.trace import TrainingRunTrace
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer
from vrl.utils.memory import HostMemoryMonitor
from vrl.utils.paths import RootedPaths
from vrl.utils.profiling import profile_range

logger = logging.getLogger(__name__)
_host_memory = HostMemoryMonitor(logger=logger)

_RAY_ADDRESS_ENV = "RAY_ADDRESS"


@dataclass(slots=True)
class _RayClusterSession:
    """Connection owned by this recipe invocation.

    A pre-initialized Ray client belongs to the embedding caller and is left
    connected. Connections opened here are always closed: for a local cluster
    ``ray.shutdown`` terminates only the processes spawned by this driver; for an
    explicitly attached cluster it disconnects this driver without stopping the
    cluster.
    """

    ray: Any
    shutdown_on_exit: bool
    _closed: bool = False

    @classmethod
    def connect(
        cls,
        ray: Any,
        *,
        cross_node: bool,
        environ: Mapping[str, str] | None = None,
        local_num_cpus: int | None = None,
        local_gpu_ids: tuple[int, ...] | None = None,
    ) -> _RayClusterSession:
        """Connect to exactly the Ray cluster selected by the resource topology.

        Single-node and per-rank-local runs always start a fresh local cluster,
        even when the host has a stale ``RAY_ADDRESS`` or another user's Ray
        instance. Cross-node runs require a concrete address; implicit
        ``address='auto'`` discovery is unsafe on a shared host.
        ``local_gpu_ids`` restricts the owned node to physical actor reservations;
        attached and pre-initialized clusters retain their operator-owned view.
        """

        if ray.is_initialized():
            logger.info(
                "Ray cluster session: ownership=preinitialized driver_pid=%d ray_version=%s",
                os.getpid(),
                getattr(ray, "__version__", "unknown"),
            )
            return cls(ray=ray, shutdown_on_exit=False)

        environment = os.environ if environ is None else environ
        if cross_node:
            address = str(environment.get(_RAY_ADDRESS_ENV, "")).strip()
            if not address or address in {"auto", "local"}:
                raise ValueError(
                    "distributed.resources.cross_node=true requires a concrete "
                    "RAY_ADDRESS (for example 10.0.0.1:6379); 'auto' and 'local' "
                    "do not identify the operator-owned multi-node cluster",
                )
            ownership = "attached"
        else:
            # ``local`` bypasses RAY_ADDRESS and Ray's latest-cluster discovery.
            address = "local"
            ownership = "owned_local"

        # Ray overwrites driver SIGTERM handling inside ray.init(). Signal
        # ownership belongs to the launcher, whose cooperative handlers must
        # survive cluster initialization so terminal cleanup still runs.
        import signal

        previous_handlers = {
            signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        init_kwargs: dict[str, Any] = {"address": address}
        original_cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        narrow_cuda_mask = ownership == "owned_local" and local_gpu_ids is not None
        if ownership == "owned_local":
            if local_num_cpus is not None:
                init_kwargs["num_cpus"] = local_num_cpus
            init_kwargs["include_dashboard"] = False
            if local_gpu_ids is not None:
                if original_cuda_mask is not None:
                    allowed = {token.strip() for token in original_cuda_mask.split(",")}
                    if any(str(gpu) not in allowed for gpu in local_gpu_ids):
                        raise ValueError(
                            "local Ray GPU IDs must remain within CUDA_VISIBLE_DEVICES"
                        )
                init_kwargs["num_gpus"] = len(local_gpu_ids)
        try:
            # Only Ray's child node inherits this mask. The driver keeps its
            # existing CUDA ordinal space, including trainer-only devices.
            if narrow_cuda_mask:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
            context = ray.init(**init_kwargs)
        finally:
            if narrow_cuda_mask:
                if original_cuda_mask is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_mask
            for signum, handler in previous_handlers.items():
                if handler is not None:
                    signal.signal(signum, handler)
        address_info = getattr(context, "address_info", None)
        if not isinstance(address_info, Mapping):
            address_info = {}
        resolved_address = (
            address_info.get("gcs_address") or address_info.get("address") or address
        )
        logger.info(
            "Ray cluster session: ownership=%s driver_pid=%d address=%s "
            "session_dir=%s ray_version=%s",
            ownership,
            os.getpid(),
            resolved_address,
            address_info.get("session_dir", "unknown"),
            getattr(ray, "__version__", "unknown"),
        )
        return cls(ray=ray, shutdown_on_exit=True)

    def shutdown(self) -> None:
        if self._closed:
            return
        if self.shutdown_on_exit:
            self.ray.shutdown()
        self._closed = True


@dataclass(slots=True)
class _OnlineRecipeLifecycle:
    """Own partial-construction state and terminal teardown for one recipe."""

    placement_owner: GlobalRayPlacementOwner
    strategy: Strategy
    process_group: TrainerProcessGroup | None = None
    ray_session: _RayClusterSession | None = None
    reward_runtime: RewardRuntime | None = None
    collector: RolloutCollector | None = None
    rollout_schedule: RolloutSchedule | None = None
    _shutdown_errors: list[tuple[str, Exception]] = field(
        default_factory=list,
        init=False,
    )
    _closed: bool = field(default=False, init=False)

    async def shutdown(self, *, run_error: BaseException | None) -> None:
        """Release every acquired role in dependency order.

        Construction may fail after any field is assigned. The lifecycle keeps
        those partial acquisitions together so terminal cleanup does not need a
        parallel seven-argument state bundle.
        """

        if self._closed:
            return
        self._shutdown_errors.clear()

        # The schedule owns the complete rollout pipeline. Before trainer
        # construction, fall back to the collector; before collector
        # construction, only the standalone reward runtime exists.
        if self.rollout_schedule is not None:
            pipeline_released = await self._shutdown_one(
                "rollout_schedule",
                self.rollout_schedule,
            )
        elif self.collector is not None:
            pipeline_released = await self._shutdown_one(
                "collector",
                self.collector,
            )
        else:
            pipeline_released = await self._shutdown_one(
                "reward_runtime",
                self.reward_runtime,
            )

        await self._shutdown_one("placement_owner", self.placement_owner)

        # A failed role cleanup leaves shared-GPU ownership unknown. The
        # strategy may not restore a parked trainer into memory whose release
        # was not proven.
        await self._shutdown_one("strategy", self.strategy, restore_parked=pipeline_released)
        # Ray follows: actor and placement cleanup above still need its client.
        await self._shutdown_one("ray_session", self.ray_session)
        # The process group is last: every role above may still coordinate
        # over it while it shuts down.
        await self._shutdown_one("process_group", self.process_group)
        self._closed = True

        if not self._shutdown_errors:
            return
        for name, exc in self._shutdown_errors:
            logger.error(
                "%s shutdown failed during online recipe cleanup",
                name,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        if run_error is None:
            name, exc = self._shutdown_errors[0]
            raise RuntimeError(f"{name} shutdown failed during online recipe cleanup") from exc

    async def _shutdown_one(self, name: str, target: Any, **shutdown_kwargs: Any) -> bool:
        """Run one terminal cleanup, recording its failure."""

        if target is None:
            return True
        try:
            result = target.shutdown(**shutdown_kwargs)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self._shutdown_errors.append((name, exc))
            return False
        return True


def _log_rollout_memory_plan(
    batch_config: OnlineBatchConfig,
    *,
    samples_per_generation_batch: int | None,
) -> None:
    """Log how many rollout tensors one optimizer update can hold at once."""

    prompts_per_batch = batch_config.prompts_per_batch
    samples_per_prompt = batch_config.n_samples_per_prompt
    target_samples = prompts_per_batch * samples_per_prompt
    replay_width = batch_config.training_microbatch_size

    def describe_batch_width(value: int | None) -> str:
        size = int(value or 0)
        return str(samples_per_prompt if size <= 0 else min(samples_per_prompt, size))

    generation_batch_text = describe_batch_width(samples_per_generation_batch)
    replay_batch_text = describe_batch_width(replay_width)
    collection_count = batch_config.collections_per_update
    if batch_config.streaming:
        collection_prompts = batch_config.prompts_per_collection
        collection_samples = collection_prompts * samples_per_prompt
        logger.info(
            "Rollout memory plan: streaming accumulation enabled "
            "(prompts_per_batch=%d, collections_per_update=%d, "
            "collection_prompts=%d, collection_samples=%d, "
            "samples_per_generation_batch=%s, training_microbatch_size=%s, "
            "target_samples_per_update=%d)",
            prompts_per_batch,
            collection_count,
            collection_prompts,
            collection_samples,
            generation_batch_text,
            replay_batch_text,
            target_samples,
        )
        return

    logger.info(
        "Rollout memory plan: legacy full-batch accumulation "
        "(prompts_per_batch=%d, samples_per_generation_batch=%s, "
        "training_microbatch_size=%s, "
        "target_samples_per_update=%d)",
        prompts_per_batch,
        generation_batch_text,
        replay_batch_text,
        target_samples,
    )
    if prompts_per_batch > 1:
        logger.warning(
            "Legacy full-batch rollout accumulation is enabled; host RAM may hold "
            "up to %d prompt groups (%d samples) before backward. Set "
            "actor.prompts_per_collection to a divisor of prompts_per_batch "
            "to stream rollout collection_batches and fail earlier on memory issues.",
            prompts_per_batch,
            target_samples,
        )


def _load_sft_latents_from_config(
    built: BuiltConfigs, *, sft_weight: float
) -> dict[str, Any] | None:
    """Load the clean-latents shard when the objective's SFT regularizer is on.

    ``sft_weight`` is the built objective's declared weight. ``vrl/config/rules.py``
    already rejected sft_weight>0 without data.sft_latents, so this only turns
    the configured path into tensors (and fails loud on a family-mismatched or
    malformed shard).
    """

    if sft_weight <= 0:
        return None
    from vrl.trainers.data.sft_latents import load_sft_latents

    model = built.root.model
    return load_sft_latents(
        str(built.root.data.sft_latents),
        family=built.family.family,
        model_path=str(model.path or ""),
        model_revision=str(model.revision or ""),
    )


@dataclass(slots=True)
class OnlineRecipeRun:
    """Execution controller for one ``run_online_recipe`` invocation.

    Owns the wired training objects and the per-run CSV/checkpoint state. These
    fields used to sit in a one-owner nested wrapper; keeping them
    here makes the controller the single source of its own checkpoint inputs.
    """

    bundle: Any
    trainer: Any
    strategy: Any
    family: str
    adapter_exports: dict[str, AdapterExport] | None
    rng: Any
    model_identity: dict[str, Any]

    metrics_csv: OnlineMetricsCSV | None = None

    def initialize_metrics(
        self,
        *,
        component_names: Sequence[str],
        resume_epoch: int | None,
    ) -> None:
        """Initialize on rank 0 and propagate errors before training collectives."""

        def initialize() -> None:
            self.metrics_csv = OnlineMetricsCSV(
                Path(self.trainer.config.output_dir),
                component_names=component_names,
                resume_epoch=resume_epoch,
            )

        run_on_primary_rank(
            self.strategy.context,
            initialize,
            description="metrics CSV preflight",
        )

    def save_checkpoint(self, path: Path, *, epoch: int) -> None:
        # Called on EVERY rank: save_training_checkpoint runs the checkpoint-state
        # gather (a collective under FSDP2) on all ranks and writes files on the
        # primary only. Adapter artifacts use a separately gathered full state
        # under EMA and never read live DTensor shards during rank0 IO. The save
        # boundary also propagates publication success/failure to every rank.
        save_training_checkpoint(
            path,
            trainer=self.trainer,
            bundle=self.bundle,
            family=self.family,
            progress={
                "completed_epoch": epoch,
                "next_epoch": epoch,
                "next_step": self.trainer.state.global_step,
                "global_step": self.trainer.state.global_step,
            },
            rng_state=capture_rng_state(prompt_generator=self.rng),
            adapter_exports=self.adapter_exports,
            export_ema=getattr(self.trainer, "_ema", None),
            model_identity=self.model_identity,
            strategy=self.strategy,
        )


async def run_online_recipe(
    cfg: DictConfig, *, prompt_items: Sequence[PromptExample | OwnedCollection] | None = None
) -> None:
    """Run a family online training job through shared recipe glue.

    ``prompt_items`` replaces the items ``cfg.data`` would load, for a caller
    above vrl whose items a manifest row cannot express (an owned collection,
    such as a multi-step edit chain). ``cfg.data`` still declares the sampler.
    No caller inside vrl passes it: the sibling agentic repo's
    ``train_edit_chains`` does, so a vrl-only reference scan must keep it.
    """

    built = BuiltConfigs.from_cfg(cfg)
    reward_config = built.reward
    if reward_config is None:
        raise ValueError("online recipe requires a reward section")
    # The reward deployment is admitted before any resource resolution.
    reward_config.require_online_training()
    resources = ResourcePlan.from_root(built.root)
    run_config = OnlineRunConfig.from_root(built.root)
    worker_config = RolloutWorkerConfig.from_root(built.root)
    collector_config = RolloutCollectorConfig.from_root(built.root)
    # Identical initialization on every rank, before any randomized model build.
    run_config.initialize_process_rng()
    family_entry = built.family
    device = torch.device(resources.trainer_torch_device)
    trainer_config = built.trainer
    if trainer_config is None:
        raise ValueError("online recipe cannot use an offline-only trainer config")
    batch_config = trainer_config.batch_config
    _log_rollout_memory_plan(
        batch_config,
        samples_per_generation_batch=(
            built.root.rollout.samples_per_generation_batch
            if built.root.rollout is not None
            else None
        ),
    )
    if batch_config.streaming and built.algorithm.global_std:
        logger.info(
            "global_std streaming: normalize all %d prompt groups before clipping/filtering; "
            "temporarily spool trajectories under trainer.output_dir and replay %d groups "
            "at a time",
            batch_config.prompts_per_batch,
            batch_config.prompts_per_collection,
        )
    if trainer_config.profile:
        os.environ["VRL_PROFILE"] = "1"

    # Config rules already refused model.lora.path beside a resume, and resolved the
    # resume policy, so the recipe only loads the raw checkpoint here. The
    # checkpoint-identity preflight below consumes it directly; the epoch/step/dir
    # fields are derived after the preflight, next to the trainer that reads them.
    resume_checkpoint = (
        TrainingCheckpoint.load(built.resume_from) if built.resume_from is not None else None
    )
    resources.validate_schedule_mode(trainer_config.rollout_orchestration.schedule_mode)
    family_entry.validate_gpus_per_engine(resources.rollout_gpus_per_engine)
    if resources.rollout_num_engines < 1:
        raise ValueError("distributed.resources.rollout.num_engines must be >= 1")
    logger.info("%s", resources)
    # Resolve the training process identity (rank/device) and fail-fast on
    # strategies the online recipe can't yet drive end-to-end, before building the
    # model / Ray runtime.
    training_context = DistributedTrainingContext.from_root(built.root, device=device)
    # The implemented DDP/FSDP orchestration is per-rank-local and colocated: each
    # rank owns its own Ray runtime on its trainer GPU. With disjoint rollout GPUs,
    # every rank would instead resolve the same global rollout plan, start or attach
    # Ray independently, and launch duplicate workers onto those GPUs. Supporting
    # that topology needs one explicit owner (for example rank-0 collection plus a
    # broadcast), not a resource-only YAML change.
    if (
        training_context.distributed
        and not bool(resources.colocated)
        and int(resources.rollout_num_gpus) != 0
    ):
        raise NotImplementedError(
            "distributed training with disjoint rollout GPUs is not supported by "
            "run_online_recipe: every torchrun rank would independently initialize Ray "
            "and launch the same rollout device plan. Use single_process with dedicated "
            "rollout GPUs. Multi-rank DDP/FSDP needs rank-owned rollout placement or "
            "rank-0 collection/broadcast before this topology can run.",
        )
    # Construct the strategy before any model or Ray actor. Shared-GPU on-demand
    # execution needs complete trainer-state parking; distributed strategies must
    # reject that topology here instead of failing after expensive launch work.
    strategy = build_strategy(built.root, training_context)
    if resources.lifecycle.offload_train:
        strategy.validate_training_state_parking()
    # Under ddp every torchrun rank owns a distinct GPU: DistributedTrainingContext.from_root
    # returns cuda:<local_rank>, which overrides the resolver's (rank-agnostic)
    # trainer device so the trainer model, rollout, and weight sync all land on
    # this rank's card. single_process passes the resolver device straight through.
    device = training_context.device
    # Resolve device intent before actor construction. Dedicated Ray placement
    # maps its reservation to an actor-local GPU; shared actors borrow this
    # trainer's physical device under the topology's parking lease.
    reward_device = resources.reward_torch_device(trainer_device=str(device))
    reward_parking = resources.reward_parking_mechanism
    # A GPU-sharing reward must be able to park; checked before any model or Ray work.
    if reward_parking is not None:
        reward_config.validate_memory_parking(reward_device)
    data_config = built.root.data
    if family_entry.task in {"i2v", "v2w"}:
        preprocessing = data_config.preprocessing if data_config is not None else None
        conditioning = preprocessing.conditioning if preprocessing is not None else None
        if conditioning != "reference_image":
            raise ValueError(
                f"{family_entry.family} requires data.preprocessing.conditioning=reference_image",
            )

    replay_build = family_entry.resolve_model_build(
        built.root, device, precision=built.precision, for_rollout=False
    )
    model_identity = replay_build.checkpoint_identity()
    if resume_checkpoint is not None:
        resume_checkpoint.validate_compatibility(
            family=family_entry.family,
            expected_model_identity=model_identity,
        )
        # Every process checks every rank before model/Ray construction so a
        # missing peer stream cannot leave other ranks starting expensive work.
        for rank in range(training_context.world_size):
            validate_rng_state(
                resume_checkpoint.rng_state,
                rank=rank,
                world_size=training_context.world_size,
                generator_names=("prompt_generator",),
            )

    examples = (
        list(prompt_items)
        if prompt_items is not None
        else load_prompt_examples_from_config(data_config)
    )
    roots = RootedPaths.data_root(
        data_config.artifact_data_root if data_config is not None else None, allow_absolute=True
    )
    # An owned collection (see ``OwnedCollection``) resolved its own paths.
    examples = [
        example
        if isinstance(example, OwnedCollection)
        else example.with_resolved_references(roots)
        for example in examples
    ]
    if family_entry.task in {"i2v", "v2w"}:
        manifest = Path(str(data_config.manifest or "manifest"))
        default_reference_image = (
            data_config.preprocessing.reference_image if data_config.preprocessing else None
        )
        examples = [
            example.with_required_reference_images(
                manifest=manifest,
                row_index=row_index,
                default_reference_image=default_reference_image,
            )
            for row_index, example in enumerate(examples)
        ]
    # Derive the per-rank resume verdict the trainer reads below. Kept
    # after the checkpoint-identity preflight so an incompatible checkpoint fails
    # fast before we start reading its epoch/step/dir fields.
    resumed = resume_checkpoint is not None
    resume_epoch = resume_checkpoint.next_epoch if resume_checkpoint is not None else None
    resume_dir = resume_checkpoint.checkpoint_dir if resume_checkpoint is not None else None

    _host_memory.log("before_trainer_bundle_build")
    bundle = family_entry.build_replay(
        replace(replay_build, materialize_weights=strategy.materialize_weights),
        expected_identity=model_identity,
    )
    _host_memory.log("after_trainer_bundle_build")
    enable_transformer_gradient_checkpointing(bundle, built.root)
    model = require_runtime_model(
        bundle.model,
        owner=f"{family_entry.family}.bundle.model",
    )
    # Scheduler feeds the flow-matching evaluator when the family bundle has one.
    scheduler = getattr(bundle, "scheduler", None)

    # One run-level Ray placement group owns the trainer/reward reservations and
    # rollout bundles for the whole run. Both actor roles receive their resolved
    # ownership before construction; shared rewards borrow the trainer/rollout
    # GPU only through the phase parking lease.
    placement_owner = GlobalRayPlacementOwner(resources, worker_config)
    ray = require_ray()
    lifecycle = _OnlineRecipeLifecycle(
        placement_owner=placement_owner,
        strategy=strategy,
    )
    run_error: BaseException | None = None
    try:
        # Preflight is over: this process owns its training process group from
        # here until the lifecycle shuts it down after every other role.
        lifecycle.process_group = TrainerProcessGroup.open(
            training_context, output_dir=trainer_config.output_dir
        )
        lifecycle.ray_session = _RayClusterSession.connect(
            ray,
            cross_node=resources.cross_node,
            local_num_cpus=placement_owner.required_local_cluster_cpus(),
            local_gpu_ids=tuple(
                gpu for gpu in placement_owner.layout.bundle_gpu_ids if gpu is not None
            ),
        )
        if resources.cross_node:
            cross_node_preflight(ray, resources)
        placement_owner.create()
        if reward_parking is not None and reward_device.startswith("cuda"):
            # A parking reward sits on another role's card and borrows it in place.
            reward_placement = RayRewardPlacement.on_shared_gpu(reward_device)
        else:
            reward_placement = RayRewardPlacement(placement=placement_owner.reward_placement)
        reward_runtime = lifecycle.reward_runtime = RewardFunctionRuntime(
            reward_config.build_function(
                device=reward_device,
                parking=reward_parking,
                ray_placement=reward_placement,
            ),
        )
        algorithm_and_evaluator = AlgorithmEvaluatorPair.from_built(
            built,
            scheduler=scheduler,
        )
        # An unreachable or wrong-identity external reward service must fail
        # here, before the expensive rollout backend launch — not after the
        # first generation batch reaches scoring.
        await reward_runtime.preflight()
        collector = lifecycle.collector = RolloutCollector.from_family(
            family_entry,
            reward_runtime=reward_runtime,
            config=collector_config,
            lifecycle=resources.lifecycle,
        )
        generation_launcher = RayGenerationLauncher()
        generation_launch_inputs = RayGenerationLaunchInputs.from_run(
            built, resources, replay_build, model_identity
        )
        _host_memory.log("before_rollout_backend_build")
        generation_runtime = generation_launcher.create_runtime(
            resources,
            worker_config,
            generation_launch_inputs,
            placement=placement_owner.rollout_placement,
        )
        collector.set_generation_runtime(generation_runtime)
        _host_memory.log("after_rollout_backend_build")

        algorithm = algorithm_and_evaluator.algorithm
        # The strategy built during preflight is the single owner of trainable-state
        # export for both rollout weight sync and checkpointing. prepare_model
        # (called once in the trainer) creates any process group and wraps the
        # trainable transformer only after topology/capability validation passed.
        trainer = OnlineTrainer(
            algorithm=algorithm_and_evaluator.algorithm,
            collector=collector,
            evaluator=algorithm_and_evaluator.evaluator,
            model=model,
            weight_syncer=RayRuntimeWeightSyncer(collector.generation_runtime),
            # Rollout weight sync re-reads live trainable state on every push, so
            # bind the strategy export lazily instead of snapshotting once.
            sync_state_getter=lambda: strategy.export_rollout_state(bundle),
            config=trainer_config,
            strategy=strategy,
            sft_latents=_load_sft_latents_from_config(built, sft_weight=algorithm.sft_weight),
        )
        lifecycle.rollout_schedule = trainer.rollout_schedule

        if resume_checkpoint is not None:
            resume_checkpoint.restore_training(
                trainer=trainer,
                bundle=bundle,
                family=family_entry.family,
                expected_model_identity=model_identity,
            )
            logger.info(
                "Resuming from %s, start_epoch=%d",
                resume_dir,
                resume_epoch,
            )

        # Under any multi-rank strategy (ddp/fsdp) only rank0 owns run IO
        # (metrics/checkpoint/eval/resolved-config): the cross-rank collectives in
        # trainer.step (DDP grad all-reduce / FSDP all-gather+reduce-scatter) keep
        # ranks in lockstep, so rank0's gathered checkpoint is complete and a single
        # writer avoids N ranks racing the same files (on 2 servers, rank0's host).
        is_primary = training_context.is_primary
        output_dir = Path(trainer_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def prepare_launch_files() -> None:
            save_resolved_config(cfg, output_dir, resumed=resumed)
            run_trace = TrainingRunTrace.capture(
                cfg,
                output_dir,
                model_identity=model_identity,
                resumed=resumed,
            )
            logger.info("Training launch evidence: %s", run_trace.launch_path)

        run_on_primary_rank(
            training_context, prepare_launch_files, description="training launch evidence"
        )

        component_names = tuple(reward_config.weights)

        # Isolate training randomness from model/reward construction, while
        # giving data-parallel ranks distinct reproducible streams. Checkpoint
        # restoration below overrides this for resumed runs.
        run_config.initialize_process_rng(rank=training_context.rank)
        rng = torch.Generator().manual_seed(run_config.seed)
        start_epoch = resume_epoch if resume_epoch is not None else 0
        if start_epoch > run_config.total_epochs:
            raise ValueError(
                "resume checkpoint starts after configured total_epochs: "
                f"start_epoch={start_epoch}, total_epochs={run_config.total_epochs}",
            )
        if resume_checkpoint is not None:
            restore_rng_state(
                resume_checkpoint.rng_state,
                rank=training_context.rank,
                world_size=training_context.world_size,
                prompt_generator=rng,
            )
            # A full-param checkpoint is ~20 GB. Each torchrun rank loads its own
            # CPU payload, so retaining these dicts while three colocated rollout
            # models park on CPU exceeds this host's Ray memory threshold. All
            # consumers above have restored/captured their values; clear the
            # mutable payload before rollout actors load, then return pages now.
            resume_checkpoint.payload.clear()
            resume_checkpoint = None
            gc.collect()
            _host_memory.log("after_resume_checkpoint_release")

        adapter_exports = build_adapter_exports(
            bundle,
            use_lora=bool(built.root.model.use_lora),
        )
        run = OnlineRecipeRun(
            bundle=bundle,
            trainer=trainer,
            strategy=strategy,
            family=family_entry.family,
            adapter_exports=adapter_exports,
            rng=rng,
            model_identity=model_identity,
        )
        run.initialize_metrics(
            component_names=component_names,
            resume_epoch=resume_epoch,
        )

        logger.info(
            "Starting %s online recipe: epochs=%d examples=%d n=%d",
            family_entry.family,
            run_config.total_epochs,
            len(examples),
            batch_config.n_samples_per_prompt,
        )

        rank_batch = batch_config.prompts_per_batch
        prompt_sampler = PromptBatchSampler(
            generator=rng,
            num_examples=len(examples),
            prompts_per_rank=rank_batch,
            # CP peers must draw identical prompts: they replay one sample set.
            num_replicas=training_context.dp_size,
            rank=training_context.dp_rank,
            strategy=str(data_config.sampler.type),
        )
        for epoch in range(start_epoch, run_config.total_epochs):
            indices = prompt_sampler.sample(epoch=epoch)
            example_batch = [examples[i] for i in indices]
            next_example_batch: list[Any] | None = None
            if epoch + 1 < run_config.total_epochs:
                next_indices = prompt_sampler.preview(epoch=epoch + 1)
                next_example_batch = [examples[i] for i in next_indices]
            # This wall range deliberately encloses collect, replay/backward,
            # optimizer step, and the post-step rollout weight sync. It is the
            # denominator for update-level barrier attribution in nsys traces.
            with profile_range("trainer.optimizer_update"):
                metrics = await trainer.step(example_batch, next_prompts=next_example_batch)
            if is_primary:
                assert run.metrics_csv is not None
                run.metrics_csv.append(epoch, metrics)

            # Checkpoint on EVERY rank (NOT gated by is_primary): the trainable-state
            # export inside is a collective under FSDP2 (all ranks all-gather), and
            # save_checkpoint writes files on the primary only. Gating the call to
            # rank0 deadlocks FSDP (rank0 waits at the gather for peers that skipped).
            if run_config.save_freq > 0 and (epoch + 1) % run_config.save_freq == 0:
                run.save_checkpoint(output_dir / f"checkpoint-{epoch + 1}", epoch=epoch + 1)

        # Final checkpoint on EVERY rank too (collective gather inside; rank0 writes).
        run.save_checkpoint(
            output_dir / "checkpoint-final",
            epoch=run_config.total_epochs,
        )
        if is_primary:
            logger.info("Training complete. Final checkpoint: %s", output_dir / "checkpoint-final")
    except BaseException as exc:
        run_error = exc
        # Log BEFORE lifecycle.shutdown: shutdown crosses collective barriers
        # (park_training_state -> all_ranks_succeeded), so on a multi-rank run
        # a peer still inside the update path deadlocks the barrier and the
        # re-raise below never prints. Without this line the failing rank's
        # traceback dies with the NCCL watchdog SIGABRT (observed on the hpsv3
        # fsdp 4-rank smoke, 2026-08-16).
        logger.exception("online recipe failed before shutdown: %r", exc)
        raise
    finally:
        await lifecycle.shutdown(run_error=run_error)


__all__ = [
    "run_online_recipe",
]
