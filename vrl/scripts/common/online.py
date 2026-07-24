"""Common runner skeleton for online training recipes."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from vrl.config.builders import build_configs
from vrl.config.schema import RootConfig
from vrl.config.validation import require
from vrl.families.registry import (
    get_model_family_entry,
)
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.models.interfaces import require_runtime_model
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import GlobalRayPlacementOwner, cross_node_preflight
from vrl.ray.resources import (
    format_distributed_resource_plan,
    resolve_distributed_resources,
    reward_torch_device,
    trainer_torch_device,
)
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.orchestration import validate_rollout_schedule_topology
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator_from_cfg,
    build_reward,
    validate_reward_memory_parking,
)
from vrl.trainers.activation_checkpointing import (
    enable_transformer_gradient_checkpointing,
)
from vrl.trainers.checkpointing import (
    LORA_WEIGHTS_NAME,
    AdapterExport,
    capture_rng_state,
    load_training_checkpoint_for_resume,
    prepare_metrics_csv,
    restore_rng_state,
    restore_training_checkpoint,
    save_resolved_config,
    save_training_checkpoint,
    validate_checkpoint_compatibility,
)
from vrl.trainers.data import PromptBatchSampler, load_prompt_examples_from_config
from vrl.trainers.distributed import DistributedTrainingContext, resolve_training_context
from vrl.trainers.metrics_io import (
    build_online_metric_row,
    format_online_metric_row,
    online_metric_columns,
)
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan
from vrl.trainers.strategy import build_strategy
from vrl.trainers.weight_sync import build_runtime_weight_syncer
from vrl.utils.memory import capture_host_memory, format_host_memory, log_host_memory
from vrl.utils.profiling import profile_range
from vrl.utils.stats import RolloutStats

logger = logging.getLogger(__name__)

_RAY_ADDRESS_ENV = "RAY_ADDRESS"


def _run_integer(value: object, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer (got {value!r})")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum} (got {value})")
    return value


@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    """Controller-owned epoch, checkpoint cadence, and prompt RNG policy."""

    total_epochs: int
    save_freq: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        _run_integer(self.total_epochs, path="trainer.total_epochs", minimum=0)
        _run_integer(self.save_freq, path="trainer.save_freq", minimum=0)
        _run_integer(self.seed, path="trainer.seed")

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
        return cls(**values)


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

    def shutdown(self) -> None:
        if self._closed:
            return
        if self.shutdown_on_exit:
            self.ray.shutdown()
        self._closed = True


def _initialize_ray_cluster(
    resources: Any,
    ray: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> _RayClusterSession:
    """Connect to exactly the Ray cluster selected by the resource topology.

    Single-node and per-rank-local runs always start a fresh local cluster, even
    when the host has a stale ``RAY_ADDRESS`` or another user's Ray instance.
    Cross-node runs must provide a concrete address explicitly; implicit
    ``address='auto'`` discovery is unsafe on a shared host because the most
    recently started test cluster can win the startup race.
    """

    if ray.is_initialized():
        logger.info(
            "Ray cluster session: ownership=preinitialized driver_pid=%d ray_version=%s",
            os.getpid(),
            getattr(ray, "__version__", "unknown"),
        )
        return _RayClusterSession(ray=ray, shutdown_on_exit=False)

    environment = os.environ if environ is None else environ
    if bool(resources.cross_node):
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

    # ray.init() documents that it overwrites the driver-process SIGTERM
    # handler (ray._private.worker: "This method overwrite sigterm handler of
    # the driver process"), replacing it with a bare sys.exit that skips the
    # recipe's cleanup path. Signal ownership belongs to whoever launched this
    # recipe (the vrl-train CLI installs cooperative-cancel handlers); snapshot
    # and restore around init so Ray cannot silently take it.
    import signal

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        context = ray.init(address=address)
    finally:
        for signum, handler in previous_handlers.items():
            if handler is not None:
                signal.signal(signum, handler)
    address_info = getattr(context, "address_info", None)
    if not isinstance(address_info, Mapping):
        address_info = {}
    resolved_address = address_info.get("gcs_address") or address_info.get("address") or address
    logger.info(
        "Ray cluster session: ownership=%s driver_pid=%d address=%s session_dir=%s ray_version=%s",
        ownership,
        os.getpid(),
        resolved_address,
        address_info.get("session_dir", "unknown"),
        getattr(ray, "__version__", "unknown"),
    )
    return _RayClusterSession(ray=ray, shutdown_on_exit=True)


def _require_supported_online_strategy(context: DistributedTrainingContext) -> None:
    """Fail-fast when the online recipe cannot yet drive the resolved strategy.

    Both multi-rank strategies ride the same per-rank-local symmetric-colocated
    path (SPRINT_symmetric_colocated_ddp.md / SPRINT_multi_gpu_training.md Phase 4):
    each torchrun rank runs its own local Ray + local colocated rollout on its 1
    GPU, draws a DISJOINT prompt slice, and the recipe writes IO from rank0 only.
    ``ddp`` replicates the full module and all-reduces gradients; ``fsdp`` shards
    params/grads/optimizer as DTensor and the per-layer all-gather / reduce-scatter
    crosses ranks instead. The export seam already gathers a full, unwrapped,
    policy-facing state on every rank (FSDPStrategy.export_rollout_state /
    gather_full_state_dict), so rollout weight sync and checkpointing are identical
    to single-process from the recipe's point of view. Only a genuinely unsupported
    strategy reaches the raise below.
    """

    if context.strategy not in {"single_process", "ddp", "fsdp"}:
        raise NotImplementedError(
            f"distributed.training.strategy={context.strategy!r} is not supported by "
            "run_online_recipe. Use single_process, ddp, or fsdp "
            "(SPRINT_multi_gpu_training.md / SPRINT_symmetric_colocated_ddp.md).",
        )


def _require_supported_distributed_rollout_topology(
    context: DistributedTrainingContext,
    resources: Any,
) -> None:
    """Reject multi-rank rollout ownership that the recipe cannot coordinate.

    The implemented DDP/FSDP orchestration is per-rank-local and colocated: each
    rank owns its own Ray runtime on its trainer GPU. With disjoint rollout GPUs,
    every rank would instead resolve the same global rollout plan, start or attach
    Ray independently, and launch duplicate workers onto those GPUs. Supporting
    that topology needs one explicit owner (for example rank-0 collection plus a
    broadcast), not a resource-only YAML change.
    """

    if not context.distributed:
        return
    if bool(resources.colocated) or int(resources.rollout_num_gpus) == 0:
        return
    raise NotImplementedError(
        "distributed training with disjoint rollout GPUs is not supported by "
        "run_online_recipe: every torchrun rank would independently initialize Ray "
        "and launch the same rollout device plan. Use single_process with dedicated "
        "rollout GPUs. Multi-rank DDP/FSDP needs rank-owned rollout placement or "
        "rank-0 collection/broadcast before this topology can run.",
    )


def _log_rollout_memory_plan(
    batch_plan: OnlineBatchPlan,
    *,
    generation_samples_per_chunk: int | str | None,
) -> None:
    """Log how many rollout tensors one optimizer update can hold at once."""

    prompts_per_batch = batch_plan.prompts_per_batch
    samples_per_prompt = batch_plan.n_samples_per_prompt
    target_samples = prompts_per_batch * samples_per_prompt
    replay_chunk = batch_plan.replay_samples_per_chunk

    def describe_chunk(value: Any) -> str:
        if value == "auto":
            return "auto(pending)"
        size = int(value or 0)
        return str(samples_per_prompt if size <= 0 else min(samples_per_prompt, size))

    generation_chunk_text = describe_chunk(generation_samples_per_chunk)
    replay_chunk_text = describe_chunk(replay_chunk)
    gas = batch_plan.gradient_accumulation_steps
    if batch_plan.streaming:
        microbatch_prompts = batch_plan.microbatch_size
        microbatch_samples = microbatch_prompts * samples_per_prompt
        logger.info(
            "Rollout memory plan: streaming accumulation enabled "
            "(prompts_per_batch=%d, gradient_accumulation_steps=%d, "
            "microbatch_prompts=%d, microbatch_samples=%d, "
            "generation_samples_per_chunk=%s, replay_samples_per_chunk=%s, "
            "target_samples_per_update=%d)",
            prompts_per_batch,
            gas,
            microbatch_prompts,
            microbatch_samples,
            generation_chunk_text,
            replay_chunk_text,
            target_samples,
        )
        return

    logger.info(
        "Rollout memory plan: legacy full-batch accumulation "
        "(prompts_per_batch=%d, generation_samples_per_chunk=%s, "
        "replay_samples_per_chunk=%s, "
        "target_samples_per_update=%d)",
        prompts_per_batch,
        generation_chunk_text,
        replay_chunk_text,
        target_samples,
    )
    if prompts_per_batch > 1:
        logger.warning(
            "Legacy full-batch rollout accumulation is enabled; host RAM may hold "
            "up to %d prompt groups (%d samples) before backward. Set "
            "actor.gradient_accumulation_steps to a divisor of prompts_per_batch "
            "to stream rollout microbatches and fail earlier on memory issues.",
            prompts_per_batch,
            target_samples,
        )


def _warn_global_std_streaming_divergence(
    batch_plan: OnlineBatchPlan,
    *,
    global_std: bool,
) -> None:
    """Warn when global_std advantage normalization is silently per-microbatch.

    GRPO ``global_std=true`` normalizes advantages by the std across ALL prompt
    groups in the optimizer-target batch. Streaming accumulation computes
    advantages per microbatch (collect_training_batch runs once per slice), so
    with >1 group per microbatch the std is taken over the microbatch's groups
    only -- not the full batch -- and the gradient diverges from the full-batch
    global-std intent. ``microbatch_size=1`` is exempt: one group per microbatch
    makes per-group and "global" std identical. Surfaced, not blocked, because
    keeping global_std is an experiment-owner decision.

    Same signature shape as ``_log_rollout_memory_plan``: the batch plan the
    diagnostic reasons about, plus its one value from another owner as a keyword.
    ``global_std`` belongs to the algorithm config, so the caller passes the
    typed field rather than re-reading a YAML path whose default would silently
    win if the key ever moved.
    """
    gas = batch_plan.gradient_accumulation_steps
    if not batch_plan.streaming:
        return
    if not global_std:
        return
    rbs = batch_plan.prompts_per_batch
    groups_per_microbatch = batch_plan.microbatch_size
    if groups_per_microbatch <= 1:
        return
    logger.warning(
        "algorithm.global_std=true with streaming accumulation "
        "(gradient_accumulation_steps=%d, %d prompt groups per microbatch): the "
        "global-std advantage normalization is computed per microbatch, not over "
        "the full %d-group batch, so the gradient differs from the full-batch "
        "global-std intent. Set algorithm.global_std=false (per-group std, which "
        "is streaming-equivalent), actor.microbatch_size=1 (one group per "
        "microbatch), or drop streaming to keep the full-batch global std.",
        gas,
        groups_per_microbatch,
        rbs,
    )


def _default_reference_model(bundle: Any, cfg: Any) -> Any | None:
    """Reference model for KL: the (LoRA) policy itself when use_lora and kl_coef>0, else None."""

    # Convention for config reads in this module: keys that family configs
    # legitimately omit (e.g. cosmos full-param dropped use_lora from its model
    # yaml) are read with OmegaConf.select + explicit default, which also
    # tolerates a missing parent section. Required keys keep raw attribute
    # access on purpose — a missing required key must fail loudly at startup,
    # not silently default.
    kl_coef = float(OmegaConf.select(cfg, "algorithm.kl_coef", default=0.0) or 0.0)
    if bool(OmegaConf.select(cfg, "model.use_lora", default=False)) and kl_coef > 0:
        return bundle.model
    return None


def _load_sft_latents_from_config(cfg: DictConfig, family: str) -> dict[str, Any] | None:
    """Load the clean-latents shard when the diffusion-loss regularizer is on.

    The schema cross-check already rejected sft_weight>0 without
    data.sft_latents, so this only turns a configured path into tensors (and
    fails loud on a family-mismatched or malformed shard).
    """

    weight = float(OmegaConf.select(cfg, "algorithm.sft_weight", default=0.0) or 0.0)
    if weight <= 0:
        return None
    path = OmegaConf.select(cfg, "data.sft_latents", default=None)
    if not path:
        raise ValueError("algorithm.sft_weight > 0 requires data.sft_latents")
    from vrl.trainers.data.sft_latents import load_sft_latents

    return load_sft_latents(
        str(path),
        family=family,
        model_path=str(OmegaConf.select(cfg, "model.path", default="")),
        model_revision=str(OmegaConf.select(cfg, "model.revision", default="") or ""),
    )


def _export_transformer_lora(
    bundle: Any,
    cfg: DictConfig,
) -> dict[str, AdapterExport] | None:
    """Export diffusion transformer LoRA weights when configured."""

    if not bool(OmegaConf.select(cfg, "model.use_lora", default=False)):
        return None
    exportable = {
        str(name): module
        for name, module in getattr(bundle, "trainable_modules", {}).items()
        if hasattr(module, "save_pretrained")
    }
    if len(exportable) == 1:
        return {LORA_WEIGHTS_NAME: AdapterExport(next(iter(exportable.values())))}
    if len(exportable) > 1:
        # checkpoint.pt remains the resume source of truth. Namespaced adapter
        # artifacts make each expert independently inspectable/publishable while
        # preserving the legacy lora_weights/ path for ordinary one-root models.
        return {
            f"{LORA_WEIGHTS_NAME}/{name}": AdapterExport(module)
            for name, module in exportable.items()
        }
    return None


def _export_language_model_lora(
    bundle: Any,
    cfg: DictConfig,
) -> dict[str, AdapterExport] | None:
    """Export AR language-model LoRA weights when configured."""

    if bool(OmegaConf.select(cfg, "model.use_lora", default=False)):
        return {LORA_WEIGHTS_NAME: AdapterExport(bundle.model.language_model)}
    return None


def _check_host_memory_budget(
    budget_fraction: float,
    *,
    microbatch_prompts: int,
    n_samples_per_prompt: int,
) -> None:
    """Fail fast if one streamed microbatch already pushes host RAM past budget.

    Streaming accumulation holds ~one microbatch of rollout/replay tensors at a
    time, so if system memory is already over budget right after collecting the
    first microbatch, a larger ``microbatch_size`` (or simply more
    epochs) would only OOM later in the run. Raising now — with the measured
    snapshot — turns a delayed mid-run OOM into an immediate, actionable error.
    Real RSS is measured (``capture_host_memory`` reads /proc), not estimated
    from tensor byte counts, because the Ray OOM monitor kills on RSS.
    """
    snapshot = capture_host_memory()
    used = snapshot.used_fraction
    if used is None or used <= budget_fraction:
        return
    raise MemoryError(
        f"Host RAM is at used={used:.1%} after collecting one streamed microbatch "
        f"({microbatch_prompts} prompt group(s) x {n_samples_per_prompt} samples), "
        f"above actor.host_memory_budget_fraction={budget_fraction:.1%} "
        f"({format_host_memory(snapshot)}). One microbatch already does not fit the "
        "host-RAM budget; reduce actor.microbatch_size to stream smaller "
        "slices, or lower rollout.n_samples_per_prompt / sample resolution if it is "
        "already 1.",
    )


async def _run_streaming_optimizer_update(
    trainer: OnlineTrainer,
    example_batch: list[Any],
    *,
    batch_plan: OnlineBatchPlan,
    next_example_batch: list[Any] | None = None,
) -> Any:
    """One optimizer update streamed over ``gradient_accumulation_steps`` microbatches.

    Splits the ``prompts_per_batch`` prompts into microbatches and runs
    collect -> backward -> RELEASE for each before the next, so host RAM holds
    ~one microbatch of rollout/replay tensors instead of the whole target batch
    (the memory fix that lets bigger models train on limited GPUs). One
    optimizer.step / EMA / weight-sync / metric row per update; gradients
    accumulate across microbatches with a global loss scale, so the update is
    gradient-equivalent to the legacy full-batch path.

    When ``host_memory_budget_fraction`` > 0, the first collected microbatch is
    checked against the host-RAM budget and the run fails fast if it is already
    over budget (SPRINT_memory_budgeted_microbatch T2).
    """
    if not batch_plan.streaming:
        raise ValueError("_run_streaming_optimizer_update requires a streaming batch plan")
    micro = batch_plan.microbatch_size
    microbatches = [example_batch[k : k + micro] for k in range(0, len(example_batch), micro)]
    total_groups = batch_plan.prompts_per_batch

    trainer.begin_optimizer_update()

    update_stats = RolloutStats()
    reward_mean_w = reward_std_w = adv_mean_w = adv_zero_w = adv_sat_w = 0.0
    weight_total = 0
    trained_prompt_num = 0
    group_size = float(batch_plan.n_samples_per_prompt)
    reward_component_values: dict[str, list[float]] = {}
    for mb_index, microbatch in enumerate(microbatches):
        if mb_index + 1 < len(microbatches):
            next_prompts = microbatches[mb_index + 1]
        elif next_example_batch:
            next_prompts = next_example_batch[:micro]
        else:
            next_prompts = None
        batch = await trainer.collect_training_batch(
            microbatch,
            next_prompts=next_prompts,
        )
        try:
            # Host-RAM fail-fast on the first microbatch: one slice is the host
            # peak under streaming, so if it is already over budget, stop now.
            if batch_plan.host_memory_budget_fraction > 0.0 and mb_index == 0:
                _check_host_memory_budget(
                    batch_plan.host_memory_budget_fraction,
                    microbatch_prompts=len(microbatch),
                    n_samples_per_prompt=batch_plan.n_samples_per_prompt,
                )
            await trainer.backward_on_training_batch(batch, total_groups=total_groups)
            # Sample-count-weighted aggregation of this microbatch's pre-filter stats
            # so the one metric row reflects ALL samples, not the last microbatch.
            weight = max(1, len(microbatch) * batch_plan.n_samples_per_prompt)
            reward_mean_w += batch.pre_filter_reward_mean * weight
            reward_std_w += batch.pre_filter_reward_std * weight
            adv_mean_w += batch.pre_filter_adv_mean * weight
            adv_zero_w += batch.adv_zero_rate * weight
            adv_sat_w += batch.adv_saturation * weight
            weight_total += weight
            trained_prompt_num += int(batch.trained_prompt_num)
            if batch.group_size:
                group_size = float(batch.group_size)
            for name, values in getattr(batch, "reward_components", {}).items():
                reward_component_values.setdefault(name, []).extend(values)
            update_stats.merge(trainer._step_stats(batch.iteration, batch.timer))
        finally:
            # Release this microbatch's rollout/replay tensors before the next,
            # including exception paths where traceback locals can otherwise keep
            # large batches alive longer than needed.
            del batch

    weight_total = max(1, weight_total)
    return await trainer.finish_optimizer_update(
        stats=update_stats,
        reward_mean=reward_mean_w / weight_total,
        reward_std=reward_std_w / weight_total,
        adv_mean=adv_mean_w / weight_total,
        adv_zero_rate=adv_zero_w / weight_total,
        adv_saturation=adv_sat_w / weight_total,
        group_size=group_size,
        trained_prompt_num=trained_prompt_num,
        reward_components={
            name: sum(values) / len(values)
            for name, values in reward_component_values.items()
            if values
        },
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
    component_names: tuple[str, ...]
    adapter_exports: dict[str, AdapterExport] | None
    csv_path: Path
    rng: Any
    resume_epoch: int | None
    model_identity: dict[str, Any]

    def prepare_metrics_csv(self) -> None:
        prepare_metrics_csv(
            self.csv_path,
            online_metric_columns(self.component_names),
            resume_at=("epoch", self.resume_epoch) if self.resume_epoch is not None else None,
        )

    def write_metric_row(self, epoch: int, metrics: Any) -> None:
        row = build_online_metric_row(epoch, metrics, self.component_names)
        with self.csv_path.open("a", encoding="utf-8") as handle:
            handle.write(format_online_metric_row(row))

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
                "global_step": self.trainer.state.global_step,
            },
            rng_state=capture_rng_state(prompt_generator=self.rng),
            adapter_exports=self.adapter_exports,
            export_ema=getattr(self.trainer, "_ema", None),
            model_identity=self.model_identity,
            strategy=self.strategy,
        )


def _prepare_metrics_csv_rank_consistent(
    run: OnlineRecipeRun,
    training_context: DistributedTrainingContext,
) -> None:
    """Run the rank-0 CSV preflight and propagate its verdict to every rank.

    Only rank 0 owns the output path in multi-node runs, so peers cannot safely
    inspect the header themselves. Broadcasting the small error description
    keeps every rank on the same side of the first training collective instead
    of relying on torchrun to notice and terminate a lone rank-0 exception.
    """

    if not training_context.distributed:
        run.prepare_metrics_csv()
        return

    failure: str | None = None
    if training_context.is_primary:
        try:
            run.prepare_metrics_csv()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"

    payload = [failure]
    torch.distributed.broadcast_object_list(
        payload,
        src=0,
        device=training_context.device,
    )
    if payload[0] is not None:
        raise RuntimeError(f"metrics CSV preflight failed on rank 0: {payload[0]}")


def _resolve_reference_artifacts(examples: list[Any], cfg: DictConfig) -> None:
    """Resolve manifest-relative REFERENCE paths once, at prompt load time.

    Rollout executors, family collector hooks, and rewards all consume
    reference paths; resolving the typed fields here means every consumer sees
    the same absolute path instead
    of each re-deriving it against its own data root. Rows without reference
    fields (t2v recipes) pass through untouched; required-ness stays with the
    family collector hooks. Absolute manifest paths pass through unchanged.

    TARGET fields are deliberately NOT resolved: ``target_video`` is the
    identity key into sft-latents shards (see ``PromptExample``), so rewriting
    it to a machine-local absolute path would break every shard lookup.
    """

    from vrl.utils.artifacts import coerce_data_root, resolve_artifact_path

    data_root = coerce_data_root(
        OmegaConf.select(cfg, "data.artifact_data_root", default=None),
    )
    for example in examples:
        for field_name in ("reference_image", "reference_video"):
            raw = str(getattr(example, field_name, "") or "").strip()
            if not raw:
                continue
            resolved = str(
                resolve_artifact_path(raw, data_root=data_root, allow_absolute=True),
            )
            setattr(example, field_name, resolved)
        references = list(getattr(example, "references", None) or [])
        if references:
            resolved_refs = [
                str(resolve_artifact_path(item, data_root=data_root, allow_absolute=True))
                for item in references
            ]
            example.references = resolved_refs


async def run_online_recipe(
    cfg: DictConfig,
) -> None:
    """Run a family online training job through shared recipe glue."""

    _preflight_production_video_reward(cfg)
    built = build_configs(cfg)
    run_config = OnlineRunConfig.from_root(built.root)
    if built.root.model is None:
        raise ValueError("online recipe requires model configuration")
    family_entry = get_model_family_entry(str(built.root.model.family))
    trainer_config = built.trainer
    if trainer_config is None:
        raise ValueError("online recipe cannot use an offline-only trainer config")
    batch_plan = trainer_config.batch_plan
    reward_config = built.reward
    if reward_config is None:
        raise ValueError("online recipe requires a reward section")
    _log_rollout_memory_plan(
        batch_plan,
        generation_samples_per_chunk=(
            built.root.rollout.samples_per_chunk if built.root.rollout is not None else None
        ),
    )
    _warn_global_std_streaming_divergence(
        batch_plan,
        global_std=built.algorithm.global_std,
    )
    if trainer_config.profile:
        os.environ["VRL_PROFILE"] = "1"

    resume_config = built.resume
    resume_checkpoint = load_training_checkpoint_for_resume(resume_config)
    resources = resolve_distributed_resources(cfg)
    generation_config = RayGenerationConfig.from_cfg(
        built.root,
        resources=resources,
    )
    validate_rollout_schedule_topology(trainer_config.rollout_orchestration, resources)
    validate_reward_memory_parking(resources=resources, built=built)
    logger.info(format_distributed_resource_plan(resources))
    device = torch.device(trainer_torch_device(resources))
    # Resolve the training process identity (rank/device) and fail-fast on
    # strategies the online recipe can't yet drive end-to-end, before building the
    # model / Ray runtime.
    training_context = resolve_training_context(cfg, device=device)
    _require_supported_online_strategy(training_context)
    _require_supported_distributed_rollout_topology(training_context, resources)
    # Construct the strategy before any model or Ray actor. Shared-GPU on-demand
    # execution needs complete trainer-state parking; distributed strategies must
    # reject that topology here instead of failing after expensive launch work.
    strategy = build_strategy(built.root, training_context)
    if (
        resources.lifecycle.handoff.release_rollout_before_train
        or resources.lifecycle.handoff.release_trainer_before_reward
    ):
        strategy.validate_training_state_parking()
    # Under ddp every torchrun rank owns a distinct GPU: resolve_training_context
    # returns cuda:<local_rank>, which overrides the resolver's (rank-agnostic)
    # trainer device so the trainer model, rollout, and weight sync all land on
    # this rank's card. single_process passes the resolver device straight through.
    device = training_context.device
    # Rewards execute in this driver process. A dedicated local reward reservation
    # must therefore select the reward model's actual CUDA device; cross-node reward
    # ordinals are remote budget tokens and fail here before any model is loaded.
    reward_device = reward_torch_device(
        resources,
        trainer_device=device,
    )
    if family_entry.task in {"i2v", "v2w"}:
        conditioning = OmegaConf.select(
            cfg,
            "data.preprocessing.conditioning",
            default=None,
        )
        if conditioning != "reference_image":
            raise ValueError(
                f"{family_entry.family} requires data.preprocessing.conditioning=reference_image",
            )

    replay_build = family_entry.resolve_model_build(
        built.root,
        device,
        precision=built.precision,
        for_rollout=False,
    )
    model_identity = resolve_checkpoint_model_identity(replay_build)
    validate_checkpoint_compatibility(
        resume_checkpoint,
        family=family_entry.family,
        expected_model_identity=model_identity,
        strict=resume_config.strict,
    )

    examples = load_prompt_examples_from_config(cfg.data)
    _resolve_reference_artifacts(examples, cfg)
    if family_entry.task in {"i2v", "v2w"}:
        from vrl.trainers.data.artifacts import require_reference_images

        require_reference_images(
            examples,
            manifest_path=Path(
                str(OmegaConf.select(cfg, "data.manifest", default="manifest")),
            ),
            default_reference_image=OmegaConf.select(
                cfg,
                "data.preprocessing.reference_image",
                default=None,
            ),
        )
    log_host_memory("before_trainer_bundle_build", log=logger)
    bundle = family_entry.build_replay(replay_build)
    loaded_model_identity = resolve_checkpoint_model_identity(replay_build)
    if loaded_model_identity != model_identity:
        raise RuntimeError(
            "model checkpoint source changed during replay bundle construction; "
            f"before={model_identity!r}, after={loaded_model_identity!r}",
        )
    log_host_memory("after_trainer_bundle_build", log=logger)
    if family_entry.policy_semantics.step_kind == "denoise":
        enable_transformer_gradient_checkpointing(bundle, built.root)
    model = require_runtime_model(
        bundle.model,
        owner=f"{family_entry.family}.bundle.model",
    )
    # Scheduler feeds the flow-matching evaluator when the family bundle has one.
    scheduler = getattr(bundle, "scheduler", None)

    # One run-level Ray placement group owns the trainer/reward reservations and
    # rollout bundles for the whole run. It is created after the trainer model is
    # placed and before reward/rollout construction: the rollout actor receives
    # its role placement, while the local reward model uses the reserved physical
    # device selected above.
    placement_owner = GlobalRayPlacementOwner(
        resources,
        generation_config.worker,
    )
    ray = require_ray()
    ray_session: _RayClusterSession | None = None
    collector: Any | None = None
    reward_fn: Any | None = None
    trainer: OnlineTrainer | None = None
    run_error: BaseException | None = None
    try:
        ray_session = _initialize_ray_cluster(resources, ray)
        if resources.cross_node:
            cross_node_preflight(ray, resources)
        placement_owner.create()
        collector_config = RolloutCollectorConfig.from_cfg(built.root)
        reward_fn = build_reward(
            built=built,
            resources=resources,
            device=reward_device,
        )
        algorithm_and_evaluator = build_algorithm_and_evaluator_from_cfg(
            cfg,
            family_entry=family_entry,
            built=built,
            collector_config=collector_config,
            scheduler=scheduler,
        )
        # An unreachable or wrong-identity external reward service must fail
        # here, before the expensive rollout backend launch — not after the
        # first generation batch reaches scoring.
        await reward_fn.preflight()
        collector = build_rollout_collector(
            family_entry,
            reward_fn=reward_fn,
            config=collector_config,
            lifecycle=resources.lifecycle,
        )
        generation_launcher = RayGenerationLauncher()
        log_host_memory("before_rollout_backend_build", log=logger)
        collector.set_runtime(
            generation_launcher.launch_from_cfg(
                built.root,
                precision=built.precision,
                config=generation_config,
                entry=family_entry,
                driver_bundle=bundle,
                expected_model_identity=model_identity,
                placement=placement_owner.rollout_placement,
            ),
        )
        log_host_memory("after_rollout_backend_build", log=logger)

        ref_model = (
            _default_reference_model(bundle, cfg)
            if family_entry.policy_semantics.step_kind == "denoise"
            and str(cfg.algorithm.kind) != "diffusion_nft"
            else None
        )
        # The strategy built during preflight is the single owner of trainable-state
        # export for both rollout weight sync and checkpointing. prepare_model
        # (called once in the trainer) creates any process group and wraps the
        # trainable transformer only after topology/capability validation passed.
        trainer = OnlineTrainer(
            algorithm=algorithm_and_evaluator.algorithm,
            collector=collector,
            evaluator=algorithm_and_evaluator.evaluator,
            model=model,
            ref_model=ref_model,
            weight_syncer=build_runtime_weight_syncer(
                collector.runtime,
                initial_policy_version=resume_checkpoint.next_step
                if resume_checkpoint is not None
                else None,
            ),
            # Rollout weight sync re-reads live trainable state on every push, so
            # bind the strategy export lazily instead of snapshotting once.
            sync_state_getter=lambda: strategy.export_rollout_state(bundle),
            config=trainer_config,
            device=device,
            strategy=strategy,
            sft_latents=_load_sft_latents_from_config(
                cfg,
                family_entry.family,
            ),
        )

        if resume_checkpoint is not None:
            restore_training_checkpoint(
                resume_checkpoint,
                trainer=trainer,
                bundle=bundle,
                family=family_entry.family,
                expected_model_identity=model_identity,
                strict=resume_config.strict,
            )
            logger.info(
                "Resuming from %s, start_epoch=%d",
                resume_checkpoint.checkpoint_dir,
                resume_checkpoint.next_epoch,
            )

        # Under any multi-rank strategy (ddp/fsdp) only rank0 owns run IO
        # (metrics/checkpoint/eval/resolved-config): the cross-rank collectives in
        # trainer.step (DDP grad all-reduce / FSDP all-gather+reduce-scatter) keep
        # ranks in lockstep, so rank0's gathered checkpoint is complete and a single
        # writer avoids N ranks racing the same files (on 2 servers, rank0's host).
        is_primary = training_context.is_primary
        output_dir = Path(trainer_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_primary:
            save_resolved_config(cfg, output_dir, resumed=resume_checkpoint is not None)

        component_names = tuple(reward_config.weights)

        rng = torch.Generator().manual_seed(run_config.seed)
        start_epoch = resume_checkpoint.next_epoch if resume_checkpoint is not None else 0
        if start_epoch > run_config.total_epochs:
            raise ValueError(
                "resume checkpoint starts after configured total_epochs: "
                f"start_epoch={start_epoch}, total_epochs={run_config.total_epochs}",
            )
        if resume_checkpoint is not None:
            restore_rng_state(resume_checkpoint.rng_state, prompt_generator=rng)

        adapter_exports = (
            _export_transformer_lora(bundle, cfg)
            if family_entry.policy_semantics.step_kind == "denoise"
            else _export_language_model_lora(bundle, cfg)
        )
        run = OnlineRecipeRun(
            bundle=bundle,
            trainer=trainer,
            strategy=strategy,
            family=family_entry.family,
            component_names=component_names,
            adapter_exports=adapter_exports,
            csv_path=output_dir / "metrics.csv",
            rng=rng,
            resume_epoch=(resume_checkpoint.next_epoch if resume_checkpoint is not None else None),
            model_identity=model_identity,
        )
        _prepare_metrics_csv_rank_consistent(run, training_context)

        logger.info(
            "Starting %s online recipe: epochs=%d examples=%d n=%d",
            family_entry.family,
            run_config.total_epochs,
            len(examples),
            batch_plan.n_samples_per_prompt,
        )

        rank_batch = batch_plan.prompts_per_batch
        prompt_sampler = PromptBatchSampler(
            generator=rng,
            num_examples=len(examples),
            prompts_per_rank=rank_batch,
            num_replicas=training_context.world_size,
            rank=training_context.rank,
            strategy=str(require(cfg, "data.sampler.type")),
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
                if batch_plan.streaming:
                    # Streaming accumulation: split the optimizer-target batch into
                    # microbatches collected/trained/released one at a time so host
                    # RAM does not have to hold the whole batch at once.
                    metrics = await _run_streaming_optimizer_update(
                        trainer,
                        example_batch,
                        batch_plan=batch_plan,
                        next_example_batch=next_example_batch,
                    )
                else:
                    metrics = await trainer.step(
                        example_batch,
                        next_prompts=next_example_batch,
                    )
            if is_primary:
                run.write_metric_row(epoch, metrics)

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
        raise
    finally:
        await _shutdown_online_recipe_runtime(
            rollout_schedule=(
                getattr(trainer, "rollout_schedule", None) if trainer is not None else None
            ),
            collector=collector,
            reward_fn=reward_fn,
            placement_owner=placement_owner,
            strategy=strategy,
            ray_session=ray_session,
            run_error=run_error,
        )


def _preflight_production_video_reward(cfg: DictConfig) -> None:
    """Fail fast on the driver if the production reward backend is unimportable."""

    if not bool(OmegaConf.select(cfg, "production.kling_video_reward.enabled", default=False)):
        return
    from vrl.rewards.models.kling_video_reward import preflight_kling_video_reward_backend

    try:
        preflight_kling_video_reward_backend()
    except Exception as exc:
        raise RuntimeError(
            "production.kling_video_reward requires the repo-owned Kling VideoReward "
            "inference backend under vrl/rewards/models/kling_video_reward.py.",
        ) from exc


async def _shutdown_online_recipe_runtime(
    *,
    rollout_schedule: Any | None,
    collector: Any | None,
    reward_fn: Any | None,
    placement_owner: Any | None,
    strategy: Any | None,
    ray_session: _RayClusterSession | None,
    run_error: BaseException | None,
) -> None:
    shutdown_errors: list[tuple[str, Exception]] = []

    async def _run_shutdown(
        name: str,
        target: Any,
        method_name: str = "shutdown",
        *,
        attempts: int = 1,
        call_kwargs: Mapping[str, Any] | None = None,
    ) -> bool:
        if target is None:
            return True
        shutdown = getattr(target, method_name, None)
        if not callable(shutdown):
            error = TypeError(f"{name} does not expose callable {method_name}()")
            shutdown_errors.append((name, error))
            return False
        last_error: Exception | None = None
        for _attempt in range(max(1, int(attempts))):
            try:
                result = shutdown(**dict(call_kwargs or {}))
                if inspect.isawaitable(result):
                    await result
                return True
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        shutdown_errors.append((name, last_error))
        return False

    # One owner releases the whole rollout pipeline, including reward runtimes.
    # A schedule keeps continuous teardown on its owner loop and lets strict park
    # the trainer before a shared reward pool is woken and destroyed. Construction
    # failures fall back to the collector; only a failure before collector creation
    # may shut down the standalone reward directly.
    if rollout_schedule is not None:
        pipeline_released = await _run_shutdown(
            "rollout_schedule",
            rollout_schedule,
            attempts=2,
        )
    elif collector is not None:
        pipeline_released = await _run_shutdown(
            "collector",
            collector,
            attempts=2,
        )
    else:
        pipeline_released = await _run_shutdown(
            "reward_fn",
            reward_fn,
            attempts=2,
        )
    rollout_released = pipeline_released
    reward_released = pipeline_released
    await _run_shutdown("placement_owner", placement_owner, attempts=2)

    # Distributed strategies must still destroy their process groups after a role
    # cleanup failure. Single-process cleanup receives the ownership proof and
    # abandons (rather than restores) a parked trainer when the shared GPU remains
    # unknown.
    await _run_shutdown(
        "strategy",
        strategy,
        attempts=2,
        call_kwargs={"restore_parked": rollout_released and reward_released},
    )
    # Ray is last: every handle-based actor/PG cleanup above needs the connection.
    await _run_shutdown("ray_session", ray_session, attempts=2)

    if not shutdown_errors:
        return
    for name, exc in shutdown_errors:
        logger.error(
            "%s shutdown failed during online recipe cleanup",
            name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    if run_error is None:
        name, exc = shutdown_errors[0]
        raise RuntimeError(f"{name} shutdown failed during online recipe cleanup") from exc


__all__ = [
    "run_online_recipe",
]
