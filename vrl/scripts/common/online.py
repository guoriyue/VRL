"""Common runner skeleton for online training recipes."""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from vrl.config.builders import build_configs
from vrl.config.precision import resolve_precision_policy
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.interfaces import require_runtime_model
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import GlobalRayPlacementOwner
from vrl.ray.resources import (
    format_distributed_resource_plan,
    resolve_distributed_resources,
    trainer_torch_device,
)
from vrl.rollouts.families import build_ray_generation_inputs_for_family
from vrl.scripts.common.factory import (
    build_collector_from_cfg,
    build_online_recipe_components,
)
from vrl.scripts.common.types import (
    OnlineRecipeDefinition,
    OnlineRecipeStack,
    RecipeDeviceContext,
)
from vrl.trainers.checkpointing import (
    LORA_WEIGHTS_NAME,
    capture_rng_state,
    load_training_checkpoint_from_config,
    prepare_metrics_csv,
    prepare_model_config_for_training_resume,
    restore_rng_state,
    restore_training_checkpoint,
    sample_prompt_indices,
    save_resolved_config,
    save_training_checkpoint,
)
from vrl.trainers.data import load_prompt_examples_from_config
from vrl.trainers.distributed import assert_strategy_executable, resolve_training_context
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.precision import torch_dtype_for_trainer_precision
from vrl.trainers.strategy import SingleProcessStrategy
from vrl.trainers.weight_sync import (
    build_runtime_weight_syncer,
    build_trainable_state_sync_getter,
)
from vrl.utils.memory import capture_host_memory, format_host_memory, log_host_memory

logger = logging.getLogger(__name__)


def _apply_precision_policy(cfg: DictConfig, trainer_config: Any) -> None:
    """Bridge public ``precision:`` onto trainer fields.

    Public config exposes one forward dtype. Internally we still report compute
    and rollout separately so the precision guard can prove both sides stayed
    aligned.
    """

    policy = resolve_precision_policy(cfg)
    trainer_config.mixed_precision = policy.compute
    trainer_config.bf16 = policy.compute == "bf16"
    # Expose rollout separately for debug/guard records even though public config
    # derives it from the same forward precision as replay compute.
    trainer_config.rollout_precision = policy.rollout
    trainer_config.math_precision = policy.math


def _log_rollout_memory_plan(trainer_config: Any) -> None:
    """Log how many rollout tensors one optimizer update can hold at once."""

    rollout_batch_size = int(trainer_config.rollout_batch_size)
    samples_per_prompt = int(trainer_config.n_samples_per_prompt)
    target_samples = rollout_batch_size * samples_per_prompt
    gas = int(getattr(trainer_config, "gradient_accumulation_steps", 0))
    if gas > 0:
        microbatch_prompts = rollout_batch_size // gas
        microbatch_samples = microbatch_prompts * samples_per_prompt
        logger.info(
            "Rollout memory plan: streaming accumulation enabled "
            "(rollout_batch_size=%d, gradient_accumulation_steps=%d, "
            "microbatch_prompts=%d, microbatch_samples=%d, "
            "target_samples_per_update=%d)",
            rollout_batch_size,
            gas,
            microbatch_prompts,
            microbatch_samples,
            target_samples,
        )
        return

    logger.info(
        "Rollout memory plan: legacy full-batch accumulation "
        "(rollout_batch_size=%d, target_samples_per_update=%d)",
        rollout_batch_size,
        target_samples,
    )
    if rollout_batch_size > 1:
        logger.warning(
            "Legacy full-batch rollout accumulation is enabled; host RAM may hold "
            "up to %d prompt groups (%d samples) before backward. Set "
            "actor.gradient_accumulation_steps to a divisor of rollout_batch_size "
            "to stream rollout microbatches and fail earlier on memory issues.",
            rollout_batch_size,
            target_samples,
        )


def _warn_global_std_streaming_divergence(cfg: Any, trainer_config: Any) -> None:
    """Warn when global_std advantage normalization is silently per-microbatch.

    GRPO ``global_std=true`` normalizes advantages by the std across ALL prompt
    groups in the optimizer-target batch. Streaming accumulation computes
    advantages per microbatch (collect_training_batch runs once per slice), so
    with >1 group per microbatch the std is taken over the microbatch's groups
    only -- not the full batch -- and the gradient diverges from the full-batch
    global-std intent. ``microbatch_size=1`` is exempt: one group per microbatch
    makes per-group and "global" std identical. Surfaced, not blocked, because
    keeping global_std is an experiment-owner decision.
    """
    gas = int(getattr(trainer_config, "gradient_accumulation_steps", 0))
    if gas <= 0:
        return
    if not bool(OmegaConf.select(cfg, "algorithm.global_std", default=False)):
        return
    rbs = int(trainer_config.rollout_batch_size)
    groups_per_microbatch = rbs // gas
    if groups_per_microbatch <= 1:
        return
    logger.warning(
        "algorithm.global_std=true with streaming accumulation "
        "(gradient_accumulation_steps=%d, %d prompt groups per microbatch): the "
        "global-std advantage normalization is computed per microbatch, not over "
        "the full %d-group batch, so the gradient differs from the full-batch "
        "global-std intent. Set algorithm.global_std=false (per-group std, which "
        "is streaming-equivalent), rollout.microbatch_size=1 (one group per "
        "microbatch), or drop streaming to keep the full-batch global std.",
        gas,
        groups_per_microbatch,
        rbs,
    )


def default_reference_model(bundle: Any, cfg: Any) -> Any | None:
    """Reference model for KL: the (LoRA) policy itself when use_lora and init_kl_coef>0, else None."""

    # Convention for config reads in this module: keys that family configs
    # legitimately omit (e.g. cosmos full-param dropped use_lora from its model
    # yaml) are read with OmegaConf.select + explicit default, which also
    # tolerates a missing parent section. Required keys keep raw attribute
    # access on purpose — a missing required key must fail loudly at startup,
    # not silently default.
    init_kl_coef = float(OmegaConf.select(cfg, "algorithm.init_kl_coef", default=0.0) or 0.0)
    if bool(OmegaConf.select(cfg, "model.use_lora", default=False)) and init_kl_coef > 0:
        return bundle.model
    return None


def export_transformer_lora(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    """Export diffusion transformer LoRA weights when configured."""

    if not bool(OmegaConf.select(cfg, "model.use_lora", default=False)):
        return None
    exportable = [
        module
        for module in getattr(bundle, "trainable_modules", {}).values()
        if hasattr(module, "save_pretrained")
    ]
    if len(exportable) == 1:
        return {LORA_WEIGHTS_NAME: exportable[0]}
    if len(exportable) > 1:
        raise ValueError(
            "export_transformer_lora only supports one exportable transformer; "
            "set model.trainable_transformers to a single module before exporting",
        )
    return None


def export_language_model_lora(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    """Export AR language-model LoRA weights when configured."""

    if bool(OmegaConf.select(cfg, "model.use_lora", default=False)):
        return {LORA_WEIGHTS_NAME: bundle.model.language_model}
    return None


def enable_transformer_gradient_checkpointing(
    bundle: Any,
    cfg: DictConfig,
    *,
    require_method: bool = True,
) -> None:
    """Enable transformer gradient checkpointing while preserving family policy."""

    from vrl.trainers.core.types import TrainerConfig

    # Optional key: base yaml no longer restates the dataclass default, so an
    # absent key means "use the TrainerConfig default" — derived, not copied.
    enabled = OmegaConf.select(cfg, "actor.gradient_checkpointing")
    if enabled is None:
        enabled = TrainerConfig.__dataclass_fields__["gradient_checkpointing"].default
    if not bool(enabled):
        return

    trainable_modules = getattr(bundle, "trainable_modules", None) or {
        "transformer": bundle.model.transformer,
    }
    for name, module in trainable_modules.items():
        enable = getattr(module, "enable_gradient_checkpointing", None)
        if enable is None:
            if require_method:
                raise AttributeError(
                    f"trainable module {name!r} does not expose enable_gradient_checkpointing",
                )
            continue
        enable()


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
        f"above rollout.host_memory_budget_fraction={budget_fraction:.1%} "
        f"({format_host_memory(snapshot)}). One microbatch already does not fit the "
        "host-RAM budget; reduce rollout.microbatch_size to stream smaller "
        "slices, or lower rollout.n_samples_per_prompt / sample resolution if it is "
        "already 1.",
    )


async def _run_streaming_optimizer_update(
    trainer: OnlineTrainer,
    reward_fn: Any,
    example_batch: list[Any],
    *,
    gradient_accumulation_steps: int,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    host_memory_budget_fraction: float = 0.0,
) -> Any:
    """One optimizer update streamed over ``gradient_accumulation_steps`` microbatches.

    Splits the ``rollout_batch_size`` prompts into microbatches and runs
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
    micro = rollout_batch_size // gradient_accumulation_steps
    microbatches = [
        example_batch[k : k + micro] for k in range(0, len(example_batch), micro)
    ]
    total_groups = int(rollout_batch_size)

    reward_fn.reset_components()
    trainer.begin_optimizer_update()

    phase_times: dict[str, float] = {}
    reward_mean_w = reward_std_w = adv_mean_w = adv_zero_w = adv_sat_w = 0.0
    weight_total = 0
    trained_prompt_num = 0
    group_size = float(n_samples_per_prompt)
    for mb_index, microbatch in enumerate(microbatches):
        batch = await trainer.collect_training_batch(microbatch)
        try:
            # Host-RAM fail-fast on the first microbatch: one slice is the host
            # peak under streaming, so if it is already over budget, stop now.
            if host_memory_budget_fraction > 0.0 and mb_index == 0:
                _check_host_memory_budget(
                    host_memory_budget_fraction,
                    microbatch_prompts=len(microbatch),
                    n_samples_per_prompt=n_samples_per_prompt,
                )
            await trainer.backward_on_training_batch(batch, total_groups=total_groups)
            # Sample-count-weighted aggregation of this microbatch's pre-filter stats
            # so the one metric row reflects ALL samples, not the last microbatch.
            weight = max(1, len(microbatch) * n_samples_per_prompt)
            reward_mean_w += batch.pre_filter_reward_mean * weight
            reward_std_w += batch.pre_filter_reward_std * weight
            adv_mean_w += batch.pre_filter_adv_mean * weight
            adv_zero_w += batch.adv_zero_rate * weight
            adv_sat_w += batch.adv_saturation * weight
            weight_total += weight
            trained_prompt_num += int(batch.trained_prompt_num)
            if batch.group_size:
                group_size = float(batch.group_size)
            mb_phase = trainer._step_stats(batch.iteration, batch.timer).as_phase_dict()
            for key, value in mb_phase.items():
                phase_times[key] = phase_times.get(key, 0.0) + value
        finally:
            # Release this microbatch's rollout/replay tensors before the next,
            # including exception paths where traceback locals can otherwise keep
            # large batches alive longer than needed.
            del batch

    weight_total = max(1, weight_total)
    return await trainer.finish_optimizer_update(
        phase_times=phase_times,
        reward_mean=reward_mean_w / weight_total,
        reward_std=reward_std_w / weight_total,
        adv_mean=adv_mean_w / weight_total,
        adv_zero_rate=adv_zero_w / weight_total,
        adv_saturation=adv_sat_w / weight_total,
        group_size=group_size,
        trained_prompt_num=trained_prompt_num,
    )


async def run_online_recipe(
    cfg: DictConfig,
    definition: OnlineRecipeDefinition,
) -> None:
    """Run a family online training job through shared recipe glue."""

    _preflight_production_video_reward(cfg)
    built = build_configs(cfg)
    trainer_config = built["trainer"]
    if definition.configure_trainer is not None:
        definition.configure_trainer(cfg, trainer_config)
    _apply_precision_policy(cfg, trainer_config)
    _log_rollout_memory_plan(trainer_config)
    _warn_global_std_streaming_divergence(cfg, trainer_config)
    gradient_accumulation_steps = int(getattr(trainer_config, "gradient_accumulation_steps", 0))
    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    resume_checkpoint = load_training_checkpoint_from_config(cfg)
    prepare_model_config_for_training_resume(
        cfg,
        resume_checkpoint,
        strict=trainer_config.resume_strict,
    )

    resources = resolve_distributed_resources(cfg)
    logger.info(format_distributed_resource_plan(resources))
    device = torch.device(trainer_torch_device(resources))
    # Resolve the training process identity (rank/device) and fail-fast on
    # not-yet-implemented strategies before building the model / Ray runtime.
    training_context = resolve_training_context(cfg, device=device)
    assert_strategy_executable(training_context)
    # Replay/training model storage follows ``compute`` (via trainer_config);
    # the generation (rollout) model can use a different ``rollout`` dtype.
    weight_dtype = (
        definition.weight_dtype_getter(cfg, trainer_config, torch)
        if definition.weight_dtype_getter is not None
        else torch_dtype_for_trainer_precision(trainer_config, torch)
    )
    rollout_weight_dtype = resolve_torch_dtype(resolve_precision_policy(cfg).rollout)
    context = RecipeDeviceContext(
        device=device,
        weight_dtype=weight_dtype,
        distributed_resources=resources,
    )
    examples = load_prompt_examples_from_config(cfg.data)

    bundle_builder = definition.build_replay_bundle or definition.build_bundle
    log_host_memory("before_trainer_bundle_build", log=logger)
    bundle = bundle_builder(cfg, context.device, context.weight_dtype)
    log_host_memory("after_trainer_bundle_build", log=logger)
    if definition.after_bundle_built is not None:
        definition.after_bundle_built(bundle, cfg)
    model = require_runtime_model(
        definition.model_getter(bundle),
        owner=f"{definition.family}.model_getter",
    )
    scheduler = definition.scheduler_getter(bundle)

    # One run-level Ray placement group owns trainer/rollout/reward bundles for
    # the whole run. Created after the trainer model is on its GPU (so Ray init
    # keeps following model placement) and before reward/rollout are built, so
    # both receive owner-managed placement into the same group instead of each
    # building (and per-epoch rebuilding) its own.
    placement_owner = GlobalRayPlacementOwner(
        resources,
        rollout_cpus_per_worker=float(
            OmegaConf.select(cfg, "distributed.rollout.cpus_per_worker", default=1.0),
        ),
    )
    ray = require_ray()
    if not ray.is_initialized():
        ray.init()
    if resources.cross_node:
        from vrl.generation.ray.launcher import _cross_node_preflight

        _cross_node_preflight(ray, resources)

    collector: Any | None = None
    reward_fn: Any | None = None
    run_error: BaseException | None = None
    try:
        placement_owner.create()
        components = build_online_recipe_components(
            cfg,
            family=definition.family,
            device=str(device),
            scheduler=scheduler,
            built=built,
            reward_placement=placement_owner.reward_placement,
        )
        reward_fn = components.reward_fn
        rollout_executor_kwargs = (
            definition.collector_kwargs_getter(cfg, examples)
            if definition.collector_kwargs_getter is not None
            else {}
        )
        collector = build_collector_from_cfg(
            cfg,
            family=components.family_entry,
            model=model,
            reward_fn=components.reward_fn,
            collector_config=components.collector_config,
        )
        runtime_inputs = build_ray_generation_inputs_for_family(
            cfg,
            components.family,
            weight_dtype=rollout_weight_dtype,
            executor_kwargs=dict(rollout_executor_kwargs),
        )
        log_host_memory("before_rollout_backend_build", log=logger)
        collector.set_runtime(
            RayGenerationLauncher().launch_from_cfg(
                cfg,
                driver_bundle=bundle,
                launch_contract=runtime_inputs.launch_contract,
                gatherer=runtime_inputs.gatherer,
                placement=placement_owner.rollout_placement,
            ),
        )
        log_host_memory("after_rollout_backend_build", log=logger)

        ref_model = (
            definition.reference_model_getter(bundle, cfg)
            if definition.reference_model_getter is not None
            else None
        )
        trainer = OnlineTrainer(
            algorithm=components.algorithm,
            collector=collector,
            evaluator=components.evaluator,
            model=model,
            ref_model=ref_model,
            weight_syncer=build_runtime_weight_syncer(
                collector.runtime,
                initial_policy_version=resume_checkpoint.next_step
                if resume_checkpoint is not None
                else None,
            ),
            sync_state_getter=build_trainable_state_sync_getter(bundle),
            config=trainer_config,
            device=device,
            # Strategy carries the resolved rank/device identity. fsdp already
            # fail-fasted in assert_strategy_executable above, so this is always
            # single_process here; the FSDP strategy slots in by context.strategy.
            strategy=SingleProcessStrategy(training_context),
        )

        if resume_checkpoint is not None:
            restore_training_checkpoint(
                resume_checkpoint,
                trainer=trainer,
                bundle=bundle,
                strict=trainer_config.resume_strict,
            )
            logger.info(
                "Resuming from %s, start_epoch=%d",
                resume_checkpoint.checkpoint_dir,
                resume_checkpoint.next_epoch,
            )

        output_dir = Path(trainer_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(cfg, output_dir, resumed=resume_checkpoint is not None)

        component_names = tuple(components.built["reward"][0].keys())
        csv_path = output_dir / "metrics.csv"
        _prepare_metrics_csv(
            csv_path,
            component_names,
            resume=resume_checkpoint is not None,
            prepare_metrics_csv=prepare_metrics_csv,
        )

        rng = torch.Generator().manual_seed(trainer_config.seed)
        start_epoch = resume_checkpoint.next_epoch if resume_checkpoint is not None else 0
        if start_epoch > trainer_config.total_epochs:
            raise ValueError(
                "resume checkpoint starts after configured total_epochs: "
                f"start_epoch={start_epoch}, total_epochs={trainer_config.total_epochs}",
            )
        if resume_checkpoint is not None:
            restore_rng_state(resume_checkpoint.rng_state, prompt_generator=rng)

        stack = OnlineRecipeStack(
            cfg=cfg,
            definition=definition,
            bundle=bundle,
            model=model,
            reward_fn=components.reward_fn,
            collector=collector,
            algorithm=components.algorithm,
            evaluator=components.evaluator,
            trainer=trainer,
            trainer_config=trainer_config,
            collector_config=components.collector_config,
            family=components.family,
            output_dir=output_dir,
            component_names=component_names,
        )

        logger.info(
            "Starting %s online recipe: epochs=%d examples=%d n=%d",
            components.family,
            trainer_config.total_epochs,
            len(examples),
            trainer_config.n_samples_per_prompt,
        )
        for epoch in range(start_epoch, trainer_config.total_epochs):
            idx = sample_prompt_indices(
                rng,
                num_examples=len(examples),
                rollout_batch_size=trainer_config.rollout_batch_size,
                strategy=str(
                    OmegaConf.select(
                        cfg,
                        "data.sampler.type",
                        default="random_without_replacement",
                    ),
                ),
                epoch=epoch,
            )
            example_batch = [examples[i] for i in idx]
            if definition.before_step is not None:
                maybe_awaitable = definition.before_step(stack, epoch, example_batch)
                if maybe_awaitable is not None:
                    await maybe_awaitable

            if gradient_accumulation_steps > 0:
                # Streaming accumulation: split the optimizer-target batch into
                # microbatches collected/trained/released one at a time so host
                # RAM does not have to hold the whole batch at once.
                metrics = await _run_streaming_optimizer_update(
                    trainer,
                    components.reward_fn,
                    example_batch,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    rollout_batch_size=int(trainer_config.rollout_batch_size),
                    n_samples_per_prompt=int(trainer_config.n_samples_per_prompt),
                    host_memory_budget_fraction=float(
                        getattr(trainer_config, "host_memory_budget_fraction", 0.0),
                    ),
                )
            else:
                components.reward_fn.reset_components()
                metrics = await trainer.step(example_batch)
            _write_metric_row(
                csv_path,
                epoch,
                metrics,
                reward_fn=components.reward_fn,
                component_names=component_names,
                metric_row_hook=definition.metric_row_hook,
            )

            if definition.after_step is not None:
                maybe_awaitable = definition.after_step(stack, epoch, example_batch)
                if maybe_awaitable is not None:
                    await maybe_awaitable

            if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
                _save_checkpoint(
                    output_dir / f"checkpoint-{epoch + 1}",
                    stack,
                    epoch=epoch + 1,
                    rng=rng,
                    save_training_checkpoint=save_training_checkpoint,
                    capture_rng_state=capture_rng_state,
                )

        _save_checkpoint(
            output_dir / "checkpoint-final",
            stack,
            epoch=trainer_config.total_epochs,
            rng=rng,
            save_training_checkpoint=save_training_checkpoint,
            capture_rng_state=capture_rng_state,
        )
        logger.info("Training complete. Final checkpoint: %s", output_dir / "checkpoint-final")
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        await _shutdown_online_recipe_runtime(
            collector=collector,
            reward_fn=reward_fn,
            placement_owner=placement_owner,
            run_error=run_error,
        )


def _preflight_production_video_reward(cfg: DictConfig) -> None:
    """Fail fast on the driver if the production reward backend is unimportable."""

    if not bool(
        OmegaConf.select(cfg, "production.kling_video_reward.enabled", default=False)
        or OmegaConf.select(cfg, "production.video_reward.enabled", default=False)
    ):
        return
    from vrl.rewards.models.kling_video_reward import preflight_kling_video_reward_backend

    try:
        preflight_kling_video_reward_backend()
    except Exception as exc:
        raise RuntimeError(
            "production.kling_video_reward requires the repo-owned Kling VideoReward "
            "inference backend under vrl/rewards/models/kling_video_reward.py.",
        ) from exc


def _prepare_metrics_csv(
    path: Path,
    component_names: tuple[str, ...],
    *,
    resume: bool,
    prepare_metrics_csv: Any,
) -> None:
    component_cols = ",".join(f"r_{name}" for name in component_names)
    header = (
        "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
        "clip_fraction,approx_kl,logprob_abs_diff_mean,logprob_abs_diff_max,"
        "ratio_abs_dev_mean,ratio_abs_dev_max,mismatch_kl,mismatch_k3_kl,"
        "advantage_mean,grad_norm,adv_saturation,"
        "adv_zero_rate,group_size,trained_prompt_num"
    )
    if component_cols:
        header = f"{header},{component_cols}"
    prepare_metrics_csv(path, header + "\n", resume=resume)


def _write_metric_row(
    path: Path,
    epoch: int,
    metrics: Any,
    *,
    reward_fn: Any,
    component_names: tuple[str, ...],
    metric_row_hook: Any,
) -> None:
    last = getattr(reward_fn, "last_components", {}) or {}
    component_means = {
        name: (sum(last.get(name, [])) / len(last.get(name, [])))
        if last.get(name)
        else float("nan")
        for name in component_names
    }
    row = {
        "epoch": epoch,
        "loss": metrics.loss,
        "policy_loss": metrics.policy_loss,
        "kl_penalty": metrics.kl_penalty,
        "reward_mean": metrics.reward_mean,
        "reward_std": metrics.reward_std,
        "clip_fraction": metrics.clip_fraction,
        "approx_kl": metrics.approx_kl,
        "logprob_abs_diff_mean": metrics.logprob_abs_diff_mean,
        "logprob_abs_diff_max": metrics.logprob_abs_diff_max,
        "ratio_abs_dev_mean": metrics.ratio_abs_dev_mean,
        "ratio_abs_dev_max": metrics.ratio_abs_dev_max,
        "mismatch_kl": metrics.mismatch_kl,
        "mismatch_k3_kl": metrics.mismatch_k3_kl,
        "advantage_mean": metrics.advantage_mean,
        "grad_norm": metrics.grad_norm,
        "adv_saturation": metrics.adv_saturation,
        "adv_zero_rate": metrics.adv_zero_rate,
        "group_size": metrics.group_size,
        "trained_prompt_num": metrics.trained_prompt_num,
        **{f"r_{name}": component_means[name] for name in component_names},
    }
    if metric_row_hook is not None:
        metric_row_hook(row, metrics)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            ",".join(
                [
                    str(row["epoch"]),
                    f"{row['loss']:.6f}",
                    f"{row['policy_loss']:.6f}",
                    f"{row['kl_penalty']:.6f}",
                    f"{row['reward_mean']:.4f}",
                    f"{row['reward_std']:.4f}",
                    f"{row['clip_fraction']:.4f}",
                    f"{row['approx_kl']:.6f}",
                    f"{row['logprob_abs_diff_mean']:.6f}",
                    f"{row['logprob_abs_diff_max']:.6f}",
                    f"{row['ratio_abs_dev_mean']:.6f}",
                    f"{row['ratio_abs_dev_max']:.6f}",
                    f"{row['mismatch_kl']:.6f}",
                    f"{row['mismatch_k3_kl']:.6f}",
                    f"{row['advantage_mean']:.6f}",
                    f"{row['grad_norm']:.6f}",
                    f"{row['adv_saturation']:.4f}",
                    f"{row['adv_zero_rate']:.4f}",
                    f"{row['group_size']:.2f}",
                    str(row["trained_prompt_num"]),
                    *(f"{row[f'r_{name}']:.4f}" for name in component_names),
                ],
            )
            + "\n",
        )


async def _shutdown_online_recipe_runtime(
    *,
    collector: Any | None,
    reward_fn: Any | None,
    placement_owner: Any | None,
    run_error: BaseException | None,
) -> None:
    shutdown_errors: list[tuple[str, Exception]] = []

    async def _run_shutdown(name: str, target: Any, method_name: str = "shutdown") -> None:
        shutdown = getattr(target, method_name, None)
        if shutdown is None:
            return
        try:
            result = shutdown()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            shutdown_errors.append((name, exc))

    await _run_shutdown("collector", collector)
    await _run_shutdown("reward_fn", reward_fn)
    await _run_shutdown("placement_owner", placement_owner)

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


def _save_checkpoint(
    path: Path,
    stack: OnlineRecipeStack,
    *,
    epoch: int,
    rng: Any,
    save_training_checkpoint: Any,
    capture_rng_state: Any,
) -> None:
    export_modules = (
        stack.definition.export_modules_getter(stack.bundle, stack.cfg)
        if stack.definition.export_modules_getter is not None
        else None
    )
    save_training_checkpoint(
        path,
        trainer=stack.trainer,
        bundle=stack.bundle,
        family=stack.family,
        progress={
            "completed_epoch": epoch,
            "next_epoch": epoch,
            "global_step": stack.trainer.state.global_step,
        },
        rng_state=capture_rng_state(prompt_generator=rng),
        export_modules=export_modules,
        export_ema=getattr(stack.trainer, "_ema", None),
    )


__all__ = [
    "default_reference_model",
    "enable_transformer_gradient_checkpointing",
    "export_language_model_lora",
    "export_transformer_lora",
    "run_online_recipe",
]
