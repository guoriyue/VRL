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
from vrl.utils.memory import log_host_memory

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

    transformer = bundle.model.transformer
    if bool(OmegaConf.select(cfg, "model.use_lora", default=False)) and hasattr(
        transformer, "save_pretrained"
    ):
        return {LORA_WEIGHTS_NAME: transformer}
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

    transformer = bundle.model.transformer
    # Optional key: base yaml no longer restates the dataclass default, so an
    # absent key means "use the TrainerConfig default" — derived, not copied.
    enabled = OmegaConf.select(cfg, "actor.gradient_checkpointing")
    if enabled is None:
        enabled = TrainerConfig.__dataclass_fields__["gradient_checkpointing"].default
    if not bool(enabled):
        return

    enable = getattr(transformer, "enable_gradient_checkpointing", None)
    if enable is None:
        if require_method:
            raise AttributeError(
                "bundle.model.transformer does not expose enable_gradient_checkpointing",
            )
        return
    enable()


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
