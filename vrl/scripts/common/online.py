"""Common runner skeleton for online training recipes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from vrl.config.loader import build_configs
from vrl.distributed.resources import (
    format_distributed_resource_plan,
    resolve_distributed_resources,
    trainer_torch_device,
)
from vrl.models.interfaces import require_runtime_model
from vrl.rollouts.runtime.backend import build_rollout_backend_from_cfg
from vrl.rollouts.runtime.launch_inputs import build_rollout_runtime_inputs
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
from vrl.trainers.data import load_prompt_manifest
from vrl.trainers.memory import log_host_memory
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.precision import torch_dtype_for_trainer_precision
from vrl.trainers.weight_sync import (
    build_runtime_weight_syncer,
    build_trainable_state_sync_getter,
)

logger = logging.getLogger(__name__)


async def run_online_recipe(
    cfg: DictConfig,
    definition: OnlineRecipeDefinition,
) -> None:
    """Run a family online training job through shared recipe glue."""

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    if definition.configure_trainer is not None:
        definition.configure_trainer(cfg, trainer_config)
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
    weight_dtype = (
        definition.weight_dtype_getter(cfg, trainer_config, torch)
        if definition.weight_dtype_getter is not None
        else torch_dtype_for_trainer_precision(trainer_config, torch)
    )
    context = RecipeDeviceContext(
        device=device,
        weight_dtype=weight_dtype,
        distributed_resources=resources,
    )
    examples = load_prompt_manifest(Path(str(cfg.data.manifest)))

    log_host_memory("before_bundle_build", log=logger)
    bundle = definition.build_bundle(cfg, context.device, context.weight_dtype)
    log_host_memory("after_bundle_build", log=logger)
    if definition.after_bundle_built is not None:
        definition.after_bundle_built(bundle, cfg)
    model = require_runtime_model(
        definition.model_getter(bundle),
        owner=f"{definition.family}.model_getter",
    )
    scheduler = definition.scheduler_getter(bundle)

    components = build_online_recipe_components(
        cfg,
        family=definition.family,
        device=str(device),
        scheduler=scheduler,
        built=built,
    )
    collector = build_collector_from_cfg(
        cfg,
        family=components.family_entry,
        model=model,
        reward_fn=components.reward_fn,
        collector_config=components.collector_config,
        **(
            definition.collector_kwargs_getter(cfg, examples)
            if definition.collector_kwargs_getter is not None
            else {}
        ),
    )
    runtime_inputs = build_rollout_runtime_inputs(
        cfg,
        components.family,
        weight_dtype=weight_dtype,
        executor_kwargs=dict(getattr(collector, "executor_kwargs", {}) or {}),
    )
    log_host_memory("before_rollout_backend_build", log=logger)
    collector.set_runtime(
        build_rollout_backend_from_cfg(
            cfg,
            driver_bundle=bundle,
            runtime_spec=runtime_inputs.runtime_spec,
            gatherer=runtime_inputs.gatherer,
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
        stat_tracker=_build_stat_tracker(cfg, components.algorithm),
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
        trainer_config.n,
    )
    for epoch in range(start_epoch, trainer_config.total_epochs):
        idx = sample_prompt_indices(
            rng,
            num_examples=len(examples),
            rollout_batch_size=trainer_config.rollout_batch_size,
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


def _build_stat_tracker(cfg: DictConfig, algorithm: Any) -> Any | None:
    if not bool(cfg.algorithm.get("per_prompt_stat_tracking", True)):
        return None
    from vrl.algorithms.stat_tracking import PerPromptStatTracker

    config = getattr(algorithm, "config", None)
    return PerPromptStatTracker(global_std=bool(getattr(config, "global_std", False)))


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
        "clip_fraction,approx_kl,advantage_mean,grad_norm,adv_saturation,"
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


__all__ = ["run_online_recipe"]
