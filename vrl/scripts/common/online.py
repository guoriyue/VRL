"""Common runner skeleton for online training recipes."""

from __future__ import annotations

import gc
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from vrl.config.builders import build_configs
from vrl.config.precision import resolve_precision_policy
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.interfaces import require_runtime_model
from vrl.ray.dependencies import inspect_cluster, require_ray
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
from vrl.trainers.data import (
    load_eval_prompt_examples_from_config,
    load_prompt_examples_from_config,
)
from vrl.trainers.distributed import DistributedTrainingContext, resolve_training_context
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.precision import torch_dtype_for_trainer_precision
from vrl.trainers.strategy import build_strategy
from vrl.trainers.weight_sync import build_runtime_weight_syncer
from vrl.utils.memory import capture_host_memory, format_host_memory, log_host_memory

logger = logging.getLogger(__name__)


def _require_supported_online_strategy(context: DistributedTrainingContext) -> None:
    """Fail-fast when the online recipe cannot yet drive the resolved strategy.

    ``ddp`` is supported via the per-rank-local symmetric-colocated path: each
    torchrun rank runs its own local Ray + local colocated rollout on its 1 GPU
    and trains a full DDP replica; only the gradient all-reduce crosses ranks
    (SPRINT_symmetric_colocated_ddp.md). ``fsdp`` is still gated — its sharded
    DTensor state needs the rank0-full-gather collect/checkpoint orchestration the
    online recipe does not implement yet (SPRINT_multi_gpu_training.md Phase 4).
    """

    if context.strategy not in {"single_process", "ddp"}:
        raise NotImplementedError(
            f"distributed.training.strategy={context.strategy!r}: the FSDPStrategy "
            "layer exists (vrl/trainers/strategy.py) and is unit-tested, but "
            "run_online_recipe does not yet implement the sharded multi-rank GRPO "
            "orchestration that drives it (rank0-full-gather collect/checkpoint + "
            "torchrun↔Ray coordination, SPRINT_multi_gpu_training.md Phase 4/§6.5). "
            "Use strategy=single_process, or strategy=ddp for symmetric colocated "
            "multi-GPU (SPRINT_symmetric_colocated_ddp.md).",
        )


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
    sample_batch_size = int(getattr(trainer_config, "sample_batch_size", 0) or 0)
    # One knob bounds both the generation forward chunk and the train replay
    # chunk, so this per-call sample count applies to generation and backward.
    sample_chunk_size = (
        samples_per_prompt
        if sample_batch_size <= 0
        else min(samples_per_prompt, sample_batch_size)
    )
    gas = int(getattr(trainer_config, "gradient_accumulation_steps", 0))
    if gas > 0:
        # Read the reconciled microbatch_size (TrainerConfig.__post_init__ sets it
        # to rollout_batch_size // gas) rather than recomputing the same quotient.
        microbatch_prompts = int(trainer_config.microbatch_size)
        microbatch_samples = microbatch_prompts * samples_per_prompt
        logger.info(
            "Rollout memory plan: streaming accumulation enabled "
            "(rollout_batch_size=%d, gradient_accumulation_steps=%d, "
            "microbatch_prompts=%d, microbatch_samples=%d, "
            "sample_chunk_size_per_call=%d, "
            "target_samples_per_update=%d)",
            rollout_batch_size,
            gas,
            microbatch_prompts,
            microbatch_samples,
            sample_chunk_size,
            target_samples,
        )
        return

    logger.info(
        "Rollout memory plan: legacy full-batch accumulation "
        "(rollout_batch_size=%d, sample_chunk_size_per_call=%d, "
        "target_samples_per_update=%d)",
        rollout_batch_size,
        sample_chunk_size,
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


@dataclass(slots=True)
class OnlineRecipeRun:
    """Execution controller for one ``run_online_recipe`` invocation.

    Holds the wired runtime (``stack``) plus the per-run execution state the
    recipe loop mutates: the two metrics-CSV paths, the prompt-sampling RNG, and
    whether this is a resume. The metrics-CSV and checkpoint side effects live
    here as methods so the loop calls ``run.write_metric_row(epoch, metrics)``
    instead of threading ``csv_path`` / ``reward_fn`` / ``component_names`` /
    ``rng`` through free functions on every call.

    This is NOT a second owner of ``stack``'s fields -- component_names,
    reward_fn, trainer, definition, etc. are all read through ``self.stack``.
    ``OnlineRecipeStack`` stays the family-hook payload (handed to ``before_step``
    / ``after_step``); ``OnlineRecipeRun`` is the IO/execution controller and is
    never exposed to family hooks.
    """

    stack: OnlineRecipeStack
    csv_path: Path
    eval_csv_path: Path
    rng: Any
    resume: bool

    def prepare_metrics_csv(self) -> None:
        component_cols = ",".join(f"r_{name}" for name in self.stack.component_names)
        header = (
            "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
            "clip_fraction,approx_kl,logprob_abs_diff_mean,logprob_abs_diff_max,"
            "ratio_abs_dev_mean,ratio_abs_dev_max,mismatch_kl,mismatch_k3_kl,"
            "advantage_mean,grad_norm,adv_saturation,"
            "adv_zero_rate,group_size,trained_prompt_num,"
            # Continuous-rollout async diagnostics (0 in strict_on_policy mode). These
            # answer "is the run actually async?": observed staleness of consumed
            # samples, prefetched ready-queue depth, weight-sync barrier pause, and
            # producer starvation. The two *_dropped/discarded columns quantify
            # wasted generation: groups dropped past the staleness window at receipt
            # (producer) and ready items purged right after a weight sync (schedule).
            # Both stay ~0 on a single fast-refilling card and only grow under real
            # disaggregated overlap. Sourced from TrainStepMetrics.phase_times.
            "continuous_stale_versions,continuous_ready_groups,"
            "continuous_weight_sync_pause_s,continuous_producer_max_gap_s,"
            "continuous_producer_discarded_stale,continuous_post_sync_dropped_stale,"
            # 0 = draining weight-sync barrier (waited for in-flight generation),
            # 1 = non-draining (versioned trainable-state slots let it skip the wait).
            "continuous_weight_sync_barrier_mode"
        )
        if component_cols:
            header = f"{header},{component_cols}"
        prepare_metrics_csv(self.csv_path, header + "\n", resume=self.resume)

    def write_metric_row(self, epoch: int, metrics: Any) -> None:
        reward_fn = self.stack.reward_fn
        component_names = self.stack.component_names
        last = getattr(reward_fn, "last_components", {}) or {}
        component_means = {
            name: (sum(last.get(name, [])) / len(last.get(name, [])))
            if last.get(name)
            else float("nan")
            for name in component_names
        }
        # Continuous async diagnostics live in TrainStepMetrics.phase_times (attached
        # per iteration by ContinuousRolloutSchedule); empty in strict_on_policy mode.
        phases = getattr(metrics, "phase_times", None) or {}
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
            "continuous_stale_versions": phases.get("continuous.stale_policy_versions", 0.0),
            "continuous_ready_groups": phases.get("continuous.queue_ready_groups", 0.0),
            "continuous_weight_sync_pause_s": phases.get("continuous.weight_sync_pause_s", 0.0),
            "continuous_producer_max_gap_s": phases.get("continuous.producer_max_tick_gap_s", 0.0),
            "continuous_producer_discarded_stale": phases.get(
                "continuous.producer_discarded_stale", 0.0,
            ),
            "continuous_post_sync_dropped_stale": phases.get(
                "continuous.post_sync_dropped_stale", 0.0,
            ),
            "continuous_weight_sync_barrier_mode": phases.get(
                "continuous.weight_sync_barrier_mode", 0.0,
            ),
            **{f"r_{name}": component_means[name] for name in component_names},
        }
        metric_row_hook = self.stack.definition.metric_row_hook
        if metric_row_hook is not None:
            metric_row_hook(row, metrics)
        with self.csv_path.open("a", encoding="utf-8") as handle:
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
                        f"{row['continuous_stale_versions']:.1f}",
                        f"{row['continuous_ready_groups']:.1f}",
                        f"{row['continuous_weight_sync_pause_s']:.4f}",
                        f"{row['continuous_producer_max_gap_s']:.4f}",
                        f"{row['continuous_producer_discarded_stale']:.1f}",
                        f"{row['continuous_post_sync_dropped_stale']:.1f}",
                        f"{row['continuous_weight_sync_barrier_mode']:.1f}",
                        *(f"{row[f'r_{name}']:.4f}" for name in component_names),
                    ],
                )
                + "\n",
            )

    def prepare_eval_metrics_csv(self) -> None:
        """eval_metrics.csv header. ``epoch=-1`` is the pre-RL baseline row."""

        component_cols = ",".join(f"r_{name}" for name in self.stack.component_names)
        header = "epoch,eval_reward_mean,eval_reward_std,eval_reward_stderr,eval_n,global_step"
        if component_cols:
            header = f"{header},{component_cols}"
        prepare_metrics_csv(self.eval_csv_path, header + "\n", resume=self.resume)

    def write_eval_metric_row(
        self,
        epoch: int,
        result: _FixedEvalResult,
        *,
        global_step: int,
    ) -> None:
        means = result.component_means
        columns = [
            str(epoch),
            f"{result.reward_mean:.4f}",
            f"{result.reward_std:.4f}",
            f"{result.reward_stderr:.4f}",
            str(result.n),
            str(global_step),
            *(f"{means.get(name, float('nan')):.4f}" for name in self.stack.component_names),
        ]
        with self.eval_csv_path.open("a", encoding="utf-8") as handle:
            handle.write(",".join(columns) + "\n")

    def save_checkpoint(self, path: Path, *, epoch: int) -> None:
        stack = self.stack
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
            rng_state=capture_rng_state(prompt_generator=self.rng),
            export_modules=export_modules,
            export_ema=getattr(stack.trainer, "_ema", None),
            strategy=stack.strategy,
        )


def _maybe_autodetect_cross_node(cfg: DictConfig, ray: Any) -> None:
    """Enable distributed.resources.cross_node automatically on a multi-node cluster.

    Mirrors slime/cosmos-rl, where the operator brings the cluster up first
    (``ray start``) and the driver attaches to it. When ``cross_node`` is not set
    explicitly we ATTACH to an already-running cluster (``address="auto"``) and, if
    it exposes GPUs on non-driver nodes, set ``cross_node=true`` before resolving so
    the user need not hand-set it. A single-node run with no external cluster raises
    ``ConnectionError`` on attach and is left untouched -- the normal ``ray.init()``
    later starts the local instance -- so single-node behaviour is unchanged. An
    explicit ``cross_node`` (true or false) is an override and is always respected.
    """

    if OmegaConf.select(cfg, "distributed.resources.cross_node", default=None) is not None:
        return
    if not ray.is_initialized():
        try:
            ray.init(address="auto")
        except ConnectionError:
            return
    try:
        multinode = inspect_cluster(ray).has_non_driver_gpus
    except Exception:
        # Best-effort: if the live cluster cannot be inspected, leave cross_node
        # unset (single-node default; the user can set it explicitly).
        return
    if not multinode:
        return
    OmegaConf.update(cfg, "distributed.resources.cross_node", True, force_add=True)
    logger.info(
        "Auto-detected a multi-node Ray cluster (GPUs on non-driver nodes); "
        "enabling distributed.resources.cross_node=true. Set it explicitly to "
        "override.",
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

    # Auto-enable cross_node when an already-running multi-node Ray cluster is
    # detected, so the user need not hand-set it. Must run before resolve, since
    # the resolver sizes the GPU budget differently under cross_node. No-op (and
    # ordering unchanged) for single-node runs with no external cluster.
    _maybe_autodetect_cross_node(cfg, require_ray())
    resources = resolve_distributed_resources(cfg)
    logger.info(format_distributed_resource_plan(resources))
    device = torch.device(trainer_torch_device(resources))
    # Resolve the training process identity (rank/device) and fail-fast on
    # strategies the online recipe can't yet drive end-to-end, before building the
    # model / Ray runtime.
    training_context = resolve_training_context(cfg, device=device)
    _require_supported_online_strategy(training_context)
    # Under ddp every torchrun rank owns a distinct GPU: resolve_training_context
    # returns cuda:<local_rank>, which overrides the resolver's (rank-agnostic)
    # trainer device so the trainer model, rollout, and weight sync all land on
    # this rank's card. single_process passes the resolver device straight through.
    device = training_context.device
    # Replay/training model storage follows ``compute`` (via trainer_config);
    # the generation (rollout) model can use a different ``rollout`` dtype.
    weight_dtype = (
        definition.weight_dtype_getter(cfg, trainer_config, torch)
        if definition.weight_dtype_getter is not None
        else torch_dtype_for_trainer_precision(trainer_config, torch)
    )
    policy = resolve_precision_policy(cfg)
    rollout_precision = policy.rollout
    if rollout_precision == "fp4":
        # fp4 has a precision token + drift/TIS support but no GEMM kernel yet
        # (Fp8Linear is e4m3 only). Fail loudly rather than crash in the forward.
        raise NotImplementedError(
            "precision.rollout='fp4': the fp4 rollout GEMM is not built "
            "(Fp8Linear is e4m3/fp8 only). Use fp8 or bf16/fp16/fp32 for live runs.",
        )
    if rollout_precision == "fp8":
        # fp8 rollout is a quantized GEMM, not float8 storage: the rollout model
        # loads its bf16 master and the runtime builder swaps the big linears to
        # Fp8Linear (torch._scaled_mm). So storage stays the compute (bf16) dtype;
        # the swap is driven by spec.rollout_quantization (extract_runtime_spec).
        rollout_weight_dtype = resolve_torch_dtype(policy.compute)
    else:
        rollout_weight_dtype = resolve_torch_dtype(rollout_precision)
    context = RecipeDeviceContext(
        device=device,
        weight_dtype=weight_dtype,
    )
    examples = load_prompt_examples_from_config(cfg.data)
    eval_cfg = getattr(trainer_config, "eval", None)
    eval_enabled = bool(getattr(eval_cfg, "enabled", False))
    eval_examples = (
        load_eval_prompt_examples_from_config(cfg.data) if eval_enabled else []
    )

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
        # The strategy is the single owner of trainable-state export for both
        # rollout weight sync and checkpointing. build_strategy maps the resolved
        # context to a concrete strategy; fsdp is fail-fasted by
        # _require_supported_online_strategy above, so this is single_process in
        # the live online path. The FSDPStrategy slots in here once the online
        # rank-split orchestration lands, without the recipe changing how state
        # leaves the trainer.
        strategy = build_strategy(cfg, training_context)
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
            # Rollout weight sync re-reads live trainable state on every push, so
            # bind the strategy export lazily instead of snapshotting once.
            sync_state_getter=lambda: strategy.export_rollout_state(bundle),
            config=trainer_config,
            device=device,
            strategy=strategy,
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

        # Under ddp only rank0 owns run IO (metrics/checkpoint/eval/resolved-config):
        # each rank trains a full DDP replica and the grad all-reduce in trainer.step
        # keeps ranks in lockstep, so rank0's checkpoint is complete and a single
        # writer avoids N ranks racing the same files (on 2 servers, rank0's host).
        is_primary = training_context.is_primary
        output_dir = Path(trainer_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_primary:
            save_resolved_config(cfg, output_dir, resumed=resume_checkpoint is not None)

        component_names = tuple(components.built["reward"][0].keys())

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
            strategy=strategy,
            trainer_config=trainer_config,
            collector_config=components.collector_config,
            family=components.family,
            output_dir=output_dir,
            component_names=component_names,
        )
        # Execution controller for this run: owns the metrics-CSV paths, prompt
        # RNG, and resume flag, and carries the metrics/checkpoint IO as methods so
        # the loop below stops threading csv_path / reward_fn / component_names /
        # rng through free helpers. Holds `stack` (single owner of the wired
        # runtime); never handed to family hooks, which still receive `stack`.
        run = OnlineRecipeRun(
            stack=stack,
            csv_path=output_dir / "metrics.csv",
            eval_csv_path=output_dir / "eval_metrics.csv",
            rng=rng,
            resume=resume_checkpoint is not None,
        )
        if is_primary:
            run.prepare_metrics_csv()
            if eval_enabled:
                run.prepare_eval_metrics_csv()

        logger.info(
            "Starting %s online recipe: epochs=%d examples=%d n=%d",
            components.family,
            trainer_config.total_epochs,
            len(examples),
            trainer_config.n_samples_per_prompt,
        )

        async def _fixed_eval_and_log(eval_epoch: int) -> None:
            """Run the fixed eval, append its row, log the signal. eval_epoch=-1 = baseline."""
            result = await _run_fixed_eval(
                collector,
                components.reward_fn,
                eval_examples,
                samples_per_prompt=int(eval_cfg.samples_per_prompt),
                base_seed=int(eval_cfg.seed),
                max_prompts=int(eval_cfg.max_prompts),
                component_names=component_names,
            )
            run.write_eval_metric_row(
                eval_epoch,
                result,
                global_step=int(trainer.state.global_step),
            )
            logger.info(
                "fixed eval epoch=%d: eval_reward_mean=%.4f +/- %.4f (n=%d)",
                eval_epoch,
                result.reward_mean,
                result.reward_stderr,
                result.n,
            )

        # Pre-RL baseline on the fixed grid (fresh runs only; resume keeps its rows).
        # rank0-only under ddp: non-primary ranks block at the next trainer.step
        # grad all-reduce until rank0 finishes, so this stays in lockstep.
        if eval_enabled and resume_checkpoint is None and is_primary:
            await _fixed_eval_and_log(-1)

        # DDP is data-parallel: every rank shares the prompt RNG (identical draw),
        # so draw world_size * rollout_batch_size prompts and hand each rank a
        # DISJOINT slice. The all-reduced gradient then covers world_size *
        # rollout_batch_size DISTINCT prompts (the effective batch) instead of every
        # rank redundantly training the SAME rollout_batch_size prompts and averaging
        # identical gradients. single_process (world_size=1, rank=0) takes the whole
        # draw unchanged. Requires the prompt manifest to hold >= world_size *
        # rollout_batch_size prompts for a without-replacement sampler.
        rank_batch = int(trainer_config.rollout_batch_size)
        sampler_strategy = str(
            OmegaConf.select(cfg, "data.sampler.type", default="random_without_replacement"),
        )
        for epoch in range(start_epoch, trainer_config.total_epochs):
            idx = sample_prompt_indices(
                rng,
                num_examples=len(examples),
                rollout_batch_size=rank_batch * training_context.world_size,
                strategy=sampler_strategy,
                epoch=epoch,
            )
            shard = idx[
                training_context.rank * rank_batch : (training_context.rank + 1) * rank_batch
            ]
            example_batch = [examples[i] for i in shard]
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
            if is_primary:
                run.write_metric_row(epoch, metrics)

            # Fixed eval AFTER the training row (eval overwrites reward_fn
            # components; the next epoch's collect resets them). rank0-only.
            if eval_enabled and (epoch + 1) % int(eval_cfg.freq) == 0 and is_primary:
                await _fixed_eval_and_log(epoch)

            if definition.after_step is not None:
                maybe_awaitable = definition.after_step(stack, epoch, example_batch)
                if maybe_awaitable is not None:
                    await maybe_awaitable

            if (
                is_primary
                and trainer_config.save_freq > 0
                and (epoch + 1) % trainer_config.save_freq == 0
            ):
                run.save_checkpoint(output_dir / f"checkpoint-{epoch + 1}", epoch=epoch + 1)

        if is_primary:
            run.save_checkpoint(
                output_dir / "checkpoint-final",
                epoch=trainer_config.total_epochs,
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


@dataclass(frozen=True, slots=True)
class _FixedEvalResult:
    """Aggregated fixed-eval reward over the held-out prompt+seed grid."""

    reward_mean: float
    reward_std: float
    reward_stderr: float
    n: int
    component_means: dict[str, float]


def _fixed_eval_collect_kwargs(item: Any, *, group_size: int, seed: int) -> dict[str, Any]:
    """Collect kwargs for one eval prompt group at a fixed seed.

    Mirrors prompt_collection._prompt_example_kwargs so image/caption/reference
    eval prompts carry their conditioning, and adds the deterministic ``seed``
    the request builder threads into sampling (requests.py).
    """

    kwargs: dict[str, Any] = {"group_size": int(group_size), "seed": int(seed)}
    if not isinstance(item, str):
        for key, attr in (
            ("target_text", "target_text"),
            ("references", "references"),
            ("task_type", "task_type"),
            ("request_overrides", "request_overrides"),
            ("sample_metadata", "metadata"),
        ):
            value = getattr(item, attr, None)
            if value is not None:
                kwargs[key] = value
        reference_image = getattr(item, "reference_image", None)
        if reference_image:
            kwargs["reference_image"] = reference_image
        reference_video = getattr(item, "reference_video", None)
        if reference_video:
            kwargs["reference_video"] = reference_video
    return kwargs


async def _run_fixed_eval(
    collector: Any,
    reward_fn: Any,
    eval_examples: list[Any],
    *,
    samples_per_prompt: int,
    base_seed: int,
    max_prompts: int,
    component_names: tuple[str, ...],
) -> _FixedEvalResult:
    """Score a held-out prompt set on a FIXED seed grid — the learning signal.

    Reuses the training ``collector``/runtime/``reward_fn`` (same sampling path as
    training) but does NOT touch the trainer: no collect_training_batch, no
    backward, no optimizer/EMA/previous-policy sync, no trainer.prompts mutation.
    Each prompt gets ``samples_per_prompt`` videos at a deterministic seed derived
    from ``base_seed``, so the same grid reproduces across epochs and resumes.
    Rollout/reward artifacts are released before returning to avoid host-RAM creep.
    """

    examples = list(eval_examples)
    if max_prompts > 0:
        examples = examples[:max_prompts]
    if not examples:
        raise ValueError(
            "fixed eval has no prompts (check data.eval_manifest and trainer.eval.max_prompts)",
        )

    # Snapshot only this eval's reward components, not training's.
    reward_fn.reset_components()
    unscored: list[Any] = []
    batches: list[Any] = []
    try:
        for prompt_index, item in enumerate(examples):
            seed = int(base_seed) + prompt_index * int(samples_per_prompt)
            prompt = str(getattr(item, "prompt", item))
            unscored.append(
                await collector.collect_unscored(
                    [prompt],
                    **_fixed_eval_collect_kwargs(item, group_size=samples_per_prompt, seed=seed),
                ),
            )
        batches = await collector.score_rollouts(unscored)
        rewards = torch.cat([b.rewards.detach().float().reshape(-1).cpu() for b in batches])
        n = int(rewards.numel())
        mean = float(rewards.mean().item()) if n else float("nan")
        std = float(rewards.std(unbiased=False).item()) if n > 1 else 0.0
        stderr = std / (n**0.5) if n > 0 else 0.0
        last = getattr(reward_fn, "last_components", {}) or {}
        component_means = {
            name: (sum(last.get(name, [])) / len(last.get(name, [])))
            if last.get(name)
            else float("nan")
            for name in component_names
        }
        return _FixedEvalResult(mean, std, stderr, n, component_means)
    finally:
        # Drop rollout/reward artifact refs before the next training epoch.
        del batches, unscored
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


__all__ = [
    "default_reference_model",
    "enable_transformer_gradient_checkpointing",
    "export_language_model_lora",
    "export_transformer_lora",
    "run_online_recipe",
]
