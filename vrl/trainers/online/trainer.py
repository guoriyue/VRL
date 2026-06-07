"""Online RL trainer — CEA pipeline (Collector + Evaluator + Algorithm).

collect -> evaluate -> advantage -> loss -> backward -> step.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch, stack_batches
from vrl.rollouts.batch.ops import (
    apply_sample_mask,
    move_training_batch_to_device,
    nonzero_advantage_mask,
    pad_zero_advantage_mask,
    shuffle_and_rebatch_batches,
)
from vrl.rollouts.orchestration import build_rollout_schedule
from vrl.trainers.core.base import Trainer
from vrl.trainers.core.types import TrainerConfig, TrainState
from vrl.trainers.online.ema import EMAModuleWrapper
from vrl.trainers.online.precision_guard import run_precision_drift_guard
from vrl.trainers.precision import trainer_mixed_precision
from vrl.trainers.weight_sync import TrainableStateGetter, WeightSyncer
from vrl.utils.model_diagnostics import (
    parameter_state_summary,
    tensor_stats,
    trainable_state_digest,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def _create_optimizer(
    parameters: Any,
    config: TrainerConfig,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer."""
    optim = config.optim
    return torch.optim.AdamW(
        parameters,
        lr=optim.lr,
        betas=(optim.adam_beta1, optim.adam_beta2),
        weight_decay=optim.weight_decay,
        eps=optim.eps,
    )


# ---------------------------------------------------------------------------
# Phase profiler
# ---------------------------------------------------------------------------


class PhaseTimer:
    """Accumulating phase timer with optional CUDA sync.

    Each ``time(name)`` call returns a context manager whose wall time is
    added to ``self.times[name]``. When ``sync=True`` and CUDA is available,
    ``torch.cuda.synchronize()`` is called on both ends so async GPU kernels
    are captured.
    """

    def __init__(self, enabled: bool = False, sync: bool = True) -> None:
        self.enabled = enabled
        self.sync = sync and torch.cuda.is_available()
        self.times: dict[str, float] = defaultdict(float)
        self.events: list[tuple[str, float, float]] = []

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        wall0 = time.time()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.times[name] += t1 - t0
            self.events.append((name, wall0, time.time()))


# ---------------------------------------------------------------------------
# Autocast helper
# ---------------------------------------------------------------------------


def _resolve_mixed_precision(config: TrainerConfig) -> str:
    return trainer_mixed_precision(config)


def _get_autocast(
    config: TrainerConfig,
    device: torch.device,
    model: Any | None = None,
) -> Any:
    """Return the configured autocast context manager."""
    autocast_dtype = _trainer_autocast_dtype(config, device, model=model)
    if autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.amp.autocast(str(device), dtype=autocast_dtype)


def _trainer_autocast_dtype(
    config: TrainerConfig,
    device: torch.device,
    model: Any | None = None,
) -> torch.dtype | None:
    if bool(getattr(model, "disable_train_autocast", False)):
        return None
    precision = _resolve_mixed_precision(config)
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16" and device.type == "cuda":
        return torch.float16
    return None


def _precision_label(value: Any) -> str:
    token = str(value or "").strip().lower()
    return "fp32" if token in ("", "no") else token.removeprefix("torch.")


def _dtype_label(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def _model_transformer_dtype(model: Any) -> str | None:
    getter = getattr(model, "_transformer_dtype", None)
    if callable(getter):
        try:
            return _dtype_label(getter())
        except Exception:
            pass
    transformer = getattr(model, "transformer", None)
    dtype = getattr(transformer, "dtype", None)
    if dtype is not None:
        return _dtype_label(dtype)
    parameters = getattr(transformer if transformer is not None else model, "parameters", None)
    if callable(parameters):
        try:
            return _dtype_label(next(parameters()).dtype)
        except (StopIteration, RuntimeError, TypeError):
            return None
    return None


def _trainer_precision_metadata(
    config: TrainerConfig,
    device: torch.device,
    model: Any,
) -> dict[str, Any]:
    compute_precision = _precision_label(_resolve_mixed_precision(config))
    rollout_precision = _precision_label(config.rollout_precision or compute_precision)
    autocast_dtype = _trainer_autocast_dtype(config, device, model=model)
    return {
        "compute_precision": compute_precision,
        "rollout_precision": rollout_precision,
        "math_precision": _precision_label(config.math_precision),
        "mixed_precision": _resolve_mixed_precision(config),
        "trainer_autocast_enabled": autocast_dtype is not None,
        "trainer_autocast_dtype": _dtype_label(autocast_dtype),
        "trainer_transformer_dtype": _model_transformer_dtype(model),
        "allow_tf32_config": bool(config.optim.allow_tf32),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def _merge_rollout_precision_context(
    metadata: dict[str, Any],
    batch_context: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(metadata)
    for key in ("rollout_transformer_dtype", "rollout_autocast_enabled"):
        if key in batch_context:
            merged[key] = batch_context[key]
    return merged


# ---------------------------------------------------------------------------
# OnlineTrainer
# ---------------------------------------------------------------------------


class OnlineTrainer(Trainer):
    """Orchestrates the CEA online RL loop.

    Pipeline: collect -> evaluate -> advantage -> loss -> backward -> step.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        collector: Any,
        evaluator: Any,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        weight_syncer: WeightSyncer | None = None,
        sync_state_getter: TrainableStateGetter | None = None,
        config: TrainerConfig | None = None,
        prompts: list[str] | None = None,
        device: torch.device | str = "cuda",
        accelerator: Any | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.collector = collector
        self.evaluator = evaluator
        self.model = model
        self.ref_model = ref_model
        self.weight_syncer = weight_syncer
        if weight_syncer is not None and sync_state_getter is None:
            raise ValueError(
                "OnlineTrainer weight sync requires an explicit trainable-state "
                "getter; syncing model.state_dict() would send frozen modules.",
            )
        self.sync_state_getter = sync_state_getter
        self.config = config or TrainerConfig()
        self.prompts = prompts or []
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state = TrainState()
        self.accelerator = accelerator

        self._optimizer: torch.optim.Optimizer | None = None
        self._ema: EMAModuleWrapper | None = None
        self._rollout_weights_initialized = False
        self.rollout_schedule = build_rollout_schedule(
            self.config.rollout_orchestration,
            collector=self.collector,
            model=self.model,
            device=self.device,
            weight_syncer=self.weight_syncer,
            sync_state_getter=self.sync_state_getter,
            weights_initialized=lambda: self._rollout_weights_initialized,
            set_weights_initialized=self._set_rollout_weights_initialized,
        )

        if self.config.optim.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = _create_optimizer(trainable, self.config)
        return self._optimizer

    def _ensure_ema(self) -> EMAModuleWrapper | None:
        if not self.config.ema.enable:
            return None
        if self._ema is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._ema = EMAModuleWrapper(
                trainable,
                decay=self.config.ema.decay,
                update_step_interval=self.config.ema.update_interval,
                device=self.device,
            )
        return self._ema

    def _set_rollout_weights_initialized(self, value: bool) -> None:
        self._rollout_weights_initialized = bool(value)

    # ------------------------------------------------------------------
    # Accelerator-aware backward/step helpers
    # ------------------------------------------------------------------

    def _backward(self, loss: Any) -> None:
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def _clip_and_step(self, optimizer: Any) -> float:
        """Clip grads, step optimizer, return pre-clip total grad-norm (float)."""
        cfg = self.config
        grad_norm: Any = 0.0
        if self.accelerator is not None:
            if self.accelerator.sync_gradients and cfg.max_norm > 0:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.max_norm)
        else:
            if cfg.max_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_norm)
            else:
                # no clip — compute norm manually for diagnostic
                sq_sum = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        sq_sum += float(p.grad.detach().pow(2).sum().item())
                grad_norm = sq_sum**0.5
        optimizer.step()
        optimizer.zero_grad()
        return float(grad_norm)

    # ------------------------------------------------------------------
    # Training step — CEA pipeline
    # ------------------------------------------------------------------

    async def step(self, prompts: list[str] | None = None) -> TrainStepMetrics:
        """Run one full training step: collect -> evaluate -> advantage -> loss -> backward -> step."""
        from vrl.utils.profiling import torch_profiler_step

        with torch_profiler_step(
            self.config.torch_profiler,
            output_dir=self.config.output_dir,
            step=self.state.step,
            device=self.device,
            worker_name="online_trainer",
        ):
            return await self._step_impl(prompts)

    async def _step_impl(self, prompts: list[str] | None = None) -> TrainStepMetrics:
        """Run one full training step without profiler wrapping."""
        from vrl.algorithms.trajectory import AlgorithmAdapter, AlgorithmInput
        from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
        from vrl.utils.profiling import record_function

        if prompts is not None:
            self.prompts = prompts

        cfg = self.config
        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()

        timer = PhaseTimer(enabled=cfg.profile)
        runtime_debug_collect = bool(cfg.debug.first_step and self.state.step == 0)

        # 1. The rollout schedule owns collect/offload/release/sync timing.
        iteration = await self.rollout_schedule.next_iteration(
            list(self.prompts),
            group_size=cfg.n,
            runtime_debug=runtime_debug_collect,
        )
        all_batches: list[RolloutBatch] = iteration.batches

        # 2. Compute advantages (per-prompt normalization).
        # Rewards are concatenated across all collected batches, normalized
        # per prompt-group by the algorithm (the single source of truth for
        # advantage math), then split back per-batch for the training loop.
        with timer.time("advantage"):
            all_rewards = torch.cat([b.rewards for b in all_batches])
            all_group_ids = torch.cat([b.group_ids for b in all_batches])
            advantages_all = self.algorithm.compute_advantages_from_tensors(
                all_rewards,
                all_group_ids,
            )
            # Per-prompt grouping stats for logging (mean group size + unique prompts).
            _unique_groups, _group_counts = torch.unique(all_group_ids, return_counts=True)
            group_size = (
                float(_group_counts.float().mean().item()) if _group_counts.numel() else 0.0
            )
            trained_prompt_num = int(_unique_groups.numel())

        # Advantage diagnostics on the full (pre-filter) advantages.
        _adv_abs = advantages_all.detach().abs()
        _total = max(advantages_all.numel(), 1)
        adv_zero_rate = float((_adv_abs < 1e-6).sum().item()) / _total
        _clip_max = getattr(self.algorithm.config, "adv_clip_max", None)
        adv_saturation = (
            float((_adv_abs >= _clip_max - 1e-6).sum().item()) / _total
            if _clip_max is not None
            else 0.0
        )

        pre_filter_reward_mean = all_rewards.mean().item()
        pre_filter_reward_std = all_rewards.std().item() if all_rewards.numel() > 1 else 0.0
        pre_filter_adv_mean = advantages_all.mean().item()

        # Split advantages back per-batch for the gradient-accumulation loop.
        split_sizes = [b.rewards.shape[0] for b in all_batches]
        adv_split = list(torch.split(advantages_all, split_sizes))

        # Zero-advantage sample filter. Flow-GRPO applies this globally across
        # the outer rollout epoch, pads enough zero-advantage rows back in for
        # exact rebatching, then shuffles/rebatches into
        # num_batches_per_epoch training microbatches.
        filtered_batches: list[RolloutBatch] = []
        filtered_advs: list[torch.Tensor] = []
        if cfg.drop_zero_advantage:
            if cfg.gradient_accumulation_steps > 0:
                combined = stack_batches(all_batches)
                mask = pad_zero_advantage_mask(
                    nonzero_advantage_mask(advantages_all),
                    num_batches=len(all_batches),
                )
                if bool(mask.any()):
                    combined = apply_sample_mask(combined, mask)
                    adv_all = advantages_all[mask.to(advantages_all.device)]
                    filtered_batches, filtered_advs = shuffle_and_rebatch_batches(
                        [combined],
                        [adv_all],
                        num_batches=len(all_batches),
                    )
            else:
                for b, adv_b in zip(all_batches, adv_split, strict=True):
                    mask = nonzero_advantage_mask(adv_b)
                    if not bool(mask.any()):
                        continue
                    if not bool(mask.all()):
                        b = apply_sample_mask(b, mask)
                        adv_b = adv_b[mask.to(adv_b.device)]
                    if b.rewards.shape[0] > 0:
                        filtered_batches.append(b)
                        filtered_advs.append(adv_b)
        else:
            filtered_batches = list(all_batches)
            filtered_advs = adv_split
            if cfg.gradient_accumulation_steps > 0:
                filtered_batches, filtered_advs = shuffle_and_rebatch_batches(
                    filtered_batches,
                    filtered_advs,
                    num_batches=len(all_batches),
                )

        # 3. Train loop — gradient accumulation across per-prompt batches.
        self.model.train()
        autocast_ctx = _get_autocast(cfg, self.device, model=self.model)
        agg_metrics: dict[str, list[float]] = defaultdict(list)
        uses_evaluator = bool(getattr(self.algorithm, "uses_evaluator", True))
        algorithm_adapter = AlgorithmAdapter()
        if not uses_evaluator and not callable(getattr(self.algorithm, "compute_loss", None)):
            raise RuntimeError(
                f"{type(self.algorithm).__name__} disabled evaluator use but does "
                "not expose compute_loss(AlgorithmInput)",
            )

        # If every batch was filtered out (all dead), skip training this step.
        if not filtered_batches:
            logger.info(
                "step %d: all batches filtered (zero advantages); skipping backward",
                self.state.step,
            )
            # Early exit — still advance state + return metrics with zeros.
            self.state.step += 1
            reward_mean = pre_filter_reward_mean
            reward_std = pre_filter_reward_std
            return TrainStepMetrics(
                loss=0.0,
                policy_loss=0.0,
                kl_penalty=0.0,
                reward_mean=reward_mean,
                reward_std=reward_std,
                advantage_mean=pre_filter_adv_mean,
                clip_fraction=0.0,
                approx_kl=0.0,
                grad_norm=0.0,
                adv_saturation=adv_saturation,
                adv_zero_rate=adv_zero_rate,
                group_size=group_size,
                trained_prompt_num=trained_prompt_num,
                phase_times={**iteration.phase_times, **dict(timer.times)},
            )

        defer_replay_tensor_move = uses_evaluator and bool(
            getattr(self.evaluator, "supports_deferred_replay_tensor_move", False),
        )
        filtered_batches = [
            move_training_batch_to_device(
                batch,
                self.device,
                defer_replay_tensors=defer_replay_tensor_move,
            )
            for batch in filtered_batches
        ]
        filtered_advs = [adv.to(self.device) for adv in filtered_advs]

        # Timestep schedule — same num_timesteps across all batches (collector
        # uses the same scheduler), so pick from first filtered batch.
        num_timesteps = filtered_batches[0].observations.shape[1]
        train_timestep_count = max(1, int(num_timesteps * cfg.timestep_fraction))
        if train_timestep_count < num_timesteps:
            step_size = num_timesteps / train_timestep_count
            train_indices = [int(i * step_size) for i in range(train_timestep_count)]
        else:
            train_indices = list(range(num_timesteps))

        # Number of rollout micro-batches per optimizer update. ``0`` preserves
        # legacy VRL behavior: one optimizer update after all collected batches.
        grad_accum_batches = int(cfg.gradient_accumulation_steps)
        if grad_accum_batches <= 0 or grad_accum_batches > len(filtered_batches):
            grad_accum_batches = len(filtered_batches)

        # Debug first step: compare old vs fresh log-probs on first timestep
        # (using first filtered batch so memory footprint is bounded).
        first_step_debug_record: dict[str, Any] | None = None
        precision_metadata = _merge_rollout_precision_context(
            _trainer_precision_metadata(cfg, self.device, self.model),
            filtered_batches[0].context,
        )
        if cfg.debug.first_step and self.state.step == 0 and uses_evaluator:
            _dbg_batch = filtered_batches[0]
            with torch.no_grad(), autocast_ctx, record_function("trainer.replay"):
                _dbg_signals = self.evaluator.evaluate(
                    self.model,
                    _dbg_batch,
                    0,
                    ref_model=self.ref_model,
                    signal_request=SignalRequest(need_ref=False, need_kl_intermediates=False),
                )
            if not isinstance(_dbg_signals, TrajectorySignalBatch):
                raise TypeError(
                    "evaluator output must be TrajectorySignalBatch; "
                    f"got {type(_dbg_signals).__name__}",
                )
            _dbg_trajectory_signals = _dbg_signals
            _dbg_log_prob = _dbg_trajectory_signals.primary.log_prob
            _old_lp_0 = _dbg_trajectory_signals.primary.old_log_prob
            _diff = (_dbg_log_prob - _old_lp_0).abs()
            _ratio = torch.exp(_dbg_log_prob - _old_lp_0)
            _old_lp_first = _old_lp_0.reshape(-1)[0]
            _fresh_lp_first = _dbg_log_prob.reshape(-1)[0]
            logger.info(
                "DEBUG first-step log-prob diff: mean=%.6f max=%.6f | "
                "old_lp[0]=%.6f fresh_lp[0]=%.6f",
                _diff.mean().item(),
                _diff.max().item(),
                _old_lp_first.item(),
                _fresh_lp_first.item(),
            )
            first_step_debug_record = {
                "event": "first_step_logprob_parity",
                "trainer_step": int(self.state.step),
                "global_step": int(self.state.global_step),
                "device": str(self.device),
                "mixed_precision": _resolve_mixed_precision(cfg),
                "autocast_enabled": precision_metadata["trainer_autocast_enabled"],
                "precision_policy": precision_metadata,
                "old_log_prob": tensor_stats(_old_lp_0),
                "fresh_log_prob": tensor_stats(_dbg_log_prob),
                "abs_diff": tensor_stats(_diff),
                "ratio": tensor_stats(_ratio),
                "driver_trainable_before_step": trainable_state_digest(self.model),
                "driver_parameter_state_before_step": parameter_state_summary(self.model),
                "rollout_context": {
                    key: value
                    for key, value in _dbg_batch.context.items()
                    if key != "runtime_debug"
                },
                "runtime_debug": _dbg_batch.context.get("runtime_debug"),
            }

        # Precision drift guard: on the first step (before any optimizer update),
        # check rollout-vs-replay logprob parity. `auto` protects unsafe precision
        # splits; explicit warn/fail is used for same-precision acceptance runs.
        if self.state.step == 0 and uses_evaluator and self.evaluator is not None:
            _guard_batch = filtered_batches[0]

            def _guard_evaluate(timestep_idx: int) -> TrajectorySignalBatch:
                with torch.no_grad(), autocast_ctx, record_function("trainer.replay"):
                    _sig = self.evaluator.evaluate(
                        self.model,
                        _guard_batch,
                        timestep_idx,
                        ref_model=self.ref_model,
                        signal_request=SignalRequest(
                            need_ref=False, need_kl_intermediates=False,
                        ),
                    )
                if not isinstance(_sig, TrajectorySignalBatch):
                    raise TypeError(
                        "evaluator output must be TrajectorySignalBatch; "
                        f"got {type(_sig).__name__}",
                    )
                return _sig

            _guard_record = run_precision_drift_guard(
                cfg.precision_drift_guard,
                compute_precision=_resolve_mixed_precision(cfg),
                rollout_precision=cfg.rollout_precision or _resolve_mixed_precision(cfg),
                math_precision=cfg.math_precision,
                timestep_indices=train_indices,
                evaluate_fn=_guard_evaluate,
                metadata=precision_metadata,
                logger=logger,
            )
            if _guard_record is not None and first_step_debug_record is not None:
                first_step_debug_record["precision_drift_guard"] = _guard_record

        for _inner_epoch in range(cfg.ppo_epochs):
            # Accumulate a configurable number of rollout micro-batches per
            # optimizer update. Flow-GRPO sets this to num_batches_per_epoch//2,
            # so an epoch can intentionally contain multiple optimizer updates.
            for batch_start in range(0, len(filtered_batches), grad_accum_batches):
                chunk_batches = filtered_batches[batch_start : batch_start + grad_accum_batches]
                chunk_advs = filtered_advs[batch_start : batch_start + grad_accum_batches]
                # Flow-GRPO uses Accelerate accumulation over both rollout
                # microbatches and denoising timesteps:
                # gradient_accumulation_steps = microbatches * timesteps.
                # Mirror that normalization explicitly in the native trainer.
                loss_scale = len(chunk_batches) * len(train_indices)

                for b, adv_b in zip(chunk_batches, chunk_advs, strict=True):
                    for j in train_indices:
                        with timer.time("evaluate"), autocast_ctx:
                            if not uses_evaluator:
                                with record_function("trainer.loss"):
                                    loss, metrics = algorithm_adapter.compute_loss(
                                        self.algorithm,
                                        AlgorithmInput(
                                            rewards=b.rewards,
                                            group_ids=b.group_ids,
                                            advantages=adv_b,
                                            metadata={
                                                "model": self.model,
                                                "rollout_batch": b,
                                                "timestep_index": j,
                                            },
                                        ),
                                    )
                            else:
                                if self.evaluator is None:
                                    raise RuntimeError(
                                        f"{type(self.algorithm).__name__} requires an evaluator",
                                    )
                                init_kl_coef = float(
                                    getattr(self.algorithm.config, "init_kl_coef", 0.0),
                                )
                                with record_function("trainer.replay"):
                                    signals = self.evaluator.evaluate(
                                        self.model,
                                        b,
                                        j,
                                        ref_model=self.ref_model,
                                        signal_request=SignalRequest(
                                            need_ref=init_kl_coef > 0,
                                            need_kl_intermediates=init_kl_coef > 0,
                                        ),
                                    )
                                with record_function("trainer.loss"):
                                    if not isinstance(signals, TrajectorySignalBatch):
                                        raise TypeError(
                                            "evaluator output must be TrajectorySignalBatch; "
                                            f"got {type(signals).__name__}",
                                        )
                                    trajectory_signals = signals
                                    loss, metrics = algorithm_adapter.compute_loss(
                                        self.algorithm,
                                        AlgorithmInput(
                                            signals=trajectory_signals,
                                            advantages=adv_b,
                                            group_ids=b.group_ids,
                                        ),
                                    )
                            # Average across rollout micro-batches inside this
                            # optimizer update; timestep accumulation follows
                            # Flow-GRPO's per-denoise-step surrogate structure.
                            loss = loss / loss_scale

                        with timer.time("backward"):
                            self._backward(loss)

                        self._clear_algorithm_diagnostics()

                        agg_metrics["loss"].append(metrics.loss)
                        agg_metrics["policy_loss"].append(metrics.policy_loss)
                        agg_metrics["kl_penalty"].append(metrics.kl_penalty)
                        agg_metrics["clip_fraction"].append(metrics.clip_fraction)
                        agg_metrics["approx_kl"].append(metrics.approx_kl)
                        agg_metrics["logprob_abs_diff_mean"].append(metrics.logprob_abs_diff_mean)
                        agg_metrics["logprob_abs_diff_max"].append(metrics.logprob_abs_diff_max)
                        agg_metrics["ratio_abs_dev_mean"].append(metrics.ratio_abs_dev_mean)
                        agg_metrics["ratio_abs_dev_max"].append(metrics.ratio_abs_dev_max)
                        agg_metrics["mismatch_kl"].append(metrics.mismatch_kl)
                        agg_metrics["mismatch_k3_kl"].append(metrics.mismatch_k3_kl)

                        # Continuous rollout production runs on the same asyncio
                        # loop as training orchestration. Yield once per
                        # timestep so producer admit/harvest can progress while
                        # this synchronous CUDA-heavy loop is still computing.
                        await asyncio.sleep(0)

                with timer.time("optim_step"):
                    _gn = self._clip_and_step(optimizer)
                    agg_metrics["grad_norm"].append(_gn)

                after_optimizer_step = getattr(self.algorithm, "after_optimizer_step", None)
                if callable(after_optimizer_step):
                    after_optimizer_step(self.model, self.state.global_step)

                if ema is not None:
                    trainable = [p for p in self.model.parameters() if p.requires_grad]
                    ema.step(trainable, self.state.global_step)

                self.state.global_step += 1

        # Aggregate metrics — each metric averages over its own count (loss/policy
        # appended per-timestep, grad_norm appended per-inner-epoch).
        def avg(key: str) -> float:
            vals = agg_metrics.get(key, [])
            return sum(vals) / len(vals) if vals else 0.0

        def mx(key: str) -> float:
            vals = agg_metrics.get(key, [])
            return max(vals) if vals else 0.0

        reward_mean = pre_filter_reward_mean
        reward_std = pre_filter_reward_std
        adv_mean = pre_filter_adv_mean

        phase_times = dict(iteration.phase_times)
        for key, value in timer.times.items():
            phase_times[key] = phase_times.get(key, 0.0) + value
        if cfg.profile:
            phase_times.update(getattr(self.collector, "last_collect_phases", {}))
        if cfg.profile and phase_times:
            total = sum(v for k, v in phase_times.items() if not k.startswith("collect."))
            parts = " | ".join(
                f"{k}={v:.3f}s ({100 * v / total:.1f}%)" for k, v in phase_times.items()
            )
            logger.info("phase_times[step=%d] total=%.3fs | %s", self.state.step, total, parts)
            try:
                import json
                import os

                _evt_path = os.path.join(cfg.output_dir, "phase_events.jsonl")
                os.makedirs(cfg.output_dir, exist_ok=True)
                with open(_evt_path, "a") as _f:
                    for _n, _s, _e in timer.events:
                        _f.write(
                            json.dumps(
                                {
                                    "step": self.state.step,
                                    "phase": _n,
                                    "start": _s,
                                    "end": _e,
                                }
                            )
                            + "\n"
                        )
                timer.events.clear()
            except Exception:
                pass

        metrics = TrainStepMetrics(
            loss=avg("loss"),
            policy_loss=avg("policy_loss"),
            kl_penalty=avg("kl_penalty"),
            reward_mean=reward_mean,
            reward_std=reward_std,
            advantage_mean=adv_mean,
            clip_fraction=avg("clip_fraction"),
            approx_kl=avg("approx_kl"),
            logprob_abs_diff_mean=avg("logprob_abs_diff_mean"),
            logprob_abs_diff_max=mx("logprob_abs_diff_max"),
            ratio_abs_dev_mean=avg("ratio_abs_dev_mean"),
            ratio_abs_dev_max=mx("ratio_abs_dev_max"),
            mismatch_kl=avg("mismatch_kl"),
            mismatch_k3_kl=avg("mismatch_k3_kl"),
            grad_norm=avg("grad_norm"),
            adv_saturation=adv_saturation,
            adv_zero_rate=adv_zero_rate,
            group_size=group_size,
            trained_prompt_num=trained_prompt_num,
            phase_times=phase_times,
        )

        # Update state
        self.state.step += 1
        self.state.total_reward += metrics.reward_mean
        self.state.total_loss += metrics.loss

        sync_phase_times = await self.rollout_schedule.after_train_step()
        for key, value in sync_phase_times.items():
            phase_times[key] = phase_times.get(key, 0.0) + value

        if first_step_debug_record is not None:
            first_step_debug_record["driver_trainable_after_step"] = trainable_state_digest(
                self.model
            )
            first_step_debug_record["driver_parameter_state_after_step"] = parameter_state_summary(
                self.model
            )
            first_step_debug_record["post_step_global_step"] = int(self.state.global_step)
            write_jsonl(
                f"{cfg.output_dir}/training_debug.jsonl",
                first_step_debug_record,
            )

        return metrics

    def _clear_algorithm_diagnostics(self) -> None:
        for attr in ("_last_policy_loss_tensor", "_last_kl_term_tensor"):
            if hasattr(self.algorithm, attr):
                setattr(self.algorithm, attr, None)

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d: dict[str, Any] = {
            "step": self.state.step,
            "global_step": self.state.global_step,
            "total_reward": self.state.total_reward,
            "total_loss": self.state.total_loss,
        }
        if self._optimizer is not None:
            d["optimizer"] = self._optimizer.state_dict()
        if self._ema is not None:
            d["ema"] = self._ema.state_dict()
        return d

    def load_state_dict(self, state: dict, *, strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("OnlineTrainer.load_state_dict expects a dict")

        self.state.step = int(state.get("step", 0))
        self.state.global_step = int(state.get("global_step", 0))
        self.state.total_reward = float(state.get("total_reward", 0.0))
        self.state.total_loss = float(state.get("total_loss", 0.0))

        if "optimizer" in state:
            optimizer = self._ensure_optimizer()
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception:
                if strict:
                    raise
                logger.warning("Skipping incompatible optimizer state during non-strict load")

        if "ema" in state:
            if not self.config.ema.enable:
                if strict:
                    raise ValueError("checkpoint contains EMA state but trainer.ema.enable=false")
                logger.warning("Skipping EMA state because trainer.ema.enable=false")
            else:
                ema = self._ensure_ema()
                assert ema is not None
                _validate_ema_state_shapes(
                    state["ema"],
                    self.model,
                    strict=strict,
                )
                try:
                    ema.load_state_dict(state["ema"])
                except Exception:
                    if strict:
                        raise
                    logger.warning("Skipping incompatible EMA state during non-strict load")
        elif strict and self.config.ema.enable:
            raise ValueError("checkpoint missing EMA state but trainer.ema.enable=true")

        # A resumed trainer must push the restored driver weights before the
        # next Ray rollout. The policy version and worker state are runtime
        # concerns, not persisted as an initialized rollout flag.
        self._rollout_weights_initialized = False
        self.rollout_schedule.reset()


def _validate_ema_state_shapes(
    ema_state: dict[str, Any],
    model: nn.Module,
    *,
    strict: bool,
) -> None:
    ema_parameters = ema_state.get("ema_parameters") if isinstance(ema_state, dict) else None
    if not isinstance(ema_parameters, list):
        if strict:
            raise ValueError("checkpoint EMA state missing ema_parameters")
        return
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(ema_parameters) != len(trainable):
        if strict:
            raise ValueError(
                "checkpoint EMA parameter count mismatch: "
                f"checkpoint={len(ema_parameters)} current={len(trainable)}",
            )
        return
    for idx, (ema_param, param) in enumerate(zip(ema_parameters, trainable, strict=True)):
        if not isinstance(ema_param, torch.Tensor):
            if strict:
                raise ValueError(f"checkpoint EMA parameter {idx} is not a tensor")
            return
        if tuple(ema_param.shape) != tuple(param.shape):
            if strict:
                raise ValueError(
                    "checkpoint EMA parameter shape mismatch at index "
                    f"{idx}: checkpoint={tuple(ema_param.shape)} "
                    f"current={tuple(param.shape)}",
                )
            return
