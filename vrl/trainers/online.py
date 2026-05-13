"""Online RL trainer — CEA pipeline (Collector + Evaluator + Algorithm).

collect -> evaluate -> advantage -> loss -> backward -> step.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import TrainStepMetrics
from vrl.engine.trajectory.ops import move_trajectory_batch, select_trajectory_batch
from vrl.rollouts.batch import RolloutBatch, stack_batches
from vrl.trainers.base import Trainer
from vrl.trainers.diagnostics import (
    parameter_state_summary,
    tensor_stats,
    trainable_state_digest,
    write_jsonl,
)
from vrl.trainers.ema import EMAModuleWrapper
from vrl.trainers.types import TrainerConfig, TrainState
from vrl.trainers.weight_sync import WeightSyncer

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
    if optim.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as err:
            raise ImportError(
                "Install bitsandbytes for 8-bit Adam: pip install bitsandbytes"
            ) from err
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    return optimizer_cls(
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
    precision = str(config.mixed_precision or "").lower().strip()
    if not precision:
        precision = "bf16" if config.bf16 else "no"
    aliases = {
        "none": "no",
        "off": "no",
        "fp32": "no",
        "float32": "no",
        "float16": "fp16",
        "bfloat16": "bf16",
    }
    precision = aliases.get(precision, precision)
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "actor.mixed_precision must be one of 'no', 'fp16', or 'bf16', "
            f"got {config.mixed_precision!r}",
        )
    return precision


def _get_autocast(
    config: TrainerConfig,
    device: torch.device,
    model: Any | None = None,
) -> Any:
    """Return the configured autocast context manager."""
    if bool(getattr(model, "disable_train_autocast", False)):
        return contextlib.nullcontext()
    precision = _resolve_mixed_precision(config)
    if precision == "bf16":
        return torch.amp.autocast(str(device), dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.amp.autocast(str(device), dtype=torch.float16)
    return contextlib.nullcontext()


def _select_batch(batch: RolloutBatch, selector: torch.Tensor) -> RolloutBatch:
    """Select RolloutBatch rows by a boolean mask or long indices.

    All per-sample tensors (observations, actions, rewards, dones, group_ids,
    videos, and extras whose leading dim matches the batch) are indexed by
    ``selector``. Non-per-sample extras and context are carried through unchanged.
    """
    selector = selector.detach()
    new_extras: dict[str, Any] = {}
    batch_size = batch.rewards.shape[0]
    for k, v in batch.extras.items():
        new_extras[k] = _select_tensor_tree(v, selector, batch_size)
    videos = (
        batch.videos[selector.to(batch.videos.device)]
        if batch.videos is not None
        else None
    )
    if batch.prompts is not None:
        selector_cpu = selector.cpu()
        if selector_cpu.dtype == torch.bool:
            positions = torch.where(selector_cpu)[0].tolist()
        else:
            positions = [int(i) for i in selector_cpu.reshape(-1).tolist()]
        prompts = [batch.prompts[i] for i in positions]
    else:
        prompts = None
    return RolloutBatch(
        observations=batch.observations[selector.to(batch.observations.device)],
        actions=batch.actions[selector.to(batch.actions.device)],
        rewards=batch.rewards[selector.to(batch.rewards.device)],
        dones=batch.dones[selector.to(batch.dones.device)],
        group_ids=batch.group_ids[selector.to(batch.group_ids.device)],
        extras=new_extras,
        context=batch.context,
        videos=videos,
        prompts=prompts,
        trajectory=select_trajectory_batch(batch.trajectory, selector),
        training_view=batch.training_view,
    )


def _select_tensor_tree(value: Any, selector: torch.Tensor, batch_size: int) -> Any:
    """Select per-sample tensor leaves inside nested rollout metadata."""

    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == batch_size:
            return value[selector.to(value.device)]
        return value
    if isinstance(value, dict):
        return {
            key: _select_tensor_tree(inner, selector, batch_size)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_select_tensor_tree(inner, selector, batch_size) for inner in value]
    if isinstance(value, tuple):
        return tuple(_select_tensor_tree(inner, selector, batch_size) for inner in value)
    return value


def _apply_sample_mask(batch: RolloutBatch, mask: torch.Tensor) -> RolloutBatch:
    """Filter RolloutBatch along sample dim by a boolean mask."""

    return _select_batch(batch, mask)


def _nonzero_advantage_mask(advantages: torch.Tensor) -> torch.Tensor:
    """Return Flow-GRPO's mask for samples with non-zero total advantage."""

    adv_abs = advantages.detach().abs()
    if adv_abs.dim() <= 1:
        return adv_abs != 0
    reduce_dims = tuple(range(1, adv_abs.dim()))
    return adv_abs.sum(dim=reduce_dims) != 0


def _pad_zero_advantage_mask(mask: torch.Tensor, num_batches: int) -> torch.Tensor:
    """Pad zero-advantage samples back in so rebatching divides evenly.

    Flow-GRPO filters all-zero-advantage samples, then randomly re-includes
    enough zero-advantage rows to make the remaining batch count divisible by
    ``num_batches_per_epoch``. That keeps the later reshape/rebatch step exact.
    """

    if num_batches <= 0:
        return mask
    padded = mask.clone()
    true_count = int(padded.sum().item())
    if true_count == 0 or true_count % num_batches == 0:
        return padded
    false_indices = torch.where(~padded)[0]
    num_to_change = num_batches - (true_count % num_batches)
    if false_indices.numel() >= num_to_change:
        random_indices = torch.randperm(false_indices.numel(), device=false_indices.device)[
            :num_to_change
        ]
        padded[false_indices[random_indices]] = True
    return padded


def _shuffle_and_rebatch_batches(
    batches: list[RolloutBatch],
    advantages: list[torch.Tensor],
    *,
    num_batches: int,
) -> tuple[list[RolloutBatch], list[torch.Tensor]]:
    """Shuffle across all rollout rows and split into Flow-GRPO microbatches."""

    combined = stack_batches(batches)
    adv_all = torch.cat(advantages)
    total_batch_size = int(combined.rewards.shape[0])
    if total_batch_size == 0:
        return [], []
    if num_batches <= 0 or total_batch_size % num_batches != 0:
        raise ValueError(
            "Flow-GRPO rebatch requires total samples divisible by rollout batches: "
            f"total_batch_size={total_batch_size}, num_batches={num_batches}",
        )

    perm = torch.randperm(total_batch_size, device=combined.rewards.device)
    combined = _select_batch(combined, perm)
    adv_all = adv_all[perm.to(adv_all.device)]

    microbatch_size = total_batch_size // num_batches
    rebatches: list[RolloutBatch] = []
    rebatch_advs: list[torch.Tensor] = []
    for start in range(0, total_batch_size, microbatch_size):
        idx = torch.arange(
            start,
            start + microbatch_size,
            device=combined.rewards.device,
        )
        rebatches.append(_select_batch(combined, idx))
        rebatch_advs.append(adv_all[idx.to(adv_all.device)])
    return rebatches, rebatch_advs


def _split_batch_by_group(batch: RolloutBatch) -> list[RolloutBatch]:
    """Split a rollout batch into group-local batches for bounded training memory."""

    group_ids = batch.group_ids
    ordered_ids: list[int] = []
    seen: set[int] = set()
    for group_id in group_ids.detach().cpu().tolist():
        gid = int(group_id)
        if gid not in seen:
            seen.add(gid)
            ordered_ids.append(gid)
    if len(ordered_ids) <= 1:
        return [batch]
    return [_apply_sample_mask(batch, group_ids == group_id) for group_id in ordered_ids]


def _remap_group_ids_(batch: RolloutBatch, global_prompt_indices: list[int]) -> None:
    """Map collector-local prompt groups back to trainer-global prompt indices."""

    if not global_prompt_indices:
        return
    remapped = batch.group_ids.clone()
    for local_idx, global_idx in enumerate(global_prompt_indices):
        remapped[batch.group_ids == local_idx] = global_idx
    batch.group_ids = remapped
    if batch.trajectory is not None:
        trajectory_group_ids = batch.trajectory.group_ids
        device = getattr(trajectory_group_ids, "device", remapped.device)
        batch.trajectory.group_ids = remapped.to(device)


def _move_tensor_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tensor_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tensor_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(v, device) for v in value)
    return value


def _move_training_batch_to_device(batch: RolloutBatch, device: torch.device) -> RolloutBatch:
    """Move replay tensors to the trainer device.

    Ray rollout workers return CPU tensors so the driver can gather chunks
    without owning the worker GPU memory. Training replay runs on the driver
    policy, so latent trajectories, actions, log-probs, embeds, and timesteps
    must move to the trainer device before evaluator/model forward. Videos stay
    on CPU because reward scoring already consumed them and replay does not use
    decoded frames.
    """

    return RolloutBatch(
        observations=batch.observations.to(device),
        actions=batch.actions.to(device),
        rewards=batch.rewards.to(device),
        dones=batch.dones.to(device),
        group_ids=batch.group_ids.to(device),
        extras=_move_tensor_tree(batch.extras, device),
        context=_move_tensor_tree(batch.context, device),
        videos=batch.videos,
        prompts=batch.prompts,
        trajectory=move_trajectory_batch(batch.trajectory, device),
        training_view=batch.training_view,
    )


async def _release_collector_runtime_memory(collector: Any) -> None:
    release = getattr(collector, "release_runtime_memory", None)
    if callable(release):
        await release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collector_runtime_requires_driver_model_offload(collector: Any) -> bool:
    try:
        runtime = collector.runtime
    except Exception:
        runtime = getattr(collector, "_runtime", None)
    return bool(getattr(runtime, "requires_driver_model_offload", False))


def _move_model_to_device(model: nn.Module, device: torch.device | str) -> None:
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _algorithm_mask_key(algorithm: Any) -> str:
    config = getattr(algorithm, "config", None)
    return str(getattr(config, "mask_key", "token_mask"))


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
        config: TrainerConfig | None = None,
        prompts: list[str] | None = None,
        device: torch.device | str = "cuda",
        accelerator: Any | None = None,
        stat_tracker: Any | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.collector = collector
        self.evaluator = evaluator
        self.model = model
        self.ref_model = ref_model
        self.weight_syncer = weight_syncer
        self.config = config or TrainerConfig()
        self.prompts = prompts or []
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state = TrainState()
        self.accelerator = accelerator
        # Optional per-prompt history stat tracker (e.g. PerPromptStatTracker).
        # When set, trainer uses tracker-derived advantages (long-horizon
        # normalization) and applies zero-advantage sample filtering.
        self.stat_tracker = stat_tracker

        self._optimizer: torch.optim.Optimizer | None = None
        self._ema: EMAModuleWrapper | None = None
        self._rollout_weights_initialized = False

        if self.config.optim.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

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

    async def _ensure_rollout_weights_initialized(self) -> None:
        """Synchronize driver weights before the first remote rollout.

        Ray workers build their own model/adapters. Without an explicit initial
        sync, rollout old log-probs can come from a different random adapter
        initialization than the driver replay path.
        """

        if self._rollout_weights_initialized or self.weight_syncer is None:
            return
        await self.weight_syncer.push(self.model.state_dict())
        self._rollout_weights_initialized = True

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
        from vrl.trainers.profiling import torch_profiler_step

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
        from vrl.rollouts.evaluators.trajectory import (
            to_trajectory_signals,
        )
        from vrl.rollouts.evaluators.types import SignalRequest
        from vrl.trainers.data import PromptExample
        from vrl.trainers.profiling import record_function

        if prompts is not None:
            self.prompts = prompts

        cfg = self.config
        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()

        timer = PhaseTimer(enabled=cfg.profile)
        runtime_debug_collect = bool(cfg.debug.first_step and self.state.step == 0)

        await self._ensure_rollout_weights_initialized()

        offload_driver_model_for_rollout = (
            self.device.type == "cuda"
            and _collector_runtime_requires_driver_model_offload(self.collector)
        )
        if offload_driver_model_for_rollout:
            with timer.time("offload_driver_model"):
                _move_model_to_device(self.model, "cpu")

        # 1. Collect group_size samples per prompt.
        # Plain string prompts are batched into one rollout request so the
        # executor can plan prompt x group_size micro-batches directly. Rich
        # PromptExample items stay single-prompt because reward metadata and
        # request_overrides are per prompt.
        all_batches: list[RolloutBatch] = []
        with timer.time("collect"):
            pending_prompts: list[str] = []
            pending_indices: list[int] = []

            async def flush_pending_prompts() -> None:
                if not pending_prompts:
                    return
                collect_kwargs: dict[str, Any] = {"group_size": cfg.n}
                if runtime_debug_collect:
                    collect_kwargs["runtime_debug"] = True
                b = await self.collector.collect(
                    list(pending_prompts),
                    **collect_kwargs,
                )
                _remap_group_ids_(b, pending_indices)
                all_batches.extend(_split_batch_by_group(b))
                pending_prompts.clear()
                pending_indices.clear()

            for prompt_idx, item in enumerate(self.prompts):
                if isinstance(item, PromptExample):
                    await flush_pending_prompts()
                    prompt_str = item.prompt
                    collect_kwargs: dict[str, Any] = {
                        "target_text": item.target_text,
                        "references": item.references,
                        "task_type": item.task_type,
                        "request_overrides": item.request_overrides,
                        "sample_metadata": item.metadata,
                    }
                    if item.reference_image:
                        collect_kwargs["reference_image"] = item.reference_image
                    if item.reference_video:
                        collect_kwargs["reference_video"] = item.reference_video
                    if runtime_debug_collect:
                        collect_kwargs["runtime_debug"] = True
                    # Group-batched collect: one call produces cfg.n samples.
                    b = await self.collector.collect(
                        [prompt_str],
                        group_size=cfg.n,
                        **collect_kwargs,
                    )
                    b.group_ids[:] = prompt_idx
                    all_batches.extend(_split_batch_by_group(b))
                else:
                    pending_prompts.append(str(item))
                    pending_indices.append(prompt_idx)

            await flush_pending_prompts()

            # Gradient accumulation: collect can run prompt x group_size as one
            # large rollout request, then this trainer splits by group so each
            # forward/backward still sees only one prompt group. That keeps
            # training memory bounded while rollout gets a larger execution plan.

        with timer.time("release_rollout"):
            await _release_collector_runtime_memory(self.collector)

        if offload_driver_model_for_rollout:
            with timer.time("restore_driver_model"):
                _move_model_to_device(self.model, self.device)

        # 2. Compute advantages (per-prompt normalization).
        # Rewards + prompts are concatenated across all collected batches so
        # the tracker sees every prompt together and groups properly. The
        # resulting advantages are then split back per-batch for the
        # per-batch training loop.
        tracker_group_size: float = 0.0
        tracker_trained_prompt_num: int = 0
        with timer.time("advantage"):
            all_rewards = torch.cat([b.rewards for b in all_batches])
            all_prompts_flat: list[str] = []
            for b in all_batches:
                assert b.prompts is not None, "batch.prompts must be populated"
                all_prompts_flat.extend(b.prompts)
            all_group_ids = torch.cat([b.group_ids for b in all_batches])

            if self.stat_tracker is not None:
                adv_np = self.stat_tracker.update(all_prompts_flat, all_rewards)
                tracker_group_size, tracker_trained_prompt_num = self.stat_tracker.get_stats()
                self.stat_tracker.clear()
                advantages_all = torch.as_tensor(
                    adv_np,
                    dtype=torch.float32,
                    device=all_rewards.device,
                )
                adv_clip_max = getattr(self.algorithm.config, "adv_clip_max", None)
                if adv_clip_max is not None:
                    advantages_all = torch.clamp(advantages_all, -adv_clip_max, adv_clip_max)
            else:
                advantages_all = self.algorithm.compute_advantages_from_tensors(
                    all_rewards,
                    all_group_ids,
                )

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
        if self.stat_tracker is not None:
            if cfg.gradient_accumulation_steps > 0:
                combined = stack_batches(all_batches)
                mask = _pad_zero_advantage_mask(
                    _nonzero_advantage_mask(advantages_all),
                    num_batches=len(all_batches),
                )
                if bool(mask.any()):
                    combined = _apply_sample_mask(combined, mask)
                    adv_all = advantages_all[mask.to(advantages_all.device)]
                    filtered_batches, filtered_advs = _shuffle_and_rebatch_batches(
                        [combined],
                        [adv_all],
                        num_batches=len(all_batches),
                    )
            else:
                for b, adv_b in zip(all_batches, adv_split, strict=True):
                    mask = _nonzero_advantage_mask(adv_b)
                    if not bool(mask.all()):
                        b = _apply_sample_mask(b, mask)
                        adv_b = adv_b[mask.to(adv_b.device)]
                    if b.rewards.shape[0] > 0:
                        filtered_batches.append(b)
                        filtered_advs.append(adv_b)
        else:
            filtered_batches = list(all_batches)
            filtered_advs = adv_split
            if cfg.gradient_accumulation_steps > 0:
                filtered_batches, filtered_advs = _shuffle_and_rebatch_batches(
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
                group_size=tracker_group_size,
                trained_prompt_num=tracker_trained_prompt_num,
                phase_times=dict(timer.times),
            )

        filtered_batches = [
            _move_training_batch_to_device(batch, self.device)
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
        if cfg.debug.first_step and self.state.step == 0 and uses_evaluator:
            _dbg_batch = filtered_batches[0]
            with autocast_ctx, record_function("trainer.replay"):
                _dbg_signals = self.evaluator.evaluate(
                    self.model,
                    _dbg_batch,
                    0,
                    ref_model=self.ref_model,
                    signal_request=SignalRequest(need_ref=False, need_kl_intermediates=False),
                )
            _dbg_trajectory_signals = to_trajectory_signals(_dbg_signals)
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
                "autocast_enabled": _resolve_mixed_precision(cfg) != "no",
                "old_log_prob": tensor_stats(_old_lp_0),
                "fresh_log_prob": tensor_stats(_dbg_log_prob),
                "abs_diff": tensor_stats(_diff),
                "ratio": tensor_stats(_ratio),
                "driver_trainable_before_step": trainable_state_digest(self.model),
                "driver_parameter_state_before_step": parameter_state_summary(self.model),
                "runtime_debug": _dbg_batch.context.get("runtime_debug"),
            }

        if cfg.debug.grad_split:
            import sys

            _msg = (
                f"\n[GRAD-SPLIT TRACER] about to enter inner loop: "
                f"step={self.state.step} ppo_epochs={cfg.ppo_epochs} "
                f"num_filtered_batches={len(filtered_batches)} "
                f"grad_accum_batches={grad_accum_batches} "
                f"num_train_indices={len(train_indices)}\n"
            )
            print(_msg, file=sys.stderr, flush=True)
            print(_msg, flush=True)
            logger.info(_msg.strip())
            try:
                with open("/tmp/grad_split_debug.log", "a") as _f:
                    _f.write(_msg)
            except Exception:
                pass
        for _inner_epoch in range(cfg.ppo_epochs):
            # Accumulate a configurable number of rollout micro-batches per
            # optimizer update. Flow-GRPO sets this to num_batches_per_epoch//2,
            # so an epoch can intentionally contain multiple optimizer updates.
            for batch_start in range(0, len(filtered_batches), grad_accum_batches):
                chunk_batches = filtered_batches[batch_start:batch_start + grad_accum_batches]
                chunk_advs = filtered_advs[batch_start:batch_start + grad_accum_batches]
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
                                            trajectory=b.trajectory,
                                            training_view=b.training_view,
                                            rewards=b.rewards,
                                            group_ids=b.group_ids,
                                            advantages=adv_b,
                                            metadata={
                                                "model": self.model,
                                                "rollout_batch": b,
                                                "timestep_index": j,
                                                "signal_context": b.context,
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
                                    mask_key = _algorithm_mask_key(self.algorithm)
                                    trajectory_signals = to_trajectory_signals(signals)
                                    loss, metrics = algorithm_adapter.compute_loss(
                                        self.algorithm,
                                        AlgorithmInput(
                                            trajectory=b.trajectory,
                                            training_view=b.training_view,
                                            signals=trajectory_signals,
                                            advantages=adv_b,
                                            old_log_probs=trajectory_signals.primary.old_log_prob,
                                            group_ids=b.group_ids,
                                            metadata={
                                                "signal_context": b.context,
                                                "mask_key": mask_key,
                                            },
                                        ),
                                    )
                            # Average across rollout micro-batches inside this
                            # optimizer update; timestep accumulation follows
                            # Flow-GRPO's per-denoise-step surrogate structure.
                            loss = loss / loss_scale

                        # Grad-split diagnostic: fire ONCE per process on first
                        # backward we actually reach, to verify the KL term is
                        # not drowning the policy gradient. Stamps a class flag
                        # to ensure single-shot.
                        _grad_split_fired = getattr(
                            OnlineTrainer,
                            "_grad_split_already_fired",
                            False,
                        )
                        if cfg.debug.grad_split and not _grad_split_fired:
                            OnlineTrainer._grad_split_already_fired = True  # type: ignore[attr-defined]
                            import sys

                            _enter = (
                                f"\n[GRAD-SPLIT] entering diagnostic block "
                                f"(step={self.state.step}, j={j})\n"
                            )
                            print(_enter, file=sys.stderr, flush=True)
                            print(_enter, flush=True)
                            logger.info(_enter.strip())
                            try:
                                with open("/tmp/grad_split_debug.log", "a") as _f:
                                    _f.write(_enter)
                            except Exception:
                                pass
                            try:
                                p_t = getattr(self.algorithm, "_last_policy_loss_tensor", None)
                                k_t = getattr(self.algorithm, "_last_kl_term_tensor", None)
                                params = [p for p in self.model.parameters() if p.requires_grad]
                                p_norm = float("nan")
                                k_norm = float("nan")
                                if p_t is not None and p_t.requires_grad:
                                    p_grads = torch.autograd.grad(
                                        p_t,
                                        params,
                                        retain_graph=True,
                                        allow_unused=True,
                                    )
                                    p_norm = (
                                        sum(
                                            (g.detach() ** 2).sum().item()
                                            for g in p_grads
                                            if g is not None
                                        )
                                    ) ** 0.5
                                if k_t is not None and k_t.requires_grad:
                                    k_grads = torch.autograd.grad(
                                        k_t,
                                        params,
                                        retain_graph=True,
                                        allow_unused=True,
                                    )
                                    k_norm = (
                                        sum(
                                            (g.detach() ** 2).sum().item()
                                            for g in k_grads
                                            if g is not None
                                        )
                                    ) ** 0.5
                                ratio = p_norm / k_norm if k_norm and k_norm > 0 else float("inf")
                                _result = (
                                    f"\n[GRAD-SPLIT RESULT] step={self.state.step} j={j} "
                                    f"||grad(policy)||={p_norm:.4e} "
                                    f"||grad(beta*kl)||={k_norm:.4e} "
                                    f"policy/kl_ratio={ratio:.3f} "
                                    f"policy_loss={p_t.item() if p_t is not None else float('nan'):.4e} "
                                    f"kl_term={k_t.item() if k_t is not None else float('nan'):.4e}\n"
                                )
                                import sys

                                print(_result, file=sys.stderr, flush=True)
                                print(_result, flush=True)
                                logger.info(_result.strip())
                                try:
                                    with open("/tmp/grad_split_debug.log", "a") as _f:
                                        _f.write(_result)
                                except Exception:
                                    pass
                            except Exception as _e:
                                logger.warning("debug_grad_split failed: %s", _e)

                        with timer.time("backward"):
                            self._backward(loss)

                        self._clear_algorithm_diagnostics()

                        agg_metrics["loss"].append(metrics.loss)
                        agg_metrics["policy_loss"].append(metrics.policy_loss)
                        agg_metrics["kl_penalty"].append(metrics.kl_penalty)
                        agg_metrics["clip_fraction"].append(metrics.clip_fraction)
                        agg_metrics["approx_kl"].append(metrics.approx_kl)

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

        reward_mean = pre_filter_reward_mean
        reward_std = pre_filter_reward_std
        adv_mean = pre_filter_adv_mean

        phase_times = dict(timer.times)
        if cfg.profile:
            try:
                from vrl.rollouts.collector import LAST_COLLECT_PHASES

                phase_times.update(LAST_COLLECT_PHASES)
            except Exception:
                pass
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
            grad_norm=avg("grad_norm"),
            adv_saturation=adv_saturation,
            adv_zero_rate=adv_zero_rate,
            group_size=tracker_group_size,
            trained_prompt_num=tracker_trained_prompt_num,
            phase_times=phase_times,
        )

        # Update state
        self.state.step += 1
        self.state.total_reward += metrics.reward_mean
        self.state.total_loss += metrics.loss

        # Sync weights
        if self.weight_syncer is not None:
            state_dict = self.model.state_dict()
            await self.weight_syncer.push(state_dict)

        if first_step_debug_record is not None:
            first_step_debug_record["driver_trainable_after_step"] = (
                trainable_state_digest(self.model)
            )
            first_step_debug_record["driver_parameter_state_after_step"] = (
                parameter_state_summary(self.model)
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
