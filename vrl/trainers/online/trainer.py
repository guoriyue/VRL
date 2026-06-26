"""Online RL trainer — CEA pipeline (Collector + Evaluator + Algorithm).

collect -> evaluate -> advantage -> loss -> backward -> step.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import (
    move_training_batch_to_device,
    nonzero_advantage_mask,
    select_batch,
)
from vrl.rollouts.orchestration import build_rollout_schedule
from vrl.trainers.core.base import Trainer
from vrl.trainers.core.types import TrainerConfig, TrainState
from vrl.trainers.online.ema import EMAModuleWrapper
from vrl.trainers.online.precision_guard import run_precision_drift_guard
from vrl.trainers.precision import normalize_mixed_precision
from vrl.trainers.strategy import SingleProcessStrategy, Strategy
from vrl.trainers.weight_sync import TrainableStateGetter, WeightSyncer
from vrl.utils.model_diagnostics import (
    parameter_state_summary,
    tensor_stats,
    trainable_state_digest,
    write_jsonl,
)
from vrl.utils.stats import LoggingStatsSink, RolloutStats, StatsSink

logger = logging.getLogger(__name__)


def _global_reward_stats(rewards: Any) -> tuple[float, float]:
    """Population (mean, std) of ``rewards`` over **all DDP ranks**.

    The logged reward curve is written by rank0 only (online.py is_primary), and
    each rank holds just its own prompt slice (16 of the 32 global prompts for
    rbs=16), so a plain ``rewards.mean()`` reports a per-rank metric — half the
    true optimization objective. All-reduce the sufficient statistics (sum,
    sum-of-squares, count) so the logged mean/std reflect the full cross-rank
    batch. Falls back to local stats with no process group or world_size==1
    (single-GPU), where local already equals global. Unconditional on every rank
    (the writer-gating happens upstream), so the collective stays balanced.
    """

    n = rewards.numel()
    dist = torch.distributed
    distributed = (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    if not distributed:
        mean = rewards.mean().item() if n else 0.0
        std = rewards.std().item() if n > 1 else 0.0
        return mean, std
    stats = torch.stack(
        [rewards.sum(), rewards.mul(rewards).sum(), rewards.new_tensor(float(n))],
    )
    # NCCL collectives require GPU tensors, but rewards live on CPU; move stats to
    # the rank's GPU for nccl (gloo handles CPU directly, e.g. in tests).
    if dist.get_backend() == "nccl":
        stats = stats.cuda()
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    g_sum, g_sumsq, g_count = stats[0], stats[1], stats[2]
    g_mean = g_sum / g_count
    mean = float(g_mean.item())
    if g_count <= 1:
        return mean, 0.0
    g_var = (g_sumsq / g_count) - g_mean * g_mean
    std = float(torch.sqrt(torch.clamp(g_var, min=0.0)).item())
    return mean, std


def _all_ranks_have_work(has_work: bool, device: torch.device) -> bool:
    """True iff EVERY training rank has a non-empty (post-filter) microbatch.

    A backward pass fires cross-rank collectives — FSDP2 per-layer all-gather +
    reduce-scatter, or DDP's gradient all-reduce. If one rank skips backward on an
    all-filtered (zero-advantage) microbatch while another rank runs it, those
    collectives mismatch and the job DEADLOCKS: an unrecoverable NCCL hang, not an
    exception. So the skip decision must be unanimous. All-reduce the local
    ``has_work`` flag with MIN, so every rank takes the SAME branch — the
    microbatch runs only when all ranks have work, otherwise all ranks skip it
    together (matched: no rank issues backward collectives). Dropping a microbatch
    because one rank's slice came back empty wastes the other ranks' work for that
    slice, but empty slices are rare (reward spread) and a dropped slice beats a
    hung run.

    No process group / world_size==1 (single-GPU) returns the local value
    unchanged. Must be called UNCONDITIONALLY on every rank, the same number of
    times, so this collective itself stays balanced (the recipe runs a fixed
    microbatch/step count per rank regardless of filtering).
    """

    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return has_work
    flag = torch.tensor([1 if has_work else 0], dtype=torch.int64)
    # NCCL collectives require GPU tensors; gloo (tests) handles CPU directly.
    if dist.get_backend() == "nccl":
        flag = flag.to(device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() > 0)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def _create_optimizer(
    parameters: Any,
    config: TrainerConfig,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer."""
    optim = config.optim
    parameters = list(parameters)
    # fused=True collapses the per-parameter optimizer step into a handful of
    # kernels (the loop variant launched ~1.7k/step on LoRA models). It is
    # only valid for CUDA float params; anything else falls back to default.
    use_fused = bool(parameters) and all(
        isinstance(p, torch.Tensor) and p.is_cuda and p.is_floating_point()
        for p in parameters
    )
    return torch.optim.AdamW(
        parameters,
        lr=optim.lr,
        betas=(optim.adam_beta1, optim.adam_beta2),
        weight_decay=optim.weight_decay,
        eps=optim.eps,
        fused=use_fused or None,
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
    return normalize_mixed_precision(getattr(config, "train_precision", ""))


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


def _needs_grad_scaler(
    config: TrainerConfig,
    device: torch.device,
    model: Any,
    accelerator: Any | None,
) -> bool:
    """Use dynamic loss scaling for native CUDA fp16 training."""

    if accelerator is not None:
        return False
    return _trainer_autocast_dtype(config, device, model=model) == torch.float16


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
    train_precision = _precision_label(_resolve_mixed_precision(config))
    rollout_precision = _precision_label(config.rollout_precision or train_precision)
    autocast_dtype = _trainer_autocast_dtype(config, device, model=model)
    return {
        "train_precision": train_precision,
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


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """The data half of one step: filtered rollouts + advantages + diagnostics.

    ``collect_training_batch`` produces it and ``train_on_rollout_batch`` consumes
    it; everything the train half reads — the filtered batches/advantages, the
    pre-filter reward/advantage diagnostics, and the shared timer/iteration whose
    timings both halves accumulate into — travels in here. The split is the seam
    a future rank split would collect/train across.
    """

    iteration: Any
    timer: PhaseTimer
    batches: list[RolloutBatch]
    advantages: list[torch.Tensor]
    group_size: float
    trained_prompt_num: int
    adv_zero_rate: float
    adv_saturation: float
    pre_filter_reward_mean: float
    pre_filter_reward_std: float
    pre_filter_adv_mean: float


@dataclass(frozen=True, slots=True)
class _TrainingSampleChunk:
    batch: RolloutBatch
    advantages: torch.Tensor
    loss_weight: float
    is_dummy: bool = False


def _training_sample_chunks(
    batch: RolloutBatch,
    advantages: torch.Tensor,
    sample_batch_size: int,
) -> list[_TrainingSampleChunk]:
    """Split one prompt group for replay without changing full-group loss math."""

    batch_size = int(batch.rewards.shape[0])
    if batch_size != int(advantages.shape[0]):
        raise ValueError(
            "rollout batch and advantages must have the same sample count "
            f"({batch_size} != {int(advantages.shape[0])})",
        )
    if batch_size <= 0:
        return []
    chunk_size = int(sample_batch_size)
    if chunk_size <= 0 or chunk_size >= batch_size:
        return [_TrainingSampleChunk(batch=batch, advantages=advantages, loss_weight=1.0)]

    chunks: list[_TrainingSampleChunk] = []
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        selector = torch.arange(start, stop, device=batch.rewards.device)
        chunks.append(
            _TrainingSampleChunk(
                batch=select_batch(batch, selector),
                advantages=advantages[selector.to(advantages.device)],
                loss_weight=float(stop - start) / float(batch_size),
            ),
        )
    return chunks


def _distributed_max_int(value: int, device: torch.device) -> int:
    """Return the maximum integer value across training ranks."""

    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.int64)
    if dist.get_backend() == "nccl":
        tensor = tensor.to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _balanced_training_sample_chunks(
    batches: list[RolloutBatch],
    advantages: list[torch.Tensor],
    sample_batch_size: int,
    device: torch.device,
) -> list[_TrainingSampleChunk]:
    """Plan replay execution slots with equal slot counts across ranks.

    Local zero-advantage filtering can leave different ranks with different
    numbers of prompt groups or sample chunks. DDP/FSDP forward/backward issue
    collectives, so the number of replay slots must be globally balanced even
    when only some slots carry training signal.
    """

    chunks: list[_TrainingSampleChunk] = []
    for batch, adv in zip(batches, advantages, strict=True):
        chunks.extend(_training_sample_chunks(batch, adv, sample_batch_size))

    target_count = _distributed_max_int(len(chunks), device)
    if target_count == len(chunks):
        return chunks
    if target_count <= 0:
        return []
    if not chunks:
        raise RuntimeError(
            "distributed replay planner cannot synthesize dummy slots without a "
            "local real chunk; call _all_ranks_have_work before planning replay chunks",
        )
    # Use the smallest available local chunk as the dummy template to minimize
    # the extra zero-loss forward/backward work needed for collective balance.
    template = min(chunks, key=lambda chunk: int(chunk.batch.rewards.shape[0]))
    chunks.extend(
        _TrainingSampleChunk(
            batch=template.batch,
            advantages=torch.zeros_like(template.advantages),
            loss_weight=0.0,
            is_dummy=True,
        )
        for _ in range(target_count - len(chunks))
    )
    return chunks


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
        config: TrainerConfig,
        ref_model: nn.Module | None = None,
        weight_syncer: WeightSyncer | None = None,
        sync_state_getter: TrainableStateGetter | None = None,
        prompts: list[str] | None = None,
        device: torch.device | str = "cuda",
        accelerator: Any | None = None,
        strategy: Strategy | None = None,
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
        self.config = config
        # Precision correction (TIS) is a trainer-level precision-drift concern, not
        # an algorithm hyperparameter; inject it into algorithms that apply it
        # (importance-ratio algorithms hold a `precision_correction` slot).
        if hasattr(algorithm, "precision_correction"):
            algorithm.precision_correction = config.precision_correction
        self.prompts = prompts or []
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state = TrainState()
        self.accelerator = accelerator
        # How a step runs on the hardware (backward / clip / state export). The
        # default keeps current single-GPU behavior; FSDP2 swaps this in later
        # without the trainer loop changing. See vrl/trainers/strategy.py.
        self._strategy: Strategy = strategy or SingleProcessStrategy()
        # Route the model through the strategy once: identity for single process,
        # fully_shard wrapping for FSDP2. Done before optimizer / grad-scaler / EMA
        # so they bind to the (possibly sharded) parameters the strategy returns.
        self.model = self._strategy.prepare_model(self.model)
        # Sink for the per-step phase-timing line (recording decoupled from
        # emitting); swap for a jsonl/Prometheus sink later.
        self._stats_sink: StatsSink = LoggingStatsSink(logger)
        self._grad_scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda")
            if _needs_grad_scaler(self.config, self.device, self.model, self.accelerator)
            else None
        )

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
            # Default True: only likelihood-free objectives (DiffusionNFT) opt out,
            # which makes a continuous max_stale>0 config fail fast as unsound.
            algorithm_tolerates_off_policy_staleness=bool(
                getattr(self.algorithm, "tolerates_off_policy_staleness", True),
            ),
        )
        self._validate_trust_region_engages()

        if self.config.optim.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _validate_trust_region_engages(self) -> None:
        """Refuse configs where a trust-region algorithm's ratio term is inert.

        Flow-DPPO / GRPO-Guard are *defined* by a clipped/guarded importance ratio
        ``r = pi_new / pi_old``. With ``strict_on_policy`` + ``ppo_epochs == 1`` the
        behavior and target policy are identical on the single replay pass, so
        ``r == 1``, the trust-region term is identically zero, and the run is
        byte-equivalent to plain GRPO — the documented flat-curve root cause. Fail
        fast instead of silently training a no-op mechanism.

        Scoped to ``strict_on_policy``: ``continuous`` is the explicit off-policy
        path (stale ``behavior_policy_version`` makes ``r != 1`` even at one epoch),
        so the user has already opted into a moving ratio there.
        """
        if not bool(getattr(self.algorithm, "requires_active_trust_region", False)):
            return
        cfg = self.config
        schedule_mode = cfg.rollout_orchestration.schedule_mode
        if schedule_mode == "strict_on_policy" and int(cfg.ppo_epochs) <= 1:
            raise ValueError(
                f"{type(self.algorithm).__name__} is defined by its importance-ratio "
                "trust region, but rollout_orchestration.schedule_mode='strict_on_policy' "
                "with actor.ppo_epochs=1 makes the ratio identically 1 (behavior == "
                "target on the single replay pass), so the clip/guard term is a no-op "
                "and the run is equivalent to plain GRPO. Set actor.ppo_epochs>1 — which "
                "needs the legacy full-batch path (actor.gradient_accumulation_steps=0 "
                "and rollout.microbatch_size=0, since streaming releases each microbatch "
                "and cannot replay it across epochs) — or use schedule_mode='continuous' "
                "with staleness for an off-policy ratio."
            )

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
        self._strategy.backward(loss, grad_scaler=self._grad_scaler)

    def _clip_and_step(self, optimizer: Any) -> tuple[float, bool]:
        """Clip grads and step the optimizer.

        Returns ``(pre_clip_grad_norm, stepped)``. ``stepped`` is False only when
        the fp16 ``GradScaler`` skipped the step because it found inf/nan grads;
        callers must not run EMA / ``after_optimizer_step`` on a skipped step (it
        would fold an update that never happened into the averaged/adapter state).
        """
        cfg = self.config
        grad_norm: Any = 0.0
        if self._grad_scaler is not None:
            self._grad_scaler.unscale_(optimizer)
        if cfg.max_norm > 0:
            grad_norm = self._strategy.clip_grad_norm(self.model.parameters(), cfg.max_norm)
        else:
            # no clip — compute norm manually for diagnostic
            sq_sum = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    sq_sum += float(p.grad.detach().pow(2).sum().item())
            grad_norm = sq_sum**0.5
        stepped = True
        if self._grad_scaler is not None:
            scale_before = self._grad_scaler.get_scale()
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            # The scaler lowers its scale by the backoff factor only when it
            # skipped the step on inf/nan grads; an unchanged or grown scale
            # means the optimizer actually applied the update.
            stepped = self._grad_scaler.get_scale() >= scale_before
        else:
            optimizer.step()
        optimizer.zero_grad()
        return float(grad_norm), stepped

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
        batch = await self.collect_training_batch(prompts)
        return await self.train_on_rollout_batch(batch)

    async def collect_training_batch(
        self,
        prompts: list[str] | None = None,
    ) -> TrainingBatch:
        """Collect rollouts and compute + filter advantages — the data half.

        Returns everything ``train_on_rollout_batch`` needs; in single-process
        ``step()`` the two run back-to-back. The shared ``timer`` is created here
        and carried through the batch so both halves' phase timings land in one
        accumulator, exactly as the previous single method did.
        """
        if prompts is not None:
            self.prompts = prompts

        cfg = self.config

        timer = PhaseTimer(enabled=cfg.profile)
        runtime_debug_collect = bool(cfg.debug.first_step and self.state.step == 0)

        # 1. The rollout schedule owns collect/offload/release/sync timing.
        iteration = await self.rollout_schedule.next_iteration(
            list(self.prompts),
            group_size=cfg.n_samples_per_prompt,
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

        # Cross-rank so the logged curve is the full 32-prompt objective, not
        # rank0's local 16-prompt slice (see _global_reward_stats).
        pre_filter_reward_mean, pre_filter_reward_std = _global_reward_stats(all_rewards)
        pre_filter_adv_mean = advantages_all.mean().item()

        # Split advantages back per-batch for the gradient-accumulation loop.
        split_sizes = [b.rewards.shape[0] for b in all_batches]
        adv_split = list(torch.split(advantages_all, split_sizes))

        # Zero-advantage sample filter. Flow-GRPO applies this globally across
        # the outer rollout epoch, pads enough zero-advantage rows back in for
        # exact rebatching, then shuffles/rebatches into
        # num_batches_per_epoch training microbatches.
        # Per-group batches (one RolloutBatch per prompt group). Streaming
        # accumulation (gradient_accumulation_steps>0) splits the optimizer
        # target into prompt microbatches in the recipe loop, so the collector
        # no longer rebatches internally; advantage stays per-group either way.
        filtered_batches: list[RolloutBatch] = []
        filtered_advs: list[torch.Tensor] = []
        if cfg.drop_zero_advantage:
            for b, adv_b in zip(all_batches, adv_split, strict=True):
                mask = nonzero_advantage_mask(adv_b)
                if not bool(mask.any()):
                    continue
                if not bool(mask.all()):
                    b = select_batch(b, mask)
                    adv_b = adv_b[mask.to(adv_b.device)]
                if b.rewards.shape[0] > 0:
                    filtered_batches.append(b)
                    filtered_advs.append(adv_b)
        else:
            filtered_batches = list(all_batches)
            filtered_advs = adv_split

        return TrainingBatch(
            iteration=iteration,
            timer=timer,
            batches=filtered_batches,
            advantages=filtered_advs,
            group_size=group_size,
            trained_prompt_num=trained_prompt_num,
            adv_zero_rate=adv_zero_rate,
            adv_saturation=adv_saturation,
            pre_filter_reward_mean=pre_filter_reward_mean,
            pre_filter_reward_std=pre_filter_reward_std,
            pre_filter_adv_mean=pre_filter_adv_mean,
        )

    @staticmethod
    def _train_timestep_indices(
        num_timesteps: int,
        timestep_fraction: float,
        selection: str = "strided",
    ) -> list[int]:
        """Denoise timesteps that receive loss (single source of truth).

        ``selection`` controls which subset gets gradient when
        ``timestep_fraction < 1``:
        - ``"strided"`` — a fixed evenly-spaced subset, identical every update.
        - ``"random"`` (DanceGRPO) — a fresh random subset resampled each call,
          so gradient coverage of the denoise trajectory decorrelates across
          updates instead of always landing on the same strided steps. Returned
          sorted; the loss loop sums over the indices, so order is irrelevant.
        """
        train_timestep_count = max(1, int(num_timesteps * timestep_fraction))
        if train_timestep_count >= num_timesteps:
            return list(range(num_timesteps))
        if selection == "random":
            import torch

            picks = torch.randperm(num_timesteps)[:train_timestep_count].tolist()
            return sorted(int(i) for i in picks)
        step_size = num_timesteps / train_timestep_count
        return [int(i * step_size) for i in range(train_timestep_count)]

    # ------------------------------------------------------------------
    # Streaming accumulation boundary (gradient_accumulation_steps>0):
    # begin -> backward(microbatch)* -> finish. One optimizer update spans
    # several separately-collected prompt microbatches so the whole target
    # batch never has to be materialized at once (SPRINT_streaming_rollout_
    # accumulation). The legacy full-batch path keeps train_on_rollout_batch.
    # ------------------------------------------------------------------

    def begin_optimizer_update(self) -> None:
        """Start one streaming optimizer update: reset grads + metric accumulator.

        The recipe pairs this with ``reward_fn.reset_components()`` once per
        update (the trainer holds no reward_fn reference).
        """
        self._update_optimizer = self._ensure_optimizer()
        self._update_ema = self._ensure_ema()
        self.model.train()
        self._update_agg_metrics: dict[str, list[float]] = defaultdict(list)
        self._update_optimizer.zero_grad(set_to_none=True)

    async def backward_on_training_batch(
        self,
        batch: TrainingBatch,
        *,
        total_groups: int,
    ) -> None:
        """Evaluate + loss + backward for ONE collected microbatch; no optimizer step.

        ``total_groups`` is the global prompt-group count across the whole
        optimizer update (= rollout_batch_size); loss divides by
        ``total_groups * num_train_timesteps`` so the accumulated gradient over
        all microbatches equals the legacy full-batch path.
        """
        from vrl.algorithms.trajectory import AlgorithmAdapter, AlgorithmInput
        from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
        from vrl.utils.profiling import record_function

        cfg = self.config
        # Unanimous skip across ranks: a backward fires cross-rank collectives, so
        # one rank skipping an empty microbatch while another runs it deadlocks
        # (see _all_ranks_have_work). Called once per microbatch on every rank, in
        # lockstep with the fixed gradient-accumulation count, so this collective
        # is balanced.
        if not _all_ranks_have_work(bool(batch.batches), self.device):
            return
        autocast_ctx = _get_autocast(cfg, self.device, model=self.model)
        uses_evaluator = bool(getattr(self.algorithm, "uses_evaluator", True))
        algorithm_adapter = AlgorithmAdapter()
        defer = uses_evaluator and bool(
            getattr(self.evaluator, "supports_deferred_replay_tensor_move", False),
        )
        num_timesteps = batch.batches[0].observations.shape[1]
        train_indices = self._train_timestep_indices(
            num_timesteps, cfg.timestep_fraction, cfg.timestep_selection,
        )
        loss_scale = int(total_groups) * len(train_indices)
        sample_batch_size = int(getattr(cfg, "sample_batch_size", 0))
        agg = self._update_agg_metrics
        for sample_chunk in _balanced_training_sample_chunks(
            batch.batches,
            batch.advantages,
            sample_batch_size,
            self.device,
        ):
            chunk_batch = move_training_batch_to_device(
                sample_chunk.batch,
                self.device,
                defer_replay_tensors=defer,
            )
            chunk_adv = sample_chunk.advantages.to(self.device)
            for j in train_indices:
                with autocast_ctx:
                    if not uses_evaluator:
                        with record_function("trainer.loss"):
                            loss, metrics = algorithm_adapter.compute_loss(
                                self.algorithm,
                                AlgorithmInput(
                                    rewards=chunk_batch.rewards,
                                    group_ids=chunk_batch.group_ids,
                                    advantages=chunk_adv,
                                    metadata={
                                        "model": self.model,
                                        "rollout_batch": chunk_batch,
                                        "timestep_index": j,
                                    },
                                ),
                            )
                    else:
                        if self.evaluator is None:
                            raise RuntimeError(
                                f"{type(self.algorithm).__name__} requires an evaluator",
                            )
                        kl_coef = float(
                            getattr(self.algorithm.config, "kl_coef", 0.0),
                        )
                        # Trust-region SDE algorithms (Flow-DPPO / GRPO-Guard)
                        # read latent KL intermediates (dt) even at kl_coef=0.
                        need_kl_intermediates = kl_coef > 0 or bool(
                            getattr(self.algorithm, "needs_kl_intermediates", False),
                        )
                        with record_function("trainer.replay"):
                            signals = self.evaluator.evaluate(
                                self.model,
                                chunk_batch,
                                j,
                                ref_model=self.ref_model,
                                signal_request=SignalRequest(
                                    need_ref=kl_coef > 0,
                                    need_kl_intermediates=need_kl_intermediates,
                                ),
                            )
                        with record_function("trainer.loss"):
                            if not isinstance(signals, TrajectorySignalBatch):
                                raise TypeError(
                                    "evaluator output must be TrajectorySignalBatch; "
                                    f"got {type(signals).__name__}",
                                )
                            loss, metrics = algorithm_adapter.compute_loss(
                                self.algorithm,
                                AlgorithmInput(
                                    signals=signals,
                                    advantages=chunk_adv,
                                    group_ids=chunk_batch.group_ids,
                                ),
                            )
                    loss = loss * sample_chunk.loss_weight / loss_scale
                self._backward(loss)
                self._clear_algorithm_diagnostics()
                if not sample_chunk.is_dummy:
                    agg["loss"].append(metrics.loss)
                    agg["policy_loss"].append(metrics.policy_loss)
                    agg["kl_penalty"].append(metrics.kl_penalty)
                    agg["clip_fraction"].append(metrics.clip_fraction)
                    agg["approx_kl"].append(metrics.approx_kl)
                    agg["logprob_abs_diff_mean"].append(metrics.logprob_abs_diff_mean)
                    agg["logprob_abs_diff_max"].append(metrics.logprob_abs_diff_max)
                    agg["ratio_abs_dev_mean"].append(metrics.ratio_abs_dev_mean)
                    agg["ratio_abs_dev_max"].append(metrics.ratio_abs_dev_max)
                    agg["mismatch_kl"].append(metrics.mismatch_kl)
                    agg["mismatch_k3_kl"].append(metrics.mismatch_k3_kl)
                await asyncio.sleep(0)

    async def finish_optimizer_update(
        self,
        *,
        phase_times: dict[str, float],
        reward_mean: float,
        reward_std: float,
        adv_mean: float,
        adv_zero_rate: float,
        adv_saturation: float,
        group_size: float,
        trained_prompt_num: int,
    ) -> TrainStepMetrics:
        """Clip + optimizer.step + EMA/NFT (once) and build the one update's metrics.

        ``phase_times`` is the caller-aggregated collect/timer phase dict summed
        across the update's microbatches — the streaming recipe owns aggregation
        so each microbatch's tensors are released before this runs.
        """
        optimizer = self._update_optimizer
        agg = self._update_agg_metrics
        grad_norm, stepped = self._clip_and_step(optimizer)
        agg["grad_norm"].append(grad_norm)
        if stepped:
            after_optimizer_step = getattr(self.algorithm, "after_optimizer_step", None)
            if callable(after_optimizer_step):
                after_optimizer_step(self.model, self.state.global_step)
            if self._update_ema is not None:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                self._update_ema.step(trainable, self.state.global_step)
        self.state.global_step += 1

        def avg(key: str) -> float:
            vals = agg.get(key, [])
            return sum(vals) / len(vals) if vals else 0.0

        def mx(key: str) -> float:
            vals = agg.get(key, [])
            return max(vals) if vals else 0.0

        phase_times = dict(phase_times)
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
        self.state.step += 1
        self.state.total_reward += metrics.reward_mean
        self.state.total_loss += metrics.loss
        sync_phase_times = await self.rollout_schedule.after_train_step()
        for key, value in sync_phase_times.items():
            phase_times[key] = phase_times.get(key, 0.0) + value
        return metrics

    async def train_on_rollout_batch(self, batch: TrainingBatch) -> TrainStepMetrics:
        """Train on a collected batch — the compute half of one step.

        Stays async on purpose: the per-timestep ``await asyncio.sleep(0)`` in the
        inner loop lets the continuous-rollout producer advance on the shared
        asyncio loop between CUDA-heavy iterations. Making this synchronous would
        change rollout interleaving, so the split keeps it awaitable. Behavior is
        identical to the previous single method.
        """
        from vrl.algorithms.trajectory import AlgorithmAdapter, AlgorithmInput
        from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
        from vrl.utils.profiling import record_function

        cfg = self.config
        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()
        iteration = batch.iteration
        timer = batch.timer
        filtered_batches = batch.batches
        filtered_advs = batch.advantages
        group_size = batch.group_size
        trained_prompt_num = batch.trained_prompt_num
        adv_zero_rate = batch.adv_zero_rate
        adv_saturation = batch.adv_saturation
        pre_filter_reward_mean = batch.pre_filter_reward_mean
        pre_filter_reward_std = batch.pre_filter_reward_std
        pre_filter_adv_mean = batch.pre_filter_adv_mean

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
        # Unanimous across ranks (see _all_ranks_have_work): a backward fires
        # cross-rank collectives, so the skip must be agreed or the ranks that did
        # vs. did not run backward deadlock. Called once per step on every rank.
        if not _all_ranks_have_work(bool(filtered_batches), self.device):
            logger.info(
                "step %d: all batches filtered (zero advantages) on this or a peer "
                "rank; skipping backward",
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
                phase_times=self._step_stats(iteration, timer).as_phase_dict(),
            )

        defer_replay_tensor_move = uses_evaluator and bool(
            getattr(self.evaluator, "supports_deferred_replay_tensor_move", False),
        )

        # Timestep schedule — same num_timesteps across all batches (collector
        # uses the same scheduler), so pick from first filtered batch. Shares the
        # single-source selection helper with the streaming path.
        num_timesteps = filtered_batches[0].observations.shape[1]
        train_indices = self._train_timestep_indices(
            num_timesteps, cfg.timestep_fraction, cfg.timestep_selection,
        )

        # Number of rollout micro-batches per optimizer update. ``0`` preserves
        # legacy VRL behavior: one optimizer update after all collected batches.
        grad_accum_batches = int(cfg.gradient_accumulation_steps)
        if grad_accum_batches <= 0 or grad_accum_batches > len(filtered_batches):
            grad_accum_batches = len(filtered_batches)
        sample_batch_size = int(getattr(cfg, "sample_batch_size", 0))

        # Debug first step: compare old vs fresh log-probs on first timestep
        # (using first filtered batch so memory footprint is bounded).
        first_step_debug_record: dict[str, Any] | None = None
        precision_metadata = _merge_rollout_precision_context(
            _trainer_precision_metadata(cfg, self.device, self.model),
            filtered_batches[0].context,
        )
        first_debug_chunk = _training_sample_chunks(
            filtered_batches[0],
            filtered_advs[0],
            sample_batch_size,
        )[0]
        if cfg.debug.first_step and self.state.step == 0 and uses_evaluator:
            _dbg_batch = move_training_batch_to_device(
                first_debug_chunk.batch,
                self.device,
                defer_replay_tensors=defer_replay_tensor_move,
            )
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
            # Replay parity is the ratio==1 invariant: with unchanged weights
            # the fresh log-prob must reproduce the collection-time one. A
            # large gap means the training signal is garbage (e.g. the
            # Predict2 EDM-sigma-domain bug sat at mean diff ~115 in metrics
            # nobody alerted on) — shout, do not just persist a jsonl row.
            if _diff.mean().item() > 0.01:
                logger.warning(
                    "first-step log-prob parity violated: mean abs diff "
                    "%.4f > 0.01. Replay does not reproduce rollout "
                    "log-probs; GRPO ratios are untrustworthy. Suspect "
                    "replay-side conditioning/scheduler-domain drift.",
                    _diff.mean().item(),
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
        elif cfg.debug.first_step and self.state.step == 0:
            # Non-evaluator algorithms (NFT) compute no log-prob ratio, so the
            # parity probe above is blind to them. Ask the algorithm for its
            # own lr=0 invariant through the optional protocol method instead
            # of hardcoding algorithm checks here.
            _invariant_check = getattr(self.algorithm, "first_step_invariant_check", None)
            if callable(_invariant_check):
                _dbg_batch = move_training_batch_to_device(
                    first_debug_chunk.batch,
                    self.device,
                    defer_replay_tensors=defer_replay_tensor_move,
                )
                _dbg_adv = first_debug_chunk.advantages.to(self.device)
                with torch.no_grad(), autocast_ctx, record_function("trainer.replay"):
                    _invariant = _invariant_check(
                        model=self.model,
                        batch=_dbg_batch,
                        advantages=_dbg_adv,
                        timestep_index=0,
                    )
                logger.info(
                    "DEBUG first-step %s: abs_diff=%.3e (threshold %.1e)",
                    _invariant["event"],
                    _invariant["abs_diff"],
                    _invariant["threshold"],
                )
                if not _invariant.get("passed", True):
                    logger.warning(
                        "first-step %s invariant violated: abs_diff %.3e > %.1e. "
                        "The collection-time training signal is untrustworthy; "
                        "suspect replay-side conditioning/scheduler-domain drift.",
                        _invariant["invariant"],
                        _invariant["abs_diff"],
                        _invariant["threshold"],
                    )
                first_step_debug_record = {
                    **_invariant,
                    "trainer_step": int(self.state.step),
                    "global_step": int(self.state.global_step),
                    "device": str(self.device),
                    "mixed_precision": _resolve_mixed_precision(cfg),
                    "precision_policy": precision_metadata,
                    "rollout_context": {
                        key: value
                        for key, value in _dbg_batch.context.items()
                        if key != "runtime_debug"
                    },
                }

        # Precision drift guard: on the first step (before any optimizer update),
        # check rollout-vs-replay logprob parity. `auto` protects unsafe precision
        # splits; explicit warn/fail is used for same-precision acceptance runs.
        if self.state.step == 0 and uses_evaluator and self.evaluator is not None:
            _guard_batch = move_training_batch_to_device(
                first_debug_chunk.batch,
                self.device,
                defer_replay_tensors=defer_replay_tensor_move,
            )

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
                train_precision=_resolve_mixed_precision(cfg),
                rollout_precision=cfg.rollout_precision or _resolve_mixed_precision(cfg),
                math_precision=cfg.math_precision,
                timestep_indices=train_indices,
                evaluate_fn=_guard_evaluate,
                metadata=precision_metadata,
                logger=logger,
            )
            if _guard_record is not None and first_step_debug_record is not None:
                first_step_debug_record["precision_drift_guard"] = _guard_record

        for _ppo_epoch in range(cfg.ppo_epochs):
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

                for sample_chunk in _balanced_training_sample_chunks(
                    chunk_batches,
                    chunk_advs,
                    sample_batch_size,
                    self.device,
                ):
                    chunk_batch = move_training_batch_to_device(
                        sample_chunk.batch,
                        self.device,
                        defer_replay_tensors=defer_replay_tensor_move,
                    )
                    chunk_adv = sample_chunk.advantages.to(self.device)
                    for j in train_indices:
                        with timer.time("evaluate"), autocast_ctx:
                            if not uses_evaluator:
                                with record_function("trainer.loss"):
                                    loss, metrics = algorithm_adapter.compute_loss(
                                        self.algorithm,
                                        AlgorithmInput(
                                            rewards=chunk_batch.rewards,
                                            group_ids=chunk_batch.group_ids,
                                            advantages=chunk_adv,
                                            metadata={
                                                "model": self.model,
                                                "rollout_batch": chunk_batch,
                                                "timestep_index": j,
                                            },
                                        ),
                                    )
                            else:
                                if self.evaluator is None:
                                    raise RuntimeError(
                                        f"{type(self.algorithm).__name__} requires an evaluator",
                                    )
                                kl_coef = float(
                                    getattr(self.algorithm.config, "kl_coef", 0.0),
                                )
                                need_kl_intermediates = kl_coef > 0 or bool(
                                    getattr(
                                        self.algorithm,
                                        "needs_kl_intermediates",
                                        False,
                                    ),
                                )
                                with record_function("trainer.replay"):
                                    signals = self.evaluator.evaluate(
                                        self.model,
                                        chunk_batch,
                                        j,
                                        ref_model=self.ref_model,
                                        signal_request=SignalRequest(
                                            need_ref=kl_coef > 0,
                                            need_kl_intermediates=need_kl_intermediates,
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
                                            advantages=chunk_adv,
                                            group_ids=chunk_batch.group_ids,
                                        ),
                                    )
                            # Average across rollout micro-batches inside this
                            # optimizer update; timestep accumulation follows
                            # Flow-GRPO's per-denoise-step surrogate structure.
                            loss = loss * sample_chunk.loss_weight / loss_scale

                        with timer.time("backward"):
                            self._backward(loss)

                        self._clear_algorithm_diagnostics()

                        if not sample_chunk.is_dummy:
                            agg_metrics["loss"].append(metrics.loss)
                            agg_metrics["policy_loss"].append(metrics.policy_loss)
                            agg_metrics["kl_penalty"].append(metrics.kl_penalty)
                            agg_metrics["clip_fraction"].append(metrics.clip_fraction)
                            agg_metrics["approx_kl"].append(metrics.approx_kl)
                            agg_metrics["logprob_abs_diff_mean"].append(
                                metrics.logprob_abs_diff_mean,
                            )
                            agg_metrics["logprob_abs_diff_max"].append(
                                metrics.logprob_abs_diff_max,
                            )
                            agg_metrics["ratio_abs_dev_mean"].append(
                                metrics.ratio_abs_dev_mean,
                            )
                            agg_metrics["ratio_abs_dev_max"].append(
                                metrics.ratio_abs_dev_max,
                            )
                            agg_metrics["mismatch_kl"].append(metrics.mismatch_kl)
                            agg_metrics["mismatch_k3_kl"].append(metrics.mismatch_k3_kl)

                        # Continuous rollout production runs on the same asyncio
                        # loop as training orchestration. Yield once per
                        # timestep so producer admit/harvest can progress while
                        # this synchronous CUDA-heavy loop is still computing.
                        await asyncio.sleep(0)

                with timer.time("optim_step"):
                    _gn, _stepped = self._clip_and_step(optimizer)
                    agg_metrics["grad_norm"].append(_gn)

                # A scaler-skipped step (inf/nan grads) left the weights
                # unchanged — do not fold a non-update into EMA or the
                # algorithm's post-step adapter sync.
                if _stepped:
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

        # collect.* phase timings arrive inside iteration.stats: each collect
        # call owns its timings (no shared collector state), and the
        # schedule/consumer aggregates them per iteration. The trainer phases
        # (timer) merge on top into one typed accumulator, emitted via the sink.
        step_stats = self._step_stats(iteration, timer)
        phase_times = step_stats.as_phase_dict()
        if cfg.profile and phase_times:
            self._stats_sink.record(self.state.step, step_stats)
            self._write_phase_events(timer)

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

    def _step_stats(self, iteration: Any, timer: PhaseTimer) -> RolloutStats:
        """Typed per-step stats: the iteration's accumulator + trainer phases."""

        stats = RolloutStats()
        stats.merge(iteration.stats)
        stats.add_phases(timer.times)
        return stats

    def _write_phase_events(self, timer: PhaseTimer) -> None:
        """Append the per-phase start/end events to phase_events.jsonl."""

        try:
            import json
            import os

            evt_path = os.path.join(self.config.output_dir, "phase_events.jsonl")
            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(evt_path, "a") as handle:
                for name, start, end in timer.events:
                    handle.write(
                        json.dumps(
                            {
                                "step": self.state.step,
                                "phase": name,
                                "start": start,
                                "end": end,
                            },
                        )
                        + "\n",
                    )
            timer.events.clear()
        except Exception:
            pass

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
        if self._grad_scaler is not None:
            d["grad_scaler"] = self._grad_scaler.state_dict()
        ema = self._ensure_ema()
        if ema is not None:
            d["ema"] = ema.state_dict()
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

        if "grad_scaler" in state:
            if self._grad_scaler is None:
                if strict:
                    raise ValueError(
                        "checkpoint contains GradScaler state but trainer fp16 scaling is disabled",
                    )
                logger.warning("Skipping GradScaler state because fp16 scaling is disabled")
            else:
                try:
                    self._grad_scaler.load_state_dict(state["grad_scaler"])
                except Exception:
                    if strict:
                        raise
                    logger.warning("Skipping incompatible GradScaler state during non-strict load")

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
