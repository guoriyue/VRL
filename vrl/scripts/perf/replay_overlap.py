"""Experimental replay scheduling on an existing single-device OnlineTrainer.

No recipe/default changes: callers scope this adapter around a measured update.
Autograd runs backward on each op's forward stream, so the streams arm places
whole replay forwards on alternating streams and serializes gradient writers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import torch
import torch.nn.functional as F


def cuda_overlap_summary(trace_path) -> dict[str, Any]:
    """Attribute device kernel intervals via profiler External ids, not host ranges."""
    import bisect
    import json
    from collections import defaultdict

    from vrl.utils.profiling import TimeIntervals

    events = json.loads(trace_path.read_text())["traceEvents"]
    scopes = sorted(
        (event["ts"], event["ts"] + event["dur"], event["name"].removeprefix("probe.replay_"))
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") in {"probe.replay_forward", "probe.replay_backward"}
    )
    starts = [start for start, _, _ in scopes]
    external_stages = {}
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        index = bisect.bisect_right(starts, event["ts"]) - 1
        external = event.get("args", {}).get("External id")
        if external is not None and index >= 0 and event["ts"] <= scopes[index][1]:
            external_stages[external] = scopes[index][2]
    intervals = defaultdict(list)
    for event in events:
        if event.get("cat") == "kernel":
            stage = external_stages.get(event.get("args", {}).get("External id"), "other")
            intervals[stage].append((event["ts"], event["ts"] + event["dur"]))
    forward = TimeIntervals(intervals["forward"])
    backward = TimeIntervals(intervals["backward"])
    return {
        "counts": {stage: len(values) for stage, values in intervals.items()},
        "forward_active_ms": forward.duration / 1000,
        "backward_active_ms": backward.duration / 1000,
        "forward_backward_overlap_ms": forward.overlap(backward) / 1000,
        "has_forward_and_backward_kernels": bool(intervals["forward"] and intervals["backward"]),
        "scope": "GPU kernel interval unions; instrumented trace, not update latency",
    }


class _StreamLoss:
    """Keep the trainer's scalar normalization on the replay's CUDA stream."""

    def __init__(self, tensor, stream):
        self.tensor = tensor
        self.stream = stream

    def __mul__(self, value):
        with torch.cuda.stream(self.stream):
            return _StreamLoss(self.tensor * value, self.stream)

    def __truediv__(self, value):
        with torch.cuda.stream(self.stream):
            return _StreamLoss(self.tensor / value, self.stream)


class _DeferredLinear(torch.autograd.Function):
    """Split a plain Linear's dX from dW; no distributed hooks or loss scaling."""

    @staticmethod
    def forward(ctx, inputs, weight, bias, pending):
        ctx.input_dtype = inputs.dtype
        ctx.weight_parameter = weight
        ctx.bias = bias
        ctx.pending = pending
        # Match Linear autocast at the forward boundary. Backward executes
        # outside autocast and must use those same effective input/weight
        # dtypes, while accumulating into the original master Parameters.
        if torch.is_autocast_enabled(inputs.device.type):
            dtype = torch.get_autocast_dtype(inputs.device.type)
            inputs = inputs.to(dtype) if inputs.dtype != torch.float64 else inputs
            weight = weight.to(dtype) if weight.dtype != torch.float64 else weight
            if bias is not None and bias.dtype != torch.float64:
                bias = bias.to(dtype)
        ctx.save_for_backward(inputs, weight)
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        grad_input = (
            (grad_output @ weight).to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
        )
        ctx.pending.append((ctx.weight_parameter, ctx.bias, inputs.detach(), grad_output.detach()))
        return grad_input, None, None, None


@contextmanager
def replay_schedule(trainer: Any, *, mode: str) -> Iterator[dict[str, Any]]:
    """Temporarily schedule replay; always join work and restore patched methods.

    ``serial`` is the unmodified baseline. ``streams`` schedules private-leaf
    replicas, while backwards accumulate into the original parameters in order.
    ``deferred_wgrad`` postpones trainable plain Linear weight gradients until
    the real trainer's clipping/optimizer boundary. Neither arm is distributed
    support or a claim that the GPU actually executes kernels concurrently.
    """

    if mode not in {"serial", "streams", "deferred_wgrad"}:
        raise ValueError(f"unknown replay schedule: {mode}")
    stats: dict[str, Any] = {"forward_calls": 0, "deferred_linears": [], "queued_bytes_peak": 0}
    if mode == "serial":
        yield stats
        return
    if trainer._strategy.context.distributed:
        raise ValueError(
            "replay overlap experiment requires single_process; no collective scheduling"
        )
    if trainer._grad_scaler is not None:
        raise ValueError("replay overlap experiment does not support GradScaler")
    if trainer.config.profile:
        raise ValueError("trainer.profile synchronizes every phase; use the probe's CUDA trace")
    if trainer.config.ema.enable or trainer.algorithm.sft_weight:
        raise ValueError("replay overlap experiment requires EMA and SFT regularization disabled")
    if trainer._strategy.offload is not None:
        raise ValueError("replay overlap experiment cannot share a block-offload prefetcher")
    from vrl.nn.offload import HostStreaming

    if HostStreaming.of(trainer.model.transformer) is not None:
        raise ValueError("replay overlap experiment cannot share streamed model state")
    if any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in trainer.model.modules()):
        raise ValueError("replay overlap experiment requires deterministic forwards (no dropout)")
    if mode == "streams":
        from vrl.trainers.strategy import SingleProcessStrategy

        if type(trainer._strategy) is not SingleProcessStrategy:
            raise ValueError(
                "explicit-gradient stream experiment requires the native SingleProcessStrategy"
            )
        if any(
            p._backward_hooks or p._post_accumulate_grad_hooks
            for p in trainer.model.parameters()
            if p.requires_grad
        ):
            raise ValueError("explicit-gradient stream experiment cannot bypass parameter hooks")
    if mode == "streams" and trainer.device.type != "cuda":
        raise ValueError("streams schedule requires CUDA")

    pending: list[tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]] = []
    streams = (
        [torch.cuda.Stream(device=trainer.device) for _ in range(2)] if mode == "streams" else []
    )
    previous_backward: torch.cuda.Event | None = None
    active_stream: torch.cuda.Stream | None = None
    compute_loss = trainer._compute_replay_loss
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    clip_and_step = trainer._clip_and_step
    original_model = trainer.model
    replicas = []
    replica_parameters = []
    stepped = False
    if streams:
        # Each stream needs its own trainable leaves and checkpoint module
        # state. Share frozen weights, but never shared AccumulateGrad nodes.
        frozen = {id(p): p for p in original_model.parameters() if not p.requires_grad}
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream(trainer.device))
            with torch.cuda.stream(stream):
                replica = deepcopy(original_model, memo=frozen.copy())
                replica.train()
            original_named = dict(original_model.named_parameters())
            replica_named = dict(replica.named_parameters())
            if original_named.keys() != replica_named.keys() or any(
                (replica_named[name] is parameter) == parameter.requires_grad
                for name, parameter in original_named.items()
            ):
                raise ValueError("replica must share frozen weights and isolate trainable leaves")
            replicas.append(replica)
            replica_parameters.append([p for p in replica.parameters() if p.requires_grad])
        stats["private_trainable_leaves"] = sum(p.numel() for p in parameters) * len(streams)

    def compute(*args, **kwargs):
        nonlocal active_stream
        if stepped:
            raise ValueError("stream replicas support one optimizer update per scope")
        index = stats["forward_calls"] % 2
        active_stream = streams[index]
        stats["forward_calls"] += 1
        caller = torch.cuda.current_stream(trainer.device)
        active_stream.wait_stream(caller)
        from vrl.trajectory.device import map_tensor_tree

        def record_input(tensor):
            if tensor.is_cuda:
                tensor.record_stream(active_stream)
            return tensor

        # Microbatch tensors may have been allocated on the caller stream.
        # Their last Python reference can die while backward still reads them;
        # an event dependency alone does not prevent allocator reuse.
        map_tensor_tree(
            (args, kwargs), record_input, is_leaf=lambda x: isinstance(x, torch.Tensor)
        )
        with (
            torch.cuda.stream(active_stream),
            torch.profiler.record_function("probe.replay_forward"),
            patch.object(trainer, "model", replicas[index]),
        ):
            loss, metrics = compute_loss(*args, **kwargs)
        # A root loss op on the caller stream causes autograd to join backward
        # there before returning. The next forward would then wait for all of
        # the preceding backward, even with independent gradient collection.
        return _StreamLoss(loss, active_stream), metrics

    def run_backward(loss):
        nonlocal previous_backward
        if not isinstance(loss, _StreamLoss):
            raise TypeError("stream backward requires the stream-owned replay loss")
        loss = loss.tensor
        if active_stream is None:
            raise RuntimeError("backward without a scheduled replay forward")
        active_stream.wait_stream(torch.cuda.current_stream(trainer.device))
        # Forward may run alongside the previous backward, but grad writers
        # must not race across microbatches (including AccumulateGrad kernels).
        if previous_backward is not None:
            active_stream.wait_event(previous_backward)
        with (
            torch.cuda.stream(active_stream),
            torch.profiler.record_function("probe.replay_backward"),
        ):
            loss.record_stream(active_stream)
            # Each replica owns its AccumulateGrad nodes. Collect its gradients,
            # then accumulate into optimizer-owned Parameters in original order.
            leaves = replica_parameters[(stats["forward_calls"] - 1) % 2]
            gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients, strict=True):
                    if gradient is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = gradient.detach()
                    else:
                        parameter.grad.record_stream(active_stream)
                        parameter.grad.add_(gradient)
            previous_backward = torch.cuda.Event()
            previous_backward.record(active_stream)

    def finish_update(optimizer):
        nonlocal stepped
        if streams:
            caller = torch.cuda.current_stream(trainer.device)
            for stream in streams:
                caller.wait_stream(stream)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.record_stream(caller)
        if pending:
            stats["queued_bytes_peak"] = max(
                stats["queued_bytes_peak"],
                sum(
                    (x.numel() * x.element_size() + dy.numel() * dy.element_size())
                    for _, _, x, dy in pending
                ),
            )
            with torch.no_grad(), torch.profiler.record_function("probe.deferred_wgrad"):
                for weight, bias, inputs, grad_output in pending:
                    x = inputs.reshape(-1, inputs.shape[-1])
                    dy = grad_output.reshape(-1, grad_output.shape[-1])
                    grad = (dy.T @ x).to(weight.dtype)
                    if weight.grad is None:
                        weight.grad = grad
                    else:
                        weight.grad.add_(grad)
                    if bias is not None and bias.requires_grad:
                        dtype = torch.float64 if dy.dtype == torch.float64 else torch.float32
                        grad_bias = dy.sum(0, dtype=dtype).to(dy.dtype).to(bias.dtype)
                        if bias.grad is None:
                            bias.grad = grad_bias
                        else:
                            bias.grad.add_(grad_bias)
            pending.clear()
        result = clip_and_step(optimizer)
        stepped = True
        return result

    with ExitStack() as stack:
        if mode == "streams":
            stack.enter_context(patch.object(trainer, "_compute_replay_loss", compute))
            stack.enter_context(patch.object(trainer, "_backward", run_backward))
        else:
            # Plain leaves only: PEFT wrappers, fused modules and custom
            # forwards keep their semantics. Parameter hooks would be bypassed
            # by delayed accumulation, so reject them rather than dropping them.
            for name, module in trainer.model.transformer.named_modules():
                if type(module) is not torch.nn.Linear or not module.weight.requires_grad:
                    continue
                if any(
                    p._backward_hooks or p._post_accumulate_grad_hooks
                    for p in module.parameters(recurse=False)
                ):
                    raise ValueError(f"cannot defer parameter hooks: {name}")
                if (
                    "forward" in module.__dict__
                    or module._forward_hooks
                    or module._forward_pre_hooks
                    or module._backward_hooks
                ):
                    raise ValueError(f"cannot replace hooked Linear: {name}")

                # apply takes positional arguments; capture this module, not
                # the loop variable. This callback owns the queue lifetime.
                def forward(input, module=module):
                    return _DeferredLinear.apply(input, module.weight, module.bias, pending)

                stack.enter_context(patch.object(module, "forward", forward))
                stats["deferred_linears"].append(name)
            if not stats["deferred_linears"]:
                raise ValueError("no trainable plain Linear found; deferred arm would be a no-op")
        stack.enter_context(patch.object(trainer, "_clip_and_step", finish_update))
        try:
            yield stats
            if pending:
                raise RuntimeError("deferred gradients were not flushed by an optimizer boundary")
        finally:
            # Error paths must finish outstanding CUDA reads before restoring
            # module methods or releasing saved activations. Never step on error.
            for stream in streams:
                stream.synchronize()
            pending.clear()


class UpdateComparisonError(RuntimeError):
    """A rejected arm, with the completed baseline evidence retained."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(report["error"])


def compare_updates(
    trainer: Any,
    batch: Any,
    *,
    modes: tuple[str, ...],
    warmup: int,
    repeats: int,
    atol: float,
    rtol: float,
    trace_dir: Any = None,
    measure: bool = True,
) -> dict[str, Any]:
    """Compare complete updates from identical initial state after kernel warmup.

    The caller supplies a complete fixed batch. Timings include loss,
    backward, clipping and optimizer, not collection/reward/weight publication.
    Correctness clones and optional tracing are outside timing samples.
    """

    import asyncio
    import copy
    import math
    import random
    import statistics
    import time

    import numpy as np

    if warmup < 0 or repeats < 1 or not modes or len(set(modes)) != len(modes):
        raise ValueError("need warmup >= 0, repeats >= 1 and distinct nonempty modes")
    if "serial" not in modes or modes[0] != "serial" or atol < 0 or rtol < 0:
        raise ValueError("serial must be first; comparison tolerances must be nonnegative")
    if trainer._rollout_runtime.weight_syncer is not None:
        raise ValueError(
            "fixed-batch probe must not publish weights; construct an offline trainer"
        )
    # Reject an unsupported arm before warmup can mutate the trainer.
    for mode in modes:
        with replay_schedule(trainer, mode=mode):
            pass
    device = trainer.device
    parameters = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    if not parameters:
        raise ValueError("no trainable parameters")
    optimizer = trainer._ensure_optimizer()

    def snapshot(tree):
        if isinstance(tree, torch.Tensor):
            result = tree.detach().cpu().clone()
            if result.is_floating_point() and not bool(torch.isfinite(result).all()):
                raise RuntimeError("nonfinite tensor in update comparison")
            return result
        if isinstance(tree, float) and not math.isfinite(tree):
            raise RuntimeError("nonfinite scalar in update comparison")
        if isinstance(tree, dict):
            return {k: snapshot(v) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(snapshot(v) for v in tree)
        return copy.deepcopy(tree)

    initial = snapshot({n: p for n, p in trainer.model.named_parameters() if p.requires_grad})
    buffers = snapshot(dict(trainer.model.named_buffers()))
    optimizer_state = snapshot(optimizer.state_dict())
    trainer_state = copy.deepcopy(trainer.state)
    algorithm_state = copy.deepcopy(trainer.algorithm.__dict__)
    parity_passed = trainer._replay_parity_passed
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    def reset():
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(initial[name])
            for name, buffer in trainer.model.named_buffers():
                buffer.copy_(buffers[name])
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        trainer.state = copy.deepcopy(trainer_state)
        trainer.algorithm.__dict__.clear()
        trainer.algorithm.__dict__.update(copy.deepcopy(algorithm_state))
        trainer._replay_parity_passed = parity_passed
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)

    reference = None
    report: dict[str, Any] = {
        "scope": "fixed-batch trainer update; no reward/collection/transport",
        "timing_measured": measure,
        "arms": {},
    }
    try:
        # Warm kernels from the same policy that produced the trajectory. Keeping
        # warmup updates would make the fixed rollout stale and can clip every
        # gradient to zero before the comparison even begins.
        for _ in range(warmup):
            reset()
            asyncio.run(trainer.train_on_rollout_batch(batch))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        reset()

        for mode in modes:
            try:
                reset()
                gradients = {}
                clip = trainer._clip_and_step

                def capture(optim, gradients=gradients, clip=clip):
                    gradients.update(
                        {
                            n: p.grad.detach().clone()
                            for n, p in parameters.items()
                            if p.grad is not None
                        }
                    )
                    return clip(optim)

                # The replay adapter flushes dW before calling this capture.
                with (
                    patch.object(trainer, "_clip_and_step", capture),
                    replay_schedule(trainer, mode=mode) as stats,
                ):
                    metrics = asyncio.run(trainer.train_on_rollout_batch(batch))
                state = snapshot(
                    {
                        "gradients": gradients,
                        "parameters": parameters,
                        "buffers": dict(trainer.model.named_buffers()),
                        "optimizer": optimizer.state_dict(),
                        "loss": metrics.loss,
                        "grad_norm": metrics.grad_norm,
                    }
                )
                gradient_max_abs = max(
                    (float(g.abs().max()) for g in state["gradients"].values()),
                    default=0.0,
                )
                parameter_max_abs_change = max(
                    (float((p - initial[n]).abs().max()) for n, p in state["parameters"].items()),
                    default=0.0,
                )
                if gradient_max_abs == 0.0 or parameter_max_abs_change == 0.0:
                    raise RuntimeError(
                        f"insufficient update signal: gradient_max_abs={gradient_max_abs}, "
                        f"parameter_max_abs_change={parameter_max_abs_change}, "
                        f"clip_fraction={metrics.update.clip_fraction}"
                    )
                gradients.clear()
                if reference is None:
                    reference = state
                torch.testing.assert_close(state, reference, atol=atol, rtol=rtol, equal_nan=False)
                samples = []
                peaks = []
                for _ in range(repeats if measure else 0):
                    reset()
                    optimizer.zero_grad(set_to_none=True)
                    # Installing the experimental adapter and resetting state
                    # are setup, not work performed by an optimizer update.
                    with replay_schedule(trainer, mode=mode):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            memory_start = {
                                "allocated_start": torch.cuda.memory_allocated(device),
                                "reserved_start": torch.cuda.memory_reserved(device),
                            }
                        started = time.perf_counter()
                        asyncio.run(trainer.train_on_rollout_batch(batch))
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - started
                        if device.type == "cuda":
                            peaks.append(
                                {
                                    **memory_start,
                                    "allocated": torch.cuda.max_memory_allocated(device),
                                    "reserved": torch.cuda.max_memory_reserved(device),
                                }
                            )
                    samples.append(elapsed)
                    # Check every timed repetition too. A one-off good update
                    # must not hide a later stream/lifetime race; copies and
                    # assertions happen after the timer and peak sampling.
                    actual_state = snapshot(
                        {
                            "parameters": parameters,
                            "buffers": dict(trainer.model.named_buffers()),
                            "optimizer": optimizer.state_dict(),
                        }
                    )
                    expected_state = {key: reference[key] for key in actual_state}
                    torch.testing.assert_close(
                        actual_state, expected_state, atol=atol, rtol=rtol, equal_nan=False
                    )
                report["arms"][mode] = {
                    "correctness_passed": True,
                    "gradient_max_abs": gradient_max_abs,
                    "parameter_max_abs_change": parameter_max_abs_change,
                    "loss": metrics.loss,
                    "grad_norm": metrics.grad_norm,
                    "clip_fraction": metrics.update.clip_fraction,
                    "logprob_abs_diff_max": metrics.logprob_mismatch.logprob_abs_diff_max,
                    "seconds": samples,
                    "median_seconds": statistics.median(samples) if samples else None,
                    "peaks": peaks,
                    "schedule": stats,
                }
            except Exception as error:
                report["passed"] = False
                report["error"] = f"{mode}: {type(error).__name__}: {error}"
                report["arms"][mode] = {"correctness_passed": False, "error": str(error)}
                raise UpdateComparisonError(report) from error

        # Capture after every timing arm, so profiler allocation/destruction
        # cannot contaminate the next arm's latency or allocator history.
        if trace_dir is not None:
            for mode in modes:
                try:
                    reset()
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with (
                        torch.profiler.profile(activities=activities) as prof,
                        replay_schedule(trainer, mode=mode),
                    ):
                        asyncio.run(trainer.train_on_rollout_batch(batch))
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    prof.export_chrome_trace(str(trace_dir / f"{mode}.trace.json"))
                    del prof
                    if mode == "streams":
                        report["arms"][mode]["cuda_overlap"] = cuda_overlap_summary(
                            trace_dir / f"{mode}.trace.json"
                        )
                except Exception as error:
                    report["passed"] = False
                    report["error"] = f"{mode} trace: {type(error).__name__}: {error}"
                    report["arms"][mode]["trace_error"] = str(error)
                    raise UpdateComparisonError(report) from error

    finally:
        reset()
        optimizer.zero_grad(set_to_none=True)
    report["passed"] = True
    return report
