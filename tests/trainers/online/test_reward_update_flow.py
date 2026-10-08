"""OnlineTrainer rollout consume + update loop: prompt-kwarg forwarding, batching, gradient
accumulation, loss scaling, streaming release, and batch-op field preservation.

The trainer-driving tests run the online recipe's own wiring on the tiny SANA
stack (``real_trainer``): real GRPO, the real denoise SDE evaluator, the real
collector and in-process rollout runtime. Observation goes through wrappers
around the real methods, never through substituted behaviour.
"""

from __future__ import annotations

import asyncio
import gc
import random
import weakref
from typing import Any

import pytest
import torch

from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.rollouts.batch import RolloutBatch
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.data.prompts import PromptExample
from vrl.trainers.online.config import OnlineBatchPlan

# One replay epoch and per-group std keep the streaming and full-batch paths
# comparable: streaming refuses multi-epoch updates, and a global std would
# normalize over a different sample set per microbatch.
_ONE_EPOCH = ("actor.ppo_epochs=1", "algorithm.global_std=false")
# Three sampling steps with the full timestep fraction train two denoise
# transitions per group (the SDE window covers the first two).
_TWO_TRAIN_TIMESTEPS = ("sampling.num_steps=3", "actor.timestep_fraction=1.0")
_STREAM_FOUR = ("rollout.prompts_per_batch=4", "actor.prompts_per_collection=1")


def _stream(bench: TrainerBench, prompts: list[Any], **kwargs: Any) -> Any:
    return asyncio.run(
        _run_streaming_optimizer_update(
            bench.trainer,
            prompts,
            **kwargs,
        ),
    )


def _requested_prompts(bench: TrainerBench) -> list[list[str]]:
    return [list(request.prompts) for request in bench.collector.trace.requests]


def _pushes(bench: TrainerBench) -> list[tuple[Any, int]]:
    return [args for event, args in bench.collector.trace.calls if event == "update_weights"]


def _record_evaluations(monkeypatch, bench: TrainerBench) -> list[tuple[int, list[int], int]]:
    """(batch size, group ids, timestep index) of every real evaluator replay."""

    calls: list[tuple[int, list[int], int]] = []
    real = bench.trainer.evaluator.evaluate

    def evaluate(model, batch, timestep_idx, **kwargs):
        calls.append((int(batch.rewards.shape[0]), batch.group_ids.tolist(), int(timestep_idx)))
        return real(model, batch, timestep_idx, **kwargs)

    monkeypatch.setattr(bench.trainer.evaluator, "evaluate", evaluate)
    return calls


def _record_loss_scales(monkeypatch, bench: TrainerBench) -> list[float]:
    """d(backpropagated loss) / d(replay loss) for every backward.

    The trainer backpropagates ``replay_loss * loss_weight / loss_scale``; the
    derivative is that factor exactly, even when the replay loss itself is 0.
    """

    scales: list[float] = []
    last: dict[str, torch.Tensor] = {}
    trainer = bench.trainer
    real_loss = trainer._compute_replay_loss
    real_backward = trainer._backward

    def compute_replay_loss(*args, **kwargs):
        result = real_loss(*args, **kwargs)
        last["loss"] = result[0]
        return result

    def backward(loss):
        (factor,) = torch.autograd.grad(loss, last["loss"], retain_graph=True)
        scales.append(float(factor))
        return real_backward(loss)

    monkeypatch.setattr(trainer, "_compute_replay_loss", compute_replay_loss)
    monkeypatch.setattr(trainer, "_backward", backward)
    return scales


def _record_optimizer_steps(monkeypatch, bench: TrainerBench) -> list[dict[str, torch.Tensor]]:
    """The accumulated gradient at every real optimizer step."""

    grads: list[dict[str, torch.Tensor]] = []
    real = bench.trainer._clip_and_step

    def clip_and_step(optimizer):
        grads.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in bench.trainable_parameters().items()
                if parameter.grad is not None
            }
        )
        return real(optimizer)

    monkeypatch.setattr(bench.trainer, "_clip_and_step", clip_and_step)
    return grads


def test_step_forwards_prompt_example_fields(monkeypatch, tmp_path) -> None:
    """PromptExample conditioning reaches the generation input; its reward targets reach
    the reward samples as group metadata."""

    bench = real_trainer(monkeypatch, tmp_path, overrides=("actor.ppo_epochs=1",))
    example = PromptExample(
        prompt="sign says HELLO",
        target_text="HELLO",
        reference_images=["/tmp/reference.png"],
        task_type="text_to_video",
        metadata={"difficulty": "easy"},
    )

    asyncio.run(bench.trainer.step([example]))

    (request,) = bench.collector.trace.requests
    assert request.samples_per_prompt == 2
    assert len(request.inputs) == 1
    assert request.inputs[0].reference_images == ["/tmp/reference.png"]
    assert request.inputs[0].task_type == "text_to_video"
    metadata = bench.collector.reward.calls[0]["metadata"]
    assert all(row["target_text"] == "HELLO" for row in metadata)
    assert all(row["difficulty"] == "easy" for row in metadata)


def test_plain_prompts_collect_together_but_train_group_locally(monkeypatch, tmp_path) -> None:
    """Plain prompts share one rollout request, then every replay sees exactly one group."""

    bench = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(*_ONE_EPOCH, *_TWO_TRAIN_TIMESTEPS, "rollout.prompts_per_batch=2"),
    )
    evaluations = _record_evaluations(monkeypatch, bench)

    asyncio.run(bench.trainer.step(["prompt-a", "prompt-b"]))

    assert _requested_prompts(bench) == [["prompt-a", "prompt-b"]]
    assert [(size, groups) for size, groups, _ in evaluations] == [
        (2, [0, 0]),
        (2, [0, 0]),
        (2, [1, 1]),
        (2, [1, 1]),
    ]
    assert [timestep for _, _, timestep in evaluations] == [0, 1, 0, 1]


def test_streaming_accumulation_runs_one_optimizer_step(monkeypatch, tmp_path) -> None:
    """prompts_per_collection>0 streams collection batches into ONE optimizer update."""

    bench = real_trainer(monkeypatch, tmp_path, overrides=(*_ONE_EPOCH, *_STREAM_FOUR))
    steps = _record_optimizer_steps(monkeypatch, bench)
    bench.collector.trace.watch(
        bench.trainer.rollout_schedule, "after_train_step", "after_train_step"
    )

    metrics = _stream(bench, ["prompt-a", "prompt-b", "prompt-c", "prompt-d"])

    # 4 microbatches of 1 prompt each, collected/trained/released separately...
    assert _requested_prompts(bench) == [["prompt-a"], ["prompt-b"], ["prompt-c"], ["prompt-d"]]
    # ...but ONE optimizer update, published once after the train half.
    assert len(steps) == 1
    assert bench.trainer.state.step == 1
    assert bench.trainer.state.global_step == 1
    assert [version for _, version in _pushes(bench)] == [1, 2]
    # The post-train sync's own phases reach the update's metrics.
    (sync_stats,) = bench.collector.trace.results["after_train_step"]
    assert sync_stats.phase_seconds
    assert set(sync_stats.phase_seconds) <= set(metrics.phase_times)


def test_streaming_releases_microbatch_before_next_collect(monkeypatch, tmp_path) -> None:
    """Streaming must not retain the previous rollout batch while collecting the next."""

    bench = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(*_ONE_EPOCH, "rollout.prompts_per_batch=2", "actor.prompts_per_collection=1"),
    )
    released_before_collect: list[bool] = []
    batch_refs: list[weakref.ref] = []
    real = bench.trainer.collect_training_batch

    async def collect_training_batch(prompts, *, next_prompts=None):
        if batch_refs:
            gc.collect()
            released_before_collect.append(batch_refs[-1]() is None)
        batch = await real(prompts, next_prompts=next_prompts)
        batch_refs.extend(weakref.ref(rollout) for rollout in batch.batches)
        return batch

    monkeypatch.setattr(bench.trainer, "collect_training_batch", collect_training_batch)

    _stream(bench, ["prompt-a", "prompt-b"])
    gc.collect()

    assert released_before_collect == [True]
    assert all(ref() is None for ref in batch_refs)


def test_streaming_stats_sum_phases_and_keep_peak_gauges(monkeypatch, tmp_path) -> None:
    """Microbatch stats retain summed durations and the peak continuous state."""

    monkeypatch.setenv("VRL_PROFILE", "1")
    bench = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(
            *_ONE_EPOCH,
            "rollout.prompts_per_batch=2",
            "actor.prompts_per_collection=1",
            "/base/rollout/orchestration=continuous",
        ),
    )
    microbatch_stats: list[Any] = []
    real = bench.trainer._step_stats

    def step_stats(iteration, timer):
        stats = real(iteration, timer)
        microbatch_stats.append(stats)
        return stats

    monkeypatch.setattr(bench.trainer, "_step_stats", step_stats)

    async def run_two_updates():
        try:
            await _run_streaming_optimizer_update(
                bench.trainer,
                ["prompt-a", "prompt-b"],
                next_example_batch=["prompt-c", "prompt-d"],
            )
            microbatch_stats.clear()
            # The second update consumes groups produced before the first
            # update's weight sync, so one microbatch is a version stale.
            return await _run_streaming_optimizer_update(
                bench.trainer,
                ["prompt-c", "prompt-d"],
            )
        finally:
            await bench.trainer.rollout_schedule.shutdown()

    metrics = asyncio.run(run_two_updates())

    assert len(microbatch_stats) == 2
    collect_phases = {
        name
        for stats in microbatch_stats
        for name in stats.phase_seconds
        if name.startswith("collect.")
    }
    assert "collect.engine_generate" in collect_phases
    for name in collect_phases:
        assert metrics.phase_times[name] == pytest.approx(
            sum(stats.phase_seconds.get(name, 0.0) for stats in microbatch_stats)
        )
    gauges = {name for stats in microbatch_stats for name in stats.gauges}
    assert "continuous.stale_policy_versions" in gauges
    for name in gauges:
        assert metrics.phase_times[name] == max(
            stats.gauges[name] for stats in microbatch_stats if name in stats.gauges
        )
    staleness = [stats.gauges["continuous.stale_policy_versions"] for stats in microbatch_stats]
    assert max(staleness) > min(staleness)


def test_streaming_announces_the_next_prompt_batch_before_backward(monkeypatch, tmp_path) -> None:
    """Each collect announces the prompt batch that runs during its backward."""

    bench = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(*_ONE_EPOCH, "rollout.prompts_per_batch=2", "actor.prompts_per_collection=1"),
    )
    announced: list[tuple[list[str], list[str] | None]] = []
    real = bench.trainer.collect_training_batch

    async def collect_training_batch(prompts, *, next_prompts=None):
        announced.append((list(prompts), next_prompts))
        return await real(prompts, next_prompts=next_prompts)

    monkeypatch.setattr(bench.trainer, "collect_training_batch", collect_training_batch)

    _stream(bench, ["prompt-a", "prompt-b"], next_example_batch=["prompt-c", "prompt-d"])

    assert announced == [
        (["prompt-a"], ["prompt-b"]),
        (["prompt-b"], ["prompt-c"]),
    ]


def test_flow_grpo_loss_scaling_includes_timesteps(monkeypatch, tmp_path) -> None:
    """Streaming accumulation scales every backward by 1 / (groups * train timesteps)."""

    bench = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(*_ONE_EPOCH, *_TWO_TRAIN_TIMESTEPS, *_STREAM_FOUR),
    )
    scales = _record_loss_scales(monkeypatch, bench)
    steps = _record_optimizer_steps(monkeypatch, bench)

    _stream(bench, ["prompt-a", "prompt-b", "prompt-c", "prompt-d"])

    # 4 microbatches (1 group each) * 2 train timesteps, one optimizer update:
    # each backward carries 1 / (4 * 2), so the accumulated gradient is the
    # mean over every trained (group, timestep), not 2x or 8x it.
    assert len(scales) == 8
    assert scales == pytest.approx([1 / 8] * 8)
    assert len(steps) == 1
    assert bench.trainer.state.global_step == 1


def test_streaming_matches_full_batch_gradient(monkeypatch, tmp_path) -> None:
    """Streaming microbatches accumulate exactly the full-batch gradient.

    Both trainers start from the same snapshot weights and draw the same
    request seeds (one request per PromptExample either way), so both see the
    same real rollouts; only the update's batching differs.
    """

    prompts = [PromptExample(prompt=text) for text in ("p0", "p1", "p2", "p3")]
    full = real_trainer(
        monkeypatch,
        tmp_path / "full",
        overrides=(*_ONE_EPOCH, "rollout.prompts_per_batch=4"),
    )
    # Each stack serves one snapshot: load the first policy before the second
    # stack installs its pipeline.
    asyncio.run(full.collector.collector.activate_generation_runtime())
    streaming = real_trainer(
        monkeypatch,
        tmp_path / "streaming",
        overrides=(*_ONE_EPOCH, *_STREAM_FOUR),
    )
    full_steps = _record_optimizer_steps(monkeypatch, full)
    streaming_steps = _record_optimizer_steps(monkeypatch, streaming)
    initial = full.trainable_parameters()
    assert all(
        torch.equal(initial[name], value)
        for name, value in streaming.trainable_parameters().items()
    )

    random.seed(0)
    asyncio.run(full.trainer.step(list(prompts)))
    random.seed(0)
    _stream(streaming, list(prompts))

    assert [r.sampling["seed"] for r in full.collector.trace.requests] == [
        r.sampling["seed"] for r in streaming.collector.trace.requests
    ]
    assert len(full_steps) == len(streaming_steps) == 1
    assert full.trainer.state.global_step == streaming.trainer.state.global_step == 1
    full_grad, streaming_grad = full_steps[0], streaming_steps[0]
    assert full_grad.keys() == streaming_grad.keys()
    assert any(bool(grad.abs().sum() > 0) for grad in full_grad.values())
    for name, grad in full_grad.items():
        torch.testing.assert_close(streaming_grad[name], grad, rtol=1e-6, atol=1e-7)


def test_training_microbatch_size_splits_backward_and_preserves_gradient(
    monkeypatch, tmp_path
) -> None:
    """The replay-only batch integer changes call shape without changing gradients."""

    device_move_sizes: list[int] = []
    original_to_device = RolloutBatch.to_device

    def recording_to_device(batch, *args, **kwargs):
        device_move_sizes.append(int(batch.rewards.shape[0]))
        return original_to_device(batch, *args, **kwargs)

    monkeypatch.setattr(RolloutBatch, "to_device", recording_to_device)

    def run(name: str, microbatch: int, *, streaming: bool, activate_after: bool):
        overrides = (
            *_ONE_EPOCH,
            "rollout.n_samples_per_prompt=4",
            "rollout.samples_per_generation_batch=4",
            f"actor.training_microbatch_size={microbatch}",
        )
        if streaming:
            overrides += ("actor.prompts_per_collection=1",)
        bench = real_trainer(monkeypatch, tmp_path / name, overrides=overrides)
        evaluations = _record_evaluations(monkeypatch, bench)
        steps = _record_optimizer_steps(monkeypatch, bench)
        device_move_sizes.clear()
        random.seed(0)
        if streaming:
            _stream(bench, [PromptExample(prompt="prompt")])
        else:
            asyncio.run(bench.trainer.step([PromptExample(prompt="prompt")]))
        if activate_after:
            # Load this stack's policy before the next stack installs its pipeline.
            asyncio.run(bench.collector.collector.activate_generation_runtime())
        assert len(steps) == 1
        return steps[0], [size for size, _, _ in evaluations], list(device_move_sizes)

    full_grad, full_calls, full_moves = run("full", 0, streaming=False, activate_after=True)
    split_grad, split_calls, split_moves = run("split", 2, streaming=False, activate_after=True)
    stream_grad, stream_calls, stream_moves = run(
        "stream", 2, streaming=True, activate_after=False
    )

    assert full_calls == [4]
    assert 4 in full_moves
    assert split_calls == [2, 2]
    assert stream_calls == [2, 2]
    assert max(split_moves) == 2
    assert max(stream_moves) == 2
    assert any(bool(grad.abs().sum() > 0) for grad in full_grad.values())
    for name, grad in full_grad.items():
        torch.testing.assert_close(split_grad[name], grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(stream_grad[name], grad, rtol=1e-5, atol=1e-7)


def test_rollout_memory_plan_logs_streaming_and_legacy_warning(caplog) -> None:
    """Startup logs should make rollout microbatch residency visible."""
    import logging

    from vrl.scripts.common.online import _log_rollout_memory_plan

    def _plan(rbs: int, gas: int) -> OnlineBatchPlan:
        return OnlineBatchPlan(
            prompts_per_batch=rbs,
            n_samples_per_prompt=2,
            prompts_per_collection=(rbs // gas) if gas else 0,
        )

    logger_name = "vrl.scripts.common.online"
    with caplog.at_level(logging.INFO, logger=logger_name):
        _log_rollout_memory_plan(
            _plan(4, 4),
            samples_per_generation_batch=2,
        )
    streaming_messages = [record.getMessage() for record in caplog.records]
    assert any("streaming accumulation enabled" in msg for msg in streaming_messages)
    assert any("collection_prompts=1" in msg for msg in streaming_messages)
    assert any("samples_per_generation_batch=2" in msg for msg in streaming_messages)
    assert any("training_microbatch_size=1" in msg for msg in streaming_messages)
    assert any("target_samples_per_update=8" in msg for msg in streaming_messages)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger_name):
        _log_rollout_memory_plan(
            _plan(4, 0),
            samples_per_generation_batch=2,
        )
    legacy_messages = [record.getMessage() for record in caplog.records]
    assert any("legacy full-batch accumulation" in msg for msg in legacy_messages)
    assert any("samples_per_generation_batch=2" in msg for msg in legacy_messages)
    assert any("training_microbatch_size=1" in msg for msg in legacy_messages)
    # The legacy path must emit a host-RAM residency WARNING. Assert the warning
    # level fired (the behavioral contract) rather than pinning its exact prose,
    # which a benign reword would redden with no real regression.
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_global_std_streaming_scope_logging(caplog) -> None:
    """Both one- and multi-group slices require update-wide normalization."""
    import logging

    from vrl.scripts.common.online import _log_global_std_streaming_scope

    def _plan(rbs: int, gas: int) -> OnlineBatchPlan:
        return OnlineBatchPlan(
            prompts_per_batch=rbs,
            n_samples_per_prompt=2,
            prompts_per_collection=(rbs // gas) if gas else 0,
        )

    logger_name = "vrl.scripts.common.online"

    def _warns(plan, *, global_std: bool) -> bool:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=logger_name):
            _log_global_std_streaming_scope(plan, global_std=global_std)
        return any("global_std streaming:" in r.getMessage() for r in caplog.records)

    assert _warns(_plan(8, 4), global_std=True)
    assert _warns(_plan(8, 8), global_std=True)
    # Exempt: global_std=false (per-group std is streaming-equivalent).
    assert not _warns(_plan(8, 4), global_std=False)
    # Exempt: legacy full-batch (gas=0, no streaming).
    assert not _warns(_plan(8, 0), global_std=True)


def test_host_memory_budget_fail_fast(monkeypatch) -> None:
    """The host-RAM guard raises over budget and passes under it (injected RSS)."""
    import pytest

    from vrl.scripts.common import online
    from vrl.utils.memory import HostMemoryMonitor, HostMemorySnapshot

    def _inject(used_fraction: float) -> None:
        # total=100GiB; available carved so used_fraction comes out as asked.
        total = 100_000.0
        snap = HostMemorySnapshot(
            rss_mb=total * used_fraction,
            available_mb=total * (1.0 - used_fraction),
            total_mb=total,
        )
        monkeypatch.setattr(HostMemoryMonitor, "capture", lambda self: snap)

    # Over budget -> fail fast with an actionable message.
    _inject(0.95)
    with pytest.raises(MemoryError, match=r"actor\.prompts_per_collection"):
        online._check_host_memory_budget(0.9, collection_prompts=1, n_samples_per_prompt=8)

    # Exactly at / under budget -> pass (<= budget does not trip).
    _inject(0.90)
    online._check_host_memory_budget(0.9, collection_prompts=1, n_samples_per_prompt=8)
    _inject(0.50)
    online._check_host_memory_budget(0.9, collection_prompts=1, n_samples_per_prompt=8)

    # Unreadable host memory (used_fraction None) -> never raises (no false kill).
    monkeypatch.setattr(
        HostMemoryMonitor,
        "capture",
        lambda self: HostMemorySnapshot(rss_mb=None, available_mb=None, total_mb=None),
    )
    online._check_host_memory_budget(0.9, collection_prompts=1, n_samples_per_prompt=8)


def test_select_move_and_remap_preserve_rollout_trajectory_fields() -> None:
    """Selecting, moving and remapping a batch keep every rollout trajectory field (token ids,
    log-probs, masks, prompt and uncond ids) attached and consistent with the selected rows.
    """
    import torch

    from vrl.generation import GenerationRequest, GenerationSampleRow
    from vrl.rollouts.batch import RolloutBatch
    from vrl.trajectory.builders import build_diffusion_trajectory

    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["a", "b"],
        samples_per_prompt=2,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=index // 2,
            sample_index=index % 2,
            prompt=request.prompts[index // 2],
            sample_id=f"s{index}",
        )
        for index in range(4)
    ]
    actions = torch.arange(8, dtype=torch.float32).view(4, 2, 1)
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros_like(actions),
        actions=actions,
        old_log_prob=torch.zeros(4, 2),
        timesteps=torch.zeros(4, 2),
        replay_tensors={},
        context={"model_family": "sd3_5"},
    )
    batch = RolloutBatch(
        rewards=torch.arange(4, dtype=torch.float32),
        group_ids=torch.tensor([0, 0, 1, 1]),
        trajectory=trajectory,
    )

    selected = batch.select(torch.tensor([True, False, True, False]))

    assert selected.trajectory is not None
    assert selected.trajectory.primary_segment == "denoise"
    assert selected.trajectory.axes["sample"].length == 2
    assert [row.prompt_index for row in selected.trajectory.sample_rows] == [0, 1]
    assert torch.equal(
        selected.trajectory.segments["denoise"].tensors["actions"].value,
        torch.tensor([[[0.0], [1.0]], [[4.0], [5.0]]]),
    )

    moved = selected.to_device(torch.device("cpu"))
    assert moved.trajectory is not None

    moved.remap_group_ids_([10, 11])
    assert torch.equal(moved.group_ids, torch.tensor([10, 11]))
    assert moved.trajectory is not None
    assert [row.prompt_index for row in moved.trajectory.sample_rows] == [0, 1]
