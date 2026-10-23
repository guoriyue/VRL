"""Rollout collector orchestration over the real generation stack (``real_collector``)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch

from tests.rollouts.collector._helpers import (
    CollectorBench,
    IndexReward,
    real_collector,
)
from vrl.generation import GenerationOutput
from vrl.ray.resources import RoleLifecycle
from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.base import RewardCleanupError
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.core import RolloutCollector
from vrl.trainers.weight_sync import flatten_trainable_module_state, to_cpu_snapshot
from vrl.trajectory.storage import TrajectoryStoragePolicy
from vrl.utils.stats import elapsed, recording


async def collect_scored(
    collector: RolloutCollector,
    inputs: list[Any],
    *,
    group_size: int,
    metadata: Mapping[str, Any] | None = None,
    request_overrides: Mapping[str, Any] | None = None,
    policy_version: int | None = None,
) -> RolloutBatch:
    """Collect one group and score it in a single call.

    Tests needing a single unsplit batch compose the low-level phases here.
    The public prompt-group API additionally restores prompt IDs and splits batches.
    """

    unscored = await collector.generate_rollout(
        collector.request_builder.build(
            inputs,
            group_size=group_size,
            metadata=metadata,
            request_overrides=request_overrides,
            policy_version=policy_version,
        )
    )
    return (collector.assemble_training_batches(await collector.evaluate_rollout([unscored])))[0]


def _sync_policy(bench: CollectorBench, version: int) -> None:
    """Push a policy version the way the trainer does: a CPU snapshot of the
    trainable modules. The worker serves a request only at its installed version."""

    asyncio.run(bench.runtime.activate())
    model = bench.runtime.worker.executor.model
    payload = to_cpu_snapshot(flatten_trainable_module_state(model.trainable_modules))
    asyncio.run(bench.runtime.update_weights(payload, version))


def _generated(bench: CollectorBench, prompts: list[str]) -> GenerationOutput:
    """One real single-sample generation per prompt, for batch-builder tests."""

    request = bench.collector.request_builder.build(prompts, 1).request
    return asyncio.run(bench.runtime.generate(request))


def test_collector_requires_runtime_before_collect(monkeypatch, tmp_path) -> None:
    bench = real_collector(monkeypatch, tmp_path)
    # The same run's collector before the trainer attaches its runtime.
    collector = RolloutCollector.from_family(
        bench.stack.family,
        reward_runtime=RewardFunctionRuntime(bench.reward),
        config=bench.stack.collector_config(),
        lifecycle=bench.stack.resources.lifecycle,
    )
    assert collector.generation_runtime is None

    with pytest.raises(RuntimeError, match="runtime is not initialized"):
        asyncio.run(collect_scored(collector, ["p0"], group_size=1))
    with pytest.raises(RuntimeError, match="runtime is not initialized"):
        asyncio.run(collector.activate_generation_runtime())

    collector.set_generation_runtime(bench.runtime)
    assert collector.generation_runtime is bench.runtime
    asyncio.run(collector.activate_generation_runtime())


@pytest.mark.asyncio
async def test_collector_shutdown_retries_in_safe_runtime_then_reward_order(
    monkeypatch, tmp_path
) -> None:
    bench = real_collector(monkeypatch, tmp_path)
    collector, trace = bench.collector, bench.trace
    trace.fail("runtime_shutdown", "runtime shutdown failed")
    trace.fail("reward_shutdown", "reward shutdown failed")

    with pytest.raises(RuntimeError, match="runtime shutdown failed"):
        await collector.shutdown()
    assert trace.events == ["runtime_shutdown"]
    assert collector._generation_runtime is bench.runtime
    assert collector._reward_shutdown_complete is False

    with pytest.raises(RuntimeError, match="reward shutdown failed"):
        await collector.shutdown()
    assert trace.events == ["runtime_shutdown", "runtime_shutdown", "reward_shutdown"]
    assert collector._generation_runtime is None
    assert collector._reward_shutdown_complete is False

    await collector.shutdown()
    await collector.shutdown()

    assert trace.events == [
        "runtime_shutdown",
        "runtime_shutdown",
        "reward_shutdown",
        "reward_shutdown",
    ]
    assert collector._generation_runtime is None
    assert collector._reward_shutdown_complete is True


def test_collector_routes_request_through_runtime_reward_and_trajectory_batch(
    monkeypatch, tmp_path
) -> None:
    """One collect call builds one request (prompts, group size as samples_per_prompt, overrides
    as sampling, policy version), scores every sample through the reward runtime with the
    collector metadata attached, and returns a batch whose rewards, group ids and sample rows
    line up prompt-major.
    """

    bench = real_collector(monkeypatch, tmp_path)
    _sync_policy(bench, 7)

    batch = asyncio.run(
        collect_scored(
            bench.collector,
            ["p0", "p1"],
            group_size=2,
            metadata={"collector": "metadata"},
            request_overrides={"seed": 5},
            policy_version=7,
        ),
    )

    assert len(bench.trace.requests) == 1
    request = bench.trace.requests[0]
    assert request.prompts == ["p0", "p1"]
    assert request.samples_per_prompt == 2
    assert request.sampling["seed"] == 5
    assert request.policy_version == 7
    call = bench.reward.calls[0]
    assert all(metadata["collector"] == "metadata" for metadata in call["metadata"])
    assert call["prompts"] == ["p0", "p0", "p1", "p1"]
    assert torch.stack(call["outputs"]).shape == (4, 3, 32, 32)
    assert bench.trace.events == ["activate", "update_weights", "generate", "score"]
    assert batch.rewards.tolist() == [0.0, 1.0, 2.0, 3.0]
    # The denoise pack path copies the reward metadata beside the request context.
    assert batch.context["reward_metadata"] == {
        "collector": "metadata",
        "rollout_policy_version": 7,
        "task_type": "text_to_image",
    }
    assert batch.trajectory is not None
    assert batch.group_ids.tolist() == [0, 0, 1, 1]
    assert [row.prompt_index for row in batch.trajectory.sample_rows] == [0, 0, 1, 1]


@pytest.mark.asyncio
async def test_profiled_collector_builds_cpu_batch_without_trainer_cuda_sync(
    monkeypatch, tmp_path
) -> None:
    bench = real_collector(monkeypatch, tmp_path)
    monkeypatch.setenv("VRL_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def reject_trainer_sync(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("collector touched the trainer CUDA device")

    monkeypatch.setattr(torch.cuda, "synchronize", reject_trainer_sync)

    batch = await collect_scored(bench.collector, ["p0"], group_size=1)

    assert batch.rewards.device.type == "cpu"
    assert batch.group_ids.device.type == "cpu"


def test_collector_offloads_runtime_memory_before_reward_scoring(monkeypatch, tmp_path) -> None:
    """Under a lifecycle plan that shares the reward GPU, scoring is bracketed by a rollout
    offload before the reward model activates and a reward park afterwards; the phase-final
    offload parks the rollout again and asks the reward to park again (the function knows
    it holds nothing by then).
    """

    # Shared reward GPU: the lifecycle plan (not the runtime) tells the collector
    # to park rollout GPU memory before the in-process reward model scores.
    bench = real_collector(
        monkeypatch,
        tmp_path,
        lifecycle=RoleLifecycle(trainer=(0,), rollout=(0,), reward=(0,)),
    )

    asyncio.run(collect_scored(bench.collector, ["p0"], group_size=1))
    asyncio.run(bench.collector.offload_generation_runtime_memory())

    assert bench.trace.events == [
        "generate",
        "offload",
        "score",
        "reward_park",
        "offload",
        "reward_park",
    ]


def test_collector_does_not_offload_runtime_before_independent_reward(
    monkeypatch, tmp_path
) -> None:
    """Checks collector keeps rollout active for an independent reward."""

    # Dedicated reward GPU: the plan keeps both roles resident, so the collector
    # never releases before reward.
    bench = real_collector(
        monkeypatch,
        tmp_path,
        lifecycle=RoleLifecycle(trainer=(0,), rollout=(1,), reward=(2,)),
    )

    asyncio.run(collect_scored(bench.collector, ["p0"], group_size=1))

    assert bench.trace.events == ["generate", "score"]


def test_collector_blocks_trainer_handoff_when_reward_parking_fails(monkeypatch, tmp_path) -> None:
    """A failed reward park remains terminal even after rollout itself parks."""

    bench = real_collector(
        monkeypatch,
        tmp_path,
        lifecycle=RoleLifecycle(trainer=(0,), rollout=(0,), reward=(0,)),
    )
    bench.trace.fail("reward_park", "reward park failed")
    bench.trace.fail("reward_park", "reward park failed")

    with pytest.raises(RuntimeError, match="reward park failed"):
        asyncio.run(collect_scored(bench.collector, ["p0"], group_size=1))
    with pytest.raises(RuntimeError, match="reward park failed"):
        asyncio.run(bench.collector.offload_generation_runtime_memory())

    # The park owed from scoring is retried at the phase-final handoff.
    assert bench.trace.events == [
        "generate",
        "offload",
        "score",
        "reward_park",
        "offload",
        "reward_park",
    ]


def test_collector_phase_final_gate_retries_reward_parking(monkeypatch, tmp_path) -> None:
    """The final rollout offload retries a transient reward park failure."""

    bench = real_collector(
        monkeypatch,
        tmp_path,
        lifecycle=RoleLifecycle(trainer=(0,), rollout=(0,), reward=(0,)),
    )
    bench.trace.fail("reward_park", "transient reward park failure")

    with pytest.raises(RuntimeError, match="transient reward park failure"):
        asyncio.run(collect_scored(bench.collector, ["p0"], group_size=1))

    asyncio.run(bench.collector.offload_generation_runtime_memory())

    assert bench.trace.events.count("reward_park") == 2
    assert bench.reward.memory_parked is True


def test_collector_aggregates_rollout_offload_and_owed_reward_park_failures(
    monkeypatch, tmp_path
) -> None:
    """Rollout failure cannot skip the reward park still owed from scoring."""

    bench = real_collector(
        monkeypatch,
        tmp_path,
        lifecycle=RoleLifecycle(trainer=(0,), rollout=(0,), reward=(0,)),
    )
    bench.trace.fail("reward_park", "reward park failed")
    bench.trace.fail("reward_park", "reward park failed")

    # Scoring parks and fails, so the reward still owes a park at the handoff.
    with pytest.raises(RuntimeError, match="reward park failed"):
        asyncio.run(collect_scored(bench.collector, ["p0"], group_size=1))
    bench.trace.fail("offload", "rollout offload failed")
    with pytest.raises(RewardCleanupError) as error:
        asyncio.run(bench.collector.offload_generation_runtime_memory())

    assert len(error.value.errors) == 2
    assert bench.trace.events[-2:] == ["offload", "reward_park"]


def test_collector_reports_rollout_offload_failure_alone_when_the_park_succeeds(
    monkeypatch, tmp_path
) -> None:
    bench = real_collector(
        monkeypatch,
        tmp_path,
        lifecycle=RoleLifecycle(trainer=(0,), rollout=(0,), reward=(0,)),
    )
    bench.trace.fail("offload", "rollout offload failed")

    with pytest.raises(RuntimeError, match="rollout offload failed"):
        asyncio.run(bench.collector.offload_generation_runtime_memory())

    assert bench.trace.events == ["offload", "reward_park"]


def _reward_sample_builder(
    bench: CollectorBench,
    prompts: list[str],
    *,
    outputs: Any = None,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryRolloutBatchBuilder:
    output = _generated(bench, prompts)
    if outputs is not None:
        output.output = outputs
    return TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata=dict(metadata or {})),
    )


def test_reward_samples_forward_boxed_media_without_resolving(monkeypatch, tmp_path) -> None:
    from vrl.utils.media_reference import MediaReference

    def unexpected_resolve(*args, **kwargs):
        pytest.fail("collector must not resolve media references")

    monkeypatch.setattr(MediaReference, "resolve", unexpected_resolve)
    refs = [MediaReference("boxed-ref", i, nbytes=48) for i in range(2)]
    builder = _reward_sample_builder(
        real_collector(monkeypatch, tmp_path), ["p0", "p1"], outputs=refs
    )

    samples = builder.reward_samples()

    assert [sample.output for sample in samples] == refs


def test_reward_samples_reject_sample_row_output_mismatch(monkeypatch, tmp_path) -> None:
    builder = _reward_sample_builder(
        real_collector(monkeypatch, tmp_path),
        ["p0"],
        outputs=torch.ones(2, 3),
    )

    with pytest.raises(ValueError, match="sample-row/output batch mismatch"):
        builder.reward_samples()


def test_collector_uses_one_reward_call_and_splits_scores_per_group(monkeypatch, tmp_path) -> None:
    """One reward call preserves every group's metadata and sample identity, and
    reward components travel with the group they describe."""

    class _ComponentReward(IndexReward):
        async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
            output = await super().score_batch(samples)
            return RewardOutput(
                scores=output.scores,
                components={
                    "observer": tuple(value + 10.0 for value in output.scores),
                },
            )

    bench = real_collector(monkeypatch, tmp_path, reward=_ComponentReward())
    collector = bench.collector

    async def _collect_groups():
        first = await collector.generate_rollout(
            collector.request_builder.build(
                ["same prompt"],
                group_size=2,
                metadata={"target_text": "group-0"},
            )
        )
        second = await collector.generate_rollout(
            collector.request_builder.build(
                ["same prompt"],
                group_size=3,
                metadata={"target_text": "group-1"},
            )
        )
        return collector.assemble_training_batches(
            await collector.evaluate_rollout([first, second])
        )

    first, second = asyncio.run(_collect_groups())

    assert len(bench.reward.calls) == 1
    call = bench.reward.calls[0]
    assert call["prompts"] == ["same prompt"] * 5
    assert [metadata["target_text"] for metadata in call["metadata"]] == [
        "group-0",
        "group-0",
        "group-1",
        "group-1",
        "group-1",
    ]
    assert len(set(call["sample_ids"])) == 5
    assert first.rewards.tolist() == [0.0, 1.0]
    assert first.extras["reward_components"]["observer"].tolist() == [10.0, 11.0]
    assert second.rewards.tolist() == [2.0, 3.0, 4.0]
    assert second.extras["reward_components"]["observer"].tolist() == [12.0, 13.0, 14.0]


def test_prepare_training_batches_collects_reward_timing_into_stats(monkeypatch, tmp_path) -> None:
    """What the reward records during the call's scoring lands in the caller's stats."""

    class _TimedReward(IndexReward):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
            self.batch_sizes.append(len(samples))
            elapsed("reward.latency_s", 0.012)
            elapsed("reward.inference_s", 0.009)
            return RewardOutput(scores=tuple(float(index + 1) for index in range(len(samples))))

    reward = _TimedReward()
    collector = real_collector(monkeypatch, tmp_path, reward=reward).collector
    with recording() as stats:
        batches = asyncio.run(
            collector.prepare_training_batches(
                prompts=["p0"],
                group_size=2,
                runtime_debug=False,
                policy_version=None,
            ),
        )

    assert reward.batch_sizes == [2]
    assert len(batches) == 1
    assert batches[0].rewards.tolist() == [1.0, 2.0]
    metrics = stats.as_metrics_dict()
    assert metrics["reward.latency_s"] == 0.012
    assert metrics["reward.inference_s"] == 0.009
    assert metrics["reward.call_count"] == 1.0


def test_reward_output_is_independent_of_trajectory_storage_dtype(monkeypatch, tmp_path) -> None:
    """Trainer replay compression must not change the artifact being scored."""

    output = _generated(real_collector(monkeypatch, tmp_path), ["p0"])
    canonical = torch.tensor(
        [[[[0.1234567, -0.2345678], [0.3456789, -0.4567891]]]],
        dtype=torch.float32,
    )
    output.output = canonical

    builder = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(
            metadata={},
            trajectory_storage_policy=TrajectoryStoragePolicy(dtype="float16"),
        ),
    )
    replay_log_probs = builder.trajectory.segments["denoise"].tensors["old_log_prob"].value

    assert replay_log_probs.dtype == torch.float16
    assert output.output is canonical
    assert torch.equal(builder.reward_outputs(), canonical)


def test_reward_outputs_reconstructs_uint8_wire_video_exactly(monkeypatch, tmp_path) -> None:
    """Checks uint8 wire-packed video reconstructs to k/255 floats.

    The worker packs decoded video as uint8 before the wire (wire diet T1);
    reward_outputs must hand consumers [0, 1] floats that round-trip
    bit-exactly through every downstream to_uint8 quantization, keeping
    reward scores identical to the fp32-over-wire path.
    """

    from vrl.utils.media import to_uint8

    output = _generated(real_collector(monkeypatch, tmp_path), ["p0"])
    packed = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    output.output = packed

    reconstructed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}),
    ).reward_outputs()

    assert reconstructed.dtype == torch.float32
    assert float(reconstructed.min()) >= 0.0
    assert float(reconstructed.max()) <= 1.0
    # The G3 guarantee: re-quantizing recovers every byte value exactly.
    assert torch.equal(to_uint8(reconstructed), packed)


def test_collector_does_not_coerce_group_size_before_request_validation(
    monkeypatch, tmp_path
) -> None:
    bench = real_collector(monkeypatch, tmp_path)
    collector = bench.collector
    with pytest.raises(ValueError, match="samples_per_prompt must be an integer"):
        asyncio.run(
            collector.generate_rollout(collector.request_builder.build(["p0"], group_size=True))
        )
    assert bench.trace.requests == []


def test_request_generation_reward_and_assembly_are_separate_stages(monkeypatch, tmp_path) -> None:
    bench = real_collector(monkeypatch, tmp_path)
    _sync_policy(bench, 7)
    collector, trace, reward = bench.collector, bench.trace, bench.reward
    requests = list(
        collector.build_generation_requests(
            prompts=["first", "second"],
            group_size=2,
            runtime_debug=False,
            policy_version=7,
        )
    )
    assert trace.requests == []
    assert reward.calls == []
    assert len(requests) == 1
    request, indices = requests[0]
    assert indices == [0, 1]
    assert request.request.samples_per_prompt == 2

    assembled = []
    original_build = TrajectoryRolloutBatchBuilder.build

    def record_build(self, reward_values):
        assembled.append(reward_values)
        return original_build(self, reward_values)

    monkeypatch.setattr(TrajectoryRolloutBatchBuilder, "build", record_build)

    async def run():
        rollout = await collector.generate_rollout(request)
        assert len(trace.requests) == 1
        assert reward.calls == []
        evaluation = await collector.evaluate_rollout([rollout])
        assert len(reward.calls) == 1
        assert assembled == []
        batches = collector.assemble_training_batches(evaluation)
        assert len(assembled) == 1
        return batches

    batches = asyncio.run(run())
    assert len(batches) == 1
    assert batches[0].rewards.tolist() == [0.0, 1.0, 2.0, 3.0]
