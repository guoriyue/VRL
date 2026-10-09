"""Shared helpers for rollout collector and orchestration tests.

``real_collector`` is the one construction every test in this family starts
from: ``RolloutCollector.from_family`` on the tiny SANA family with
``InProcessGenerationRuntime`` running the production worker body and a real
``RewardFunction``. ``Trace`` wraps the real coroutines to record their order
and to inject the one-shot failures handoff tests need.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.generation._in_process_runtime import InProcessGenerationRuntime
from tests.scripts.eval.fixtures import TinySanaStack, tiny_sana_stack
from vrl.generation import GenerationRequest
from vrl.models.parking import TrainingMemoryState
from vrl.ray.resources import RayLifecyclePlan
from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.base import RewardFunction
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.trainers.strategy import SingleProcessStrategy
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer


class Trace:
    """Order of the real runtime and reward operations, plus one-shot faults.

    ``watch`` wraps one coroutine method of a real object: the call is
    recorded, a failure queued for that event raises instead of delegating,
    otherwise the real method runs.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.threads: list[int] = []
        self.results: dict[str, list[Any]] = {}
        self._failures: dict[str, list[BaseException]] = {}

    @property
    def events(self) -> list[str]:
        return [event for event, _ in self.calls]

    @property
    def requests(self) -> list[GenerationRequest]:
        return [args[0] for event, args in self.calls if event == "generate"]

    def thread_ids(self, event: str) -> list[int]:
        """The thread each call of ``event`` ran on, in call order."""

        return [
            thread
            for (name, _), thread in zip(self.calls, self.threads, strict=True)
            if name == event
        ]

    def fail(self, event: str, error: str | BaseException, *, times: int = 1) -> None:
        """Queue ``times`` failures for the next calls of ``event``."""

        for _ in range(times):
            self._failures.setdefault(event, []).append(
                RuntimeError(error) if isinstance(error, str) else error
            )

    def watch(self, target: Any, method: str, event: str) -> None:
        real = getattr(target, method)

        def enter() -> None:
            pending = self._failures.get(event)
            if pending:
                raise pending.pop(0)

        def record(result: Any) -> Any:
            self.results.setdefault(event, []).append(result)
            return result

        if inspect.iscoroutinefunction(real):

            async def traced(*args: Any, **kwargs: Any) -> Any:
                self.calls.append((event, args))
                self.threads.append(threading.get_ident())
                enter()
                return record(await real(*args, **kwargs))

        else:

            def traced(*args: Any, **kwargs: Any) -> Any:
                self.calls.append((event, args))
                self.threads.append(threading.get_ident())
                enter()
                return record(real(*args, **kwargs))

        self._monkeypatch.setattr(target, method, traced)


class IndexReward(RewardFunction):
    """Scores each sample by its position and records what it was asked to score."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.memory_parked = False

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        self.calls.append(
            {
                "outputs": [sample.output for sample in samples],
                "prompts": [sample.prompt for sample in samples],
                "metadata": [sample.metadata for sample in samples],
                "sample_ids": [sample.sample_id for sample in samples],
            },
        )
        self.memory_parked = False
        return RewardOutput(scores=tuple(float(index) for index in range(len(samples))))

    async def park_memory(self) -> None:
        self.memory_parked = True


@dataclass(slots=True)
class CollectorBench:
    stack: TinySanaStack
    collector: RolloutCollector
    runtime: InProcessGenerationRuntime
    reward: RewardFunction
    trace: Trace


@dataclass
class TrainerSide:
    """The trainer's half of a rollout schedule: the run's real policy bundle,
    its single-process strategy, and whether the runtime already holds its
    weights when the coordinator starts."""

    bundle: Any
    strategy: SingleProcessStrategy
    initialized: bool = False

    def training_state(self) -> TrainingMemoryState:
        return TrainingMemoryState(
            model=self.bundle.model,
            ref_model=None,
            optimizer=None,
            ema=None,
            grad_scaler=None,
            device=torch.device("cpu"),
        )

    def export(self) -> dict[str, Any]:
        return self.strategy.export_rollout_state(self.bundle)

    def coordinator(
        self, bench: CollectorBench, *, syncer: bool = True
    ) -> RolloutRuntimeCoordinator:
        """The coordinator the trainer builds, over this bench and this trainer side."""

        coordinator = RolloutRuntimeCoordinator(
            collector=bench.collector,
            strategy=self.strategy,
            training_state_getter=self.training_state,
            weight_syncer=RayRuntimeWeightSyncer(bench.runtime) if syncer else None,
            sync_state_getter=self.export if syncer else None,
        )
        coordinator.weights_initialized = self.initialized
        return coordinator


def trainer_side(bench: CollectorBench, *, initialized: bool = False) -> TrainerSide:
    return TrainerSide(
        bundle=bench.stack.trainer_bundle(),
        strategy=SingleProcessStrategy(),
        initialized=initialized,
    )


# Versioned trainable-state slots: a continuous LoRA run on a family that can
# hold two versions, so the launch contract allows non-draining weight sync.
_VERSIONED_SLOTS = (
    "model.use_lora=true",
    "trainer.rollout_orchestration.schedule_mode=continuous",
    "trainer.rollout_orchestration.continuous.max_stale_policy_versions=1",
)


def real_collector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    reward: RewardFunction | None = None,
    lifecycle: RayLifecyclePlan | None = None,
    attach_runtime: bool = True,
    versioned_slots: bool = False,
    overrides: tuple[str, ...] = (),
) -> CollectorBench:
    """The real collector on the tiny SANA stack.

    ``versioned_slots`` resolves the run as continuous LoRA, so the in-process
    runtime declares non-draining weight sync for real; the default full
    fine-tune run overwrites weights in place and drains.
    """

    stack = tiny_sana_stack(
        monkeypatch,
        tmp_path,
        overrides=(_VERSIONED_SLOTS if versioned_slots else ()) + overrides,
    )
    reward = reward or IndexReward()
    trace = Trace(monkeypatch)
    trace.watch(stack.runtime, "activate", "activate")
    trace.watch(stack.runtime, "generate", "generate")
    trace.watch(stack.runtime, "update_weights", "update_weights")
    trace.watch(stack.runtime, "offload", "offload")
    trace.watch(stack.runtime, "shutdown", "runtime_shutdown")
    trace.watch(reward, "score_batch", "score")
    trace.watch(reward, "park_memory", "reward_park")
    trace.watch(reward, "shutdown", "reward_shutdown")
    collector = RolloutCollector.from_family(
        stack.family,
        reward_runtime=RewardFunctionRuntime(reward),
        config=stack.collector_config(),
        # A test passes a plan to model a shared-GPU topology; the run's own
        # plan is the tiny CPU run's disjoint one.
        lifecycle=lifecycle or stack.resolved.resources.lifecycle,
    )
    if attach_runtime:
        collector.set_generation_runtime(stack.runtime)
    return CollectorBench(stack, collector, stack.runtime, reward, trace)


async def collect_scored(
    collector: RolloutCollector,
    inputs: list[Any],
    *,
    group_size: int,
    metadata: Mapping[str, Any] | None = None,
    request_overrides: Mapping[str, Any] | None = None,
    runtime_debug: bool = False,
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
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        )
    )
    return (collector.assemble_training_batches(await collector.evaluate_rollout([unscored])))[0]
