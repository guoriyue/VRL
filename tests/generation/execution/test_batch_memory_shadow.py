"""Batch memory readings + startup batch-size probe (SPRINT_chunk_size_probe).

Covers the three seams: the pure affine-fit math, the worker-side probe
contract (trials -> fit -> confirm/bisect -> knee, against the CONTRACT
budget), and the wire plumbing (readings cross ungated; the Ray runtime
resolves ``samples_per_generation_batch: auto`` once and rewrites requests).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vrl.generation.execution.worker as worker_module
from tests.generation.execution._helpers import launch_contract
from tests.generation.ray._helpers import FakeRayActor
from vrl.generation.execution.batch_memory import (
    AffinePeakFit,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import (
    BatchMemoryReading,
    BatchProduceFence,
    BatchSizeProbeResult,
    BatchSizeProbeTrial,
    GenerationBatchEnvelope,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.types import GenerationRequest
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher

GB = 1024**3


@pytest.mark.parametrize("value", [True, 1.5, float("nan")])
@pytest.mark.parametrize(
    "field", ["completed_batches", "n", "samples_per_generation_batch", "budget_bytes"]
)
def test_execution_counts_reject_nonintegers(field, value):
    with pytest.raises(ValueError, match=field):
        if field == "completed_batches":
            BatchProduceFence(completed_batches=value, event=None)
        elif field == "n":
            BatchSizeProbeTrial(n=value, oom=True, label="probe")
        else:
            kwargs = {"samples_per_generation_batch": 1, "budget_bytes": 0, "trials": ()}
            kwargs[field] = value
            BatchSizeProbeResult(**kwargs)


# Carried by every `fake_cuda` consumer: they assert exact byte arithmetic, and
# real hardware cannot pin the inputs to it (see the fixture's docstring below).
_EXACT_BYTES_NEED_A_FIXED_CARD = pytest.mark.real_cover(
    None,
    why=(
        "the probe's budget arithmetic is asserted to the byte (a 10GB + 2GB*n peak model, "
        "non_torch = (32-24) - 8 = 0), and torch.cuda.mem_get_info varies with the machine "
        "and its current load; a real GPU would turn a deterministic arithmetic assertion "
        "into an unreproducible approximation"
    ),
    tracked_in="docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md",
)


def _runtime(executor: RayGenerationExecutor) -> RayGenerationRuntime:
    return RayGenerationRuntime(
        session=RayGenerationSession(executor, None, []),
    )


def _reading(
    *,
    sample_count: int = 4,
    baseline: int = 10 * GB,
    denoise_peak: int = 18 * GB,
    decode_peak: int = 14 * GB,
    reserved_start: int = 11 * GB,
    free_start: int = 18 * GB,
    total: int = 32 * GB,
) -> BatchMemoryReading:
    return BatchMemoryReading(
        sample_count=sample_count,
        baseline_allocated_bytes=baseline,
        denoise_peak_bytes=denoise_peak,
        decode_peak_bytes=decode_peak,
        reserved_start_bytes=reserved_start,
        free_start_bytes=free_start,
        total_bytes=total,
    )


# -- affine fit ---------------------------------------------------------------


def test_affine_fit_recovers_slope_intercept_and_budget_division() -> None:
    fit = AffinePeakFit.from_trials(1, 12 * GB, 4, 18 * GB)

    assert fit.slope_bytes_per_sample == 2 * GB
    assert fit.intercept_bytes == 10 * GB
    # (32 - 10) // 2 = 11 samples fit the full card.
    assert fit.max_samples_within(32 * GB, max_samples=64) == 11
    # Not even the intercept fits.
    assert fit.max_samples_within(9 * GB, max_samples=64) == 0
    assert fit.max_samples_within(32 * GB, max_samples=4) == 4


@pytest.mark.parametrize("ceiling", [1, 16, 2**21])
@pytest.mark.parametrize("high_peak", [12 * GB, 11 * GB])
def test_non_growing_fit_uses_actual_request_ceiling(ceiling, high_peak) -> None:
    fit = AffinePeakFit.from_trials(1, 12 * GB, 4, high_peak)
    assert fit.max_samples_within(32 * GB, max_samples=ceiling) == ceiling
    assert fit.max_samples_within(9 * GB, max_samples=ceiling) == 0


def test_affine_fit_rejects_invalid_request_ceiling() -> None:
    fit = AffinePeakFit.from_trials(1, 12 * GB, 4, 18 * GB)
    with pytest.raises(ValueError, match="max_samples"):
        fit.max_samples_within(32 * GB, max_samples=0)


def test_affine_fit_rejects_degenerate_points() -> None:
    with pytest.raises(ValueError, match="two distinct n"):
        AffinePeakFit.from_trials(4, 18 * GB, 4, 18 * GB)


# -- reading + shadow rows ----------------------------------------------------


def test_reading_normalizes_binding_mapping_and_rejects_partial_data() -> None:
    reading = _reading()

    assert BatchMemoryReading.from_metrics(asdict(reading)) == reading
    partial = asdict(reading)
    del partial["decode_peak_bytes"]
    assert BatchMemoryReading.from_metrics(partial) is None
    # non-torch = device-used minus torch-reserved: (32-18) - 11 = 3GB.
    assert reading.non_torch_bytes == 3 * GB
    assert reading.budget_bytes == 29 * GB


# -- worker probe -------------------------------------------------------------


class _ProbeExecutor:
    """Diffusion-stage fake: peak(n) = 10GB + 2GB*n, OOM above ``oom_limit``."""

    family = "sd3_5"
    task = "t2i"

    def __init__(self, oom_limit: int | None = None) -> None:
        self.model = SimpleNamespace(device="cpu")
        self.oom_limit = oom_limit
        self.executed_steps: list[int | None] = []

    def forward_probe_batch(
        self,
        request: Any,
        batch: GenerationSampleBatch,
        *,
        execute_steps: int,
    ) -> Any:
        del request
        n = batch.sample_count
        if self.oom_limit is not None and n > self.oom_limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        self.executed_steps.append(execute_steps)
        return SimpleNamespace(
            memory=asdict(
                _reading(
                    sample_count=n,
                    denoise_peak=10 * GB + 2 * GB * n,
                    decode_peak=10 * GB + 1 * GB * n,
                    reserved_start=8 * GB,
                    free_start=24 * GB,  # non_torch = (32-24) - 8 = 0
                ),
            ),
        )

    def merge_generation_batches(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _probe_core(executor: Any) -> GenerationWorkerCore:
    contract = launch_contract(policy_version=1)
    core = GenerationWorkerCore("rollout-0", contract, executor)
    core.executor = executor
    return core


def _request(samples_per_prompt: int = 10) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-1",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=samples_per_prompt,
        sampling={"num_steps": 20},
        samples_per_generation_batch="auto",
        policy_version=1,
    )


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed 24GB-free/32GB-total card. Kept as a fake on purpose: the probe's
    budget arithmetic asserts exact byte values, which no real GPU can pin
    (mem_get_info is machine- and load-dependent)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # The probe tests' arithmetic assumes no safety margin and a knee that
    # CANNOT fire: the knee compares wall-clock per-sample times, and on a
    # loaded machine the micro-trials jitter by more than any finite factor —
    # -inf is the only load-proof "disabled". The knee test re-pins its own
    # threshold, because there the knee IS the theorem.
    monkeypatch.setattr(worker_module, "_PROBE_MEMORY_MARGIN", 0.0)
    monkeypatch.setattr(worker_module, "_PROBE_KNEE_THRESHOLD", float("-inf"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (24 * GB, 32 * GB))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@_EXACT_BYTES_NEED_A_FIXED_CARD
def test_probe_fits_confirms_and_truncates_steps(fake_cuda: None) -> None:
    executor = _ProbeExecutor()
    core = _probe_core(executor)

    # fit from (1, 12GB) and (4, 18GB) -> slope 2GB, intercept 10GB;
    # budget 32GB, margin 0 -> 11 fit, capped by samples_per_prompt=10.
    result = core.probe_batch_size(
        _request(),
        max_samples=10,
    )

    assert result.samples_per_generation_batch == 10
    assert [trial.label for trial in result.trials] == [
        "warmup",
        "fit-low",
        "fit-high",
        "confirm",
    ]
    assert not any(trial.oom for trial in result.trials)
    # Every executed trial ran truncated (2 steps), never the full schedule.
    assert set(executor.executed_steps) == {2}


@_EXACT_BYTES_NEED_A_FIXED_CARD
def test_probe_knee_refuses_growth_without_throughput_gain(
    fake_cuda: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _probe_core(_ProbeExecutor())

    # A 2.0 throughput-gain bar can never be met -> settle at the fit anchor n=4.
    monkeypatch.setattr(worker_module, "_PROBE_KNEE_THRESHOLD", 2.0)
    result = core.probe_batch_size(
        _request(),
        max_samples=10,
    )

    assert result.samples_per_generation_batch == 4


@_EXACT_BYTES_NEED_A_FIXED_CARD
def test_probe_bisects_when_confirm_ooms(fake_cuda: None) -> None:
    core = _probe_core(_ProbeExecutor(oom_limit=6))

    result = core.probe_batch_size(
        _request(),
        max_samples=10,
    )

    # candidate 10 OOMs; bisection between known-good 4 and 10 lands on 6.
    assert result.samples_per_generation_batch == 6
    assert any(trial.oom for trial in result.trials)


@_EXACT_BYTES_NEED_A_FIXED_CARD
def test_probe_budgets_against_whole_phase_gpu(fake_cuda: None) -> None:
    core = _probe_core(_ProbeExecutor())

    # Shared roles hand the GPU over before this probe, so rollout owns the whole
    # 32GB device for the phase rather than a persistent fractional share.
    result = core.probe_batch_size(
        _request(),
        max_samples=10,
    )

    assert result.samples_per_generation_batch == 10
    assert result.budget_bytes == 32 * GB


def test_probe_requires_cuda_and_single_sample_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    core = _probe_core(_ProbeExecutor())
    with pytest.raises(RuntimeError, match="requires CUDA"):
        core.probe_batch_size(_request(), max_samples=4)


@_EXACT_BYTES_NEED_A_FIXED_CARD
def test_probe_fails_loud_when_one_sample_ooms(fake_cuda: None) -> None:
    core = _probe_core(_ProbeExecutor(oom_limit=0))
    with pytest.raises(RuntimeError, match="single sample does not fit"):
        core.probe_batch_size(_request(), max_samples=4)


# -- runtime auto resolution --------------------------------------------------


def _probe_worker(
    worker_id: str,
    answer: int,
    calls: list[str],
) -> RayGenerationEngine:
    """Build a probe engine wearing the Ray actor method face.

    This covers result ordering and cache reuse without pretending to exercise
    Ray serialization or ObjectRef deadlines.
    """

    def probe(request: Any, *, max_samples: int) -> BatchSizeProbeResult:
        calls.append(worker_id)
        return BatchSizeProbeResult(
            samples_per_generation_batch=answer,
            budget_bytes=32 * GB,
            trials=(),
        )

    return RayGenerationEngine(
        worker_id,
        [
            RayActorHandle(
                worker_id=worker_id,
                actor=FakeRayActor(
                    SimpleNamespace(probe_batch_size=probe),
                    "probe_batch_size",
                ),
            ),
        ],
    )


@pytest.mark.real_cover(
    "tests/generation/ray/test_runtime_config.py"
    "::test_real_ray_probe_fan_out_resolves_auto_once_across_the_fleet",
    why=(
        "the executor's local-callable branch cannot exercise remote fleet fan-out, shared "
        "actor admission, the generation-stall ObjectRef deadline, or GenerationRequest "
        "serialization; the slow_test twin drives all four on a live cluster"
    ),
)
def test_runtime_resolves_auto_once_and_rewrites_requests() -> None:
    """Fleet answer is the min across workers, resolved once and reused: the
    second request is rewritten from the cache without a second probe."""

    calls: list[str] = []
    executed: list[Any] = []

    workers = [
        _probe_worker("w0", 6, calls),
        _probe_worker("w1", 4, calls),
    ]
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        workers,
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("w0", "w1")),
        generation_stall_timeout_s=30.0,
    )

    async def execute(request: Any) -> Any:
        executed.append(request)
        return SimpleNamespace(request_id=request.request_id)

    executor.execute = execute
    runtime = _runtime(executor)

    async def go() -> None:
        await runtime.generate(_request())
        await runtime.generate(_request())

    asyncio.run(go())

    # Fleet answer = min across workers; probed once, cached for request 2.
    assert calls == ["w0", "w1"]
    assert [req.samples_per_generation_batch for req in executed] == [4, 4]


@pytest.mark.parametrize("engine_ids", ["w0", b"w0", [1], [b"worker-0"], [["worker-0"]]])
def test_planner_rejects_invalid_engine_id_types_before_batch_planning(engine_ids) -> None:
    from vrl.generation.execution.batch_placement import DistributedExecutionPlanner

    # The unresolved auto width would fail later: identity errors belong first.
    with pytest.raises(ValueError, match="engine IDs"):
        DistributedExecutionPlanner().plan_with_engine(_request(), engine_ids)


def test_planner_rejects_unresolved_auto() -> None:
    from vrl.generation.execution.batch_placement import DistributedExecutionPlanner

    planner = DistributedExecutionPlanner()
    with pytest.raises(ValueError, match="samples_per_generation_batch: auto requires"):
        planner.plan_with_engine(
            _request(),
            ["w0"],
        )


# -- worker wire contract (readings cross ungated) -----------------------------


class _MemoryExecutor:
    family = "sd3_5"
    task = "t2i"

    def __init__(self) -> None:
        self.model = SimpleNamespace(device="cpu")

    def forward_batch(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(memory=asdict(_reading()))

    def merge_generation_batches(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_worker_forwards_batch_memory_without_runtime_debug() -> None:
    contract = launch_contract(policy_version=1)
    executor = _MemoryExecutor()
    core = GenerationWorkerCore("rollout-0", contract, executor)
    core.executor = executor
    request = GenerationRequest(
        request_id="req-1",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=1,
        policy_version=1,
    )
    envelope = GenerationBatchEnvelope(
        request=request,
        batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1),
    )

    result = core.execute_batch(envelope)

    assert result.error is None
    assert result.memory == _reading()
    assert result.output.memory is None
    assert "batch_memory" not in result.metrics
    assert "engine_counters" not in result.metrics


def test_cuda_occupancy_is_absent_only_without_cuda(monkeypatch) -> None:
    import torch

    from vrl.generation.execution.batch_memory import cuda_occupancy_snapshot

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cuda_occupancy_snapshot() is None


@pytest.mark.parametrize("operation", ["mem_get_info", "memory_allocated", "memory_reserved"])
def test_cuda_occupancy_preserves_query_errors(monkeypatch, operation) -> None:
    import torch

    from vrl.generation.execution.batch_memory import cuda_occupancy_snapshot

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8, 16))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 4)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 6)
    failure = RuntimeError("CUDA query failed")

    def fail():
        raise failure

    monkeypatch.setattr(torch.cuda, operation, fail)
    with pytest.raises(RuntimeError, match="CUDA query failed") as caught:
        cuda_occupancy_snapshot()
    assert caught.value is failure


@pytest.mark.parametrize("failure_stage", ["forward", "synchronize"])
def test_probe_releases_failed_trial_tensors_before_cache_cleanup(
    fake_cuda, monkeypatch, failure_stage
):
    import weakref

    retained = []
    sync_failure_pending = False

    class FailingProbe(_ProbeExecutor):
        def forward_probe_batch(self, request, batch, *, execute_steps):
            nonlocal sync_failure_pending
            result = super().forward_probe_batch(request, batch, execute_steps=execute_steps)
            if batch.sample_count > 6:
                temporary = torch.ones(2)
                retained.append(weakref.ref(temporary))
                if failure_stage == "forward":
                    raise RuntimeError("CUDA out of memory")
                result.temporary = temporary
                sync_failure_pending = True
            return result

    def synchronize():
        nonlocal sync_failure_pending
        if sync_failure_pending:
            sync_failure_pending = False
            raise RuntimeError("CUDA out of memory")

    def empty_cache():
        assert all(ref() is None for ref in retained)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    result = _probe_core(FailingProbe()).probe_batch_size(_request(), max_samples=10)
    assert retained
    assert result.samples_per_generation_batch == 6


def test_multirank_probe_rejected_before_any_engine_dispatch():
    calls = []
    first = _probe_worker("single", 4, calls)
    rank0 = _probe_worker("rank0", 4, calls).primary
    rank1 = _probe_worker("rank1", 6, calls).primary
    multi = RayGenerationEngine("multi", [rank0, rank1])
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        [first, multi],
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("single", "multi")),
        generation_stall_timeout_s=30.0,
    )
    with pytest.raises(ValueError, match="single-rank engines"):
        asyncio.run(executor.probe_batch_sizes(_request(), max_samples=8))
    assert calls == []


def test_direct_multirank_probe_rejected_before_cuda_or_model_work(monkeypatch):
    from vrl.generation.execution.rank_group import RankGroupSpec

    core = _probe_core(_ProbeExecutor())
    core.rank_group_spec = RankGroupSpec("127.0.0.1", 29500, 0, 2)

    def unexpected():
        raise AssertionError("unsupported probe must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    with pytest.raises(ValueError, match="single-rank engine"):
        core.probe_batch_size(_request(), max_samples=8)
