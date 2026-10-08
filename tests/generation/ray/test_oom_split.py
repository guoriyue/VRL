"""Batch OOM degradation and dispatch routing in the Ray generation executor.

Every fleet here is real: the real launcher starts real ``RayGenerationWorker``
actors over the tiny SANA snapshot into a real placement group, and the real
executor, engines, dispatcher and finalizers drive them. A CPU worker cannot
run out of CUDA memory, so the launcher starts a subclass of the real worker
(``_fault_worker``) whose real SANA executor raises torch's own CUDA OOM error
above a sample capacity -- the worker's recovery, the typed results and the
driver's split-and-retry all run unchanged. The same subclass records which
batches each actor was asked to run and, where a theorem is about a
malformed reply, corrupts its real result.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Any

import pytest
import torch

import vrl.generation.ray.launcher as launcher_module
from tests.generation.ray._helpers import RaySanaRuntime, ray_sana_runtime
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import BatchMemoryReading, StaleSlotDiscard
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.finalizer import RayGenerationFinalizer
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.actor_pool import RayActorDispatcher
from vrl.utils.cuda_memory import is_cuda_out_of_memory
from vrl.utils.media_reference import MediaReference

# torch's allocator wire format, pinned against the real allocator by
# test_oom_matcher_accepts_the_real_torch_allocator_message below.
_OOM_PREFIX = "CUDA out of memory. Tried to allocate "
_OOM_MESSAGE = f"{_OOM_PREFIX}4.00 GiB"

# The injected OOM is torch's real error type carrying a hand-copied allocator
# message; the gpu-lane twin pins that the format is still torch's.
_OOM_WIRE_FORMAT = pytest.mark.real_cover(
    "tests/generation/ray/test_oom_split.py"
    "::test_oom_matcher_accepts_the_real_torch_allocator_message",
    why=(
        "a CPU Ray worker cannot exhaust CUDA memory, so the real worker raises torch's "
        "CUDA OOM error above a sample capacity; the gpu-lane twin pins its message format"
    ),
)

# A continuous LoRA run keeps versioned trainable-state slots.
_VERSIONED = (
    "model.use_lora=true",
    "trainer.rollout_orchestration.schedule_mode=continuous",
    "trainer.rollout_orchestration.continuous.max_stale_policy_versions=1",
)


def _key(start: int, count: int) -> str:
    """Derive a batch_key from the source template, not a hand-copied f-string."""

    return GenerationSampleBatch(prompt_index=0, sample_start=start, sample_count=count).batch_key


def _fault_worker(
    *,
    capacity: int | Mapping[str, int] | None = None,
    failure: str = _OOM_MESSAGE,
    request_path_oom: frozenset[str] = frozenset(),
    corrupt: frozenset[str] = frozenset(),
    evict_on_first_oom: bool = False,
    memory_reading: BatchMemoryReading | None = None,
) -> type[RayGenerationWorker]:
    """The real Ray generation worker with faults injected into its real executor.

    ``capacity`` (per worker id when a mapping) is the largest batch the
    executor's forward accepts; a larger one raises torch's CUDA OOM error, or
    ``ValueError(failure)`` for a non-allocator failure. Workers named in
    ``request_path_oom`` run out of memory only inside the per-request loop.
    ``evict_on_first_oom`` installs eight newer weight versions before the
    first OOM, which evicts the request's own slot from the real retention
    window. ``corrupt`` names reply fields the worker returns wrong.
    """

    class _FaultWorker(RayGenerationWorker):
        def __init__(self, worker_id: str, launch_inputs: Any) -> None:
            super().__init__(worker_id, launch_inputs)
            self.batch_calls: list[str] = []
            self.request_calls: list[str] = []
            self.request_batches: list[list[str]] = []
            self._in_request = False
            self._installed: tuple[Any, int] | None = None
            self._evicted = False

        def load_policy(self) -> None:
            super().load_policy()
            executor = self.core.executor
            if "forward_batch" in vars(executor):
                return
            real_forward = executor.forward_batch
            worker_id = self.core.worker_id
            limit = capacity.get(worker_id) if isinstance(capacity, Mapping) else capacity

            def forward_batch(request: Any, batch: GenerationSampleBatch) -> Any:
                if self._in_request and worker_id in request_path_oom:
                    raise torch.cuda.OutOfMemoryError(_OOM_MESSAGE)
                if limit is not None and batch.sample_count > limit:
                    if evict_on_first_oom and not self._evicted:
                        self._evict_installed_slot()
                    if failure == _OOM_MESSAGE:
                        raise torch.cuda.OutOfMemoryError(failure)
                    raise ValueError(failure)
                return real_forward(request, batch)

            executor.forward_batch = forward_batch

        def update_weights(self, trainable_state: Any, policy_version: int) -> int:
            self._installed = (trainable_state, policy_version)
            return super().update_weights(trainable_state, policy_version)

        def _evict_installed_slot(self) -> None:
            assert self._installed is not None
            state, version = self._installed
            for newer in range(version + 1, version + 9):
                self.core.update_weights(state, newer)
            self._evicted = True

        def execute_batch(self, envelope: Any) -> Any:
            self.batch_calls.append(envelope.batch.batch_key)
            result = super().execute_batch(envelope)
            if "request_id" in corrupt:
                result = replace(result, request_id="wrong-request")
            if "media" in corrupt and result.output is not None:
                # Boxed references must obey the same sample-count contract as tensors.
                result.output.video = [MediaReference("batch-ref", i) for i in range(2)]
            if memory_reading is not None:
                result = replace(result, memory=memory_reading)
            return result

        def execute_request_batches(self, request: Any, engine_plan: Any) -> Any:
            self.request_calls.append(request.request_id)
            self.request_batches.append([batch.batch_key for batch in engine_plan.sample_batches])
            self._in_request = True
            try:
                result = super().execute_request_batches(request, engine_plan)
            finally:
                self._in_request = False
            if "pipeline_request_id" in corrupt:
                result = replace(result, request_id="wrong-request")
            if "pipeline_worker_id" in corrupt:
                result = replace(result, worker_id="wrong-worker")
            return result

        def history(self) -> dict[str, list[Any]]:
            return {
                "batch_calls": list(self.batch_calls),
                "request_calls": list(self.request_calls),
                "request_batches": list(self.request_batches),
            }

    return _FaultWorker


class _CountingFinalizer(RayGenerationFinalizer):
    """The real finalizer, recording how many staged references each merge received."""

    def __init__(self, finalizer_id: str, gatherer: Any) -> None:
        super().__init__(finalizer_id, gatherer)
        self.merged: list[int] = []

    def merge_request(self, request: Any, batch_refs: Any) -> Any:
        self.merged.append(len(batch_refs))
        return super().merge_request(request, batch_refs)

    def merges(self) -> list[int]:
        return list(self.merged)


@contextlib.asynccontextmanager
async def _fleet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    snapshot: Any,
    worker: type[RayGenerationWorker],
    *,
    batch: int,
    engines: int = 1,
    pipelined: bool = False,
    versioned: bool = False,
) -> AsyncIterator[RaySanaRuntime]:
    """Launch the real fleet with ``worker`` as its rank actor class, activated."""

    monkeypatch.setattr(launcher_module, "RayGenerationWorker", worker)
    monkeypatch.setattr(launcher_module, "RayGenerationFinalizer", _CountingFinalizer)
    overrides = (
        f"rollout.samples_per_generation_batch={batch}",
        f"distributed.resources.rollout.num_engines={engines}",
        f"distributed.rollout.pipelined={str(pipelined).lower()}",
        *(_VERSIONED if versioned else ()),
    )
    async with ray_sana_runtime(monkeypatch, tmp_path, snapshot, overrides=overrides) as run:
        await run.runtime.activate()
        yield run


def _histories(ray: Any, run: RaySanaRuntime) -> list[dict[str, list[Any]]]:
    engines = run.runtime._session.executor.engines
    return ray.get([engine.primary.actor.history.remote() for engine in engines])


def _merges(ray: Any, run: RaySanaRuntime) -> list[int]:
    handles = run.runtime._session.finalizer_handles
    return [
        count for merged in ray.get([h.actor.merges.remote() for h in handles]) for count in merged
    ]


def _one_engine_of_every_rank(run: RaySanaRuntime) -> RayGenerationExecutor:
    """Assemble the launched single-rank engines into ONE multi-rank engine.

    A CPU fleet cannot group ranks (gloo rank groups need a GPU fleet), so each
    real actor stands for one rank of the engine the executor drives.
    """

    session = run.runtime._session
    engine = RayGenerationEngine("engine", [e.primary for e in session.executor.engines])
    return RayGenerationExecutor(
        engines=[engine],
        gatherer=run.resolved.family.new_gatherer(),
        actor_dispatcher=RayActorDispatcher(("engine",)),
        generation_stall_timeout_s=30.0,
    )


@pytest.mark.gpu
def test_oom_matcher_accepts_the_real_torch_allocator_message() -> None:
    """`_OOM_MESSAGE` is only an honest fixture while torch still emits that prefix."""

    with pytest.raises(torch.OutOfMemoryError) as caught:
        torch.empty(1024**4, dtype=torch.float32, device="cuda")

    real = str(caught.value)
    assert real.startswith(_OOM_PREFIX), real
    # The production matcher is a substring test on "out of memory"; assert it
    # against the real message, not only against our own fixture.
    assert is_cuda_out_of_memory(real) is True
    assert is_cuda_out_of_memory(_OOM_MESSAGE) is True


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_failed_gather_rejects_misaligned_media_references(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _fault_worker(corrupt=frozenset({"media"}))
    async with _fleet(monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=1) as run:
        with pytest.raises(ValueError, match="has 2 rows, expected 1"):
            await run.runtime.generate(run.request(["p"], group_size=1, reward_media_refs=True))


@_OOM_WIRE_FORMAT
@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_nonprimary_oom_retries_whole_engine_and_reports_every_rank(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _fault_worker(capacity={"rollout-0": 4, "rollout-1": 2})
    async with _fleet(monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=4, engines=2) as run:
        executor = _one_engine_of_every_rank(run)

        output = await executor.execute(run.request(["p"], group_size=4, runtime_debug=True))

        histories = _histories(local_ray, run)
        assert [h["batch_calls"] for h in histories] == [[_key(0, 4), _key(0, 2), _key(2, 2)]] * 2
        assert output.output.shape[0] == 4
        rows = output.runtime_debug["ray_chunks"]
        assert {row["worker_id"] for row in rows} == {"rollout-0", "rollout-1"}
        assert {row["batch_key"] for row in rows} == {_key(0, 2), _key(2, 2)}


@_OOM_WIRE_FORMAT
@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_oom_chunk_splits_until_it_fits(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """An 8-sample batch on a 2-sample worker degrades to four 2-sample batches."""

    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, _fault_worker(capacity=2), batch=8
    ) as run:
        output = await run.runtime.generate(run.request(["p"], group_size=8, runtime_debug=True))

        assert output.output.shape[0] == 8
        assert [row.sample_index for row in output.sample_rows] == list(range(8))
        splits = output.runtime_debug["batch_oom_splits"]
        # Recursion order: 8 -> [0:4] + [4:8] -> 2-sample leaves.
        assert [row["batch_key"] for row in splits] == [_key(0, 8), _key(0, 4), _key(4, 4)]
        assert all(row["worker_id"] == "rollout-0" for row in splits)
        (history,) = _histories(local_ray, run)
        assert sorted(history["batch_calls"][3:]) == sorted(
            [_key(0, 2), _key(2, 2), _key(4, 2), _key(6, 2)]
        )


@_OOM_WIRE_FORMAT
@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_single_sample_oom_still_raises(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A batch that OOMs at one sample is a hard failure, not an infinite loop."""

    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, _fault_worker(capacity=0), batch=4
    ) as run:
        with pytest.raises(RuntimeError, match="out of memory"):
            await run.runtime.generate(run.request(["p"], group_size=4))


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_non_oom_error_is_not_retried(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Only allocator failures degrade; other worker errors fail fast."""

    worker = _fault_worker(capacity=2, failure="bad scheduler state")
    async with _fleet(monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=4) as run:
        with pytest.raises(RuntimeError, match="bad scheduler state"):
            await run.runtime.generate(run.request(["p"], group_size=4))

        (history,) = _histories(local_ray, run)
        assert history["batch_calls"] == [_key(0, 4)]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_healthy_chunks_skip_degradation_path(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """No OOM: results and telemetry are exactly the pre-split behavior."""

    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, _fault_worker(capacity=2), batch=2
    ) as run:
        output = await run.runtime.generate(run.request(["p"], group_size=4))

        assert output.output.shape[0] == 4
        assert output.runtime_debug is None
        (history,) = _histories(local_ray, run)
        assert history["batch_calls"] == [_key(0, 2), _key(2, 2)]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_result_request_id_must_match_submitted_envelope(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _fault_worker(corrupt=frozenset({"request_id"}))
    async with _fleet(monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=2) as run:
        with pytest.raises(RuntimeError, match="request_id mismatch"):
            await run.runtime.generate(run.request(["p"], group_size=2))

        (history,) = _histories(local_ray, run)
        assert history["batch_calls"] == [_key(0, 2)]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_stale_slot_routes_to_graceful_discard_not_failure(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A request whose version slot the worker does not hold raises StaleSlotDiscard (a
    typed discard), NOT a generic RuntimeError, so the producer counts it as a stale
    discard, not a collect error. It skips OOM-split retries entirely."""

    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, _fault_worker(), batch=2, versioned=True
    ) as run:
        await run.runtime.update_weights(run.trainable_state(), 1)

        with pytest.raises(StaleSlotDiscard, match="policy_version=7"):
            await run.runtime.generate(run.request(["p"], group_size=2, policy_version=7))

        # Routed before scheduling any OOM retry, so the batch ran exactly once.
        (history,) = _histories(local_ray, run)
        assert history["batch_calls"] == [_key(0, 2)]


def test_stale_slot_discard_is_not_runtime_error() -> None:
    """StaleSlotDiscard must be its OWN type, not a RuntimeError subclass — the
    producer's generic ``except Exception`` (error_count) must not catch it first."""

    assert not issubclass(StaleSlotDiscard, RuntimeError)
    assert issubclass(StaleSlotDiscard, Exception)


@_OOM_WIRE_FORMAT
@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_slot_evicted_during_oom_retry_discards_request(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Eight newer installs land between the OOM and its retry; the request's own
    version leaves the retention window, so the retry is a typed discard."""

    worker = _fault_worker(capacity=2, evict_on_first_oom=True)
    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=4, versioned=True
    ) as run:
        await run.runtime.update_weights(run.trainable_state(), 1)

        with pytest.raises(StaleSlotDiscard, match="policy_version=1"):
            await run.runtime.generate(run.request(["p"], group_size=4, policy_version=1))

        (history,) = _histories(local_ray, run)
        assert history["batch_calls"] == [_key(0, 4), _key(0, 2), _key(2, 2)]


def test_is_oom_error_classifier() -> None:
    assert is_cuda_out_of_memory(_OOM_MESSAGE)
    assert is_cuda_out_of_memory("torch.OutOfMemoryError: HIP out of memory")
    assert not is_cuda_out_of_memory("ValueError: shape mismatch")


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_routes_single_worker_to_per_request_path(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """pipelined=True + one engine => the whole request runs via the per-request
    path (execute_request_batches) and its staged references are merged once by
    the finalizer, NOT per-batch dispatch and NOT a driver-side gather."""

    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, _fault_worker(), batch=2, pipelined=True
    ) as run:
        output = await run.runtime.generate(run.request(["p"], group_size=4))

        (history,) = _histories(local_ray, run)
        assert len(history["request_calls"]) == 1
        assert history["request_batches"] == [[_key(0, 2), _key(2, 2)]]
        assert history["batch_calls"] == []
        assert _merges(local_ray, run) == [2]
        assert output.output.shape[0] == 4
        assert [row.sample_index for row in output.sample_rows] == [0, 1, 2, 3]


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_splits_batches_over_engines_and_merges_once(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Every engine runs its round-robin share in one call; one finalizer merges
    all the references in plan order regardless of which engine staged them."""

    async with _fleet(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        _fault_worker(),
        batch=2,
        engines=2,
        pipelined=True,
    ) as run:
        output = await run.runtime.generate(run.request(["p"], group_size=8))

        histories = _histories(local_ray, run)
        assert histories[0]["request_batches"] == [[_key(0, 2), _key(4, 2)]]
        assert histories[1]["request_batches"] == [[_key(2, 2), _key(6, 2)]]
        assert all(history["batch_calls"] == [] for history in histories)
        assert _merges(local_ray, run) == [4]
        assert [row.sample_index for row in output.sample_rows] == list(range(8))


@_OOM_WIRE_FORMAT
@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_oom_on_one_engine_retries_the_request_per_batch(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _fault_worker(request_path_oom=frozenset({"rollout-1"}))
    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=2, engines=2, pipelined=True
    ) as run:
        output = await run.runtime.generate(run.request(["p"], group_size=8))

        histories = _histories(local_ray, run)
        assert [len(history["request_calls"]) for history in histories] == [1, 1]
        assert _merges(local_ray, run) == []
        assert sorted(histories[0]["batch_calls"] + histories[1]["batch_calls"]) == sorted(
            _key(start, 2) for start in (0, 2, 4, 6)
        )
        assert output.output.shape[0] == 8


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_stale_slot_is_a_graceful_discard_that_frees_the_engine(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _fleet(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        _fault_worker(),
        batch=2,
        pipelined=True,
        versioned=True,
    ) as run:
        await run.runtime.update_weights(run.trainable_state(), 1)

        with pytest.raises(StaleSlotDiscard, match="slot evicted"):
            await run.runtime.generate(run.request(["p"], group_size=4, policy_version=7))

        output = await run.runtime.generate(run.request(["p"], group_size=4, policy_version=1))
        assert output.output.shape[0] == 4
        (history,) = _histories(local_ray, run)
        assert len(history["request_calls"]) == 2


@_OOM_WIRE_FORMAT
@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_uses_per_chunk_path_for_one_chunk(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """A one-batch request has nothing to overlap and keeps OOM admission."""

    async with _fleet(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        _fault_worker(capacity=2),
        batch=4,
        pipelined=True,
    ) as run:
        output = await run.runtime.generate(run.request(["p"], group_size=4))

        (history,) = _histories(local_ray, run)
        assert history["request_calls"] == []
        assert history["batch_calls"] == [_key(0, 4), _key(0, 2), _key(2, 2)]
        assert output.output.shape[0] == 4


@_OOM_WIRE_FORMAT
@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_oom_retries_through_per_chunk_split_admission(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    async with _fleet(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        _fault_worker(capacity=2),
        batch=4,
        pipelined=True,
    ) as run:
        output = await run.runtime.generate(run.request(["p"], group_size=8))

        (history,) = _histories(local_ray, run)
        assert len(history["request_calls"]) == 1
        assert history["batch_calls"][:2] == [_key(0, 4), _key(4, 4)]
        assert sorted(history["batch_calls"][2:]) == sorted(
            _key(start, 2) for start in (0, 2, 4, 6)
        )
        assert _merges(local_ray, run) == []
        assert output.output.shape[0] == 8


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_result_request_id_must_match_request(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _fault_worker(corrupt=frozenset({"pipeline_request_id"}))
    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=2, pipelined=True
    ) as run:
        with pytest.raises(RuntimeError, match="request_id mismatch"):
            await run.runtime.generate(run.request(["p"], group_size=4))

        (history,) = _histories(local_ray, run)
        assert len(history["request_calls"]) == 1
        assert history["batch_calls"] == []


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_pipelined_oom_worker_id_must_match_actor(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    worker = _fault_worker(
        request_path_oom=frozenset({"rollout-0"}),
        corrupt=frozenset({"pipeline_worker_id"}),
    )
    async with _fleet(
        monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=2, pipelined=True
    ) as run:
        with pytest.raises(RuntimeError, match="rank mismatch"):
            await run.runtime.generate(run.request(["p"], group_size=4))

        (history,) = _histories(local_ray, run)
        assert len(history["request_calls"]) == 1
        assert history["batch_calls"] == []


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_default_uses_per_chunk_path(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
) -> None:
    """Default (pipelined=False) is the unchanged per-batch dispatch."""

    async with _fleet(monkeypatch, tmp_path, ray_sana_snapshot, _fault_worker(), batch=4) as run:
        await run.runtime.generate(run.request(["p"], group_size=4))

        (history,) = _histories(local_ray, run)
        assert history["request_calls"] == []
        assert history["batch_calls"] == [_key(0, 4)]


@pytest.mark.slow_test
@pytest.mark.asyncio
@pytest.mark.parametrize("with_reading", [True, False])
async def test_executor_logs_measured_batch_memory(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, caplog, with_reading
):
    # A CPU worker measures no CUDA memory; the reading a CUDA worker reports
    # is attached to its real result.
    mib = 2**20
    reading = (
        BatchMemoryReading(
            sample_count=1,
            baseline_allocated_bytes=10 * mib,
            denoise_peak_bytes=18 * mib,
            decode_peak_bytes=14 * mib,
            reserved_start_bytes=11 * mib,
            free_start_bytes=18 * mib,
            total_bytes=32 * mib,
        )
        if with_reading
        else None
    )
    worker = _fault_worker(memory_reading=reading)
    async with _fleet(monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=1) as run:
        with caplog.at_level("INFO", logger="vrl.generation.ray.executor"):
            await run.runtime.generate(run.request(["p"], group_size=1))

    messages = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("batch memory:")
    ]
    assert messages == (
        [
            f"batch memory: batch={_key(0, 1)} n=1 peak=18MB "
            "(denoise=18MB decode=14MB baseline=10MB) budget=29MB non_torch=3MB"
        ]
        if with_reading
        else []
    )


@pytest.mark.parametrize(
    "message, expected",
    [
        ("CUDA out of memory", True),
        ("HIP out of memory", True),
        ("CPU out of memory", False),
        ("shape mismatch", False),
    ],
)
def test_local_and_remote_oom_classification_agree(message, expected):
    assert is_cuda_out_of_memory(message) is expected
    assert is_cuda_out_of_memory(RuntimeError(message)) is expected


@pytest.mark.slow_test
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [_OOM_MESSAGE, "decode failed"])
async def test_real_multirank_nonprimary_failure_reaches_driver(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path, failure
):
    worker = _fault_worker(capacity={"rollout-0": 8, "rollout-1": 2}, failure=failure)
    async with _fleet(monkeypatch, tmp_path, ray_sana_snapshot, worker, batch=8, engines=2) as run:
        executor = _one_engine_of_every_rank(run)
        request = run.request(["p"], group_size=8, runtime_debug=True)

        if failure == _OOM_MESSAGE:
            output = await executor.execute(request)
            assert output.output.shape[0] == 8
            assert all(
                row["worker_id"] == "rollout-1" for row in output.runtime_debug["batch_oom_splits"]
            )
            histories = _histories(local_ray, run)
            assert histories[0]["batch_calls"] == histories[1]["batch_calls"]
            assert len(histories[0]["batch_calls"]) == 7
        else:
            with pytest.raises(RuntimeError, match="decode failed"):
                await executor.execute(request)
            assert [h["batch_calls"] for h in _histories(local_ray, run)] == [[_key(0, 8)]] * 2
