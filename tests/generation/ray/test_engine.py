"""Multi-rank fan-out/aggregate semantics of the driver-side engine.

Today every engine owns one rank; these tests drive RayGenerationEngine over
recording fake rank actors at N=2/3 so the coordination layer is pinned before
a multi-GPU engine backend exists: broadcast order, rank-0 result selection,
uniform-echo validation, fail-closed on any rank failure (with sibling
cancellation), and parking-snapshot aggregation.
"""

from __future__ import annotations

from typing import Any

import pytest

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.generation.protocols import GenerationRankActor
from vrl.generation.ray.engine import (
    EngineCallRef,
    RayGenerationEngine,
    rank_handles,
    uniform_rank_result,
)
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.actor_group import RayActorHandle


class _Ref:
    def __init__(self, value: Any = None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error

    def __await__(self):
        async def _resolve() -> Any:
            if self.error is not None:
                raise self.error
            return self.value

        return _resolve().__await__()


class _Method:
    """Recording remote method: returns scripted refs in submission order."""

    def __init__(self, calls: list[tuple[str, tuple, dict]], rank_id: str, ref: _Ref) -> None:
        self._calls = calls
        self._rank_id = rank_id
        self._ref = ref

    def remote(self, *args: Any, **kwargs: Any) -> _Ref:
        self._calls.append((self._rank_id, args, kwargs))
        return self._ref


def _engine(
    calls: list[tuple[str, tuple, dict]],
    refs: dict[str, _Ref],
    *,
    method: str = "execute_batch",
) -> RayGenerationEngine:
    ranks = []
    for rank_id, ref in refs.items():
        actor = type("_Actor", (), {method: _Method(calls, rank_id, ref)})()
        ranks.append(RayActorHandle(worker_id=rank_id, actor=actor))
    return RayGenerationEngine("engine-0", ranks)


@pytest.mark.asyncio
async def test_broadcast_submits_to_every_rank_in_order_and_returns_rank0() -> None:
    calls: list[tuple[str, tuple, dict]] = []
    engine = _engine(
        calls,
        {"r0": _Ref("rank0-result"), "r1": _Ref("rank1-result"), "r2": _Ref("rank2-result")},
    )

    ref = engine.remote("execute_batch")("payload", flag=True)
    assert isinstance(ref, EngineCallRef)
    result = await ref

    assert result == "rank0-result"
    assert calls == [
        ("r0", ("payload",), {"flag": True}),
        ("r1", ("payload",), {"flag": True}),
        ("r2", ("payload",), {"flag": True}),
    ]


@pytest.mark.asyncio
async def test_single_rank_returns_the_raw_rank_ref() -> None:
    calls: list[tuple[str, tuple, dict]] = []
    raw = _Ref("only")
    engine = _engine(calls, {"r0": raw})

    ref = engine.remote("execute_batch")("payload")

    # No aggregate wrapper for the degenerate case: identical to pre-engine
    # behavior byte-for-byte (completion/cancellation timing included).
    assert ref is raw


@pytest.mark.asyncio
async def test_any_rank_failure_fails_the_engine_call_and_cancels_siblings() -> None:
    calls: list[tuple[str, tuple, dict]] = []
    cancelled: list[Any] = []
    import vrl.generation.ray.engine as engine_module

    boom = RuntimeError("rank r1 died")
    engine = _engine(calls, {"r0": _Ref("ok"), "r1": _Ref(error=boom), "r2": _Ref("ok")})

    def _record_cancel(_ray: Any, refs: Any, *, root_error: Any) -> tuple:
        cancelled.extend(refs)
        return ()

    original = engine_module.cancel_ray_refs
    engine_module.cancel_ray_refs = _record_cancel
    try:
        with pytest.raises(RuntimeError, match="rank r1 died"):
            await engine.remote("execute_batch")("payload")
    finally:
        engine_module.cancel_ray_refs = original

    # Waiting only on rank 0 would have HUNG here once ranks run collectives;
    # the aggregate surfaces the failure and cancels every sibling ref.
    assert len(cancelled) == 3


@pytest.mark.asyncio
async def test_uniform_combine_requires_every_rank_to_agree() -> None:
    calls: list[tuple[str, tuple, dict]] = []
    agree = _engine(calls, {"r0": _Ref(7), "r1": _Ref(7)}, method="update_weights")
    result = await agree.remote(
        "update_weights",
        combine=uniform_rank_result("update_weights"),
    )("state", policy_version=7)
    assert result == 7

    disagree = _engine(calls, {"r0": _Ref(7), "r1": _Ref(6)}, method="update_weights")
    with pytest.raises(RuntimeError, match="ranks disagree on update_weights"):
        await disagree.remote(
            "update_weights",
            combine=uniform_rank_result("update_weights"),
        )("state", policy_version=7)


@pytest.mark.asyncio
async def test_sleep_validates_and_aggregates_per_rank_snapshots() -> None:
    def snapshot(rank_id: str) -> WorkerMemoryParkingSnapshot:
        return WorkerMemoryParkingSnapshot(
            worker_id=rank_id,
            backend="cumem",
            baseline_gpu_used_bytes=0,
            loaded_gpu_used_bytes=1,
            residual_gpu_used_bytes=0,
            residual_bytes_limit=1,
        )

    calls: list[tuple[str, tuple, dict]] = []
    engine = _engine(
        calls,
        {"r0": _Ref(snapshot("r0")), "r1": _Ref(snapshot("r1"))},
        method="sleep",
    )
    snapshots = await engine.sleep()
    assert [snap.worker_id for snap in snapshots] == ["r0", "r1"]

    mismatched = _engine(
        calls,
        {"r0": _Ref(snapshot("someone-else"))},
        method="sleep",
    )
    with pytest.raises(RuntimeError, match="mismatched rank memory-parking report"):
        await mismatched.sleep()


def test_engine_requires_at_least_one_unique_rank() -> None:
    with pytest.raises(ValueError, match="at least one rank"):
        RayGenerationEngine("engine-0", [])
    handle = RayActorHandle(worker_id="r0", actor=object())
    with pytest.raises(ValueError, match="duplicate rank ids"):
        RayGenerationEngine("engine-0", [handle, handle])


def test_rank_handles_flattens_in_engine_order() -> None:
    first = RayActorHandle(worker_id="a", actor=object())
    second = RayActorHandle(worker_id="b", actor=object())
    engines = [
        RayGenerationEngine("e0", [first]),
        RayGenerationEngine("e1", [second]),
    ]
    assert rank_handles(engines) == [first, second]


def test_rank_actor_satisfies_the_rank_protocol() -> None:
    """The engine's method-name strings cannot drift from the actor surface."""

    missing = [
        name
        for name in dir(GenerationRankActor)
        if not name.startswith("_") and not hasattr(RayGenerationWorker, name)
    ]
    assert missing == []
