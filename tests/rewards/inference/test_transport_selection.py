"""Rewards score in-process; the removed pool transport fails loud."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.pickscore import PickScoreReward
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.runtime import LocalRewardRuntime, make_reward_runtime

_FACTORY = "vrl.rewards.models.pickscore:pickscore_reward_model"


def _request() -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0", path="/tmp/a0.mp4", media_type="video", prompt="p",
            ),
        ),
        reward_name="fake",
        score_key="overall",
    )


def test_make_reward_runtime_inline() -> None:
    """Checks make reward runtime inline -> in-process transport."""
    runtime = make_reward_runtime(
        "inline", model_factory=_FACTORY, worker_config={"device": "cpu"},
    )
    assert isinstance(runtime, LocalRewardRuntime)


def test_make_reward_runtime_rejects_removed_pool() -> None:
    """Checks the removed pool transport fails loud with the migration hint."""
    with pytest.raises(ValueError, match="sleep_offload"):
        make_reward_runtime("pool", model_factory=_FACTORY)


def test_make_reward_runtime_rejects_unknown() -> None:
    """Checks make reward runtime rejects unknown."""
    with pytest.raises(ValueError, match="execution"):
        make_reward_runtime("bogus", model_factory=_FACTORY)


def test_reward_class_uses_inline_transport() -> None:
    """Checks reward class builds the in-process transport."""
    reward = PickScoreReward(device="cpu")
    assert isinstance(reward.runtime, LocalRewardRuntime)


class _FakeCumemAllocator:
    """Records pool/sleep/wake calls (the cumem contract, no CUDA needed).

    Kept as a fake on purpose: the real allocator needs vLLM installed plus a
    CUDA context; a memory-effect twin belongs in a vLLM-equipped GPU lane
    when one exists (the allocator-missing branch is already tested for real).
    """

    def __init__(self) -> None:
        self.pool_tags: list[str] = []
        self.sleeps: list[tuple[str, ...]] = []
        self.wakes: list[list[str]] = []

    def use_memory_pool(self, *, tag: str):
        import contextlib

        self.pool_tags.append(tag)
        return contextlib.nullcontext()

    def sleep(self, *, offload_tags) -> None:
        self.sleeps.append(tuple(offload_tags))

    def wake_up(self, *, tags) -> None:
        self.wakes.append(list(tags))


def _immovable_factory(worker_config):
    """Module-level factory: a reward model WITHOUT .to() (cumem needs none)."""

    class _Immovable:
        def __call__(self, *, artifact, request):
            return {"overall": 2.0}

    return _Immovable()


@pytest.mark.asyncio
async def test_sleep_offload_uses_cumem_pool(monkeypatch) -> None:
    """cumem path: model built inside the pool, per-runtime tag, no .to() needed."""
    import vrl.utils.cuda_memory as cuda_memory_mod

    allocator = _FakeCumemAllocator()
    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: allocator)

    factory = f"{__name__}:_immovable_factory"
    a = LocalRewardRuntime({"sleep_offload": True, "model_factory": factory})
    b = LocalRewardRuntime({"sleep_offload": True, "model_factory": factory})

    await a.score_batch(_request())
    await a.score_batch(_request())
    await b.score_batch(_request())

    # Build happened inside each runtime's own pool; tags never shared, so
    # waking one heavyweight reward cannot drag another's pages back to GPU.
    assert len(allocator.pool_tags) == 2
    assert allocator.pool_tags[0] != allocator.pool_tags[1]
    # a: score->sleep, wake->score->sleep; b: score->sleep.
    assert allocator.sleeps == [
        (allocator.pool_tags[0],),
        (allocator.pool_tags[0],),
        (allocator.pool_tags[1],),
    ]
    assert allocator.wakes == [[allocator.pool_tags[0]]]

    await a.shutdown()  # slept pool wakes before drop (frees pinned host copies)
    assert allocator.wakes[-1] == [allocator.pool_tags[0]]


def test_sleep_offload_requires_cumem(monkeypatch) -> None:
    """sleep_offload has no naive fallback: missing cumem fails loud at build."""
    import asyncio

    import vrl.utils.cuda_memory as cuda_memory_mod

    monkeypatch.setattr(cuda_memory_mod, "_cumem_allocator", lambda: None)
    runtime = LocalRewardRuntime(
        {"sleep_offload": True, "model_factory": f"{__name__}:_immovable_factory"},
    )
    with pytest.raises(RuntimeError, match="CuMemAllocator"):
        asyncio.run(runtime.score_batch(_request()))


def test_sleep_offload_rejects_injected_model() -> None:
    """An already-built model cannot be pooled; sleep_offload rejects it."""
    with pytest.raises(ValueError, match="model_factory"):
        LocalRewardRuntime({"sleep_offload": True}, model=object())
