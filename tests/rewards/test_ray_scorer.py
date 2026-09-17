"""CPU integration of real Ray actors, including forced process teardown."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
import torch

from vrl.ray.placement import RolePlacement
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.protocols import MemoryParkingScorer, RemoteReadyScorer, RewardScorer
from vrl.rewards.ray import (
    RayRewardCancelled,
    RayRewardError,
    RayRewardPlacement,
    RayRewardScorer,
    RayRewardTimeout,
)
from vrl.utils.lifecycle import RuntimePhase
from vrl.utils.media_reference import MediaReference


@pytest.fixture(scope="module")
def local_ray():
    import ray

    if ray.is_initialized():
        raise RuntimeError("Ray reward tests require their own CPU-only cluster")
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=80 * 1024 * 1024,
    )
    try:
        yield ray
    finally:
        ray.shutdown()


def _scorer(**kwargs):
    worker_config = {
        "model_factory": "tests.rewards._ray_model:TinyRewardModel",
        "device": "cpu",
        "scale": 2.0,
    }
    worker_config.update(kwargs.pop("worker_config", {}))
    return RayRewardScorer(worker_config, startup_timeout_s=60, shutdown_timeout_s=10, **kwargs)


def _request(media=None, **metadata):
    return RewardInferenceRequest(
        request_id="request",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="artifact",
                sample_id="sample",
                path="",
                media=torch.tensor([0.25, 0.75]) if media is None else media,
                metadata=metadata,
            ),
        ),
    )


@pytest.mark.slow_test
def test_real_ray_reward_score_media_and_reload_parking(local_ray):
    async def exercise():
        scorer = _scorer(worker_config={"sleep_offload": True, "memory_parking_mode": "reload"})
        assert isinstance(scorer, RewardScorer)
        assert isinstance(scorer, RemoteReadyScorer)
        assert isinstance(scorer, MemoryParkingScorer)
        assert scorer.requires_memory_parking
        try:
            await scorer.ensure_ready()
            await scorer.activate()
            batch_ref = local_ray.put(torch.tensor([[0.25, 0.75]]))
            result = await scorer.score_batch(_request(MediaReference(batch_ref, 0)))
            assert result[0].scores["score"] == 1.0
            assert result[0].scores["pid"] != os.getpid()
            await scorer.park_memory()
            await scorer.activate()
            assert (await scorer.score_batch(_request()))[0].scores["score"] == 1.0
        finally:
            await scorer.shutdown()
        await scorer.shutdown()
        assert scorer.lifecycle.phase is RuntimePhase.TERMINATED
        with pytest.raises(RuntimeError, match="terminated"):
            await scorer.score_batch(_request())

    asyncio.run(exercise())


@pytest.mark.slow_test
def test_real_ray_reward_timeout_kills_blocking_actor(local_ray):
    async def exercise():
        scorer = _scorer(timeout_s=0.1)
        try:
            await scorer.ensure_ready()
            # Load before starting the short scoring deadline.
            scorer._timeout = 10
            await scorer.activate()
            scorer._timeout = 0.1
            with pytest.raises(RayRewardTimeout) as caught:
                await scorer.score_batch(_request(delay=30))
            assert caught.value.retain_reward_artifacts
            with pytest.raises(RuntimeError, match="shutting down"):
                await scorer.score_batch(_request())
        finally:
            await scorer.shutdown()
        assert scorer._actor is None

    asyncio.run(exercise())


@pytest.mark.slow_test
def test_real_ray_reward_cancellation_and_failure_close_admission(local_ray):
    async def exercise():
        for cancel in (False, True):
            scorer = _scorer()
            try:
                await scorer.ensure_ready()
                await scorer.activate()
                if cancel:
                    task = asyncio.create_task(scorer.score_batch(_request(delay=30)))
                    await asyncio.sleep(0.1)
                    task.cancel()
                    with pytest.raises(RayRewardCancelled):
                        await task
                else:
                    with pytest.raises(RayRewardError, match="injected reward model failure"):
                        await scorer.score_batch(_request(fail=True))
                assert scorer.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
            finally:
                await scorer.shutdown()

    asyncio.run(exercise())


@pytest.mark.slow_test
def test_real_ray_shutdown_retains_actor_when_kill_fails(local_ray, monkeypatch):
    async def exercise():
        scorer = _scorer()
        await scorer.ensure_ready()
        actor = scorer._actor
        kill = local_ray.kill

        def fail_kill(*args, **kwargs):
            raise RuntimeError("injected kill failure")

        try:
            monkeypatch.setattr(local_ray, "kill", fail_kill)
            with pytest.raises(RuntimeError, match="injected kill failure"):
                await scorer.shutdown()
            assert scorer._actor is actor
            assert scorer.lifecycle.phase is RuntimePhase.SHUTTING_DOWN
        finally:
            monkeypatch.setattr(local_ray, "kill", kill)
            await scorer.shutdown()
        assert scorer._actor is None

    asyncio.run(exercise())


@pytest.mark.slow_test
def test_real_ray_shutdown_interrupts_inflight_score(local_ray):
    async def exercise():
        scorer = _scorer()
        await scorer.activate()
        task = asyncio.create_task(scorer.score_batch(_request(delay=30)))
        await asyncio.sleep(0.1)
        await scorer.shutdown()
        with pytest.raises(RayRewardError):
            await task
        assert scorer.lifecycle.phase is RuntimePhase.TERMINATED

    asyncio.run(exercise())


@pytest.mark.slow_test
def test_real_ray_startup_deadline_cancels_unschedulable_cpu_actor(local_ray):
    async def exercise():
        scorer = RayRewardScorer(
            {"device": "cpu", "model_factory": "tests.rewards._ray_model:TinyRewardModel"},
            cpus_per_worker=3,  # The test-owned cluster exposes only two CPUs.
            startup_timeout_s=0.1,
            shutdown_timeout_s=10,
        )
        try:
            with pytest.raises(RayRewardTimeout, match=r"reward\.ready"):
                await scorer.ensure_ready()
        finally:
            await scorer.shutdown()
        assert scorer._actor is None

    asyncio.run(exercise())


@pytest.mark.slow_test
def test_real_ray_file_reward_cancellation_cleans_only_actor_root(local_ray, tmp_path):
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")

    async def exercise():
        scorer = _scorer(
            worker_config={
                "model_factory": "tests.rewards._ray_model:TinyFileRewardModel",
            }
        )
        await scorer.activate()
        root = Path(scorer._actor_directory[1])
        assert not root.exists()  # Readiness/activation never creates media files.
        task = asyncio.create_task(scorer.score_batch(_request(delay=30)))
        try:
            async with asyncio.timeout(10):
                while not (root / "model-started").exists():
                    await asyncio.sleep(0.01)
            assert list(root.rglob("*.pt"))
            task.cancel()
            with pytest.raises(RayRewardCancelled):
                await task
        finally:
            await scorer.shutdown()
        assert not root.exists()
        assert scorer._actor_directory is None
        assert unrelated.read_text(encoding="utf-8") == "keep"

    asyncio.run(exercise())


def test_ray_reward_requires_planner_owned_cuda_and_honors_cpu_downgrade():
    with pytest.raises(ValueError, match="resolved RayRewardPlacement"):
        RayRewardScorer({"device": "cuda:2"})
    with pytest.raises(ValueError, match="explicit node_id"):
        RayRewardScorer({"device": "cuda:2"}, placement=RayRewardPlacement(shared_gpu_id=2))
    with pytest.raises(ValueError, match="sleep_offload"):
        RayRewardScorer(
            {"device": "cuda:2"},
            placement=RayRewardPlacement(shared_gpu_id=2, node_id="node"),
        )
    scorer = RayRewardScorer(
        {"device": "cpu"},
        placement=RayRewardPlacement(shared_gpu_id=2, node_id="node"),
    )
    assert scorer._actor_options()["num_gpus"] == 0
    assert "scheduling_strategy" not in scorer._actor_options()


def test_ray_reward_fractional_dedicated_and_shared_node_options():
    dedicated = RayRewardScorer(
        {"device": "cuda:3"},
        placement=RayRewardPlacement(
            placement=RolePlacement(None, (4,), (3,)),
            gpu_fraction=0.5,
        ),
    )
    assert dedicated._actor_options()["num_gpus"] == 0.5
    shared = RayRewardScorer(
        {"device": "cuda:3", "sleep_offload": True},
        placement=RayRewardPlacement(shared_gpu_id=3, node_id="a" * 56),
    )
    options = shared._actor_options()
    assert options["num_gpus"] == 0
    assert options["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert options["scheduling_strategy"].node_id == "a" * 56


def test_ray_reward_media_cleanup_rejects_unowned_directory(tmp_path):
    from vrl.rewards.ray import _remove_actor_media_directory

    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the owned"):
        _remove_actor_media_directory(str(tmp_path), uuid.uuid4().hex)
    assert marker.read_text(encoding="utf-8") == "keep"
