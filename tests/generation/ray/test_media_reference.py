"""Only the Ray generation adapter boxes online media; the driver never resolves it."""

from types import SimpleNamespace

import pytest
import torch

from tests.generation.ray._helpers import ray_sana_runtime
from vrl.generation.bindings.full_sequence import (
    DenoiseBatchGatherer,
    DenoiseBatchResult,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.ray.reward_media import reference_reward_media
from vrl.generation.types import GenerationRequest
from vrl.utils.media_reference import MediaReference


def _request(*, references=True, count=2):
    return GenerationRequest("r", "sd3_5", "t2i", ["p"], count, reward_media_refs=references)


def _batch(start=0, count=2):
    return DenoiseBatchResult(
        batch=GenerationSampleBatch(0, start, count),
        latents=torch.ones(count, 3, 3),
        log_probs=torch.zeros(count, 2),
        timesteps=torch.ones(count, 2),
        video=torch.full((count, 3, 4, 4), 128, dtype=torch.uint8),
        replay_tensors={},
        context={"model_family": "sd3_5"},
    )


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_ray_worker_puts_one_batch_and_returns_boxed_sample_views(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
):
    """The real rank stages one object per generation batch; the driver gets
    per-sample boxed views of it and never the pixels."""

    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=("rollout.samples_per_generation_batch=2",),
    ) as ray_run:
        await ray_run.runtime.activate()
        output = await ray_run.runtime.generate(
            ray_run.request(["p"], group_size=4, reward_media_refs=True)
        )

    refs = output.output
    assert [ref.sample_index for ref in refs] == [0, 1, 0, 1]
    assert all(isinstance(ref.object_ref, local_ray.ObjectRef) for ref in refs)
    # One object per generation batch, shared by that batch's samples.
    assert refs[0].object_ref == refs[1].object_ref != refs[2].object_ref == refs[3].object_ref
    assert {ref.nbytes for ref in refs} == {3 * 32 * 32}


@pytest.mark.parametrize("references,rank", [(False, None), (False, 0), (True, 1)])
def test_direct_eval_and_secondary_rank_never_put_media(monkeypatch, references, rank):
    def unexpected_put(media):
        pytest.fail("unexpected object-store materialization")

    monkeypatch.setattr("ray.put", unexpected_put)
    batch = _batch()
    original = batch.video
    reference_reward_media(batch, _request(references=references), primary=rank in (None, 0))
    assert batch.video is (None if references else original)


def test_gather_orders_refs_without_resolving_or_materializing(monkeypatch):
    def unexpected_resolve(*args, **kwargs):
        pytest.fail("gather must not fetch media")

    monkeypatch.setattr(MediaReference, "resolve", unexpected_resolve)
    request = _request(count=4)
    first, second = _batch(0), _batch(2)
    first.video = [MediaReference("first", i) for i in range(2)]
    second.video = [MediaReference("second", i) for i in range(2)]
    output = DenoiseBatchGatherer().merge_generation_batches(
        request, request.sample_rows(), [second, first]
    )
    assert output.output == first.video + second.video


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_finalized_pipelined_output_uses_same_reference_adapter(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
):
    """The real finalizer boxes the merged media exactly as the per-batch rank does:
    one object for the merged request, per-sample views of it."""

    async with ray_sana_runtime(
        monkeypatch,
        tmp_path,
        ray_sana_snapshot,
        overrides=("distributed.rollout.pipelined=true", "rollout.samples_per_generation_batch=1"),
    ) as ray_run:
        await ray_run.runtime.activate()
        output = await ray_run.runtime.generate(
            ray_run.request(["p"], group_size=2, reward_media_refs=True)
        )

    refs = output.output
    assert [ref.sample_index for ref in refs] == [0, 1]
    assert refs[0].object_ref == refs[1].object_ref
    assert all(isinstance(ref, MediaReference) for ref in refs)
    assert {ref.nbytes for ref in refs} == {3 * 32 * 32}


def test_media_reference_payload_counts_toward_pending_budget_without_resolving(monkeypatch):
    from vrl.trajectory.storage import trajectory_tensor_bytes

    def unexpected_resolve(*args, **kwargs):
        pytest.fail("byte accounting must not fetch media")

    monkeypatch.setattr(MediaReference, "resolve", unexpected_resolve)
    refs = [MediaReference("one-batch", i, nbytes=48) for i in range(2)]
    assert trajectory_tensor_bytes({"media": refs, "again": refs[0]}) == 96


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_real_ray_boxed_media_is_consumed_in_receiver(
    local_ray, ray_sana_snapshot, monkeypatch, tmp_path
):
    """The generation worker owns the media while another actor consumes it."""

    ray = local_ray

    @ray.remote(num_cpus=0)
    class Consumer:
        def score(self, samples):
            # Nested ObjectRefs remain boxed on arrival, not auto-dereferenced.
            assert all(isinstance(sample, MediaReference) for sample in samples)
            cache = {}
            shapes = [tuple(sample.resolve(cache).shape) for sample in samples]
            return shapes, len(cache)

    consumer = Consumer.remote()
    try:
        async with ray_sana_runtime(monkeypatch, tmp_path, ray_sana_snapshot) as ray_run:
            await ray_run.runtime.activate()
            output = await ray_run.runtime.generate(
                ray_run.request(["p"], group_size=2, reward_media_refs=True)
            )
            refs = output.output
            assert sum(sample.nbytes for sample in refs) == 2 * 3 * 32 * 32
            shapes, cache_size = ray.get(consumer.score.remote(refs), timeout=30)
        assert shapes == [(3, 32, 32), (3, 32, 32)]
        assert cache_size == 1
    finally:
        ray.kill(consumer, no_restart=True)


def _cross_node_media_probe():
    """Subprocess entry point: isolated two-node CPU object-transfer test."""

    import os
    from pathlib import Path

    import ray
    from ray.cluster_utils import Cluster
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    root = str(Path(__file__).resolve().parents[3])
    cluster = Cluster()
    producer = consumer = None
    original_resolve = MediaReference.resolve

    def reject_driver_decode(*args, **kwargs):
        raise AssertionError("driver must forward references without fetching media")

    MediaReference.resolve = reject_driver_decode
    try:
        first = cluster.add_node(
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=80 * 1024 * 1024,
        )
        second = cluster.add_node(
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=80 * 1024 * 1024,
        )
        ray.init(
            address=cluster.address,
            _skip_env_hook=True,
            runtime_env={"env_vars": {"PYTHONPATH": root, "CUDA_VISIBLE_DEVICES": ""}},
        )
        runtime_env = ray.get_runtime_context().runtime_env
        assert "working_dir" not in runtime_env
        assert "py_executable" not in runtime_env

        @ray.remote(num_cpus=1)
        class Producer:
            def generate(self):
                output = SimpleNamespace(
                    reward_media=torch.full((2, 3, 4, 4), 128, dtype=torch.uint8)
                )
                reference_reward_media(output, _request(), primary=True)
                return output.reward_media, ray.get_runtime_context().get_node_id(), os.getpid()

        @ray.remote(num_cpus=1)
        class Consumer:
            def score(self, samples):
                assert all(isinstance(sample, MediaReference) for sample in samples)
                cache = {}
                scores = [sample.resolve(cache).mean().item() for sample in samples]
                return scores, len(cache), ray.get_runtime_context().get_node_id(), os.getpid()

        producer = Producer.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=first.node_id,
                soft=False,
            )
        ).remote()
        consumer = Consumer.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=second.node_id,
                soft=False,
            )
        ).remote()
        refs, producer_node, producer_pid = ray.get(producer.generate.remote(), timeout=45)
        assert all(isinstance(sample.object_ref, ray.ObjectRef) for sample in refs)
        scores, cache_size, consumer_node, consumer_pid = ray.get(
            consumer.score.remote(refs),
            timeout=45,
        )
        assert producer_node != consumer_node
        assert producer_pid != consumer_pid != os.getpid()
        assert scores == pytest.approx([128 / 255, 128 / 255])
        assert cache_size == 1
        assert sum(sample.nbytes for sample in refs) == 96
    finally:
        MediaReference.resolve = original_resolve
        if producer is not None:
            ray.kill(producer, no_restart=True)
        if consumer is not None:
            ray.kill(consumer, no_restart=True)
        ray.shutdown()
        cluster.shutdown()


@pytest.mark.slow_test
def test_two_ray_nodes_forward_media_without_driver_decode():
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.conftest import ray_uv_hook_disabled\n"
                "from tests.generation.ray.test_media_reference import _cross_node_media_probe\n"
                # A subprocess still has the outer `uv run` ancestor. The
                # parent's module-level hook guard does not cross processes.
                "with ray_uv_hook_disabled():\n"
                "    _cross_node_media_probe()\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
