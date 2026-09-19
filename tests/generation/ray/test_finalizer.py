"""The finalizer merges staged batch references off the GPU rank's critical path."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DenoiseBatchGatherer,
    DenoiseBatchResult,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.ray.finalizer import RayGenerationFinalizer
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.utils.media_reference import MediaReference


def _request(count: int = 2, *, references: bool = True) -> GenerationRequest:
    return GenerationRequest("r", "sd3_5", "t2i", ["p"], count, reward_media_refs=references)


def _batch(start: int, count: int) -> DenoiseBatchResult:
    return DenoiseBatchResult(
        batch=GenerationSampleBatch(0, start, count),
        latents=torch.full((count, 3, 3), float(start)),
        log_probs=torch.zeros(count, 2),
        timesteps=torch.ones(count, 2),
        kl=torch.zeros(count, 2),
        video=torch.full((count, 3, 4, 4), 128, dtype=torch.uint8),
        replay_tensors={},
        context={"model_family": "sd3_5"},
    )


def test_finalizer_requires_a_gatherer() -> None:
    with pytest.raises(TypeError, match="GenerationBatchGatherer"):
        RayGenerationFinalizer("finalize-0", object())


def test_merge_fetches_references_gathers_in_plan_order_and_boxes_media(monkeypatch) -> None:
    store = {"ref-a": _batch(2, 2), "ref-b": _batch(0, 2)}
    fetched: list[list[str]] = []
    put: list[object] = []

    def fake_get(refs):
        fetched.append(list(refs))
        return [store[ref] for ref in refs]

    monkeypatch.setattr("ray.get", fake_get)
    monkeypatch.setattr("ray.put", lambda media: put.append(media) or "media-ref")
    request = _request(4)

    output = RayGenerationFinalizer("finalize-0", DenoiseBatchGatherer()).merge_request(
        request, request.sample_rows(), ["ref-a", "ref-b"]
    )

    assert isinstance(output, GenerationOutput)
    assert fetched == [["ref-a", "ref-b"]]  # one object-store fetch for the request
    # Coverage validation orders by sample_start, not by reference order.
    assert output.trajectory.sample_rows == request.sample_rows()
    assert output.output == [MediaReference("media-ref", i, nbytes=48) for i in range(4)]
    assert len(put) == 1 and put[0].shape == (4, 3, 4, 4)


def test_merge_rejects_an_empty_reference_list() -> None:
    request = _request()
    with pytest.raises(ValueError, match="no batch references"):
        RayGenerationFinalizer("finalize-0", DenoiseBatchGatherer()).merge_request(
            request, request.sample_rows(), []
        )


@pytest.mark.slow_test
def test_real_ray_finalizer_merges_staged_batches(local_ray) -> None:
    """Staged payloads round-trip through the object store into one output."""

    ray = local_ray
    finalizer = ray.remote(num_cpus=0)(RayGenerationFinalizer).remote(
        "finalize-0", DenoiseBatchGatherer()
    )
    try:
        request = _request(4)
        refs = [ray.put(_batch(0, 2)), ray.put(_batch(2, 2))]
        output = ray.get(
            finalizer.merge_request.remote(request, request.sample_rows(), refs),
            timeout=60,
        )
        assert isinstance(output, GenerationOutput)
        assert output.request_id == "r"
        assert [row.sample_index for row in output.trajectory.sample_rows] == [0, 1, 2, 3]
        assert all(isinstance(sample, MediaReference) for sample in output.output)
        assert sum(sample.nbytes for sample in output.output) == 4 * 48
        assert ray.get(finalizer.health.remote(), timeout=30) == "finalize-0"
    finally:
        ray.kill(finalizer, no_restart=True)
