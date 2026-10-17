"""The finalizer merges staged batch references off the GPU rank's critical path."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.bindings.full_sequence import (
    DenoiseBatchGatherer,
    DenoiseBatchResult,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.ray.finalizer import RayGenerationFinalizer
from vrl.generation.types import GenerationRequest


def _request(count: int = 2, *, references: bool = True) -> GenerationRequest:
    return GenerationRequest("r", "sd3_5", "t2i", ["p"], count, reward_media_refs=references)


def _batch(start: int, count: int) -> DenoiseBatchResult:
    return DenoiseBatchResult(
        batch=GenerationSampleBatch(0, start, count),
        latents=torch.full((count, 3, 3), float(start)),
        log_probs=torch.zeros(count, 2),
        timesteps=torch.ones(count, 2),
        video=torch.full((count, 3, 4, 4), 128, dtype=torch.uint8),
        replay_tensors={},
        context={"model_family": "sd3_5"},
    )


def test_merge_rejects_an_empty_reference_list() -> None:
    request = _request()
    with pytest.raises(ValueError, match="no batch references"):
        RayGenerationFinalizer("finalize-0", DenoiseBatchGatherer()).merge_request(request, [])
