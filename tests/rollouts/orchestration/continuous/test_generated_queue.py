"""Capacity remains owned across generation, scoring retries and cancellation."""

import pytest

from vrl.rollouts.orchestration.continuous.generated_queue import GeneratedRolloutQueue
from vrl.rollouts.orchestration.prompt_collection import GeneratedPromptGroup


def _receipt() -> GeneratedPromptGroup:
    return GeneratedPromptGroup(object(), [0], 1.0, 2.0)


def test_reservation_bounds_unfinished_generation_before_receipt() -> None:
    queue = GeneratedRolloutQueue(max_items=3, max_bytes=10)
    assert queue.reserve((0, 0), max_group_bytes=6)
    assert not queue.reserve((0, 1), max_group_bytes=6)
    queue.put((0, 0), _receipt(), nbytes=4)
    assert queue.reserve((0, 1), max_group_bytes=6)
    assert queue.stats()["reserved_bytes"] == 10


def test_scoring_and_retry_retain_artifact_and_capacity() -> None:
    queue = GeneratedRolloutQueue(max_items=1, max_bytes=10)
    receipt = _receipt()
    assert queue.reserve((0, 0), max_group_bytes=10)
    queue.put((0, 0), receipt, nbytes=8)
    assert queue.take() == ((0, 0), receipt)
    assert not queue.reserve((0, 1), max_group_bytes=1)
    # A retry uses the borrowed receipt while its capacity remains charged.
    assert queue.take() is None
    assert queue.stats()["reserved_bytes"] == 8
    queue.release((0, 0))
    assert queue.reserve((0, 1), max_group_bytes=10)


def test_overflow_and_duplicate_receipt_fail_before_mutation() -> None:
    queue = GeneratedRolloutQueue(max_items=2, max_bytes=10)
    receipt = _receipt()
    assert queue.reserve((0, 0), max_group_bytes=5)
    with pytest.raises(ValueError, match="ceiling"):
        queue.put((0, 0), receipt, nbytes=6)
    assert queue.take() is None
    assert queue.stats()["reserved_bytes"] == 5
    queue.put((0, 0), receipt, nbytes=4)
    with pytest.raises(RuntimeError, match="duplicate"):
        queue.put((0, 0), _receipt(), nbytes=1)
    assert queue.take() == ((0, 0), receipt)
    assert queue.stats()["reserved_bytes"] == 4


def test_cancel_at_every_stage_releases_capacity_and_close_stops_admission() -> None:
    queue = GeneratedRolloutQueue(max_items=3, max_bytes=30)
    for slot in range(3):
        assert queue.reserve((0, slot), max_group_bytes=10)
    queue.put((0, 1), _receipt(), nbytes=8)
    queue.put((0, 2), _receipt(), nbytes=6)
    queue.take()
    # One generating, one scoring, one queued.
    for slot in range(3):
        queue.release((0, slot))
    assert queue.stats()["reserved_bytes"] == 0
    assert queue.take() is None
    queue.close()
    queue.close()
    with pytest.raises(RuntimeError, match="closed"):
        queue.reserve((1, 0), max_group_bytes=10)


def test_impossible_or_missing_reservations_fail_instead_of_waiting_forever() -> None:
    queue = GeneratedRolloutQueue(max_items=1, max_bytes=10)
    with pytest.raises(ValueError, match="fit"):
        queue.reserve((0, 0), max_group_bytes=11)
    with pytest.raises(RuntimeError, match="unreserved"):
        queue.put((0, 0), _receipt(), nbytes=1)
    assert queue.reserve((0, 0), max_group_bytes=10)
    with pytest.raises(RuntimeError, match="duplicate"):
        queue.reserve((0, 0), max_group_bytes=10)
    queue.release((0, 0))
    with pytest.raises(RuntimeError, match="no reservation"):
        queue.release((0, 0))


def test_unscored_payload_estimate_counts_dataclass_media_without_alias_duplicates() -> None:
    from dataclasses import dataclass

    import torch
    from PIL import Image

    from vrl.trajectory import trajectory_tensor_bytes

    @dataclass
    class Receipt:
        media: object
        metadata: object

    tensor = torch.zeros(8, dtype=torch.float32)
    media = Image.new("RGB", (10, 20))
    receipt = Receipt([tensor, media], {"alias": tensor, "encoded": b"abc"})
    assert trajectory_tensor_bytes(receipt) == 32 + 800 + 3
