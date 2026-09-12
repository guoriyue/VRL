"""Capacity remains owned across generation, scoring retries and cancellation."""

import pytest

from vrl.rollouts.orchestration.continuous.pending_reward_capacity import PendingRewardCapacity


def test_reservation_bounds_unfinished_generation_before_receipt() -> None:
    capacity = PendingRewardCapacity(max_groups=3, max_bytes=10)
    assert capacity.reserve((0, 0), max_group_bytes=6)
    assert not capacity.reserve((0, 1), max_group_bytes=6)
    capacity.record_generated((0, 0), nbytes=4)
    assert capacity.reserve((0, 1), max_group_bytes=6)
    assert capacity.stats()["reserved_bytes"] == 10


def test_scoring_and_retry_retain_capacity() -> None:
    capacity = PendingRewardCapacity(max_groups=1, max_bytes=10)
    assert capacity.reserve((0, 0), max_group_bytes=10)
    capacity.record_generated((0, 0), nbytes=8)
    capacity.start_scoring((0, 0))
    assert not capacity.reserve((0, 1), max_group_bytes=1)
    # A retry uses the locally held receipt while its capacity remains charged.
    assert capacity.stats()["scoring_groups"] == 1
    assert capacity.stats()["unscored_items"] == 0
    assert capacity.stats()["reserved_bytes"] == 8
    capacity.release((0, 0))
    assert capacity.reserve((0, 1), max_group_bytes=10)


def test_overflow_and_duplicate_size_report_fail_before_mutation() -> None:
    capacity = PendingRewardCapacity(max_groups=2, max_bytes=10)
    assert capacity.reserve((0, 0), max_group_bytes=5)
    with pytest.raises(ValueError, match="ceiling"):
        capacity.record_generated((0, 0), nbytes=6)
    assert capacity.stats()["unscored_items"] == 0
    assert capacity.stats()["reserved_bytes"] == 5
    capacity.record_generated((0, 0), nbytes=4)
    with pytest.raises(RuntimeError, match="duplicate"):
        capacity.record_generated((0, 0), nbytes=1)
    capacity.start_scoring((0, 0))
    assert capacity.stats()["reserved_bytes"] == 4


def test_cancel_at_every_stage_releases_capacity_and_close_stops_admission() -> None:
    capacity = PendingRewardCapacity(max_groups=3, max_bytes=30)
    for slot in range(3):
        assert capacity.reserve((0, slot), max_group_bytes=10)
    capacity.record_generated((0, 1), nbytes=8)
    capacity.record_generated((0, 2), nbytes=6)
    capacity.start_scoring((0, 1))
    # One generating, one scoring, one waiting for scoring.
    for slot in range(3):
        capacity.release((0, slot))
    assert capacity.stats()["reserved_bytes"] == 0
    assert capacity.stats()["unscored_items"] == 0
    capacity.close()
    capacity.close()
    with pytest.raises(RuntimeError, match="closed"):
        capacity.reserve((1, 0), max_group_bytes=10)


def test_impossible_or_missing_reservations_fail_instead_of_waiting_forever() -> None:
    capacity = PendingRewardCapacity(max_groups=1, max_bytes=10)
    with pytest.raises(ValueError, match="fit"):
        capacity.reserve((0, 0), max_group_bytes=11)
    with pytest.raises(RuntimeError, match="unreserved"):
        capacity.record_generated((0, 0), nbytes=1)
    assert capacity.reserve((0, 0), max_group_bytes=10)
    with pytest.raises(RuntimeError, match="duplicate"):
        capacity.reserve((0, 0), max_group_bytes=10)
    capacity.release((0, 0))
    with pytest.raises(RuntimeError, match="no reservation"):
        capacity.release((0, 0))


def test_unscored_payload_estimate_counts_dataclass_media_without_alias_duplicates() -> None:
    from dataclasses import dataclass

    import torch
    from PIL import Image

    from vrl.trajectory.storage import trajectory_tensor_bytes

    @dataclass
    class Receipt:
        media: object
        metadata: object

    tensor = torch.zeros(8, dtype=torch.float32)
    media = Image.new("RGB", (10, 20))
    receipt = Receipt([tensor, media], {"alias": tensor, "encoded": b"abc"})
    assert trajectory_tensor_bytes(receipt) == 32 + 800 + 3


def test_scoring_requires_pending_reward_capacity_and_cannot_start_twice() -> None:
    capacity = PendingRewardCapacity(max_groups=1, max_bytes=10)
    assert capacity.reserve((0, 0), max_group_bytes=10)
    with pytest.raises(RuntimeError, match="not waiting"):
        capacity.start_scoring((0, 0))
    capacity.record_generated((0, 0), nbytes=4)
    capacity.start_scoring((0, 0))
    with pytest.raises(RuntimeError, match="not waiting"):
        capacity.start_scoring((0, 0))
    with pytest.raises(RuntimeError, match="duplicate"):
        capacity.record_generated((0, 0), nbytes=2)
    assert capacity.stats()["reserved_bytes"] == 4
    assert capacity.stats()["scoring_groups"] == 1


@pytest.mark.parametrize("nbytes", [-1])
def test_invalid_byte_counts_leave_reservation_unchanged(nbytes) -> None:
    capacity = PendingRewardCapacity(max_groups=2, max_bytes=8)
    assert capacity.reserve((0, 0), max_group_bytes=4)
    with pytest.raises(ValueError, match="bytes"):
        capacity.record_generated((0, 0), nbytes=nbytes)
    assert capacity.stats()["reserved_bytes"] == 4
    assert capacity.stats()["unscored_items"] == 0
    capacity.release((0, 0))
    assert capacity.stats()["reserved_bytes"] == 0


@pytest.mark.parametrize(
    "name",
    [
        "max_inflight_groups",
        "max_ready_bytes_mb",
        "max_unscored_groups",
        "max_unscored_bytes_mb",
        "max_generated_group_bytes_mb",
        "fail_fast_errors",
    ],
)
@pytest.mark.parametrize("value", [1.5, True, "2"])
def test_capacity_configuration_does_not_coerce_counts(name, value) -> None:
    from vrl.trainers.core.types import ContinuousRolloutConfig

    with pytest.raises(ValueError, match=name):
        ContinuousRolloutConfig(**{name: value})
