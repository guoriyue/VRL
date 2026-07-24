"""Tests for shared model-side helpers (vrl/models/utils.py)."""

from __future__ import annotations

import pytest

from vrl.models.utils import TrainableStateSlots


def _state(tag: str) -> dict[str, str]:
    # The container is payload-agnostic; a marker dict is enough to assert identity.
    return {"transformer.weight": tag}


def test_install_and_get_round_trip() -> None:
    slots = TrainableStateSlots()
    slots.install(1, _state("v1"))
    slots.install(2, _state("v2"))

    assert slots.has(1) and slots.has(2)
    assert not slots.has(3)
    assert slots.get(1)["transformer.weight"] == "v1"


def test_eviction_keeps_most_recent_versions() -> None:
    slots = TrainableStateSlots(max_retained=2)
    for version in (1, 2, 3):
        slots.install(version, _state(f"v{version}"))

    # Oldest (v1) evicted; the two most recent retained.
    assert not slots.has(1)
    assert slots.has(2) and slots.has(3)


def test_none_payload_aliases_newest_slot() -> None:
    """A version-only bump (no new weights) reuses the latest state."""
    slots = TrainableStateSlots()
    slots.install(1, _state("v1"))
    slots.install(2, None)

    assert slots.has(2)
    # v2 resolves to v1's state rather than an empty slot.
    assert slots.get(2)["transformer.weight"] == "v1"


def test_none_payload_with_no_prior_slot_is_noop() -> None:
    slots = TrainableStateSlots()
    slots.install(1, None)
    assert not slots.has(1)


def test_max_retained_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_retained"):
        TrainableStateSlots(max_retained=0)
