"""The prefetcher: which unit it copies ahead, and that the copies are the units."""

from __future__ import annotations

import pytest
import torch

from vrl.nn.offload import Prefetcher
from vrl.nn.offload.prefetch import IN_FLIGHT_UNITS


def test_pass_direction_reads_forward_recompute_and_a_new_pass_from_its_end() -> None:
    count = 3
    # (last, current) -> the unit to copy while ``current`` is in use.
    expected = {
        (None, 0): 1,  # first call at the front: a forward pass
        (0, 1): 2,
        (1, 2): None,  # the forward pass ends
        (2, 2): 1,  # the recompute starts where the forward ended: backward
        (2, 1): 0,
        (1, 0): None,
        (None, 2): 1,  # first call at the back: already backward
        (0, 2): 1,  # a jump to the back starts a backward pass
        (2, 0): 1,  # a jump to the front starts a forward pass
    }
    assert {key: Prefetcher.pass_direction(*key, count) for key in expected} == expected


def test_prefetcher_hands_out_each_unit_and_copies_the_predicted_one_ahead() -> None:
    units = [{"w": torch.full((2,), float(i)), "b": torch.tensor(float(-i))} for i in range(3)]
    prefetcher = Prefetcher(units, "cpu")

    for index in (0, 1, 2, 2, 1, 0):
        copies = prefetcher.acquire(index)
        assert copies.keys() == units[index].keys()
        for name, tensor in copies.items():
            torch.testing.assert_close(tensor, units[index][name])

    # Only the first unit of each pass misses.
    assert (prefetcher.hits, prefetcher.misses) == (4, 2)


@pytest.mark.gpu
def test_the_host_waits_once_the_device_falls_more_than_the_in_flight_units_behind() -> None:
    """Each queued unit holds its own copies, so an unbounded host lead grows the cache."""

    units = [{"w": torch.full((1024,), float(i)).pin_memory()} for i in range(6)]
    prefetcher = Prefetcher(units, "cuda")
    done: list[torch.cuda.Event] = []
    for index in range(len(units)):
        prefetcher.acquire(index)
        torch.cuda._sleep(20_000_000)  # tens of ms of device work per unit
        done.append(torch.cuda.Event())
        done[-1].record()
        # The work queued IN_FLIGHT_UNITS + 1 acquires ago has run.
        lagging = index - IN_FLIGHT_UNITS - 1
        if lagging >= 0:
            assert done[lagging].query()
    assert not done[-1].query()
    torch.cuda.synchronize()
