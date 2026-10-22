"""The prefetcher: which unit it copies ahead, and that the copies are the units."""

from __future__ import annotations

import torch

from vrl.nn.offload import Prefetcher


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
