"""The shared tensor-tree walker behind device moves, offload, and selection."""

from __future__ import annotations

import dataclasses

import torch

from vrl.trajectory.device import map_tensor_tree, move_value_to_device


@dataclasses.dataclass
class _Payload:
    latents: torch.Tensor
    meta: str


def test_map_tensor_tree_reaches_every_container_kind() -> None:
    """dicts, lists, tuples, and dataclasses all recurse; non-leaves pass through."""

    tree = {
        "a": torch.zeros(2),
        "b": [_Payload(torch.ones(3), "x"), (torch.full((1,), 2.0), "keep")],
        "c": None,
    }

    out = map_tensor_tree(
        tree,
        lambda leaf: leaf + 1,
        is_leaf=lambda v: isinstance(v, torch.Tensor),
    )

    assert torch.equal(out["a"], torch.ones(2))
    assert torch.equal(out["b"][0].latents, torch.full((3,), 2.0))
    assert out["b"][0].meta == "x"
    assert torch.equal(out["b"][1][0], torch.full((1,), 3.0))
    assert out["b"][1][1] == "keep"
    assert out["c"] is None


def test_move_value_to_device_is_none_safe_and_duck_typed() -> None:
    tree = {"t": torch.zeros(2), "s": "str"}
    assert move_value_to_device(tree, None) is tree
    moved = move_value_to_device(tree, "cpu")
    assert moved["t"].device.type == "cpu"
    assert moved["s"] == "str"
