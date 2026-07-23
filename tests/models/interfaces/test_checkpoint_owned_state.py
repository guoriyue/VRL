from __future__ import annotations

import pytest
import torch
from torch import nn

from vrl.models.interfaces.runtime import (
    checkpoint_owned_state_names,
    register_checkpoint_owned_state,
)


class _PolicyState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Parameter(torch.ones(1))
        self.frozen_base = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.register_buffer("previous_policy", torch.ones(1))
        self.register_buffer("cache", torch.zeros(1))


def test_checkpoint_ownership_derives_trainable_and_registered_state() -> None:
    module = _PolicyState()

    register_checkpoint_owned_state(module, ["previous_policy"])

    assert checkpoint_owned_state_names(module) == {
        "trainable",
        "previous_policy",
    }


def test_checkpoint_ownership_excludes_unregistered_frozen_state() -> None:
    module = _PolicyState()

    assert checkpoint_owned_state_names(module) == {"trainable"}


def test_checkpoint_ownership_rejects_unknown_names_without_mutation() -> None:
    module = _PolicyState()

    with pytest.raises(ValueError, match="do not exist"):
        register_checkpoint_owned_state(module, ["missing"])

    assert checkpoint_owned_state_names(module) == {"trainable"}


def test_checkpoint_ownership_rejects_redundant_trainable_registration() -> None:
    module = _PolicyState()

    with pytest.raises(ValueError, match="already checkpoint-owned"):
        register_checkpoint_owned_state(module, ["trainable"])

    assert checkpoint_owned_state_names(module) == {"trainable"}


@pytest.mark.parametrize("names", ["trainable", [], [""]])
def test_checkpoint_ownership_rejects_invalid_name_collections(names) -> None:
    module = _PolicyState()

    with pytest.raises((TypeError, ValueError), match="names"):
        register_checkpoint_owned_state(module, names)
