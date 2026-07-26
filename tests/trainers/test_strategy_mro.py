"""Strategy implementation ownership and structural protocol boundaries."""

from __future__ import annotations

import pytest
import torch

from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import (
    DDPStrategy,
    FSDPStrategy,
    SingleProcessStrategy,
    Strategy,
    _UnshardedStateStrategy,
)

# The five bodies whose shared precondition is "every rank already holds the
# full unsharded tensor". FSDP overrides all of them.
_UNSHARDED_STATE_METHODS = (
    "export_checkpoint_state",
    "load_checkpoint_state",
    "load_full_checkpoint_state",
    "export_optimizer_state",
    "load_optimizer_state",
)

_STRATEGY_CLASSES = (SingleProcessStrategy, DDPStrategy, FSDPStrategy)
_STRATEGY_METHODS = tuple(
    name
    for name, value in Strategy.__dict__.items()
    if not name.startswith("_") and callable(value)
)


def _owner(cls: type, name: str) -> type | None:
    """The class in ``cls``'s MRO that actually defines ``name``."""

    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


@pytest.mark.parametrize("strategy_cls", _STRATEGY_CLASSES, ids=lambda cls: cls.__name__)
def test_concrete_strategy_keeps_consumer_protocol_out_of_mro(strategy_cls: type) -> None:
    assert Strategy not in strategy_cls.__mro__


@pytest.mark.parametrize("strategy_cls", _STRATEGY_CLASSES, ids=lambda cls: cls.__name__)
def test_concrete_strategy_exposes_structural_contract(strategy_cls: type) -> None:
    strategy = _new_strategy(strategy_cls)

    assert isinstance(strategy.context, DistributedTrainingContext)
    assert all(callable(getattr(strategy, name, None)) for name in _STRATEGY_METHODS)


@pytest.mark.parametrize(
    ("strategy_cls", "expected_owner"),
    [
        (SingleProcessStrategy, _UnshardedStateStrategy),
        (DDPStrategy, _UnshardedStateStrategy),
        (FSDPStrategy, FSDPStrategy),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
@pytest.mark.parametrize("method_name", _UNSHARDED_STATE_METHODS)
def test_shared_state_methods_resolve_to_a_real_body(
    strategy_cls: type,
    expected_owner: type,
    method_name: str,
) -> None:
    """Single process and DDP share the mixin; FSDP owns its gathering override."""

    assert _owner(strategy_cls, method_name) is expected_owner


def _new_strategy(strategy_cls: type) -> Strategy:
    context = DistributedTrainingContext(
        strategy={
            SingleProcessStrategy: "single_process",
            DDPStrategy: "ddp",
            FSDPStrategy: "fsdp",
        }[strategy_cls],
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    if strategy_cls is SingleProcessStrategy:
        return SingleProcessStrategy(context)
    if strategy_cls is DDPStrategy:
        return DDPStrategy(context, find_unused_parameters=False)
    return FSDPStrategy(
        context,
        mesh_dims=["dp"],
        precision_policy="actor",
        reshard_after_forward=True,
        cpu_offload=False,
    )
