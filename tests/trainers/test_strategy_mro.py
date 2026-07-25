"""Where each training strategy resolves its shared state methods.

``Strategy`` is a ``Protocol``, and an explicit Protocol base is a real class:
its ``...`` method stubs are bodies that return ``None``. So listing a mixin
AFTER ``Strategy`` does not raise — it silently replaces every shared method
with a no-op that returns ``None``, while ``checkpointing.py``'s
``callable(getattr(strategy, name, None))`` probe still reports ``True`` and
skips straight past the load. That is a wrong-answer failure at resume time,
far from the edit, so the bases order is pinned here by MRO owner rather than
by behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from vrl.trainers.strategy import (
    DDPStrategy,
    FSDPStrategy,
    SingleProcessStrategy,
    Strategy,
    _TrainingStateParking,
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


def _owner(cls: type, name: str) -> type | None:
    """The class in ``cls``'s MRO that actually defines ``name``."""

    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


@pytest.mark.parametrize("strategy_cls", _STRATEGY_CLASSES, ids=lambda cls: cls.__name__)
def test_every_mixin_precedes_the_strategy_protocol(strategy_cls: type) -> None:
    """No mixin may sit behind ``Strategy``, whose stubs would then win."""

    mro = strategy_cls.__mro__
    protocol_index = mro.index(Strategy)
    for mixin in (_TrainingStateParking, _UnshardedStateStrategy):
        if mixin in mro:
            assert mro.index(mixin) < protocol_index, (
                f"{mixin.__name__} must precede Strategy in {strategy_cls.__name__}"
            )


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


@pytest.mark.parametrize("method_name", _UNSHARDED_STATE_METHODS)
def test_protocol_stub_would_silently_win_if_ordered_first(method_name: str) -> None:
    """The false side: this is why the order above is asserted, not assumed."""

    class _MisorderedStrategy(Strategy, _UnshardedStateStrategy):
        pass

    assert _owner(_MisorderedStrategy, method_name) is Strategy
    stub = getattr(_MisorderedStrategy, method_name)
    # Fails silently rather than loudly on both counts: the stub is callable, so
    # checkpointing's capability probe accepts it, and it returns None instead of
    # raising, so a skipped checkpoint load looks like a successful one.
    assert callable(stub)
    assert _call_state_method(stub, method_name) is None


def _call_state_method(stub: Any, method_name: str) -> Any:
    """Invoke one Protocol stub with the arity its real signature declares."""

    if method_name == "export_checkpoint_state":
        return stub(None, object())
    if method_name in {"load_checkpoint_state", "load_full_checkpoint_state"}:
        return stub(None, object(), {})
    if method_name == "export_optimizer_state":
        return stub(None, object(), object())
    return stub(None, object(), object(), {})
