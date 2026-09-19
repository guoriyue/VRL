"""Tests for the model-side weight-sync receiver (vrl/models/weight_utils.py)."""

from __future__ import annotations

import pytest

from vrl.models.weight_utils import TrainableStateSlots


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


def test_tensor_impostor_is_rejected_before_any_weight_copy() -> None:
    from types import SimpleNamespace

    import torch

    from vrl.models.weight_utils import load_weights_into

    module = torch.nn.Linear(2, 2)
    original = {name: value.clone() for name, value in module.state_dict().items()}
    payload = {
        "transformer.weight": torch.full_like(module.weight, 99),
        "transformer.bias": SimpleNamespace(shape=module.bias.shape, dtype=module.bias.dtype),
    }

    with pytest.raises(TypeError, match="must be a tensor"):
        load_weights_into(module, payload, prefix="transformer")

    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, original[name])


@pytest.mark.parametrize("version", [True, 1.9, "1", -1])
@pytest.mark.parametrize("operation", ["install", "has", "get"])
def test_state_slots_reject_ambiguous_versions_without_replacing_state(version, operation):
    slots = TrainableStateSlots()
    original = _state("original")
    slots.install(1, original)
    with pytest.raises(ValueError, match="policy version"):
        if operation == "install":
            slots.install(version, _state("replacement"))
        else:
            getattr(slots, operation)(version)
    assert slots.get(1) is original


@pytest.mark.parametrize("limit", [True, 1.9, "1"])
def test_state_slots_require_exact_retention_limit(limit):
    with pytest.raises(ValueError, match="max_retained"):
        TrainableStateSlots(max_retained=limit)


@pytest.mark.parametrize("child_name", ["module", "_orig_mod"])
def test_unwrap_preserves_ordinary_named_submodules(child_name) -> None:
    import torch

    from vrl.models.weight_utils import load_weights_into, unwrap_compile_and_ddp
    from vrl.trainers.weight_sync import flatten_trainable_module_state

    model = torch.nn.Module()
    model.register_parameter("scale", torch.nn.Parameter(torch.tensor([2.0])))
    model.add_module(child_name, torch.nn.Linear(2, 1, bias=False))
    assert unwrap_compile_and_ddp(model) is model

    payload = flatten_trainable_module_state({"policy": model})
    assert set(payload) == {"policy.scale", f"policy.{child_name}.weight"}
    replacement = {name: torch.full_like(value, 7) for name, value in payload.items()}
    load_weights_into(model, replacement, prefix="policy")
    assert torch.equal(model.scale, torch.tensor([7.0]))
    assert torch.all(getattr(model, child_name).weight == 7)
