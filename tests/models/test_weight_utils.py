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


@pytest.mark.parametrize("dtype", ["float32", "float64", "bfloat16"])
def test_installed_weight_readback_detects_noop_and_swapped_parameters(dtype):
    import torch

    from vrl.models.weight_utils import load_weights_into, verify_weights_in

    module = torch.nn.Linear(2, 2, bias=False).to(getattr(torch, dtype))
    payload = {"transformer.weight": torch.tensor([[1, 2], [3, 4]], dtype=module.weight.dtype)}
    with torch.no_grad():
        module.weight.fill_(-77)
    with pytest.raises(RuntimeError, match="installed weight content"):
        verify_weights_in(module, payload, prefix="transformer")
    load_weights_into(module, payload, prefix="transformer")
    verify_weights_in(module, payload, prefix="transformer")
    with torch.no_grad():
        module.weight.copy_(module.weight.flip(0))
    with pytest.raises(RuntimeError, match="installed weight content"):
        verify_weights_in(module, payload, prefix="transformer")
    assert payload["transformer.weight"][0, 0].item() == 1


def test_weight_readback_compares_bits_including_signed_zero_and_nan():
    import torch

    from vrl.models.weight_utils import load_weights_into, verify_weights_in

    module = torch.nn.Linear(2, 1, bias=False)
    payload = {"transformer.weight": torch.tensor([[float("nan"), -0.0]])}
    load_weights_into(module, payload, prefix="transformer")
    verify_weights_in(module, payload, prefix="transformer")
    with torch.no_grad():
        module.weight[0, 1] = 0.0
    with pytest.raises(RuntimeError, match="installed weight content"):
        verify_weights_in(module, payload, prefix="transformer")


def test_multi_root_readback_cannot_ignore_missing_expert_or_frozen_parameter():
    import torch

    from vrl.models.weight_utils import verify_trainable_modules

    modules = {name: torch.nn.Linear(1, 1) for name in ("low", "high")}
    for module in modules.values():
        module.bias.requires_grad_(False)
    payload = {
        f"{name}.weight": module.weight.detach().clone() for name, module in modules.items()
    }
    verify_trainable_modules(modules, payload)
    with pytest.raises(ValueError, match="empty"):
        verify_trainable_modules(modules, {"low.weight": payload["low.weight"]})
    with pytest.raises(ValueError, match="exactly trainable keys"):
        verify_trainable_modules(modules, {**payload, "low.bias": modules["low"].bias.detach()})
