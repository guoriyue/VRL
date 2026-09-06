"""Theorems every ``QuantizedLinear`` scheme must satisfy, stated once.

The per-scheme files (``test_fp4.py`` / ``test_fp8.py``) keep the numeric
parity and targeting tests that differ between formats; the master-weight
lifecycle and the swap's fail-before-mutation rule are the shared base class
contract and are asserted here for both schemes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vrl.nn.quantization import QUANTIZATION_SCHEMES, drop_quantized_masters

_SCHEMES = pytest.mark.parametrize("scheme", sorted(QUANTIZATION_SCHEMES))


def _quantized(scheme: str, *, in_features: int = 64, out_features: int = 32, bias: bool = False):
    # Both alignment rules (fp8 rowwise, nvfp4 K % 16 / N % 8) accept 64 x 32.
    return QUANTIZATION_SCHEMES[scheme](nn.Linear(in_features, out_features, bias=bias))


@_SCHEMES
def test_master_weight_is_the_only_state_dict_entry(scheme: str) -> None:
    """The quantized cache is derived, never serialized: weight sync loads ``weight``."""
    quantized = _quantized(scheme, bias=True)

    assert set(quantized.state_dict()) == {"weight", "bias"}


@_SCHEMES
def test_requantizes_on_state_dict_load(scheme: str) -> None:
    quantized = _quantized(scheme)
    before = [buffer.clone() for buffer in quantized.buffers()]

    quantized.load_state_dict({"weight": torch.randn(32, 64)})

    assert any(
        not torch.equal(old.view(torch.uint8), new.view(torch.uint8))
        for old, new in zip(before, quantized.buffers(), strict=True)
    )


@_SCHEMES
def test_drop_master_frees_and_blocks_base_load(scheme: str) -> None:
    quantized = _quantized(scheme)

    assert quantized.drop_master() == 32 * 64 * 4
    assert quantized.weight is None
    assert quantized.drop_master() == 0
    with pytest.raises(RuntimeError, match="master-free"):
        quantized.load_state_dict({"weight": torch.randn(32, 64)})
    quantized.load_state_dict({}, strict=False)


@_SCHEMES
def test_drop_quantized_masters_walks_the_tree(scheme: str) -> None:
    root = nn.Sequential(_quantized(scheme))

    assert drop_quantized_masters(root) > 0
    assert root[0].weight is None


@_SCHEMES
def test_invalid_target_profile_raises_before_mutation(scheme: str) -> None:
    root = nn.Sequential(nn.Linear(1024, 1024))

    with pytest.raises(ValueError, match="bogus"):
        QUANTIZATION_SCHEMES[scheme].swap_linears(root, target_profile="bogus")
    assert isinstance(root[0], nn.Linear) and type(root[0]) is nn.Linear
