"""Theorems every ``QuantizedLinear`` scheme must satisfy, stated once.

The per-scheme files (``test_fp4.py`` / ``test_fp8.py``) keep the numeric
parity and targeting tests that differ between formats; the master-weight
lifecycle is the shared base class contract and is asserted here for both
schemes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vrl.nn.quantization import QUANTIZATION_SCHEMES

_SCHEMES = pytest.mark.parametrize("scheme", sorted(QUANTIZATION_SCHEMES))


def _quantized(scheme: str, *, in_features: int = 64, out_features: int = 32, bias: bool = False):
    # Both schemes accept this shape, including NVFP4's K % 32 / N % 16 gate.
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
def test_master_free_module_holds_only_its_cache_and_blocks_base_load(scheme: str) -> None:
    quantized = QUANTIZATION_SCHEMES[scheme](nn.Linear(64, 32, bias=False), keep_master=False)

    assert quantized.weight is None
    assert set(quantized.state_dict()) == set()
    with pytest.raises(RuntimeError, match="master-free"):
        quantized.load_state_dict({"weight": torch.randn(32, 64)})
    quantized.load_state_dict({}, strict=False)


@pytest.mark.gpu
@_SCHEMES
def test_a_sample_is_bitwise_independent_of_its_batch(scheme: str) -> None:
    torch.manual_seed(0)
    linear = nn.Linear(256, 512).cuda().to(torch.bfloat16)
    quantized = QUANTIZATION_SCHEMES[scheme](linear, keep_master=False)
    batch = torch.randn(4, 48, 256, device="cuda", dtype=torch.bfloat16)
    # Unequal magnitudes: a whole-batch activation scale would follow the largest sample.
    batch *= torch.tensor([1.0, 8.0, 0.1, 30.0], device="cuda", dtype=torch.bfloat16)[
        :, None, None
    ]

    with torch.no_grad():
        together = quantized(batch)
        alone = torch.cat([quantized(sample) for sample in batch.split(1)])

    assert torch.equal(together, alone)
