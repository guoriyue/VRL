"""Precision normalization tests."""

from __future__ import annotations

import pytest
import torch

from vrl.trainers.precision import (
    normalize_mixed_precision,
    torch_dtype_for_mixed_precision,
)


def test_false_precision_means_full_precision() -> None:
    assert normalize_mixed_precision(False, bf16=True) == "no"
    assert normalize_mixed_precision("false", bf16=True) == "no"
    assert normalize_mixed_precision("no", bf16=True) == "no"
    assert normalize_mixed_precision("fp32", bf16=True) == "no"


def test_empty_precision_uses_bf16_flag_without_fp16_fallback() -> None:
    assert normalize_mixed_precision("", bf16=True) == "bf16"
    assert normalize_mixed_precision("", bf16=False) == "no"


def test_torch_dtype_uses_fp16_only_when_explicit() -> None:
    assert torch_dtype_for_mixed_precision("no", bf16=False, torch=torch) == torch.float32
    assert torch_dtype_for_mixed_precision("bf16", bf16=False, torch=torch) == torch.bfloat16
    assert torch_dtype_for_mixed_precision("fp16", bf16=False, torch=torch) == torch.float16
    assert torch_dtype_for_mixed_precision("", bf16=False, torch=torch) == torch.float32


def test_true_precision_is_rejected_as_ambiguous() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_mixed_precision(True, bf16=True)
