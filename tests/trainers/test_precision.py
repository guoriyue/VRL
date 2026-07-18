"""Precision normalization tests."""

from __future__ import annotations

import pytest
import torch

from vrl.trainers.precision import (
    normalize_mixed_precision,
    torch_dtype_for_mixed_precision,
)


def test_false_precision_means_full_precision() -> None:
    """Checks false precision means full precision."""
    assert normalize_mixed_precision(False) == "no"
    assert normalize_mixed_precision("false") == "no"
    assert normalize_mixed_precision("no") == "no"
    assert normalize_mixed_precision("fp32") == "no"


def test_empty_precision_means_full_precision() -> None:
    """Empty/unset mixed_precision resolves to fp32 ('no')."""
    assert normalize_mixed_precision("") == "no"


def test_torch_dtype_uses_fp16_only_when_explicit() -> None:
    """Checks torch dtype uses FP16 only when explicit."""
    assert torch_dtype_for_mixed_precision("no", torch=torch) == torch.float32
    assert torch_dtype_for_mixed_precision("bf16", torch=torch) == torch.bfloat16
    assert torch_dtype_for_mixed_precision("fp16", torch=torch) == torch.float16
    assert torch_dtype_for_mixed_precision("", torch=torch) == torch.float32


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("fp16+no-autocast", torch.float16),
        ("bf16+fp8", torch.bfloat16),
        ("bf16+nvfp4+no-autocast", torch.bfloat16),
    ],
)
def test_torch_dtype_extracts_base_dtype_from_role_label(label, expected) -> None:
    assert torch_dtype_for_mixed_precision(label, torch=torch) is expected


def test_true_precision_is_rejected_as_ambiguous() -> None:
    """Checks true precision is rejected as ambiguous."""
    with pytest.raises(ValueError, match="AMP mixed precision") as exc_info:
        normalize_mixed_precision(True)
    assert "actor.mixed_precision" not in str(exc_info.value)


def test_invalid_precision_uses_source_neutral_amp_protocol_language() -> None:
    with pytest.raises(ValueError, match="AMP mixed precision") as exc_info:
        normalize_mixed_precision("int8")
    assert "actor.mixed_precision" not in str(exc_info.value)


@pytest.mark.parametrize(
    "label",
    ["bf16+unknown", "bf16+fp8+fp8"],
)
def test_role_label_rejects_unknown_or_duplicate_execution_policy(label) -> None:
    with pytest.raises(ValueError, match="unknown or duplicate execution policy"):
        torch_dtype_for_mixed_precision(label, torch=torch)
