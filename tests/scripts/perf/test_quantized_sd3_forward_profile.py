"""CPU tests for the SD3 quantized-forward profile's accounting helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vrl.nn.quantization import LinearTargetProfile
from vrl.scripts.perf.quantized_sd3_forward_profile import (
    _accuracy,
    _add_relative_speedups,
    _apply_scheme,
    _manifest_sha256,
    _require_matching_manifest,
    _target_manifest,
    _unique_tensor_bytes,
)


def test_unique_tensor_bytes_counts_shared_storage_once() -> None:
    module = nn.Module()
    parameter = nn.Parameter(torch.zeros(4, 8))
    module.register_parameter("weight", parameter)
    module.register_buffer("weight_view", parameter.detach().view(8, 4))
    module.register_buffer("independent", torch.zeros(3))

    assert _unique_tensor_bytes(module) == parameter.untyped_storage().nbytes() + 3 * 4


def test_accuracy_reports_normalized_metrics() -> None:
    reference = torch.ones(2, 4)
    output = torch.zeros_like(reference)

    metrics = _accuracy(output, reference)

    assert metrics["normalized_l1"] == pytest.approx(1.0)
    assert metrics["nrmse"] == pytest.approx(1.0)
    assert metrics["cosine"] == pytest.approx(0.0)
    assert metrics["abs_p99"] == pytest.approx(1.0)
    assert metrics["abs_max"] == pytest.approx(1.0)


class _MatchedTargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(1024, 1024, bias=False)
        self.mlp = nn.Module()
        self.mlp.up = nn.Linear(1024, 1024, bias=False)
        self.proj_out = nn.Linear(1024, 1024, bias=False)


@pytest.mark.parametrize(
    ("profile", "expected_paths"),
    [
        (LinearTargetProfile.MLP_ONLY, ["mlp.up"]),
        (LinearTargetProfile.ATTENTION_MLP, ["attn.to_q", "mlp.up"]),
    ],
)
def test_fp8_and_fp4_build_identical_explicit_target_manifests(
    profile: LinearTargetProfile,
    expected_paths: list[str],
) -> None:
    fp8_model = _MatchedTargetModel()
    fp4_model = _MatchedTargetModel()

    fp8_paths = _apply_scheme(fp8_model, "fp8", profile)
    fp4_paths = _apply_scheme(fp4_model, "fp4", profile)
    fp8_manifest = _target_manifest(fp8_model, fp8_paths)
    fp4_manifest = _target_manifest(fp4_model, fp4_paths)

    assert fp8_paths == expected_paths
    assert fp4_paths == expected_paths
    _require_matching_manifest("fp8", fp8_manifest, "fp4", fp4_manifest)
    assert _manifest_sha256(fp8_manifest) == _manifest_sha256(fp4_manifest)


def test_manifest_mismatch_fails_loud() -> None:
    fp8_manifest = (("attn.to_q", 1024, 1024), ("mlp.up", 1024, 1024))
    fp4_manifest = (("mlp.up", 1024, 1024),)

    with pytest.raises(RuntimeError, match=r"only_fp8=.*attn\.to_q"):
        _require_matching_manifest("fp8", fp8_manifest, "fp4", fp4_manifest)


def test_relative_speedups_include_direct_fp4_vs_fp8_ratio() -> None:
    results = [
        {"scheme": "bf16", "rows": [{"batch": 32, "median_ms": 100.0}]},
        {
            "scheme": "fp8",
            "profile": "mlp_only",
            "target_manifest_sha256": "same",
            "rows": [{"batch": 32, "median_ms": 80.0}],
        },
        {
            "scheme": "fp4",
            "profile": "mlp_only",
            "target_manifest_sha256": "same",
            "rows": [{"batch": 32, "median_ms": 50.0}],
        },
    ]

    _add_relative_speedups(results)

    assert results[1]["rows"][0]["speedup_vs_bf16"] == pytest.approx(1.25)
    assert results[2]["rows"][0]["speedup_vs_bf16"] == pytest.approx(2.0)
    assert results[2]["rows"][0]["speedup_vs_fp8"] == pytest.approx(1.6)


def test_unmatched_scopes_do_not_report_direct_format_speedup() -> None:
    results = [
        {"scheme": "bf16", "rows": [{"batch": 32, "median_ms": 100.0}]},
        {
            "scheme": "fp8",
            "profile": "attention_mlp",
            "target_manifest_sha256": "fp8",
            "rows": [{"batch": 32, "median_ms": 80.0}],
        },
        {
            "scheme": "fp4",
            "profile": "mlp_only",
            "target_manifest_sha256": "fp4",
            "rows": [{"batch": 32, "median_ms": 50.0}],
        },
    ]

    _add_relative_speedups(results)

    assert "speedup_vs_fp8" not in results[2]["rows"][0]
