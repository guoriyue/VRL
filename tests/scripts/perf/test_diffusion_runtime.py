"""Regression tests for shared diffusion performance-probe model loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.scripts.perf.common.diffusion_runtime import build_runtime


@pytest.mark.parametrize("use_lora", [False, True])
def test_build_runtime_preserves_the_resolved_lora_contract(
    monkeypatch,
    use_lora: bool,
) -> None:
    import vrl.families.registry as families

    root = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "sd3_5",
                    "path": "unit-checkpoint",
                    "use_lora": use_lora,
                },
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                },
            },
        ),
    )
    precision = resolve_precision_policy(root)
    device = torch.device("cpu")
    resolved_build = object()
    runtime = object()
    calls: dict[str, object] = {}

    def resolve_build(
        actual_root,
        actual_device,
        *,
        precision,
    ):
        calls["resolver"] = (actual_root, actual_device, precision)
        return resolved_build

    def build_bundle(actual_build):
        calls["builder"] = actual_build
        return runtime

    entry = SimpleNamespace(
        family="sd3_5",
        resolve_model_build=resolve_build,
        build_rollout=build_bundle,
    )
    monkeypatch.setattr(families, "get_model_family_entry", lambda _family: entry)

    assert build_runtime(root, device, precision=precision) is runtime
    assert calls == {
        "resolver": (root, device, precision),
        "builder": resolved_build,
    }
    assert root.model is not None
    assert root.model.use_lora is use_lora
