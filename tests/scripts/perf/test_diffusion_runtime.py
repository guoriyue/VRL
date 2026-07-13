"""Regression tests for shared diffusion performance-probe model loading."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.scripts.perf.common.diffusion_runtime import build_model


def test_build_model_passes_dtype_as_named_parameter_override(monkeypatch) -> None:
    """The probe override must not depend on the resolver's argument order."""
    import vrl.rollouts.families as families
    import vrl.utils.config as config_utils

    cfg = SimpleNamespace(model=SimpleNamespace(family="sd3_5", use_lora=False))
    device = torch.device("cpu")
    dtype = torch.bfloat16
    resolved_build = SimpleNamespace(family="sd3_5")
    model = object()
    calls: dict[str, object] = {}

    def resolve_build(
        actual_cfg,
        actual_device,
        *,
        parameter_dtype_override,
    ):
        calls["resolver"] = (actual_cfg, actual_device, parameter_dtype_override)
        return resolved_build

    def build_bundle(actual_build):
        calls["builder"] = actual_build
        return SimpleNamespace(model=model)

    entry = SimpleNamespace(
        family="sd3_5",
        model_build_resolver="test:resolve_build",
        runtime_builder="test:build_bundle",
    )
    monkeypatch.setattr(families, "get_rollout_family_entry", lambda _family: entry)
    monkeypatch.setattr(
        config_utils,
        "import_from_path",
        lambda path: {
            "test:resolve_build": resolve_build,
            "test:build_bundle": build_bundle,
        }[path],
    )

    assert build_model(cfg, device, dtype) is model
    assert calls == {
        "resolver": (cfg, device, dtype),
        "builder": resolved_build,
    }
    assert cfg.model.use_lora is True
