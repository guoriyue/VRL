"""Regression tests for shared diffusion performance-probe model loading."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.scripts.perf.common.diffusion_runtime import build_runtime


def test_build_runtime_uses_the_resolved_rollout_contract(monkeypatch) -> None:
    import vrl.families.registry as families

    cfg = SimpleNamespace(model=SimpleNamespace(family="sd3_5", use_lora=False))
    device = torch.device("cpu")
    resolved_build = SimpleNamespace(family="sd3_5")
    runtime = object()
    calls: dict[str, object] = {}

    def resolve_build(
        actual_cfg,
        actual_device,
    ):
        calls["resolver"] = (actual_cfg, actual_device)
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

    assert build_runtime(cfg, device) is runtime
    assert calls == {
        "resolver": (cfg, device),
        "builder": resolved_build,
    }
    assert cfg.model.use_lora is True
