"""Regression tests for shared diffusion performance-probe model loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.scripts.perf.common.diffusion_runtime import (
    build_runtime,
    prepare_sampling_state,
)


def test_build_runtime_hands_the_resolved_build_to_the_family_rollout_builder(
    monkeypatch,
) -> None:
    """The registry entry is ``build_runtime``'s only exit.

    ``resolve_model_build`` must receive the caller's own ``(root, device,
    precision)`` objects, and ``build_rollout`` must receive exactly what the
    resolver returned — the function adds no projection of its own.
    """
    import vrl.models.families.registry as families

    root = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "sd3_5",
                    "path": "unit-checkpoint",
                    "use_lora": False,
                },
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                },
            },
        ),
    )
    precision = PrecisionPolicy.from_section(root.precision)
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


@pytest.mark.parametrize(
    ("sampling_max_sequence_length", "executor_max_sequence_length", "expected"),
    [
        (64, 128, 64),
        (None, 128, 128),
        (None, None, None),
    ],
)
def test_prepare_sampling_state_uses_only_real_sequence_length_sources(
    sampling_max_sequence_length: int | None,
    executor_max_sequence_length: int | None,
    expected: int | None,
) -> None:
    sampling = {
        "width": 256,
        "height": 256,
        "num_frames": 9,
        "num_steps": 4,
        "guidance_scale": 1.0,
    }
    if sampling_max_sequence_length is not None:
        sampling["max_sequence_length"] = sampling_max_sequence_length
    model_config: dict[str, object] = {"family": "wan_2_1"}
    if executor_max_sequence_length is not None:
        model_config["executor"] = {
            "max_sequence_length": executor_max_sequence_length,
        }
    cfg = OmegaConf.create({"model": model_config, "sampling": sampling})

    class CapturingModel:
        def __init__(self) -> None:
            self.encode_kwargs: dict[str, object] | None = None
            self.request = None

        def encode_prompt(self, prompts, negative_prompt, **kwargs):
            assert prompts == ["a physical scene, high quality"]
            assert negative_prompt is None
            self.encode_kwargs = kwargs
            return {"encoded": True}

        def prepare_sampling(self, request, encoded):
            assert encoded == {"encoded": True}
            self.request = request
            return "prepared"

    model = CapturingModel()

    assert prepare_sampling_state(model, parse_config(cfg)) == "prepared"
    assert model.encode_kwargs is not None
    assert model.request is not None
    if expected is None:
        assert "max_sequence_length" not in model.encode_kwargs
    else:
        assert model.encode_kwargs["max_sequence_length"] == expected
