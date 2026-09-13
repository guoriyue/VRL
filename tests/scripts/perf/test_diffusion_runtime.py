"""Regression tests for shared diffusion performance-probe model loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tests.scripts.eval.fixtures import TinySanaPipeline, write_tiny_sana_snapshot
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.families.sana.model import SanaModel
from vrl.scripts.perf.common.diffusion_runtime import (
    build_runtime,
    prepare_sampling_state,
)


def test_build_runtime_returns_the_family_rollout_bundle(monkeypatch, tmp_path) -> None:
    """``build_runtime`` is registry resolve + family build with no projection of
    its own: on the tiny SANA snapshot it yields the real family model at the
    caller's precision, loaded once."""

    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)
    root = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "sana",
                    "path": str(snapshot),
                    "revision": None,
                    "use_lora": False,
                },
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                    "rollout": {"dtype": "fp32"},
                },
            },
        ),
    )
    precision = PrecisionPolicy.from_section(root.precision)

    bundle = build_runtime(root, torch.device("cpu"), precision=precision)

    assert isinstance(bundle.model, SanaModel)
    assert bundle.model.pipeline is pipeline
    assert pipeline.loads == 1
    assert bundle.model.transformer.dtype is torch.float32


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


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_sampling_preparation_does_not_retain_autograd_graphs(grad_enabled):
    root = parse_config(
        OmegaConf.create(
            {
                "model": {"family": "sd3_5"},
                "sampling": {"num_steps": 4, "guidance_scale": 1.0, "width": 256, "height": 256},
            }
        ),
    )
    weight = torch.tensor(2.0, requires_grad=True)
    stages = []

    class Model:
        def encode_prompt(self, *args, **kwargs):
            stages.append(torch.is_grad_enabled())
            return weight * 3

        def prepare_sampling(self, request, encoded):
            stages.append(torch.is_grad_enabled())
            return encoded * weight

    with torch.set_grad_enabled(grad_enabled):
        state = prepare_sampling_state(Model(), root)
        assert torch.is_grad_enabled() is grad_enabled
    assert stages == [False, False]
    assert state.item() == 12
    assert not state.requires_grad
    assert state.grad_fn is None


def test_e2e_duration_uses_monotonic_performance_clock(monkeypatch, capsys):
    from vrl.scripts.perf.common import diffusion_runtime

    ticks = iter([10.0, 10.1, 20.0, 20.2, 30.0, 30.3])
    calls = []

    def wall_clock():
        pytest.fail("elapsed latency must not use the adjustable wall clock")

    monkeypatch.setattr(diffusion_runtime.time, "time", wall_clock)
    monkeypatch.setattr(diffusion_runtime.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(diffusion_runtime, "_e2e_once", lambda *a: calls.append(None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    diffusion_runtime.run_e2e(None, SimpleNamespace(sampling=SimpleNamespace(num_steps=2)), "cuda")
    assert len(calls) == 5
    assert "200 ms/img (median of 3)" in capsys.readouterr().out
