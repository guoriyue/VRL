"""Download-free ``from_spec`` loading test for Cosmos Predict2 Video2World.

Mirrors ``tests/models/diffusion/sd3_5/test_model_loading.py``: monkeypatch the
diffusers pipeline ``from_pretrained`` to return a fake pipeline (no Hub fetch,
no real weights) and assert the loader-constructed state, NOT literal YAML
config values. The Cosmos wrapper has three load-time behaviors the sd3 test
does not exercise:

  * the diffusers ``CosmosSafetyChecker`` is swapped for a passthrough during
    ``from_pretrained`` so RL training does not depend on the safety classifier
    weights, then restored afterwards;
  * diffusers' ``from_pretrained`` disables autograd globally, which the wrapper
    re-enables via ``torch.set_grad_enabled(True)``;
  * frozen modules (vae fp32, text_encoder spec dtype) are staged to the spec
    device and have grad disabled.

``transformers`` is a declared dependency of this repo but is not installed in
the test venv used by this suite; the wrapper eagerly imports the cosmos
pipeline submodule (to swap the safety checker), so a minimal ``transformers``
stub is injected only when the real package is absent. This keeps the test
exercising the real submodule swap on the real ``CosmosSafetyChecker`` symbol
rather than skipping.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from types import SimpleNamespace
from typing import Any

import torch


class _FakeModule:
    def __init__(self) -> None:
        self.dtype: torch.dtype | None = None
        self.requires_grad_enabled: bool | None = None
        self.to_calls: list[tuple[Any, torch.dtype | None]] = []

    def requires_grad_(self, enabled: bool) -> None:
        self.requires_grad_enabled = enabled

    def to(self, device: Any, dtype: torch.dtype | None = None) -> _FakeModule:
        self.to_calls.append((device, dtype))
        if dtype is not None:
            self.dtype = dtype
        return self


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = _FakeModule()
        self.vae = _FakeModule()
        self.text_encoder = _FakeModule()
        self.device = "cpu"
        self.progress_bar_disabled: bool | None = None

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.progress_bar_disabled = disable


def _ensure_transformers_importable() -> None:
    """Inject a minimal ``transformers`` stub when the real one is absent.

    The cosmos pipeline submodule does ``from transformers import ...`` at module
    import time; the wrapper imports that submodule eagerly. The real dependency
    is declared in pyproject but not installed in this venv, so provide a stub
    with a valid ``__spec__`` (diffusers' import-availability probe inspects it).
    """
    if "transformers" in sys.modules:
        return
    try:  # real package present -> use it
        import transformers  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    stub = types.ModuleType("transformers")
    stub.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    stub.__version__ = "4.44.0"
    stub.T5EncoderModel = type("T5EncoderModel", (), {})
    stub.T5TokenizerFast = type("T5TokenizerFast", (), {})
    sys.modules["transformers"] = stub


def test_cosmos_predict2_from_spec_swaps_safety_checker_and_re_enables_grad(
    monkeypatch,
) -> None:
    """Cosmos Predict2 ``from_spec`` swaps the safety checker, re-enables grad, stages frozen modules."""
    _ensure_transformers_importable()

    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as v2w_mod
    from diffusers import Cosmos2VideoToWorldPipeline

    from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2Model

    original_safety_checker = v2w_mod.CosmosSafetyChecker

    calls: list[dict[str, Any]] = []
    pipeline = _FakePipeline()
    # Record the swapped-in safety checker as seen *during* the load call, since
    # the wrapper restores the original in its ``finally`` before returning.
    captured: dict[str, Any] = {}

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> _FakePipeline:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        captured["safety_checker_during_load"] = v2w_mod.CosmosSafetyChecker
        # diffusers' real from_pretrained disables autograd globally; emulate so
        # the wrapper's re-enable is actually observable.
        torch.set_grad_enabled(False)
        return pipeline

    monkeypatch.setattr(
        Cosmos2VideoToWorldPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    spec = SimpleNamespace(
        model_name_or_path="nvidia/Cosmos-Predict2-2B-Video2World",
        dtype=torch.bfloat16,
        device="cuda:0",
    )

    # The wrapper flips the process-global grad flag; restore it after the test
    # so neighbouring tests are not affected regardless of outcome.
    grad_was_enabled = torch.is_grad_enabled()
    try:
        model = CosmosPredict2Model.from_spec(spec)

        # Wrapper wraps the fake pipeline and forwards only the model name + dtype.
        assert model.pipeline is pipeline
        assert calls == [
            {
                "model_name_or_path": "nvidia/Cosmos-Predict2-2B-Video2World",
                "torch_dtype": torch.bfloat16,
            },
        ]

        # Safety-checker passthrough swap: during the load the symbol was
        # replaced by a passthrough (text always safe, video echoed back), and
        # it is restored afterwards. Assert on behavior, not class identity.
        swapped = captured["safety_checker_during_load"]
        assert swapped is not original_safety_checker
        passthrough = swapped()
        assert passthrough.to("cuda:0") is passthrough
        assert passthrough.check_text_safety("anything") is True
        sentinel = object()
        assert passthrough.check_video_safety(sentinel) is sentinel
        assert v2w_mod.CosmosSafetyChecker is original_safety_checker

        # Grad re-enable: from_pretrained turned grad off, wrapper turns it on.
        assert torch.is_grad_enabled() is True

        # Frozen-module staging: vae fp32 + text_encoder spec dtype, both frozen
        # and moved to the spec device; progress bar disabled.
        assert pipeline.progress_bar_disabled is True
        assert pipeline.vae.requires_grad_enabled is False
        assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]
        assert pipeline.text_encoder.requires_grad_enabled is False
        assert pipeline.text_encoder.to_calls == [("cuda:0", torch.bfloat16)]
    finally:
        torch.set_grad_enabled(grad_was_enabled)
