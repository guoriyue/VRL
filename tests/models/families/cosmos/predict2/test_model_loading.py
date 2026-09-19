"""Download-free ``from_build`` loading test for Cosmos Predict2 Video2World.

Mirrors ``tests/models/sd3_5/test_model_loading.py``: monkeypatch the
diffusers pipeline ``from_pretrained`` to return a fake pipeline (no Hub fetch,
no real weights) and assert the loader-constructed state, NOT literal YAML
config values. The Cosmos wrapper has three load-time behaviors the sd3 test
does not exercise:

  * the diffusers ``CosmosSafetyChecker`` is swapped for a passthrough during
    ``from_pretrained`` so RL training does not depend on the safety classifier
    weights, then restored afterwards;
  * loader changes to thread-local autograd mode are restored to the caller's
    mode, including when loading fails;
  * frozen modules (vae fp32, text_encoder build dtype) are staged to the build
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
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import RecordingModule
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = RecordingModule()
        self.vae = RecordingModule()
        self.text_encoder = RecordingModule()
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


def test_cosmos_predict2_from_build_swaps_safety_checker_and_restores_grad(
    monkeypatch,
) -> None:
    """Cosmos Predict2 ``from_build`` swaps the safety checker, restores grad mode, stages frozen modules."""
    _ensure_transformers_importable()

    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as v2w_mod
    from diffusers import Cosmos2VideoToWorldPipeline

    from vrl.models.families.cosmos.predict2.model import CosmosPredict2Model

    original_safety_checker = v2w_mod.CosmosSafetyChecker

    calls: list[dict[str, Any]] = []
    pipeline = _FakePipeline()
    # Record the swapped-in safety checker as seen *during* the load call, since
    # the wrapper restores the original in its ``finally`` before returning.
    captured: dict[str, Any] = {}

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> _FakePipeline:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        captured["safety_checker_during_load"] = v2w_mod.CosmosSafetyChecker
        # Simulate a loader that changes the caller thread's grad mode.
        torch.set_grad_enabled(False)
        return pipeline

    monkeypatch.setattr(
        Cosmos2VideoToWorldPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    build = ModelBuild(
        model_name_or_path="nvidia/Cosmos-Predict2-2B-Video2World",
        revision=None,
        device="cuda:0",
        parameter_dtype=torch.bfloat16,
        family="cosmos-predict2",
        precision=RolePrecision("bf16", "tf32"),
    )

    with torch.set_grad_enabled(True):
        model = CosmosPredict2Model.from_build(build)

        # Wrapper wraps the fake pipeline and forwards only the model name + dtype.
        assert model.pipeline is pipeline
        assert calls == [
            {
                "model_name_or_path": "nvidia/Cosmos-Predict2-2B-Video2World",
                "torch_dtype": {
                    "default": torch.bfloat16,
                    "transformer": torch.bfloat16,
                    "vae": torch.float32,
                },
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

        # Restore the enabled mode established by the enclosing context.
        assert torch.is_grad_enabled() is True

        # Frozen-module staging: vae fp32 + text_encoder build dtype, both frozen
        # and moved to the build device; progress bar disabled.
        assert pipeline.progress_bar_disabled is True
        assert pipeline.vae.requires_grad_enabled is False
        assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]
        assert pipeline.text_encoder.requires_grad_enabled is False
        assert pipeline.text_encoder.to_calls == [("cuda:0", torch.bfloat16)]


def test_custom_cosmos_loaders_apply_component_dtypes(monkeypatch) -> None:
    import diffusers

    from vrl.models.families.cosmos.cosmos3.model import Cosmos3Model
    from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model
    from vrl.models.interfaces.runtime import RolloutBuildOptions

    for model_class, pipeline_name, has_encoder in (
        (CosmosPredict25Model, "Cosmos2_5_PredictBasePipeline", True),
        (Cosmos3Model, "Cosmos3OmniPipeline", False),
    ):
        pipeline = _FakePipeline()
        calls = []

        def load(path, *, recorded=calls, loaded=pipeline, **kwargs):
            recorded.append(kwargs)
            return loaded

        monkeypatch.setattr(
            diffusers,
            pipeline_name,
            types.SimpleNamespace(from_pretrained=load),
            raising=False,
        )
        build = ModelBuild(
            model_name_or_path="local-model",
            revision="snapshot",
            device="cpu",
            parameter_dtype=torch.bfloat16,
            family="test",
            precision=RolePrecision("bf16", "tf32"),
            model_config={"local_files_only": True},
            rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.float32),
        )
        model_class.from_build(build)
        # Frozen components load at the rollout encoder dtype by default; the
        # trainable transformer and the fp32 VAE are the named exceptions.
        expected = {
            "default": torch.float32,
            "transformer": torch.bfloat16,
            "vae": torch.float32,
        }
        if has_encoder:
            assert pipeline.text_encoder.to_calls == [("cpu", torch.float32)]
        assert calls[0]["torch_dtype"] == expected
        assert calls[0]["revision"] == "snapshot"
        assert calls[0]["local_files_only"] is True


@pytest.mark.parametrize(
    "family", ["predict2", "predict2_5", "predict2_5_without_encoder", "cosmos3"]
)
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("fail_load", [False, True])
def test_cosmos_load_preserves_caller_grad_mode(monkeypatch, family, grad_enabled, fail_load):
    import diffusers

    from vrl.models.families.cosmos.cosmos3.model import Cosmos3Model
    from vrl.models.families.cosmos.predict2.model import CosmosPredict2Model
    from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model

    model_class, pipeline_name = {
        "predict2": (CosmosPredict2Model, "Cosmos2VideoToWorldPipeline"),
        "predict2_5": (CosmosPredict25Model, "Cosmos2_5_PredictBasePipeline"),
        "cosmos3": (Cosmos3Model, "Cosmos3OmniPipeline"),
        "predict2_5_without_encoder": (CosmosPredict25Model, "Cosmos2_5_PredictBasePipeline"),
    }[family]
    failure = RuntimeError("loader failed after changing grad mode")

    def load(*args, **kwargs):
        torch.set_grad_enabled(not grad_enabled)
        if fail_load:
            raise failure
        return _FakePipeline()

    monkeypatch.setattr(
        diffusers,
        pipeline_name,
        types.SimpleNamespace(from_pretrained=load),
        raising=False,
    )
    skip_encoder = family == "predict2_5_without_encoder"
    if skip_encoder:
        monkeypatch.setattr(
            "vrl.models.families.cosmos.predict2_5.model._load_pipeline_without_text_encoder",
            load,
        )
    build = ModelBuild(
        model_config={"skip_text_encoder": skip_encoder},
        model_name_or_path="local-model",
        revision=None,
        device="cpu",
        parameter_dtype=torch.bfloat16,
        family=family,
        precision=RolePrecision("bf16", "tf32"),
    )
    with torch.set_grad_enabled(grad_enabled):
        if fail_load:
            with pytest.raises(RuntimeError, match="loader failed") as caught:
                model_class.from_build(build)
            assert caught.value is failure
        else:
            model_class.from_build(build)
        assert torch.is_grad_enabled() is grad_enabled
