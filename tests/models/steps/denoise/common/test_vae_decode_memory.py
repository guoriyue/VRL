from __future__ import annotations

import importlib
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest

from tests.models.steps.denoise.fixtures import build_tiny_autoencoder_kl
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
from vrl.nn.optimization.passes import VaeDecodeMemoryPass


def _apply_memory_policy(model: Any, memory: GenerationMemoryPolicy | None) -> None:
    """Run the vae_decode_memory pass the way the optimization sequence does."""

    build = SimpleNamespace(generation_memory=memory, family="test VAE")
    memory_pass = VaeDecodeMemoryPass()
    if memory_pass.enabled(build):
        memory_pass.apply(model, build)


def test_configure_vae_decode_memory_flips_the_real_vae_state() -> None:
    """The knobs must land on state diffusers actually keeps, not on method names.

    A double that records ``enable_tiling``/``enable_slicing`` proves we called two
    names; it cannot prove diffusers still HAS those names or that they flip
    anything. A real ``AutoencoderKL`` (2.9 ms, 4,367 params) proves both, and a
    renamed upstream method reddens here instead of passing forever.
    """

    vae = build_tiny_autoencoder_kl()
    assert (vae.use_tiling, vae.use_slicing) == (False, False)

    _apply_memory_policy(
        _FakeModel(vae),
        GenerationMemoryPolicy(vae_decode=VaeDecodeMemory(tiling=True, slicing=True)),
    )

    assert (vae.use_tiling, vae.use_slicing) == (True, True)


def test_configure_vae_decode_memory_leaves_unrequested_knobs_alone() -> None:
    """A knob the config declines must stay off, not ride along with its sibling.

    Every other case here requests tiling, so nothing pinned the negative arm: a
    production change to ``if mem.tiling`` that always enabled it passed the whole
    file. Found by mutating that branch and watching this file stay green.
    """

    vae = build_tiny_autoencoder_kl()

    _apply_memory_policy(
        _FakeModel(vae),
        GenerationMemoryPolicy(vae_decode=VaeDecodeMemory(tiling=False, slicing=True)),
    )

    assert (vae.use_tiling, vae.use_slicing) == (False, True)


def test_model_build_rehydrates_generation_memory_from_wire_payload() -> None:
    build = ModelBuild(
        model_name_or_path="fake/model",
        revision=None,
        device="cpu",
        parameter_dtype="float32",
        family="sana",
        precision=RolePrecision("fp32", "tf32"),
        rollout={"prompt_encoder_dtype": "float16"},
        generation_memory={
            "vae_decode": {
                "tiling": True,
                "slicing": False,
            },
        },
    )

    assert build.generation_memory == GenerationMemoryPolicy(
        vae_decode=VaeDecodeMemory(tiling=True, slicing=False),
    )
    assert ModelBuild(**asdict(build)) == build


def test_model_build_rejects_unresolved_or_replay_only_memory() -> None:
    common = {
        "model_name_or_path": "fake/model",
        "revision": None,
        "device": "cpu",
        "parameter_dtype": "float32",
        "family": "sana",
        "precision": RolePrecision("fp32", "tf32"),
    }

    with pytest.raises(ValueError, match=r"must not carry model\.memory"):
        ModelBuild(
            **common,
            rollout=RolloutBuildOptions(prompt_encoder_dtype="float16"),
            model_config={"memory": {"vae_decode": {"tiling": True}}},
        )
    with pytest.raises(ValueError, match="generation_memory is rollout-only"):
        ModelBuild(
            **common,
            generation_memory=GenerationMemoryPolicy(
                vae_decode=VaeDecodeMemory(tiling=True),
            ),
        )


class _FakeModel:
    """Model exposing (or not) the vae_decode generation memory target."""

    def __init__(self, vae: object | None) -> None:
        self._vae = vae

    def generation_memory_targets(self) -> dict[str, object]:
        return {} if self._vae is None else {"vae_decode": self._vae}


class _TargetlessRuntimeModel:
    """In-process runtime fake that deliberately exposes no memory target."""

    def __init__(self) -> None:
        self.trainable_modules: dict[str, Any] = {}
        self.adapter_roots: dict[str, Any] = {}
        self.scheduler = object()

    @classmethod
    def from_build(cls, _build: ModelBuild) -> _TargetlessRuntimeModel:
        return cls()

    def generation_memory_targets(self) -> dict[str, Any]:
        return {}

    def apply_full_finetune(self, *_args: Any) -> None:
        return None


def _direct_rollout_build(
    family: str,
    *,
    memory: dict[str, Any] | None,
) -> ModelBuild:
    return ModelBuild(
        model_name_or_path="fake/model",
        revision=None,
        device="cpu",
        parameter_dtype="float32",
        family=family,
        precision=RolePrecision("fp32", "tf32"),
        rollout=RolloutBuildOptions(prompt_encoder_dtype="float16"),
        generation_memory=(
            None if memory is None else GenerationMemoryPolicy(vae_decode=memory["vae_decode"])
        ),
    )


def test_runtime_bundle_applies_model_build_memory_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared rollout builder carries model-build memory knobs all the way to the VAE.

    Every descriptor family (sd3_5, sana, wan, ...) builds through the same
    ``build_denoise_runtime_bundle``; sd3_5 stands in for them.
    """

    class _VaeRuntimeModel:
        def __init__(self) -> None:
            self.vae = build_tiny_autoencoder_kl()
            self.trainable_modules: dict[str, Any] = {}
            self.adapter_roots: dict[str, Any] = {}
            self.scheduler = object()

        @classmethod
        def from_build(cls, _build: Any) -> _VaeRuntimeModel:
            return cls()

        def generation_memory_targets(self) -> dict[str, Any]:
            return {"vae_decode": self.vae}

        def apply_full_finetune(self, _build: Any) -> None:
            return None

        def set_num_steps(self, _num_steps: int) -> None:
            return None

    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.families.sd3_5 import model as sd35_model

    monkeypatch.setattr(sd35_model, "SD3_5PipelineModel", _VaeRuntimeModel)

    bundle = get_model_family_entry("sd3_5").build_rollout(
        ModelBuild(
            model_name_or_path="fake/model",
            revision=None,
            device="cpu",
            parameter_dtype="float32",
            family="sd3_5",
            precision=RolePrecision("fp32", "tf32"),
            rollout=RolloutBuildOptions(prompt_encoder_dtype="float16"),
            generation_memory=GenerationMemoryPolicy(
                vae_decode=VaeDecodeMemory(tiling=True, slicing=False),
            ),
            sampling_config={"num_steps": 2},
        ),
        expected_identity=None,
    )

    assert (bundle.model.vae.use_tiling, bundle.model.vae.use_slicing) == (True, False)


def test_configured_section_without_target_fails() -> None:
    vae_decode_config: dict[str, bool] = {"tiling": True}
    with pytest.raises(
        ValueError,
        match=r"unsupported model\.memory section\(s\) vae_decode.*<none>",
    ):
        _apply_memory_policy(
            _FakeModel(vae=None),
            GenerationMemoryPolicy(
                vae_decode=VaeDecodeMemory(**vae_decode_config),
            ),
        )


@pytest.mark.parametrize(
    ("family", "model_module_name", "model_class_name"),
    [
        ("causvid", "vrl.models.families.causvid.model", "CausVidPipelineModel"),
    ],
)
def test_targetless_in_process_runtime_rejects_direct_model_build_memory(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    model_module_name: str,
    model_class_name: str,
) -> None:
    """The runtime policy protects callers that bypass typed config validation."""

    from vrl.models.families.registry import get_model_family_entry

    model_module = importlib.import_module(model_module_name)
    monkeypatch.setattr(model_module, model_class_name, _TargetlessRuntimeModel)
    entry = get_model_family_entry(family)

    bundle = entry.build_rollout(
        _direct_rollout_build(family, memory=None), expected_identity=None
    )
    assert isinstance(bundle.model, _TargetlessRuntimeModel)

    from dataclasses import replace

    with pytest.raises(RuntimeError, match=r"did not honor model\.memory\.cpu_resident"):
        entry.build_rollout(
            replace(
                _direct_rollout_build(family, memory=None),
                generation_memory=GenerationMemoryPolicy(cpu_resident=("text_encoder",)),
            ),
            expected_identity=None,
        )

    with pytest.raises(
        ValueError,
        match=r"unsupported model\.memory section\(s\) vae_decode.*<none>",
    ):
        entry.build_rollout(
            _direct_rollout_build(family, memory={"vae_decode": {"tiling": True}}),
            expected_identity=None,
        )


def test_unconfigured_future_target_does_not_change_current_policy() -> None:
    """Exposing another target does not make it configured implicitly."""

    class _TwoTargetModel:
        def __init__(self) -> None:
            self.vae = build_tiny_autoencoder_kl()
            self.encoder = build_tiny_autoencoder_kl()

        def generation_memory_targets(self) -> dict[str, object]:
            return {"vae_decode": self.vae, "image_encoder": self.encoder}

    model = _TwoTargetModel()
    _apply_memory_policy(
        model,
        GenerationMemoryPolicy(
            vae_decode=VaeDecodeMemory(tiling=True),
        ),
    )

    assert (model.vae.use_tiling, model.vae.use_slicing) == (True, False)
    assert (model.encoder.use_tiling, model.encoder.use_slicing) == (False, False)
