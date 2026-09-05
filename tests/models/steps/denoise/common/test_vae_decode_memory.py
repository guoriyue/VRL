from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import build_tiny_autoencoder_kl
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
from vrl.models.steps.denoise.common.vae_decode_memory import (
    apply_generation_memory_policy,
    configure_vae_decode_memory,
)


def test_configure_vae_decode_memory_flips_the_real_vae_state() -> None:
    """The knobs must land on state diffusers actually keeps, not on method names.

    A double that records ``enable_tiling``/``enable_slicing`` proves we called two
    names; it cannot prove diffusers still HAS those names or that they flip
    anything. A real ``AutoencoderKL`` (2.9 ms, 4,367 params) proves both, and a
    renamed upstream method reddens here instead of passing forever.
    """

    vae = build_tiny_autoencoder_kl()
    assert (vae.use_tiling, vae.use_slicing) == (False, False)

    configure_vae_decode_memory(vae, VaeDecodeMemory(tiling=True, slicing=True), owner="test VAE")

    assert (vae.use_tiling, vae.use_slicing) == (True, True)


def test_configure_vae_decode_memory_leaves_unrequested_knobs_alone() -> None:
    """A knob the config declines must stay off, not ride along with its sibling.

    Every other case here requests tiling, so nothing pinned the negative arm: a
    production change to ``if mem.tiling`` that always enabled it passed the whole
    file. Found by mutating that branch and watching this file stay green.
    """

    vae = build_tiny_autoencoder_kl()

    configure_vae_decode_memory(vae, VaeDecodeMemory(tiling=False, slicing=True), owner="test VAE")

    assert (vae.use_tiling, vae.use_slicing) == (False, True)


def test_generation_memory_wire_mapping_rehydrates_nested_policy() -> None:
    memory = GenerationMemoryPolicy(
        vae_decode={"tiling": True, "slicing": False},
    )

    assert memory.vae_decode == VaeDecodeMemory(tiling=True, slicing=False)


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
        self.raw_handle = object()

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


def test_apply_generation_memory_policy_from_resolved_policy() -> None:
    """Policy resolves the model's vae_decode target and applies knobs."""
    vae = build_tiny_autoencoder_kl()

    apply_generation_memory_policy(
        _FakeModel(vae),
        memory=GenerationMemoryPolicy(
            vae_decode=VaeDecodeMemory(tiling=True, slicing=False),
        ),
        owner="test VAE",
    )

    assert (vae.use_tiling, vae.use_slicing) == (True, False)


def test_apply_generation_memory_policy_has_no_python_defaults() -> None:
    """No config means no knob: a default applied here would be invisible in YAML."""

    vae = build_tiny_autoencoder_kl()

    apply_generation_memory_policy(
        _FakeModel(vae),
        memory=None,
        owner="test VAE",
    )

    assert (vae.use_tiling, vae.use_slicing) == (False, False)


def test_wan_runtime_bundle_applies_model_build_memory_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Wan runtime must carry model-build memory knobs all the way to the VAE."""
    family, model_class_name = "wan_2_1_i2v", "WanI2VDiffusersModel"

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = build_tiny_autoencoder_kl()
            self.trainable_modules: dict[str, Any] = {}
            self.adapter_roots: dict[str, Any] = {}
            self.scheduler = object()
            self.raw_handle = object()

        @classmethod
        def from_build(cls, _build: Any) -> _FakeModel:
            return cls()

        def generation_memory_targets(self) -> dict[str, Any]:
            return {"vae_decode": self.vae}

        def apply_full_finetune(self, _build: Any) -> None:
            return None

        def set_num_steps(self, _num_steps: int) -> None:
            return None

    # Wan is a descriptor family: its registry entry resolves the shared builder
    # and model class, so the test follows the same boundary as production.
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.families.wan_2_1 import model as _wan_model

    monkeypatch.setattr(_wan_model, model_class_name, _FakeModel)

    bundle = get_model_family_entry(family).build_rollout(
        ModelBuild(
            model_name_or_path="fake/model",
            revision=None,
            device="cpu",
            parameter_dtype="float32",
            family=family,
            precision=RolePrecision("fp32", "tf32"),
            rollout=RolloutBuildOptions(
                prompt_encoder_dtype="float16",
            ),
            generation_memory=GenerationMemoryPolicy(
                vae_decode=VaeDecodeMemory(tiling=True, slicing=True),
            ),
            sampling_config={"num_steps": 2},
        ),
    )

    assert (bundle.model.vae.use_tiling, bundle.model.vae.use_slicing) == (True, True)


@pytest.mark.parametrize(
    (
        "model_module_name",
        "model_class_name",
        "build_family",
    ),
    [
        # sd3_5 is a registry-descriptor family: the generic builder resolves
        # its model class from the registry recipe, keyed by build.family.
        (
            "vrl.models.families.sd3_5.model",
            "SD3_5Model",
            "sd3_5",
        ),
        (
            "vrl.models.families.sana.model",
            "SanaModel",
            "sana",
        ),
    ],
)
def test_full_generation_runtime_bundles_apply_model_build_memory_policy(
    monkeypatch: pytest.MonkeyPatch,
    model_module_name: str,
    model_class_name: str,
    build_family: str,
) -> None:
    """A full-generation bundle must apply the resolved VAE knobs during build."""
    loaded_builds: list[ModelBuild] = []

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = build_tiny_autoencoder_kl()
            self.trainable_modules: dict[str, Any] = {}
            self.adapter_roots: dict[str, Any] = {}
            self.scheduler = object()
            self.raw_handle = object()

        def generation_memory_targets(self) -> dict[str, Any]:
            return {"vae_decode": self.vae}

        @classmethod
        def from_build(cls, build: ModelBuild) -> _FakeModel:
            loaded_builds.append(build)
            return cls()

        def apply_full_finetune(self, _build: Any) -> None:
            return None

        def set_num_steps(self, _num_steps: int) -> None:
            return None

        def torch_compile_transformer(self, _mode: str) -> None:
            return None

        def apply_lora(self, _build: Any) -> None:
            return None

    model_module = importlib.import_module(model_module_name)
    monkeypatch.setattr(model_module, model_class_name, _FakeModel)

    from vrl.models.families.registry import get_model_family_entry

    bundle = get_model_family_entry(build_family).build_rollout(
        ModelBuild(
            model_name_or_path="fake/model",
            revision=None,
            device="cpu",
            parameter_dtype="float16" if build_family == "sana" else "float32",
            family=build_family,
            precision=RolePrecision(
                "fp16" if build_family == "sana" else "fp32",
                "ieee" if build_family == "sana" else "tf32",
                outer_autocast=build_family != "sana",
            ),
            rollout=RolloutBuildOptions(
                prompt_encoder_dtype="float16",
            ),
            generation_memory=GenerationMemoryPolicy(
                vae_decode=VaeDecodeMemory(tiling=True, slicing=False),
            ),
            sampling_config={"num_steps": 2},
            model_config={
                # LoRA path so LoRA-only descriptor families (predict2_5) pass
                # their requires_lora guard; the fake's apply_lora is a no-op.
                "use_lora": True,
                "lora": {"rank": 1, "alpha": 1, "target_modules": ["to_q"]},
            },
        ),
    )

    assert (bundle.model.vae.use_tiling, bundle.model.vae.use_slicing) == (True, False)
    assert loaded_builds
    if build_family == "sana":
        assert loaded_builds[-1].parameter_dtype is torch.float16


def test_configured_section_without_target_fails() -> None:
    vae_decode_config: dict[str, bool] = {"tiling": True}
    with pytest.raises(
        ValueError,
        match=r"unsupported model\.memory section\(s\) vae_decode.*<none>",
    ):
        apply_generation_memory_policy(
            _FakeModel(vae=None),
            memory=GenerationMemoryPolicy(
                vae_decode=VaeDecodeMemory(**vae_decode_config),
            ),
            owner="test VAE",
        )


def test_targetless_model_passes_when_nothing_configured() -> None:
    apply_generation_memory_policy(
        _FakeModel(vae=None),
        memory=None,
        owner="test VAE",
    )


@pytest.mark.parametrize(
    ("family", "model_module_name", "model_class_name"),
    [
        ("causvid", "vrl.models.families.causvid.model", "CausVidModel"),
        ("echo", "vrl.models.families.echo.model", "EchoModel"),
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

    bundle = entry.build_rollout(_direct_rollout_build(family, memory=None))
    assert isinstance(bundle.model, _TargetlessRuntimeModel)

    with pytest.raises(
        ValueError,
        match=r"unsupported model\.memory section\(s\) vae_decode.*<none>",
    ):
        entry.build_rollout(
            _direct_rollout_build(family, memory={"vae_decode": {"tiling": True}}),
        )


@pytest.mark.parametrize(
    "family",
    ["magi_1", "janus_pro"],
)
def test_non_vae_runtime_families_keep_memory_at_the_registered_boundary(
    family: str,
) -> None:
    from vrl.models.families.registry import get_model_family_entry

    entry = get_model_family_entry(family)
    entry.validate_model_runtime_sections(
        executor_config=None,
        memory_config=None,
    )
    entry.validate_model_runtime_sections(
        executor_config={},
        memory_config={},
    )
    with pytest.raises(ValueError, match=r"does not support model\.memory"):
        entry.validate_model_runtime_sections(
            executor_config=None,
            memory_config={"vae_decode": {}},
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
    apply_generation_memory_policy(
        model,
        memory=GenerationMemoryPolicy(
            vae_decode=VaeDecodeMemory(tiling=True),
        ),
        owner="test",
    )

    assert (model.vae.use_tiling, model.vae.use_slicing) == (True, False)
    assert (model.encoder.use_tiling, model.encoder.use_slicing) == (False, False)
