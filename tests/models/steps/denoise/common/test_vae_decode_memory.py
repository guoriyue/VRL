from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from vrl.config.precision import RolePrecision
from vrl.models.steps.denoise.common.vae_decode_memory import (
    VaeDecodeMemory,
    apply_generation_memory_policy,
    configure_memory_mechanisms,
    vae_decode_memory_from_config,
)


class _FakeVAE:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enable_tiling(self) -> None:
        self.calls.append("enable_tiling")

    def enable_slicing(self) -> None:
        self.calls.append("enable_slicing")


def test_configure_memory_mechanisms_calls_declared_methods() -> None:
    """Checks configure VAE decode calls declared methods."""
    vae = _FakeVAE()
    mem = VaeDecodeMemory(tiling=True, slicing=True)

    configure_memory_mechanisms(vae, mem, owner="test VAE")

    assert vae.calls == ["enable_tiling", "enable_slicing"]


def test_configure_memory_mechanisms_raises_on_missing_method() -> None:
    """Checks configure VAE decode raises on missing method."""
    mem = VaeDecodeMemory(tiling=True)
    with pytest.raises(TypeError, match="does not support requested enable_tiling"):
        configure_memory_mechanisms(object(), mem, owner="test VAE")


def test_vae_decode_memory_rejects_unknown_keys() -> None:
    """Checks VAE decode memory rejects unknown keys."""
    with pytest.raises(ValueError, match=r"unknown model\.memory\.vae_decode key"):
        vae_decode_memory_from_config({"mystery": True})


def test_vae_decode_memory_uses_yaml_values_only() -> None:
    """Checks VAE decode memory uses YAML values only."""
    assert vae_decode_memory_from_config(None) == VaeDecodeMemory()
    assert vae_decode_memory_from_config({}) == VaeDecodeMemory()
    assert vae_decode_memory_from_config(
        {"tiling": True, "slicing": False},
    ) == VaeDecodeMemory(tiling=True, slicing=False)


class _FakeModel:
    """Model exposing (or not) the vae_decode generation memory target."""

    def __init__(self, vae: object | None) -> None:
        self._vae = vae

    def generation_memory_targets(self) -> dict[str, object]:
        return {} if self._vae is None else {"vae_decode": self._vae}


def test_apply_generation_memory_policy_from_config() -> None:
    """Policy resolves the model's vae_decode target and applies knobs."""
    vae = _FakeVAE()

    apply_generation_memory_policy(
        _FakeModel(vae),
        memory_config={"vae_decode": {"tiling": True, "slicing": False}},
        owner="test VAE",
    )

    assert vae.calls == ["enable_tiling"]


def test_apply_generation_memory_policy_has_no_python_defaults() -> None:
    """Checks the policy applies nothing when config carries nothing."""
    vae = _FakeVAE()

    apply_generation_memory_policy(
        _FakeModel(vae),
        memory_config=None,
        owner="test VAE",
    )

    assert vae.calls == []


def test_wan_runtime_bundle_applies_model_build_memory_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the Wan runtime applies its model-build memory policy."""
    from vrl.models.interfaces.runtime import (
        ModelBuild,
        RolloutBuildOptions,
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
            self.trainable_modules: dict[str, Any] = {}
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
    from vrl.families.registry import get_model_family_entry
    from vrl.models.families.wan_2_1 import model as _wan_model

    monkeypatch.setattr(_wan_model, "WanT2VDiffusersModel", _FakeModel)

    bundle = get_model_family_entry("wan_2_1").build_rollout(
        ModelBuild(
            model_name_or_path="fake/model",
            device="cpu",
            parameter_dtype="float32",
            family="wan_2_1",
            precision=RolePrecision("fp32", "tf32"),
            rollout=RolloutBuildOptions(
                prompt_encoder_dtype="float16",
            ),
            sampling_config={"num_steps": 2},
            model_config={"memory": {"vae_decode": {"tiling": True, "slicing": True}}},
        ),
    )

    assert bundle.model.vae.calls == ["enable_tiling", "enable_slicing"]


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
            "vrl.models.families.cosmos.predict2.model",
            "CosmosPredict2Model",
            "cosmos-predict2",
        ),
        # predict2_5 is also a registry-descriptor family (LoRA-only, so the
        # shared fake build below carries a minimal lora block).
        (
            "vrl.models.families.cosmos.predict2_5.model",
            "CosmosPredict25Model",
            "cosmos-predict2.5",
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
    """Checks full-generation runtime bundles apply VAE memory policy."""
    from vrl.models.interfaces.runtime import (
        ModelBuild,
        RolloutBuildOptions,
    )

    loaded_builds: list[ModelBuild] = []

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
            self.trainable_modules: dict[str, Any] = {}
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

    from vrl.families.registry import get_model_family_entry

    bundle = get_model_family_entry(build_family).build_rollout(
        ModelBuild(
            model_name_or_path="fake/model",
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
            sampling_config={"num_steps": 2},
            model_config={
                "memory": {"vae_decode": {"tiling": True, "slicing": False}},
                # LoRA path so LoRA-only descriptor families (predict2_5) pass
                # their requires_lora guard; the fake's apply_lora is a no-op.
                "use_lora": True,
                "lora": {"rank": 1, "alpha": 1, "target_modules": ["to_q"]},
            },
        ),
    )

    assert bundle.model.vae.calls == ["enable_tiling"]
    assert loaded_builds
    if build_family == "sana":
        assert loaded_builds[-1].parameter_dtype is torch.float16


def test_declared_section_without_target_is_skipped_not_applied() -> None:
    """A declared section the model exposes no target for applies nothing.

    Namespace validity is owned once by MODEL_MEMORY_SECTIONS (shared with the
    schema lint); only a genuinely unknown section name (typo) fails. A real
    generation model always exposes its VAE target, so this target-less path is
    defensive — it must not call a mechanism on a target the model does not own.
    """
    apply_generation_memory_policy(
        _FakeModel(vae=None),
        memory_config={"vae_decode": {"tiling": True}},
        owner="test VAE",
    )


def test_targetless_model_passes_when_nothing_configured() -> None:
    apply_generation_memory_policy(
        _FakeModel(vae=None),
        memory_config=None,
        owner="test VAE",
    )


def test_family_loaders_do_not_apply_memory_policy() -> None:
    """Family model.py files declare targets only; the policy applies knobs.

    Architecture pin for the generation-memory-hardening sprint: a loader
    importing the policy module (or carrying memory_metadata state) recreates
    the per-family drift this layer exists to remove.
    """

    from pathlib import Path

    families = Path("vrl/models/families").rglob("model.py")
    offenders = []
    for path in families:
        source = path.read_text()
        if "vae_decode_memory" in source or "memory_metadata" in source:
            offenders.append(str(path))
    assert not offenders, (
        f"family loaders must not apply memory policy or carry its state: {offenders}"
    )


def test_runtime_builders_apply_generation_memory_policy() -> None:
    """Every in-process generation builder routes through the shared policy.

    Registry-descriptor families ship no family builder functions. Their shared
    ``build_family_runtime_bundle`` is the single home of the policy call.
    Explicitly registered subprocess runtimes own decode memory in their
    upstream process/config and must not pretend to expose an in-process VAE.
    """

    import importlib
    from pathlib import Path

    from vrl.families.registry import FAMILY_REGISTRY, DenoiseFamilyBuild

    # The shared denoise-step builder must own the policy call.
    shared = Path("vrl/models/steps/denoise/build.py").read_text()
    assert "apply_generation_memory_policy" in shared, (
        "shared denoise builder must apply the generation memory policy"
    )

    import re

    isolated_runtime_paths: set[Path] = set()
    for entry in FAMILY_REGISTRY.values():
        if not entry.runtime_capabilities.runs_in_isolated_subprocess:
            continue
        build = entry.family_build
        assert isinstance(build, DenoiseFamilyBuild)
        assert build.rollout_runtime_builder is not None
        module_name = build.rollout_runtime_builder.partition(":")[0]
        module_file = getattr(importlib.import_module(module_name), "__file__", None)
        assert module_file is not None
        isolated_runtime_paths.add(Path(module_file).resolve())

    runtimes = sorted(Path("vrl/models/families").rglob("runtime.py"))
    missing = []
    for path in runtimes:
        source = path.read_text()
        builder_names = re.findall(r"def (build_\w+_runtime_bundle)\(", source)
        # Replay builders own no VAE and never apply the policy; only files
        # that still define a full-generation (rollout) builder must route.
        defines_rollout_builder = any("replay" not in name for name in builder_names)
        if path.resolve() in isolated_runtime_paths:
            assert "apply_generation_memory_policy" not in source
            continue
        if defines_rollout_builder and "apply_generation_memory_policy" not in source:
            missing.append(str(path))
    assert not missing, f"runtime builders missing the shared policy call: {missing}"


def test_unknown_memory_section_fails_loud() -> None:
    """A typo'd target section must error, never silently skip mechanisms."""
    import pytest

    with pytest.raises(ValueError, match="vae_deocde"):
        apply_generation_memory_policy(
            _FakeModel(_FakeVAE()),
            memory_config={"vae_deocde": {"tiling": True}},
            owner="test VAE",
        )


def test_future_targets_apply_through_the_same_policy() -> None:
    """Targets beyond vae_decode flow through the same policy unchanged."""

    class _TwoTargetModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
            self.encoder = _FakeVAE()

        def generation_memory_targets(self) -> dict[str, object]:
            return {"vae_decode": self.vae, "image_encoder": self.encoder}

    model = _TwoTargetModel()
    apply_generation_memory_policy(
        model,
        memory_config={
            "vae_decode": {"tiling": True},
            "image_encoder": {"slicing": True},
        },
        owner="test",
    )

    assert model.vae.calls == ["enable_tiling"]
    assert model.encoder.calls == ["enable_slicing"]
