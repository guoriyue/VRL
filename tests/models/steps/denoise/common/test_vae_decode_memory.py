from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import Any

import pytest
import torch

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


class _FakeVAE:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enable_tiling(self) -> None:
        self.calls.append("enable_tiling")

    def enable_slicing(self) -> None:
        self.calls.append("enable_slicing")


def test_configure_vae_decode_memory_calls_declared_methods() -> None:
    """Checks configure VAE decode calls declared methods."""
    vae = _FakeVAE()
    mem = VaeDecodeMemory(tiling=True, slicing=True)

    configure_vae_decode_memory(vae, mem, owner="test VAE")

    assert vae.calls == ["enable_tiling", "enable_slicing"]


def test_configure_vae_decode_memory_raises_on_missing_method() -> None:
    """Checks configure VAE decode raises on missing method."""
    mem = VaeDecodeMemory(tiling=True)
    with pytest.raises(TypeError, match="does not support requested enable_tiling"):
        configure_vae_decode_memory(object(), mem, owner="test VAE")


def test_generation_memory_wire_mapping_rejects_unknown_keys() -> None:
    """The typed wire boundary fails closed without a parallel key set."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'mystery'"):
        GenerationMemoryPolicy(vae_decode={"mystery": True})


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
    vae = _FakeVAE()

    apply_generation_memory_policy(
        _FakeModel(vae),
        memory=GenerationMemoryPolicy(
            vae_decode=VaeDecodeMemory(tiling=True, slicing=False),
        ),
        owner="test VAE",
    )

    assert vae.calls == ["enable_tiling"]


def test_apply_generation_memory_policy_has_no_python_defaults() -> None:
    """Checks the policy applies nothing when config carries nothing."""
    vae = _FakeVAE()

    apply_generation_memory_policy(
        _FakeModel(vae),
        memory=None,
        owner="test VAE",
    )

    assert vae.calls == []


def test_apply_generation_memory_policy_raises_on_missing_target_method() -> None:
    with pytest.raises(TypeError, match="does not support requested enable_tiling"):
        apply_generation_memory_policy(
            _FakeModel(object()),
            memory=GenerationMemoryPolicy(
                vae_decode=VaeDecodeMemory(tiling=True),
            ),
            owner="test VAE",
        )


@pytest.mark.parametrize(
    ("family", "model_class_name"),
    [
        ("wan_2_1", "WanT2VDiffusersModel"),
        ("wan_2_1_i2v", "WanI2VDiffusersModel"),
    ],
)
def test_wan_runtime_bundle_applies_model_build_memory_policy(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    model_class_name: str,
) -> None:
    """Checks the Wan runtime applies its model-build memory policy."""

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
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
    from vrl.families.registry import get_model_family_entry
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
        (
            "vrl.models.families.cosmos.cosmos3.model",
            "Cosmos3Model",
            "cosmos3",
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
    loaded_builds: list[ModelBuild] = []

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
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

    from vrl.families.registry import get_model_family_entry

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

    assert bundle.model.vae.calls == ["enable_tiling"]
    assert loaded_builds
    if build_family == "sana":
        assert loaded_builds[-1].parameter_dtype is torch.float16


@pytest.mark.parametrize(
    "vae_decode_config",
    [{}, {"tiling": False}, {"tiling": True}],
)
def test_configured_section_without_target_fails(
    vae_decode_config: dict[str, bool],
) -> None:
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


@pytest.mark.parametrize("memory", [None, GenerationMemoryPolicy()])
def test_targetless_model_passes_when_nothing_configured(
    memory: GenerationMemoryPolicy | None,
) -> None:
    apply_generation_memory_policy(
        _FakeModel(vae=None),
        memory=memory,
        owner="test VAE",
    )


def test_causvid_and_echo_do_not_advertise_incompatible_vae_targets() -> None:
    from vrl.models.families.causvid.model import _CausVidPolicyModel
    from vrl.models.families.echo.model import EchoModel
    from vrl.models.steps.denoise.base import DiffusionModelBase

    for model_cls in (_CausVidPolicyModel, EchoModel):
        assert model_cls.generation_memory_targets is DiffusionModelBase.generation_memory_targets
        model = model_cls.__new__(model_cls)
        torch.nn.Module.__init__(model)
        object.__setattr__(model, "_backend", type("_Backend", (), {"vae": object()})())
        object.__setattr__(model, "_video_vae", object())

        assert model.generation_memory_targets() == {}


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

    from vrl.families.registry import get_model_family_entry

    model_module = importlib.import_module(model_module_name)
    monkeypatch.setattr(model_module, model_class_name, _TargetlessRuntimeModel)
    entry = get_model_family_entry(family)

    bundle = entry.build_rollout(_direct_rollout_build(family, memory=None))
    assert isinstance(bundle.model, _TargetlessRuntimeModel)

    for vae_decode_config in ({}, {"tiling": False}, {"tiling": True}):
        with pytest.raises(
            ValueError,
            match=r"unsupported model\.memory section\(s\) vae_decode.*<none>",
        ):
            entry.build_rollout(
                _direct_rollout_build(
                    family,
                    memory={"vae_decode": vae_decode_config},
                ),
            )


@pytest.mark.parametrize(
    "family",
    [
        "magi_1",
        "janus_pro",
        "janus_pro_r1",
        "nextstep_1",
        "emu3",
        "glm_image",
        "llamagen",
    ],
)
def test_non_vae_runtime_families_keep_memory_at_the_registered_boundary(
    family: str,
) -> None:
    from vrl.families.registry import get_model_family_entry

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
    token = Path("vrl/models/steps/token/build.py").read_text()
    assert "apply_generation_memory_policy" not in token, (
        "token-family builders own no diffusers VAE memory targets"
    )

    import re

    # This is a source-layout exception for the architecture test, not a
    # production runtime capability. MAGI owns VAE memory in its subprocess.
    isolated_runtime_paths: set[Path] = set()
    for family in ("magi_1",):
        entry = FAMILY_REGISTRY[family]
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
    """A typo'd target section must fail at the typed build boundary."""
    with pytest.raises(TypeError, match="vae_deocde"):
        GenerationMemoryPolicy(
            **{"vae_deocde": {"tiling": True}},
        )


def test_unconfigured_future_target_does_not_change_current_policy() -> None:
    """Exposing another target does not make it configured implicitly."""

    class _TwoTargetModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
            self.encoder = _FakeVAE()

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

    assert model.vae.calls == ["enable_tiling"]
    assert model.encoder.calls == []
