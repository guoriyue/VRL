from __future__ import annotations

import importlib
from typing import Any

import pytest

from vrl.models.diffusion.common.vae_decode_memory import (
    VaeDecodeMemory,
    apply_generation_memory_policy,
    configure_memory_mechanisms,
    vae_decode_memory_from_config,
)
from vrl.models.interfaces.runtime import MEMORY_POLICY_METADATA_KEY


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

    applied = configure_memory_mechanisms(vae, mem, owner="test VAE")

    assert applied == ("tiling", "slicing")
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

    metadata = apply_generation_memory_policy(
        _FakeModel(vae),
        memory_config={"vae_decode": {"tiling": True, "slicing": False}},
        owner="test VAE",
    )

    assert vae.calls == ["enable_tiling"]
    assert metadata == {
        MEMORY_POLICY_METADATA_KEY: {
            "model_build": {
                "vae_tiling": True,
                "vae_slicing": False,
            },
        },
    }


def test_apply_generation_memory_policy_has_no_python_defaults() -> None:
    """Checks the policy applies nothing when config carries nothing."""
    vae = _FakeVAE()

    metadata = apply_generation_memory_policy(
        _FakeModel(vae),
        memory_config=None,
        owner="test VAE",
    )

    assert vae.calls == []
    assert metadata == {
        MEMORY_POLICY_METADATA_KEY: {
            "model_build": {
                "vae_tiling": False,
                "vae_slicing": False,
            },
        },
    }


def test_wan_runtime_bundle_records_model_build_memory_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Wan runtime bundle records model build memory metadata."""
    from vrl.models.diffusion.wan_2_1 import runtime
    from vrl.models.interfaces.runtime import RuntimeBuildSpec

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
            self.trainable_modules: dict[str, Any] = {}
            self.scheduler = object()
            self.raw_handle = object()

        @classmethod
        def from_spec(cls, _spec: Any) -> _FakeModel:
            return cls()

        def generation_memory_targets(self) -> dict[str, Any]:
            return {"vae_decode": self.vae}

        def apply_full_finetune(self) -> None:
            return None

        def set_num_steps(self, _num_steps: int) -> None:
            return None

    monkeypatch.setattr(runtime, "_resolve_model_cls", lambda _task: _FakeModel)

    bundle = runtime.build_wan_2_1_runtime_bundle(
        RuntimeBuildSpec(
            model_name_or_path="fake/model",
            device="cpu",
            dtype="float32",
            sampling_config={"num_steps": 2},
            model_config={"memory": {"vae_decode": {"tiling": True, "slicing": True}}},
        ),
    )

    assert bundle.metadata[MEMORY_POLICY_METADATA_KEY]["model_build"] == {
        "vae_tiling": True,
        "vae_slicing": True,
    }


@pytest.mark.parametrize(
    ("runtime_module_name", "model_module_name", "model_class_name", "build_fn_name"),
    [
        (
            "vrl.models.diffusion.sd3_5.runtime",
            "vrl.models.diffusion.sd3_5.model",
            "SD3_5Model",
            "build_sd3_5_runtime_bundle",
        ),
        (
            "vrl.models.diffusion.cosmos.predict2.runtime",
            "vrl.models.diffusion.cosmos.predict2.model",
            "CosmosPredict2Model",
            "build_cosmos_predict2_runtime_bundle",
        ),
        (
            "vrl.models.diffusion.cosmos.predict2_5.runtime",
            "vrl.models.diffusion.cosmos.predict2_5.model",
            "CosmosPredict25Model",
            "build_cosmos_predict25_runtime_bundle",
        ),
    ],
)
def test_full_generation_runtime_bundles_record_model_build_memory_metadata(
    monkeypatch: pytest.MonkeyPatch,
    runtime_module_name: str,
    model_module_name: str,
    model_class_name: str,
    build_fn_name: str,
) -> None:
    """Checks full-generation runtime bundles report VAE memory policy."""
    from vrl.models.interfaces.runtime import RuntimeBuildSpec

    class _FakeModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
            self.trainable_modules: dict[str, Any] = {}
            self.scheduler = object()
            self.raw_handle = object()

        def generation_memory_targets(self) -> dict[str, Any]:
            return {"vae_decode": self.vae}

        @classmethod
        def from_spec(cls, _spec: Any) -> _FakeModel:
            return cls()

        def apply_full_finetune(self) -> None:
            return None

        def set_num_steps(self, _num_steps: int) -> None:
            return None

        def torch_compile_transformer(self, _mode: str) -> None:
            return None

        def apply_lora(self, _spec: Any) -> None:
            return None

    runtime_module = importlib.import_module(runtime_module_name)
    model_module = importlib.import_module(model_module_name)
    monkeypatch.setattr(model_module, model_class_name, _FakeModel)

    bundle = getattr(runtime_module, build_fn_name)(
        RuntimeBuildSpec(
            model_name_or_path="fake/model",
            device="cpu",
            dtype="float32",
            sampling_config={"num_steps": 2},
            model_config={"memory": {"vae_decode": {"tiling": True, "slicing": False}}},
        ),
    )

    assert bundle.metadata[MEMORY_POLICY_METADATA_KEY]["model_build"] == {
        "vae_tiling": True,
        "vae_slicing": False,
    }


def test_declared_section_without_target_is_skipped_not_applied() -> None:
    """A declared section the model exposes no target for applies nothing.

    Namespace validity is owned once by MODEL_MEMORY_SECTIONS (shared with the
    schema lint); only a genuinely unknown section name (typo) fails. A real
    generation model always exposes its VAE target, so this target-less path is
    defensive — it must not fabricate metadata for a mechanism it never ran.
    """
    metadata = apply_generation_memory_policy(
        _FakeModel(vae=None),
        memory_config={"vae_decode": {"tiling": True}},
        owner="test VAE",
    )

    assert metadata == {MEMORY_POLICY_METADATA_KEY: {"model_build": {}}}


def test_targetless_model_passes_when_nothing_configured() -> None:
    metadata = apply_generation_memory_policy(
        _FakeModel(vae=None),
        memory_config=None,
        owner="test VAE",
    )
    # A model with no targets reports no per-target keys — fabricating
    # vae_tiling for a model that owns no VAE would be a lie in telemetry.
    assert metadata == {MEMORY_POLICY_METADATA_KEY: {"model_build": {}}}


def test_family_loaders_do_not_apply_memory_policy() -> None:
    """Family model.py files declare targets only; the policy applies knobs.

    Architecture pin for the generation-memory-hardening sprint: a loader
    importing the policy module (or carrying memory_metadata state) recreates
    the per-family drift this layer exists to remove.
    """

    from pathlib import Path

    families = Path("vrl/models/diffusion").rglob("model.py")
    offenders = []
    for path in families:
        source = path.read_text()
        if "vae_decode_memory" in source or "memory_metadata" in source:
            offenders.append(str(path))
    assert not offenders, (
        "family loaders must not apply memory policy or carry its state: "
        f"{offenders}"
    )


def test_runtime_builders_apply_generation_memory_policy() -> None:
    """Every full-generation runtime builder routes through the shared policy."""

    from pathlib import Path

    runtimes = sorted(Path("vrl/models/diffusion").rglob("runtime.py"))
    missing = [
        str(path)
        for path in runtimes
        if "apply_generation_memory_policy" not in path.read_text()
    ]
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


def test_future_targets_apply_and_report_with_own_prefix() -> None:
    """Targets beyond vae_decode flow through the same policy untouched."""

    class _TwoTargetModel:
        def __init__(self) -> None:
            self.vae = _FakeVAE()
            self.encoder = _FakeVAE()

        def generation_memory_targets(self) -> dict[str, object]:
            return {"vae_decode": self.vae, "image_encoder": self.encoder}

    model = _TwoTargetModel()
    metadata = apply_generation_memory_policy(
        model,
        memory_config={
            "vae_decode": {"tiling": True},
            "image_encoder": {"slicing": True},
        },
        owner="test",
    )

    assert model.vae.calls == ["enable_tiling"]
    assert model.encoder.calls == ["enable_slicing"]
    assert metadata[MEMORY_POLICY_METADATA_KEY]["model_build"] == {
        "vae_tiling": True,
        "vae_slicing": False,
        "image_encoder_tiling": False,
        "image_encoder_slicing": True,
    }
