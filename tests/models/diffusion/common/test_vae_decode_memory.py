from __future__ import annotations

from typing import Any

import pytest

from vrl.models.diffusion.common.vae_decode_memory import (
    VaeDecodeMemory,
    apply_vae_decode_memory,
    configure_vae_decode,
    vae_decode_memory_from_config,
    vae_decode_memory_metadata,
)
from vrl.models.interfaces.runtime import MEMORY_POLICY_METADATA_KEY


class _FakeVAE:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enable_tiling(self) -> None:
        self.calls.append("enable_tiling")

    def enable_slicing(self) -> None:
        self.calls.append("enable_slicing")


def test_configure_vae_decode_calls_declared_methods() -> None:
    vae = _FakeVAE()
    mem = VaeDecodeMemory(tiling=True, slicing=True)

    applied = configure_vae_decode(vae, mem, owner="test VAE")

    assert applied == ("tiling", "slicing")
    assert vae.calls == ["enable_tiling", "enable_slicing"]


def test_configure_vae_decode_raises_on_missing_method() -> None:
    mem = VaeDecodeMemory(tiling=True)
    with pytest.raises(TypeError, match="does not support requested enable_tiling"):
        configure_vae_decode(object(), mem, owner="test VAE")


def test_vae_decode_memory_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match=r"unknown model\.memory\.vae_decode key"):
        vae_decode_memory_from_config({"mystery": True})


def test_vae_decode_memory_uses_yaml_values_only() -> None:
    assert vae_decode_memory_from_config(None) == VaeDecodeMemory()
    assert vae_decode_memory_from_config({}) == VaeDecodeMemory()
    assert vae_decode_memory_from_config(
        {"tiling": True, "slicing": False},
    ) == VaeDecodeMemory(tiling=True, slicing=False)


def test_vae_decode_memory_metadata_shape() -> None:
    mem = VaeDecodeMemory(tiling=True, slicing=False)
    assert vae_decode_memory_metadata(mem, applied=("tiling",)) == {
        MEMORY_POLICY_METADATA_KEY: {
            "model_build": {
                "vae_tiling": True,
                "vae_slicing": False,
            },
        },
    }


def test_apply_vae_decode_memory_from_config() -> None:
    vae = _FakeVAE()

    metadata = apply_vae_decode_memory(
        vae,
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


def test_apply_vae_decode_memory_has_no_python_defaults() -> None:
    vae = _FakeVAE()

    metadata = apply_vae_decode_memory(
        vae,
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
    from vrl.models.diffusion.wan_2_1 import runtime
    from vrl.models.interfaces.runtime import RuntimeBuildSpec

    class _FakeModel:
        def __init__(self) -> None:
            self.memory_metadata = {
                MEMORY_POLICY_METADATA_KEY: {
                    "model_build": {"vae_tiling": True, "vae_slicing": True},
                },
            }
            self.trainable_modules: dict[str, Any] = {}
            self.scheduler = object()
            self.backend_handle = object()

        @classmethod
        def from_spec(cls, _spec: Any) -> _FakeModel:
            return cls()

        def enable_full_finetune(self) -> None:
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
        ),
    )

    assert bundle.metadata[MEMORY_POLICY_METADATA_KEY]["model_build"] == {
        "vae_tiling": True,
        "vae_slicing": True,
    }
