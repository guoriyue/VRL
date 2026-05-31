from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vrl.models.interfaces.runtime import MEMORY_POLICY_METADATA_KEY
from vrl.trainers.frozen_module import (
    FrozenModuleOffload,
    frozen_module_offload_from_config,
    frozen_module_offload_metadata,
    park_frozen_modules,
)


class _FakeModule:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> None:
        self.devices.append(device)


def test_park_frozen_modules_moves_only_declared_present_modules() -> None:
    owner = type(
        "_Owner",
        (),
        {
            "text_encoder": _FakeModule(),
            "vae": _FakeModule(),
            "transformer": _FakeModule(),
        },
    )()
    offload = FrozenModuleOffload(
        enable=True,
        module_names=("text_encoder", "vae", "missing"),
        empty_cache=False,
    )

    moved = park_frozen_modules(owner, offload)

    assert moved == ("text_encoder", "vae")
    assert owner.text_encoder.devices == ["cpu"]
    assert owner.vae.devices == ["cpu"]
    assert owner.transformer.devices == []


def test_park_frozen_modules_disabled_is_noop() -> None:
    owner = SimpleNamespace(vae=_FakeModule())
    moved = park_frozen_modules(
        owner,
        FrozenModuleOffload(enable=False, module_names=("vae",)),
    )
    assert moved == ()
    assert owner.vae.devices == []


def test_frozen_offload_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match=r"unknown model\.memory\.frozen_offload key"):
        frozen_module_offload_from_config({"mystery_offload": True})


def test_frozen_offload_merges_over_defaults() -> None:
    defaults = FrozenModuleOffload(
        enable=True,
        module_names=("text_encoder", "vae"),
        empty_cache=True,
    )
    assert frozen_module_offload_from_config(None, defaults=defaults) == defaults
    merged = frozen_module_offload_from_config(
        {"modules": ["vae"], "empty_cache": False},
        defaults=defaults,
    )
    assert merged == FrozenModuleOffload(
        enable=True,
        module_names=("vae",),
        empty_cache=False,
    )


def test_frozen_offload_normalizes_modules_to_tuple() -> None:
    offload = frozen_module_offload_from_config(
        OmegaConf.create({"enable": True, "modules": ["text_encoder", "", "vae"]}),
    )
    assert offload.module_names == ("text_encoder", "vae")


def test_frozen_offload_metadata_shape() -> None:
    offload = FrozenModuleOffload(
        enable=True,
        module_names=("vae",),
        empty_cache=False,
    )
    assert frozen_module_offload_metadata(offload, moved=("vae",)) == {
        MEMORY_POLICY_METADATA_KEY: {
            "runtime_lifecycle": {
                "driver_frozen_module_cpu_offload": True,
                "driver_frozen_module_names": ["vae"],
                "driver_frozen_modules_moved": ["vae"],
                "empty_cuda_cache_after_driver_offload": False,
            },
        },
    }


def test_sd3_after_bundle_hook_records_driver_offload_metadata() -> None:
    from vrl.scripts.diffusion.sd3_5.train import _after_bundle_built

    transformer = SimpleNamespace(enable_gradient_checkpointing=lambda: None)
    pipeline = SimpleNamespace(
        text_encoder=_FakeModule(),
        text_encoder_2=_FakeModule(),
        text_encoder_3=None,
        vae=_FakeModule(),
    )
    bundle = SimpleNamespace(
        model=SimpleNamespace(transformer=transformer, _pipeline=pipeline),
        metadata={},
    )
    cfg = OmegaConf.create(
        {
            "actor": {"gradient_checkpointing": False},
            "model": {},
        },
    )

    _after_bundle_built(bundle, cfg)

    assert bundle.metadata[MEMORY_POLICY_METADATA_KEY]["runtime_lifecycle"] == {
        "driver_frozen_module_cpu_offload": True,
        "driver_frozen_module_names": [
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
        ],
        "driver_frozen_modules_moved": ["text_encoder", "text_encoder_2", "vae"],
        "empty_cuda_cache_after_driver_offload": True,
    }


def test_sd3_after_bundle_hook_respects_disabled_frozen_offload() -> None:
    from vrl.scripts.diffusion.sd3_5.train import _after_bundle_built

    transformer = SimpleNamespace(enable_gradient_checkpointing=lambda: None)
    pipeline = SimpleNamespace(
        text_encoder=_FakeModule(),
        text_encoder_2=_FakeModule(),
        text_encoder_3=None,
        vae=_FakeModule(),
    )
    bundle = SimpleNamespace(
        model=SimpleNamespace(transformer=transformer, _pipeline=pipeline),
        metadata={},
    )
    cfg = OmegaConf.create(
        {
            "actor": {"gradient_checkpointing": False},
            "model": {
                "memory": {
                    "frozen_offload": {
                        "enable": False,
                        "modules": ["text_encoder", "vae"],
                    },
                },
            },
        },
    )

    _after_bundle_built(bundle, cfg)

    assert pipeline.text_encoder.devices == []
    assert pipeline.vae.devices == []
    assert bundle.metadata[MEMORY_POLICY_METADATA_KEY]["runtime_lifecycle"] == {
        "driver_frozen_module_cpu_offload": False,
        "driver_frozen_module_names": ["text_encoder", "vae"],
        "driver_frozen_modules_moved": [],
        "empty_cuda_cache_after_driver_offload": True,
    }
