"""LoRA config views and admission belong to the typed model build."""

from __future__ import annotations

import pytest
import torch

from vrl.config.model_schema import LoraSection
from vrl.config.precision import RolePrecision
from vrl.models.families.registry import FAMILY_REGISTRY
from vrl.models.interfaces.runtime import ModelBuild


def _build(model_config: dict | None) -> ModelBuild:
    return ModelBuild(
        model_name_or_path="test-model",
        revision=None,
        device="cpu",
        parameter_dtype=torch.float32,
        family="flux",
        precision=RolePrecision("fp32", "ieee"),
        model_config=model_config,
    )


@pytest.mark.parametrize("model_config", [None, {}, {"use_lora": False}])
def test_disabled_lora_has_no_attach_config_or_previous_request(model_config) -> None:
    build = _build(model_config)

    assert build.lora is None
    assert not build.previous_policy_adapter_requested
    build.require_lora_for_previous_policy_adapter()
    with pytest.raises(ValueError, match=r"requires model\.lora configuration"):
        build.require_lora_config()


@pytest.mark.parametrize("use_lora", [False, True])
def test_previous_adapter_admission_reads_build_config(use_lora: bool) -> None:
    build = _build({"use_lora": use_lora, "lora": {"previous_adapter": True}})

    assert build.previous_policy_adapter_requested
    if use_lora:
        build.require_lora_for_previous_policy_adapter()
    else:
        with pytest.raises(RuntimeError, match="previous_adapter requires LoRA"):
            build.require_lora_for_previous_policy_adapter()


@pytest.mark.parametrize(
    "extras",
    [{}, {"dropout": 0.2, "init_lora_weights": False}],
)
def test_required_lora_config_resolves_defaults_and_path(extras) -> None:
    values = {"rank": 2, "alpha": 4, "target_modules": ["proj"], **extras}
    build = _build({"use_lora": True, "lora": {**values, "path": "/adapter"}})

    assert build.require_lora_config() == LoraSection(
        **{
            "dropout": 0.0,
            "init_lora_weights": "gaussian",
            "autocast_adapter_dtype": True,
            "previous_adapter": False,
            **values,
            "path": "/adapter",
        }
    )
    assert build.require_lora_config().path == "/adapter"
    assert not build.previous_policy_adapter_requested


@pytest.mark.parametrize("family", FAMILY_REGISTRY)
def test_family_lora_defaults_are_resolved_by_the_build(family: str) -> None:
    build = _build({"use_lora": True, "lora": {"rank": 2, "alpha": 4, "target_modules": ["proj"]}})
    build.family = family

    config = build.require_lora_config()

    assert config.init_lora_weights == (
        True if family in {"wan_2_1", "wan_2_1_i2v", "causvid"} else "gaussian"
    )
    assert config.autocast_adapter_dtype is (family not in {"wan_2_1", "wan_2_1_i2v"})
    assert config.previous_adapter is (family == "cosmos-predict2.5")
    assert config.dropout == 0.0
