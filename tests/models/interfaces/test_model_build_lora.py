"""LoRA config views and admission belong to the typed model build."""

from __future__ import annotations

import pytest
import torch

from vrl.config.model_schema import LoraSection
from vrl.config.precision import RolePrecision
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
def test_disabled_lora_has_no_attach_config(model_config) -> None:
    build = _build(model_config)

    assert build.lora is None
    with pytest.raises(ValueError, match=r"requires model\.lora configuration"):
        build.require_lora_config()


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
            **values,
            "path": "/adapter",
        }
    )
    assert build.require_lora_config().path == "/adapter"


# One row per distinct ``lora_defaults``: the shared ModelSection default
# (sd3_5), WanModelSection (both wan entries) and CausVidModelSection.
@pytest.mark.parametrize(
    ("family", "init_lora_weights", "autocast_adapter_dtype"),
    [("sd3_5", "gaussian", True), ("wan_2_1", True, False), ("causvid", True, True)],
)
def test_family_lora_defaults_are_resolved_by_the_build(
    family: str,
    init_lora_weights: str | bool,
    autocast_adapter_dtype: bool,
) -> None:
    build = _build({"use_lora": True, "lora": {"rank": 2, "alpha": 4, "target_modules": ["proj"]}})
    build.family = family

    config = build.require_lora_config()

    assert config.init_lora_weights == init_lora_weights
    assert config.autocast_adapter_dtype is autocast_adapter_dtype
    assert config.dropout == 0.0
