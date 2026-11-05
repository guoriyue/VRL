"""LoRA config views and admission belong to the typed model build."""

from __future__ import annotations

import pytest
import torch

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


@pytest.mark.parametrize("missing", ["rank", "alpha", "target_modules"])
def test_lora_attach_requires_the_topology(missing: str) -> None:
    values = {"rank": 2, "alpha": 4, "target_modules": ["proj"], "path": "/adapter"}
    del values[missing]
    build = _build({"use_lora": True, "lora": values})

    with pytest.raises(ValueError, match=r"requires rank, alpha, and target_modules"):
        build.require_lora_config()


# One row per distinct family decision: the shared ModelSection (sd3_5) and
# WanModelSection; the build reads them off the family's schema class.
@pytest.mark.parametrize(
    ("family", "init_lora_weights", "autocast_adapter_dtype"),
    [("sd3_5", "gaussian", True), ("wan_2_1", True, False)],
)
def test_family_lora_decisions_follow_the_model_family(
    family: str,
    init_lora_weights: str | bool,
    autocast_adapter_dtype: bool,
) -> None:
    build = _build({"use_lora": True, "lora": {"rank": 2, "alpha": 4, "target_modules": ["proj"]}})
    build.family = family

    assert build.lora_init_weights == init_lora_weights
    assert build.lora_autocast_adapter_dtype is autocast_adapter_dtype
