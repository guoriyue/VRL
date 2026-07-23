from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from vrl.config.precision import RolePrecision
from vrl.families.registry import TokenFamilyBuild, get_model_family_entry
from vrl.models.interfaces.runtime import ModelBuild
from vrl.utils.config import import_from_path

_FAMILY_TARGETS = (
    ("janus_pro", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("emu3", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("glm_image", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("nextstep_1", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("llamagen", ("wqkv", "wo")),
)


def _project_and_resolve(
    family: str,
    *,
    use_lora: bool,
    lora: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    entry = get_model_family_entry(family)
    recipe = entry.family_build
    assert isinstance(recipe, TokenFamilyBuild)
    model_config: dict[str, Any] = {"use_lora": use_lora}
    if lora is not None:
        model_config["lora"] = lora
    build = ModelBuild(
        model_name_or_path=recipe.default_model_path,
        device="cpu",
        parameter_dtype="fp32",
        family=family,
        precision=RolePrecision(dtype="fp32", float32_precision="tf32"),
        model_config=model_config,
        sampling_config={},
    )
    projected = import_from_path(recipe.config_builder)(build)
    resolved = import_from_path(recipe.config_cls)(**projected)
    return projected, resolved


def _assert_default_lora_values(config: Any, expected_targets: tuple[str, ...]) -> None:
    assert config.lora_rank == 32
    assert config.lora_alpha == 64
    assert config.lora_target_modules == expected_targets
    assert config.lora_dropout == 0.0
    assert config.lora_init == "gaussian"


def _lora_field_names(config: Any) -> set[str]:
    return {field.name for field in fields(config) if field.name.startswith("lora_")}


@pytest.mark.parametrize(("family", "expected_targets"), _FAMILY_TARGETS)
def test_family_config_supplies_lora_defaults_when_block_is_absent(
    family: str,
    expected_targets: tuple[str, ...],
) -> None:
    projected, resolved = _project_and_resolve(family, use_lora=True)

    assert _lora_field_names(resolved).isdisjoint(projected)
    assert resolved.use_lora is True
    _assert_default_lora_values(resolved, expected_targets)


@pytest.mark.parametrize(("family", "expected_targets"), _FAMILY_TARGETS)
def test_partial_rank_override_keeps_family_config_defaults(
    family: str,
    expected_targets: tuple[str, ...],
) -> None:
    projected, resolved = _project_and_resolve(
        family,
        use_lora=True,
        lora={"rank": 8},
    )

    assert projected["lora_rank"] == 8
    assert (_lora_field_names(resolved) - {"lora_rank"}).isdisjoint(projected)
    assert resolved.lora_rank == 8
    assert resolved.lora_alpha == 64
    assert resolved.lora_target_modules == expected_targets
    assert resolved.lora_dropout == 0.0
    assert resolved.lora_init == "gaussian"


@pytest.mark.parametrize("family", [family for family, _ in _FAMILY_TARGETS])
def test_explicit_lora_fields_override_family_config_defaults(
    family: str,
) -> None:
    projected, resolved = _project_and_resolve(
        family,
        use_lora=True,
        lora={
            "alpha": 48,
            "target_modules": ["custom_q", "custom_v"],
            "dropout": 0.25,
            "init": "olora",
        },
    )

    assert "lora_rank" not in projected
    assert projected["lora_alpha"] == 48
    assert projected["lora_target_modules"] == ("custom_q", "custom_v")
    assert projected["lora_dropout"] == 0.25
    assert projected["lora_init"] == "olora"
    assert resolved.lora_rank == 32
    assert resolved.lora_alpha == 48
    assert resolved.lora_target_modules == ("custom_q", "custom_v")
    assert resolved.lora_dropout == 0.25
    assert resolved.lora_init == "olora"


@pytest.mark.parametrize("family", [family for family, _ in _FAMILY_TARGETS])
@pytest.mark.parametrize(
    ("lora", "expected"),
    (
        ({"init_lora_weights": False}, False),
        ({"init": False}, False),
        ({"init_lora_weights": "olora"}, "olora"),
        ({"init": "olora"}, "olora"),
    ),
)
def test_lora_init_aliases_preserve_bool_and_string_values(
    family: str,
    lora: dict[str, Any],
    expected: str | bool,
) -> None:
    projected, resolved = _project_and_resolve(
        family,
        use_lora=True,
        lora=lora,
    )

    assert projected["lora_init"] == expected
    assert type(projected["lora_init"]) is type(expected)
    assert resolved.lora_init == expected
    assert type(resolved.lora_init) is type(expected)


@pytest.mark.parametrize("family", [family for family, _ in _FAMILY_TARGETS])
def test_lora_init_aliases_are_mutually_exclusive(family: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"model\.lora\.init and model\.lora\.init_lora_weights are mutually exclusive",
    ):
        _project_and_resolve(
            family,
            use_lora=True,
            lora={"init": "gaussian", "init_lora_weights": False},
        )


@pytest.mark.parametrize(("family", "expected_targets"), _FAMILY_TARGETS)
def test_disabled_lora_ignores_nested_overrides(
    family: str,
    expected_targets: tuple[str, ...],
) -> None:
    projected, resolved = _project_and_resolve(
        family,
        use_lora=False,
        lora={
            "rank": 8,
            "alpha": 16,
            "target_modules": ["wrong_target"],
            "dropout": 0.5,
            "init": "olora",
            "init_lora_weights": False,
        },
    )

    assert _lora_field_names(resolved).isdisjoint(projected)
    assert resolved.use_lora is False
    _assert_default_lora_values(resolved, expected_targets)
