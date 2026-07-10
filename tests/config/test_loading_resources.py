"""Bundled config resource and external-tree loading contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from vrl.config.loading import list_bundled_configs, load_config


def test_bundled_config_inventory_contains_canonical_recipe() -> None:
    names = list_bundled_configs("experiment")

    assert "experiment/diffusion/sd3_5/online_grpo_ocr.yaml" in names
    assert all(name.startswith("experiment/") for name in names)


def test_logical_name_cannot_be_shadowed_by_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / "experiment" / "diffusion" / "sd3_5"
    shadow.mkdir(parents=True)
    (shadow / "online_grpo_ocr.yaml").write_text("model:\n  family: shadowed\n")
    monkeypatch.chdir(tmp_path)

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")

    assert cfg.model.family == "sd3_5"


def test_absolute_external_config_remains_supported(tmp_path: Path) -> None:
    external = tmp_path / "external.yaml"
    external.write_text("custom:\n  value: 7\n")

    cfg = load_config(external.resolve())

    assert cfg.custom.value == 7


def test_explicit_root_composes_external_config_tree(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("value:\n  inherited: true\n")
    (tmp_path / "experiment.yaml").write_text(
        "defaults:\n  - /base\n  - _self_\nvalue:\n  local: true\n",
    )

    cfg = load_config("experiment", root=tmp_path)

    assert cfg.value.inherited is True
    assert cfg.value.local is True


def test_bundled_logical_name_cannot_escape_resource_root() -> None:
    with pytest.raises(ValueError, match="invalid bundled config name"):
        load_config("../outside")
