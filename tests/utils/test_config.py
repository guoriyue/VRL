"""Shared configuration conversion boundaries."""

import builtins

import pytest
from omegaconf import OmegaConf

from vrl.utils.config import import_from_path, plain_mapping, to_builtin_deep


@pytest.mark.parametrize("converter", [plain_mapping, to_builtin_deep])
def test_config_conversion_preserves_dependency_import_failure(monkeypatch, converter):
    original_import = builtins.__import__
    failure = RuntimeError("broken OmegaConf initialization")

    def import_module(name, *args, **kwargs):
        if name == "omegaconf":
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(RuntimeError) as caught:
        if converter is plain_mapping:
            converter({"devices": (0, 1)}, field_name="resources")
        else:
            converter({"devices": (0, 1)})
    assert caught.value is failure


def test_deep_conversion_resolves_yaml_and_plain_nested_values():
    config = OmegaConf.create({"count": 2, "nested": {"values": ["${count}", 3]}})
    assert plain_mapping(config, field_name="config") == {"count": 2, "nested": {"values": [2, 3]}}
    assert to_builtin_deep({"config": config, "devices": (0, 1)}) == {
        "config": {"count": 2, "nested": {"values": [2, 3]}},
        "devices": [0, 1],
    }


def test_import_path_supports_class_owned_factory():
    from pathlib import Path

    factory = import_from_path("pathlib:Path.cwd")
    assert factory() == Path.cwd()
    assert import_from_path("pathlib:Path") is Path


def test_import_path_preserves_missing_attribute_error():
    with pytest.raises(AttributeError, match="missing_factory"):
        import_from_path("pathlib:Path.missing_factory")


@pytest.mark.parametrize("path", ["pathlib.Path", ":Path", "pathlib:"])
def test_import_path_requires_explicit_module_attribute_separator(path):
    with pytest.raises(ValueError, match="module:attribute"):
        import_from_path(path)
