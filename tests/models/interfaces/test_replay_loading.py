"""Tests for replay module-loading metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

from vrl.models.replay_loading import (
    LOADS_FULL_GENERATION_MODULES_KEY,
    bundle_loads_full_generation_modules,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)


def test_full_generation_bundle_declares_full_generation_modules() -> None:
    """A full-generation bundle declares it owns generation-only modules."""
    metadata = full_generation_bundle_metadata()
    assert metadata == {LOADS_FULL_GENERATION_MODULES_KEY: True}
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata=metadata)) is True


def test_minimal_replay_bundle_does_not_declare_full_generation_modules() -> None:
    """A minimal-replay bundle declares it does not own generation-only modules."""
    metadata = minimal_replay_bundle_metadata()
    assert metadata == {LOADS_FULL_GENERATION_MODULES_KEY: False}
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata=metadata)) is False


def test_bundle_loads_full_generation_modules_defaults_false() -> None:
    """Missing flag, missing metadata, or null metadata all read as not-owning."""
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata={})) is False
    assert bundle_loads_full_generation_modules(SimpleNamespace()) is False
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata=None)) is False
