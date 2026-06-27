"""Tests for replay module-loading metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

from vrl.models.replay_loading import (
    bundle_loads_full_generation_modules,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)


def test_bundle_metadata_drives_consumer_down_opposite_branches() -> None:
    """The two bundle builders are consumed into opposite ownership decisions.

    Asserts the behavior the metadata exists for — ``bundle_loads_full_generation_modules``
    returning True for a full-generation bundle and False for a minimal one — instead
    of mirroring each builder's literal ``{KEY: bool}`` return value.
    """
    full = SimpleNamespace(metadata=full_generation_bundle_metadata())
    minimal = SimpleNamespace(metadata=minimal_replay_bundle_metadata())

    assert bundle_loads_full_generation_modules(full) is True
    assert bundle_loads_full_generation_modules(minimal) is False


def test_bundle_loads_full_generation_modules_defaults_false() -> None:
    """Missing flag, missing metadata, or null metadata all read as not-owning."""
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata={})) is False
    assert bundle_loads_full_generation_modules(SimpleNamespace()) is False
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata=None)) is False
