"""Tests for replay module-loading metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.models.replay_loading import (
    FULL_GENERATION_RUNTIME_ROLE,
    MINIMAL_REPLAY_RUNTIME_ROLE,
    ReplayModuleLoadingProfile,
    bundle_loads_full_generation_modules,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
    module_loading_profile_from_metadata,
    require_minimal_replay_bundle,
)


def test_full_generation_metadata_round_trips() -> None:
    """Checks full generation metadata round trips."""
    metadata = full_generation_bundle_metadata(
        replay_modules=("transformer",),
        generation_only_modules=("text_encoder", "vae"),
    )

    profile = module_loading_profile_from_metadata(metadata)

    assert profile.runtime_role == FULL_GENERATION_RUNTIME_ROLE
    assert profile.loads_full_generation_modules is True
    assert profile.requires_minimal_replay_loader is True
    assert profile.replay_modules == ("transformer",)
    assert profile.generation_only_modules == ("text_encoder", "vae")
    assert metadata["replay_modules"] == ["transformer"]
    assert metadata["generation_only_modules"] == ["text_encoder", "vae"]


def test_minimal_replay_metadata_round_trips_and_satisfies_guard() -> None:
    """Checks minimal replay metadata round trips and satisfies guard."""
    bundle = SimpleNamespace(
        metadata=minimal_replay_bundle_metadata(replay_modules=("transformer",)),
    )

    require_minimal_replay_bundle(bundle)

    profile = module_loading_profile_from_metadata(bundle.metadata)
    assert profile.runtime_role == MINIMAL_REPLAY_RUNTIME_ROLE
    assert profile.loads_full_generation_modules is False
    assert profile.requires_minimal_replay_loader is False
    assert bundle_loads_full_generation_modules(bundle) is False


def test_full_generation_bundle_fails_minimal_replay_requirement() -> None:
    """Checks full generation bundle fails minimal replay requirement."""
    bundle = SimpleNamespace(metadata=full_generation_bundle_metadata())

    with pytest.raises(ValueError, match="minimal_replay_model"):
        require_minimal_replay_bundle(bundle, owner="test.bundle")


def test_inconsistent_minimal_replay_metadata_is_rejected() -> None:
    """Checks inconsistent minimal replay metadata is rejected."""
    with pytest.raises(ValueError, match="cannot declare"):
        ReplayModuleLoadingProfile(
            runtime_role=MINIMAL_REPLAY_RUNTIME_ROLE,
            loads_full_generation_modules=True,
            requires_minimal_replay_loader=False,
        )


def test_missing_metadata_key_fails_fast() -> None:
    """Checks missing metadata key fails fast."""
    with pytest.raises(ValueError, match="requires_minimal_replay_loader"):
        module_loading_profile_from_metadata(
            {
                "runtime_role": FULL_GENERATION_RUNTIME_ROLE,
                "loads_full_generation_modules": True,
            },
        )
