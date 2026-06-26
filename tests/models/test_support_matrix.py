"""Physical-AI model-support matrix (P0) — structural invariants.

Pins the table as a consumed asset (not a rotting doc): keys are unique, every
row is internally consistent, and the sprint's seam/tier/logprob policy holds
(64B Super is track-only; action models name an env and start un-RL-eligible).
"""

from __future__ import annotations

from vrl.models.support_matrix import (
    MODEL_SUPPORT_MATRIX,
    ModelSupportEntry,
    get_support_entry,
)


def test_keys_are_unique() -> None:
    keys = [e.key for e in MODEL_SUPPORT_MATRIX]
    assert len(keys) == len(set(keys))


def test_get_support_entry_roundtrip_and_miss() -> None:
    for entry in MODEL_SUPPORT_MATRIX:
        assert get_support_entry(entry.key) is entry
    try:
        get_support_entry("does-not-exist")
    except KeyError as err:
        assert "unknown model-support key" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for unknown key")


def test_action_models_declare_env_and_payload() -> None:
    """Any action-emitting model must name a closed-loop env (and vice versa)."""
    for entry in MODEL_SUPPORT_MATRIX:
        emits_action = "action" in entry.outputs
        if emits_action and entry.action_payload is not None:
            assert entry.env is not None, f"{entry.key}: action model without env"
        if entry.env is not None:
            assert entry.action_payload is not None, f"{entry.key}: env without action payload"


def test_super_64b_is_track_only() -> None:
    """The sprint forbids binding the system to 64B; Super stays track-only."""
    for entry in MODEL_SUPPORT_MATRIX:
        if entry.params_b == 64.0:
            assert entry.tier == "tier4"
            assert entry.status == "track"


def test_supported_models_have_replayable_logprob() -> None:
    """Anything marked trainable-in-repo today must have a replayable logprob."""
    for entry in MODEL_SUPPORT_MATRIX:
        if entry.status == "supported":
            assert entry.logprob == "replayable", entry.key


def test_unprobed_models_are_not_supported() -> None:
    """A model with unknown logprob contract must not be marked supported."""
    for entry in MODEL_SUPPORT_MATRIX:
        if entry.logprob == "unknown":
            assert entry.status in ("probe", "track"), entry.key


def test_post_init_rejects_env_without_action() -> None:
    try:
        ModelSupportEntry(
            key="bad",
            display_name="bad",
            params_b=None,
            seam="vla_token",
            inputs=("image",),
            outputs=("action",),
            action_payload=None,
            logprob="unknown",
            env="libero",  # env set but no action payload
            tier="tier2",
            status="probe",
            path=None,
        )
    except ValueError as err:
        assert "env set but no action payload" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
