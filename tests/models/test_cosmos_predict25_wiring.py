from __future__ import annotations

from vrl.models.registry import resolve_model
from vrl.rollouts.family_registry import get_rollout_family_entry


def test_cosmos_predict25_model_registry_resolves_without_loading_weights() -> None:
    cls = resolve_model("cosmos-predict2.5")

    assert cls.__name__ == "CosmosPredict25Policy"


def test_cosmos_predict25_family_uses_independent_runtime_paths() -> None:
    entry = get_rollout_family_entry("cosmos-predict2.5")

    assert entry.family == "cosmos-predict2.5"
    assert "predict2_5" in entry.executor_cls
    assert "predict2_5" in entry.runtime_builder
    assert "predict2_5" in entry.runtime_spec_extractor


def test_cosmos_predict2_family_uses_predict2_runtime_paths() -> None:
    entry = get_rollout_family_entry("cosmos-predict2")

    assert entry.family == "cosmos-predict2"
    assert "cosmos.predict2" in entry.executor_cls
    assert "cosmos.predict2" in entry.runtime_builder
    assert "cosmos.predict2" in entry.runtime_spec_extractor
