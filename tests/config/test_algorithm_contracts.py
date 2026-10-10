"""Algorithm-owned facts control validation without becoming YAML knobs."""

from dataclasses import asdict
from typing import get_args

import pytest

from vrl.algorithms.v_grpo import VGRPO
from vrl.config.algorithm import algorithm_class, algorithm_config_class
from vrl.config.schema import AlgorithmConfig, RootConfig

_FACTS = (
    "needs_sde_rollout",
    "sft_source",
    "requires_previous_policy",
    "requires_reference_policy",
    "requires_active_trust_region",
    "consumed_sections",
)


@pytest.mark.parametrize("kind", get_args(AlgorithmConfig.model_fields["kind"].annotation))
def test_every_algorithm_declares_non_configurable_facts(kind: str) -> None:
    config_cls = algorithm_config_class(kind)
    algorithm_cls = algorithm_class(kind)
    for name in _FACTS:
        getattr(algorithm_cls, name)
        assert name not in asdict(config_cls())
        with pytest.raises(ValueError, match=rf"unknown algorithm.{name}"):
            AlgorithmConfig.model_validate({"kind": kind, name: True})


def test_rules_follow_declared_facts_without_changing_kind(monkeypatch) -> None:
    payload = {"algorithm": {"kind": "v_grpo"}}
    RootConfig.model_validate(payload)
    monkeypatch.setattr(VGRPO, "needs_sde_rollout", True)
    with pytest.raises(ValueError, match=r"rollout.sde.type"):
        RootConfig.model_validate(payload)


def test_offline_surface_is_owned_by_algorithm_config(monkeypatch) -> None:
    from vrl.algorithms.dpo import DiffusionDPO

    payload = {
        "algorithm": {"kind": "diffusion_dpo"},
        "trainer": {"seed": 123},
    }
    with pytest.raises(ValueError, match=r"trainer.seed"):
        RootConfig.model_validate(payload)
    monkeypatch.setattr(
        DiffusionDPO,
        "consumed_sections",
        tuple(
            (name, allowed | {"seed"} if name == "trainer" else allowed)
            for name, allowed in DiffusionDPO.consumed_sections
        ),
    )
    RootConfig.model_validate(payload)


@pytest.mark.parametrize("reward", [{"components": {}}, {"components": {"test": 1.0}}])
def test_section_presence_restriction_is_declared_independently_of_fields(
    monkeypatch, reward: dict
) -> None:
    from vrl.algorithms.dpo import DiffusionDPO

    payload = {"algorithm": {"kind": "diffusion_dpo"}, "reward": reward}
    with pytest.raises(ValueError, match="does not consume the reward config section"):
        RootConfig.model_validate(payload)

    monkeypatch.setattr(
        DiffusionDPO,
        "consumed_sections",
        tuple(
            (name, allowed) for name, allowed in DiffusionDPO.consumed_sections if name != "reward"
        ),
    )
    RootConfig.model_validate(payload)
