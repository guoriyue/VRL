"""Algorithm-owned facts control validation without becoming YAML knobs."""

from dataclasses import asdict, replace
from typing import get_args

import pytest

from vrl.algorithms.requirements import AlgorithmRequirements
from vrl.algorithms.v_grpo import VGRPOConfig
from vrl.config.algorithm import algorithm_config_class
from vrl.config.schema import AlgorithmConfig, RootConfig


@pytest.mark.parametrize("kind", get_args(AlgorithmConfig.model_fields["kind"].annotation))
def test_every_algorithm_declares_non_configurable_facts(kind: str) -> None:
    config = algorithm_config_class(kind)()
    assert isinstance(config.requirements, AlgorithmRequirements)
    assert "requirements" not in asdict(config)
    with pytest.raises(ValueError, match=r"unknown algorithm.requirements"):
        AlgorithmConfig.model_validate({"kind": kind, "requirements": {}})


@pytest.mark.parametrize(
    ("changes", "algorithm_fields", "error"),
    [
        ({"needs_sde_rollout": True}, {}, r"rollout.sde.type"),
    ],
)
def test_rules_follow_declared_facts_without_changing_kind(
    monkeypatch, changes: dict, algorithm_fields: dict, error: str
) -> None:
    payload = {"algorithm": {"kind": "v_grpo", **algorithm_fields}}
    RootConfig.model_validate(payload)
    monkeypatch.setattr(VGRPOConfig, "requirements", replace(VGRPOConfig.requirements, **changes))
    with pytest.raises(ValueError, match=error):
        RootConfig.model_validate(payload)


def test_offline_surface_is_owned_by_algorithm_config(monkeypatch) -> None:
    from vrl.algorithms.dpo import DiffusionDPOConfig

    payload = {
        "algorithm": {"kind": "diffusion_dpo"},
        "trainer": {"seed": 123},
    }
    with pytest.raises(ValueError, match=r"trainer.seed"):
        RootConfig.model_validate(payload)
    requirements = DiffusionDPOConfig.requirements
    monkeypatch.setattr(
        DiffusionDPOConfig,
        "requirements",
        replace(
            requirements,
            consumed_sections=tuple(
                (name, allowed | {"seed"} if name == "trainer" else allowed)
                for name, allowed in requirements.consumed_sections
            ),
        ),
    )
    RootConfig.model_validate(payload)


@pytest.mark.parametrize("reward", [{"components": {}}, {"components": {"test": 1.0}}])
def test_section_presence_restriction_is_declared_independently_of_fields(
    monkeypatch, reward: dict
) -> None:
    from vrl.algorithms.dpo import DiffusionDPOConfig

    payload = {"algorithm": {"kind": "diffusion_dpo"}, "reward": reward}
    with pytest.raises(ValueError, match="does not consume the reward config section"):
        RootConfig.model_validate(payload)

    requirements = DiffusionDPOConfig.requirements
    monkeypatch.setattr(
        DiffusionDPOConfig,
        "requirements",
        replace(
            requirements,
            consumed_sections=tuple(
                (name, allowed)
                for name, allowed in requirements.consumed_sections
                if name != "reward"
            ),
        ),
    )
    RootConfig.model_validate(payload)
