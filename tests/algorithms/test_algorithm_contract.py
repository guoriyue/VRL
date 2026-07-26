"""Algorithm implementation ownership and structural protocol boundaries."""

from __future__ import annotations

from typing import get_type_hints

import pytest

from vrl.algorithms.base import Algorithm
from vrl.algorithms.diffusion_nft import DiffusionNFT
from vrl.algorithms.grpo.continuous import GRPO
from vrl.algorithms.trajectory import AlgorithmAdapter, AlgorithmInput


class _MissingInputContract:
    def compute_advantages_from_tensors(self, rewards, group_ids):
        return rewards

    def compute_loss(self, inputs):
        return inputs


def _algorithms() -> tuple[Algorithm, ...]:
    return (GRPO(), DiffusionNFT())


@pytest.mark.parametrize("algorithm", _algorithms(), ids=lambda value: type(value).__name__)
def test_concrete_algorithm_exposes_protocol_members_without_nominal_inheritance(
    algorithm: Algorithm,
) -> None:
    contract_members = set(get_type_hints(Algorithm))
    contract_members.update(
        name
        for name, member in vars(Algorithm).items()
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    )

    assert Algorithm not in type(algorithm).__mro__
    assert not contract_members.difference(dir(algorithm))


def test_algorithm_contract_carries_no_behavior_defaults() -> None:
    declarations = get_type_hints(Algorithm)

    assert declarations["required_signal_keys"] == tuple[str, ...]
    assert declarations["required_data_keys"] == tuple[str, ...]
    assert all(name not in vars(Algorithm) for name in declarations)
    assert isinstance(Algorithm.config, property)


def test_adapter_does_not_default_a_missing_input_contract_to_empty() -> None:
    with pytest.raises(AttributeError, match="required_signal_keys"):
        AlgorithmAdapter().validate_inputs(_MissingInputContract(), AlgorithmInput())


def test_root_algorithms_own_behavior_and_input_contract_values() -> None:
    assert GRPO.__dict__["uses_evaluator"] is True
    assert GRPO.__dict__["tolerates_off_policy_staleness"] is True
    assert GRPO.__dict__["requires_active_trust_region"] is False
    assert GRPO.__dict__["needs_kl_intermediates"] is False
    assert GRPO.__dict__["required_signal_keys"] == ("log_prob", "old_log_prob")
    assert GRPO.__dict__["required_data_keys"] == ()

    assert DiffusionNFT.__dict__["uses_evaluator"] is False
    assert DiffusionNFT.__dict__["tolerates_off_policy_staleness"] is False
    assert DiffusionNFT.__dict__["requires_active_trust_region"] is False
    assert DiffusionNFT.__dict__["needs_kl_intermediates"] is False
    assert DiffusionNFT.__dict__["required_signal_keys"] == ()
    assert DiffusionNFT.__dict__["required_data_keys"] == (
        "latents_clean",
        "prompt_embeds",
        "timesteps",
    )
