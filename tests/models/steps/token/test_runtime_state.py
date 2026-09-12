"""CPU contracts for token-loop runtime state ownership."""

from __future__ import annotations

import torch

from vrl.models.steps.token.base import ARDiscreteTokenState


def test_discrete_token_count_is_derived_from_token_storage() -> None:
    state = ARDiscreteTokenState(
        token_ids=torch.empty(2, 3, dtype=torch.long),
        logprobs=torch.empty(2, 3),
    )

    assert state.total_token_num == 3
