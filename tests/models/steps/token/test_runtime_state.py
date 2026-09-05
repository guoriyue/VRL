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


def _paged_state_kwargs() -> dict[str, object]:
    return {
        "token_ids": torch.empty(1, 2, dtype=torch.long),
        "logprobs": torch.empty(1, 2),
        "guidance_scale": 1.0,
        "temperature": 1.0,
        "paged_cond_states": [object()],
        "paged_uncond_states": [object()],
    }
