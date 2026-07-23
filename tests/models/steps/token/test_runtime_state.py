"""CPU contracts for token-loop runtime state ownership."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from vrl.models.steps.token.base import ARDiscreteTokenState
from vrl.models.steps.token.paged_attention_helpers import PagedCFGARState


def test_discrete_token_count_is_derived_from_token_storage() -> None:
    state = ARDiscreteTokenState(
        token_ids=torch.empty(2, 3, dtype=torch.long),
        logprobs=torch.empty(2, 3),
    )

    assert state.total_token_num == 3
    assert "total_token_num" not in {field.name for field in fields(state)}


def test_discrete_token_state_rejects_duplicate_count_input() -> None:
    with pytest.raises(TypeError):
        ARDiscreteTokenState(
            token_ids=torch.empty(1, 2, dtype=torch.long),
            logprobs=torch.empty(1, 2),
            total_token_num=1,
        )


def _paged_state_kwargs() -> dict[str, object]:
    return {
        "token_ids": torch.empty(1, 2, dtype=torch.long),
        "logprobs": torch.empty(1, 2),
        "guidance_scale": 1.0,
        "temperature": 1.0,
        "paged_cond_states": [object()],
        "paged_uncond_states": [object()],
    }


def test_paged_cfg_state_owns_both_branch_states() -> None:
    state = PagedCFGARState(**_paged_state_kwargs())

    assert len(state.paged_cond_states) == 1
    assert len(state.paged_uncond_states) == 1


@pytest.mark.parametrize("missing", ["paged_cond_states", "paged_uncond_states"])
def test_paged_cfg_state_requires_both_branch_states(missing: str) -> None:
    kwargs = _paged_state_kwargs()
    kwargs.pop(missing)

    with pytest.raises(TypeError, match=missing):
        PagedCFGARState(**kwargs)
