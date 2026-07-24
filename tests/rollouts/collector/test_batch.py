"""Tests for vrl.rollouts.batch (RolloutBatch defaults)."""

from __future__ import annotations

import pytest


class TestRolloutBatch:
    """Groups tests for RolloutBatch construction defaults."""

    def test_context_defaults_to_empty(self) -> None:
        """Without context, field defaults to empty dict."""
        import torch

        from vrl.rollouts.batch import RolloutBatch

        b = RolloutBatch(
            rewards=torch.tensor([1.0]),
            group_ids=torch.tensor([0]),
        )
        assert b.context == {}

    @pytest.mark.parametrize(
        "field_name",
        ["observations", "actions", "dones", "videos", "prompts", "training_view"],
    )
    def test_removed_fields_are_not_constructor_inputs(
        self,
        field_name: str,
    ) -> None:
        import torch

        from vrl.rollouts.batch import RolloutBatch

        values = {
            "rewards": torch.tensor([1.0]),
            "group_ids": torch.tensor([0]),
            field_name: object(),
        }
        with pytest.raises(TypeError, match=field_name):
            RolloutBatch(**values)
