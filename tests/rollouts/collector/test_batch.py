"""Tests for vrl.rollouts.batch (RolloutBatch defaults)."""

from __future__ import annotations


class TestRolloutBatch:
    """Groups tests for RolloutBatch construction defaults."""

    def test_context_defaults_to_empty(self) -> None:
        """Without context, field defaults to empty dict."""
        import torch

        from vrl.rollouts.batch import RolloutBatch

        b = RolloutBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
        )
        assert b.context == {}
