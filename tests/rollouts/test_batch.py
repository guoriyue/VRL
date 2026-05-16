"""Tests for vrl.rollouts.batch (RolloutBatch stacking)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Regression: Bug 4 — stack_batches helper for group collection
# ---------------------------------------------------------------------------

class TestStackBatches:
    def test_stacks_two_batches(self) -> None:
        """stack_batches concatenates tensor fields along batch dim."""
        import torch

        from vrl.rollouts.batch import RolloutBatch, stack_batches

        b1 = RolloutBatch(
            observations=torch.randn(1, 3, 4),
            actions=torch.randn(1, 3, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            prompts=["hello"],
        )
        b2 = RolloutBatch(
            observations=torch.randn(1, 3, 4),
            actions=torch.randn(1, 3, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([1]),
            prompts=["world"],
        )
        combined = stack_batches([b1, b2])
        assert combined.observations.shape[0] == 2
        assert combined.rewards.shape[0] == 2
        assert combined.group_ids.tolist() == [0, 1]
        assert combined.prompts == ["hello", "world"]

    def test_stacks_tensor_extras(self) -> None:
        """Tensor extras are concatenated; non-tensor extras kept from first."""
        import torch

        from vrl.rollouts.batch import RolloutBatch, stack_batches

        b1 = RolloutBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            extras={
                "debug_tensor": torch.tensor([[0.1, 0.2]]),
                "scheduler": "shared_object",
            },
        )
        b2 = RolloutBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            extras={
                "debug_tensor": torch.tensor([[0.3, 0.4]]),
                "scheduler": "shared_object",
            },
        )
        combined = stack_batches([b1, b2])
        assert combined.extras["debug_tensor"].shape == (2, 2)
        assert combined.extras["scheduler"] == "shared_object"

    def test_single_batch_passthrough(self) -> None:
        """Single batch → returned as-is (no copy overhead)."""
        import torch

        from vrl.rollouts.batch import RolloutBatch, stack_batches

        b = RolloutBatch(
            observations=torch.randn(2, 3, 4),
            actions=torch.randn(2, 3, 4),
            rewards=torch.tensor([1.0, 2.0]),
            dones=torch.tensor([True, True]),
            group_ids=torch.tensor([0, 0]),
        )
        result = stack_batches([b])
        assert result is b  # same object

    def test_context_taken_from_first_batch(self) -> None:
        """context dict is taken from the first batch (not stacked)."""
        import torch

        from vrl.rollouts.batch import RolloutBatch, stack_batches

        ctx = {"guidance_scale": 7.0, "model_family": "cosmos"}
        b1 = RolloutBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            context=ctx,
        )
        b2 = RolloutBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([1]),
            context={"guidance_scale": 7.0, "model_family": "cosmos"},
        )
        combined = stack_batches([b1, b2])
        assert combined.context["guidance_scale"] == 7.0
        assert combined.context["model_family"] == "cosmos"

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

    def test_only_tensor_extras_stacked(self) -> None:
        """Only tensor values in extras get stacked; non-tensors kept from first."""
        import torch

        from vrl.rollouts.batch import RolloutBatch, stack_batches

        b1 = RolloutBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            extras={
                "debug_tensor": torch.tensor([[0.1, 0.2]]),
                "label": "first",
            },
            context={"cfg": True},
        )
        b2 = RolloutBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([1]),
            extras={
                "debug_tensor": torch.tensor([[0.3, 0.4]]),
                "label": "second",
            },
            context={"cfg": True},
        )
        combined = stack_batches([b1, b2])
        # Tensor extras stacked
        assert combined.extras["debug_tensor"].shape == (2, 2)
        # Non-tensor extras kept from first
        assert combined.extras["label"] == "first"
        # Context preserved
        assert combined.context["cfg"] is True

    def test_stack_preserves_trajectory_and_training_view(self) -> None:
        """Trajectory-backed batches keep first-class trajectory metadata when stacked."""
        import torch

        from vrl.engine.trajectory import build_training_view
        from vrl.rollouts.batch import RolloutBatch, stack_batches

        t1 = _trajectory("req-a", torch.tensor([[1, 2]]))
        t2 = _trajectory("req-b", torch.tensor([[3, 4]]))
        view = build_training_view(t1)
        b1 = RolloutBatch(
            observations=torch.ones(1, 1, 2, dtype=torch.long),
            actions=torch.tensor([[1, 2]]),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            trajectory=t1,
            training_view=view,
        )
        b2 = RolloutBatch(
            observations=torch.ones(1, 1, 2, dtype=torch.long),
            actions=torch.tensor([[3, 4]]),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            trajectory=t2,
            training_view=build_training_view(t2),
        )

        combined = stack_batches([b1, b2])

        assert combined.trajectory is not None
        assert combined.training_view == view
        assert combined.trajectory.axes["sample"].length == 2
        assert torch.equal(
            combined.trajectory.segments["image_tokens"].tensors["token_ids"].value,
            torch.tensor([[1, 2], [3, 4]]),
        )


def _trajectory(request_id: str, token_ids):
    import torch

    from vrl.engine import GenerationRequest, GenerationSampleRow
    from vrl.engine.trajectory import build_ar_discrete_trajectory

    request = GenerationRequest(
        request_id=request_id,
        family="janus_pro",
        task="ar_t2i",
        prompts=["p"],
        samples_per_prompt=1,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=0,
            prompt="p",
            prompt_id=f"{request_id}:prompt",
            group_id=f"{request_id}:group",
            sample_id=f"{request_id}:sample",
            trajectory_id=f"{request_id}:trajectory",
            seed=None,
        )
    ]
    return build_ar_discrete_trajectory(
        request=request,
        sample_rows=sample_rows,
        token_ids=token_ids,
        token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
        token_mask=torch.ones_like(token_ids, dtype=torch.float32),
        prompt_input_ids=torch.ones(1, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(1, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(1, 3, dtype=torch.long),
        context={"model_family": "janus_pro"},
    )
