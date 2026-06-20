"""Algorithm Protocol — advantage computation and policy loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vrl.algorithms.types import TrainStepMetrics

if TYPE_CHECKING:
    from vrl.algorithms.trajectory import AlgorithmInput


class Algorithm(Protocol):
    """Structural interface for RL algorithms (GRPO, REINFORCE, etc.).

    CEA pipeline interface:
    - compute_advantages_from_tensors(rewards, group_ids)
    - compute_loss(inputs)
    """

    # Declarative input contract, read by ``AlgorithmAdapter.validate_inputs``
    # to fail fast — with available-vs-missing diagnostics — when the rollout
    # payload lacks a tensor the loss consumes. Mirrors verl-omni's
    # ``DiffusionLossFn.required_data_keys`` / ``validate_inputs``, and belongs
    # to the same "algorithm self-describes" family as ``uses_evaluator`` /
    # ``tolerates_off_policy_staleness``.
    #
    # required_signal_keys: ``SegmentSignal`` fields the loss reads from the
    #   evaluator replay (signal branch, ``uses_evaluator=True``).
    # required_data_keys: replay-tensor names the loss reads straight off the
    #   rollout batch (replay branch, ``uses_evaluator=False``).
    # Empty by default so an algorithm opts into validation by declaring keys.
    required_signal_keys: tuple[str, ...] = ()
    required_data_keys: tuple[str, ...] = ()

    def compute_advantages_from_tensors(
        self,
        rewards: Any,        # [B] tensor
        group_ids: Any,      # [B] tensor — prompt group assignment
    ) -> Any:                # [B] tensor of advantages
        """Compute per-sample advantages from reward tensors."""
        ...

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute loss from strict trajectory-native algorithm inputs."""
        ...
