"""Algorithm Protocol — advantage computation and policy loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
    # ``tolerates_off_policy_staleness`` / ``requires_active_trust_region``.
    #
    # requires_active_trust_region: the loss is *defined* by a clipped/guarded
    #   importance ratio r = pi_new/pi_old (Flow-DPPO / GRPO-Guard). When True the
    #   trainer refuses strict_on_policy + ppo_epochs==1, where r==1 makes the
    #   trust-region term identically zero (the run degenerates to plain GRPO).
    #   False for objectives whose ratio clip is only a safety rail
    #   (plain GRPO at ppo_epochs=1 is honest REINFORCE-with-group-baseline).
    #
    # required_signal_keys: ``SegmentSignal`` fields the loss reads from the
    #   evaluator replay (signal branch, ``uses_evaluator=True``).
    # required_data_keys: replay-tensor names the loss reads straight off the
    #   rollout batch (replay branch, ``uses_evaluator=False``).
    # Every behavior and input field is a required declaration. Root objectives
    # own their values; subclasses inherit only when the family theorem is the
    # same. The consumer contract itself carries no behavioral defaults.
    uses_evaluator: bool
    tolerates_off_policy_staleness: bool
    requires_active_trust_region: bool
    needs_kl_intermediates: bool
    required_signal_keys: tuple[str, ...]
    required_data_keys: tuple[str, ...]

    @property
    def config(self) -> object:
        """Return the objective-specific config read through optional shared knobs."""

        ...

    def compute_advantages_from_tensors(
        self,
        rewards: Any,  # [B] tensor
        group_ids: Any,  # [B] tensor — prompt group assignment
    ) -> Any:  # [B] tensor of advantages
        """Compute per-sample advantages from reward tensors."""
        ...

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute loss from strict trajectory-native algorithm inputs."""
        ...


@runtime_checkable
class ComponentAdvantageAlgorithm(Protocol):
    """Optional algorithm capability for raw reward-component observations."""

    def compute_advantages_from_components(
        self,
        rewards: Any,
        component_rewards: dict[str, Any],
        group_ids: Any,
    ) -> Any:
        """Compute advantages from weighted totals and their raw components."""

        ...
