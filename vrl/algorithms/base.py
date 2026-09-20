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

    # Behavior declarations the trainer and the factory read once at startup.
    #
    # requires_active_trust_region: the loss is *defined* by a clipped/guarded
    #   importance ratio r = pi_new/pi_old (Flow-DPPO / GRPO-Guard). When True the
    #   trainer refuses strict_on_policy + ppo_epochs==1, where r==1 makes the
    #   trust-region term identically zero (the run degenerates to plain GRPO).
    #   False for objectives whose ratio clip is only a safety rail
    #   (plain GRPO at ppo_epochs=1 is honest REINFORCE-with-group-baseline).
    #   The factory also reads it as "measures drift against the rollout
    #   proposal mean" and requires ``rollout.return_prev_sample_mean``.
    #
    # What a loss reads is the type of its input (``SegmentSignal`` /
    # ``FlowSDESignal`` on the evaluator branch, ``ForwardProcessReplay`` on the
    # replay branch), not a declared key list. Every behavior flag is a required
    # declaration. Root objectives own their values; subclasses inherit only
    # when the family theorem is the same.
    uses_evaluator: bool
    tolerates_off_policy_staleness: bool
    requires_active_trust_region: bool

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
