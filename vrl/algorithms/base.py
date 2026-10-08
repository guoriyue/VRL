"""Algorithm Protocol — advantage computation and policy loss."""

from __future__ import annotations

from collections.abc import Callable
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

    # Which schedules an objective is sound under (off-policy staleness, an
    # active trust region) is a config fact: its config class declares it in
    # ``config_contract`` and config resolution enforces it before launch.
    #
    # What a loss reads is the type of its input (``SegmentSignal`` /
    # ``FlowSDESignal`` on the evaluator branch, ``ForwardProcessReplay`` on the
    # replay branch), not a declared key list. Every behavior flag is a required
    # declaration. Root objectives own their values; subclasses inherit only
    # when the family theorem is the same.
    uses_evaluator: bool
    # Weight of the KL term against the reference replay (the trainer then
    # replays the reference policy) and of the clean-target SFT regularizer
    # (the trainer then loads the clean latents); 0.0 for objectives without
    # the term.
    kl_coef: float
    sft_weight: float

    # Lifecycle entry points. The trainer calls both on every objective, the
    # way a scheduler calls a thread's fixed entry points; an objective with
    # nothing to do at one implements it as a no-op.

    def prepare_update(self, update_timesteps: Callable[[], Any]) -> None:
        """Once per optimizer update, before its first replay forward.

        ``update_timesteps()`` returns the recorded timestep of every (sample,
        trained step) the update puts loss on; an objective normalizing over
        the whole update evaluates it, others ignore it.
        """
        ...

    def after_optimizer_step(self, global_step: int) -> None:
        """After every applied (not scaler-skipped) optimizer step."""
        ...

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
