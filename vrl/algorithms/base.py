"""The algorithm base: run facts, config, and the online / offline split."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from vrl.algorithms.advantages import (
    GroupAdvantageConfig,
    _standardize_group_rewards,
    group_relative_advantages,
)
from vrl.algorithms.types import TrainStepMetrics

if TYPE_CHECKING:
    from vrl.algorithms.trajectory import AlgorithmInput


class Algorithm(ABC):
    """An algorithm: what it needs from the run, and its config.

    Every ``algorithm.kind`` names a subclass. Its class attributes below are
    facts of the algorithm's math, not YAML knobs: config resolution reads
    them before launch and the trainer reads them to decide which policies it
    holds. A subclass states the first three and overrides only the facts
    that are true for it. Construction takes the algorithm's resolved config
    and nothing else; the facts an algorithm needs from other config sections
    are bridged into that config by ``BuiltConfigs.from_cfg``.

    The first fork is how the algorithm is trained: ``OnlineAlgorithm`` from
    rollouts the trainer collects, ``OfflineAlgorithm`` from a fixed dataset.
    """

    # uses_evaluator: the loss reads evaluator signals (log-probs from a replay
    #   of the rollout's reverse SDE) rather than replaying the forward process.
    # needs_sde_rollout: the algorithm trains on reverse-SDE transitions, so
    #   generation must run the stochastic sampler (``rollout.sde``).
    # sft_source: where a clean-target SFT term takes its targets, or that
    #   there is no such term.
    # requires_previous_policy: the algorithm scores the trainable policy
    #   against theta_old (the detached current prediction) through the
    #   full-sequence replay forward; such an algorithm cannot train on a
    #   stale sample, so continuous scheduling is rejected for it.
    # requires_reference_policy: the loss reads the pre-training weights
    #   unconditionally, not only behind a KL weight.
    # requires_active_trust_region: the loss is *defined* by a clipped or
    #   guarded ratio against the rollout policy, so it needs a behaviour
    #   policy that differs from the trained one and the stored proposal mean.
    #   Plain GRPO's clip is only a safety rail and leaves this False.
    # consumed_sections: for an offline algorithm, the config sections it
    #   consumes (a listed section allows only the named fields; an empty set
    #   only an empty section; None forbids the section); None = unrestricted.
    uses_evaluator: ClassVar[bool]
    needs_sde_rollout: ClassVar[bool]
    sft_source: ClassVar[Literal["unsupported", "latents", "preference_winner"]]
    requires_previous_policy: ClassVar[bool] = False
    requires_reference_policy: ClassVar[bool] = False
    requires_active_trust_region: ClassVar[bool] = False
    consumed_sections: ClassVar[tuple[tuple[str, frozenset[str] | None], ...] | None] = None

    # Weight of the clean-target SFT term the trainer adds to the loss (it
    # needs a model forward, which algorithms never run). Only an algorithm
    # whose config carries the knob overrides this.
    sft_weight: float = 0.0
    config: Any

    def __init__(self, config: Any) -> None:
        self.config = config


class OnlineAlgorithm(Algorithm):
    """Trains from rollouts: rewards become group-relative advantages, and the
    loss is formed over the trainer's replay of those rollouts.

    The online trainer calls the two lifecycle entry points on every
    algorithm, the way a scheduler calls a thread's fixed entry points; the
    defaults do nothing and an algorithm that normalizes over an update
    overrides them. What a loss reads is the type of its input
    (``SegmentSignal`` / ``FlowSDESignal`` on the evaluator branch,
    ``ForwardProcessReplay`` on the replay branch), not a declared key list.
    """

    config: GroupAdvantageConfig

    # -- lifecycle entry points -------------------------------------------

    def prepare_update(self, update_timesteps: Callable[[], Any]) -> None:
        """Once per optimizer update, before its first replay forward.

        ``update_timesteps()`` returns the recorded timestep of every (sample,
        trained step) the update puts loss on; an algorithm normalizing over
        the whole update evaluates it, others ignore it.
        """

        del update_timesteps

    def after_optimizer_step(self, global_step: int) -> None:
        """After every applied (not scaler-skipped) optimizer step."""

        del global_step

    # -- advantages -----------------------------------------------------------

    def compute_advantages_from_tensors(self, rewards: Any, group_ids: Any) -> Any:
        """Standardize and clamp the weighted reward total within each prompt group."""

        cfg = self.config
        return group_relative_advantages(
            rewards,
            group_ids,
            eps=cfg.eps,
            adv_clip_max=cfg.adv_clip_max,
            global_std=cfg.global_std,
        )

    def compute_advantages_from_components(
        self,
        rewards: Any,
        component_rewards: Mapping[str, Any],
        group_ids: Any,
    ) -> Any:
        """Advantages from the weighted total and its raw component observations.

        Under ``normalized_sum`` each configured objective is standardized in
        its group, weighted by the bridged reward weights and summed, then
        clamped once. A reward also reports observation axes beside its
        objectives (``MultiReward`` namespaces them as ``<component>/<axis>``)
        for logging; those never enter the advantage. With no configured
        objective in hand (a reward that reports only observations) there is
        exactly one objective, the weighted total, so standardizing it is the
        whole strategy. Some but not all configured objectives is a
        misconfiguration.
        """

        import torch

        cfg = self.config
        objectives = {
            name: values
            for name, values in component_rewards.items()
            if name in cfg.component_weights
        }
        if cfg.advantage_combine != "normalized_sum" or not objectives:
            return self.compute_advantages_from_tensors(rewards, group_ids)
        missing = sorted(set(cfg.component_weights) - set(objectives))
        if missing:
            raise ValueError(
                f"reward reports configured components {sorted(objectives)} but not {missing}",
            )

        total = None
        # Stable ordering keeps global-std collectives aligned across DDP ranks.
        for name in sorted(objectives):
            advantage = _standardize_group_rewards(
                objectives[name],
                group_ids,
                eps=cfg.eps,
                global_std=cfg.global_std,
            )
            weighted = cfg.component_weights[name] * advantage
            total = weighted if total is None else total + weighted
        return torch.clamp(total, -cfg.adv_clip_max, cfg.adv_clip_max)

    # -- the loss -------------------------------------------------------------

    @abstractmethod
    def compute_loss(self, inputs: AlgorithmInput) -> tuple[Any, TrainStepMetrics]:
        """Compute loss from strict trajectory-native algorithm inputs."""


class OfflineAlgorithm(Algorithm):
    """Trains from a fixed dataset: no rollout, no evaluator, no rewards.

    The loss is over the model's predictions on a dataset batch, so its
    signature belongs to the concrete algorithm; the offline trainer that
    owns the data pipeline (``vrl/trainers/offline``) calls it directly.
    """

    uses_evaluator = False
    needs_sde_rollout = False


__all__ = ["Algorithm", "OfflineAlgorithm", "OnlineAlgorithm"]
