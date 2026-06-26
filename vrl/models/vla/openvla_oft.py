"""OpenVLA-OFT action policy adapter (Tier 2, eval).

Wraps a loaded OpenVLA-OFT model behind the ``ActionPolicy`` contract so a
closed-loop env rollout can drive it. The heavy model/repo imports (the
``prismatic`` package + ``experiments.robot.*`` helpers from the openvla-oft
checkout) are deferred into method bodies, so this module imports cleanly in a
plain VRL env that only has the P1 contract.

Logprob contract (Tier 2 finding): the released OpenVLA-OFT LIBERO checkpoints
use a **continuous L1-regression action head**, not discrete action-token
sampling — ``predict_action`` is deterministic and exposes no token
distribution. So ``can_replay_logprob`` is False and every ``ActionChunk`` has
``log_prob=None``: this policy is eval-only, and an RL trainer must refuse it.
A replayable action logprob would require the discrete-token base OpenVLA (or
the diffusion-action PI0.5 flow logprob), tracked separately in the sprint.

The model is loaded by the eval entrypoint (which sets up the openvla-oft repo
on ``sys.path``) and injected here, so this adapter stays loader-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vrl.rollouts.envs.contract import ActionChunk, EnvObservation, EnvResetSpec


class OpenVLAOFTPolicy:
    """``ActionPolicy`` over a loaded OpenVLA-OFT model + processor.

    Parameters mirror the openvla-oft eval ``initialize_model`` outputs so the
    adapter can call the repo's own ``get_vla_action`` (faithful run-shape reuse,
    not a re-implementation of OFT decoding).
    """

    def __init__(
        self,
        *,
        cfg: Any,
        model: Any,
        processor: Any,
        resize_size: int,
        action_head: Any | None = None,
        proprio_projector: Any | None = None,
        noisy_action_projector: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._processor = processor
        self._resize_size = resize_size
        self._action_head = action_head
        self._proprio_projector = proprio_projector
        self._noisy_action_projector = noisy_action_projector

    @property
    def action_horizon(self) -> int:
        """Actions returned per inference (open-loop chunk executed by the env)."""
        return int(self._cfg.num_open_loop_steps)

    @property
    def can_replay_logprob(self) -> bool:
        """False: OFT's L1-regression head is deterministic — no action logprob."""
        return False

    def reset(self, spec: EnvResetSpec) -> None:
        """OFT has no per-episode policy state to reset (no KV/action cache here)."""
        return None

    def act(self, obs: EnvObservation) -> ActionChunk:
        """Query the VLA for one action chunk and wrap it in the contract type."""
        # Deferred: pulls the openvla-oft repo (must be on sys.path) + torch.
        from experiments.robot.robot_utils import get_action

        observation = {
            "full_image": obs.images["full_image"],
            "state": obs.state,
        }
        if "wrist_image" in obs.images:
            observation["wrist_image"] = obs.images["wrist_image"]

        actions = get_action(
            self._cfg,
            self._model,
            observation,
            obs.instruction or "",
            processor=self._processor,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            noisy_action_projector=self._noisy_action_projector,
            use_film=self._cfg.use_film,
        )
        chunk = np.asarray(actions, dtype=np.float32)  # [horizon, action_dim]
        return ActionChunk(
            actions=chunk,
            log_prob=None,  # deterministic L1 regression -> eval-only, never faked
            distribution="l1_regression",
            extras={"model_family": self._cfg.model_family},
        )


__all__ = ["OpenVLAOFTPolicy"]
