"""LIBERO closed-loop environment adapter (Tier 2).

Wraps a LIBERO ``OffScreenRenderEnv`` behind the ``Env`` contract. LIBERO is the
sprint's first lightweight closed-loop robot env (MuJoCo, long-horizon Franka
manipulation). The environment is the sole owner of success/reward — the policy
never invents one.

All heavy imports (``libero``, the openvla-oft ``experiments.robot.*`` image
helpers, torch) are deferred into method bodies so this module imports in a
plain VRL env. The underlying LIBERO env is built + injected by the eval
entrypoint (which sets ``MUJOCO_GL=egl`` and puts the repos on ``sys.path``),
keeping this adapter loader-agnostic.

Chunked control: a policy emits an ``ActionChunk`` of ``num_open_loop_steps``
actions; ``step`` executes the whole chunk open-loop (matching the openvla-oft
LIBERO eval loop) and returns the resulting observation + an env reward signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vrl.rollouts.envs.contract import (
    ActionChunk,
    EnvObservation,
    EnvResetSpec,
    EnvRewardSignal,
    EpisodeArtifact,
)


class LiberoEnv:
    """``Env`` over a LIBERO ``OffScreenRenderEnv`` for one task."""

    def __init__(
        self,
        *,
        env: Any,
        task_description: str,
        resize_size: int,
        model_family: str = "openvla",
        num_steps_wait: int = 10,
    ) -> None:
        self._env = env
        self._task_description = task_description
        self._resize_size = resize_size
        self._model_family = model_family
        self._num_steps_wait = num_steps_wait
        self._t = 0
        self._raw_obs: Any = None
        self._replay_frames: list[np.ndarray] = []

    @property
    def action_dim(self) -> int:
        return 7  # LIBERO Franka: 6-DoF EEF delta + gripper

    def reset(self, spec: EnvResetSpec) -> EnvObservation:
        """Reset the task, optionally set an initial state, settle the scene."""
        from experiments.robot.libero.libero_utils import get_libero_dummy_action

        self._env.reset()
        initial_state = spec.options.get("initial_state") if spec.options else None
        if initial_state is not None:
            self._raw_obs = self._env.set_init_state(initial_state)
        else:
            self._raw_obs = self._env.get_observation()
        self._t = 0
        self._replay_frames = []

        # Let objects settle before the policy acts (sim physics warm-up).
        dummy = get_libero_dummy_action(self._model_family)
        for _ in range(self._num_steps_wait):
            self._raw_obs, _, _, _ = self._env.step(dummy)
        return self._observe()

    def step(self, action: ActionChunk) -> tuple[EnvObservation, EnvRewardSignal]:
        """Execute an action chunk open-loop; return next obs + env reward."""
        from experiments.robot.robot_utils import (
            invert_gripper_action,
            normalize_gripper_action,
        )

        chunk = np.asarray(action.actions, dtype=np.float32)
        total_reward = 0.0
        success = False
        for single in chunk:
            processed = normalize_gripper_action(single, binarize=True)
            if self._model_family == "openvla":
                processed = invert_gripper_action(processed)
            self._raw_obs, reward, done, _ = self._env.step(processed.tolist())
            total_reward += float(reward)
            self._t += 1
            if done:
                success = True
                break

        obs = self._observe(done=success)
        signal = EnvRewardSignal(
            reward=total_reward,
            success=success,
            terminal=success,
            components={"env_reward": total_reward},
        )
        return obs, signal

    def render_episode(self) -> EpisodeArtifact:
        """Return the collected RGB frames as a [T, H, W, C] episode video."""
        video = np.stack(self._replay_frames) if self._replay_frames else None
        return EpisodeArtifact(video=video, info={"steps": self._t})

    def _observe(self, *, done: bool = False) -> EnvObservation:
        """Build an EnvObservation from the raw LIBERO obs (policy-ready images)."""
        from experiments.robot.libero.libero_utils import (
            get_libero_image,
            get_libero_wrist_image,
            quat2axisangle,
        )
        from experiments.robot.openvla_utils import resize_image_for_policy

        img = get_libero_image(self._raw_obs)
        wrist = get_libero_wrist_image(self._raw_obs)
        self._replay_frames.append(img)  # full-res frame for episode video

        state = np.concatenate(
            (
                self._raw_obs["robot0_eef_pos"],
                quat2axisangle(self._raw_obs["robot0_eef_quat"]),
                self._raw_obs["robot0_gripper_qpos"],
            )
        )
        return EnvObservation(
            step=self._t,
            images={
                "full_image": resize_image_for_policy(img, self._resize_size),
                "wrist_image": resize_image_for_policy(wrist, self._resize_size),
            },
            state=state,
            instruction=self._task_description,
            done=done,
        )


__all__ = ["LiberoEnv"]
