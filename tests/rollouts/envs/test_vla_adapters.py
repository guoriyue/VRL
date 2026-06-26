"""LIBERO env + OpenVLA-OFT policy adapters — protocol conformance (CPU).

The adapters defer all heavy imports (libero, prismatic, torch) into method
bodies, so they must construct and satisfy the Env / ActionPolicy protocols
with no simulator or model present. This pins that contract without GPU.
The actual model+sim rollout is exercised by
``vrl/scripts/eval/openvla_oft_libero_eval.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

from vrl.models.vla import ActionPolicy
from vrl.models.vla.openvla_oft import OpenVLAOFTPolicy
from vrl.rollouts.envs import Env
from vrl.rollouts.envs.libero import LiberoEnv


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(num_open_loop_steps=8, use_film=False, model_family="openvla")


def test_openvla_oft_policy_satisfies_protocol() -> None:
    policy = OpenVLAOFTPolicy(
        cfg=_cfg(), model=object(), processor=object(), resize_size=224,
    )
    assert isinstance(policy, ActionPolicy)
    assert policy.action_horizon == 8
    # OFT's L1-regression head is deterministic -> eval-only, no replay logprob.
    assert policy.can_replay_logprob is False
    assert policy.reset(SimpleNamespace(task="t")) is None


def test_libero_env_satisfies_protocol() -> None:
    env = LiberoEnv(
        env=object(), task_description="put the bowl in the drawer", resize_size=224,
    )
    assert isinstance(env, Env)
    assert env.action_dim == 7


def test_libero_env_render_before_step_is_empty() -> None:
    env = LiberoEnv(env=object(), task_description="t", resize_size=224)
    artifact = env.render_episode()
    assert artifact.video is None
    assert artifact.info["steps"] == 0
