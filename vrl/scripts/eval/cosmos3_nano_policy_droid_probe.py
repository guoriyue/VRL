"""Probe: Cosmos3-Nano-Policy-DROID action payload + ActionTrajectoryBatch draft.

Tier-1 robotics entry probe (SPRINT_physical_ai_model_support.md §Tier-1 /
§P0). It does NOT train and does NOT run a simulator. It answers:

  1. Can we get an action trajectory from the policy (server or local infer)?
  2. Can that trajectory be packed into the P1 ``ActionTrajectoryBatch`` draft?
  3. What is the DROID embodiment gap vs LIBERO / BEHAVIOR-1K / RoboTwin?

When the real policy is unreachable (no weights / no server — the expected
case today) the probe records the blocker and still exercises the typed seam on
a clearly-labelled SYNTHETIC trajectory, so the contract wiring is validated
end-to-end regardless. A server-level JSON action has no trainable distribution,
so the batch is built eval-only (``log_probs=None``) — the sprint forbids
fabricating a logprob.

Usage:
  python -m vrl.scripts.eval.cosmos3_nano_policy_droid_probe \
    --out outputs/cosmos3_nano_probe/droid_decision.json \
    [--server http://127.0.0.1:8000/act] [--steps 8]
"""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vrl.models.support_matrix import get_support_entry
from vrl.rollouts.envs.contract import (
    ActionChunk,
    ActionTrajectoryBatch,
    EnvObservation,
    EnvResetSpec,
    EnvRewardSignal,
)

_ENTRY_KEY = "cosmos3_nano_policy_droid"

# DROID is a 7-DoF Franka: 6-DoF end-effector delta + 1 gripper. The probe pins
# this so the embodiment gap below is concrete, not hand-wavy.
_DROID_ACTION_DIM = 7
_DROID_ACTION_HORIZON = 16  # Cosmos/PI-style action chunk length

# action_dim per embodiment, used to spell out the gap the sprint asks for.
_EMBODIMENT_ACTION_DIM = {
    "droid": 7,        # Franka 6-DoF EEF delta + gripper
    "libero": 7,       # Franka delta-EEF + gripper (compatible action space)
    "robotwin": 14,    # dual-arm (2x 7-DoF)
    "behavior": 26,    # mobile manipulator (base + arm + gripper), embodiment-heavy
}


def _embodiment_gap() -> dict[str, Any]:
    """Record DROID vs LIBERO/BEHAVIOR/RoboTwin action-space compatibility."""
    droid = _EMBODIMENT_ACTION_DIM["droid"]
    gap = {}
    for name, dim in _EMBODIMENT_ACTION_DIM.items():
        if name == "droid":
            continue
        gap[name] = {
            "action_dim": dim,
            "action_space_compatible": dim == droid,
            "note": (
                "same Franka delta-EEF action space"
                if dim == droid
                else f"different embodiment ({dim}-dim vs DROID {droid}-dim); "
                "policy cannot be run unchanged"
            ),
        }
    return gap


def _query_policy_server(server: str, obs: EnvObservation) -> ActionChunk | None:
    """Best-effort POST one observation to a policy server; None on failure."""
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {"step": obs.step, "instruction": obs.instruction, "state": None},
    ).encode()
    req = urllib.request.Request(
        server, data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    actions = np.asarray(body.get("actions"), dtype=np.float32)
    # Server returns raw JSON actions -> no replayable logprob (eval-only).
    return ActionChunk(actions=actions, log_prob=None, distribution=None)


def _synthetic_chunk(rng: np.random.Generator) -> ActionChunk:
    """A clearly-synthetic eval-only action chunk that matches DROID's shape."""
    actions = rng.normal(0.0, 0.05, size=(_DROID_ACTION_HORIZON, _DROID_ACTION_DIM))
    actions = actions.astype(np.float32)
    return ActionChunk(
        actions=actions,
        log_prob=None,  # server-level JSON => eval-only, never a fabricated logprob
        distribution=None,
        extras={"synthetic": True},
    )


def _build_trajectory_draft(
    report: dict[str, Any], server: str | None, steps: int,
) -> ActionTrajectoryBatch:
    """Roll out one episode's worth of action chunks into the P1 batch draft."""
    rng = np.random.default_rng(0)
    spec = EnvResetSpec(
        task="droid/pick_place",
        seed=0,
        embodiment="droid",
        instruction="pick up the object and place it in the bin",
    )

    chunks: list[ActionChunk] = []
    used_server = False
    for t in range(steps):
        obs = EnvObservation(step=t, instruction=spec.instruction)
        chunk: ActionChunk | None = None
        if server is not None:
            chunk = _query_policy_server(server, obs)
            used_server = used_server or chunk is not None
        if chunk is None:
            chunk = _synthetic_chunk(rng)
        chunks.append(chunk)

    report["action_payload"] = {
        "kind": "action_chunk",
        "source": "policy_server" if used_server else "synthetic_placeholder",
        "horizon": _DROID_ACTION_HORIZON,
        "action_dim": _DROID_ACTION_DIM,
        "has_replayable_logprob": any(c.is_trainable for c in chunks),
    }
    if not used_server:
        report["blockers"].append(
            "policy unreachable (no --server response and no local weights); "
            "trajectory below is SYNTHETIC and only validates the contract wiring",
        )
    report["questions"]["action_output_kind"] = "json_action" if used_server else "unknown"

    # Stack the chunks into the trainer-facing batch (B=1 episode here).
    actions = np.stack([c.actions for c in chunks])[None]  # [1, T, horizon, dim]
    reward = EnvRewardSignal(reward=0.0, success=False, terminal=True)
    return ActionTrajectoryBatch(
        actions=actions,
        log_probs=None,  # eval-only: no trainable distribution from a server policy
        rewards=np.asarray([reward.reward], dtype=np.float32),
        success=np.asarray([reward.success]),
        dones=np.asarray([[t == steps - 1 for t in range(steps)]]),
        group_ids=np.zeros((1,), dtype=np.int64),
        episode_lengths=np.asarray([steps], dtype=np.int64),
        extras={"embodiment": spec.embodiment, "task": spec.task},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    entry = get_support_entry(_ENTRY_KEY)
    parser.add_argument(
        "--server",
        default=None,
        help="optional policy-server /act endpoint to query for actions",
    )
    parser.add_argument("--steps", type=int, default=8, help="control steps to roll")
    parser.add_argument(
        "--out",
        default="outputs/cosmos3_nano_probe/droid_decision.json",
        help="where to write the decision note JSON",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "probe": "cosmos3_nano_policy_droid",
        "matrix_entry": asdict(entry),
        "questions": {
            "action_trajectory_obtainable": "unknown",
            "packs_into_action_trajectory_batch": "unknown",
            "action_output_kind": "unknown",
        },
        "action_payload": {},
        "embodiment_gap": _embodiment_gap(),
        "blockers": [],
        "decision": "undecided",
        "env": {"python": platform.python_version(), "numpy": np.__version__},
    }

    batch = _build_trajectory_draft(report, args.server, args.steps)

    # Validate the contract actually represents the trajectory.
    report["questions"]["action_trajectory_obtainable"] = (
        "yes_server" if report["action_payload"]["source"] == "policy_server" else "synthetic_only"
    )
    report["questions"]["packs_into_action_trajectory_batch"] = "yes"
    report["action_trajectory_batch"] = {
        "actions_shape": list(batch.actions.shape),
        "is_trainable": batch.is_trainable,  # False: eval-only
        "episode_lengths": batch.episode_lengths.tolist(),
        "extras": batch.extras,
    }
    # Eval-only contract is a clean, defensible outcome (the sprint's "不能训练
    # 也算成功"): the seam holds, training is correctly gated off.
    report["decision"] = (
        "positive_eval_only"
        if report["action_payload"]["source"] == "policy_server"
        else "contract_validated_synthetic"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[decision] {report['decision']}")
    for blocker in report["blockers"]:
        print(f"  blocker: {blocker}")
    print(f"[note] {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
