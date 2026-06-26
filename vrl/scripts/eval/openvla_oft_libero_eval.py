"""Real Tier-2 eval: OpenVLA-OFT on LIBERO-10 through the VRL P1 contract.

Loads the matched moojink OpenVLA-OFT LIBERO-10 checkpoint + the LIBERO sim and
drives a closed-loop rollout through the vrl ``Env`` / ``ActionPolicy`` /
``ActionTrajectoryBatch`` contract, recording success / episode length / video /
and a per-episode batch. This is the first real model+simulator validation of
the P1 seam (SPRINT_physical_ai_model_support.md §Tier-2 / §P2).

The released OFT checkpoint uses a deterministic L1-regression action head, so
the produced batch is eval-only (``log_probs=None``) — an honest "action logprob
not available for OFT" conclusion, not a fabricated distribution.

Requires an env with: cu128 torch (RTX 5090 / sm_120), the openvla-oft checkout
(``prismatic`` + ``experiments.robot.*``) and LIBERO on ``sys.path``, and the
VRL repo importable. The script wires both repos onto ``sys.path`` from
``--openvla-root`` so it can be launched from anywhere.

Usage:
  MUJOCO_GL=egl python -m vrl.scripts.eval.openvla_oft_libero_eval \
    --checkpoint ~/Desktop/vla_models/openvla-7b-oft-finetuned-libero-10 \
    --task-suite libero_10 --tasks 2 --trials 2 --max-steps 400 \
    --out outputs/openvla_oft_libero/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# MuJoCo must use EGL (headless GPU rendering) — set before any libero import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class _LenientStub:
    """Returns itself for any non-dunder attribute / call / index.

    Lets import-time references (``tf.config.set_visible_devices(...)``, type
    annotations, decorators) resolve to a harmless no-op. Dunders raise
    ``AttributeError`` so ``inspect``/``dataclasses`` introspection still falls
    back correctly instead of receiving a stub for ``__file__`` etc.
    """

    def __getattr__(self, name: str) -> _LenientStub:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> _LenientStub:
        return self

    def __getitem__(self, _key: Any) -> _LenientStub:
        return self


_STUB = _LenientStub()

# Optional deps the openvla-oft eval import path pulls in but evaluation never
# executes: the RLDS *training* data loader (``prismatic.vla.datasets`` ->
# ``dlimp`` / ``tensorflow_datasets`` / ``tensorflow_graphics``) and robosuite's
# teleop input backend (``pynput``, whose native ``evdev`` build is broken under
# the conda toolchain). We stub these to avoid heavy/broken installs.
#
# NOTE: real ``tensorflow`` is deliberately NOT stubbed — ``resize_image_for_policy``
# uses ``tf.image.resize`` (lanczos3 + JPEG round-trip) at INFERENCE time to match
# the training data pipeline, so stubbing it would shift the input distribution
# and corrupt success rates.
_STUB_ROOTS = frozenset(
    {"tensorflow_datasets", "tensorflow_graphics", "dlimp", "pynput"}
)


def _stub_unused_native_modules() -> None:
    """Install a meta-path finder faking any submodule under ``_STUB_ROOTS``.

    A finder (vs. enumerating modules) handles arbitrarily deep imports like
    ``import tensorflow_graphics.geometry.transformation`` without listing each
    one. Imported before the heavy openvla/libero imports.
    """
    import importlib.abc
    import importlib.machinery
    import types

    def _module_getattr(name: str) -> _LenientStub:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _STUB

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec: Any) -> Any:
            module = types.ModuleType(spec.name)
            module.__path__ = []  # mark as package so submodule imports resolve
            module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
            return module

        def exec_module(self, module: Any) -> None:
            return None

    class _StubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
            if name.split(".")[0] in _STUB_ROOTS:
                return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
            return None

    sys.meta_path.insert(0, _StubFinder())


def _patch_torch_load_for_libero() -> None:
    """Default ``torch.load`` to ``weights_only=False`` for LIBERO init states.

    torch >= 2.6 flipped ``weights_only`` to True by default; LIBERO's pickled
    per-task init-state files (numpy arrays) predate that and fail to unpickle
    under the new default. The files are trusted local sim assets, so we restore
    the old behavior. Scoped to this eval process only.
    """
    import torch

    original_load = torch.load

    def _load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load


def _build_cfg(args: argparse.Namespace) -> Any:
    """Construct the openvla-oft GenerateConfig (defaults match moojink ckpts)."""
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    # The repo's defaults (L1 head, 2 images, proprio, center-crop, chunk=8) are
    # exactly what the released LIBERO OFT checkpoints expect; only point it at
    # our checkpoint + suite.
    return GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite,
        center_crop=True,
        use_wandb=False,
        seed=args.seed,
    )


def _run_episode(env: Any, policy: Any, spec: Any, max_chunks: int) -> dict[str, Any]:
    """Drive one closed-loop episode through the contract; return its record."""
    from vrl.rollouts.envs.contract import ActionTrajectoryBatch

    policy.reset(spec)
    obs = env.reset(spec)

    chunks: list[np.ndarray] = []
    total_reward = 0.0
    success = False
    n = 0
    while n < max_chunks and not obs.done:
        chunk = policy.act(obs)
        chunks.append(np.asarray(chunk.actions, dtype=np.float32))
        obs, signal = env.step(chunk)
        total_reward += signal.reward
        n += 1
        if signal.terminal:
            success = signal.success
            break

    actions = np.stack(chunks)[None] if chunks else np.zeros((1, 0, 0, 7), np.float32)
    batch = ActionTrajectoryBatch(
        observations=np.zeros((1, len(chunks)), dtype=np.float32),
        actions=actions,  # [1, n_chunks, horizon, action_dim]
        log_probs=None,   # OFT L1 regression: eval-only, no replayable logprob
        rewards=np.asarray([total_reward], dtype=np.float32),
        success=np.asarray([success]),
        dones=np.asarray([[i == len(chunks) - 1 for i in range(len(chunks))]]),
        group_ids=np.asarray([spec.options.get("task_id", 0)], dtype=np.int64),
        episode_lengths=np.asarray([len(chunks)], dtype=np.int64),
        distribution="l1_regression",
        extras={"task": spec.task},
    )
    return {
        "success": bool(success),
        "chunks": len(chunks),
        "reward": round(total_reward, 3),
        "is_trainable": batch.is_trainable,
        "actions_shape": list(batch.actions.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="local path or HF id of the OFT ckpt")
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--tasks", type=int, default=2, help="number of tasks to eval")
    parser.add_argument("--trials", type=int, default=2, help="trials per task")
    parser.add_argument("--max-steps", type=int, default=400, help="env-step cap per episode")
    parser.add_argument("--openvla-root", default=str(Path.home() / "Desktop/openvla-oft"))
    parser.add_argument("--libero-root", default=str(Path.home() / "Desktop/LIBERO"))
    parser.add_argument("--out", default="outputs/openvla_oft_libero/eval.json")
    parser.add_argument("--video-dir", default="outputs/openvla_oft_libero/videos")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # LIBERO is an implicit namespace package (no top-level __init__), so it must
    # be on sys.path directly; the PEP-660 editable install registers nothing.
    sys.path.insert(0, args.libero_root)
    sys.path.insert(0, args.openvla_root)
    _stub_unused_native_modules()

    report: dict[str, Any] = {
        "eval": "openvla_oft_libero",
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "episodes": [],
        "blockers": [],
    }

    try:
        from experiments.robot.libero.libero_utils import get_libero_env, save_rollout_video
        from experiments.robot.libero.run_libero_eval import initialize_model
        from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
        from libero.libero import benchmark

        from vrl.models.vla.openvla_oft import OpenVLAOFTPolicy
        from vrl.rollouts.envs.contract import EnvResetSpec
        from vrl.rollouts.envs.libero import LiberoEnv

        set_seed_everywhere(args.seed)
        _patch_torch_load_for_libero()
        cfg = _build_cfg(args)
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)
        horizon = int(cfg.num_open_loop_steps)
        max_chunks = max(1, math.ceil(args.max_steps / horizon))

        suite = benchmark.get_benchmark_dict()[args.task_suite]()
        n_tasks = min(args.tasks, suite.n_tasks)
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

        successes = 0
        total = 0
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            libero_env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
            policy = OpenVLAOFTPolicy(
                cfg=cfg, model=model, processor=processor, resize_size=resize_size,
                action_head=action_head, proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
            )
            for trial in range(args.trials):
                env = LiberoEnv(
                    env=libero_env, task_description=task_description,
                    resize_size=resize_size, model_family=cfg.model_family,
                )
                spec = EnvResetSpec(
                    task=f"{args.task_suite}/{task_description}",
                    seed=args.seed + trial,
                    embodiment="libero-franka",
                    instruction=task_description,
                    options={"initial_state": init_states[trial % len(init_states)], "task_id": task_id},
                )
                rec = _run_episode(env, policy, spec, max_chunks)
                rec.update({"task_id": task_id, "trial": trial, "task": task_description})
                report["episodes"].append(rec)
                successes += int(rec["success"])
                total += 1
                artifact = env.render_episode()
                if artifact.video is not None:
                    save_rollout_video(
                        artifact.video, total, success=rec["success"],
                        task_description=task_description, log_file=None,
                    )
                print(f"[task {task_id} trial {trial}] success={rec['success']} "
                      f"chunks={rec['chunks']} reward={rec['reward']}", flush=True)

        report["success_rate"] = round(successes / total, 4) if total else None
        report["n_episodes"] = total
        print(f"\n[SUCCESS RATE] {successes}/{total} = {report['success_rate']}")
    except Exception as err:  # record any blocker; this is a probe-style eval
        report["blockers"].append(f"{type(err).__name__}: {err}")
        traceback.print_exc()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[note] {out}")


if __name__ == "__main__":
    main()
