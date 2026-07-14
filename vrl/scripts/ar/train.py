"""Family-generic AR GRPO training recipe.

AR families train through this single entrypoint instead of a per-family
``train.py`` (the janus_pro / nextstep_1 scripts it replaces were pure
forwarding). ``model.family`` names the exact registry entry; algorithm choice
never rewrites model identity behind the config's back.
The common runner resolves and builds replay through the same registry entry
used by Ray rollout workers; this module carries recipe hooks only.

Mirrors ``vrl/scripts/diffusion/train.py`` (the registry-descriptor diffusion
entrypoint) so both sides read the same way.

YAML wiring: ``trainer.entrypoint: vrl.scripts.ar.train:train_ar_grpo``.
"""

from __future__ import annotations

from omegaconf import DictConfig

from vrl.scripts.common.online import run_online_recipe


async def train_ar_grpo(cfg: DictConfig) -> None:
    """Run token GRPO for any AR family (reward chosen by config)."""

    await run_online_recipe(cfg)


__all__ = ["train_ar_grpo"]
