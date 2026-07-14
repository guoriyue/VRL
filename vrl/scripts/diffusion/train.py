"""Family-generic online diffusion training recipe.

Registry-descriptor families (those with a ``DiffusionFamilyBuild`` on their
rollout registry entry) train through this single entrypoint instead of a
per-family ``train.py``: family and algorithm both come from config, while the
replay bundle comes from the registry. Cosmos uses this same path; its current
upstream transformers implement the shared activation-checkpointing contract.

YAML wiring: ``trainer.entrypoint: vrl.scripts.diffusion.train:train_diffusion_online``.
"""

from __future__ import annotations

from omegaconf import DictConfig

from vrl.scripts.common.online import run_online_recipe


async def train_diffusion_online(cfg: DictConfig) -> None:
    """Run a config-selected online algorithm for any diffusion family."""

    await run_online_recipe(cfg)


__all__ = ["train_diffusion_online"]
