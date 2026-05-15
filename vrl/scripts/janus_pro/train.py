"""Janus-Pro online GRPO training recipes."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME


async def train_janus_pro_grpo(cfg: DictConfig) -> None:
    """Run Janus-Pro token GRPO."""

    await _run_janus_recipe(cfg, family="janus_pro")


async def train_janus_pro_ocr_grpo(cfg: DictConfig) -> None:
    """Run Janus-Pro OCR token GRPO."""

    await _run_janus_recipe(cfg, family="janus_pro")


async def train_janus_pro_r1_ocr_grpo(cfg: DictConfig) -> None:
    """Run Janus-Pro-R1 multi-segment OCR GRPO."""

    await _run_janus_recipe(cfg, family="janus_pro_r1")


async def train_janus_pro_r1_codex_qa_grpo(cfg: DictConfig) -> None:
    """Run Janus-Pro-R1 multi-segment Codex-QA GRPO."""

    await _run_janus_recipe(cfg, family="janus_pro_r1")


async def _run_janus_recipe(cfg: DictConfig, *, family: str) -> None:
    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family=family,
            build_bundle=_build_bundle,
            configure_trainer=_configure_trainer,
            export_modules_getter=_export_modules,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.families.janus_pro.runtime import (
        build_janus_pro_runtime_bundle,
        extract_janus_pro_runtime_spec,
    )

    return build_janus_pro_runtime_bundle(
        extract_janus_pro_runtime_spec(cfg, device, weight_dtype),
    )


def _configure_trainer(cfg: DictConfig, trainer_config: Any) -> None:
    trainer_config.n = int(cfg.rollout.n_samples_per_prompt)
    trainer_config.rollout_batch_size = int(cfg.rollout.rollout_batch_size)


def _export_modules(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    if bool(cfg.model.use_lora):
        return {LORA_WEIGHTS_NAME: bundle.model.language_model}
    return None


__all__ = [
    "train_janus_pro_grpo",
    "train_janus_pro_ocr_grpo",
    "train_janus_pro_r1_codex_qa_grpo",
    "train_janus_pro_r1_ocr_grpo",
]
