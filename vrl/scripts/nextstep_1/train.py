"""NextStep-1 online GRPO training recipe."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME


async def train_nextstep_1_ocr_grpo(cfg: DictConfig) -> None:
    """Run NextStep-1 OCR GRPO through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="nextstep_1",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            configure_trainer=_configure_trainer,
            export_modules_getter=_export_modules,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.families.nextstep_1.runtime import (
        build_nextstep_1_runtime_bundle,
        extract_nextstep_1_runtime_spec,
    )

    return build_nextstep_1_runtime_bundle(
        extract_nextstep_1_runtime_spec(cfg, device, weight_dtype),
    )


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.families.nextstep_1.runtime import (
        build_nextstep_1_replay_runtime_bundle,
        extract_nextstep_1_runtime_spec,
    )

    return build_nextstep_1_replay_runtime_bundle(
        extract_nextstep_1_runtime_spec(cfg, device, weight_dtype),
    )


def _configure_trainer(cfg: DictConfig, trainer_config: Any) -> None:
    trainer_config.n = int(cfg.rollout.n_samples_per_prompt)
    trainer_config.rollout_batch_size = int(cfg.rollout.rollout_batch_size)


def _export_modules(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    if bool(cfg.model.use_lora):
        return {LORA_WEIGHTS_NAME: bundle.model.language_model}
    return None


__all__ = ["train_nextstep_1_ocr_grpo"]
