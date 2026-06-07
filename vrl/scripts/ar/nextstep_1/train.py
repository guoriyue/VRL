"""NextStep-1 online GRPO training recipe."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import (
    configure_ar_rollout,
    export_language_model_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition


async def train_nextstep_1_ocr_grpo(cfg: DictConfig) -> None:
    """Run NextStep-1 OCR GRPO through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="nextstep_1",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            configure_trainer=configure_ar_rollout,
            export_modules_getter=export_language_model_lora,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.ar.nextstep_1.runtime import (
        build_nextstep_1_runtime_bundle,
        extract_nextstep_1_runtime_spec,
    )

    return build_nextstep_1_runtime_bundle(
        extract_nextstep_1_runtime_spec(cfg, device, weight_dtype),
    )


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.ar.nextstep_1.runtime import (
        build_nextstep_1_replay_runtime_bundle,
        extract_nextstep_1_runtime_spec,
    )

    return build_nextstep_1_replay_runtime_bundle(
        extract_nextstep_1_runtime_spec(cfg, device, weight_dtype),
    )


__all__ = ["train_nextstep_1_ocr_grpo"]
