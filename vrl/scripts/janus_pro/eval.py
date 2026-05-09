"""Evaluate Janus-Pro checkpoints."""

from __future__ import annotations

from vrl.scripts.eval_common import FamilyEvalDefinition, main_sync

DEFINITION = FamilyEvalDefinition(
    family="janus_pro",
    default_config="experiment/janus_pro_1b_ocr_grpo",
    default_run_dir="outputs/janus_pro_1b_ocr_grpo",
    default_splits=("train",),
    default_checkpoints=("base", "final"),
    default_weights=("raw",),
    default_steps=(),
    step_override_key="image_token_num",
    description="Evaluate Janus-Pro checkpoints.",
)


if __name__ == "__main__":
    main_sync(DEFINITION)
