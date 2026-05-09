"""Evaluate Wan 2.1 checkpoints."""

from __future__ import annotations

from vrl.scripts.eval_common import FamilyEvalDefinition, main_sync

DEFINITION = FamilyEvalDefinition(
    family="wan_2_1",
    default_config="experiment/wan_2_1_1_3b_ocr_grpo",
    default_run_dir="outputs/wan_1_3b_ocr_grpo",
    default_splits=("train",),
    default_checkpoints=("base", "final"),
    default_weights=("raw",),
    default_steps=(),
    step_override_key="num_steps",
    description="Evaluate Wan 2.1 checkpoints.",
)


if __name__ == "__main__":
    main_sync(DEFINITION)
