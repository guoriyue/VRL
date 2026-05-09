"""Evaluate SD3.5 OCR checkpoints."""

from __future__ import annotations

from vrl.scripts.eval_common import FamilyEvalDefinition, main_sync

DEFINITION = FamilyEvalDefinition(
    family="sd3_5",
    default_config="experiment/sd3_5_ocr_grpo",
    default_run_dir="outputs/sd3_5_ocr_grpo",
    default_splits=("eval",),
    default_checkpoints=("base", "final"),
    default_weights=("raw", "ema"),
    default_steps=(10, 40),
    step_override_key="num_steps",
    description="Evaluate SD3.5 OCR checkpoints.",
)


if __name__ == "__main__":
    main_sync(DEFINITION)
