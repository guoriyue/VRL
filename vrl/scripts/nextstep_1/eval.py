"""Evaluate NextStep-1 checkpoints."""

from __future__ import annotations

from vrl.scripts.eval_common import FamilyEvalDefinition, main_sync

DEFINITION = FamilyEvalDefinition(
    family="nextstep_1",
    default_config="experiment/nextstep_1_ocr_grpo",
    default_run_dir="outputs/nextstep_1_ocr_grpo",
    default_splits=("train",),
    default_checkpoints=("base", "final"),
    default_weights=("raw",),
    default_steps=(),
    step_override_key="num_flow_steps",
    description="Evaluate NextStep-1 checkpoints.",
)


if __name__ == "__main__":
    main_sync(DEFINITION)
