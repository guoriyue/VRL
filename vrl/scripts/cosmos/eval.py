"""Evaluate Cosmos Predict2 checkpoints."""

from __future__ import annotations

from vrl.scripts.eval_common import FamilyEvalDefinition, main_sync

DEFINITION = FamilyEvalDefinition(
    family="cosmos",
    default_config="experiment/cosmos_predict2_2b_grpo",
    default_run_dir="outputs/cosmos_pred2_2b_grpo",
    default_splits=("eval",),
    default_checkpoints=("base", "final"),
    default_weights=("raw",),
    default_steps=(),
    step_override_key="num_steps",
    description="Evaluate Cosmos Predict2 checkpoints.",
)


if __name__ == "__main__":
    main_sync(DEFINITION)
