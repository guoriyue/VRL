"""Online trainer configuration ownership tests."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import ActorSection, TrainerSection, parse_config
from vrl.trainers.core.types import DebugConfig, OptimConfig, ReplayParityConfig
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig


def test_trainer_config_public_facade_shares_online_owner() -> None:
    from vrl.trainers.online import OnlineBatchPlan as online_plan
    from vrl.trainers.online import TrainerConfig as online_export
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    assert online_export is TrainerConfig
    assert online_plan is OnlineBatchPlan


def test_config_contracts_import_without_torch() -> None:
    """The online facade must reach TrainerConfig without the trainer runtime.

    ``vrl.config.schema`` imports TrainerConfig to validate every recipe, so a
    torch-heavy path here taxes all config parsing. Run in a subprocess because
    the in-process test session has already imported torch.
    """

    probe = (
        "import sys; from vrl.trainers.online import TrainerConfig; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_core_package_does_not_claim_online_trainer_config() -> None:
    import vrl.trainers.core as core

    assert "TrainerConfig" not in core.__all__
    assert not hasattr(core, "TrainerConfig")


def test_trainer_config_does_not_mirror_controller_lifecycle() -> None:
    trainer_fields = {trainer_field.name for trainer_field in fields(TrainerConfig)}

    assert trainer_fields.isdisjoint({"total_epochs", "save_freq", "seed"})


def test_replay_parity_is_a_correctness_config_not_a_debug_toggle() -> None:
    trainer_fields = {trainer_field.name for trainer_field in fields(TrainerConfig)}

    assert [debug_field.name for debug_field in fields(DebugConfig)] == ["first_step"]
    assert "replay_parity" in trainer_fields
    assert "replay_parity" in TrainerSection.model_fields
    assert ReplayParityConfig().max_abs_logprob_diff == pytest.approx(0.01)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_replay_parity_rejects_invalid_limits(limit: float) -> None:
    with pytest.raises(ValueError, match=r"trainer\.replay_parity\.max_abs_logprob_diff"):
        ReplayParityConfig(max_abs_logprob_diff=limit)


def test_trainer_config_fields_are_declared_by_exactly_one_public_section() -> None:
    """The projection reads each field from the section that declares its name."""
    bridged = {"batch_plan", "train_precision", "rollout_precision"}
    for trainer_field in fields(TrainerConfig):
        if trainer_field.name in bridged:
            continue
        owners = [
            section
            for section in (ActorSection, TrainerSection)
            if trainer_field.name in section.model_fields
        ]
        assert len(owners) == 1, trainer_field.name
    for plan_field in fields(OnlineBatchPlan):
        if plan_field.name in ("prompts_per_batch", "n_samples_per_prompt"):
            continue
        assert plan_field.name in ActorSection.model_fields, plan_field.name


def _public_batch_config(
    *,
    prompts_per_batch: object = 32,
    n_samples_per_prompt: object = 2,
    **actor: object,
):
    return parse_config(
        OmegaConf.create(
            {
                "rollout": {
                    "prompts_per_batch": prompts_per_batch,
                    "n_samples_per_prompt": n_samples_per_prompt,
                },
                "actor": actor,
            },
        ),
    )


def _trainer_config(batch_plan: OnlineBatchPlan, *, ppo_epochs: int = 1) -> TrainerConfig:
    return TrainerConfig(
        optim=OptimConfig(lr=1e-4),
        batch_plan=batch_plan,
        timestep_fraction=0.5,
        output_dir="x",
        drop_zero_advantage=False,
        ppo_epochs=ppo_epochs,
    )


def test_batch_plan_resolves_size_and_count_to_the_same_state() -> None:
    size_only = OnlineBatchPlan.from_root(_public_batch_config(microbatch_size=4))
    count_only = OnlineBatchPlan.from_root(
        _public_batch_config(gradient_accumulation_steps=8),
    )
    both = OnlineBatchPlan.from_root(
        _public_batch_config(
            microbatch_size=4,
            gradient_accumulation_steps=8,
        ),
    )

    assert size_only == count_only == both
    assert size_only.gradient_accumulation_steps == 8
    assert size_only.microbatch_size == 4
    assert size_only.streaming is True


def test_unsplit_batch_plan_derives_the_full_batch_size() -> None:
    plan = OnlineBatchPlan.from_root(_public_batch_config())

    assert plan.gradient_accumulation_steps == 0
    assert plan.microbatch_size == 32
    assert plan.host_memory_budget_fraction == 0.0
    assert isinstance(plan.host_memory_budget_fraction, float)
    assert plan.streaming is False


@pytest.mark.parametrize(
    ("actor", "message"),
    [
        ({"microbatch_size": 5}, "evenly divide"),
        (
            {"microbatch_size": 4, "gradient_accumulation_steps": 4},
            "set only one",
        ),
        ({"microbatch_size": -1}, "must be >= 0"),
        ({"gradient_accumulation_steps": -1}, "must be >= 0"),
        ({"samples_per_replay_batch": -1}, "samples_per_replay_batch"),
        ({"samples_per_replay_batch": "auto"}, "samples_per_replay_batch"),
    ],
)
def test_batch_plan_rejects_invalid_public_geometry(
    actor: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OnlineBatchPlan.from_root(_public_batch_config(**actor))


@pytest.mark.parametrize(
    ("rollout", "message"),
    [
        ({"prompts_per_batch": 0}, "prompts_per_batch"),
        ({"prompts_per_batch": True}, "prompts_per_batch"),
        ({"n_samples_per_prompt": 0}, "n_samples_per_prompt"),
        ({"n_samples_per_prompt": False}, "n_samples_per_prompt"),
    ],
)
def test_batch_plan_rejects_non_positive_batch_dimensions(
    rollout: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OnlineBatchPlan.from_root(_public_batch_config(**rollout))


def test_streaming_plan_requires_one_ppo_epoch() -> None:
    streaming = OnlineBatchPlan(
        prompts_per_batch=4,
        n_samples_per_prompt=2,
        gradient_accumulation_steps=2,
    )

    assert _trainer_config(streaming, ppo_epochs=1).ppo_epochs == 1
    with pytest.raises(ValueError, match="ppo_epochs"):
        _trainer_config(streaming, ppo_epochs=2)

    unsplit = OnlineBatchPlan(prompts_per_batch=4, n_samples_per_prompt=2)
    assert _trainer_config(unsplit, ppo_epochs=2).ppo_epochs == 2


@pytest.mark.parametrize("budget", [0.5, 0.9, 0.999])
def test_host_memory_budget_requires_streaming(budget: float) -> None:
    plan = OnlineBatchPlan(
        prompts_per_batch=4,
        n_samples_per_prompt=2,
        gradient_accumulation_steps=4,
        host_memory_budget_fraction=budget,
    )

    assert plan.host_memory_budget_fraction == budget
    with pytest.raises(ValueError, match="requires streaming"):
        OnlineBatchPlan(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            host_memory_budget_fraction=budget,
        )


@pytest.mark.parametrize("budget", [-0.1, 1.0, 1.5, float("inf"), float("nan"), True])
def test_host_memory_budget_rejects_invalid_values(budget: object) -> None:
    with pytest.raises(ValueError, match="host_memory_budget_fraction"):
        OnlineBatchPlan(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            gradient_accumulation_steps=4,
            host_memory_budget_fraction=budget,  # type: ignore[arg-type]
        )


def test_batch_plan_is_frozen() -> None:
    plan = OnlineBatchPlan(prompts_per_batch=4, n_samples_per_prompt=2)

    with pytest.raises(FrozenInstanceError):
        plan.prompts_per_batch = 8  # type: ignore[misc]
