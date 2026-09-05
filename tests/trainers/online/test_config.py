"""Online trainer configuration ownership tests."""

from __future__ import annotations

from dataclasses import fields

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import ActorSection, TrainerSection, parse_config
from vrl.trainers.core.types import OptimConfig, ReplayParityConfig
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig


def test_replay_parity_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match=r"trainer\.replay_parity\.max_abs_logprob_diff"):
        ReplayParityConfig(max_abs_logprob_diff=-1.0)


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
        ({"samples_per_replay_batch": -1}, "samples_per_replay_batch"),
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
        ({"n_samples_per_prompt": 0}, "n_samples_per_prompt"),
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


def test_host_memory_budget_requires_streaming() -> None:
    plan = OnlineBatchPlan(
        prompts_per_batch=4,
        n_samples_per_prompt=2,
        gradient_accumulation_steps=4,
        host_memory_budget_fraction=0.9,
    )

    assert plan.host_memory_budget_fraction == 0.9
    with pytest.raises(ValueError, match="requires streaming"):
        OnlineBatchPlan(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            host_memory_budget_fraction=0.9,
        )


@pytest.mark.parametrize("budget", [-0.1, float("nan")])
def test_host_memory_budget_rejects_invalid_values(budget: object) -> None:
    with pytest.raises(ValueError, match="host_memory_budget_fraction"):
        OnlineBatchPlan(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            gradient_accumulation_steps=4,
            host_memory_budget_fraction=budget,  # type: ignore[arg-type]
        )
