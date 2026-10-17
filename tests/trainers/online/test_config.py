"""Online trainer configuration ownership tests."""

from __future__ import annotations

from dataclasses import fields

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import ActorSection, TrainerSection, parse_config
from vrl.trainers.core.types import OptimConfig, ReplayParityConfig
from vrl.trainers.online.config import OnlineBatchConfig, TrainerConfig


def test_replay_parity_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match=r"trainer\.replay_parity\.max_abs_logprob_diff"):
        ReplayParityConfig(max_abs_logprob_diff=-1.0)


def test_trainer_config_fields_are_declared_by_exactly_one_public_section() -> None:
    """The projection reads each field from the section that declares its name."""
    bridged = {"batch_config", "versioned_weight_sync"}
    for trainer_field in fields(TrainerConfig):
        if trainer_field.name in bridged:
            continue
        owners = [
            section
            for section in (ActorSection, TrainerSection)
            if trainer_field.name in section.model_fields
        ]
        assert len(owners) == 1, trainer_field.name
    for config_field in fields(OnlineBatchConfig):
        if config_field.name in ("prompts_per_batch", "n_samples_per_prompt"):
            continue
        assert config_field.name in ActorSection.model_fields, config_field.name


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


def _trainer_config(batch_config: OnlineBatchConfig, *, ppo_epochs: int = 1) -> TrainerConfig:
    return TrainerConfig(
        optim=OptimConfig(lr=1e-4),
        batch_config=batch_config,
        timestep_fraction=0.5,
        output_dir="x",
        drop_zero_advantage=False,
        ppo_epochs=ppo_epochs,
    )


def test_batch_config_derives_collection_count() -> None:
    config = OnlineBatchConfig.from_root(_public_batch_config(prompts_per_collection=4))
    assert config.collections_per_update == 8
    assert config.prompts_per_collection == 4
    assert config.streaming is True


def test_online_plan_rejects_retired_accumulation_setting() -> None:
    with pytest.raises(ValueError, match="only supported by offline DPO"):
        OnlineBatchConfig.from_root(_public_batch_config(gradient_accumulation_steps=8))


@pytest.mark.parametrize(
    ("actor", "message"),
    [
        ({"prompts_per_collection": 5}, "evenly divide"),
        ({"prompts_per_collection": -1}, "must be >= 0"),
        ({"training_microbatch_size": -1}, "training_microbatch_size"),
    ],
)
def test_batch_config_rejects_invalid_public_geometry(
    actor: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OnlineBatchConfig.from_root(_public_batch_config(**actor))


@pytest.mark.parametrize(
    ("rollout", "message"),
    [
        ({"prompts_per_batch": 0}, "prompts_per_batch"),
        ({"n_samples_per_prompt": 0}, "n_samples_per_prompt"),
    ],
)
def test_batch_config_rejects_non_positive_batch_dimensions(
    rollout: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OnlineBatchConfig.from_root(_public_batch_config(**rollout))


def test_streaming_plan_requires_one_ppo_epoch() -> None:
    streaming = OnlineBatchConfig(
        prompts_per_batch=4,
        n_samples_per_prompt=2,
        prompts_per_collection=2,
    )

    assert _trainer_config(streaming, ppo_epochs=1).ppo_epochs == 1
    with pytest.raises(ValueError, match="ppo_epochs"):
        _trainer_config(streaming, ppo_epochs=2)

    unsplit = OnlineBatchConfig(prompts_per_batch=4, n_samples_per_prompt=2)
    assert _trainer_config(unsplit, ppo_epochs=2).ppo_epochs == 2


def test_host_memory_budget_requires_streaming() -> None:
    config = OnlineBatchConfig(
        prompts_per_batch=4,
        n_samples_per_prompt=2,
        prompts_per_collection=1,
        host_memory_budget_fraction=0.9,
    )

    assert config.host_memory_budget_fraction == 0.9
    with pytest.raises(ValueError, match="requires streaming"):
        OnlineBatchConfig(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            host_memory_budget_fraction=0.9,
        )


@pytest.mark.parametrize("budget", [-0.1, float("nan")])
def test_host_memory_budget_rejects_invalid_values(budget: object) -> None:
    with pytest.raises(ValueError, match="host_memory_budget_fraction"):
        OnlineBatchConfig(
            prompts_per_batch=4,
            n_samples_per_prompt=2,
            prompts_per_collection=1,
            host_memory_budget_fraction=budget,  # type: ignore[arg-type]
        )
