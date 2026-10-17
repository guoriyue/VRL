"""Cosmos four-card preset budget and rank-local resource ownership."""

from vrl.config.builders import BuiltConfigs
from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.ray.resources import ResourcePlan

PRESET = "experiment/cosmos_predict2_5/online_grpo_kling_video_reward_colocated_fsdp_4x_l40s"


def test_four_card_preset_preserves_global_recipe_budget():
    base = parse_config(load_config("experiment/cosmos_predict2_5/online_grpo_kling_video_reward"))
    root = parse_config(load_config(PRESET))
    assert (
        root.rollout.prompts_per_batch * root.distributed.training.gpus_per_node
        == base.rollout.prompts_per_batch
    )
    assert root.rollout.n_samples_per_prompt == base.rollout.n_samples_per_prompt == 8
    assert root.actor.model_dump(exclude={"gradient_checkpointing"}) == base.actor.model_dump(
        exclude={"gradient_checkpointing"}
    )
    assert root.actor.gradient_checkpointing_mode == "full"
    assert root.algorithm == base.algorithm
    assert root.sampling == base.sampling
    assert root.trainer.total_epochs == base.trainer.total_epochs
    assert not root.model.torch_compile.enable
    assert root.distributed.training.fsdp.shard_trainable_only


def test_four_card_preset_resolves_rank_local_phase_owners():
    built = BuiltConfigs.from_cfg(
        load_config(
            PRESET,
            overrides=[
                "distributed.resources.visible_devices=[0]",
            ],
        )
    )
    resources = ResourcePlan.from_root(built.root)
    assert tuple(resources.trainer_devices) == (0,)
    assert tuple(resources.rollout_devices) == (0,)
    # The unpinned Kling reward follows the rank's own card: no reservation,
    # but the lifecycle plan sees the sharing.
    assert tuple(resources.reward_devices) == ()
    assert resources.reward_follows_trainer
    assert tuple(resources.lifecycle.reward) == (0,)
    assert resources.lifecycle.park_rollout_for_train
    assert resources.lifecycle.park_trainer_for_reward
