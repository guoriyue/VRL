"""Cosmos four-card preset budget and rank-local resource ownership."""

import pytest

from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.run import resolve_online_run


PRESET = 'experiment/cosmos_predict2_5/online_grpo_kling_video_reward_colocated_fsdp_4x_l40s'


def test_four_card_preset_preserves_global_recipe_budget():
    base = parse_config(load_config('experiment/cosmos_predict2_5/online_grpo_kling_video_reward'))
    root = parse_config(load_config(PRESET))
    assert root.rollout.prompts_per_batch * root.distributed.training.gpus_per_node == base.rollout.prompts_per_batch
    assert root.rollout.n_samples_per_prompt == base.rollout.n_samples_per_prompt == 8
    assert root.actor == base.actor
    assert root.algorithm == base.algorithm
    assert root.sampling == base.sampling
    assert root.trainer.total_epochs == base.trainer.total_epochs
    assert not root.model.torch_compile.enable
    assert root.distributed.training.fsdp.shard_trainable_only


@pytest.mark.parametrize('rank', range(4))
def test_four_card_preset_resolves_rank_local_phase_owners(rank):
    resolved = resolve_online_run(load_config(PRESET, overrides=[
        f'distributed.resources.visible_devices=[{rank}]',
    ]))
    resources = resolved.resources
    assert tuple(resources.trainer_devices) == (rank,)
    assert tuple(resources.rollout_devices) == (rank,)
    assert tuple(resources.reward_devices) == (rank,)
    assert resources.lifecycle.release_rollout_before_train
    assert resources.lifecycle.release_trainer_before_reward
