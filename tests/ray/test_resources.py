"""Tests for role-level distributed resource resolution."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.ray.resources import (
    build_bundle_layout,
    format_distributed_resource_plan,
    resolve_distributed_resources,
    trainer_torch_device,
)


def _cfg(
    resources: dict,
    *,
    colocate_with_trainer: float | None = None,
    rollout_runtime: dict | None = None,
    kling_video_reward: bool = False,
    reward_components: dict[str, float] | None = None,
    reward_kwargs: dict[str, dict] | None = None,
) -> object:
    # Release scheduling is derived from topology, so the public rollout block only
    # carries the colocate_with_trainer intent (memory_fraction cap). Tests that
    # exercise the removed-key rejection pass a raw rollout_runtime instead.
    if rollout_runtime is None:
        rollout_runtime = {}
        if colocate_with_trainer is not None:
            rollout_runtime["colocate_with_trainer"] = {
                "memory_fraction": colocate_with_trainer,
            }
    data = {
        "distributed": {
            "resources": resources,
            "rollout": rollout_runtime,
        },
    }
    if kling_video_reward:
        reward_components = {"kling_video_reward": 1.0}
        reward_kwargs = {"kling_video_reward": {"execution": "pool"}}
    if reward_components is not None:
        data["reward"] = {
            "components": reward_components,
            "kwargs": reward_kwargs or {},
        }
    return OmegaConf.create(
        data,
    )


def test_auto_split_uses_remaining_visible_gpus_for_rollout() -> None:
    """Checks auto split uses remaining visible gpus for rollout."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2, 3],
                "trainer": {"num_gpus": 1},
                "rollout": {
                    "num_gpus": "auto",
                    "gpus_per_worker": 1,
                    "num_workers": "auto",
                },
                "allow_overlap": False,
            },
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1, 2, 3)
    assert resolved.reward_devices == ()
    assert resolved.rollout_num_workers == 3
    assert resolved.reward_num_workers == 0
    assert resolved.requires_trainer_reservation is True
    assert trainer_torch_device(resolved) == "cuda:0"


def test_explicit_split_devices_do_not_overlap() -> None:
    """Checks explicit split devices do not overlap."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2, 3],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1, 2, 3], "gpus_per_worker": 1},
                "allow_overlap": False,
            },
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1, 2, 3)
    assert resolved.colocated is False


def test_explicit_overlap_requires_allow_overlap() -> None:
    """Checks explicit overlap requires allow overlap."""
    with pytest.raises(ValueError, match="overlap"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0], "gpus_per_worker": 1},
                    "allow_overlap": False,
                },
            ),
        )


def test_explicit_overlap_marks_colocated_when_allowed() -> None:
    """Checks explicit overlap marks colocated when allowed."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "gpus_per_worker": 1},
                "allow_overlap": True,
            },
        ),
    )

    assert resolved.colocated is True
    assert resolved.requires_trainer_reservation is False


def test_single_gpu_colocate_with_trainer_keeps_worker_resident() -> None:
    """Checks colocate_with_trainer keeps the single-GPU rollout worker resident."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "gpus_per_worker": 1},
                "allow_overlap": True,
            },
            colocate_with_trainer=0.45,
        ),
    )

    assert resolved.colocated is True
    assert resolved.rollout_release_after_collect is False
    assert resolved.rollout_persistent_colocated_workers is True
    assert resolved.rollout_gpu_memory_fraction == 0.45


def test_colocate_with_trainer_implies_overlap_without_allow_overlap() -> None:
    """colocate_with_trainer is itself the overlap permission; allow_overlap unset."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "gpus_per_worker": 1},
            },
            colocate_with_trainer=0.45,
        ),
    )

    assert resolved.colocated is True
    assert resolved.rollout_persistent_colocated_workers is True


def test_colocate_with_trainer_auto_pins_rollout_to_trainer_gpu() -> None:
    """Auto rollout placement is forced onto the trainer GPU, not a spare one."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0]},
                "rollout": {"num_gpus": "auto", "gpus_per_worker": 1},
            },
            colocate_with_trainer=0.4,
        ),
    )

    assert resolved.rollout_devices == (0,)
    assert resolved.colocated is True


def test_colocate_with_trainer_requires_memory_fraction() -> None:
    """Checks a colocate block without memory_fraction fails inside the block."""
    with pytest.raises(ValueError, match="colocate_with_trainer requires memory_fraction"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0], "gpus_per_worker": 1},
                    "allow_overlap": True,
                },
                rollout_runtime={"colocate_with_trainer": {}},
            ),
        )


def test_colocate_with_trainer_memory_fraction_rejects_out_of_range() -> None:
    """Checks the GPU budget cap is validated to the (0, 1] range."""
    with pytest.raises(ValueError, match=r"memory_fraction must be in \(0, 1\]"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0], "gpus_per_worker": 1},
                    "allow_overlap": True,
                },
                colocate_with_trainer=1.5,
            ),
        )


def test_colocate_with_trainer_rejects_explicit_disjoint_rollout_devices() -> None:
    """Checks colocate cannot be silently reinterpreted as split-GPU resident."""
    with pytest.raises(ValueError, match="disjoint from trainer"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                },
                colocate_with_trainer=0.45,
            ),
        )


@pytest.mark.parametrize(
    ("rollout_runtime", "match"),
    [
        ({"release_after_collect": False}, "release_after_collect was removed"),
        (
            {"release_before_reward_model": True},
            "release_before_reward_model was removed",
        ),
        (
            {"persistent_colocated_workers": True},
            "persistent_colocated_workers moved into",
        ),
        ({"gpu_memory_fraction": 0.45}, "gpu_memory_fraction moved into"),
    ],
)
def test_removed_rollout_keys_are_rejected(rollout_runtime, match) -> None:
    """Removed public rollout keys fail with a migration pointer, not silently."""
    with pytest.raises(ValueError, match=match):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0], "gpus_per_worker": 1},
                    "allow_overlap": True,
                },
                rollout_runtime=rollout_runtime,
            ),
        )


def test_removed_reward_release_after_score_key_is_rejected() -> None:
    """The public distributed.reward.release_after_score key is rejected."""
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                },
                "reward": {"release_after_score": True},
            },
        },
    )
    with pytest.raises(ValueError, match="release_after_score was removed"):
        resolve_distributed_resources(cfg)


def test_devices_must_be_subset_of_visible_devices() -> None:
    """Checks devices must be subset of visible devices."""
    with pytest.raises(ValueError, match=r"outside distributed\.resources\.visible_devices"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [2]},
                    "rollout": {"num_gpus": 1, "gpus_per_worker": 1},
                },
            ),
        )


def test_num_workers_auto_requires_divisible_gpu_budget() -> None:
    """Checks num workers auto requires divisible GPU budget."""
    with pytest.raises(ValueError, match="gpus_per_worker currently supports 0 or 1"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"num_gpus": 0},
                    "rollout": {
                        "num_gpus": 1,
                        "gpus_per_worker": 0.5,
                        "num_workers": "auto",
                    },
                },
            ),
        )


def test_single_gpu_auto_split_fails_without_overlap() -> None:
    """Checks single GPU auto split fails without overlap."""
    with pytest.raises(ValueError, match="Not enough non-overlapping rollout GPUs"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 1},
                    "rollout": {
                        "num_gpus": 1,
                        "gpus_per_worker": 1,
                        "num_workers": 1,
                    },
                    "allow_overlap": False,
                },
            ),
        )


def test_cpu_only_rollout_uses_no_gpu_bundles() -> None:
    """Checks CPU only rollout uses no GPU bundles."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [],
                "trainer": {"num_gpus": 0},
                "rollout": {
                    "num_gpus": 0,
                    "gpus_per_worker": 0,
                    "num_workers": 2,
                },
            },
        ),
    )

    assert resolved.trainer_devices == ()
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_workers == 2
    assert trainer_torch_device(resolved) == "cpu"


def test_trainer_only_plan_allows_zero_rollout_workers() -> None:
    """Checks trainer only plan allows zero rollout workers."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"num_gpus": 1},
                "rollout": {
                    "num_gpus": 0,
                    "gpus_per_worker": 0,
                    "num_workers": 0,
                },
            },
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_workers == 0
    assert trainer_torch_device(resolved) == "cuda:0"


def test_resource_plan_formatter_includes_key_fields() -> None:
    """Checks resource plan formatter includes key fields."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1],
                "trainer": {"num_gpus": 1},
                "rollout": {"num_gpus": "auto", "gpus_per_worker": 1},
            },
        ),
    )

    text = format_distributed_resource_plan(resolved)

    assert "trainer=[0]" in text
    assert "rollout=[1]" in text
    assert "reward=[]" in text
    assert "rollout_persistent_colocated_workers=False" in text
    assert "rollout_gpu_memory_fraction=None" in text
    assert "trainer_reservation=True" in text


def test_cross_node_rollout_satisfies_budget_from_explicit_counts() -> None:
    """Checks cross-node rollout satisfies budget from explicit counts."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": "auto",
                "cross_node": True,
                "trainer": {"num_gpus": 1},
                "rollout": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
                "allow_overlap": False,
            },
        ),
    )

    assert resolved.cross_node is True
    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1,)
    assert resolved.rollout_num_workers == 1
    assert resolved.colocated is False
    assert resolved.requires_trainer_reservation is False
    assert trainer_torch_device(resolved) == "cuda:0"


def test_cross_node_scales_to_multiple_rollout_workers() -> None:
    """Checks cross-node scales to multiple rollout workers."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": "auto",
                "cross_node": True,
                "trainer": {"num_gpus": 1},
                "rollout": {"num_gpus": 3, "gpus_per_worker": 1, "num_workers": 3},
            },
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1, 2, 3)
    assert resolved.rollout_num_workers == 3
    assert resolved.requires_trainer_reservation is False


def test_cross_node_requires_explicit_rollout_count() -> None:
    """Checks cross-node requires explicit rollout count."""
    with pytest.raises(ValueError, match="cross_node"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": "auto",
                    "cross_node": True,
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": "auto", "gpus_per_worker": 1},
                },
            ),
        )


def test_cross_node_plan_formatter_reports_flag() -> None:
    """Checks cross-node plan formatter reports flag."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": "auto",
                "cross_node": True,
                "trainer": {"num_gpus": 1},
                "rollout": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
            },
        ),
    )

    assert "cross_node=True" in format_distributed_resource_plan(resolved)


def test_cross_node_preset_resolves() -> None:
    """Checks cross-node preset resolves."""
    from pathlib import Path

    from omegaconf import OmegaConf

    preset = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "base"
        / "distributed"
        / "ray_rollout_cross_node.yaml"
    )
    resolved = resolve_distributed_resources(OmegaConf.load(preset))

    assert resolved.cross_node is True
    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1,)
    assert resolved.rollout_num_workers == 1
    assert resolved.requires_trainer_reservation is False


def test_reward_role_resolves_after_trainer_and_rollout_devices() -> None:
    """Checks reward role resolves after trainer and rollout devices."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
            },
        ),
    )

    assert resolved.reward_devices == (2,)
    assert resolved.reward_num_workers == 1
    assert resolved.reward_gpus_per_worker == 1.0
    assert resolved.reward_shared_with_rollout is False
    assert resolved.requires_trainer_reservation is True
    assert resolved.rollout_release_before_reward_model is False


def test_ray_video_reward_requires_reward_gpu_budget() -> None:
    """Checks Ray video reward requires reward GPU budget."""
    with pytest.raises(ValueError, match=r"distributed\.resources\.reward\.num_gpus > 0"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                },
                kling_video_reward=True,
            ),
        )


def test_ray_reward_requires_reward_gpu_budget_for_any_component() -> None:
    """Checks any active Ray reward requires reward GPU budget."""
    with pytest.raises(ValueError, match=r"distributed\.resources\.reward\.num_gpus > 0"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                },
                reward_components={"custom_gpu_reward": 1.0},
                reward_kwargs={"custom_gpu_reward": {"execution": "pool"}},
            ),
        )


def test_reward_shared_pool_derives_release_lifecycle_when_unset() -> None:
    """Checks unset release flags derive to true for a shared reward pool."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {
                    "num_gpus": 1,
                    "gpus_per_worker": 1,
                    "num_workers": 1,
                    "share_with_rollout": True,
                },
            },
        ),
    )

    assert resolved.reward_shared_with_rollout is True
    assert resolved.rollout_release_after_collect is True
    assert resolved.rollout_release_before_reward_model is True
    assert resolved.reward_release_after_score is True


def test_dedicated_reward_gpu_derives_resident_lifecycle_when_unset() -> None:
    """Checks unset release flags derive to false when reward has its own GPU."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {"devices": [2], "gpus_per_worker": 1, "num_workers": 1},
            },
        ),
    )

    assert resolved.reward_shared_with_rollout is False
    assert resolved.rollout_release_after_collect is False
    assert resolved.rollout_release_before_reward_model is False
    assert resolved.reward_release_after_score is False


def test_multiple_ray_rewards_derive_release_after_score_on_dedicated_gpu() -> None:
    """Checks sequential Ray reward models release a shared reward placement."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {"devices": [2], "gpus_per_worker": 1, "num_workers": 1},
            },
            reward_components={"first_gpu_reward": 1.0, "second_gpu_reward": 1.0},
            reward_kwargs={
                "first_gpu_reward": {"execution": "pool"},
                "second_gpu_reward": {"execution": "pool"},
            },
        ),
    )

    assert resolved.reward_shared_with_rollout is False
    assert resolved.reward_release_after_score is True


def test_lifecycle_plan_resident_when_roles_disjoint() -> None:
    """Fully disjoint trainer/rollout/reward GPUs -> every role resident, no handoff."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {"devices": [2], "gpus_per_worker": 1, "num_workers": 1},
            },
        ),
    )

    plan = resolved.lifecycle
    assert plan.rollout.mode == "resident"
    assert plan.reward.mode == "resident"
    assert plan.handoff.release_rollout_before_train is False
    assert plan.handoff.release_rollout_before_reward is False
    assert plan.handoff.release_reward_after_score is False
    # Flat flags mirror the plan (one derivation, no divergence).
    assert resolved.rollout_release_after_collect is False
    assert resolved.reward_release_after_score is False


def test_lifecycle_plan_on_demand_for_shared_reward() -> None:
    """Shared reward GPU -> rollout/reward on_demand, but no trainer handoff."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {
                    "num_gpus": 1,
                    "gpus_per_worker": 1,
                    "num_workers": 1,
                    "share_with_rollout": True,
                },
            },
        ),
    )

    plan = resolved.lifecycle
    assert plan.rollout.mode == "on_demand"
    assert plan.reward.mode == "on_demand"
    assert plan.handoff.release_rollout_before_train is False
    assert plan.handoff.release_rollout_before_reward is True
    assert plan.handoff.release_reward_after_score is True
    # Flat compatibility flags still request an on-demand rollout lease.
    assert resolved.rollout_release_after_collect is True
    assert resolved.rollout_release_before_reward_model is True
    assert resolved.reward_release_after_score is True


def test_lifecycle_plan_colocated_rollout_is_on_demand_before_train() -> None:
    """Trainer/rollout share a GPU -> rollout on_demand, releases before train only."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "gpus_per_worker": 1},
                "allow_overlap": True,
            },
        ),
    )

    plan = resolved.lifecycle
    assert resolved.colocated is True
    assert plan.rollout.mode == "on_demand"
    assert plan.handoff.release_rollout_before_train is True
    # No reward role shares the rollout GPU, so reward stays resident.
    assert plan.reward.mode == "resident"
    assert plan.handoff.release_rollout_before_reward is False


def test_lifecycle_plan_colocate_with_trainer_stays_resident() -> None:
    """colocate_with_trainer overrides the colocated on_demand default."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "gpus_per_worker": 1},
                "allow_overlap": True,
            },
            colocate_with_trainer=0.45,
        ),
    )

    plan = resolved.lifecycle
    assert plan.rollout.mode == "resident"
    assert plan.handoff.release_rollout_before_train is False


def test_resource_plan_formatter_includes_lifecycle() -> None:
    """Acceptance #8: the resource plan log shows the lifecycle plan at a glance."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {
                    "num_gpus": 1,
                    "gpus_per_worker": 1,
                    "num_workers": 1,
                    "share_with_rollout": True,
                },
            },
        ),
    )

    text = format_distributed_resource_plan(resolved)
    assert "lifecycle=rollout:on_demand/reward:on_demand" in text
    assert "before_reward:True" in text


def test_reward_auto_placement_prefers_dedicated_spare_gpu() -> None:
    """Checks unset share_with_rollout takes the spare GPU on multi-GPU boxes."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
            },
        ),
    )

    assert resolved.reward_devices == (2,)
    assert resolved.reward_shared_with_rollout is False
    assert resolved.reward_release_after_score is False


def test_reward_auto_placement_falls_back_to_shared_pool_on_single_gpu() -> None:
    """Checks unset share_with_rollout shares the rollout GPU when none is spare."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "gpus_per_worker": 1},
                "allow_overlap": True,
                "reward": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
            },
        ),
    )

    assert resolved.reward_devices == (0,)
    assert resolved.reward_shared_with_rollout is True
    assert resolved.rollout_release_before_reward_model is True
    assert resolved.reward_release_after_score is True


def test_reward_can_share_rollout_pool_when_phases_release() -> None:
    """Checks reward can share rollout pool when phases release."""
    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [1], "gpus_per_worker": 1},
                "reward": {
                    "num_gpus": 1,
                    "gpus_per_worker": 1,
                    "num_workers": 1,
                    "share_with_rollout": True,
                },
                "allow_overlap": False,
            },
        ),
    )

    assert resolved.reward_devices == (1,)
    assert resolved.reward_shared_with_rollout is True
    assert resolved.rollout_release_before_reward_model is True
    assert resolved.reward_release_after_score is True


def test_reward_shared_pool_cannot_request_more_gpus_than_rollout_pool() -> None:
    """Checks reward shared pool cannot request more gpus than rollout pool."""
    with pytest.raises(ValueError, match="Not enough rollout GPUs"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                    "reward": {
                        "num_gpus": 2,
                        "gpus_per_worker": 1,
                        "share_with_rollout": True,
                    },
                    "allow_overlap": False,
                },
            ),
        )


def test_reward_trainer_overlap_requires_explicit_allow_overlap() -> None:
    """Checks reward trainer overlap requires explicit allow overlap."""
    with pytest.raises(ValueError, match="Trainer and reward devices overlap"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"num_gpus": 0, "gpus_per_worker": 0},
                    "reward": {"devices": [0], "gpus_per_worker": 1},
                    "allow_overlap": False,
                },
            ),
        )


def test_colocated_reward_on_dedicated_gpu_owns_its_own_bundle() -> None:
    """Colocated trainer+rollout on GPU 0; a dedicated reward GPU 1 owns its own
    bundle (the run-level layout targets it directly, no reservation offset)."""

    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0, 1],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "num_workers": 1},
                "reward": {"devices": [1], "num_workers": 1, "gpus_per_worker": 1.0},
                "allow_overlap": True,
            },
            kling_video_reward=True,
        ),
    )
    assert resolved.colocated is True
    assert resolved.rollout_release_after_collect is True

    layout = build_bundle_layout(resolved)
    # Colocated trainer+rollout -> no reserved trainer bundle; reward owns a
    # bundle distinct from rollout (its own GPU 1).
    assert layout.trainer_bundle_indices == ()
    assert layout.reward_bundle_indices != ()
    assert set(layout.rollout_bundle_indices).isdisjoint(layout.reward_bundle_indices)


def test_shared_single_gpu_reward_reuses_rollout_bundle() -> None:
    """Shared single-GPU reward sits on the rollout device: same bundle index."""

    resolved = resolve_distributed_resources(
        _cfg(
            {
                "visible_devices": [0],
                "trainer": {"devices": [0]},
                "rollout": {"devices": [0], "num_workers": 1},
                "reward": {
                    "devices": [0],
                    "num_workers": 1,
                    "gpus_per_worker": 1.0,
                    "share_with_rollout": True,
                },
                "allow_overlap": True,
            },
            kling_video_reward=True,
        ),
    )

    layout = build_bundle_layout(resolved)
    assert layout.rollout_bundle_indices == layout.reward_bundle_indices
    assert layout.total_bundles == 1


# ---------------------------------------------------------------- fsdp (P2)


def _cfg_training(resources: dict, training: dict) -> object:
    return OmegaConf.create(
        {
            "distributed": {
                "resources": resources,
                "rollout": {},
                "reward": {},
                "training": training,
            },
        },
    )


def test_fsdp_trainer_allows_multi_gpu_disjoint_from_rollout() -> None:
    """fsdp lifts the single-device cap: trainer can own N GPUs."""
    resolved = resolve_distributed_resources(
        _cfg_training(
            {
                "visible_devices": [0, 1, 2],
                "trainer": {"devices": [0, 1]},
                "rollout": {"devices": [2], "gpus_per_worker": 1},
            },
            {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
        ),
    )
    assert resolved.trainer_devices == (0, 1)
    assert resolved.rollout_devices == (2,)


def test_fsdp_trainer_count_must_equal_world_size() -> None:
    """fsdp trainer device count must match num_nodes*gpus_per_node."""
    with pytest.raises(ValueError, match=r"must own num_nodes\*gpus_per_node=2"):
        resolve_distributed_resources(
            _cfg_training(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                },
                {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
            ),
        )


def test_fsdp_trainer_must_be_disjoint_from_rollout_even_with_overlap() -> None:
    """fsdp rejects trainer/rollout GPU overlap regardless of allow_overlap."""
    with pytest.raises(ValueError, match="fsdp requires trainer GPUs disjoint"):
        resolve_distributed_resources(
            _cfg_training(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0, 1]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                    "allow_overlap": True,
                },
                {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
            ),
        )


def test_single_process_still_rejects_multi_gpu_trainer() -> None:
    """The single-device cap is preserved for single_process (default)."""
    with pytest.raises(ValueError, match="0 or 1 GPU"):
        resolve_distributed_resources(
            _cfg_training(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0, 1]},
                    "rollout": {"num_gpus": 0, "gpus_per_worker": 0, "num_workers": 1},
                },
                {"strategy": "single_process"},
            ),
        )
