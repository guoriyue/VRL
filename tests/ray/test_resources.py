"""Tests for role-level distributed resource resolution."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.ray.resources import (
    format_distributed_resource_plan,
    resolve_distributed_resources,
    trainer_torch_device,
)


def _cfg(
    resources: dict,
    *,
    rollout_release_after_collect: bool = False,
    rollout_release_before_reward_model: bool = False,
    reward_release_after_score: bool = False,
    video_reward: bool = False,
) -> object:
    data = {
        "distributed": {
            "backend": "ray",
            "resources": resources,
            "rollout": {
                "release_after_collect": rollout_release_after_collect,
                "release_before_reward_model": rollout_release_before_reward_model,
            },
            "reward": {"release_after_score": reward_release_after_score},
        },
    }
    if video_reward:
        data["reward"] = {
            "components": {"video_reward": 1.0},
            "kwargs": {"video_reward": {"inference_runtime": "ray"}},
        }
    return OmegaConf.create(
        data,
    )


def test_auto_split_uses_remaining_visible_gpus_for_rollout() -> None:
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
    assert resolved.total_gpu_slots == 4
    assert resolved.ray_total_bundles == 4
    assert resolved.requires_trainer_reservation is True
    assert trainer_torch_device(resolved) == "cuda:0"


def test_explicit_split_devices_do_not_overlap() -> None:
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
    assert resolved.ray_total_bundles == 1


def test_devices_must_be_subset_of_visible_devices() -> None:
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
    assert resolved.ray_total_bundles == 2
    assert trainer_torch_device(resolved) == "cpu"


def test_trainer_only_plan_allows_zero_rollout_workers() -> None:
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
    assert resolved.ray_total_bundles == 0
    assert trainer_torch_device(resolved) == "cuda:0"


def test_resource_plan_formatter_includes_key_fields() -> None:
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
    assert "trainer_reservation=True" in text


def test_cross_node_rollout_satisfies_budget_from_explicit_counts() -> None:
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
    assert resolved.ray_total_bundles == 1
    assert trainer_torch_device(resolved) == "cuda:0"


def test_cross_node_scales_to_multiple_rollout_workers() -> None:
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
    assert resolved.reward_num_gpus == 1
    assert resolved.reward_num_workers == 1
    assert resolved.reward_gpus_per_worker == 1.0
    assert resolved.reward_shared_with_rollout is False
    assert resolved.reward_gpu_reservation_count == 2


def test_ray_video_reward_requires_reward_gpu_budget() -> None:
    with pytest.raises(ValueError, match=r"distributed\.resources\.reward\.num_gpus > 0"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                },
                video_reward=True,
            ),
        )


def test_reward_rollout_overlap_requires_shared_release_lifecycle() -> None:
    with pytest.raises(ValueError, match="release_before_reward_model"):
        resolve_distributed_resources(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                    "reward": {
                        "devices": [1],
                        "gpus_per_worker": 1,
                        "share_with_rollout": True,
                    },
                },
            ),
        )


def test_reward_can_share_rollout_pool_when_phases_release() -> None:
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
            rollout_release_after_collect=True,
            rollout_release_before_reward_model=True,
            reward_release_after_score=True,
        ),
    )

    assert resolved.reward_devices == (1,)
    assert resolved.reward_shared_with_rollout is True
    assert resolved.rollout_release_before_reward_model is True
    assert resolved.reward_release_after_score is True


def test_reward_shared_pool_cannot_request_more_gpus_than_rollout_pool() -> None:
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
                rollout_release_after_collect=True,
                rollout_release_before_reward_model=True,
                reward_release_after_score=True,
            ),
        )


def test_reward_trainer_overlap_requires_explicit_allow_overlap() -> None:
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
