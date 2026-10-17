"""Family ownership tests for the generic online entrypoint."""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.ray.resources import ResourcePlan
from vrl.scripts.common.launch_environment import narrow_rank_local_cuda_visibility


def _distributed_cfg(
    *,
    strategy: str = "fsdp",
    rollout: dict[str, Any] | None = None,
    gpus_per_node: int = 4,
) -> Any:
    return OmegaConf.create(
        {
            "distributed": {
                "training": {
                    "strategy": strategy,
                    "gpus_per_node": gpus_per_node,
                },
                "resources": {"rollout": rollout or {}},
            },
        },
    )


def test_same_host_fsdp_selects_one_physical_gpu_per_rank() -> None:
    environ = {
        "LOCAL_RANK": "3",
        "LOCAL_WORLD_SIZE": "4",
        "WORLD_SIZE": "4",
        "CUDA_VISIBLE_DEVICES": "2,4,6,7",
    }
    cfg = _distributed_cfg()

    selected = narrow_rank_local_cuda_visibility(
        parse_config(cfg),
        environ=environ,
    )

    assert selected == "7"
    assert environ["CUDA_VISIBLE_DEVICES"] == "7"
    # The entrypoint feeds the physical ordinal back as a loader override.
    narrowed = OmegaConf.merge(
        cfg,
        {"distributed": {"resources": {"visible_devices": [int(selected)]}}},
    )
    resources = ResourcePlan.from_root(parse_config(narrowed))
    assert resources.trainer_devices == (7,)
    assert resources.rollout_devices == (7,)


def test_same_host_ddp_selects_local_rank_when_visibility_is_unset() -> None:
    environ = {"LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2", "WORLD_SIZE": "2"}

    selected = narrow_rank_local_cuda_visibility(
        parse_config(_distributed_cfg(strategy="ddp", gpus_per_node=2)),
        environ=environ,
    )

    assert selected == "1"
    assert environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_one_rank_per_node_preserves_an_existing_single_device_view() -> None:
    environ = {
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "WORLD_SIZE": "4",
        "CUDA_VISIBLE_DEVICES": "7",
    }

    selected = narrow_rank_local_cuda_visibility(
        parse_config(_distributed_cfg(gpus_per_node=1)),
        environ=environ,
    )

    assert selected == "7"
    assert environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_disjoint_rollout_does_not_change_cuda_visibility() -> None:
    environ = {
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "WORLD_SIZE": "2",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    }

    selected = narrow_rank_local_cuda_visibility(
        parse_config(_distributed_cfg(rollout={"devices": [2, 3]})),
        environ=environ,
    )

    assert selected is None
    assert environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_rank_local_cuda_selection_rejects_partial_device_masks() -> None:
    environ = {
        "LOCAL_RANK": "2",
        "LOCAL_WORLD_SIZE": "4",
        "WORLD_SIZE": "4",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }

    with pytest.raises(ValueError, match="cannot supply every local torchrun rank"):
        narrow_rank_local_cuda_visibility(
            parse_config(_distributed_cfg()),
            environ=environ,
        )


def test_rank_local_cuda_selection_rejects_duplicate_devices() -> None:
    environ = {
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "WORLD_SIZE": "2",
        "CUDA_VISIBLE_DEVICES": "0,0",
    }

    with pytest.raises(ValueError, match="duplicate devices"):
        narrow_rank_local_cuda_visibility(
            parse_config(_distributed_cfg(gpus_per_node=2)),
            environ=environ,
        )


@pytest.mark.parametrize("visible", ["0,,1", "0,GPU-uuid"])
def test_rank_local_selection_rejects_ambiguous_mask_without_mutation(visible: str) -> None:
    environ = {
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "2",
        "WORLD_SIZE": "2",
        "CUDA_VISIBLE_DEVICES": visible,
    }
    original = dict(environ)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        narrow_rank_local_cuda_visibility(
            parse_config(_distributed_cfg(gpus_per_node=2)), environ=environ
        )
    assert environ == original
