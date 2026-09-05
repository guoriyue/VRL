"""Family ownership tests for the generic online entrypoint."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from omegaconf import OmegaConf

import vrl.scripts.train as train
from vrl.config.schema import parse_config
from vrl.ray.resources import ResolvedDistributedResources


def _cfg(family: str, algorithm_kind: str) -> Any:
    return OmegaConf.create(
        {
            "model": {"family": family},
            "algorithm": {"kind": algorithm_kind},
        },
    )


def _distributed_cfg(
    *,
    strategy: str = "fsdp",
    gpu_pool: str = "trainer",
    gpus_per_node: int = 4,
) -> Any:
    return OmegaConf.create(
        {
            "distributed": {
                "training": {
                    "strategy": strategy,
                    "gpus_per_node": gpus_per_node,
                },
                "resources": {"rollout": {"gpu_pool": gpu_pool}},
            },
        },
    )


@pytest.mark.parametrize(
    ("family", "algorithm_kind"),
    [
        ("janus_pro_r1", "token_grpo_multisegment"),
        ("janus_pro", "token_grpo"),
    ],
)
def test_train_online_keeps_family_owned_by_config(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    algorithm_kind: str,
) -> None:
    """The generic entrypoint passes config through without a family wrapper."""
    captured: dict[str, object] = {}

    async def fake_run_online_recipe(cfg: Any) -> None:
        captured["family"] = str(cfg.model.family)

    monkeypatch.setattr(
        "vrl.scripts.common.online.run_online_recipe",
        fake_run_online_recipe,
    )

    asyncio.run(train.train_online(_cfg(family, algorithm_kind)))

    assert captured == {"family": family}


@pytest.mark.parametrize(
    ("local_rank", "expected"),
    [(0, "2"), (3, "7")],
)
def test_same_host_fsdp_selects_one_physical_gpu_per_rank(
    local_rank: int,
    expected: str,
) -> None:
    environ = {
        "LOCAL_RANK": str(local_rank),
        "LOCAL_WORLD_SIZE": "4",
        "WORLD_SIZE": "4",
        "CUDA_VISIBLE_DEVICES": "2,4,6,7",
    }
    cfg = _distributed_cfg()

    selected = train._narrow_rank_local_cuda_visibility(
        parse_config(cfg),
        environ=environ,
    )

    assert selected == expected
    assert environ["CUDA_VISIBLE_DEVICES"] == expected
    # The entrypoint feeds the physical ordinal back as a loader override.
    narrowed = OmegaConf.merge(
        cfg,
        {"distributed": {"resources": {"visible_devices": [int(selected)]}}},
    )
    resources = ResolvedDistributedResources.resolve(parse_config(narrowed))
    assert resources.trainer_devices == (int(expected),)
    assert resources.rollout_devices == (int(expected),)


def test_same_host_ddp_selects_local_rank_when_visibility_is_unset() -> None:
    environ = {"LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2", "WORLD_SIZE": "2"}

    selected = train._narrow_rank_local_cuda_visibility(
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

    selected = train._narrow_rank_local_cuda_visibility(
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

    selected = train._narrow_rank_local_cuda_visibility(
        parse_config(_distributed_cfg(gpu_pool="dedicated")),
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
        train._narrow_rank_local_cuda_visibility(
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
        train._narrow_rank_local_cuda_visibility(
            parse_config(_distributed_cfg(gpus_per_node=2)),
            environ=environ,
        )
