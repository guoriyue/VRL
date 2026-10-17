"""Tests for role-level distributed resource resolution."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.ray.placement import BundleLayout
from vrl.ray.resources import (
    ResourcePlan,
)


def _cfg(
    resources: dict,
    *,
    rollout_runtime: dict | None = None,
    kling_video_reward: bool = False,
    reward_components: dict[str, float] | None = None,
    reward_kwargs: dict[str, dict] | None = None,
    reward_inference: dict[str, dict] | None = None,
) -> object:
    if rollout_runtime is None:
        rollout_runtime = {}
    data = {
        "distributed": {
            "resources": resources,
            "rollout": rollout_runtime,
        },
    }
    if kling_video_reward:
        reward_components = {"kling_video_reward": 1.0}
        reward_kwargs = {"kling_video_reward": {}}
    if reward_components is not None:
        data["reward"] = {
            "components": reward_components,
            "kwargs": reward_kwargs or {},
            "inference": reward_inference or {},
        }
    return OmegaConf.create(
        data,
    )


def test_pinned_rollout_devices_split_the_box() -> None:
    """Pinned rollout devices give one engine per GPU and a trainer reservation on
    ``cuda:0``.
    """
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2, 3],
                    "trainer": {"num_gpus": 1},
                    "rollout": {"devices": [1, 2, 3]},
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1, 2, 3)
    assert resolved.reward_devices == ()
    assert resolved.rollout_num_engines == 3
    assert resolved.requires_trainer_reservation is True
    assert resolved.trainer_torch_device == "cuda:0"


def test_auto_device_discovery_preserves_cuda_query_failure(monkeypatch) -> None:
    import torch

    from tests.rollouts.collector._helpers import Trace

    error = RuntimeError("CUDA discovery failed")
    cuda = Trace(monkeypatch)
    cuda.watch(torch.cuda, "is_available", "cuda.is_available")
    cuda.fail("cuda.is_available", error)
    root = parse_config(_cfg({"visible_devices": "auto"}))
    with pytest.raises(RuntimeError) as caught:
        ResourcePlan.from_root(root)
    assert caught.value is error


def test_pinned_device_intersection_declares_colocation() -> None:
    """Hand-pinning intersecting devices sets IS the sharing declaration."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0]},
                },
            )
        ),
    )

    assert resolved.colocated is True
    assert resolved.requires_trainer_reservation is False


def test_unpinned_rollout_shares_the_trainer_gpu_even_with_spares() -> None:
    """An unpinned rollout lands on the trainer GPU, never on a spare one."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"num_gpus": "auto"},
                },
            )
        ),
    )

    assert resolved.rollout_devices == (0,)
    assert resolved.colocated is True


def test_devices_must_be_subset_of_visible_devices() -> None:
    """Every role spells its own full config path in the failure, not a bare role name."""
    with pytest.raises(
        ValueError,
        match=(
            r"distributed\.resources\.trainer\.devices contains devices outside "
            r"distributed\.resources\.visible_devices"
        ),
    ):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [2]},
                        "rollout": {"num_gpus": 1},
                    },
                )
            ),
        )


def test_gpus_per_engine_derives_engine_count() -> None:
    """4 rollout GPUs at 2 ranks per engine = 2 sequence-parallel engines."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2, 3, 4],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1, 2, 3, 4], "gpus_per_engine": 2},
                },
            )
        ),
    )

    assert resolved.rollout_devices == (1, 2, 3, 4)
    assert resolved.rollout_num_engines == 2
    assert resolved.rollout_gpus_per_engine == 2
    assert "gpus_per_engine=2" in str(resolved)


def test_gpus_per_engine_requires_divisible_gpu_count() -> None:
    with pytest.raises(ValueError, match="not divisible into engines of 2"):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2, 3],
                        "trainer": {"devices": [0]},
                        "rollout": {"devices": [1, 2, 3], "gpus_per_engine": 2},
                    },
                )
            ),
        )


def test_explicit_num_engines_must_match_the_engine_derivation() -> None:
    with pytest.raises(ValueError, match="num_engines must equal rollout GPUs / "):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2, 3, 4],
                        "trainer": {"devices": [0]},
                        "rollout": {
                            "devices": [1, 2, 3, 4],
                            "gpus_per_engine": 2,
                            "num_engines": 4,
                        },
                    },
                )
            ),
        )


def test_gpus_per_engine_rejects_cross_node_and_cpu_fleets() -> None:
    with pytest.raises(ValueError, match="cross_node"):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": "auto",
                        "cross_node": True,
                        "trainer": {"num_gpus": 1},
                        "rollout": {"num_gpus": 2, "gpus_per_engine": 2},
                    },
                )
            ),
        )
    with pytest.raises(ValueError, match="CPU engine"):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0],
                        "trainer": {"num_gpus": 0},
                        "rollout": {"num_gpus": 0, "gpus_per_engine": 2},
                    },
                )
            ),
        )


def test_unpinned_rollout_cannot_outgrow_the_trainer_pool() -> None:
    """A shared rollout needing more GPUs than the trainer owns must pin devices."""
    with pytest.raises(ValueError, match=r"Pin distributed\.resources\.rollout\.devices"):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"num_gpus": 1},
                        "rollout": {"num_gpus": 2},
                    },
                )
            ),
        )


def test_cpu_only_rollout_uses_no_gpu_bundles() -> None:
    """A GPU-less plan resolves to empty device tuples, keeps the requested engine count, and puts
    the trainer on CPU.
    """
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [],
                    "trainer": {"num_gpus": 0},
                    "rollout": {
                        "num_gpus": 0,
                        "num_engines": 2,
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == ()
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_engines == 2
    assert resolved.trainer_torch_device == "cpu"


def test_pinned_devices_supersede_a_zero_gpu_count() -> None:
    """Pinned devices are the authoritative count, even over num_gpus: 0."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 0},
                    "rollout": {
                        "devices": [0],
                        "num_gpus": 0,
                        "num_engines": 1,
                    },
                },
            )
        ),
    )

    assert resolved.rollout_devices == (0,)
    assert resolved.rollout_num_engines == 1


def test_trainer_only_plan_allows_zero_rollout_workers() -> None:
    """A trainer-only plan (zero rollout GPUs and engines) is valid and leaves the rollout device
    set empty.
    """
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 1},
                    "rollout": {
                        "num_gpus": 0,
                        "num_engines": 0,
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_engines == 0
    assert resolved.trainer_torch_device == "cuda:0"


def test_reward_torch_device_uses_the_reserved_local_gpu() -> None:
    """A local reward reservation, not the trainer default, owns model placement."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"devices": [2]},
                },
            )
        ),
    )

    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cuda:2"


def test_reward_torch_device_translates_narrowed_rank_plan_ordinals(cuda_devices) -> None:
    """A rank narrowed to one physical GPU addresses its reward as torch cuda:0.

    Rank-local torchrun launches keep the plan in physical ordinal space
    (visible_devices=[2] on local rank 2) while CUDA_VISIBLE_DEVICES masks the
    process to that single card; the raw physical id is then an invalid torch
    ordinal (measured: non-zero ranks of the hpsv3 fsdp 4-rank smoke).
    """
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [2],
                    "trainer": {"devices": [2]},
                    "reward": {"devices": [2]},
                },
            )
        ),
    )

    # The rank process sees exactly its one masked card.
    cuda_devices(1)
    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cuda:0"


@pytest.mark.parametrize("devices", [(3,), (3, 1), (2, 0, 3)])
def test_auto_resources_preserve_cuda_mask_ids_and_order(
    cuda_devices, monkeypatch, devices
) -> None:
    from vrl.generation.ray.config import RolloutWorkerConfig
    from vrl.ray.placement import GlobalRayPlacementOwner

    # A host whose operator mask names these physical cards, in this order.
    cuda_devices(len(devices))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, devices)))
    resolved = ResourcePlan.from_root(
        parse_config(_cfg({"rollout": {"devices": list(devices[1:] or devices)}})),
    )
    assert resolved.visible_devices == devices
    assert resolved.trainer_devices == (devices[0],)
    assert resolved.trainer_torch_device == "cuda:0"
    for local, physical in enumerate(devices):
        assert resolved._local_torch_ordinal(physical) == local
    worker = RolloutWorkerConfig(
        cpus_per_worker=1.0,
        worker_rpc_timeout_s=60.0,
        generation_stall_timeout_s=60.0,
        pipelined=False,
    )
    owner = GlobalRayPlacementOwner(resolved, worker)
    # A CPU host has no GPU probe actor to run; the probe answer is the one
    # input, with bundles reporting their cards in reverse order.
    probed = {i: gpu for i, gpu in enumerate(reversed(owner.layout.bundle_gpu_ids))}
    roles = owner.assign_roles(probed)
    assert tuple(probed[i] for i in roles["rollout"]) == resolved.rollout_devices


@pytest.mark.parametrize("mask", ["3,3", "3,", "GPU-abc", "3,1,2"])
def test_auto_resources_reject_ambiguous_cuda_mask(cuda_devices, monkeypatch, mask) -> None:
    cuda_devices(2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        ResourcePlan.from_root(parse_config(_cfg({})))


def test_reward_torch_device_without_a_reservation_follows_the_rank_local_trainer() -> None:
    """An unreserved in-process reward shares the caller's actual trainer device."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                },
                reward_components={"aesthetic": 1.0},
            )
        ),
    )

    assert resolved.reward_torch_device(trainer_device="cuda:7") == "cuda:7"


def test_multi_gpu_local_reward_is_rejected_at_resolution() -> None:
    """One driver-local reward runtime cannot consume an actor-pool-shaped plan."""
    with pytest.raises(ValueError, match="exactly one GPU"):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2, 3],
                        "trainer": {"devices": [0]},
                        "rollout": {"devices": [1]},
                        "reward": {"devices": [2, 3]},
                    },
                )
            ),
        )


def test_cross_node_reward_gpu_cannot_share_a_remote_rollout_token() -> None:
    """A shared remote GPU needs a node-local lease, not a budget-token ordinal."""
    with pytest.raises(ValueError, match="require a dedicated Ray bundle"):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": "auto",
                        "cross_node": True,
                        "trainer": {"num_gpus": 1},
                        "rollout": {
                            "devices": [1],
                            "num_engines": 1,
                        },
                        "reward": {"devices": [1]},
                    },
                )
            ),
        )


def test_cross_node_dedicated_reward_gets_its_own_budget_token() -> None:
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": "auto",
                    "cross_node": True,
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": 1},
                    "reward": {"devices": [2]},
                }
            )
        )
    )
    assert resolved.reward_devices == (2,)
    assert not resolved.lifecycle.offload_reward
    assert resolved.reward_torch_device() == "cuda:0"


def test_cross_node_rollout_satisfies_budget_from_explicit_counts() -> None:
    """``cross_node`` with explicit counts assigns trainer and rollout distinct ordinals, not
    colocated, with no trainer reservation.
    """
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": "auto",
                    "cross_node": True,
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": 1, "num_engines": 1},
                },
            )
        ),
    )

    assert resolved.cross_node is True
    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1,)
    assert resolved.rollout_num_engines == 1
    assert resolved.colocated is False
    assert resolved.requires_trainer_reservation is False
    assert resolved.trainer_torch_device == "cuda:0"


def test_cross_node_requires_explicit_rollout_count() -> None:
    with pytest.raises(ValueError, match="cross_node"):
        ResourcePlan.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": "auto",
                        "cross_node": True,
                        "trainer": {"num_gpus": 1},
                        "rollout": {"num_gpus": "auto"},
                    },
                )
            ),
        )


def test_cross_node_kling_recipe_keeps_the_local_reward_on_the_driver() -> None:
    """The shipped two-host recipe has no remote reward token after pool removal."""
    from vrl.config.loading import load_config

    cfg = load_config(
        "experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node",
    )
    resolved = ResourcePlan.from_root(parse_config(cfg))

    assert resolved.cross_node is True
    assert resolved.rollout_devices == (1,)
    assert resolved.reward_devices == ()
    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cuda:0"


def test_lifecycle_plan_resident_when_roles_disjoint() -> None:
    """Fully disjoint trainer/rollout/reward GPUs -> every role resident, no handoff."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"devices": [2]},
                },
            )
        ),
    )

    assert not (set(resolved.reward_devices) & set(resolved.rollout_devices))
    plan = resolved.lifecycle
    assert plan.rollout_mode == "resident"
    assert plan.park_rollout_for_train is False
    assert plan.park_rollout_for_reward is False
    assert plan.offload_reward is False


def test_unpinned_gpu_reward_follows_the_trainer_gpu() -> None:
    """A GPU component with no reservation shares the trainer card, visibly."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0]},
                },
                reward_components={"aesthetic": 1.0},
            )
        ),
    )

    assert resolved.reward_devices == ()
    assert resolved.reward_follows_trainer is True
    assert resolved.lifecycle.park_trainer_for_reward is True
    assert resolved.lifecycle.park_rollout_for_reward is True
    assert resolved.lifecycle.offload_reward is True


def test_explicit_cpu_reward_does_not_create_gpu_handoffs() -> None:
    """CPU execution is a resource fact, independent of parking capability."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                },
                reward_components={"ocr": 1.0},
            )
        ),
    )

    assert resolved.lifecycle.park_trainer_for_reward is False
    assert resolved.lifecycle.park_rollout_for_reward is False
    assert resolved.lifecycle.offload_reward is False
    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cpu"


def test_http_only_reward_owns_no_local_resource_or_handoff() -> None:
    """Inherited local reward reservations disappear for external-only scoring."""

    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    # A reward preset may carry this local default. HTTP deployment
                    # owns its accelerator externally and must not reserve GPU2.
                    "reward": {"devices": [2]},
                },
                reward_components={"unified_reward_video": 1.0},
                reward_inference={
                    "unified_reward_video": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "unified-reward-v1",
                    },
                },
            )
        ),
    )

    assert resolved.reward_devices == ()
    assert resolved.lifecycle.park_trainer_for_reward is False
    assert resolved.lifecycle.park_rollout_for_reward is False
    assert resolved.lifecycle.offload_reward is False
    assert BundleLayout.from_resources(resolved).reward_bundle_indices == ()


def test_mixed_http_and_local_reward_resources_cover_only_local_execution() -> None:
    """A remote sibling does not erase a real local component's CPU execution."""

    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                },
                reward_components={"ocr": 0.5, "unified_reward_video": 0.5},
                reward_inference={
                    "unified_reward_video": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "unified-reward-v1",
                    },
                },
            )
        ),
    )

    assert resolved.reward_devices == ()
    assert resolved.reward_follows_trainer is False
    assert resolved.reward_torch_device() == "cpu"
    # CPU rewards get no bundle: they execute in the driver process.
    assert BundleLayout.from_resources(resolved).reward_bundle_indices == ()


def test_pinned_reward_on_the_trainer_gpu_declares_sharing() -> None:
    """A reward reservation pinned onto the trainer GPU is a sharing declaration."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {
                        "num_gpus": 0,
                    },
                    "reward": {"devices": [0]},
                },
                kling_video_reward=True,
            )
        ),
    )

    assert resolved.reward_devices == (0,)
    assert resolved.lifecycle.park_trainer_for_reward is True


def test_colocated_reward_on_dedicated_gpu_owns_its_own_bundle() -> None:
    """Colocated trainer+rollout on GPU 0; a dedicated reward GPU 1 owns its own
    bundle (the run-level layout targets it directly, no reservation offset)."""

    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0], "num_engines": 1},
                    "reward": {"devices": [1]},
                },
                kling_video_reward=True,
            )
        ),
    )
    assert resolved.colocated is True
    assert resolved.lifecycle.rollout_mode == "on_demand"

    layout = BundleLayout.from_resources(resolved)
    # Colocated trainer+rollout -> no reserved trainer bundle; reward owns a
    # bundle distinct from rollout (its own GPU 1).
    assert layout.bundle_gpu_ids == (0, 1)
    assert layout.reward_bundle_indices != ()
    assert set(layout.rollout_bundle_indices).isdisjoint(layout.reward_bundle_indices)


# ---------------------------------------------------------------- fsdp (P2)


def _cfg_training(resources: dict, training: dict) -> object:
    return OmegaConf.create(
        {
            "distributed": {
                "resources": resources,
                "rollout": {},
                "training": training,
            },
        },
    )


def test_fsdp_trainer_allows_multi_gpu_disjoint_from_rollout() -> None:
    """fsdp lifts the single-device cap: trainer can own N GPUs."""
    resolved = ResourcePlan.from_root(
        parse_config(
            _cfg_training(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0, 1]},
                    "rollout": {"devices": [2]},
                },
                {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
            )
        ),
    )
    assert resolved.trainer_devices == (0, 1)
    assert resolved.rollout_devices == (2,)


def test_fsdp_trainer_count_must_equal_world_size() -> None:
    """fsdp trainer device count must match num_nodes*gpus_per_node."""
    with pytest.raises(ValueError, match=r"must own num_nodes\*gpus_per_node=2"):
        ResourcePlan.from_root(
            parse_config(
                _cfg_training(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [0]},
                        "rollout": {"devices": [1]},
                    },
                    {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
                )
            ),
        )


def test_fsdp_trainer_must_be_disjoint_from_rollout_even_with_overlap() -> None:
    """fsdp rejects trainer/rollout GPU overlap even when pinned deliberately."""
    with pytest.raises(ValueError, match="fsdp requires trainer GPUs disjoint"):
        ResourcePlan.from_root(
            parse_config(
                _cfg_training(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [0, 1]},
                        "rollout": {"devices": [1]},
                    },
                    {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
                )
            ),
        )


def test_single_process_still_rejects_multi_gpu_trainer() -> None:
    """The single-device cap is preserved for single_process (default)."""
    with pytest.raises(ValueError, match="0 or 1 GPU"):
        ResourcePlan.from_root(
            parse_config(
                _cfg_training(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [0, 1]},
                        "rollout": {"num_gpus": 0, "num_engines": 1},
                    },
                    {"strategy": "single_process"},
                )
            ),
        )


# ── Symmetric colocated DDP (SPRINT_symmetric_colocated_ddp) ──


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_multi_rank_colocate_resolves_per_rank_local_single_gpu(strategy: str) -> None:
    """Symmetric colocated multi-rank training: each rank resolves only its LOCAL
    single GPU (trainer + colocated rollout on it). world_size drives only the
    collectives, NOT the per-rank GPU plan -- signaled by an unpinned rollout, so
    fsdp's world-covering asymmetric trainer rule and the disjoint rule do not
    apply (SPRINT_multi_gpu_training Phase 4)."""
    resolved = ResourcePlan.from_root(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {
                        "training": {"strategy": strategy, "num_nodes": 2, "gpus_per_node": 1},
                        "resources": {
                            "visible_devices": [0],
                            "trainer": {"num_gpus": 1},
                        },
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (0,)  # colocated on the local GPU
    assert resolved.colocated is True
    assert resolved.lifecycle.rollout_mode == "on_demand"
    assert resolved.cross_node is False  # per-rank-local: no shared Ray cluster


# ── cosmos async-reward recipe (3-GPU disjoint reward overlap) ───────────────


def test_each_role_parks_with_its_default_and_refuses_other_mechanisms() -> None:
    """``parking.mechanism`` resolves to the role's default; a mechanism the role
    cannot use, or a disk directory outside the trainer, fails at parse.
    """

    resources = ResourcePlan.from_root(
        parse_config(_cfg({"reward": {"parking": {"mechanism": "reload"}}}))
    )
    assert resources.rollout_parking.mechanism == "cumem"
    assert resources.reward_parking.mechanism == "reload"
    trainer = parse_config(_cfg({})).distributed.resources.trainer
    assert trainer.parking.mechanism == "move"

    refused = {"trainer": "reload", "rollout": "reload", "reward": "move"}
    for role, mechanism in refused.items():
        with pytest.raises(ValueError, match=rf"{role}\.parking\.mechanism must be one of"):
            parse_config(_cfg({role: {"parking": {"mechanism": mechanism}}}))
    with pytest.raises(ValueError, match="disk_directory"):
        parse_config(_cfg({"reward": {"parking": {"disk_directory": "/mnt/nvme/x"}}}))
