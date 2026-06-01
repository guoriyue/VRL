"""Tests for trainer-to-rollout weight sync adapters."""

from __future__ import annotations

import asyncio
from typing import Any

import torch

from vrl.trainers.weight_sync import (
    RayRuntimeWeightSyncer,
    build_runtime_weight_syncer,
    build_trainable_state_sync_getter,
    flatten_trainable_module_state,
)


class _RuntimeWithSync:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.weight_sync = object()
        self.calls: list[tuple[dict[str, Any], int]] = []

    async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
        self.calls.append((state_ref, policy_version))
        self.current_policy_version = policy_version


class _RuntimeWithoutSync:
    async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
        del state_ref, policy_version


class _Bundle:
    def __init__(self) -> None:
        self.trainable_modules = {
            "adapter": torch.nn.Linear(2, 1, bias=False),
        }


def test_ray_runtime_weight_syncer_pushes_cpu_state_with_monotonic_versions() -> None:
    runtime = _RuntimeWithSync()
    syncer = RayRuntimeWeightSyncer(runtime)

    asyncio.run(syncer.push({"weight": torch.ones(2)}))
    asyncio.run(syncer.push({"weight": torch.full((2,), 2.0)}))

    assert [version for _, version in runtime.calls] == [1, 2]
    assert runtime.current_policy_version == 2
    assert all(call[0]["weight"].device.type == "cpu" for call in runtime.calls)
    assert torch.equal(asyncio.run(syncer.pull())["weight"], torch.full((2,), 2.0))


def test_ray_runtime_weight_syncer_serializes_concurrent_push_versions() -> None:
    class _SlowRuntime(_RuntimeWithSync):
        async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
            await asyncio.sleep(0)
            await super().update_weights(state_ref, policy_version)

    async def _push_concurrently() -> _SlowRuntime:
        runtime = _SlowRuntime()
        syncer = RayRuntimeWeightSyncer(runtime)
        await asyncio.gather(
            syncer.push({"weight": torch.ones(1)}),
            syncer.push({"weight": torch.full((1,), 2.0)}),
        )
        return runtime

    runtime = asyncio.run(_push_concurrently())

    assert [version for _, version in runtime.calls] == [1, 2]
    assert runtime.current_policy_version == 2


def test_build_runtime_weight_syncer_requires_runtime_weight_sync_handle() -> None:
    assert build_runtime_weight_syncer(_RuntimeWithSync()) is not None
    assert build_runtime_weight_syncer(_RuntimeWithoutSync()) is None


def test_flatten_trainable_module_state_prefixes_module_names() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    state = flatten_trainable_module_state({"adapter": module})

    assert set(state) == {"adapter.weight"}
    assert torch.equal(state["adapter.weight"], module.weight)


def test_flatten_trainable_module_state_skips_frozen_parameters() -> None:
    module = torch.nn.Linear(2, 1, bias=True)
    module.bias.requires_grad_(False)
    state = flatten_trainable_module_state({"adapter": module})

    assert set(state) == {"adapter.weight"}


def test_build_trainable_state_sync_getter_reads_bundle_trainable_modules() -> None:
    bundle = _Bundle()
    getter = build_trainable_state_sync_getter(bundle)
    state = getter()

    assert set(state) == {"adapter.weight"}
    assert state["adapter.weight"].shape == (1, 2)
