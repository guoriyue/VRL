"""Tests for trainer-to-rollout weight sync adapters."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.trainers.weight_sync import (
    RayRuntimeWeightSyncer,
    flatten_trainable_module_state,
    require_trainable_modules,
)


class _RuntimeWithSync:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.calls: list[tuple[dict[str, Any], int]] = []

    async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
        self.calls.append((state_ref, policy_version))
        self.current_policy_version = policy_version


def test_syncer_hands_on_the_prepared_snapshot_under_growing_versions() -> None:
    """The runtime receives the prepared snapshot itself, under a growing version.

    The strategy export already made the one immutable host copy; a second copy
    in push would double host memory and time for a full-parameter policy.
    """
    runtime = _RuntimeWithSync()
    syncer = RayRuntimeWeightSyncer(runtime)
    first = {"weight": torch.ones(2)}

    asyncio.run(syncer.push(first))
    asyncio.run(syncer.push({"weight": torch.full((2,), 2.0)}))

    assert [version for _, version in runtime.calls] == [1, 2]
    assert runtime.current_policy_version == 2
    assert runtime.calls[0][0]["weight"] is first["weight"]


def test_ray_runtime_weight_syncer_serializes_concurrent_push_versions() -> None:
    """Two overlapping pushes must not both claim the same version number."""

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


def test_flatten_trainable_module_state_skips_frozen_parameters() -> None:
    """Frozen base weights never change, so shipping them every step is wasted wire."""
    module = torch.nn.Linear(2, 1, bias=True)
    module.bias.requires_grad_(False)
    state = flatten_trainable_module_state({"adapter": module})

    assert set(state) == {"adapter.weight"}


# --------------------------------------------------------------------------
# State export contract (readiness P5): rollout payload keys must be flat
# policy-facing names — no training-time wrapper prefix leaks through.
# --------------------------------------------------------------------------
class _DDPLike(torch.nn.parallel.DistributedDataParallel):
    """Mimics DDP / FSDP1: the inner module sits under ``.module`` so the
    wrapper's own ``state_dict()`` keys carry a ``module.`` prefix.

    A real ``DistributedDataParallel`` needs an initialized process group, which
    is a lot of machinery for one key-prefix assertion. The real wrapper's export
    is covered by the gloo tests in ``tests/trainers/test_ddp.py`` (default lane):
    deleting the ``.module`` peel from ``unwrap_compile_and_ddp`` reddens all four.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        torch.nn.Module.__init__(self)
        self.module = inner


def test_flatten_strips_nested_compile_inside_a_ddp_wrapper() -> None:
    """Nesting must peel both wrappers, and only this order can prove it.

    ``compile`` has to sit *inside*: ``OptimizedModule.__getattr__`` forwards to
    ``_orig_mod``, so with compile on the outside ``getattr(module, "module")``
    reaches DDP's inner module directly and the compile peel becomes
    unobservable — a mutation that stops stripping ``_orig_mod.`` still passes.
    With compile inside, the same mutation yields ``adapter._orig_mod.weight``.
    """
    inner = torch.nn.Linear(2, 1, bias=False)
    wrapped = _DDPLike(torch.compile(inner))
    assert "module._orig_mod.weight" in wrapped.state_dict()  # both prefixes present

    state = flatten_trainable_module_state({"adapter": wrapped})

    assert set(state) == {"adapter.weight"}
    assert torch.equal(state["adapter.weight"], inner.weight)


def test_flatten_all_frozen_module_fails_fast() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    module.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        flatten_trainable_module_state({"adapter": module})


def test_flatten_reads_live_weights_rather_than_an_earlier_snapshot() -> None:
    """Rollout sync must ship what the trainer holds now, not what it held before."""
    bundle = SimpleNamespace(trainable_modules={"adapter": torch.nn.Linear(2, 1, bias=False)})
    modules = require_trainable_modules(bundle)

    state = flatten_trainable_module_state(modules)
    assert set(state) == {"adapter.weight"}
    assert state["adapter.weight"].shape == (1, 2)

    with torch.no_grad():
        modules["adapter"].weight.fill_(7.0)
    assert flatten_trainable_module_state(modules)["adapter.weight"].eq(7.0).all()
