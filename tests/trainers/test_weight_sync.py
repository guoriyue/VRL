"""Tests for trainer-to-rollout weight sync adapters."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from vrl.trainers.weight_sync import (
    RayRuntimeWeightSyncer,
    build_trainable_state_sync_getter,
    flatten_trainable_module_state,
)


class _RuntimeWithSync:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.supports_weight_sync = True
        self.calls: list[tuple[dict[str, Any], int]] = []

    async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
        self.calls.append((state_ref, policy_version))
        self.current_policy_version = policy_version


class _RuntimeWithoutSync:
    supports_weight_sync = False

    async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
        del state_ref, policy_version


class _Bundle:
    def __init__(self) -> None:
        self.trainable_modules = {
            "adapter": torch.nn.Linear(2, 1, bias=False),
        }


def test_ray_runtime_weight_syncer_pushes_cpu_state_with_monotonic_versions() -> None:
    """The rollout side must receive CPU tensors and a version that only ever grows."""
    runtime = _RuntimeWithSync()
    syncer = RayRuntimeWeightSyncer(runtime)

    asyncio.run(syncer.push({"weight": torch.ones(2)}))
    asyncio.run(syncer.push({"weight": torch.full((2,), 2.0)}))

    assert [version for _, version in runtime.calls] == [1, 2]
    assert runtime.current_policy_version == 2
    assert all(call[0]["weight"].device.type == "cpu" for call in runtime.calls)
    assert torch.equal(runtime.calls[-1][0]["weight"], torch.full((2,), 2.0))


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


def test_weight_syncer_if_supported_requires_explicit_runtime_capability() -> None:
    """Checks the adapter reads capability instead of a current inner handle."""
    assert RayRuntimeWeightSyncer.if_supported(_RuntimeWithSync()) is not None
    assert RayRuntimeWeightSyncer.if_supported(_RuntimeWithoutSync()) is None


def test_flatten_trainable_module_state_prefixes_module_names() -> None:
    """Payload keys are ``<bundle key>.<parameter>``, so the receiver can route them."""
    module = torch.nn.Linear(2, 1, bias=False)
    state = flatten_trainable_module_state({"adapter": module})

    assert set(state) == {"adapter.weight"}
    assert torch.equal(state["adapter.weight"], module.weight)


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
class _DDPLike(torch.nn.Module):
    """Mimics DDP / FSDP1: the inner module sits under ``.module`` so the
    wrapper's own ``state_dict()`` keys carry a ``module.`` prefix.

    A real ``DistributedDataParallel`` needs an initialized process group, which
    is a lot of machinery for one key-prefix assertion. The real wrapper's export
    is covered by the gloo tests in ``tests/trainers/test_ddp.py`` (default lane):
    deleting the ``.module`` peel from ``unwrap_compile_and_ddp`` reddens all four.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.module = inner


def test_flatten_strips_ddp_module_prefix() -> None:
    """A `.module`-wrapped trainable root must not leak `module.` to rollout."""
    inner = torch.nn.Linear(2, 1, bias=False)
    wrapped = _DDPLike(inner)
    assert "module.weight" in wrapped.state_dict()  # the wrapper prefixes keys

    state = flatten_trainable_module_state({"adapter": wrapped})

    assert set(state) == {"adapter.weight"}  # prefix stripped
    assert torch.equal(state["adapter.weight"], inner.weight)


def test_flatten_strips_compile_orig_mod_prefix() -> None:
    """A real ``torch.compile``'d policy — the default in live configs — exports clean keys.

    ``torch.compile`` is free until the module is called, so the compiled shape is
    torch's own ``OptimizedModule`` rather than a hand-written ``_orig_mod`` shell.
    """
    inner = torch.nn.Linear(2, 1, bias=False)
    compiled = torch.compile(inner)
    assert "_orig_mod.weight" in compiled.state_dict()  # the wrapper prefixes keys

    state = flatten_trainable_module_state({"adapter": compiled})

    assert set(state) == {"adapter.weight"}
    assert torch.equal(state["adapter.weight"], inner.weight)


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


def test_flatten_empty_mapping_fails_fast() -> None:
    with pytest.raises(ValueError, match="non-empty mapping"):
        flatten_trainable_module_state({})


def test_flatten_all_frozen_module_fails_fast() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    module.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        flatten_trainable_module_state({"adapter": module})


def test_build_trainable_state_sync_getter_reads_bundle_trainable_modules() -> None:
    """The getter reads the bundle at call time, so it tracks live weights, not a snapshot."""
    bundle = _Bundle()
    getter = build_trainable_state_sync_getter(bundle)
    state = getter()

    assert set(state) == {"adapter.weight"}
    assert state["adapter.weight"].shape == (1, 2)
