"""Tests for trainer-to-rollout weight sync adapters."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
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
    """Checks Ray runtime weight syncer pushes CPU state with monotonic versions."""
    runtime = _RuntimeWithSync()
    syncer = RayRuntimeWeightSyncer(runtime)

    asyncio.run(syncer.push({"weight": torch.ones(2)}))
    asyncio.run(syncer.push({"weight": torch.full((2,), 2.0)}))

    assert [version for _, version in runtime.calls] == [1, 2]
    assert runtime.current_policy_version == 2
    assert all(call[0]["weight"].device.type == "cpu" for call in runtime.calls)
    assert torch.equal(runtime.calls[-1][0]["weight"], torch.full((2,), 2.0))


def test_ray_runtime_weight_syncer_serializes_concurrent_push_versions() -> None:
    """Checks Ray runtime weight syncer serializes concurrent push versions."""

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


def test_build_runtime_weight_syncer_requires_explicit_runtime_capability() -> None:
    """Checks the adapter reads capability instead of a current inner handle."""
    assert build_runtime_weight_syncer(_RuntimeWithSync()) is not None
    assert build_runtime_weight_syncer(_RuntimeWithoutSync()) is None


def test_flatten_trainable_module_state_prefixes_module_names() -> None:
    """Checks flatten trainable module state prefixes module names."""
    module = torch.nn.Linear(2, 1, bias=False)
    state = flatten_trainable_module_state({"adapter": module})

    assert set(state) == {"adapter.weight"}
    assert torch.equal(state["adapter.weight"], module.weight)


def test_flatten_trainable_module_state_skips_frozen_parameters() -> None:
    """Checks flatten trainable module state skips frozen parameters."""
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
    wrapper's own ``state_dict()`` keys carry a ``module.`` prefix."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.module = inner


class _CompiledLike(torch.nn.Module):
    """Mimics torch.compile's OptimizedModule: inner under ``_orig_mod``."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner


def test_flatten_strips_ddp_module_prefix() -> None:
    """A `.module`-wrapped trainable root must not leak `module.` to rollout."""
    inner = torch.nn.Linear(2, 1, bias=False)
    wrapped = _DDPLike(inner)
    assert "module.weight" in wrapped.state_dict()  # the wrapper prefixes keys

    state = flatten_trainable_module_state({"adapter": wrapped})

    assert set(state) == {"adapter.weight"}  # prefix stripped
    assert torch.equal(state["adapter.weight"], inner.weight)


def test_flatten_strips_compile_orig_mod_prefix() -> None:
    """A compiled-like `_orig_mod` wrapper keeps the existing clean-key behavior."""
    inner = torch.nn.Linear(2, 1, bias=False)
    state = flatten_trainable_module_state({"adapter": _CompiledLike(inner)})

    assert set(state) == {"adapter.weight"}
    assert torch.equal(state["adapter.weight"], inner.weight)


def test_flatten_strips_nested_compile_and_ddp_wrappers() -> None:
    """Wrapper nesting (compile(DDP(m))) unwraps fully regardless of order."""
    inner = torch.nn.Linear(2, 1, bias=False)
    state = flatten_trainable_module_state({"adapter": _CompiledLike(_DDPLike(inner))})

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
    """Checks build trainable state sync getter reads bundle trainable modules."""
    bundle = _Bundle()
    getter = build_trainable_state_sync_getter(bundle)
    state = getter()

    assert set(state) == {"adapter.weight"}
    assert state["adapter.weight"].shape == (1, 2)
