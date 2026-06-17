"""GenerationRuntime / PolicyVersionProvider protocol contract (SOLID sub-sprint A).

Orchestration asks runtimes and weight syncers questions through these
protocols instead of probing their internal structure with getattr. These
tests pin the contract: every production implementation must answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.generation.protocols import PolicyVersionProvider
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.ray.placement import RolePlacement
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer, WeightSyncer


def _release_after_collect_runtime(
    *,
    allow_driver_gpu_overlap: bool = False,
    colocated: bool = False,
) -> RayGenerationRuntime:
    config = RayGenerationConfig(
        allow_driver_gpu_overlap=allow_driver_gpu_overlap,
        resources=SimpleNamespace(colocated=colocated),
    )
    return RayGenerationRuntime.with_release_after_collect(
        config,
        launch_contract=SimpleNamespace(policy_version=None),
        gatherer=object(),
        placement=RolePlacement(
            placement_group=object(),
            bundle_indices=(),
            expected_gpu_ids=(),
        ),
    )


# Note: the release-before-reward decision is no longer a runtime method; it is
# derived from GPU topology into the RayLifecyclePlan and read by the collector.
# See tests/ray/test_resources.py (plan derivation) and
# tests/rollouts/collector/test_runtime.py (collector consumption).


# --------------------------------------------------------------------------
# is_colocated
# --------------------------------------------------------------------------
def test_persistent_runtime_is_not_colocated() -> None:
    runtime = RayGenerationRuntime(executor=object())
    assert runtime.is_colocated() is False


@pytest.mark.parametrize(
    ("overlap", "colocated", "expected"),
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_release_after_collect_runtime_colocation(overlap, colocated, expected) -> None:
    runtime = _release_after_collect_runtime(
        allow_driver_gpu_overlap=overlap,
        colocated=colocated,
    )
    assert runtime.is_colocated() is expected


# --------------------------------------------------------------------------
# PolicyVersionProvider
# --------------------------------------------------------------------------
def test_runtimes_satisfy_policy_version_provider() -> None:
    persistent = RayGenerationRuntime(executor=object())
    release_after_collect = _release_after_collect_runtime()
    assert isinstance(persistent, PolicyVersionProvider)
    assert isinstance(release_after_collect, PolicyVersionProvider)


def test_weight_syncer_reports_its_runtime_version() -> None:
    async def _update_weights(state, version):  # pragma: no cover - signature only
        del state, version

    runtime = SimpleNamespace(
        update_weights=_update_weights,
        current_policy_version=7,
    )
    syncer = RayRuntimeWeightSyncer(runtime)
    assert isinstance(syncer, PolicyVersionProvider)
    assert syncer.current_policy_version == 7


def test_base_weight_syncer_defaults_to_no_version() -> None:
    class _NullSyncer(WeightSyncer):
        async def push(self, state_dict):  # pragma: no cover - contract only
            del state_dict

        async def pull(self):  # pragma: no cover - contract only
            return {}

    assert _NullSyncer().current_policy_version is None
