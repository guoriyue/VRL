"""GenerationRuntime / PolicyVersionProvider protocol contract (SOLID sub-sprint A).

Orchestration asks runtimes and weight syncers questions through these
protocols instead of probing their internal structure with getattr. These
tests pin the contract: every production implementation must answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import PolicyVersionProvider
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.ray.placement import RolePlacement
from vrl.ray.resources import ActorLeasePolicy, PhaseHandoffPolicy, RayLifecyclePlan
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer, WeightSyncer


def _on_demand_runtime(
    *,
    allow_driver_gpu_overlap: bool = False,
    colocated: bool = False,
) -> RayGenerationRuntime:
    config = RayGenerationConfig(
        allow_driver_gpu_overlap=allow_driver_gpu_overlap,
        # Use the real lifecycle types so the fixture cannot omit the canonical
        # on-demand selection that production resolves from shared-GPU topology.
        resources=SimpleNamespace(
            colocated=colocated,
            lifecycle=RayLifecyclePlan(
                rollout=ActorLeasePolicy(mode="on_demand"),
                reward=ActorLeasePolicy(mode="resident"),
                handoff=PhaseHandoffPolicy(
                    release_rollout_before_train=True,
                    release_rollout_before_reward=True,
                    release_reward_after_score=False,
                ),
            ),
        ),
    )
    return RayGenerationRuntime.with_on_demand_activation(
        config,
        RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family="test",
                task="test",
                runtime_builder="tests:runtime_builder",
                executor_cls="tests:executor_cls",
            ),
            gatherer=object(),
        ),
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
def test_on_demand_runtime_colocation(overlap, colocated, expected) -> None:
    runtime = _on_demand_runtime(
        allow_driver_gpu_overlap=overlap,
        colocated=colocated,
    )
    assert runtime.is_colocated() is expected


# --------------------------------------------------------------------------
# PolicyVersionProvider
# --------------------------------------------------------------------------
def test_runtimes_satisfy_policy_version_provider() -> None:
    persistent = RayGenerationRuntime(executor=object())
    on_demand = _on_demand_runtime()
    assert isinstance(persistent, PolicyVersionProvider)
    assert isinstance(on_demand, PolicyVersionProvider)


def test_runtimes_expose_explicit_activation_and_offload() -> None:
    runtime = _on_demand_runtime()
    assert callable(runtime.activate)
    assert callable(runtime.offload)


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
