"""GenerationRuntime / PolicyVersionProvider protocol contract (SOLID sub-sprint A).

Orchestration asks runtimes and weight syncers questions through these
protocols instead of probing their internal structure with getattr. These
tests pin the contract: every production implementation must answer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import GenerationRuntime, PolicyVersionProvider
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.ray.on_demand_runtime import _OnDemandRayGenerationRuntime
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.ray.placement import RolePlacement
from vrl.ray.resources import resolve_distributed_resources
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer, WeightSyncer


class _Gatherer:
    def gather_chunks(self, *_args: Any) -> Any:
        raise AssertionError("protocol fixture must not gather chunks")


def _on_demand_runtime(
    *,
    colocated: bool = False,
) -> _OnDemandRayGenerationRuntime:
    rollout = (
        {
            "gpu_pool": "trainer",
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
        }
        if colocated
        else {
            "devices": [1],
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
        }
    )
    reward = (
        None
        if colocated
        else {
            "devices": [1],
            "num_gpus": 1,
            "gpus_per_worker": 1,
            "num_workers": 1,
            "gpu_pool": "rollout",
        }
    )
    resources = {
        "visible_devices": [0] if colocated else [0, 1],
        "trainer": {"devices": [0]},
        "rollout": rollout,
    }
    if reward is not None:
        resources["reward"] = reward
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": resources,
                "rollout": {},
            },
        },
    )
    config = RayGenerationConfig.from_cfg(
        cfg,
        resources=resolve_distributed_resources(cfg),
    )
    return _OnDemandRayGenerationRuntime(
        config,
        RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family="sd3_5",
                model_build={},
                expected_model_identity={"schema": "test"},
            ),
            gatherer=_Gatherer(),
        ),
        launcher=RayGenerationLauncher(init_ray=False),
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
    ("colocated", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
def test_on_demand_runtime_colocation(colocated, expected) -> None:
    runtime = _on_demand_runtime(colocated=colocated)
    assert runtime.is_colocated() is expected


# --------------------------------------------------------------------------
# PolicyVersionProvider
# --------------------------------------------------------------------------
def test_runtimes_satisfy_policy_version_provider() -> None:
    persistent = RayGenerationRuntime(executor=object())
    on_demand = _on_demand_runtime()
    assert isinstance(persistent, PolicyVersionProvider)
    assert isinstance(on_demand, PolicyVersionProvider)


def test_concrete_runtimes_satisfy_generation_runtime_structurally() -> None:
    persistent = RayGenerationRuntime(executor=object())
    on_demand = _on_demand_runtime()
    assert isinstance(persistent, GenerationRuntime)
    assert isinstance(on_demand, GenerationRuntime)


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
