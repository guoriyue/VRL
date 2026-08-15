"""GenerationRuntime and weight-sync version contract tests.

Orchestration reads the version properties declared by each concrete boundary
instead of probing nested runtime internals.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.generation.protocols import GenerationRuntime
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer, WeightSyncer


def _runtime(
    *,
    deferred: bool = False,
    colocated: bool = False,
) -> RayGenerationRuntime:
    session = RayGenerationSession(
        executor=object(),
        weight_sync=None,
        owned_engines=[],
    )

    if deferred:

        async def create_session() -> RayGenerationSession:
            return session

        return RayGenerationRuntime(
            session=None,
            session_factory=create_session,
            supports_weight_sync=False,
            colocated=colocated,
        )
    return RayGenerationRuntime(
        session=session,
        colocated=colocated,
    )


# Note: the release-before-reward decision is no longer a runtime method; it is
# derived from GPU topology into the RayLifecyclePlan and read by the collector.
# See tests/ray/test_resources.py (plan derivation) and
# tests/rollouts/collector/test_runtime.py (collector consumption).


# --------------------------------------------------------------------------
# requires_driver_model_offload
# --------------------------------------------------------------------------
def test_persistent_runtime_does_not_require_driver_offload() -> None:
    runtime = _runtime()
    assert runtime.requires_driver_model_offload is False


@pytest.mark.parametrize(
    ("colocated", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
def test_deferred_runtime_driver_offload_requirement(colocated, expected) -> None:
    runtime = _runtime(deferred=True, colocated=colocated)
    assert runtime.requires_driver_model_offload is expected


def test_concrete_runtimes_satisfy_generation_runtime_structurally() -> None:
    persistent = _runtime()
    deferred = _runtime(deferred=True)
    assert isinstance(persistent, GenerationRuntime)
    assert isinstance(deferred, GenerationRuntime)


def test_runtimes_expose_explicit_activation_and_offload() -> None:
    runtime = _runtime(deferred=True)
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
    assert syncer.current_policy_version == 7


def test_base_weight_syncer_defaults_to_no_version() -> None:
    class _NullSyncer(WeightSyncer):
        async def push(self, state_dict):  # pragma: no cover - contract only
            del state_dict

        async def pull(self):  # pragma: no cover - contract only
            return {}

    assert _NullSyncer().current_policy_version is None
