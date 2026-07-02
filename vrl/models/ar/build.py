"""Shared AR runtime-bundle assembly.

Every AR family (janus_pro, nextstep_1, ...) assembled its rollout and replay
``RuntimeBundle`` with the same ~18 lines, differing only in the capability
constant and the rollout/replay split. That assembly lives here once; a
family's ``runtime.py`` keeps a thin ``build_<family>_*`` stub that constructs
its model (the genuinely family-specific part — config-from-spec plus the
model class) and passes it in with the capability.

Like the diffusion counterpart (``vrl.models.diffusion.build``), the family
stub — not this module — stays the registry dispatch target, and this module
never imports family model classes. AR bundles carry no scheduler and register
the whole wrapper as the single trainable module under ``"model"``; families
needing a different shape keep their own builder instead of growing knobs here.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.capabilities import FamilyCapability
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)


def build_ar_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    model: Any,
    capability: FamilyCapability,
    replay: bool = False,
) -> RuntimeBundle:
    """Assemble the canonical AR bundle around a family-constructed model.

    ``replay=True`` selects the trainer-side shape: no raw handle, chunked
    execution off, and minimal bundle metadata (the replay model loads no
    generation-only modules). The rollout shape exposes the model as its own
    raw handle and advertises chunked execution.
    """

    return RuntimeBundle(
        model=model,
        trainable_modules={"model": model},
        scheduler=None,
        raw_handle=None if replay else model,
        runtime_caps={
            "family_capability": capability.to_dict(),
            "supports_chunked_execution": not replay,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": capability.family,
            "ar_task": spec.ar_task,
            "use_lora": spec.use_lora,
            **(
                minimal_replay_bundle_metadata()
                if replay
                else full_generation_bundle_metadata()
            ),
        },
    )


__all__ = ["build_ar_runtime_bundle"]
