"""Typed inputs shared by Ray generation launch and on-demand activation."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from vrl.generation.execution.rank_group import RankGroupSpec
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import GenerationBatchGatherer

if TYPE_CHECKING:
    from vrl.config.builders import BuiltConfigs
    from vrl.ray.resources import ResolvedDistributedResources
    from vrl.run import ResolvedModel


@dataclass(frozen=True, slots=True)
class RayGenerationLaunchInputs:
    """Serializable rank build contract plus registry-owned batch gatherer.

    The driver executor and single-engine pipelined path receive the same
    binding; neither side re-resolves family identity after composition.
    ``rank_group`` is None for single-rank engines; the launcher stamps a
    per-rank rendezvous spec when an engine spans multiple GPUs.
    """

    launch_contract: GenerationRuntimeLaunchContract
    gatherer: GenerationBatchGatherer
    rank_group: RankGroupSpec | None = None

    def __post_init__(self) -> None:
        try:
            pickle.dumps(self)
        except Exception as error:
            raise TypeError(
                "RayGenerationLaunchInputs must be pickle-serializable",
            ) from error


def build_launch_inputs(
    built: BuiltConfigs,
    resources: ResolvedDistributedResources,
    replay_model: ResolvedModel,
) -> RayGenerationLaunchInputs:
    """The Ray worker contract of one online run, checked against its replay model.

    The rollout build is resolved on the rollout device and must carry the
    replay model's checkpoint identity; the Ray launcher consumes the result
    and topology only, never trainer schedule or model YAML.
    """

    import torch

    from vrl.models import checkpoint_identity
    from vrl.models.dtypes import dtype_to_wire_name

    family = built.family
    if replay_model.entry.family != family.family:
        raise ValueError(
            "replay model family does not match the online run: "
            f"{replay_model.entry.family!r} != {family.family!r}",
        )
    replay_model.build.require_replay()
    trainer = built.trainer
    if trainer is None:
        raise ValueError("online generation launch requires a trainer config")

    root = built.root
    runtime_device = torch.device("cuda" if resources.rollout_devices else "cpu")
    build = family.resolve_model_build(root, runtime_device, precision=built.precision)
    rollout_model_identity = checkpoint_identity.resolve_checkpoint_model_identity(build)
    if rollout_model_identity != replay_model.identity:
        raise ValueError(
            "rollout model identity does not match the driver replay model "
            "identity before Ray worker launch: "
            f"replay={replay_model.identity!r}, rollout={rollout_model_identity!r}",
        )

    model_build_payload = asdict(build)
    # Family is the top-level launch identity and is restored worker-side.
    model_build_payload.pop("family", None)
    model_build_payload["device"] = str(model_build_payload["device"])
    model_build_payload["parameter_dtype"] = dtype_to_wire_name(
        model_build_payload["parameter_dtype"],
    )
    rollout = model_build_payload.get("rollout")
    if rollout is not None:
        rollout["prompt_encoder_dtype"] = dtype_to_wire_name(
            rollout["prompt_encoder_dtype"],
        )
        # Absence is the wire representation of the universal no-offload
        # default. Only a selected residency mode needs to cross Ray.
        if rollout.get("pipeline_offload_mode") == "none":
            rollout.pop("pipeline_offload_mode")

    torch_profiler = root.rollout.torch_profiler if root.rollout is not None else None
    if torch_profiler is not None and not torch_profiler.output_dir:
        run_output_dir = root.trainer.output_dir if root.trainer is not None else None
        torch_profiler = replace(
            torch_profiler,
            output_dir=str(Path(run_output_dir or "outputs") / "torch_profiler"),
        )

    return RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=family.family,
            model_build=model_build_payload,
            expected_model_identity=replay_model.identity,
            executor_kwargs=family.executor_kwargs(root),
            policy_version=0,
            torch_profiler={} if torch_profiler is None else asdict(torch_profiler),
            versioned_weight_sync=trainer.versioned_weight_sync,
            # A rollout that hands its GPUs to another role between phases
            # parks its workers' memory at the handoff.
            sleep_offload=(
                resources.lifecycle.rollout_mode == "on_demand" and bool(resources.rollout_devices)
            ),
            parking_mechanism=str(resources.rollout_parking.mechanism),
        ),
        gatherer=family.new_gatherer(),
    )


__all__ = ["RayGenerationLaunchInputs", "build_launch_inputs"]
