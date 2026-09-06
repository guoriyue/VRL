"""Rollout-to-generation request adapter for collectors."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, NamedTuple

from vrl.config.schema import require_sampling_overrides
from vrl.generation import GenerationInput, GenerationRequest
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.models.families.registry import ModelFamilyEntry
from vrl.models.families.semantics import task_type_for
from vrl.rollouts.collector.config import RolloutCollectorConfig

_DENOISE_FIELDS = frozenset(item.name for item in fields(DenoiseRequestOptions))


class CollectorRequest(NamedTuple):
    request: GenerationRequest
    metadata: dict[str, Any]


class GenerationRequestBuilder:
    """Build ``GenerationRequest`` payloads from resolved rollout config.

    Callers hand over ready ``GenerationInput`` conditioning (or bare prompt
    strings) plus one opaque reward-metadata dict; this builder only applies
    family defaults and sampling config. It never picks example fields out of
    an untyped kwargs dict — ``PromptExample.generation_input()`` /
    ``reward_metadata()`` own that mapping.
    """

    def __init__(
        self,
        *,
        entry: ModelFamilyEntry,
        config: RolloutCollectorConfig,
    ) -> None:
        self.entry = entry
        self.config = config

    def build(
        self,
        inputs: list[GenerationInput | str],
        group_size: int,
        *,
        metadata: Mapping[str, Any] | None = None,
        request_overrides: Mapping[str, Any] | None = None,
        runtime_debug: bool = False,
        policy_version: int | None = None,
    ) -> CollectorRequest:
        sampling = {
            str(field_name): list(value) if isinstance(value, tuple) else value
            for field_name, value in self.config.request_sampling.items()
        }
        overrides = dict(request_overrides or {})
        # A per-prompt override of a denoise knob lands on the typed options
        # (re-validated by replace); the rest must be the family's own sampling
        # vocabulary, so an unknown key fails here rather than riding the wire.
        denoise = self.config.denoise
        denoise_overrides = {
            name: overrides.pop(name) for name in tuple(overrides) if name in _DENOISE_FIELDS
        }
        if denoise_overrides:
            denoise = replace(denoise or DenoiseRequestOptions(), **denoise_overrides)
        if overrides:
            sampling.update(require_sampling_overrides(self.entry.family, overrides))

        group_metadata = dict(metadata or {})
        if "fps" in sampling:
            group_metadata.setdefault("video_fps", sampling["fps"])

        resolved_inputs = [self._resolve_input(item) for item in inputs]
        default_task_type = task_type_for(self.entry.task)
        if default_task_type is not None and resolved_inputs:
            first = resolved_inputs[0]
            group_metadata["task_type"] = first.task_type
            if first.reference_image is not None:
                group_metadata["reference_image"] = first.reference_image
            if first.reference_video is not None:
                group_metadata["reference_video"] = first.reference_video

        request = GenerationRequest(
            request_id=f"{self.entry.family}-{uuid.uuid4()}",
            family=self.entry.family,
            task=self.entry.task,
            inputs=list(resolved_inputs),
            # GenerationRequest names this `samples_per_prompt` (generation-domain
            # wording); the value is the collector's `group_size` — the GRPO group,
            # sourced from rollout.n_samples_per_prompt. Same number, three domain
            # names (config / GRPO collector / generation request), each accurate to
            # its layer; distinct from any external evaluation sampling policy.
            samples_per_prompt=group_size,
            sampling=sampling,
            samples_per_generation_batch=self.config.samples_per_generation_batch,
            train_segments=self.config.train_segments,
            trajectory_storage=self.config.trajectory_storage,
            denoise=denoise,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        )
        return CollectorRequest(
            request=request,
            metadata=group_metadata,
        )

    def _resolve_input(
        self,
        item: GenerationInput | str,
    ) -> GenerationInput:
        """Apply family defaults to one conditioning input."""

        if isinstance(item, str):
            item = GenerationInput(prompt=item)
        return GenerationInput(
            prompt=item.prompt,
            task_type=item.task_type or task_type_for(self.entry.task),
            reference_image=item.reference_image,
            reference_video=item.reference_video,
        )


__all__ = [
    "CollectorRequest",
    "GenerationRequestBuilder",
]
