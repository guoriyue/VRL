"""Rollout-to-generation request adapter for collectors."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, NamedTuple

from vrl.generation import GenerationInput, GenerationRequest


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
        family: str,
        task: str,
        request_prefix: str,
        config: Any,
        return_artifacts: tuple[str, ...],
        default_task_type: str | None = None,
        metadata_key: str | None = None,
    ) -> None:
        if not return_artifacts:
            raise ValueError(f"{family} request builder requires return_artifacts")
        self.family = family
        self.task = task
        self.request_prefix = request_prefix
        self.config = config
        self.return_artifacts = return_artifacts
        self.default_task_type = default_task_type
        self.metadata_key = metadata_key

    def build(
        self,
        inputs: list[GenerationInput | str],
        group_size: int,
        *,
        metadata: Mapping[str, Any] | None = None,
        request_overrides: Mapping[str, Any] | None = None,
        seed: int | None = None,
        runtime_debug: bool = False,
        policy_version: int | None = None,
    ) -> CollectorRequest:
        sampling = self._sampling()
        if seed is not None:
            sampling["seed"] = seed
        sampling.update(dict(request_overrides or {}))

        group_metadata = dict(metadata or {})
        if "fps" in sampling:
            group_metadata.setdefault("video_fps", sampling["fps"])

        resolved_inputs = [self._resolve_input(item, group_metadata) for item in inputs]
        if self.default_task_type is not None and resolved_inputs:
            first = resolved_inputs[0]
            group_metadata["task_type"] = first.task_type
            if first.reference_image is not None:
                group_metadata["reference_image"] = first.reference_image
            if first.reference_video is not None:
                group_metadata["reference_video"] = first.reference_video

        request_metadata: dict[str, Any] = {}
        if runtime_debug:
            request_metadata["_runtime_debug"] = True

        request = GenerationRequest(
            request_id=f"{self.request_prefix}-{uuid.uuid4()}",
            family=self.family,
            task=self.task,
            inputs=list(resolved_inputs),
            # GenerationRequest names this `samples_per_prompt` (generation-domain
            # wording); the value is the collector's `group_size` — the GRPO group,
            # sourced from rollout.n_samples_per_prompt. Same number, three domain
            # names (config / GRPO collector / generation request), each accurate to
            # its layer; distinct from any external evaluation sampling policy.
            samples_per_prompt=group_size,
            sampling=sampling,
            return_artifacts=set(self.return_artifacts),
            metadata=request_metadata,
            policy_version=policy_version,
        )
        return CollectorRequest(
            request=request,
            metadata=group_metadata,
        )

    def _resolve_input(
        self,
        item: GenerationInput | str,
        group_metadata: Mapping[str, Any],
    ) -> GenerationInput:
        """Apply family defaults to one conditioning input."""

        if isinstance(item, str):
            item = GenerationInput(prompt=item)
        input_metadata = dict(item.metadata)
        if "video_fps" in group_metadata:
            input_metadata.setdefault("video_fps", group_metadata["video_fps"])
        # Some engines expect their per-sample metadata under a family-specific
        # key (e.g. nextstep_1's "rollout_metadata").
        if self.metadata_key is not None:
            input_metadata = {self.metadata_key: input_metadata}
        return GenerationInput(
            prompt=item.prompt,
            task_type=item.task_type or self.default_task_type,
            reference_image=item.reference_image,
            reference_video=item.reference_video,
            metadata=input_metadata,
        )

    def _sampling(self) -> dict[str, Any]:
        request_sampling = getattr(self.config, "request_sampling", None)
        if callable(request_sampling):
            raw_sampling = request_sampling()
        else:
            raw_sampling = getattr(self.config, "values", None)
        if not isinstance(raw_sampling, Mapping):
            raise TypeError(
                f"{type(self.config).__name__} must expose "
                "request_sampling() or values",
            )
        return {
            str(field_name): list(value) if isinstance(value, tuple) else value
            for field_name, value in raw_sampling.items()
        }


__all__ = [
    "CollectorRequest",
    "GenerationRequestBuilder",
]
