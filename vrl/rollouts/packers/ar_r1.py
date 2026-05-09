"""Janus-Pro-R1 multi-segment AR OutputBatch to RolloutBatch packing."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import torch

from vrl.engine import OutputBatch
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.packers.base import RolloutPackContext

R1_SEGMENT_NAMES = ("initial_image", "selfcheck_text", "final_image")


class ARR1RolloutPacker:
    """Pack Janus-Pro-R1-style segments without flattening image/text tokens."""

    primary_segment: str = "final_image"

    def reward_outputs(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> Any:
        images = output.extra.get("final_image", output.output)
        if context.rescale_to_unit:
            images = (images + 1.0) * 0.5
            images = images.clamp(0.0, 1.0)
        return images

    def reward_prompts(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> list[str]:
        del context
        return [spec.prompt for spec in output.sample_specs]

    async def pack(
        self,
        output: OutputBatch,
        rewards_raw: torch.Tensor,
        context: RolloutPackContext,
    ) -> RolloutBatch:
        raw_segments = output.extra.get("segments", output.extra.get("r1_segments"))
        if raw_segments is None:
            raise RuntimeError("ARR1RolloutPacker requires OutputBatch.extra['segments']")
        if not isinstance(raw_segments, dict):
            raise TypeError("R1 segments must be a dict")

        segments = {
            name: _normalize_segment(name, raw_segments[name])
            for name in R1_SEGMENT_NAMES
            if name in raw_segments
        }
        missing = [name for name in R1_SEGMENT_NAMES if name not in segments]
        if missing:
            raise RuntimeError(
                "ARR1RolloutPacker requires all R1 segments: " + ", ".join(missing),
            )

        primary = segments[self.primary_segment]
        token_ids = _segment_tensor(primary, "token_ids")
        token_log_probs = _segment_tensor(primary, "token_log_probs")
        token_mask = _segment_tensor(primary, "token_mask")
        prompt_ids = _segment_prompt_ids(primary, token_ids)
        prompt_mask = _segment_prompt_mask(primary, output.extra)

        device = context.device or prompt_ids.device
        final_image = output.extra.get("final_image", output.output)

        extras: dict[str, Any] = {
            # Compatibility path for OnlineTrainer debug/loop. Multi-segment
            # loss reads per-segment old log-probs from r1_segments instead.
            "log_probs": token_log_probs.detach().unsqueeze(1),
            "prompt_attention_mask": prompt_mask,
            "token_mask": token_mask,
            "r1_segments": segments,
            "primary_segment": self.primary_segment,
            "initial_image": output.extra.get("initial_image"),
            "final_image": final_image,
            "selfcheck": output.extra.get("selfcheck"),
        }
        for key in ("uncond_input_ids", "uncond_attention_mask"):
            if key in primary:
                extras[key] = primary[key]
            elif key in output.extra:
                extras[key] = output.extra[key]

        return RolloutBatch(
            observations=prompt_ids.unsqueeze(1),
            actions=token_ids,
            rewards=rewards_raw.to(device),
            dones=torch.ones(len(output.sample_specs), dtype=torch.bool, device=device),
            group_ids=torch.tensor(
                [spec.prompt_index for spec in output.sample_specs],
                dtype=torch.long,
                device=device,
            ),
            extras=extras,
            context={
                **dict(output.extra.get("context", {})),
                "r1_segment_names": tuple(segments.keys()),
            },
            videos=final_image.unsqueeze(2),
            prompts=[spec.prompt for spec in output.sample_specs],
        )


def _normalize_segment(name: str, segment: Any) -> dict[str, Any]:
    if isinstance(segment, dict):
        out = dict(segment)
    elif is_dataclass(segment):
        out = {
            field_name: getattr(segment, field_name)
            for field_name in getattr(segment, "__dataclass_fields__", {})
        }
    else:
        out = {
            field_name: getattr(segment, field_name)
            for field_name in dir(segment)
            if not field_name.startswith("_") and not callable(getattr(segment, field_name))
        }

    replay = out.pop("replay", None)
    if isinstance(replay, dict):
        for key, value in replay.items():
            out.setdefault(key, value)

    if out.get("token_log_probs") is None and out.get("old_log_probs") is not None:
        out["token_log_probs"] = out["old_log_probs"]
    out.setdefault("name", name)
    out.setdefault("modality", _infer_modality(name, out))
    out.setdefault("train", out.get("enabled", True))
    return out


def _segment_tensor(
    segment: dict[str, Any],
    key: str,
    fallback: dict[str, Any] | None = None,
) -> torch.Tensor:
    value = segment.get(key)
    if value is None and fallback is not None:
        value = fallback.get(key)
    if not isinstance(value, torch.Tensor):
        name = segment.get("name", "<unknown>")
        raise RuntimeError(f"R1 segment {name!r} is missing tensor field {key!r}")
    return value


def _segment_prompt_ids(segment: dict[str, Any], token_ids: torch.Tensor) -> torch.Tensor:
    value = segment.get("prompt_input_ids")
    if isinstance(value, torch.Tensor):
        return value
    replay = segment.get("replay")
    if isinstance(replay, dict) and isinstance(replay.get("prompt_input_ids"), torch.Tensor):
        return replay["prompt_input_ids"]
    # R1 replay uses prompt embeddings directly. Observations only provide the
    # trainer timestep axis, so a one-token placeholder is safer than forcing
    # every R1 executor to carry text ids for padded/generated contexts.
    return torch.zeros(
        token_ids.shape[0],
        1,
        dtype=torch.long,
        device=token_ids.device,
    )


def _segment_prompt_mask(
    segment: dict[str, Any],
    fallback: dict[str, Any],
) -> torch.Tensor:
    for key in ("prompt_attention_mask", "attention_mask"):
        value = segment.get(key)
        if isinstance(value, torch.Tensor):
            return value
        value = fallback.get(key)
        if isinstance(value, torch.Tensor):
            return value
    replay = segment.get("replay")
    if isinstance(replay, dict):
        for key in ("prompt_attention_mask", "attention_mask"):
            value = replay.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise RuntimeError("R1 primary segment is missing prompt attention mask")


def _infer_modality(name: str, segment: dict[str, Any]) -> str:
    visual = segment.get("visual")
    if visual is not None:
        return "image" if bool(visual) else "text"
    if name.endswith("_text"):
        return "text"
    return "image"


__all__ = ["R1_SEGMENT_NAMES", "ARR1RolloutPacker"]
