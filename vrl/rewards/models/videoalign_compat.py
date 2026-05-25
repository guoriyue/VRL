"""Compatibility shims for VideoAlign reward inference."""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def prepare_videoalign_runtime(
    inference_cls: Any,
    *,
    disable_flash_attn2: bool | None = None,
) -> None:
    """Patch VideoAlign only for dependency/API gaps required by current runtime."""

    if disable_flash_attn2 is None:
        disable_flash_attn2 = importlib.util.find_spec("flash_attn") is None
    if disable_flash_attn2:
        logger.info("Forcing VideoAlign to use SDPA attention for VideoReward")
        patch_videoalign_disable_flash_attn2(inference_cls)
    patch_videoalign_qwen2vl_forward_compat()
    patch_videoalign_checkpoint_key_compat(inference_cls)


def patch_videoalign_disable_flash_attn2(inference_cls: Any) -> None:
    """Force VideoAlign to use SDPA when flash-attn is unavailable."""

    module = sys.modules.get(str(getattr(inference_cls, "__module__", "")))
    if module is None or bool(getattr(module, "_vrl_disable_flash_attn2_patched", False)):
        return
    create_model_and_processor = getattr(module, "create_model_and_processor", None)
    if not callable(create_model_and_processor):
        return

    def _create_model_and_processor_without_flash(*args: Any, **kwargs: Any) -> Any:
        training_args = kwargs.get("training_args")
        if training_args is None and len(args) >= 3:
            training_args = args[2]
        if training_args is not None:
            training_args.disable_flash_attn2 = True
        return create_model_and_processor(*args, **kwargs)

    module.create_model_and_processor = _create_model_and_processor_without_flash
    module._vrl_disable_flash_attn2_patched = True


def patch_videoalign_checkpoint_key_compat(inference_cls: Any) -> None:
    """Patch VideoAlign checkpoint loading for current Transformers Qwen2-VL keys."""

    module = sys.modules.get(str(getattr(inference_cls, "__module__", "")))
    if module is None or bool(getattr(module, "_vrl_checkpoint_key_compat_patched", False)):
        return
    load_model_from_checkpoint = getattr(module, "load_model_from_checkpoint", None)
    if not callable(load_model_from_checkpoint):
        return

    def _load_model_from_checkpoint_compat(
        model: Any,
        checkpoint_dir: str,
        checkpoint_step: int | None,
    ) -> tuple[Any, str]:
        checkpoint_path, resolved_step = resolve_videoalign_checkpoint_path(
            checkpoint_dir,
            checkpoint_step,
        )
        full_ckpt = checkpoint_path / "model.pth"
        if full_ckpt.exists():
            state = torch.load(full_ckpt, map_location="cpu")
            if isinstance(state, Mapping):
                remapped = {
                    remap_videoalign_qwen2vl_key(str(key)): value
                    for key, value in state.items()
                }
                model_keys = set(model.state_dict().keys())
                if set(remapped) == model_keys:
                    logger.info(
                        "Loading VideoAlign checkpoint with Qwen2-VL key remap: %s",
                        checkpoint_path,
                    )
                    model.load_state_dict(remapped, strict=True)
                    return model, resolved_step
                if set(state) == model_keys:
                    logger.info("Loading VideoAlign checkpoint: %s", checkpoint_path)
                    model.load_state_dict(state, strict=True)
                    return model, resolved_step
        return load_model_from_checkpoint(model, checkpoint_dir, checkpoint_step)

    module.load_model_from_checkpoint = _load_model_from_checkpoint_compat
    module._vrl_checkpoint_key_compat_patched = True


def patch_videoalign_qwen2vl_forward_compat() -> None:
    """Patch VideoAlign's old Qwen2-VL forward path for current Transformers."""

    module = sys.modules.get("trainer")
    if module is None or bool(getattr(module, "_vrl_qwen2vl_forward_compat_patched", False)):
        return
    reward_cls = getattr(module, "Qwen2VLRewardModelBT", None)
    if reward_cls is None:
        return

    def _forward_compat(
        self: Any,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del labels, return_dict
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        model_kwargs = dict(kwargs)
        if cache_position is not None:
            model_kwargs["cache_position"] = cache_position
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            **model_kwargs,
        )

        hidden_states = outputs[0]
        logits = self.rm_head(hidden_states)
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None or input_ids is None:
            sequence_lengths = -1
        else:
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack(
                [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)],
            )
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 3, -1)
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError("Invalid reward_token")
        return {"logits": pooled_logits}

    reward_cls.forward = _forward_compat
    module._vrl_qwen2vl_forward_compat_patched = True


def resolve_videoalign_checkpoint_path(
    checkpoint_dir: str,
    checkpoint_step: int | None,
) -> tuple[Path, str]:
    root = Path(checkpoint_dir)
    checkpoint_paths = list(root.glob("checkpoint-*"))
    checkpoint_paths.sort(
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No VideoAlign checkpoints found in {root}")
    if checkpoint_step is None or int(checkpoint_step) == -1:
        checkpoint_path = checkpoint_paths[0]
    else:
        requested = root / f"checkpoint-{int(checkpoint_step)}"
        checkpoint_path = requested if requested in checkpoint_paths else checkpoint_paths[0]
    return checkpoint_path, checkpoint_path.name.split("checkpoint-")[-1]


def remap_videoalign_qwen2vl_key(key: str) -> str:
    if key.startswith("base_model.model.visual."):
        return key.replace(
            "base_model.model.visual.",
            "base_model.model.model.visual.",
            1,
        )
    for prefix in (
        "base_model.model.model.embed_tokens.",
        "base_model.model.model.layers.",
        "base_model.model.model.norm.",
    ):
        if key.startswith(prefix):
            return key.replace(
                "base_model.model.model.",
                "base_model.model.model.language_model.",
                1,
            )
    return key


__all__ = [
    "patch_videoalign_checkpoint_key_compat",
    "patch_videoalign_disable_flash_attn2",
    "patch_videoalign_qwen2vl_forward_compat",
    "prepare_videoalign_runtime",
    "remap_videoalign_qwen2vl_key",
    "resolve_videoalign_checkpoint_path",
]
