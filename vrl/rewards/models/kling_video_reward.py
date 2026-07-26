"""Repo-owned Kling VideoReward model loading and inference.

Portions of this module are adapted from KwaiVGI/VideoAlign's inference,
Qwen2-VL reward model, and checkpoint-loading code under the MIT license. VRL
owns this narrowed implementation so production reward scoring does not depend
on a VideoAlign checkout or flat-module imports.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration

from vrl.rewards.assets.kling_prompt_templates import (
    build_kling_video_reward_prompt,
)
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.base import require_prompt_and_video_path
from vrl.rewards.models.hub import parse_hf_repo_revision
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_SPECIAL_TOKENS = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
_DEFAULT_REWARD_MODEL = "KlingTeam/VideoReward"
_SCORE_KEY_MAP = {
    "overall_reward": "Overall",
    "visual_quality": "VQ",
    "motion_quality": "MQ",
    "text_alignment": "TA",
}


@dataclass
class _DataConfig:
    max_frame_pixels: int = 240 * 320
    num_frames: int | None = None
    fps: float = 2.0
    eval_dim: str | list[str] = "VQ"
    prompt_template_type: str = "none"
    sample_type: str = "uniform"


@dataclass
class _ModelConfig:
    model_name_or_path: str = ""
    model_revision: str = "main"
    output_dim: int = 1
    use_special_tokens: bool = False
    torch_dtype: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"


@dataclass
class _PeftLoraConfig:
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_namespan_exclude: list[str] | str | None = None
    lora_modules_to_save: list[str] | None = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1


class KlingVideoRewardModel:
    """One RewardModel: load KlingTeam/VideoReward weights and score a video artifact."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        self.reward_model_name = str(
            self.worker_config.get("reward_model_name", ""),
        ).strip()
        self.local_files_only = bool(self.worker_config.get("local_files_only", False))
        logger.info(
            "resolving Kling VideoReward root %s",
            kv(
                reward_model_name=self.reward_model_name,
                model_path=str(self.worker_config.get("model_path", "")).strip(),
                local_files_only=self.local_files_only,
            ),
        )
        self.model_root = _resolve_model_root(self.worker_config)
        self.dtype = _torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.use_norm = bool(self.worker_config.get("use_norm", True))
        # Optional per-frame pixel bounds for qwen smart_resize. min upscales
        # low-res rollout videos toward the reward model's training frame
        # budget (checkpoint data_config.max_frame_pixels, 200704 for
        # KlingTeam/VideoReward) so scores are calibrated in-distribution;
        # max overrides the checkpoint's own budget. None keeps checkpoint
        # behavior.
        min_frame_pixels = self.worker_config.get("min_frame_pixels")
        self.min_frame_pixels = None if min_frame_pixels is None else int(min_frame_pixels)
        max_frame_pixels = self.worker_config.get("max_frame_pixels")
        self.max_frame_pixels = None if max_frame_pixels is None else int(max_frame_pixels)
        logger.info(
            "resolved Kling VideoReward root %s",
            kv(
                root=self.model_root,
                device=self.device,
                dtype=self.dtype,
                use_norm=self.use_norm,
                min_frame_pixels=self.min_frame_pixels,
                max_frame_pixels=self.max_frame_pixels,
            ),
        )
        disable_flash_attn2 = self.worker_config.get("disable_flash_attn2", None)
        if disable_flash_attn2 is None:
            disable_flash_attn2 = importlib.util.find_spec("flash_attn") is None
        else:
            disable_flash_attn2 = bool(disable_flash_attn2)
        if disable_flash_attn2:
            logger.info("Forcing Kling VideoReward to use SDPA attention")

        logger.info("loading Kling VideoReward configs %s", kv(root=self.model_root))
        data_config, model_config, peft_config, inference_config = _load_configs(self.model_root)
        logger.info(
            "building Kling VideoReward base model %s",
            kv(
                base_model=model_config.model_name_or_path,
                revision=model_config.model_revision,
                local_files_only=self.local_files_only,
                disable_flash_attn2=disable_flash_attn2,
            ),
        )
        model, processor = _create_model_and_processor(
            model_config,
            peft_config,
            dtype=self.dtype,
            disable_flash_attn2=disable_flash_attn2,
            local_files_only=self.local_files_only,
        )
        logger.info("loading Kling VideoReward checkpoint %s", kv(root=self.model_root))
        model, _checkpoint_step = load_kling_video_reward_checkpoint(
            model,
            self.model_root,
        )
        logger.info(
            "moving Kling VideoReward model to device %s",
            kv(device=self.device, checkpoint_step=_checkpoint_step),
        )
        model.eval()
        self.model = model.to(self.device)
        self.processor = processor
        self.data_config = data_config
        self.inference_config = inference_config
        logger.info("loaded Kling VideoReward model %s", kv(device=self.device))

    def move_to(self, device: str) -> None:
        """Move the complete reward backbone for shared-GPU phase handoff."""

        self.model = self.model.to(device)
        self.device = str(device)

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        prompt, artifact_path = require_prompt_and_video_path(
            artifact,
            family="Kling VideoReward",
        )
        rewards = self._reward(
            [artifact_path],
            [prompt],
            min_pixels=self.min_frame_pixels,
            max_pixels=self.max_frame_pixels,
            use_norm=self.use_norm,
        )
        if not rewards:
            raise RuntimeError("Kling VideoReward returned no scores")
        raw_scores = rewards[0]
        if not isinstance(raw_scores, Mapping):
            raise TypeError("Kling VideoReward must return a mapping of scores")
        return _normalize_scores(raw_scores)

    def _prepare_batch(
        self,
        video_paths: list[str],
        prompts: list[str],
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> Mapping[str, Any]:
        from qwen_vl_utils import process_vision_info

        if self.data_config.sample_type != "uniform":
            raise ValueError(
                "Kling VideoReward repo-owned inference currently supports only "
                f"uniform video sampling, got {self.data_config.sample_type!r}",
            )
        fps = self.data_config.fps
        num_frames = self.data_config.num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels
        chat_data = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_path}",
                            "max_pixels": max_pixels,
                            **({"min_pixels": min_pixels} if min_pixels is not None else {}),
                            **(
                                {"nframes": num_frames} if num_frames is not None else {"fps": fps}
                            ),
                        },
                        {
                            "type": "text",
                            "text": build_kling_video_reward_prompt(
                                prompt,
                                self.data_config.eval_dim,
                                self.data_config.prompt_template_type,
                            ),
                        },
                    ],
                },
            ]
            for video_path, prompt in zip(video_paths, prompts, strict=True)
        ]
        image_inputs, video_inputs = process_vision_info(chat_data)
        batch = self.processor(
            text=self.processor.apply_chat_template(
                chat_data,
                tokenize=False,
                add_generation_prompt=True,
            ),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        if hasattr(batch, "to"):
            return batch.to(self.device)
        return {
            key: value.to(device=self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _reward(
        self,
        video_paths: list[str],
        prompts: list[str],
        max_pixels: int | None = None,
        min_pixels: int | None = None,
        use_norm: bool = True,
    ) -> list[dict[str, float]]:
        batch = self._prepare_batch(video_paths, prompts, max_pixels, min_pixels)
        with torch.no_grad():
            logits = self.model(return_dict=True, **batch)["logits"]
        rewards = [
            {
                "VQ": float(reward[0].item()),
                "MQ": float(reward[1].item()),
                "TA": float(reward[2].item()),
            }
            for reward in logits
        ]
        for reward in rewards:
            if use_norm and self.inference_config is not None:
                reward["VQ"] = (reward["VQ"] - float(self.inference_config["VQ_mean"])) / float(
                    self.inference_config["VQ_std"]
                )
                reward["MQ"] = (reward["MQ"] - float(self.inference_config["MQ_mean"])) / float(
                    self.inference_config["MQ_std"]
                )
                reward["TA"] = (reward["TA"] - float(self.inference_config["TA_mean"])) / float(
                    self.inference_config["TA_std"]
                )
            reward["Overall"] = reward["VQ"] + reward["MQ"] + reward["TA"]
        return rewards


class KlingQwen2VLRewardModel(Qwen2VLForConditionalGeneration):
    """Qwen2-VL backbone with a reward head over selected tokens."""

    def __init__(
        self,
        config: Any,
        output_dim: int = 4,
        reward_token: str = "last",
        special_token_ids: list[int] | None = None,
        # Transformers 5 forwards loading kwargs (use_cache, ...) into the model
        # constructor, so a fixed signature raises TypeError before any weight is
        # read. They describe generation behavior this reward head never uses --
        # it runs one forward and reads a head over selected tokens -- so absorb
        # them rather than widening the parent call.
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(config)
        self.output_dim = output_dim
        # Transformers 5 moved the language dimensions off the top-level
        # multimodal config into a nested text config; get_text_config() is the
        # accessor that survives both layouts.
        self.rm_head = nn.Linear(config.get_text_config().hidden_size, output_dim, bias=False)
        self.reward_token = "special" if special_token_ids is not None else reward_token
        self.special_token_ids = special_token_ids

    def forward(
        self,
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
            output_attentions if output_attentions is not None else self.config.output_attentions
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
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack(
                [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)],
            )
        elif self.reward_token == "special":
            if input_ids is None or self.special_token_ids is None:
                raise ValueError("special reward token pooling requires input_ids")
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


def preflight_kling_video_reward_backend() -> None:
    """Validate repo-owned backend imports without downloading weights."""

    import peft  # noqa: F401
    import qwen_vl_utils  # noqa: F401
    import transformers  # noqa: F401


def load_kling_video_reward_checkpoint(
    model: Any,
    checkpoint_dir: Path,
) -> tuple[Any, str]:
    checkpoint_path, resolved_step = _resolve_checkpoint_path(checkpoint_dir)
    full_ckpt = checkpoint_path / "model.pth"
    if full_ckpt.exists():
        state = torch.load(full_ckpt, map_location="cpu")
        if isinstance(state, Mapping):
            state = _remap_qwen2vl_state_dict(state, model.state_dict())
        model.load_state_dict(state, strict=True)
        return model, resolved_step

    import safetensors.torch

    lora_state = safetensors.torch.load_file(checkpoint_path / "adapter_model.safetensors")
    non_lora_state = torch.load(
        checkpoint_path / "non_lora_state_dict.pth",
        map_location="cpu",
        weights_only=True,
    )
    lora_state = _insert_adapter_name_into_state_dict(
        lora_state,
        adapter_name="default",
        parameter_prefix="lora_",
    )
    model_state = model.state_dict()
    model_state.update(non_lora_state)
    model_state.update(lora_state)
    model_state = _remap_qwen2vl_state_dict(model_state, model.state_dict())
    model.load_state_dict(model_state, strict=True)
    return model, resolved_step


def _resolve_checkpoint_path(checkpoint_dir: Path) -> tuple[Path, str]:
    checkpoint_paths = list(checkpoint_dir.glob("checkpoint-*"))
    checkpoint_paths.sort(
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No Kling VideoReward checkpoints found in {checkpoint_dir}")
    checkpoint_path = checkpoint_paths[0]
    return checkpoint_path, checkpoint_path.name.split("checkpoint-")[-1]


def _remap_qwen2vl_key(key: str) -> str:
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


def _resolve_model_root(worker_config: Mapping[str, Any]) -> Path:
    model_path = str(worker_config.get("model_path", "")).strip()
    if model_path:
        root = Path(model_path).expanduser().resolve()
    else:
        reward_model_name = str(
            worker_config.get("reward_model_name", ""),
        ).strip()
        if not reward_model_name:
            raise ValueError("Kling VideoReward requires worker_config.reward_model_name")
        model_ref = parse_hf_repo_revision(reward_model_name)
        if model_ref.repo_id != _DEFAULT_REWARD_MODEL:
            raise ValueError(
                "Kling VideoReward loader currently supports only "
                f"{_DEFAULT_REWARD_MODEL!r}, got {model_ref.repo_id!r}",
            )
        from huggingface_hub import snapshot_download

        local_files_only = bool(worker_config.get("local_files_only", False))
        try:
            root = Path(
                snapshot_download(
                    repo_id=model_ref.repo_id,
                    revision=model_ref.revision,
                    local_files_only=local_files_only,
                ),
            ).resolve()
        except Exception as exc:
            mode = "local cache" if local_files_only else "Hugging Face download"
            raise RuntimeError(
                f"Failed to resolve Kling VideoReward model root from {mode}: "
                f"repo_id={model_ref.repo_id!r} revision={model_ref.revision!r}. Set "
                "reward.kwargs.kling_video_reward.worker_config.model_path to a "
                "local snapshot path, or pre-download the model cache before training.",
            ) from exc

    if not (root / "model_config.json").exists():
        parent = root.parent
        if root.name.startswith("checkpoint-") and (parent / "model_config.json").exists():
            root = parent
    if not root.exists():
        raise FileNotFoundError(f"Kling VideoReward model path does not exist: {root}")
    if not (root / "model_config.json").exists():
        raise FileNotFoundError(
            f"Kling VideoReward model root must contain model_config.json: {root}",
        )
    if not any(root.glob("checkpoint-*")):
        raise FileNotFoundError(
            f"Kling VideoReward model root must contain a checkpoint-* subdir: {root}",
        )
    return root


def _normalize_scores(raw_scores: Mapping[str, Any]) -> dict[str, float]:
    """Map the model's raw VQ/MQ/TA/Overall to the public score keys only.

    The returned dict is the public scoring contract — it carries solely the
    aliases in ``_SCORE_KEY_MAP`` (``overall_reward``/``visual_quality``/...),
    never the raw model keys, so a reader cannot accidentally select an
    undocumented key as ``score_key``.
    """

    raw = {str(key): float(value) for key, value in raw_scores.items()}
    return {
        public_key: raw[model_key]
        for public_key, model_key in _SCORE_KEY_MAP.items()
        if model_key in raw
    }


def _create_model_and_processor(
    model_config: _ModelConfig,
    peft_config: _PeftLoraConfig,
    *,
    dtype: torch.dtype,
    disable_flash_attn2: bool,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    if model_config.load_in_8bit or model_config.load_in_4bit:
        raise ValueError("Kling VideoReward inference does not support 8-bit/4-bit loading")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    torch_dtype = _torch_dtype(model_config.torch_dtype, fallback=dtype)
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        local_files_only=local_files_only,
        padding_side="right",
    )
    special_token_ids = None
    if model_config.use_special_tokens:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": _SPECIAL_TOKENS})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(_SPECIAL_TOKENS)

    model = KlingQwen2VLRewardModel.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa" if disable_flash_attn2 else "flash_attention_2",
        revision=model_config.model_revision,
        local_files_only=local_files_only,
        use_cache=True,
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))
    if dtype in {torch.bfloat16, torch.float16}:
        model.to(dtype)

    if peft_config.lora_enable:
        excluded = peft_config.lora_namespan_exclude
        if excluded is None:
            lora_namespan_exclude = []
        elif isinstance(excluded, str):
            lora_namespan_exclude = [excluded]
        else:
            lora_namespan_exclude = list(excluded)
        target_modules = _find_target_linear_names(
            model,
            num_lora_modules=peft_config.num_lora_modules,
            lora_namespan_exclude=lora_namespan_exclude,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                target_modules=target_modules,
                r=peft_config.lora_r,
                lora_alpha=peft_config.lora_alpha,
                lora_dropout=peft_config.lora_dropout,
                task_type=peft_config.lora_task_type,
                use_rslora=peft_config.use_rslora,
                bias="none",
                modules_to_save=peft_config.lora_modules_to_save,
            ),
        )

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor


def _load_configs(
    root: Path,
) -> tuple[_DataConfig, _ModelConfig, _PeftLoraConfig, dict[str, float] | None]:
    with (root / "model_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return (
        _from_dataclass(_DataConfig, config.get("data_config", {})),
        _from_dataclass(_ModelConfig, config.get("model_config", {})),
        _from_dataclass(_PeftLoraConfig, config.get("peft_lora_config", {})),
        config.get("inference_config"),
    )


def _from_dataclass(cls: Any, values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(values).items() if key in allowed})


def _find_target_linear_names(
    model: Any,
    *,
    num_lora_modules: int = -1,
    lora_namespan_exclude: list[str] | None = None,
) -> list[str]:
    excluded = tuple(lora_namespan_exclude or ())
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
        and not any(token in name for token in excluded)
    ]
    return names[-num_lora_modules:] if num_lora_modules > 0 else names


def _torch_dtype(name: str | None, *, fallback: torch.dtype | None = None) -> torch.dtype | str:
    if name is None:
        if fallback is None:
            raise ValueError("Kling VideoReward torch dtype is required")
        return fallback
    if name.lower() == "auto":
        return "auto"
    from vrl.models.dtypes import resolve_torch_dtype

    return resolve_torch_dtype(name)


def _remap_qwen2vl_state_dict(
    state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    remapped = {_remap_qwen2vl_key(str(key)): value for key, value in state.items()}
    return remapped if set(remapped) == set(target_state) else state


def _insert_adapter_name_into_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    adapter_name: str,
    parameter_prefix: str,
) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if parameter_prefix in key:
            prefix, separator, leaf = key.rpartition(".")
            new_key = f"{prefix}.{adapter_name}.{leaf}" if separator else f"{key}.{adapter_name}"
        remapped[new_key] = value
    return remapped


__all__ = [
    "KlingVideoRewardModel",
    "preflight_kling_video_reward_backend",
]
