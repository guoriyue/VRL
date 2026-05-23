"""Cosmos Predict2 Anima text-to-image runtime.

Anima is distributed as single-file checkpoints. The diffusion backbone is
Cosmos Predict2 Text2Image, but Anima replaces the prompt path with Qwen3-0.6B
plus a learned LLM adapter before feeding Cosmos' 1024-wide text context.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.diffusion import DiffusionModelBase
from vrl.models.diffusion.common import (
    ChunkedLatentDecoder,
    LatentDecodeSpec,
    LatentDecodeTransform,
)
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult

_COSMOS_T2I_TRANSFORMER_CONFIG: dict[str, Any] = {
    "_class_name": "CosmosTransformer3DModel",
    "adaln_lora_dim": 256,
    "attention_head_dim": 128,
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
    "in_channels": 16,
    "max_size": [128, 240, 240],
    "mlp_ratio": 4.0,
    "num_attention_heads": 16,
    "num_layers": 28,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_scale": [1.0, 4.0, 4.0],
    "text_embed_dim": 1024,
}


@dataclass
class AnimaSamplingState:
    """Private Anima sampling state."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    padding_mask: torch.Tensor
    seed: int


class AnimaModel(DiffusionModelBase):
    """Single-file Anima model on the shared diffusion RL path."""

    family = "cosmos-predict2-anima-t2i"

    def __init__(
        self,
        *,
        transformer: Any,
        text_encoder: Any,
        llm_adapter: nn.Module,
        vae: Any,
        scheduler: Any,
        image_processor: Any,
        qwen_tokenizer: Any,
        t5_tokenizer: Any,
        device: Any,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.llm_adapter = llm_adapter
        self.vae = vae
        self._scheduler = scheduler
        self.image_processor = image_processor
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self._device = device
        self._dtype = dtype

    @classmethod
    def from_spec(cls, spec: Any) -> AnimaModel:
        """Load Anima's transformer, Qwen3 encoder, and VAE from local files."""

        from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        from safetensors.torch import load_file
        from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

        paths = spec.extra
        dtype = _resolve_torch_dtype(spec.dtype)

        transformer_checkpoint = load_file(paths["transformer_path"], device="cpu")
        transformer = _load_anima_transformer(transformer_checkpoint, dtype=dtype)
        llm_adapter = _load_anima_llm_adapter(transformer_checkpoint, dtype=dtype)
        del transformer_checkpoint

        text_encoder_state = load_file(paths["text_encoder_path"], device="cpu")
        text_encoder = Qwen3Model(_qwen3_06b_config())
        text_encoder.load_state_dict(
            {
                key.removeprefix("model."): value
                for key, value in text_encoder_state.items()
            },
            strict=True,
        )
        del text_encoder_state

        vae_state = convert_wan_vae_to_diffusers(
            load_file(paths["vae_path"], device="cpu"),
        )
        vae = AutoencoderKLQwenImage()
        vae.load_state_dict(vae_state, strict=True)
        del vae_state

        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=float(spec.extra.get("scheduler_shift", 3.0)),
        )
        scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)

        qwen_tokenizer = Qwen2Tokenizer.from_pretrained(
            paths["qwen_tokenizer_path"],
            local_files_only=True,
        )
        if qwen_tokenizer.pad_token is None:
            qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        t5_tokenizer = T5TokenizerFast.from_pretrained(
            paths["t5_tokenizer_path"],
            local_files_only=True,
        )

        transformer.requires_grad_(False)
        text_encoder.requires_grad_(False)
        llm_adapter.requires_grad_(False)
        vae.requires_grad_(False)

        transformer.to(spec.device, dtype=dtype)
        text_encoder.to(spec.device, dtype=dtype)
        llm_adapter.to(spec.device, dtype=dtype)
        vae.to(spec.device, dtype=torch.float32)

        return cls(
            transformer=transformer,
            text_encoder=text_encoder,
            llm_adapter=llm_adapter,
            vae=vae,
            scheduler=scheduler,
            image_processor=VaeImageProcessor(vae_scale_factor=8),
            qwen_tokenizer=qwen_tokenizer,
            t5_tokenizer=t5_tokenizer,
            device=spec.device,
            dtype=dtype,
        )

    def apply_lora(self, spec: Any) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model

        self.transformer.requires_grad_(False)
        self.transformer.to(self.device, dtype=self._dtype)

        if spec.lora_path:
            transformer = PeftModel.from_pretrained(
                self.transformer,
                spec.lora_path,
                is_trainable=True,
            )
            transformer.set_adapter("default")
            self._set_transformer(transformer)
            return

        assert spec.lora_config is not None
        cfg = LoraConfig(
            r=spec.lora_config["rank"],
            lora_alpha=spec.lora_config["alpha"],
            init_lora_weights="gaussian",
            target_modules=spec.lora_config["target_modules"],
        )
        self._set_transformer(get_peft_model(self.transformer, cfg))

    def enable_full_finetune(self) -> None:
        self.transformer.requires_grad_(True)
        self.transformer.to(self.device, dtype=self._dtype)

    def torch_compile_transformer(self, mode: str) -> None:
        self._set_transformer(
            torch.compile(self.transformer, mode=mode, fullgraph=False),
        )

    def set_num_steps(self, n: int) -> None:
        self.scheduler.set_timesteps(int(n), device=self.device)

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def backend_handle(self) -> Any:
        return self

    @property
    def device(self) -> Any:
        return self._device

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "device": str(self.device)}

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        guidance_scale = float(kwargs.get("guidance_scale", 5.0))
        do_cfg = guidance_scale > 1.0
        max_seq = int(kwargs.get("max_sequence_length", 512))
        prompt_embeds = self._encode_anima_prompt(prompt, max_sequence_length=max_seq)
        negative_prompt_embeds = None
        if do_cfg:
            negative = negative_prompt if negative_prompt is not None else ""
            negative_prompt_embeds = self._encode_anima_prompt(
                negative,
                max_sequence_length=max_seq,
            )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    def _encode_anima_prompt(
        self,
        prompt: str | list[str],
        *,
        max_sequence_length: int,
    ) -> torch.Tensor:
        prompts = _non_empty_prompts([prompt] if isinstance(prompt, str) else list(prompt))
        max_len = max(1, int(max_sequence_length))
        qwen_inputs = self.qwen_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        t5_inputs = self.t5_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            qwen_hidden = self.text_encoder(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs.get("attention_mask"),
                use_cache=False,
            ).last_hidden_state.to(self._dtype)
            embeds = self.llm_adapter(qwen_hidden, t5_inputs["input_ids"])
            if embeds.shape[1] < 512:
                embeds = F.pad(embeds, (0, 0, 0, 512 - embeds.shape[1]))
        return embeds.detach()

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> AnimaSamplingState:
        del kwargs
        device = self.device
        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        batch_size = prompt_embeds.shape[0]
        guidance_scale = float(request.guidance_scale)
        do_cfg = guidance_scale > 1.0

        self.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = self.scheduler.timesteps

        seed = (
            request.seed
            if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        scale = int(getattr(self.vae, "spatial_compression_ratio", 8))
        latent_shape = (
            batch_size,
            int(self.transformer.config.in_channels),
            1,
            request.height // scale,
            request.width // scale,
        )
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        latents = latents * float(self.scheduler.config.sigma_max)
        padding_mask = latents.new_zeros(
            1,
            1,
            request.height,
            request.width,
            dtype=self._dtype,
        )
        return AnimaSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            padding_mask=padding_mask,
            seed=seed,
        )

    def forward_step(
        self,
        state: AnimaSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        current_sigma = state.scheduler.sigmas[step_idx]
        current_t = current_sigma / (current_sigma + 1)
        c_in = 1 - current_t
        c_skip = 1 - current_t
        c_out = -current_t
        timestep = current_t.expand(state.latents.shape[0]).to(self._dtype)
        latent_model_input = (state.latents * c_in).to(self._dtype)

        cond = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=state.prompt_embeds,
            padding_mask=state.padding_mask,
            return_dict=False,
        )[0]
        noise_pred_cond = (c_skip * state.latents + c_out * cond.float()).to(self._dtype)
        noise_pred_uncond = torch.zeros_like(noise_pred_cond)
        combined = noise_pred_cond

        if state.do_cfg:
            if state.negative_prompt_embeds is None:
                raise ValueError("Anima CFG requires negative_prompt_embeds")
            uncond = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=state.negative_prompt_embeds,
                padding_mask=state.padding_mask,
                return_dict=False,
            )[0]
            noise_pred_uncond = (
                c_skip * state.latents + c_out * uncond.float()
            ).to(self._dtype)
            combined = noise_pred_cond + state.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        noise_pred = ((state.latents - combined.float()) / current_sigma).to(self._dtype)
        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": (
                (state.latents - noise_pred_cond.float()) / current_sigma
            ).to(self._dtype),
            "noise_pred_uncond": (
                (state.latents - noise_pred_uncond.float()) / current_sigma
            ).to(self._dtype),
        }

    def export_batch_context(self, state: AnimaSamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_replay_tensors(self, state: AnimaSamplingState) -> dict[str, Any]:
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "padding_mask": _align_replay_tensor(
                state.padding_mask,
                state.latents.shape[0],
            ),
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> AnimaSamplingState:
        del step_idx
        return AnimaSamplingState(
            latents=latents,
            timesteps=self.scheduler.timesteps,
            scheduler=self.scheduler,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            padding_mask=_shared_replay_tensor(replay_tensors, "padding_mask"),
            seed=0,
        )

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del request
        replay_tensors, batch_context, latents = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        state = self.restore_eval_state(
            replay_tensors,
            batch_context,
            latents,
            timestep_idx,
        )
        values = self.forward_step(state, timestep_idx)
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values=dict(values),
                ),
            },
        )

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            x = chunk.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            latents_std = (
                1.0 / torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            return x / latents_std + latents_mean

        decoder = ChunkedLatentDecoder(
            LatentDecodeSpec(
                transform=LatentDecodeTransform(_transform),
                vae_decode=lambda chunk: self.vae.decode(
                    chunk,
                    return_dict=False,
                )[0][:, :, 0],
                postprocess=lambda image: self.image_processor.postprocess(
                    image,
                    output_type="pt",
                ),
                output_layout="image_bchw",
                decode_batch_size=None,
            ),
        )
        return decoder(latents)


def _non_empty_prompts(prompts: list[str]) -> list[str]:
    # Anima uses add_special_tokens=False; empty strings produce zero-length
    # Qwen/T5 token tensors and crash Qwen attention during CFG negative prompts.
    return [prompt if str(prompt).strip() else "." for prompt in prompts]


class AnimaReplayModel(AnimaModel):
    """Replay-only Anima model without text encoder, LLM adapter, or VAE."""

    def __init__(
        self,
        *,
        transformer: Any,
        scheduler: Any,
        device: Any,
        dtype: torch.dtype,
    ) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device
        self._dtype = dtype

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("AnimaReplayModel does not own a full generation pipeline")

    @property
    def backend_handle(self) -> Any:
        return None

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt, negative_prompt, kwargs
        raise RuntimeError("AnimaReplayModel cannot encode prompts")

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> AnimaSamplingState:
        del request, encoded, kwargs
        raise RuntimeError("AnimaReplayModel cannot run rollout sampling")

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        del latents
        raise RuntimeError("AnimaReplayModel cannot decode latents")


def _load_anima_transformer(
    checkpoint: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> Any:
    from diffusers import CosmosTransformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_cosmos_transformer_checkpoint_to_diffusers,
    )

    converted = convert_cosmos_transformer_checkpoint_to_diffusers(dict(checkpoint))
    transformer = CosmosTransformer3DModel.from_config(_COSMOS_T2I_TRANSFORMER_CONFIG)
    transformer.to(dtype=dtype)
    expected = set(transformer.state_dict())
    transformer_state = {
        key: value
        for key, value in converted.items()
        if key in expected
    }
    missing, unexpected = transformer.load_state_dict(transformer_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "failed to load Anima Cosmos transformer weights: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}",
        )
    return transformer


def _load_anima_llm_adapter(
    checkpoint: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> nn.Module:
    adapter = AnimaLLMAdapter()
    adapter.to(dtype=dtype)
    state = {
        key.removeprefix("net.llm_adapter."): value
        for key, value in checkpoint.items()
        if key.startswith("net.llm_adapter.")
    }
    missing, unexpected = adapter.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(
            "failed to load Anima LLM adapter weights: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}",
        )
    return adapter


def _qwen3_06b_config() -> Any:
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=151643,
        eos_token_id=151645,
    )


class AnimaLLMAdapter(nn.Module):
    """Qwen3-to-Cosmos text adapter shipped inside Anima transformer files."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32128, 1024)
        self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(64)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    source_dim=1024,
                    model_dim=1024,
                    num_heads=16,
                    use_self_attn=True,
                )
                for _ in range(6)
            ],
        )
        self.out_proj = nn.Linear(1024, 1024)
        self.norm = nn.RMSNorm(1024, eps=1e-6)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        model_dim: int,
        num_heads: int,
        use_self_attn: bool,
    ) -> None:
        super().__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
            self.self_attn = Attention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = Attention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_context: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            x = x + self.self_attn(
                normed,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
        normed = self.norm_cross_attn(x)
        x = x + self.cross_attn(
            normed,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        return x + self.mlp(self.norm_mlp(x))


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_context: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        cos, sin = position_embeddings_context
        key_states = apply_rotary_pos_emb(key_states, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def _align_replay_tensor(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] == 1:
        return value.expand(batch_size, *value.shape[1:])
    return value


def _shared_replay_tensor(
    replay_tensors: dict[str, Any],
    key: str,
) -> torch.Tensor:
    value = replay_tensors[key]
    if isinstance(value, torch.Tensor) and value.shape[0] > 1:
        return value[:1]
    return value


def _resolve_torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    text = str(value).replace("torch.", "")
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(text, torch.bfloat16)


__all__ = [
    "AnimaLLMAdapter",
    "AnimaModel",
    "AnimaReplayModel",
    "AnimaSamplingState",
]
