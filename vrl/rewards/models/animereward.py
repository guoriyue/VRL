"""AnimeReward visual-quality head as a model-backed RewardModel.

The quality dimension of AnimeReward (arXiv 2504.10044, released inside
``IndexTeam/Index-anisora``): an Idefics2-8B sequence-classification regression
head trained on anime video frames to score clearness, resolution, brightness,
and color. Returns ``{"animereward_quality": x}``; drive it with
``score_key="animereward_quality"``.

Two deliberate deviations from the paper's video setting, both forced by scoring
STILLS from an image model:

- The head consumes ``MAX_NUM_FRAMES`` sampled frames. A still is replicated
  into a static clip, which is the closest in-distribution encoding of "this
  image, held". Measured on anima output this still separates cleanly:
  on-prompt anime 0.696 +/- 0.027 vs off-prompt collapse 0.470 +/- 0.081
  (Cohen's d = 3.7), where LAION-aesthetic and PickScore both scored the two
  sets identically.
- The five video-only dimensions (smoothness, motion, image-video consistency,
  and cross-frame character identity) are undefined for a single still and are
  not loaded.

The head is Mantis's Idefics2 fork, which targets transformers 4.x while this
repo runs 5.x. That gap is bridged here rather than by pinning: see
``_adapt_mantis_to_current_transformers``. The model therefore runs either
colocated in the trainer process or behind the standalone reward service
(``vrl/config/reward_service/animereward_quality.yaml``).

Prefer colocated on a single-GPU box. The service holds its device for its whole
lifetime and its transient scoring allocations are invisible to the rollout
worker's allocator, so they surface as unparkable ``non_torch`` bytes (measured
swing 7.0 -> 12.8 GiB) and trip the phase-handoff parking check. Colocated, the
same allocations belong to the pool and park with everything else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel
from vrl.utils.media import to_uint8

# Verbatim from the release's reward_infer.py quality branch. The head is a
# regressor -- the "output a number from [1,2,3,4,5]" rubric is the training
# prompt, not a parsed instruction -- so this text must not be paraphrased.
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 7 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) object consistency, the consistency of objects or humans in video
please judge whether the video is consistent with the text prompt.
For each dimension, output a number from [1,2,3,4,5],
in which '1' means 'Bad', '2' means 'Poor', '3' means 'Fair', '4' means 'Good', '5' means 'Perfect'.
"""

# reward_infer.py samples this many frames regardless of the 24-frame checkpoint name.
MAX_NUM_FRAMES = 8

# The head regresses onto a 0-100 scale; /100 puts it in [0, 1] like pickscore.
_SCORE_SCALE = 100.0


class AnimeRewardQualityModel(TorchRewardModel):
    """AnimeReward visual-quality regressor (Idefics2-8B), normalised to ~[0, 1]."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        self.model_name = str(self.worker_config.get("model_name", "")).strip()
        if not self.model_name:
            raise ValueError(
                "AnimeRewardQualityModel requires model_name (the quality "
                "checkpoint-final directory)",
            )
        self.num_frames = int(self.worker_config.get("num_frames", MAX_NUM_FRAMES))
        if self.num_frames < 1:
            raise ValueError("AnimeRewardQualityModel num_frames must be >= 1")
        self.load_in_4bit = bool(self.worker_config.get("load_in_4bit", False))
        self._processor: Any = None

    @staticmethod
    def _adapt_mantis_to_current_transformers(model_cls: Any) -> None:
        """Bridge the two 4.x-era API breaks that scoring actually hits.

        Mantis's fork targets transformers 4.x. Its 4.x-only *cache* calls are
        all guarded (``if past_key_value is not None`` / ``if use_cache``) or
        live in ``prepare_inputs_for_generation``, and scoring is a single
        cache-less forward, so none of them execute. Only two breaks remain,
        both signature/attribute shape rather than semantics:

        1. 5.x calls ``tie_weights(recompute_mapping=...)``; the fork's override
           takes no arguments. Widening the signature changes nothing else.
        2. 5.x no longer exposes ``pad_token_id`` on the top-level config. The
           head reads it only to choose the pooled position, and the ``None``
           branch selects the last token -- which is what a batch-of-one score
           wants anyway.

        Verified end to end under transformers 5.13: the same image scores
        0.7250, matching the pinned-4.49 service to 4 decimal places.
        """

        if getattr(model_cls, "_vrl_transformers_adapted", False):
            return
        original_tie_weights = model_cls.tie_weights

        def tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return original_tie_weights(self)

        model_cls.tie_weights = tie_weights
        model_cls._vrl_transformers_adapted = True

    def _load_module(self) -> Any:
        # Mantis's fork, not transformers' Idefics2: the released checkpoint
        # declares architectures=["Idefics2ForSequenceClassification"], which
        # only exists there.
        from mantis.models.idefics2 import Idefics2ForSequenceClassification
        from transformers import AutoProcessor

        self._adapt_mantis_to_current_transformers(Idefics2ForSequenceClassification)

        # AutoProcessor owns CPU tokenization/image transforms only; the returned
        # module is the CUDA state the pool's build frame must capture.
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        if not self.load_in_4bit:
            return self._finalize(
                Idefics2ForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                )
                .eval()
                .to(self.device)
            )

        # A regression head puts quantization error straight onto the score, so
        # this was measured rather than assumed (the sibling Kling judge refuses
        # 4-bit outright). On anima stills nf4 holds the separation that makes
        # this reward worth using -- Cohen's d 3.74 -> 3.61, Spearman +0.976 vs
        # bf16, max per-image delta 0.020 -- while the resident footprint drops
        # 17.3 GiB -> 4.2 GiB. On a single-GPU box that returned headroom is what
        # lets the trainer hold a usable batch, so it buys far more than it costs.
        import torch
        from transformers import BitsAndBytesConfig

        return self._finalize(
            Idefics2ForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
            ).eval()
        )

    @staticmethod
    def _finalize(module: Any) -> Any:
        """Restore the config field the head's pooling step reads (see above)."""

        if not hasattr(module.config, "pad_token_id"):
            module.config.pad_token_id = None
        return module

    def _as_frames(self, media: Any) -> list[Any]:
        """Normalise any supported media payload to a list of PIL frames."""

        import numpy as np
        import torch
        from PIL import Image

        if isinstance(media, torch.Tensor):
            arr = to_uint8(media).cpu().numpy()
            if arr.ndim == 3:
                arr = arr[None]
            if arr.ndim == 4:
                arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            elif arr.ndim == 5:
                arr = arr[0].transpose(0, 2, 3, 1)
            frames = [Image.fromarray(a) for a in arr]
        elif isinstance(media, np.ndarray):
            frames = (
                [Image.fromarray(media)]
                if media.ndim == 3
                else [Image.fromarray(a) for a in media]
            )
        elif isinstance(media, Image.Image):
            frames = [media]
        else:
            raise TypeError(f"AnimeRewardQualityModel cannot score media of type {type(media)!r}")
        return [frame.convert("RGB") for frame in frames]

    def score_media(self, *, media: Any, prompt: str) -> Mapping[str, float]:
        import torch

        frames = self._as_frames(media)
        if not frames:
            raise ValueError("AnimeRewardQualityModel received an empty media payload")
        # A still (or a clip shorter than the head's window) is held for the
        # whole window rather than zero-padded, keeping every frame in-distribution.
        if len(frames) < self.num_frames:
            frames = [frames[len(frames) * i // self.num_frames] for i in range(self.num_frames)]
        elif len(frames) > self.num_frames:
            step = len(frames) / self.num_frames
            frames = [frames[int(i * step)] for i in range(self.num_frames)]

        module = self._module_for_inference()
        eval_prompt = REGRESSION_QUERY_PROMPT + f"\ntext prompt: {prompt}\n"
        content: list[dict[str, Any]] = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": eval_prompt})
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
        )
        inputs = self._processor(text=text, images=frames, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            # use_cache=False is load-bearing, not an optimization: it is what
            # keeps the fork's 4.x-only cache calls unreachable (see
            # _adapt_mantis_to_current_transformers). Scoring is one forward, so
            # there is nothing to cache anyway.
            logits = module(**inputs, use_cache=False).logits
        return {"animereward_quality": float(logits[0].item()) / _SCORE_SCALE}


__all__ = ["MAX_NUM_FRAMES", "REGRESSION_QUERY_PROMPT", "AnimeRewardQualityModel"]
