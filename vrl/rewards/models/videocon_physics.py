"""VideoCon-Physics reward model for the in-process reward runtime.

Wraps the vendored ``mplug_owl_video`` module under ``third_party/`` to load the
KwaiVGI / VideoPhy ``videophysics/videocon_physics`` checkpoint and score
generated videos on two axes:

* ``physical_commonsense`` — "Does the video follow physical commonsense?"
* ``semantic_adherence``   — "Does the video entail the caption?"

Both are computed as ``P(Yes) / (P(Yes) + P(No))`` at the final non-pad token
position, matching ``entailment_inference.py`` from the upstream repo.

VideoCon-Physics is not in the HuggingFace ``transformers`` library, so we
cannot use ``AutoModelForCausalLM``. We import the model class from the
vendored ``third_party.mplug_owl_video`` module directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.assets.video_judge_prompts import (
    VIDEOCON_PHYSICS_TEMPLATE,
    VIDEOCON_SEMANTIC_TEMPLATE,
)
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.base import require_prompt_and_video_path
from vrl.rewards.models.hub import resolve_model_root
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

_DEFAULT_REWARD_MODEL = "videophysics/videocon_physics"
_DEFAULT_NUM_FRAMES = 32
_DEFAULT_VIDEO_TOKEN = "<|video|>"


class VideoConPhysicsModel:
    """Load VideoCon-Physics and score one (prompt, video) pair per call."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        # Drift fix: this family previously dropped ``local_files_only`` on the
        # snapshot_download path; the shared resolver forwards it.
        self.model_root = resolve_model_root(
            self.worker_config,
            default_model=_DEFAULT_REWARD_MODEL,
            family="VideoCon-Physics",
        )
        self.dtype = resolve_torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.num_frames = int(self.worker_config.get("num_frames", _DEFAULT_NUM_FRAMES))
        self.physics_template = str(
            self.worker_config.get("physics_template", VIDEOCON_PHYSICS_TEMPLATE),
        )
        self.semantic_template = str(
            self.worker_config.get("semantic_template", VIDEOCON_SEMANTIC_TEMPLATE),
        )

        logger.info(
            "Loading VideoCon-Physics root=%s device=%s dtype=%s frames=%d",
            self.model_root,
            self.device,
            self.dtype,
            self.num_frames,
        )

        from mplug_owl_video.modeling_mplug_owl import (
            MplugOwlForConditionalGeneration,
        )
        from mplug_owl_video.processing_mplug_owl import (
            MplugOwlImageProcessor,
            MplugOwlProcessor,
        )
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(str(self.model_root))
        image_processor = MplugOwlImageProcessor.from_pretrained(str(self.model_root))
        processor = MplugOwlProcessor(image_processor, tokenizer)
        model = MplugOwlForConditionalGeneration.from_pretrained(
            str(self.model_root),
            torch_dtype=self.dtype,
        )
        model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model.to(self.device)

        # Pre-resolve Yes/No token ids — used in every score call.
        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = tokenizer.encode("No", add_special_tokens=False)
        if not yes_ids or not no_ids:
            raise RuntimeError(
                "VideoCon-Physics tokenizer did not yield single-token Yes/No ids",
            )
        self.token_id_yes = int(yes_ids[0])
        self.token_id_no = int(no_ids[0])
        self._softmax = nn.Softmax(dim=2)

    def __call__(
        self,
        artifact: RewardInferenceArtifact,
    ) -> dict[str, float]:
        prompt, video_path = require_prompt_and_video_path(
            artifact,
            family="VideoCon-Physics",
        )

        physics_text = self.physics_template.format(
            video=_DEFAULT_VIDEO_TOKEN,
            caption=prompt,
        )
        semantic_text = self.semantic_template.format(
            video=_DEFAULT_VIDEO_TOKEN,
            caption=prompt,
        )

        physics_score = self._score_entailment(physics_text, video_path)
        semantic_score = self._score_entailment(semantic_text, video_path)
        overall = 0.5 * (physics_score + semantic_score)

        return {
            "physical_commonsense": physics_score,
            "semantic_adherence": semantic_score,
            "overall": overall,
        }

    def _score_entailment(self, prompt_text: str, video_path: str) -> float:
        """One forward pass; return P(Yes)/(P(Yes)+P(No)) at last non-pad position."""

        batch = self.processor(
            text=[prompt_text],
            videos=[video_path],
            num_frames=self.num_frames,
            return_tensors="pt",
        )
        # Floating tensors from the processor are FP32; the model is bf16.
        # Cast floating tensors (pixel / video_pixel) to the model's dtype.
        # Leave integer tensors (input_ids, attention_mask) alone.
        model_dtype = next(self.model.parameters()).dtype
        casted_batch: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if v.is_floating_point():
                    v = v.to(model_dtype)
                casted_batch[k] = v
            else:
                casted_batch[k] = v
        batch = casted_batch
        # Match upstream batchify() for the video-only path: no image tensor,
        # zero images, and one video feature block per sample.
        batch_size = int(batch["input_ids"].shape[0])
        batch.setdefault("pixel_values", None)
        if "num_images" not in batch:
            batch["num_images"] = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        if "num_videos" not in batch:
            batch["num_videos"] = torch.ones(batch_size, dtype=torch.long, device=self.device)
        input_ids = batch["input_ids"]
        with torch.no_grad():
            output = self.model(**batch)
        logits = output["logits"] if isinstance(output, dict) else output.logits
        probs = self._softmax(logits)

        pad_id = self.tokenizer.pad_token_id
        # Locate the last non-pad position in the (single-batch) row.
        row_ids = input_ids[0].tolist()
        last_idx = len(row_ids) - 1
        if pad_id is not None:
            for i, tid in enumerate(row_ids):
                if tid == pad_id:
                    last_idx = max(i - 1, 0)
                    break

        p_yes = probs[0, last_idx, self.token_id_yes]
        p_no = probs[0, last_idx, self.token_id_no]
        denom = (p_yes + p_no).clamp(min=1e-8)
        return float((p_yes / denom).item())


__all__ = ["VideoConPhysicsModel"]
