"""VideoCon-Physics reward model (Ray-actor backend).

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

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.ray.model import RewardModel

logger = logging.getLogger(__name__)

_DEFAULT_REWARD_MODEL = "videophysics/videocon_physics"
_DEFAULT_REVISION = "main"
_DEFAULT_NUM_FRAMES = 32
_DEFAULT_VIDEO_TOKEN = "<|video|>"

# Templates use the videophy paper conversational format. The {caption} slot is
# filled with the rollout prompt. {video} expands to the special media token
# (``<|video|>``) that the MplugOwlProcessor maps to a vision-feature block.
_DEFAULT_PHYSICS_TEMPLATE = (
    "\nHuman: {video}\nDoes the video follow physical commonsense?\nAI: "
)
_DEFAULT_SEMANTIC_TEMPLATE = (
    "\nHuman: {video}\nDoes the video entail the caption: \"{caption}\"?\nAI: "
)

_SCORE_KEY_ALIASES = {
    "physical_commonsense": "physical_commonsense",
    "physics": "physical_commonsense",
    "semantic_adherence": "semantic_adherence",
    "semantic": "semantic_adherence",
    "overall": "overall",
    "overall_reward": "overall",
}


class VideoConPhysicsModel(RewardModel):
    """Load VideoCon-Physics and score one (prompt, video) pair per call."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        self.reward_model_name = str(
            self.worker_config.get("reward_model_name")
            or self.worker_config.get("model_name")
            or "",
        ).strip()
        self.model_root = _resolve_model_root(self.worker_config)
        self.dtype = _torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.num_frames = int(self.worker_config.get("num_frames", _DEFAULT_NUM_FRAMES))
        self.physics_template = str(
            self.worker_config.get("physics_template", _DEFAULT_PHYSICS_TEMPLATE),
        )
        self.semantic_template = str(
            self.worker_config.get("semantic_template", _DEFAULT_SEMANTIC_TEMPLATE),
        )

        logger.info(
            "Loading VideoCon-Physics root=%s device=%s dtype=%s frames=%d",
            self.model_root,
            self.device,
            self.dtype,
            self.num_frames,
        )

        _ensure_mplug_owl_video_on_path()

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
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        del request
        prompt = artifact.prompt or str(artifact.metadata.get("prompt", ""))
        if not prompt:
            raise ValueError(
                f"VideoCon-Physics requires a caption for artifact {artifact.artifact_id!r}",
            )
        video_path = str(Path(artifact.as_path()).expanduser().resolve())

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
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
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


def preflight_videocon_physics_backend() -> None:
    """Validate vendored backend imports without downloading weights."""

    _ensure_mplug_owl_video_on_path()
    import mplug_owl_video  # noqa: F401
    import transformers  # noqa: F401


def _ensure_mplug_owl_video_on_path() -> None:
    """Add the ``mplug_owl_video`` directory from the videophy submodule to
    ``sys.path`` so we can ``import mplug_owl_video`` without prefixing a
    deep ``third_party/videophy/videocon/training/pipeline_video`` path.

    Raises a clear error if the submodule has not been initialized — fix
    with ``git submodule update --init --recursive``.
    """

    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    parent_dir = (
        repo_root / "third_party" / "videophy"
        / "videocon" / "training" / "pipeline_video"
    )
    needed = parent_dir / "mplug_owl_video" / "modeling_mplug_owl.py"
    if not needed.exists():
        raise RuntimeError(
            "VideoCon-Physics requires the vendored mPLUG-Owl-Video sources. "
            f"Missing file: {needed}. Initialize the videophy submodule with "
            "`git submodule update --init --recursive`."
        )
    parent_str = str(parent_dir)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)


def _resolve_model_root(worker_config: Mapping[str, Any]) -> Path:
    """Return a local directory containing the checkpoint files.

    Either ``worker_config.model_path`` is set (local checkpoint) or we
    snapshot_download ``videophysics/videocon_physics``.
    """

    model_path = str(worker_config.get("model_path", "")).strip()
    if model_path:
        root = Path(model_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"VideoCon-Physics model_path missing: {root}")
        return root

    reward_model_name = str(
        worker_config.get("reward_model_name")
        or worker_config.get("model_name")
        or "",
    ).strip()
    if not reward_model_name:
        reward_model_name = _DEFAULT_REWARD_MODEL

    repo_id, separator, revision = reward_model_name.rpartition("@")
    if not separator:
        repo_id = reward_model_name
        revision = _DEFAULT_REVISION
    else:
        revision = revision or _DEFAULT_REVISION

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision)).resolve()


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported VideoCon-Physics torch_dtype: {name!r}")


__all__ = [
    "VideoConPhysicsModel",
    "preflight_videocon_physics_backend",
]
