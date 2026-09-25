"""Local-edit reward: the instruction was carried out, and nothing outside its region changed.

Two measurements, both from released models, multiplied:

* **Execution** -- a trained instruction-following edit scorer (EditReward,
  Qwen2.5-VL-7B) served over HTTP, asked with the plain instruction and the
  clean source. Its unbounded log-odds-like score is squashed with a sigmoid.
* **Keep** -- the share of DINOv2 patch tokens outside the edit box (padded)
  whose cosine to the same patch of the source stays above ``patch_match``,
  times a frame-shift guard from the phase-correlation peak with the box
  blanked (both taken from the object-move reward, where kept scenes all
  scored >= 0.75 and redrawn scenes <= 0.33 on 183 blind-labelled edits).

``local_edit = sqrt(execution * keep)``; a redrawn frame has no keep and an
untouched frame has no execution, so neither shortcut scores. The training key
``local_edit_shaped = (local_edit + w * keep) / (1 + w)`` leaves a faithful
no-op a small floor, so GRPO does not push every faithful-but-unfinished sample
below a redraw (the run1 failure mode of the move reward).

Per-artifact metadata (from the prompt manifest, vrl/scripts/data/local_edit.py)::

    reference_images: [image shown to the model]   # a red-box "hint" copy or the source
    local_edit: {task, box: [x0, y0, x1, y1],      # normalized change box the edit may touch
                 hint: bool, source_image: path}   # clean source: keep and execution compare to it

The edit box comes from the dataset's reference edit, not from a detector, so
this reward has no localisation rule of its own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.base import LazyTorchModule
from vrl.rewards.models.media import artifact_middle_frame_image
from vrl.rewards.models.object_move import _global_shift, _outside_boxes
from vrl.scripts.data.local_edit import HINT_SUFFIX
from vrl.utils.artifacts import default_data_root, resolve_artifact_path


class LocalEditRewardModel(LazyTorchModule):
    """Score one edited image against its clean source, edit box and instruction."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(worker_config)
        self.device = str(cfg.get("device", "cuda"))
        self.data_root = str(cfg.get("data_root") or default_data_root())
        self._dino_model = str(cfg.get("dino_model", "facebook/dinov2-large"))
        # Operator-run EditReward service (vrl.rewards.service.server with the
        # editreward model); the execution half of the score comes from it.
        self._execution_endpoint = str(cfg.get("execution_endpoint", "http://127.0.0.1:18316"))
        self._execution_model = str(cfg.get("execution_model", "editreward-qwen25-7b"))
        self._execution_key = str(cfg.get("execution_key", "editreward"))
        self._execution_timeout_s = float(cfg.get("execution_timeout_s", 600.0))
        # A patch is kept when its DINOv2 token still matches the source's at this cosine.
        self._patch_match = float(cfg.get("patch_match", 0.5))
        # The frame stayed when its global translation is under this share of the diagonal.
        self._stay_tolerance = float(cfg.get("stay_tolerance", 0.01))
        # The edit box is grown by this share of the frame before exclusion,
        # so a change that spills a little past the reference edit's box is not charged.
        self._box_pad = float(cfg.get("box_pad", 0.02))
        self._background_weight = float(cfg.get("background_weight", 0.2))
        for name, value, low, high in (
            ("patch_match", self._patch_match, -1.0, 1.0),
            ("stay_tolerance", self._stay_tolerance, 1e-6, 1.0),
            ("box_pad", self._box_pad, 0.0, 0.5),
            ("background_weight", self._background_weight, 0.0, 1.0),
        ):
            if not low <= value <= high:
                raise ValueError(f"local_edit {name} must lie in [{low}, {high}]")
        self._processor: Any | None = None
        self._scorer: Any | None = None

    def _load_module(self) -> Any:
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self._dino_model)
        return AutoModel.from_pretrained(self._dino_model).eval().to(self.device)

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        spec = artifact.metadata.get("local_edit")
        if not isinstance(spec, Mapping):
            raise ValueError("local_edit reward needs metadata.local_edit")
        references = artifact.metadata.get("reference_images") or []
        source_ref = spec.get("source_image") or (references[0] if references else None)
        if not source_ref:
            raise ValueError(
                "local_edit reward needs local_edit.source_image or one reference image"
            )
        from PIL import Image

        source_path = resolve_artifact_path(
            str(source_ref), data_root=self.data_root, allow_absolute=True
        )
        edited = artifact_middle_frame_image(artifact)
        with Image.open(source_path) as image:
            source = image.convert("RGB").resize(edited.size, Image.Resampling.LANCZOS)
        execution = self._execution(artifact, str(source_path))
        return self.score(source, edited, spec, execution)

    def score(
        self, source: Any, edited: Any, spec: Mapping[str, Any], execution: float
    ) -> dict[str, float]:
        """Compose the keep term (measured here) with an execution score in [0, 1]."""

        import torch

        if source.size != edited.size:
            raise ValueError("local_edit compares images of equal size")
        box = spec.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("local_edit spec needs box [x0, y0, x1, y1]")
        width, height = edited.size
        x0, y0, x1, y1 = (float(v) for v in box)
        pad = self._box_pad
        region = (
            max(x0 - pad, 0.0) * width,
            max(y0 - pad, 0.0) * height,
            min(x1 + pad, 1.0) * width,
            min(y1 + pad, 1.0) * height,
        )
        first, second = self._patches(source), self._patches(edited)
        side = first.shape[0]
        rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        centers = torch.stack(
            [(cols.flatten() + 0.5) * width / side, (rows.flatten() + 0.5) * height / side], -1
        )
        outside = _outside_boxes(centers, [region])
        if bool(outside.any()):
            cosine = (first * second).sum(-1).flatten()[outside]
            kept = float((cosine >= self._patch_match).float().mean())
        else:
            kept = 1.0  # the box covers the frame: nothing outside it to keep
        shift = _global_shift(source, edited, [region])
        stayed = min(max(2.0 - shift / self._stay_tolerance, 0.0), 1.0)
        keep = kept * stayed
        execution = min(max(float(execution), 0.0), 1.0)
        score = math.sqrt(execution * keep)
        return {
            "local_edit": score,
            "local_edit_shaped": (score + self._background_weight * keep)
            / (1.0 + self._background_weight),
            "local_edit_execution": execution,
            "local_edit_keep": keep,
            "local_edit_kept_share": kept,
            "local_edit_shift": shift,
        }

    def _patches(self, image: Any) -> Any:
        """Unit-norm DINOv2 patch tokens of one image as an ``[n, n, d]`` grid."""

        import torch

        dino = self._module_for_inference()
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            tokens = dino(**inputs).last_hidden_state[0, 1:].float().cpu()
        side = round(tokens.shape[0] ** 0.5)
        return torch.nn.functional.normalize(tokens, dim=-1).reshape(side, side, -1)

    def _execution(self, artifact: RewardInferenceArtifact, source_path: str) -> float:
        """Sigmoid of the EditReward score for (clean source, edited, plain instruction)."""

        from vrl.rewards.inference import RewardInferenceRequest
        from vrl.rewards.service.client import HttpRewardScorer

        if self._scorer is None:
            self._scorer = HttpRewardScorer(
                self._execution_endpoint,
                timeout_s=self._execution_timeout_s,
                expected_model=self._execution_model,
            )
        instruction = artifact.prompt.removesuffix(HINT_SUFFIX).strip()
        request = RewardInferenceRequest(
            request_id=f"local-edit-{artifact.artifact_id}",
            artifacts=(
                RewardInferenceArtifact(
                    artifact_id=artifact.artifact_id,
                    sample_id=artifact.sample_id,
                    path="",
                    prompt=instruction,
                    metadata={"reference_images": [source_path]},
                    media=artifact.as_media(),
                ),
            ),
        )
        (result,) = _run(self._scorer.score_batch(request))
        raw = float(result.scores[self._execution_key])
        return 1.0 / (1.0 + math.exp(-raw))


def _run(coroutine: Any) -> Any:
    """Drive one coroutine to completion from sync scoring code, inside or outside a running loop."""

    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()
