"""Tag adherence reward: WD14 tagger recall over the prompt's own danbooru tags.

Every judge-based quality reward tried on this policy was too noisy to drive
GRPO (test-retest rank correlation 0.1-0.5, ties on ~60% of prompts), so most of the
within-group advantage it produced was noise. Tag adherence on NSFW prompts is,
by contrast, a measured weakness with a verifiable target: base-model recall
0.913 with only 61% of images matching every requested tag, versus 0.98 / 81%
on SFW prompts. Scoring the prompt's own tags with a deterministic tagger gives
a zero-noise signal — the same image always gets the same score — and a
headroom the policy can actually close.

Preprocessing (white square pad, 448 bicubic, RGB->BGR, raw 0-255 floats) is a
verbatim copy of imgutils' ``_prepare_image_for_tagging`` because the headroom
baseline above was measured through imgutils; any drift here would silently
move the target the reward is validated against. The tagger is loaded directly
through onnxruntime so the reward carries no extra dependency.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

# SwinV2 v3 is trained at a fixed square input; the ONNX graph has no resize.
WD14_INPUT_SIZE = 448
# selected_tags.csv category ids: 0 = general, 4 = character, 9 = rating.
_GENERAL_CATEGORY = 0


class TagAdherenceRewardModel:
    """Per-artifact recall of requested tags in ``[0, 1]`` (higher = more adherent)."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._model_repo = str(cfg.get("model_repo", "SmilingWolf/wd-swinv2-tagger-v3"))
        self._model_file = str(cfg.get("model_file", "model.onnx"))
        self._tags_file = str(cfg.get("tags_file", "selected_tags.csv"))
        self._threshold = _validate_probability("threshold", cfg.get("threshold", 0.35))
        self._metadata_key = str(cfg.get("metadata_key", "adherence_tags"))
        if not self._metadata_key:
            raise ValueError("tag_adherence metadata_key must be non-empty")
        self._providers = list(cfg.get("providers") or ["CPUExecutionProvider"])
        # Test seam: a callable returning per-image ``{tag: prob}`` bypasses onnxruntime.
        self._tagger: Callable[[list[Any]], Sequence[Mapping[str, float]]] | None = cfg.get(
            "tagger"
        )
        self._session: Any = None
        self._input_name = ""
        self._general_labels: list[str] = []
        self._general_index: Any = None

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        # Validate every artifact's tag list before running the tagger so a
        # malformed manifest row fails fast instead of after a full batch.
        wanted = [self._wanted_tags(artifact) for artifact in artifacts]
        images = [_artifact_image(artifact) for artifact in artifacts]
        return [
            {"tag_adherence": self._recall(tags, probs)}
            for tags, probs in zip(wanted, self.tag_images(images), strict=True)
        ]

    def __call__(self, artifact: Any) -> dict[str, float]:
        return self.score_batch([artifact])[0]

    def tag_images(self, images: list[Any]) -> list[dict[str, float]]:
        """Return per-image ``{general_tag: probability}`` for the whole batch."""

        if not images:
            return []
        if self._tagger is not None:
            results = self._tagger(images)
            if len(results) != len(images):
                raise ValueError(
                    "tag_adherence tagger returned wrong number of results: "
                    f"got {len(results)}, expected {len(images)}",
                )
            return [{str(tag): float(prob) for tag, prob in result.items()} for result in results]

        import numpy as np

        self._ensure_loaded()
        batch = np.concatenate([prepare_wd14_input(image) for image in images], axis=0)
        probs = self._session.run(None, {self._input_name: batch})[0]
        # v3 heads emit sigmoid outputs already; the clip only guards against
        # float noise so thresholds behave as probabilities.
        probs = np.clip(np.asarray(probs, dtype=np.float32), 0.0, 1.0)
        return [
            dict(zip(self._general_labels, row[self._general_index].tolist(), strict=True))
            for row in probs
        ]

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import numpy as np
        import onnxruntime
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(self._model_repo, self._model_file)
        tags_path = hf_hub_download(self._model_repo, self._tags_file)
        with open(tags_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        # Row order is the model's output index; only general tags are
        # matchable — ratings/characters never appear in an adherence target.
        general = [
            (index, row["name"])
            for index, row in enumerate(rows)
            if int(row["category"]) == _GENERAL_CATEGORY
        ]
        self._general_index = np.asarray([index for index, _ in general], dtype=np.int64)
        self._general_labels = [name for _, name in general]
        self._session = onnxruntime.InferenceSession(model_path, providers=self._providers)
        self._input_name = self._session.get_inputs()[0].name

    def _wanted_tags(self, artifact: Any) -> set[str]:
        raw = artifact.metadata.get(self._metadata_key)
        if isinstance(raw, str) or not isinstance(raw, (list, tuple, set, frozenset)):
            raise ValueError(
                f"tag_adherence requires metadata[{self._metadata_key!r}] to be a "
                f"non-empty list of tag strings on artifact {artifact.artifact_id!r}, "
                f"got {type(raw).__name__}",
            )
        wanted = {str(tag).strip().lower() for tag in raw if str(tag).strip()}
        if not wanted:
            raise ValueError(
                f"tag_adherence requires a non-empty metadata[{self._metadata_key!r}] "
                f"tag list on artifact {artifact.artifact_id!r}",
            )
        return wanted

    def _recall(self, wanted: set[str], probs: Mapping[str, float]) -> float:
        detected = {tag.lower() for tag, prob in probs.items() if prob >= self._threshold}
        return len(wanted & detected) / len(wanted)


def prepare_wd14_input(image: Image.Image, size: int = WD14_INPUT_SIZE) -> np.ndarray:
    """Return a ``[1, size, size, 3]`` float32 BGR batch, imgutils-identical.

    White padding (not black) and BGR order are what the tagger was trained
    with; the raw 0-255 scale is intentional — the ONNX graph normalizes.
    """

    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    max_dim = max(width, height)
    canvas = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    canvas.paste(rgb, ((max_dim - width) // 2, (max_dim - height) // 2))
    if max_dim != size:
        canvas = canvas.resize((size, size), Image.BICUBIC)
    bgr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(np.expand_dims(bgr, axis=0))


def _artifact_image(artifact: Any) -> Image.Image:
    """Middle frame of the artifact as an RGB PIL image, the tagger's input."""

    from PIL import Image

    from vrl.rewards.models.media import decode_artifact_frames
    from vrl.utils.media import to_pil_image

    if not artifact.path or artifact.path.endswith(".pt"):
        media = artifact.as_media()
        if isinstance(media, Image.Image):
            return media.convert("RGB")
    frames = decode_artifact_frames(artifact, 1)
    return to_pil_image(frames[frames.shape[0] // 2])


def _validate_probability(name: str, value: float) -> float:
    out = float(value)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must satisfy 0.0 <= {name} <= 1.0")
    return out


__all__ = ["WD14_INPUT_SIZE", "TagAdherenceRewardModel", "prepare_wd14_input"]
