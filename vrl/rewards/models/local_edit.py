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

``local_edit = sqrt(execution * keep)`` and ``local_edit_shaped`` (a small keep
floor for a faithful no-op) are reported, but the validated training key is
``local_edit_execution`` alone. On 160 blind-labelled held-out edits (sprint doc
S7-8) execution separated done from not-done at AUC 0.94, while keep separated
clean edits from judge-labelled collateral at only 0.70: what judges call
"changed something else" is mostly a small object gone, text garbled or the
framing nudged, which patch agreement cannot see, and a legitimate edit placed
outside the reference edit's box is charged as damage. Keep still reads whole-
frame redraws reliably (all <= 0.39), so it stays as a logged observation.

Per-artifact metadata (from the prompt manifest, vrl/scripts/data/local_edit.py)::

    reference_images: [image shown to the model]   # a red-box "hint" copy or the source
    local_edit: {task, box: [x0, y0, x1, y1],      # normalized change box the edit may touch
                 hint: bool, source_image: path}   # clean source: keep and execution compare to it

The edit box comes from the dataset's reference edit, not from a detector, so
this reward has no localisation rule of its own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.base import LazyTorchModule
from vrl.rewards.models.media import artifact_middle_frame_image
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
        self._request_budget_bytes = int(cfg.get("request_budget_bytes", 48 * 1024 * 1024))
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

    def _load_module(self) -> Any:
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self._dino_model)
        return AutoModel.from_pretrained(self._dino_model).eval().to(self.device)

    def score_batch(self, artifacts: Sequence[RewardInferenceArtifact]) -> list[dict[str, float]]:
        """Score one reward phase: a single wake -> score -> park cycle of the execution service.

        The service parks its 7B judge between phases (the rollout and the
        trainer own the card then); scoring per artifact would pay that reload
        every image, so the whole phase goes to it in one request.
        """

        from PIL import Image

        specs: list[Mapping[str, Any]] = []
        sources: list[Any] = []
        source_paths: list[str] = []
        instructions: list[str] = []
        editeds: list[Any] = []
        for artifact in artifacts:
            spec = artifact.metadata.get("local_edit")
            if not isinstance(spec, Mapping):
                raise ValueError("local_edit reward needs metadata.local_edit")
            references = artifact.metadata.get("reference_images") or []
            source_ref = spec.get("source_image") or (references[0] if references else None)
            if not source_ref:
                raise ValueError(
                    "local_edit reward needs local_edit.source_image or one reference image"
                )
            source_path = resolve_artifact_path(
                str(source_ref), data_root=self.data_root, allow_absolute=True
            )
            edited = artifact_middle_frame_image(artifact)
            with Image.open(source_path) as image:
                source = image.convert("RGB").resize(edited.size, Image.Resampling.LANCZOS)
            specs.append(spec)
            sources.append(source)
            source_paths.append(str(source_path))
            instructions.append(artifact.prompt.removesuffix(HINT_SUFFIX).strip())
            editeds.append(edited)
        executions = self._executions(artifacts, instructions, source_paths)
        return [
            self.score(source, edited, spec, execution)
            for source, edited, spec, execution in zip(
                sources, editeds, specs, executions, strict=True
            )
        ]

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        return self.score_batch([artifact])[0]

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

    def _executions(
        self,
        artifacts: Sequence[RewardInferenceArtifact],
        instructions: Sequence[str],
        source_paths: Sequence[str],
    ) -> list[float]:
        """Sigmoid of each EditReward score for (clean source, edited, plain instruction).

        The judge is asked exactly what it was validated on: the clean source
        and the instruction without the hint suffix. Shown the red-box copy and
        the suffixed prompt instead, its done-vs-rest AUC on the hint arm fell
        from 0.89 to 0.72.
        """

        import asyncio
        import concurrent.futures
        import uuid

        from vrl.rewards.inference import RewardInferenceRequest
        from vrl.rewards.service.client import HttpRewardScorer

        # Media goes over the wire as base64 float32; the service caps a request
        # (max_request_bytes, 64 MiB by default), so a phase is split into
        # requests under this budget -- all inside the one wake/park cycle.
        requests: list[RewardInferenceRequest] = []
        chunk: list[RewardInferenceArtifact] = []
        chunk_bytes = 0
        for artifact, instruction, source_path in zip(
            artifacts, instructions, source_paths, strict=True
        ):
            media = artifact.as_media()
            wire_bytes = 4 * media.numel() * 4 // 3 if hasattr(media, "numel") else 0
            if chunk and chunk_bytes + wire_bytes > self._request_budget_bytes:
                requests.append(
                    RewardInferenceRequest(f"local-edit-{uuid.uuid4().hex}", tuple(chunk))
                )
                chunk, chunk_bytes = [], 0
            chunk.append(
                RewardInferenceArtifact(
                    artifact_id=artifact.artifact_id,
                    sample_id=artifact.sample_id,
                    path="",
                    prompt=instruction,
                    metadata={"reference_images": [source_path]},
                    media=media,
                )
            )
            chunk_bytes += wire_bytes
        if chunk:
            requests.append(RewardInferenceRequest(f"local-edit-{uuid.uuid4().hex}", tuple(chunk)))

        async def cycle() -> list[float]:
            # A client per cycle: its HTTP session is bound to the loop that made it.
            scorer = HttpRewardScorer(
                self._execution_endpoint,
                timeout_s=self._execution_timeout_s,
                expected_model=self._execution_model,
            )
            results = []
            try:
                await scorer.activate()
                try:
                    for request in requests:
                        results += request.validate_and_order_results(
                            await scorer.score_batch(request)
                        )
                finally:
                    if scorer.requires_memory_parking:
                        await scorer.park_memory()
            finally:
                await scorer.shutdown()
            return [
                1.0 / (1.0 + math.exp(-float(result.scores[self._execution_key])))
                for result in results
            ]

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(cycle())
        # Called from inside a running loop (the reward runtime is async): drive
        # the cycle on its own loop in a worker thread instead of nesting.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, cycle()).result()


Box = tuple[float, float, float, float]


def _outside_boxes(points: Any, boxes: Sequence[Box]) -> Any:
    """Boolean mask of ``[N, 2]`` points that lie outside every box."""

    import torch

    keep = torch.ones(points.shape[0], dtype=torch.bool)
    for x0, y0, x1, y1 in boxes:
        inside = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
        keep = keep & ~inside
    return keep


def _global_shift(first: Any, second: Any, boxes: Sequence[Box], side: int = 256) -> float:
    """Translation of ``second`` relative to ``first`` (phase-correlation peak) as a share of the diagonal.

    The edited boxes are blanked first: a large object set down in a flat scene
    (a suitcase on grass) would otherwise own the correlation peak.
    """

    import torch

    width, height = first.size

    def gray(image: Any) -> Any:
        values = image.resize((side, side)).convert("L").getdata()
        plane = torch.tensor(list(values), dtype=torch.float32).reshape(side, side)
        keep = torch.ones_like(plane, dtype=torch.bool)
        margin = side // 50  # resampling halo around a blanked box
        for x0, y0, x1, y1 in boxes:
            rows = slice(
                max(int(y0 * side / height) - margin, 0), int(y1 * side / height) + margin + 1
            )
            cols = slice(
                max(int(x0 * side / width) - margin, 0), int(x1 * side / width) + margin + 1
            )
            keep[rows, cols] = False
        plane = torch.where(keep, plane - plane[keep].mean(), torch.zeros(()))
        return plane * (torch.hann_window(side)[:, None] * torch.hann_window(side)[None, :])

    spectrum = torch.fft.fft2(gray(first)) * torch.fft.fft2(gray(second)).conj()
    correlation = torch.fft.ifft2(spectrum / (spectrum.abs() + 1e-8)).real
    dy, dx = divmod(int(correlation.argmax()), side)
    dy, dx = (d - side if d > side // 2 else d for d in (dy, dx))
    return math.hypot(dx, dy) / math.hypot(side, side)


__all__ = ["LocalEditRewardModel"]
