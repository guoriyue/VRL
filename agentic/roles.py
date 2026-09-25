"""Evaluation roles for a chain run, over the framework's own runtimes.

``LocalEditor`` runs a frozen family model in-process through its batch
executor, one image per step; ``RewardJudge`` scores every state through a
``RewardRuntime`` in one call. Neither owns an optimizer or a scheduler. On a
shared GPU each role parks its memory between turns, and a failed memory
transition retires the role rather than reusing it.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agentic.chains import Artifact, EditChain, PolicyStamp, Score
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.parking import ModelParking
from vrl.rewards.protocols import RewardRuntime
from vrl.rewards.types import RewardSample
from vrl.trajectory.storage import TrajectoryStoragePolicy
from vrl.utils.media import to_pil_image, write_png


class LocalEditor:
    """Frozen native-denoise editing over an existing family model and executor."""

    def __init__(
        self,
        model: Any,
        executor: Any,
        *,
        policy: PolicyStamp,
        family: str,
        sampling: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.eval().requires_grad_(False)
        self.executor = executor
        self.policy_stamp = policy
        self.family = family
        self.sampling = dict(sampling)
        self.device = device
        self._parking = ModelParking()
        self._broken = False
        self._active = False

    async def park(self) -> None:
        if self._broken:
            raise RuntimeError("editor memory transition failed; replace its owner")
        try:
            self._parking.park(self.model, restore_device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            self._active = False
        except BaseException:
            self._broken = True
            raise

    async def activate(self) -> None:
        if self._broken:
            raise RuntimeError("editor memory transition failed; replace its owner")
        try:
            if self._parking.restore_device is not None:
                self._parking.restore()
            else:
                self.model.to(self.device)
                self.model.move_frozen_components(self.device)
            self._active = True
        except BaseException:
            self._broken = True
            raise

    async def edit(
        self, chain: EditChain, index: int, current: Artifact, *, seed: int, output_dir: Path
    ) -> Artifact:
        """One edit: the step's instruction applied to the current image."""

        if not self._active or self._broken:
            raise RuntimeError("editor must be healthy and active")
        request = GenerationRequest(
            request_id=f"chain-edit-{uuid.uuid4().hex}",
            family=self.family,
            task="t2i",
            inputs=[
                GenerationInput(prompt=chain.steps[index].prompt, reference_images=[current.path])
            ],
            samples_per_prompt=1,
            sampling={**self.sampling, "seed": seed},
            denoise=DenoiseRequestOptions(denoise_mode="native"),
            trajectory_storage=TrajectoryStoragePolicy(device="cpu", dtype="float32"),
            policy_version=self.policy_stamp.version,
        )
        with torch.inference_mode():
            batch = self.executor.forward_batch(
                request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
            )
            result = self.executor.merge_generation_batches(
                request, request.sample_rows(), [batch]
            )
        path = output_dir / f"step{index:02d}.png"
        write_png(result.output[0], path)
        return Artifact.from_path(path)


class RewardJudge:
    """Score every state against the chain's instructions and source image, in one call."""

    def __init__(
        self, runtime: RewardRuntime, *, revision: str, require_memory_release: bool
    ) -> None:
        if not revision:
            raise ValueError("judge revision is required")
        self.runtime = runtime
        self.revision = revision
        self.require_memory_release = require_memory_release
        self._activated = False
        self._broken = False
        self._initialized = False

    async def activate(self) -> None:
        if self._broken:
            raise RuntimeError("judge memory transition failed; replace its owner")
        # Count a partially completed activation, so the next park releases it.
        self._activated = True
        await self.runtime.activate()

    async def park(self) -> None:
        if self._broken:
            raise RuntimeError("judge memory transition failed; replace its owner")
        # Initialize once so an external GPU service acknowledges release before
        # any other role takes the GPU.
        if not self._initialized:
            await self.activate()
        if not self._activated:
            return
        try:
            await self.runtime.park_memory(required=self.require_memory_release)
            self._activated = False
            self._initialized = True
        except BaseException:
            self._broken = True
            raise

    async def score(self, chain: EditChain, artifacts: list[Artifact]) -> list[Score | None]:
        from PIL import Image

        if not self._activated or self._broken:
            raise RuntimeError("judge must be healthy and active")
        samples = []
        for artifact in artifacts:
            with Image.open(artifact.path) as image:
                pixels = np.array(to_pil_image(image, preserve_alpha=True), copy=True)
            media = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1).float() / 255.0
            samples.append(
                RewardSample(
                    prompt=chain.instruction,
                    output=media,
                    sample_id=f"{chain.chain_id}:{artifact.sha256}",
                    metadata={
                        "reference_images": [chain.source.path],
                        "chain_id": chain.chain_id,
                        **({"requirement": chain.requirement} if chain.requirement else {}),
                        **{name: asset.path for name, asset in chain.reward_assets.items()},
                    },
                )
            )
        result = await self.runtime.score(samples)
        return [
            Score(total, {name: values[index] for name, values in result.components.items()})
            for index, total in enumerate(result.scores)
        ]
