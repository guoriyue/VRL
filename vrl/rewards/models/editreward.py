"""Released EditReward inference behind VRL's lazy, parkable model boundary.

Run the upstream Qwen2.5 implementation in its compatible environment, normally
through the HTTP service. Generation dependencies remain independent.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.base import LazyTorchModule
from vrl.rewards.models.media import artifact_middle_frame_image
from vrl.utils.artifacts import resolve_artifact_path


class EditRewardModel(LazyTorchModule):
    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(worker_config)
        self.device = str(self.config.get("device", "cuda"))
        self.data_root = Path(self.config["data_root"]).expanduser().resolve()

    def _load_module(self) -> Any:
        import yaml
        from huggingface_hub import snapshot_download

        upstream = Path(self.config["upstream_root"]).expanduser().resolve()
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        from EditReward import EditRewardInferencer

        config = yaml.safe_load(
            (upstream / "EditReward/config/EditReward-Qwen2.5-7B-VL.yaml").read_text()
        )
        config["disable_flash_attn2"] = True
        config["model_name_or_path"] = snapshot_download(
            self.config["base_model"],
            revision=self.config["base_revision"],
            local_files_only=True,
        )
        checkpoint = snapshot_download(
            self.config["reward_model_name"],
            revision=self.config["revision"],
            local_files_only=True,
        )
        with tempfile.TemporaryDirectory(prefix="vrl-editreward-config-") as directory:
            config["output_dir"] = directory
            path = Path(directory) / "inference.yaml"
            path.write_text(yaml.safe_dump(config))
            return EditRewardInferencer(
                config_path=str(path),
                checkpoint_path=checkpoint,
                device=self.device,
                reward_dim="overall_detail",
                rm_head_type="ranknet_multi_head",
            )

    def __call__(self, artifact: RewardInferenceArtifact) -> Mapping[str, float]:
        import torch
        from PIL import Image

        references = artifact.metadata.get("reference_images") or []
        if len(references) != 1:
            raise ValueError("This EditReward checkpoint adapter supports exactly one reference")
        reference = references[0]
        if not isinstance(reference, str) or not reference or not artifact.prompt.strip():
            raise ValueError("EditReward requires a reference image and a non-empty instruction")
        source_path = resolve_artifact_path(
            reference, data_root=self.data_root, allow_absolute=True
        )
        image = artifact_middle_frame_image(artifact)
        with Image.open(source_path) as original:
            rgba = original.convert("RGBA")
            source = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert(
                "RGB"
            )
        # Score the source at the candidate's geometry.
        source = source.resize(image.size, Image.Resampling.LANCZOS)
        with torch.inference_mode():
            values = (
                self._module_for_inference()
                .reward(prompts=[artifact.prompt], image_src=[source], image_paths=[image])[0]
                .float()
                .cpu()
                .tolist()
            )
        return {"editreward": float(values[0]), "editreward_log_sigma": float(values[1])}
