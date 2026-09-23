"""Released EditReward inference behind VRL's lazy, parkable model boundary.

Run the upstream Qwen2.5 implementation in its compatible environment, normally
through the HTTP service. Generation dependencies remain independent.
"""

from __future__ import annotations

import hashlib
import json
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
        locality_path = self.config.get("locality_config")
        self.locality_config = (
            json.loads(Path(locality_path).read_text()) if locality_path else None
        )

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
        # Match the source/candidate geometry used by the baseline experiment.
        source = source.resize(image.size, Image.Resampling.LANCZOS)
        evidence = None
        if self.locality_config is not None:
            from vrl.rewards.models.color_locality import color_locality, combine_locality
            from vrl.utils.artifacts import sha256_file

            task_id = artifact.metadata.get("task_id")
            rule = self.locality_config["tasks"].get(task_id)
            if rule is None or rule["prompt"] != artifact.prompt:
                raise ValueError(f"Uncalibrated locality task/instruction: {task_id!r}")
            if sha256_file(source_path) != rule["source_sha256"]:
                raise ValueError(f"Locality source image changed: {task_id!r}")
            evidence = color_locality(source, image, rule, self.locality_config)
        with torch.inference_mode():
            values = (
                self._module_for_inference()
                .reward(prompts=[artifact.prompt], image_src=[source], image_paths=[image])[0]
                .float()
                .cpu()
                .tolist()
            )
        scores = {"editreward": float(values[0]), "editreward_log_sigma": float(values[1])}
        if evidence is not None:
            scores.update(
                combine_locality(
                    evidence,
                    scores["editreward"],
                    quality_weight=self.locality_config["quality_weight"],
                )
            )
        audit_dir = self.config.get("audit_dir")
        if audit_dir:
            root = Path(audit_dir)
            root.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256(artifact.artifact_id.encode()).hexdigest()
            image.save(root / f"{name}.png")
            record = {
                "artifact_id": artifact.artifact_id,
                "sample_id": artifact.sample_id,
                "prompt": artifact.prompt,
                "metadata": artifact.metadata,
                "scores": scores,
                "image": f"{name}.png",
                "locality_config": self.locality_config,
            }
            (root / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        return scores
