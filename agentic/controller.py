"""Image-conditioned categorical control, separate from the diffusion policy.

The next-token logits of single-token action labels define a categorical
policy over the task's actions. Replay stores the actual processor tensors,
so a library or preprocessing change cannot silently alter what the behavior
policy saw.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from agentic import prompt
from agentic.episode import Artifact, Decision, Observation, PolicyStamp, Task
from vrl.utils.artifacts import atomic_file
from vrl.utils.json_files import canonical_json_sha256


class CategoricalController:
    """One vision-language model and its replayable finite-action policy.

    Pass a separately loaded model, never the editor's text encoder. Eval mode
    disables dropout for both sampling and gradient replay; it does not disable
    autograd or trainable adapters.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        policy: PolicyStamp,
        replay_dir: Path,
        device: torch.device,
        max_image_pixels: int = 262144,
        temperature: float = 1.0,
        observation_mode: str = "rgb",
        base_identity: dict[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("controller temperature must be finite and positive")
        if observation_mode not in {"rgb", "rgba"}:
            raise ValueError("controller observation_mode must be rgb or rgba")
        self.model = model.eval()
        self.processor = processor
        self.policy_stamp = policy
        self.replay_dir = replay_dir
        self.device = device
        self.max_image_pixels = max_image_pixels
        self.temperature = temperature
        self.observation_mode = observation_mode
        self.base_identity = base_identity
        self.is_active = False
        self._broken = False

    @classmethod
    def from_qwen_checkpoint(
        cls,
        checkpoint: Path,
        *,
        policy: PolicyStamp,
        replay_dir: Path,
        device: torch.device,
        max_image_pixels: int = 65536,
        temperature: float = 1.0,
        observation_mode: str = "rgb",
    ) -> CategoricalController:
        """Load an independent offline Qwen3-VL backbone with a text-attention LoRA.

        The policy revision becomes a hash of the backbone files, the prompt
        template and the LoRA shape, so checkpoints name the weights they fit.
        """

        from importlib.metadata import version

        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from vrl.models.checkpoint_identity import LocalCheckpointContent

        checkpoint = checkpoint.resolve(strict=True)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
        )
        identity = {
            "schema": "vrl.qwen-visual-controller-base.v2",
            "declared_revision": policy.revision,
            "prompt": {"template": prompt.CONTROLLER, "stop": prompt.STOP, "alpha": prompt.ALPHA},
            "text_encoder": asdict(LocalCheckpointContent.from_path(checkpoint / "text_encoder")),
            "processor": asdict(LocalCheckpointContent.from_path(checkpoint / "processor")),
            "runtime_versions": {
                name: version(name) for name in ("torch", "transformers", "peft", "pillow")
            },
            "dtype": "bfloat16",
            "attention": "sdpa",
            "lora": {
                "rank": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "dropout": lora_config.lora_dropout,
                "targets": lora_config.target_modules,
            },
        }
        processor = AutoProcessor.from_pretrained(checkpoint / "processor", local_files_only=True)
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint / "text_encoder",
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        return cls(
            get_peft_model(backbone, lora_config),
            processor,
            policy=PolicyStamp(
                policy.name, canonical_json_sha256(identity, allow_nan=False), policy.version
            ),
            replay_dir=replay_dir,
            device=device,
            max_image_pixels=max_image_pixels,
            temperature=temperature,
            observation_mode=observation_mode,
            base_identity=identity,
        )

    def advance_policy(self) -> None:
        """Call after a successful optimizer update, between episodes."""

        stamp = self.policy_stamp
        self.policy_stamp = PolicyStamp(stamp.name, stamp.revision, stamp.version + 1)

    def restore_policy_stamp(self, policy: PolicyStamp) -> None:
        if (policy.name, policy.revision) != (self.policy_stamp.name, self.policy_stamp.revision):
            raise ValueError("cannot restore a different controller base policy")
        self.policy_stamp = policy

    async def activate(self) -> None:
        if self._broken:
            raise RuntimeError("controller memory transition failed; replace its owner")
        try:
            self.model.to(self.device)
            self.model.eval()
            self.is_active = True
        except BaseException:
            self._broken = True
            raise

    async def park(self) -> None:
        if self._broken:
            raise RuntimeError("controller memory transition failed; replace its owner")
        try:
            self.model.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            self.is_active = False
        except BaseException:
            self._broken = True
            raise

    def _prepare(
        self, task: Task, observation: Observation
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Processor tensors for this observation plus the action label token IDs."""

        from PIL import Image

        if len(task.actions) > 26:
            raise ValueError("categorical visual control supports at most 26 actions")
        labels = [chr(ord("A") + index) for index in range(len(task.actions))]
        encoded = [
            self.processor.tokenizer.encode(label, add_special_tokens=False) for label in labels
        ]
        if any(len(tokens) != 1 for tokens in encoded):
            raise ValueError("each action label must be exactly one tokenizer token")
        token_ids = [tokens[0] for tokens in encoded]
        alpha = self.observation_mode == "rgba"
        images, alpha_images = [], []
        for artifact in (task.source, observation.current, *task.context_images.values()):
            with Image.open(artifact.path) as image:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                images.append(Image.alpha_composite(background, rgba).convert("RGB"))
                if alpha and len(alpha_images) < 2:
                    alpha_images.append(rgba.getchannel("A").convert("RGB"))
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images + alpha_images],
                    {"type": "text", "text": prompt.render(task, observation, alpha=alpha)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "images_kwargs": {
                    "size": {"shortest_edge": 1024, "longest_edge": self.max_image_pixels}
                }
            },
        )
        return {key: value.detach().cpu() for key, value in inputs.items()}, token_ids

    def _log_probs(self, inputs: dict[str, torch.Tensor], token_ids: list[int]) -> torch.Tensor:
        if not self.is_active or self._broken:
            raise RuntimeError("controller must be healthy and active")
        placed = {
            key: value.to(
                device=self.device,
                dtype=self.model.dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in inputs.items()
        }
        output = self.model(**placed, use_cache=False, logits_to_keep=1)
        logits = output.logits[0, -1, token_ids].float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("non-finite controller action logits")
        return (logits / self.temperature).log_softmax(dim=-1)

    async def decide(self, task: Task, observation: Observation, *, seed: int) -> Decision:
        inputs, token_ids = self._prepare(task, observation)
        digest = observation.digest(task)
        with torch.no_grad():
            log_probs = self._log_probs(inputs, token_ids).cpu()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = int(torch.multinomial(log_probs.exp(), 1, generator=generator).item())
        path = self.replay_dir / f"{uuid.uuid4().hex}.pt"
        with atomic_file(path, binary=True, overwrite=False) as handle:
            torch.save(
                {
                    "inputs": inputs,
                    "token_ids": token_ids,
                    "observation_digest": digest,
                    "policy": asdict(self.policy_stamp),
                    "temperature": self.temperature,
                    "observation_mode": self.observation_mode,
                    "action_log_probs": log_probs.tolist(),
                },
                handle,
            )
        return Decision(
            action=task.actions[selected].name,
            policy=self.policy_stamp,
            old_log_prob=float(log_probs[selected]),
            observation_digest=digest,
            replay={
                "artifact": asdict(Artifact.from_path(path)),
                "action_names": [action.name for action in task.actions],
                "selected_index": selected,
                "action_log_probs": log_probs.tolist(),
            },
        )

    def replay_log_prob(
        self, task: Task, observation: Observation, decision: Decision
    ) -> torch.Tensor:
        """Differentiable likelihood of the recorded decision under the current weights."""

        if decision.observation_digest != observation.digest(task):
            raise ValueError("controller replay observation changed")
        record = decision.replay
        payload = torch.load(record["artifact"]["path"], map_location="cpu", weights_only=True)
        stamp = self.policy_stamp
        if (
            payload["observation_digest"] != decision.observation_digest
            or (payload["policy"]["name"], payload["policy"]["revision"])
            != (stamp.name, stamp.revision)
            or payload["temperature"] != self.temperature
            or payload.get("observation_mode", "rgb") != self.observation_mode
        ):
            raise ValueError("controller replay tensors belong to a different controller setup")
        return self._log_probs(payload["inputs"], payload["token_ids"])[record["selected_index"]]
