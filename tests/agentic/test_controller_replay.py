"""Categorical replay reproduces the sampled likelihood and trains the real VL forward."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from agentic.controller import CategoricalController
from agentic.episode import Action, Artifact, Observation, PolicyStamp, Task


@pytest.mark.asyncio
async def test_image_conditioned_likelihood_replay_and_gradient(tmp_path):
    torch.manual_seed(7)
    model = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            text_config={
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [],
            },
            image_token_id=29,
            video_token_id=28,
            vision_start_token_id=30,
            vision_end_token_id=31,
        )
    )

    class Processor:
        tokenizer = SimpleNamespace(encode=lambda label, **kwargs: [3 + ord(label) - ord("A")])

        def apply_chat_template(self, messages, **kwargs):
            # Real vision patches, two images with four 2x2 spatial patches each.
            images = messages[0]["content"][:2]
            values = [image["image"].getpixel((0, 0))[0] / 255 for image in images]
            return {
                "input_ids": torch.tensor([[30, 29, 31, 30, 29, 31, 5]]),
                "attention_mask": torch.ones(1, 7, dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 1, 0, 0]]),
                "pixel_values": torch.cat([torch.full((4, 24), value) for value in values]),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            }

    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "white").save(source)
    task = Task(
        "edit",
        "Make it black",
        Artifact.from_path(source),
        (Action("black", "edit", "Make it black"), Action("stop", "stop")),
    )
    observation = Observation(0, 2, task.source)
    controller = CategoricalController(
        model,
        Processor(),
        policy=PolicyStamp("tiny-qwen", "test", 0),
        replay_dir=tmp_path / "replay",
        device=torch.device("cpu"),
    )
    await controller.activate()
    decision = await controller.decide(task, observation, seed=12)
    replay = controller.replay_log_prob(task, observation, decision)
    assert replay.item() == pytest.approx(decision.old_log_prob, abs=1e-7)
    assert sum(np.exp(decision.replay["action_log_probs"])) == pytest.approx(1.0)
    controller.observation_mode = "rgba"
    with pytest.raises(ValueError, match="different controller setup"):
        controller.replay_log_prob(task, observation, decision)
    controller.observation_mode = "rgb"
    with pytest.raises(ValueError, match="observation changed"):
        controller.replay_log_prob(task, replace(observation, remaining_tool_calls=1), decision)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    (-replay).backward()
    assert model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
    optimizer.step()
    controller.advance_policy()
    assert controller.replay_log_prob(task, observation, decision).item() > replay.item()
    await controller.park()
    with pytest.raises(RuntimeError, match="healthy and active"):
        controller._log_probs({}, [3, 4])


def test_prompt_and_images_come_from_the_task_row(tmp_path):
    class Processor:
        tokenizer = SimpleNamespace(encode=lambda label, **kwargs: [ord(label)])

        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            images = [item["image"] for item in content if item["type"] == "image"]
            self.prompt = content[-1]["text"]
            return {"pixels": torch.from_numpy(np.stack([np.asarray(image) for image in images]))}

    source, current, style, target = [
        tmp_path / name for name in ("source.png", "current.png", "style.png", "oracle.png")
    ]
    Image.new("RGB", (8, 8), "white").save(source)
    Image.new("RGBA", (8, 8), (255, 255, 255, 128)).save(current)
    Image.new("RGB", (8, 8), "blue").save(style)
    Image.new("RGB", (8, 8), "red").save(target)
    task = Task.from_manifest_record(
        {
            "task_id": "opacity",
            "instruction": "Preserve partial transparency",
            "requirement": "Match the style reference",
            "source": "source.png",
            "context_images": {"style reference": "style.png"},
            "reward_assets": {"target_image": "oracle.png"},
            "actions": [
                {"name": "edit", "kind": "edit", "instruction": "Adjust opacity"},
                {"name": "stop", "kind": "stop"},
            ],
        },
        base_dir=tmp_path,
    )
    observation = Observation(1, 1, Artifact.from_path(current), "edit")
    processor = Processor()
    controller = CategoricalController(
        torch.nn.Linear(1, 1),
        processor,
        policy=PolicyStamp("fixture", "v1", 0),
        replay_dir=tmp_path / "replay",
        device=torch.device("cpu"),
    )
    rgb, _ = controller._prepare(task, observation)
    # Original, current (composited on white) and the declared context image.
    assert rgb["pixels"].shape[0] == 3
    assert torch.equal(rgb["pixels"][0], rgb["pixels"][1])
    assert torch.all(rgb["pixels"][2, ..., 2] == 255)
    assert processor.prompt == (
        "Image 1 is the original image.\n"
        "Image 2 is the current result.\n"
        "Image 3 is style reference (context).\n"
        "Instruction: Preserve partial transparency\n"
        "Requirements: Match the style reference\n"
        "Previous action: edit\n"
        "Remaining editing calls: 1\n"
        "Choose the next action. Reply with its letter only.\n"
        "A: Adjust opacity\n"
        "B: Stop editing."
    )
    controller.observation_mode = "rgba"
    rgba, _ = controller._prepare(task, observation)
    assert rgba["pixels"].shape[0] == 5
    assert torch.all(rgba["pixels"][3] == 255)
    assert torch.all(rgba["pixels"][4] == 128)
    assert "Image 5 is the alpha mask of image 2" in processor.prompt
    # The red oracle is a reward asset, so it never enters perception.
    red = torch.tensor([255, 0, 0], dtype=torch.uint8)
    assert not torch.any(torch.all(rgba["pixels"].reshape(5, -1, 3) == red, dim=(1, 2)))
