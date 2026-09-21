"""Manifest reference ordering and RGBA options reach the Qwen batch encoder."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.families.qwen_image_21.runtime import QwenImage21BatchExecutor
from vrl.models.families.registry import get_model_family_entry
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trainers.data.artifacts import resolve_prompt_example_references
from vrl.trainers.data.prompts import load_prompt_examples_from_jsonl_bytes


def test_manifest_references_reach_encoder_in_order_with_alpha(tmp_path) -> None:
    Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(tmp_path / "source.png")
    Image.new("RGB", (96, 64), (0, 255, 0)).save(tmp_path / "leaf.png")
    payload = json.dumps(
        {
            "prompt": "Put image 2 onto image 1",
            "reference_images": ["source.png", "leaf.png"],
            "request_overrides": {"output_mode": "rgba", "reference_resolution": 512},
        }
    ).encode()
    example = resolve_prompt_example_references(
        load_prompt_examples_from_jsonl_bytes(payload)[0], data_root=tmp_path
    )
    built = GenerationRequestBuilder(
        entry=get_model_family_entry("qwen_image_21"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 10}),
    ).build([example.generation_input()], 2, request_overrides=example.request_overrides)
    assert built.metadata["reference_images"] == [
        str(tmp_path / "source.png"),
        str(tmp_path / "leaf.png"),
    ]
    assert built.request.sampling["output_mode"] == "rgba"
    request = GenerationRequest(
        request_id="edit",
        family="qwen_image_21",
        task="t2i",
        inputs=[GenerationInput(prompt="unrelated"), example.generation_input()],
        samples_per_prompt=2,
        sampling={
            "height": 512,
            "width": 512,
            "num_steps": 10,
            "guidance_scale": 1.0,
            **example.request_overrides,
        },
    )
    model = SimpleNamespace(encode_prompt=Mock(return_value={}))
    executor = QwenImage21BatchExecutor(model)
    params = executor.parse_sampling_params(request)
    executor.encode_prompt_for_batch(
        generation_request=request,
        model_request=params.model_request,
        params=params,
        batch=GenerationSampleBatch(prompt_index=1, sample_start=0, sample_count=2),
    )
    args, kwargs = model.encode_prompt.call_args
    assert args[0] == example.prompt
    images = kwargs["reference_images"]
    assert [im.getpixel((0, 0)) for im in images] == [(255, 0, 0, 0), (0, 255, 0, 255)]
    assert kwargs["output_mode"] == "rgba" and kwargs["reference_resolution"] == 512


def test_ambiguous_and_invalid_reference_lists_fail_before_generation() -> None:
    with pytest.raises(ValueError, match="not both"):
        GenerationInput(prompt="edit", reference_image="a.png", reference_images=["b.png"])
    with pytest.raises(ValueError, match="list of non-empty"):
        GenerationInput(prompt="edit", reference_images="a.png")
    with pytest.raises(ValueError, match="reference_images must be a list"):
        load_prompt_examples_from_jsonl_bytes(b'{"prompt":"edit", "reference_images":"a.png"}')


def test_single_reference_alias_and_text_only_requests_share_the_executor(tmp_path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "red").save(source)
    model = SimpleNamespace(encode_prompt=Mock(return_value={}))
    executor = QwenImage21BatchExecutor(model)
    request = GenerationRequest(
        request_id="single",
        family="qwen_image_21",
        task="t2i",
        inputs=[
            GenerationInput(prompt="edit", reference_image=str(source)),
            GenerationInput(prompt="draw"),
        ],
        samples_per_prompt=1,
        sampling={"height": 64, "width": 64, "num_steps": 3, "guidance_scale": 1.0},
    )
    params = executor.parse_sampling_params(request)
    for index in range(2):
        executor.encode_prompt_for_batch(
            generation_request=request,
            model_request=params.model_request,
            params=params,
            batch=GenerationSampleBatch(prompt_index=index, sample_start=0, sample_count=1),
        )
    assert model.encode_prompt.call_args_list[0].kwargs["reference_images"][0].mode == "RGBA"
    assert model.encode_prompt.call_args_list[1].kwargs["reference_images"] == []
