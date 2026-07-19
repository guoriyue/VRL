from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from tests.quality.preview import build_preview_request, write_preview_image
from vrl.rollouts.collector.config import build_rollout_config_from_cfg
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trainers.data.prompts import PromptExample


@pytest.mark.parametrize(
    ("configured_chunk_size", "expected_chunk_size"),
    [(8, 8), ("auto", 1)],
)
def test_preview_request_uses_real_prompt_overrides_and_one_sample(
    configured_chunk_size: object,
    expected_chunk_size: int,
) -> None:
    builder = GenerationRequestBuilder(
        entry=SimpleNamespace(
            family="sana",
            task="t2i",
            request_metadata_namespace=None,
        ),
        config=build_rollout_config_from_cfg(
            OmegaConf.create(
                {
                    "rollout": {"samples_per_chunk": configured_chunk_size},
                    "sampling": {"num_steps": 10, "guidance_scale": 4.5},
                },
            ),
        ),
    )
    example = PromptExample(
        prompt="a red apple",
        request_overrides={"negative_prompt": "text"},
        metadata={"source": "fixture"},
        task_type="text_to_image",
    )

    request = build_preview_request(builder, example, seed=101)

    assert request.prompts == ["a red apple"]
    assert request.samples_per_prompt == 1
    assert request.sampling == {
        "guidance_scale": 4.5,
        "negative_prompt": "text",
        "num_steps": 10,
        "samples_per_chunk": expected_chunk_size,
        "seed": 101,
    }
    assert request.inputs[0].metadata == {"source": "fixture"}


def test_preview_image_preserves_uint8_and_identity(tmp_path: Path) -> None:
    image = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(1, 3, 4, 5)
    output = SimpleNamespace(
        error=None,
        output=image,
        sample_rows=[SimpleNamespace(prompt="prompt", seed=7)],
    )
    path = tmp_path / "000.png"

    write_preview_image(output, path, expected_prompt="prompt", seed=7)

    with Image.open(path) as persisted:
        actual = np.asarray(persisted.convert("RGB"))
    np.testing.assert_array_equal(actual, image[0].permute(1, 2, 0).numpy())


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (
            SimpleNamespace(error="oom", output=[], sample_rows=[]),
            "returned an error",
        ),
        (
            SimpleNamespace(
                error=None,
                output=torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
                sample_rows=[SimpleNamespace(prompt="different", seed=7)],
            ),
            "changed prompt/seed identity",
        ),
    ],
)
def test_preview_image_rejects_invalid_executor_output(output: object, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        write_preview_image(
            output,
            Path("unused.png"),
            expected_prompt="prompt",
            seed=7,
        )
