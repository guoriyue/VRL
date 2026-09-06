from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from tests.quality.preview import build_preview_request, write_preview_image
from vrl.config.schema import parse_config
from vrl.rollouts.collector.config import RolloutCollectorConfig
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
        ),
        config=RolloutCollectorConfig.from_root(
            parse_config(
                OmegaConf.create(
                    {
                        "model": {"family": "sana"},
                        "rollout": {"samples_per_generation_batch": configured_chunk_size},
                        "sampling": {"num_steps": 10, "guidance_scale": 4.5},
                    },
                )
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
    assert request.samples_per_generation_batch == expected_chunk_size
    assert request.sampling == {
        "guidance_scale": 4.5,
        "negative_prompt": "text",
        "num_steps": 10,
        "seed": 101,
    }
    assert not hasattr(request.inputs[0], "metadata")


def test_preview_image_preserves_uint8_and_identity(tmp_path: Path) -> None:
    image = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(1, 3, 4, 5)
    output = SimpleNamespace(
        request_id="request-0",
        output=image,
        sample_rows=[
            SimpleNamespace(prompt="prompt"),
        ],
    )
    path = tmp_path / "000.png"

    write_preview_image(
        output,
        path,
        expected_request_id="request-0",
        expected_prompt="prompt",
    )

    with Image.open(path) as persisted:
        actual = np.asarray(persisted.convert("RGB"))
    np.testing.assert_array_equal(actual, image[0].permute(1, 2, 0).numpy())


def test_preview_image_rejects_changed_prompt_identity() -> None:
    output = SimpleNamespace(
        request_id="request-0",
        output=torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
        sample_rows=[SimpleNamespace(prompt="different")],
    )
    with pytest.raises(RuntimeError, match="changed request/prompt identity"):
        write_preview_image(
            output,
            Path("unused.png"),
            expected_request_id="request-0",
            expected_prompt="prompt",
        )
