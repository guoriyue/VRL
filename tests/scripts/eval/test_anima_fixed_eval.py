from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.scripts.eval import anima_fixed_eval


@pytest.mark.parametrize("overrides, expected_steps", [([], 20), (["sampling.num_steps=40"], 40)])
def test_default_generation_composes_model_without_training_recipe(
    tmp_path, monkeypatch, overrides, expected_steps
) -> None:
    from vrl.models.families import registry
    from vrl.scripts.eval import _device
    from vrl.trainers import data

    class GenerationBoundaryReached(Exception):
        pass

    captured = {}

    def resolve_model_build(root, device, *, precision, parameter_dtype_override):
        captured.update(root=root, precision=precision, dtype=parameter_dtype_override)
        raise GenerationBoundaryReached

    monkeypatch.setattr(data, "load_prompt_manifest", lambda _: [])
    monkeypatch.setattr(_device, "resolve_eval_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        registry,
        "get_model_family_entry",
        lambda _: SimpleNamespace(resolve_model_build=resolve_model_build, build_rollout=None),
    )
    argv = ["--output-dir", str(tmp_path)]
    for override in overrides:
        argv.extend(["--override", override])
    args = anima_fixed_eval.build_parser().parse_args(argv)

    with pytest.raises(GenerationBoundaryReached):
        anima_fixed_eval._generate(args, tmp_path)

    root = captured["root"]
    assert root.trainer is None
    assert root.reward is None
    assert root.sampling.width == root.sampling.height == 512
    assert root.sampling.num_steps == expected_steps
    assert root.sampling.guidance_scale == 4.5
    assert root.model.use_lora is False
    assert root.model.torch_compile.enable is False
    assert captured["precision"].training.dtype == "bf16"
    assert captured["precision"].training.float32_precision == "tf32"
    assert captured["dtype"] == torch.float32
