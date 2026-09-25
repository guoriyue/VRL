"""Open the frozen Qwen editor and a reward judge for chain evaluation commands."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import torch

from agentic.chains import PolicyStamp
from agentic.roles import LocalEditor, RewardJudge
from vrl.config.precision import PrecisionPolicy
from vrl.config.reward_inference import RewardInferenceConfig
from vrl.config.schema import RewardConfig, parse_config
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards.functions.editreward import EditReward
from vrl.rewards.functions.registry import MultiReward, get_reward
from vrl.rewards.protocols import MemoryParkingScorer
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rewards.service.client import HttpRewardScorer
from vrl.utils.config import import_from_path
from vrl.utils.json_files import canonical_json_sha256, write_json
from vrl.utils.validation import require_int


def editor_sampling(args: argparse.Namespace) -> dict:
    """Resolve and validate the editor geometry before model allocation."""

    height, width = getattr(args, "height", None), getattr(args, "width", None)
    if (height is None) != (width is None):
        raise ValueError("set both --height and --width for a non-square editor canvas")
    height = args.resolution if height is None else height
    width = args.resolution if width is None else width
    for name, value in (("height", height), ("width", width), ("resolution", args.resolution)):
        require_int(value, path=name, minimum=32)
        if value % 32:
            raise ValueError(f"Qwen editor {name} must be divisible by 32")
    require_int(args.steps, path="steps", minimum=1)
    return {
        "height": height,
        "width": width,
        "num_steps": args.steps,
        "guidance_scale": 1.0,
        "reference_resolution": args.resolution,
        "output_mode": args.output_mode,
    }


@asynccontextmanager
async def open_session(args: argparse.Namespace, *, output: Path):
    """Yield ``(editor, judge)``: a frozen Qwen editor and a CPU/HTTP reward judge."""

    from omegaconf import OmegaConf

    sampling = editor_sampling(args)
    if getattr(args, "reward_config", None):
        import yaml

        if getattr(args, "reward_model", None):
            raise ValueError("--reward-model belongs to the direct --reward-endpoint mode")
        recipe = RewardConfig.model_validate(
            yaml.safe_load(Path(args.reward_config).expanduser().read_text())
        )
        if not recipe.components:
            raise ValueError("reward configuration needs at least one component")
        inference = {
            name: recipe.inference.get(name, RewardInferenceConfig(kind="in_process"))
            for name in recipe.components
        }
        for name, placement in inference.items():
            if placement.kind == "ray":
                raise ValueError("chain sessions support local CPU or operator-owned HTTP rewards")
            if placement.kind == "http" and not placement.expected_model_version:
                raise ValueError("HTTP rewards require expected_model_version")
            if (
                placement.kind == "in_process"
                and get_reward(name).resolve_execution_device(
                    device="cpu", kwargs=dict(recipe.kwargs.get(name) or {})
                )
                != "cpu"
            ):
                raise ValueError("chain sessions require local reward components to use CPU")
        revision = canonical_json_sha256(
            {"revision": args.reward_revision, "recipe": recipe.model_dump(mode="json")},
            allow_nan=False,
        )
        write_json(
            output / "reward_recipe.json",
            {"revision": revision, "recipe": recipe.model_dump(mode="json")},
        )
        combination, axis_mapping = None, None
        if recipe.calibration is not None:
            from vrl.config.builders import RewardRuntimeConfig
            from vrl.rewards.deployment import load_reward_deployment

            combination, axis_mapping = load_reward_deployment(
                RewardRuntimeConfig.from_cfg(recipe)
            )
        reward_function = MultiReward.from_dict(
            recipe.components,
            device="cpu",
            reward_kwargs=recipe.kwargs,
            inference_configs=inference,
            combination=combination,
            axis_mapping=axis_mapping,
            image_float32_inputs=combination is not None,
        )
    else:
        if not getattr(args, "reward_model", None):
            raise ValueError("--reward-endpoint requires --reward-model")
        reward_function = EditReward(
            archive_dir=str(output / "reward_archive"),
            debug_dir=str(output / "reward_debug"),
            scorer=HttpRewardScorer(
                RewardInferenceConfig(
                    kind="http",
                    endpoint=args.reward_endpoint,
                    expected_model=args.reward_model,
                    expected_model_version=args.reward_revision,
                )
            ),
        )
        revision = args.reward_revision
    runtime = RewardFunctionRuntime(reward_function)
    try:
        await runtime.preflight()
        components = (
            [component for _, _, component in reward_function.rewards]
            if isinstance(reward_function, MultiReward)
            else [reward_function]
        )
        require_memory_release = False
        for component in components:
            if component.external_accelerator_isolation_verified:
                continue
            scorer = getattr(component, "scorer", None)
            if not isinstance(scorer, MemoryParkingScorer) or not scorer.requires_memory_parking:
                raise ValueError(
                    "every external reward must prove accelerator isolation or memory parking"
                )
            require_memory_release = True
        judge = RewardJudge(
            runtime, revision=revision, require_memory_release=require_memory_release
        )
        await judge.park()
        device = torch.device(args.device)
        root = parse_config(
            OmegaConf.create(
                {
                    "model": {
                        "family": "qwen_image_21",
                        "path": args.path,
                        "revision": args.revision,
                        "use_lora": False,
                        "memory": {
                            "cpu_resident": ["text_encoder"],
                            "vae_decode": {"tiling": True, "slicing": True},
                        },
                    },
                    "precision": {
                        "float32_precision": "ieee",
                        "training": {"dtype": "bf16", "outer_autocast": False},
                    },
                }
            )
        )
        entry = get_model_family_entry("qwen_image_21")
        from vrl.run import resolve_model

        resolved = resolve_model(
            entry,
            root,
            device,
            precision=PrecisionPolicy.from_section(root.precision),
            for_rollout=True,
        )
        model = resolved.materialize(context="Qwen chain editor").model
        write_json(output / "model_identity.json", {"editor": resolved.identity})
        executor = import_from_path(entry.executor_cls)(model, gatherer=entry.new_gatherer())
        editor = LocalEditor(
            model,
            executor,
            policy=PolicyStamp(
                "qwen-image-2.1", canonical_json_sha256(resolved.identity, allow_nan=False), 0
            ),
            family="qwen_image_21",
            sampling=sampling,
            device=device,
        )
        yield editor, judge
    finally:
        await runtime.shutdown()


def add_session_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", required=True, help="Local Qwen Image 2.1 snapshot")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    reward_source = parser.add_mutually_exclusive_group(required=True)
    reward_source.add_argument("--reward-endpoint")
    reward_source.add_argument("--reward-config", help="RewardConfig YAML for CPU/HTTP components")
    parser.add_argument("--reward-model")
    parser.add_argument("--reward-revision", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--height", type=int, help="Explicit output canvas; requires --width")
    parser.add_argument("--width", type=int, help="Explicit output canvas; requires --height")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-mode", choices=("rgb", "rgba"), default="rgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
