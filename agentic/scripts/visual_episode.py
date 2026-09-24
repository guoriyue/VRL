"""Run a bounded real Qwen visual episode with an independent controller and judge.

This executable collects inference episodes. It does not train either policy.
The task JSON contains task_id, instruction, source, explicit actions
[{name, kind: edit|stop, instruction?}], and optional reward-only assets.
An HTTP judge must already be running; local CPU verifiers need no service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from agentic.controller import CategoricalController
from agentic.episode import (
    Controller,
    Episode,
    PolicyStamp,
    Task,
)
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


def visual_editor_sampling(args: argparse.Namespace) -> dict:
    """Resolve and validate the shared editor geometry before model allocation."""
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
async def open_visual_session(
    args: argparse.Namespace, *, output: Path, controller: Controller | None = None
):
    """Own the shared controller/editor/judge lifecycle for collection and training CLIs."""
    from omegaconf import OmegaConf

    sampling = visual_editor_sampling(args)
    if getattr(args, "reward_config", None):
        import yaml

        if getattr(args, "reward_model", None):
            raise ValueError("--reward-model belongs to the direct --reward-endpoint mode")
        recipe = RewardConfig.model_validate(
            yaml.safe_load(Path(args.reward_config).expanduser().read_text())
        )
        if not recipe.components:
            raise ValueError("visual reward configuration needs at least one component")
        inference = {
            name: recipe.inference.get(name, RewardInferenceConfig(kind="in_process"))
            for name in recipe.components
        }
        for name, placement in inference.items():
            if placement.kind == "ray":
                raise ValueError(
                    "visual sessions support local CPU or operator-owned HTTP rewards"
                )
            if placement.kind == "http" and not placement.expected_model_version:
                raise ValueError("visual HTTP rewards require expected_model_version")
            if (
                placement.kind == "in_process"
                and get_reward(name).resolve_execution_device(
                    device="cpu", kwargs=dict(recipe.kwargs.get(name) or {})
                )
                != "cpu"
            ):
                raise ValueError("visual sessions require local reward components to use CPU")
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
                    "every external visual reward must prove accelerator isolation or memory parking"
                )
            require_memory_release = True
        judge = RewardJudge(
            runtime, revision=revision, require_memory_release=require_memory_release
        )
        await judge.park()
        device = torch.device(args.device)
        if controller is None:
            controller = CategoricalController.from_qwen_checkpoint(
                Path(args.path),
                policy=PolicyStamp("qwen3-vl-controller", args.revision, 0),
                replay_dir=output / "controller_replay",
                device=device,
                temperature=args.temperature,
                observation_mode=getattr(args, "controller_observation", "rgb"),
            )
            controller_identity = controller.base_identity
        else:
            controller_identity = {
                "kind": "provided-controller",
                "policy": asdict(controller.policy_stamp),
            }
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
        model = resolved.materialize(context="Qwen visual episode editor").model
        write_json(
            output / "model_identity.json",
            {
                "editor": resolved.identity,
                "controller": controller_identity,
            },
        )
        executor = import_from_path(entry.executor_cls)(model, gatherer=entry.new_gatherer())
        tool = LocalEditor(
            model,
            executor,
            policy=PolicyStamp(
                "qwen-image-2.1", canonical_json_sha256(resolved.identity, allow_nan=False), 0
            ),
            family="qwen_image_21",
            sampling=sampling,
            device=device,
        )
        yield controller, tool, judge
    finally:
        await runtime.shutdown()


async def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path(args.task).expanduser().resolve()
    payload = json.loads(manifest.read_text())
    task = Task.from_manifest_record(payload, base_dir=manifest.parent)
    async with open_visual_session(args, output=output) as (controller, tool, judge):
        write_json(output / "settings.json", vars(args))
        trace = await Episode(max_tool_calls=args.max_tool_calls, tool_cost=args.tool_cost).run(
            task, controller, tool, judge, output_dir=output / "episode", seed=args.seed
        )
        print(
            json.dumps({key: trace[key] for key in ("status", "termination", "tool_calls")}),
            flush=True,
        )


def add_visual_session_arguments(parser: argparse.ArgumentParser) -> None:
    """Keep collection and controller-training runtime options consistent."""
    parser.add_argument("--path", required=True)
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
    parser.add_argument("--max-tool-calls", type=int, default=2)
    parser.add_argument("--tool-cost", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--controller-observation",
        choices=("rgb", "rgba"),
        default="rgb",
        help="rgba also shows original/current alpha masks to the controller",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_visual_session_arguments(parser)
    parser.add_argument("--task", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
