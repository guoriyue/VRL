"""Fixed-prompt Anima evaluation for before/after checkpoint comparison.

Generates a frozen prompt set at a frozen seed so two checkpoints differ only by
weights, then scores every image with the AnimeReward quality service and
reports quality alongside diversity.

Diversity is reported because reward-model optimization can present as pure mode
collapse: a policy that finds one high-scoring composition and emits it for every
prompt raises mean reward while destroying the model. Quality alone cannot see
that, so a run is only an improvement if quality rises AND diversity holds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, artist name"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-prompt Anima eval with AnimeReward scoring"
    )
    parser.add_argument("--config", default="model/cosmos/anima_preview3")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable config override or additive preset, such as +sampling/denoise=20_step_cfg_4_5.",
    )
    parser.add_argument("--manifest", default="datasets/danbooru/anatomy/eval_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lora-path", default="", help="Checkpoint to evaluate; empty = base model"
    )
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--steps", type=int, default=0, help="0 = follow the config")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8310")
    parser.add_argument("--label", default="", help="Name for this run in the report")
    parser.add_argument(
        "--with-pickscore",
        action="store_true",
        help="Also score with PickScore. Required when PickScore drove training: "
        "a rise there alongside a fall in AnimeReward is reward hacking, not gain.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    rows = _generate(args, out_dir)
    per_component = asyncio.run(
        _score(rows, endpoint=args.endpoint, with_pickscore=args.with_pickscore)
    )
    scores = per_component["animereward_quality"]
    diversity = _diversity(rows)

    report: dict[str, Any] = {
        "label": args.label or ("base" if not args.lora_path else Path(args.lora_path).name),
        "lora_path": args.lora_path,
        "seed": args.seed,
        "prompt_count": len(rows),
        "quality_mean": sum(scores) / len(scores),
        "quality_min": min(scores),
        "quality_max": max(scores),
        "diversity": diversity,
        "per_image": [
            {
                "image": row["image_path"],
                "prompt": row["prompt"],
                "quality": score,
                **(
                    {"pickscore": per_component["pickscore"][i]}
                    if "pickscore" in per_component
                    else {}
                ),
            }
            for i, (row, score) in enumerate(zip(rows, scores, strict=True))
        ],
    }
    if "pickscore" in per_component:
        picks = per_component["pickscore"]
        report["pickscore_mean"] = sum(picks) / len(picks)
    variance = sum((s - report["quality_mean"]) ** 2 for s in scores) / len(scores)
    report["quality_std"] = variance**0.5
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "per_image"},
            indent=2,
            sort_keys=True,
        )
    )


def _generate(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    import torch
    from omegaconf import OmegaConf

    from vrl.config.loading import load_config
    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import parse_config
    from vrl.generation.types import DenoiseRequest
    from vrl.models.dtypes import resolve_torch_dtype
    from vrl.models.families.registry import get_model_family_entry
    from vrl.scripts.eval._device import resolve_eval_device
    from vrl.trainers.data import load_prompt_manifest
    from vrl.utils.media import to_pil_image

    # Preserve this evaluator's established sampling and precision without importing
    # an unrelated training reward, dataset, or mandatory trainer configuration.
    overrides = (
        [
            "+sampling/image=512",
            "+sampling/denoise=20_step_cfg_4_5",
            "precision.float32_precision=tf32",
            "precision.training.dtype=bf16",
            "precision.training.outer_autocast=true",
        ]
        if args.config == "model/cosmos/anima_preview3"
        else []
    )
    overrides.extend(args.override)
    overrides.append("model.torch_compile.enable=false")
    if args.lora_path:
        overrides += ["model.use_lora=true", f"model.lora.path={args.lora_path}"]
    else:
        overrides += ["model.use_lora=false"]
    cfg = load_config(args.config, overrides=overrides)
    root = parse_config(cfg)
    precision = resolve_precision_policy(root.precision)

    prompts = [example.prompt for example in load_prompt_manifest(args.manifest)][: args.limit]
    num_steps = int(args.steps or OmegaConf.select(cfg, "sampling.num_steps", default=20))
    guidance = float(
        OmegaConf.select(cfg, "sampling.guidance_scale", default=4.5)
        if args.guidance_scale is None
        else args.guidance_scale
    )
    width = int(OmegaConf.select(cfg, "sampling.width", default=512))
    height = int(OmegaConf.select(cfg, "sampling.height", default=512))
    max_seq = int(OmegaConf.select(cfg, "sampling.max_sequence_length", default=128))

    device = resolve_eval_device("auto")
    dtype = resolve_torch_dtype("fp32" if device.type == "cpu" else precision.training.dtype)
    entry = get_model_family_entry(str(root.model.family))
    bundle = entry.build_rollout(
        entry.resolve_model_build(
            root, device, precision=precision, parameter_dtype_override=dtype
        )
    )
    model = bundle.model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, prompt in enumerate(prompts):
            encoded = model.encode_prompt(
                [prompt],
                [_DEFAULT_NEGATIVE_PROMPT],
                max_sequence_length=max_seq,
                guidance_scale=guidance,
            )
            request = DenoiseRequest(
                negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
                width=width,
                height=height,
                frame_count=1,
                num_steps=num_steps,
                guidance_scale=guidance,
                # Same seed per index across runs: checkpoints differ only by weights.
                seed=int(args.seed) + index,
            )
            state = model.prepare_sampling(request, encoded)
            for step_idx, timestep in enumerate(state.timesteps):
                step_output = model.forward_step(state, step_idx)
                state.latents = state.scheduler.step(
                    step_output["noise_pred"].float(),
                    timestep,
                    state.latents.float(),
                    return_dict=False,
                )[0]
            image = to_pil_image(model.decode_latents(state.latents)[0])
            path = out_dir / "images" / f"eval_{index:04d}.png"
            image.save(path)
            rows.append({"image_path": str(path), "prompt": prompt, "index": index})
            if (index + 1) % 8 == 0:
                logger.info("generated %s/%s", index + 1, len(prompts))
    del model, bundle
    torch.cuda.empty_cache()
    return rows


async def _score(
    rows: list[dict[str, Any]], *, endpoint: str, with_pickscore: bool = False
) -> dict[str, list[float]]:
    """Score every image with AnimeReward, and optionally PickScore too.

    Both are reported separately and neither is summed into a single number. When
    one reward drives training, a rise in *that* reward alongside a fall in the
    other is the signature of reward hacking rather than improvement — collapsing
    them into one score would hide exactly the thing this eval exists to detect.
    """
    import numpy as np
    import torch
    from PIL import Image

    from vrl.config.reward_inference import RewardInferenceConfig
    from vrl.rewards.functions.registry import MultiReward
    from vrl.rewards.types import RewardSample

    components: dict[str, float] = {"animereward_quality": 1.0}
    kwargs: dict[str, Any] = {
        "animereward_quality": {
            "media_type": "image",
            "artifact_format": "tensor",
            "artifact_dir": "outputs/reward_artifacts",
        }
    }
    inference_configs = {
        "animereward_quality": RewardInferenceConfig(
            kind="http",
            endpoint=endpoint,
            expected_model="animereward-quality",
        ),
    }
    if with_pickscore:
        components["pickscore"] = 1.0
        kwargs["pickscore"] = {"dtype": "float32"}
        inference_configs["pickscore"] = RewardInferenceConfig()

    reward = MultiReward.from_dict(
        components,
        reward_kwargs=kwargs,
        inference_configs=inference_configs,
    )
    samples = []
    for row in rows:
        arr = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        samples.append(
            RewardSample(prompt=row["prompt"], output=tensor, sample_id=f"eval-{row['index']:04d}")
        )
    output = await reward.score_batch(samples)
    per_component = {
        name: [float(v) for v in values] for name, values in (output.components or {}).items()
    }
    per_component.setdefault("animereward_quality", [float(s) for s in output.scores])
    return per_component


def _diversity(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean pairwise distance between images, in pixel and color-histogram space.

    Cheap and model-free on purpose: collapse shows up as near-zero spread here
    long before it is visible in a quality mean.
    """

    import numpy as np
    from PIL import Image

    arrays = [
        np.asarray(Image.open(row["image_path"]).convert("RGB").resize((64, 64)), dtype=np.float32)
        / 255.0
        for row in rows
    ]
    flat = np.stack([a.reshape(-1) for a in arrays])
    hists = np.stack(
        [
            np.concatenate(
                [np.histogram(a[..., c], bins=32, range=(0.0, 1.0))[0] for c in range(3)]
            ).astype(np.float32)
            for a in arrays
        ]
    )
    hists /= hists.sum(axis=1, keepdims=True) + 1e-9

    def mean_pairwise(matrix: np.ndarray) -> float:
        count = matrix.shape[0]
        total, pairs = 0.0, 0
        for i in range(count):
            for j in range(i + 1, count):
                total += float(np.linalg.norm(matrix[i] - matrix[j]))
                pairs += 1
        return total / max(pairs, 1)

    return {
        "pixel_l2": mean_pairwise(flat),
        "color_hist_l2": mean_pairwise(hists),
    }


if __name__ == "__main__":
    main()
