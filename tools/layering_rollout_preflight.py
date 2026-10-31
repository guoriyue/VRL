"""Check one real training row's production conditioning, rollout and replay on each GPU."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from PIL import Image

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.steps.denoise.config import DenoiseLoopConfig
from vrl.generation.steps.denoise.loop import run_denoise_loop
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.registry import get_model_family_entry
from vrl.rollouts.collector import RolloutCollectorConfig
from vrl.utils.config import import_from_path
from vrl.utils.media import to_pil_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/external/coco/layered_val2017/train_mixed_v3.jsonl"),
    )
    parser.add_argument(
        "--config", default="experiment/qwen_image_layered/online_grpo_layered_grouping_fsdp_4x_l4"
    )
    parser.add_argument("--row-source")
    parser.add_argument("--seed-per-rank", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if args.row_source:
        rows = [row for row in rows if row["metadata"]["source"] == args.row_source]
    row = max(rows, key=lambda item: item["request_overrides"]["layers"])
    root = parse_config(
        load_config(
            args.config,
            overrides=[f"model.lora.path={args.adapter}"],
        )
    )
    entry = get_model_family_entry("qwen_image_layered")
    build = entry.resolve_model_build(
        root, torch.device("cuda", rank), precision=PrecisionPolicy.from_section(root.precision)
    )
    model = entry.build_rollout(build, expected_identity=None).model.eval()
    collector = RolloutCollectorConfig.from_root(root)
    request = GenerationRequest(
        request_id=f"layered-preflight-{rank}",
        family=entry.family,
        task=entry.task,
        inputs=[GenerationInput(prompt=row["prompt"], reference_images=[row["reference_image"]])],
        samples_per_prompt=root.rollout.n_samples_per_prompt,
        samples_per_generation_batch=root.rollout.samples_per_generation_batch,
        sampling={
            **collector.request_sampling,
            **row["request_overrides"],
            "seed": 4401 + rank if args.seed_per_rank else 4401,
        },
        denoise=collector.denoise,
        sde_window_seed=0,
    )
    executor = import_from_path(entry.executor_cls)(model, **entry.executor_kwargs(root))
    batch = request.sample_batches()[0]
    params = executor.parse_sampling_params(request)
    loop = DenoiseLoopConfig(
        sample_start=batch.sample_start,
        sample_count=batch.sample_count,
        seed=params.model_request.seed,
        sde=params.sde,
        sde_window=params.sde_window,
        denoise_mode=params.denoise_mode,
        teacache=params.teacache,
    )
    out = args.output / f"rank_{rank}"
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        encoded = executor.encode_prompt_for_batch(
            generation_request=request, params=params, batch=batch
        )
        direct = model.encode_prompt(
            row["prompt"],
            None,
            reference_image=Image.open(row["reference_image"]).convert("RGBA"),
            height=640,
            width=640,
            input_mode="rgb",
        )
        conditioning = {
            key: (encoded[key] is None and direct[key] is None)
            if encoded[key] is None or direct[key] is None
            else bool(torch.equal(encoded[key], direct[key]))
            for key in ("condition_latents", "prompt_embeds", "prompt_embeds_mask")
        }
        if not all(conditioning.values()):
            raise RuntimeError(f"production/eval conditioning mismatch: {conditioning}")
        state = executor.prepare_denoise_state(
            request=params.model_request,
            encoded=executor.expand_conditioning_to_batch(
                encoded=encoded, generation_request=request, batch=batch
            ),
            config=loop,
            prepare_kwargs=executor.build_prepare_kwargs(
                encoded=encoded, generation_request=request, batch=batch
            ),
        )
        result = run_denoise_loop(model=model, state=state, config=loop)
        decoded = model.decode_latents(result.state.latents)
        if not bool(torch.isfinite(decoded).all()) or decoded.float().std().item() < 1e-3:
            raise RuntimeError("nonfinite or degenerate rollout")
        for i, layer in enumerate(decoded[0].cpu()):
            to_pil_image(layer, preserve_alpha=True).save(out / f"layer_{i}.png")
        tensors = {
            key: value[:1] if isinstance(value, torch.Tensor) and value.ndim else value
            for key, value in model.export_replay_tensors(result.state).items()
        }
        context = model.export_batch_context(result.state)
        differences = []
        for step in range(4):
            observation = result.observations[:1, step]
            tensors["timesteps"] = result.timesteps[:1, step].clone()
            restored = model.restore_eval_state(tensors, context, observation, step)
            prediction = model.forward_step(restored, model._replay_forward_step_index(step))[
                "noise_pred"
            ]
            sde = sde_step_with_logprob(
                result.state.scheduler,
                prediction.float(),
                result.timesteps[0, step].unsqueeze(0),
                observation.float(),
                prev_sample=result.actions[:1, step].float(),
                noise_level=loop.sde.noise_level,
                sde_type=loop.sde.sde_type,
                step_index=step,
            )
            differences.append(float((sde.log_prob - result.log_probs[:1, step]).abs().max()))
        witness = {
            "replay_tensors": {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in tensors.items()
            },
            "batch_context": context,
            "latents": observation.cpu(),
            "prediction": prediction.cpu(),
            "step": step,
            "action": result.actions[:1, step].cpu(),
            "rollout_logprob": result.log_probs[:1, step].cpu(),
        }
        torch.save(witness, out / "replay_witness.pt")
    report = {
        "conditioning_exact": conditioning,
        "logprob_abs_diff_per_stochastic_step": differences,
        "max_abs_logprob_diff": max(differences),
        "gate": 0.01,
        "passed": max(differences) <= 0.01,
        "sampling_note": "Production SDE and held-out Heun use different integrators; conditioning is checked exactly and production log-probability is replayed at microbatch one.",
        "reference_image": row["reference_image"],
        "prompt": row["prompt"],
        "layers": row["request_overrides"]["layers"],
        "seed": 4401 + rank if args.seed_per_rank else 4401,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("rollout/replay drift exceeds the recipe gate")


if __name__ == "__main__":
    main()
