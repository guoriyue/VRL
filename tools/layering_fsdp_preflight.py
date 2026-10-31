"""Restore ckpt-28 on four FSDP ranks and check a full-size backward without updating."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import torch

from vrl.config.builders import BuiltConfigs
from vrl.config.loading import load_config
from vrl.trainers.activation_checkpointing import enable_transformer_gradient_checkpointing
from vrl.trainers.checkpointing import TrainingCheckpoint, _require_equal_tensor_tree
from vrl.trainers.distributed import DistributedTrainingContext, TrainerProcessGroup
from vrl.trainers.strategy import build_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        "experiment/qwen_image_layered/online_grpo_layered_grouping_fsdp_4x_l4",
        overrides=[
            "model.lora.path=null",
            f"trainer.resume_from={args.checkpoint}",
        ],
    )
    built = BuiltConfigs.from_cfg(cfg)
    context = DistributedTrainingContext.from_root(built.root, device=torch.device("cuda"))
    strategy = build_strategy(built.root, context)
    owner = TrainerProcessGroup.open(context, output_dir=args.output)
    try:
        build = built.family.resolve_model_build(
            built.root, context.device, precision=built.precision, for_rollout=False
        )
        bundle = built.family.build_replay(
            replace(build, materialize_weights=strategy.materialize_weights),
            expected_identity=build.checkpoint_identity(),
        )
        enable_transformer_gradient_checkpointing(bundle, built.root)
        model = strategy.prepare_model(bundle.model)
        checkpoint = TrainingCheckpoint.load(args.checkpoint)
        checkpoint.restore_model(
            bundle=bundle,
            family=built.family.family,
            expected_model_identity=build.checkpoint_identity(),
            strategy=strategy,
        )
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, eps=1e-12, weight_decay=1e-4)
        strategy.load_optimizer_state(model, optimizer, checkpoint.trainer_state["optimizer"])
        exported = strategy.export_checkpoint_optimizer_state(model, optimizer)
        if context.is_primary:
            _require_equal_tensor_tree(
                checkpoint.trainer_state["optimizer"], exported, label="FSDP restored Adam"
            )
        strategy.collectives.coordination_barrier()
        print(f"rank {context.rank}: checkpoint and Adam restored", flush=True)
        witness = torch.load(args.witness, map_location="cpu", weights_only=True)
        tensors = {
            k: v.to(context.device) if isinstance(v, torch.Tensor) else v
            for k, v in witness["replay_tensors"].items()
        }
        state = model.restore_eval_state(
            tensors,
            witness["batch_context"],
            witness["latents"].to(context.device),
            witness["step"],
        )
        with torch.no_grad():
            fsdp_prediction = model.forward_step(state, 0)["noise_pred"]
        prediction_diff = float(
            (fsdp_prediction - witness["prediction"].to(context.device)).abs().max()
        )
        from vrl.math.denoise.flow_matching import sde_step_with_logprob

        model._set_dynamic_timesteps(10, 1600, context.device)
        replay = sde_step_with_logprob(
            model.scheduler,
            fsdp_prediction.float(),
            tensors["timesteps"].reshape(-1),
            state.latents.float(),
            prev_sample=witness["action"].to(context.device).float(),
            noise_level=0.5,
            sde_type="flow_grpo",
            step_index=witness["step"],
        )
        logprob_diff = float(
            (replay.log_prob - witness["rollout_logprob"].to(context.device)).abs().max()
        )
        if not strategy.collectives.all_true(logprob_diff <= 0.01):
            raise RuntimeError(f"FSDP replay parity failed: {logprob_diff}")
        print(
            f"rank {context.rank}: FSDP prediction max diff={prediction_diff}, logprob max diff={logprob_diff}",
            flush=True,
        )
        cache = torch.load(args.cache, map_location="cpu", weights_only=True)
        layers = 11
        latents = torch.randn(
            1, (layers + 1) * 1600, 64, generator=torch.Generator().manual_seed(4401)
        ).to(context.device)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.predict_velocity(
                latents,
                torch.tensor(1000.0, device=context.device),
                condition_latents=cache["condition_latents"].to(context.device),
                prompt_embeds=cache["prompt_embeds"].to(context.device),
                prompt_embeds_mask=None
                if cache["prompt_embeds_mask"] is None
                else cache["prompt_embeds_mask"].to(context.device),
                layers=layers,
                height=640,
                width=640,
            )
            loss = prediction.float().square().mean()
        strategy.backward(loss)
        grads = [p.grad.to_local() for p in parameters if p.grad is not None]
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in grads)
        expected_grads = {
            name
            for name, saved in checkpoint.trainer_state["optimizer"]["state"].items()
            if float(saved["step"]) > 0
        }
        missing = [
            name
            for name, p in model.named_parameters()
            if name in expected_grads and p.grad is None
        ]
        if not strategy.collectives.all_true(finite and not missing):
            raise RuntimeError("FSDP backward produced missing or nonfinite LoRA gradients")
        result = {
            "rank": context.rank,
            "world_size": context.world_size,
            "layers": layers,
            "loss": loss.item(),
            "gradient_tensors": len(grads),
            "all_finite": finite,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_step_performed": False,
            "prediction_max_abs_diff": prediction_diff,
            "logprob_max_abs_diff": logprob_diff,
        }
        (args.output / f"rank_{context.rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        owner.shutdown()


if __name__ == "__main__":
    main()
