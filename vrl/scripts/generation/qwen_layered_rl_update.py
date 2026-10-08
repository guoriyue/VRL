"""Run reviewed, reward-driven Flow-GRPO updates on a fixed layered SDE batch.

This small-batch integration runner reuses the repository GRPO objective. It is
not a replacement for iterative rollout collection or held-out quality testing.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

import torch

from vrl.algorithms.grpo.continuous import GRPO, FlashGRPO, FlashGRPOConfig, GRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.rollouts.evaluators.types import FlowSDESignal, TrajectorySignalBatch
from vrl.scripts.generation.qwen_layered_rl_probe import (
    load_transformer,
    load_upstream,
    predict_velocity,
    recorded_noise_levels,
    recorded_schedule,
)


def restore_optimizer(optimizer, parameters, directory, *, adapter_sha256, optimizer_sha256):
    """Restore Adam moments only when their saved policy and parameter order match."""
    directory = Path(directory)
    adapter_path = directory / "adapter.pt"
    optimizer_path = directory / "optimizer.pt"
    result = json.loads((directory / "result.json").read_text())
    if (
        hashlib.sha256(adapter_path.read_bytes()).hexdigest() != adapter_sha256
        or result["adapter_sha256"] != adapter_sha256
    ):
        raise ValueError("Optimizer parent does not match the initial adapter")
    if hashlib.sha256(optimizer_path.read_bytes()).hexdigest() != optimizer_sha256:
        raise ValueError("Optimizer checkpoint hash mismatch")
    saved_parameters = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if list(saved_parameters) != list(parameters):
        raise ValueError("Optimizer parameter order differs from the saved adapter")
    state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    current_groups = optimizer.state_dict()["param_groups"]
    if state["param_groups"] != current_groups:
        raise ValueError("Optimizer groups or hyperparameters differ from the parent")
    ids = [index for group in state["param_groups"] for index in group["params"]]
    if set(state["state"]) - set(ids):
        raise ValueError("Optimizer state contains unknown parameters")
    for index, (name, parameter) in zip(ids, parameters.items(), strict=True):
        if saved_parameters[name].shape != parameter.shape:
            raise ValueError(f"Saved adapter shape mismatch: {name}")
        moments = state["state"].get(index)
        if moments is None:
            continue  # Unconnected parameters can legitimately have no Adam state.
        for key in ("exp_avg", "exp_avg_sq"):
            value = moments[key]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid optimizer moment: {name}.{key}")
        step = moments["step"]
        if step.numel() != 1 or not torch.isfinite(step).all() or step.item() <= 0:
            raise ValueError(f"Invalid optimizer step: {name}")
    if not state["state"]:
        raise ValueError("Optimizer checkpoint has no learned state")
    optimizer.load_state_dict(state)
    return {
        "directory": str(directory.absolute()),
        "adapter_sha256": adapter_sha256,
        "optimizer_sha256": optimizer_sha256,
        "previous_optimizer_updates": result.get(
            "optimizer_updates_total", result["optimizer_updates"]
        ),
    }


def run_update(root, config):
    from diffusers import FlowMatchEulerDiscreteScheduler

    if config.get("deterministic_algorithms", False):
        # The launcher must supply CUBLAS_WORKSPACE_CONFIG before CUDA starts.
        # Fail on unsupported kernels instead of silently accepting noisy replay.
        torch.use_deterministic_algorithms(True)
    normalization = config.get("density_normalization", "mean")
    if normalization not in {"mean", "sqrt_dimension"}:
        raise ValueError(f"Unsupported density normalization: {normalization}")
    plan = json.loads((root / "plan.json").read_text())
    reward_directory = root / config.get("reward_directory", "reward_calibration")
    scores_path = reward_directory / "scores.json"
    validation = json.loads((reward_directory / "validation.json").read_text())
    assert validation["usable_for_small_update"] is True
    assert validation["scores_sha256"] == hashlib.sha256(scores_path.read_bytes()).hexdigest()
    scored = json.loads(scores_path.read_text())
    assert scored["status"] == "complete"
    cases = {case["name"]: case for case in plan["cases"] if case["split"] == "train"}
    rows = [row for row in scored["rows"] if row["kind"] == "candidate"]
    assert rows and all(row["valid"] and row["case"] in cases for row in rows)
    groups = torch.tensor([list(cases).index(row["case"]) for row in rows])
    assert all((groups == group).sum() >= 2 for group in groups.unique())
    rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float32)
    assert torch.isfinite(rewards).all()
    rectify = config.get("temporal_rectification", False)
    assert isinstance(rectify, bool)
    algorithm_class, config_class = (FlashGRPO, FlashGRPOConfig) if rectify else (GRPO, GRPOConfig)
    algorithm = algorithm_class(
        config_class(
            clip_ratio=config["clip_ratio"], kl_coef=config["kl_coef"], flow_kl_use_dt=True
        )
    )
    advantages = algorithm.compute_advantages_from_tensors(rewards, groups)
    assert torch.isfinite(advantages).all() and advantages.abs().max() > 0, "No reward signal"
    epochs = config.get("ppo_epochs", 1)
    assert isinstance(epochs, int) and epochs > 0
    output = root / "update01"
    output.mkdir(exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    upstream = load_upstream(upstream_path)
    model = load_transformer(plan)
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    initial_sha256 = hashlib.sha256((root / "initial_adapter.pt").read_bytes()).hexdigest()
    initial = torch.load(root / "initial_adapter.pt", weights_only=True)
    assert initial.keys() == parameters.keys()
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(initial[name])
    optimizer = torch.optim.AdamW(parameters.values(), lr=config["learning_rate"], weight_decay=0)
    resumed_optimizer = None
    if resume := config.get("resume_optimizer"):
        resumed_optimizer = restore_optimizer(
            optimizer,
            parameters,
            root / resume["directory"],
            adapter_sha256=initial_sha256,
            optimizer_sha256=resume["sha256"],
        )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        plan["base_path"], subfolder="scheduler", local_files_only=True
    )
    indices = config["timesteps"]
    assert len(set(indices)) == len(indices) and all(
        0 <= index < plan["steps"] for index in indices
    )
    count = len(rows) * len(indices)
    if rectify:
        # Only the temporal loss weighting changes. These remain the existing
        # full-SDE trajectories, not the full single-stochastic-step Flash recipe.
        recorded_timesteps = []
        selected_noise = set()
        schedule = None
        for row in rows:
            trajectory = torch.load(
                root / "rollouts" / row["case"] / row["candidate"] / "trajectory.pt",
                map_location="cpu",
                weights_only=True,
            )
            sigmas, times, _ = recorded_schedule(trajectory)
            noise_levels = recorded_noise_levels(trajectory, plan)
            selected_noise.update(noise_levels[index] for index in indices)
            current = (sigmas, times)
            if schedule is not None:
                assert all(torch.equal(a, b) for a, b in zip(schedule, current, strict=True)), (
                    "Temporal normalization requires a shared recorded schedule"
                )
            schedule = current
            recorded_timesteps.append(trajectory["timesteps"][indices])
        scheduler.sigmas, scheduler.timesteps = [value.cuda() for value in schedule]
        if len(selected_noise) != 1:
            raise ValueError("Temporal rectification requires equal noise at selected steps")
        algorithm.prepare_update(
            torch.cat(recorded_timesteps).cuda(),
            scheduler=scheduler,
            noise_level=selected_noise.pop(),
        )
    logs = []
    epoch_records = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for row_index, row in enumerate(rows):
            directory = root / "rollouts" / row["case"] / row["candidate"]
            assert hashlib.sha256(Path(row["view"]).read_bytes()).hexdigest() == row["view_sha256"]
            trajectory = torch.load(directory / "trajectory.pt", weights_only=True)
            behavior_sha256 = trajectory.get("behavior_adapter_sha256")
            if behavior_sha256 is not None:
                assert behavior_sha256 == initial_sha256, "Trajectory behavior checkpoint mismatch"
            noise_levels = recorded_noise_levels(trajectory, plan)
            assert trajectory["condition_case"] == row["case"]
            assert trajectory["initial_adapter_seed"] == plan["adapter_seed"]
            condition = torch.load(
                root / "conditions" / row["case"] / "condition.pt", weights_only=True
            )
            assert condition["source_sha256"] == cases[row["case"]]["sha256"]
            condition = {
                key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in condition.items()
            }
            sigmas, times, absolute_indices = recorded_schedule(trajectory)
            scheduler.sigmas = sigmas.cuda()
            scheduler.timesteps = times.cuda()
            assert all(index < len(trajectory["timesteps"]) for index in indices)
            for index in indices:
                latent = trajectory["observations"][index].cuda()
                action = trajectory["actions"][index].cuda()
                absolute_index = int(absolute_indices[index])
                timestep = scheduler.timesteps[absolute_index]
                velocity = predict_velocity(model, latent, timestep, condition, upstream)
                replay = sde_step_with_logprob(
                    scheduler,
                    velocity,
                    timestep[None],
                    latent,
                    prev_sample=action,
                    noise_level=noise_levels[index],
                    step_index=absolute_index,
                    return_dt=True,
                )
                old = trajectory["old_log_density_mean"][index].cuda()
                parity = float((replay.log_prob.detach() - old).abs().max())
                if epoch == 0:
                    assert parity < 1e-5, f"Pre-update replay mismatch: {parity}"
                with torch.no_grad():
                    reference = sde_step_with_logprob(
                        scheduler,
                        trajectory["old_velocity"][index].cuda(),
                        timestep[None],
                        latent,
                        prev_sample=action,
                        noise_level=noise_levels[index],
                        step_index=absolute_index,
                        return_dt=True,
                    )
                # Center on the fixed behavior density before scaling in float64.
                # This is only a sum/sqrt(D) ablation, not GRPO-Guard centering.
                density_scale = (
                    math.sqrt(action[0].numel()) if normalization == "sqrt_dimension" else 1.0
                )
                policy_log_prob = replay.log_prob
                behavior_log_prob = old
                reference_log_prob = reference.log_prob
                if normalization == "sqrt_dimension":
                    policy_log_prob = (replay.log_prob.double() - old.double()) * density_scale
                    behavior_log_prob = torch.zeros_like(policy_log_prob)
                    reference_log_prob = (
                        reference.log_prob.double() - old.double()
                    ) * density_scale
                signal = FlowSDESignal(
                    name="layers",
                    distribution=(
                        "flow_sde_mean_element_density"
                        if normalization == "mean"
                        else "flow_sde_sqrt_dimension_density"
                    ),
                    log_prob=policy_log_prob,
                    old_log_prob=behavior_log_prob,
                    mask=torch.ones_like(old),
                    ref_log_prob=reference_log_prob,
                    prev_sample_mean=replay.prev_sample_mean,
                    std_dev_t=replay.std_dev_t,
                    dt=replay.sqrt_neg_dt,
                    sigma=replay.sigma,
                    ref_prev_sample_mean=reference.prev_sample_mean,
                )
                loss, metrics = algorithm.compute_loss(
                    AlgorithmInput(
                        signals=TrajectorySignalBatch(
                            segments={"layers": signal},
                            group_ids=groups[row_index : row_index + 1].cuda(),
                        ),
                        advantages=advantages[row_index : row_index + 1].cuda(),
                        model=model,
                        timestep_index=index,
                    )
                )
                assert torch.isfinite(loss)
                (loss / count).backward()
                logs.append(
                    {
                        "epoch": epoch + 1,
                        "case": row["case"],
                        "candidate": row["candidate"],
                        "step": index,
                        "absolute_step": absolute_index,
                        "noise_level": noise_levels[index],
                        "density_scale": density_scale,
                        "policy_log_ratio": float(
                            (policy_log_prob.detach() - behavior_log_prob).mean()
                        ),
                        "reward": row["reward"],
                        "advantage": float(advantages[row_index]),
                        "loss": float(loss.detach()),
                        "preupdate_logprob_error": parity if epoch == 0 else None,
                        "logprob_difference_from_behavior": float(
                            (replay.log_prob.detach() - old).mean()
                        ),
                        "kl_penalty": metrics.kl_penalty,
                        "active_clip_fraction": metrics.update.active_clip_fraction,
                        "temporal_weight": float(algorithm._loss_weight(signal).mean())
                        if rectify
                        else 1.0,
                    }
                )
                print(json.dumps(logs[-1]), flush=True)
                (output / "replay.json").write_text(json.dumps(logs, indent=2))
                del velocity, replay, reference, signal, loss, metrics
        gradients = [value.grad for value in parameters.values() if value.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        norm = torch.nn.utils.clip_grad_norm_(
            list(parameters.values()), config["max_grad_norm"], error_if_nonfinite=True
        )
        assert float(norm) > 0
        optimizer.step()
        epoch_records.append({"epoch": epoch + 1, "gradient_norm_before_clip": float(norm)})
        (output / "epochs.json").write_text(json.dumps(epoch_records, indent=2))
        if epochs > 1:
            torch.save(
                {name: value.detach().cpu().clone() for name, value in parameters.items()},
                output / f"epoch{epoch + 1:02d}_adapter.pt",
            )
    saved = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    assert all(torch.isfinite(value).all() for value in saved.values())
    differences = {
        name: float((value - initial[name]).abs().max()) for name, value in saved.items()
    }
    assert max(differences.values()) > 0
    torch.save(saved, output / "adapter.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    # Test an actual recorded transition after update and after disk reload.
    with torch.no_grad():
        updated_velocity = predict_velocity(model, latent, timestep, condition, upstream)
        for value in parameters.values():
            value.zero_()
        restored = torch.load(output / "adapter.pt", weights_only=True)
        for name, value in parameters.items():
            value.copy_(restored[name])
        reloaded_velocity = predict_velocity(model, latent, timestep, condition, upstream)
    assert torch.equal(updated_velocity, reloaded_velocity)
    record = {
        "optimizer_updates": epochs,
        "optimizer_updates_total": epochs
        + (resumed_optimizer["previous_optimizer_updates"] if resumed_optimizer else 0),
        "resumed_optimizer": resumed_optimizer,
        "optimizer_sha256": hashlib.sha256((output / "optimizer.pt").read_bytes()).hexdigest(),
        "rollout_batches": 1,
        "behavior_policy": "initial_adapter.pt; fixed throughout PPO epochs",
        "gradient_norm_before_clip": float(norm),
        "gradient_tensors": len(gradients),
        "changed_tensors": sum(value > 0 for value in differences.values()),
        "max_parameter_difference": max(differences.values()),
        "reload_velocity_exact": True,
        "velocity_difference_from_initial_rms": float(
            (updated_velocity.cpu() - trajectory["old_velocity"][indices[-1]])
            .square()
            .mean()
            .sqrt()
        ),
        "adapter_sha256": hashlib.sha256((output / "adapter.pt").read_bytes()).hexdigest(),
        "scores_sha256": validation["scores_sha256"],
        "initial_adapter_sha256": initial_sha256,
        "temporal_rectification": rectify,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "density_normalization": normalization,
        "logprob_convention": "Mean Gaussian log density; sqrt_dimension optionally scales the behavior-relative difference by sqrt(D). Gaussian KL and temporal weights are unchanged. Not full GRPO-Guard.",
        "scope": "Real group-reward optimizer updates on one fixed rollout batch. Quality improvement and generalization remain untested.",
        "teacher_initialized": plan.get("teacher_initialized", False),
    }
    (output / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with Path("/tmp/vrl-preservation-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_update(args.root.resolve(), json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
