"""Audit an initial layered GRPO step separately from generated-image quality."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

import torch

from vrl.math.denoise.flow_matching import compute_kl_divergence, sde_step_with_logprob
from vrl.scripts.generation.qwen_layered_rl_probe import (
    load_transformer,
    load_upstream,
    predict_velocity,
)


def audit_optimizer(root):
    initial = torch.load(root / "initial_adapter.pt", map_location="cpu", weights_only=True)
    updated = torch.load(root / "update01/adapter.pt", map_location="cpu", weights_only=True)
    state = torch.load(root / "update01/optimizer.pt", map_location="cpu", weights_only=True)
    assert initial.keys() == updated.keys()
    assert len(state["param_groups"]) == 1
    group = state["param_groups"][0]
    assert len(group["params"]) == len(initial) and group["weight_decay"] == 0
    gradient_dot_step = 0.0
    active_gradients = []
    parameters_without_optimizer_state = []
    step_squares = 0.0
    # The update runner saves tensors in the same order used to construct Adam.
    # Imported initial checkpoints may serialize those named tensors differently.
    for name, parameter_id in zip(updated, group["params"], strict=True):
        if parameter_id not in state["state"]:
            assert torch.equal(initial[name], updated[name]), (
                f"Parameter changed without optimizer state: {name}"
            )
            parameters_without_optimizer_state.append(name)
            continue
        moment = state["state"][parameter_id]
        assert float(moment["step"]) == 1, "This diagnostic assumes a single AdamW step"
        gradient = moment["exp_avg"].double() / (1 - group["betas"][0])
        change = updated[name].double() - initial[name].double()
        gradient_dot_step += float((gradient * change).sum())
        step_squares += float(change.square().sum())
        if "lora_B" in name:
            active_gradients.append(gradient.abs().float().flatten())
    assert active_gradients, "No LoRA B optimizer state to audit"
    active = torch.cat(active_gradients)
    record = {
        "gradient_dot_parameter_step": gradient_dot_step,
        "descent_direction_for_recorded_surrogate": gradient_dot_step < 0,
        "parameter_step_l2": step_squares**0.5,
        "adam_epsilon": group["eps"],
        "parameters_without_optimizer_state": parameters_without_optimizer_state,
        "lora_B_gradient_abs_median": float(active.median()),
        "lora_B_gradient_below_adam_epsilon_fraction": float(
            (active < group["eps"]).float().mean()
        ),
        "scope": "First-order optimizer direction only; not actual post-update likelihood or image improvement.",
    }
    (root / "update01/optimizer_audit.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def audit_likelihood(root):
    from diffusers import FlowMatchEulerDiscreteScheduler

    plan = json.loads((root / "plan.json").read_text())
    config = json.loads((root / "update01/config.json").read_text())
    result = json.loads((root / "update01/result.json").read_text())
    checkpoint = root / "update01/adapter.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result["adapter_sha256"]
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    upstream = load_upstream(upstream_path)
    model = load_transformer(plan)
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    updated = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert updated.keys() == parameters.keys()
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(updated[name])
    model.requires_grad_(False)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        plan["base_path"], subfolder="scheduler", local_files_only=True
    )
    replay_rows = json.loads((root / "update01/replay.json").read_text())
    rows = []
    for case in plan["cases"]:
        if case["split"] != "train":
            continue
        condition = torch.load(
            root / "conditions" / case["name"] / "condition.pt", weights_only=True
        )
        assert condition["source_sha256"] == case["sha256"]
        condition = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in condition.items()
        }
        for candidate in sorted((root / "rollouts" / case["name"]).glob("candidate*")):
            trajectory = torch.load(candidate / "trajectory.pt", weights_only=True)
            assert trajectory["noise_level"] == plan["noise_level"]
            scheduler.sigmas = trajectory["sigmas"].cuda()
            for index in config["timesteps"]:
                original = next(
                    row
                    for row in replay_rows
                    if row["case"] == case["name"]
                    and row["candidate"] == candidate.name
                    and row["step"] == index
                )
                temporal_weight = (
                    original["temporal_weight"]
                    if config.get("temporal_rectification", False)
                    else 1.0
                )
                assert math.isfinite(temporal_weight) and temporal_weight > 0
                latent = trajectory["observations"][index].cuda()
                action = trajectory["actions"][index].cuda()
                timestep = trajectory["timesteps"][index].cuda()
                with torch.no_grad():
                    velocity = predict_velocity(model, latent, timestep, condition, upstream)
                    common = dict(
                        prev_sample=action,
                        noise_level=plan["noise_level"],
                        step_index=index,
                        math_dtype=torch.float64,
                        return_dt=True,
                    )
                    before = sde_step_with_logprob(
                        scheduler,
                        trajectory["old_velocity"][index].cuda(),
                        timestep[None],
                        latent,
                        **common,
                    )
                    after = sde_step_with_logprob(
                        scheduler, velocity, timestep[None], latent, **common
                    )
                    change = float((after.log_prob - before.log_prob).mean())
                    ratio = (after.log_prob - before.log_prob).exp()
                    advantage = original["advantage"]
                    clipped_ratio = ratio.clamp(1 - config["clip_ratio"], 1 + config["clip_ratio"])
                    policy_loss_change = float(
                        torch.maximum(-advantage * ratio, -advantage * clipped_ratio).mean()
                        + advantage
                    )
                    # Flash-GRPO weights the policy surrogate, not the KL term.
                    # Use the actual weight recorded by this training run.
                    policy_loss_change *= temporal_weight
                    kl = float(
                        compute_kl_divergence(
                            after.prev_sample_mean,
                            before.prev_sample_mean,
                            after.std_dev_t,
                            sqrt_neg_dt=after.sqrt_neg_dt,
                        ).mean()
                    )
                row = {
                    "case": case["name"],
                    "candidate": candidate.name,
                    "step": index,
                    "advantage": original["advantage"],
                    "temporal_weight": temporal_weight,
                    "mean_element_log_density_change": change,
                    "advantage_weighted_change": original["advantage"] * change,
                    "mean_element_gaussian_kl": kl,
                    "clipped_policy_loss_change": policy_loss_change,
                    "regularized_loss_change": policy_loss_change + config["kl_coef"] * kl,
                    "ratio_clip_active": bool((ratio != clipped_ratio).any()),
                }
                rows.append(row)
                (root / "update01/policy_audit_partial.json").write_text(
                    json.dumps(rows, indent=2)
                )
                print(json.dumps(row), flush=True)
    record = {
        "rows": rows,
        "mean_advantage_weighted_log_density_change": sum(
            row["advantage_weighted_change"] for row in rows
        )
        / len(rows),
        "mean_clipped_policy_loss_change": sum(row["clipped_policy_loss_change"] for row in rows)
        / len(rows),
        "mean_gaussian_kl": sum(row["mean_element_gaussian_kl"] for row in rows) / len(rows),
        "mean_regularized_loss_change": sum(row["regularized_loss_change"] for row in rows)
        / len(rows),
        "loss_interpretation": "Negative change is improvement relative to the initial policy. Mirrors the clipped surrogate with recorded temporal policy weights plus unweighted exact mean-element Gaussian KL, recomputed in float64; not the training kernel's float32 rounding.",
        "audit_math": "Float64 Gaussian density calculation from actual stored float32 SDE states/actions; model remains bf16. Baseline recorded velocities passed exact pre-update replay checks during training.",
        "scope": "Recorded-trajectory optimization diagnostic only, not new-rollout reward or held-out visual improvement.",
    }
    (root / "update01/policy_audit.json").write_text(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    audit_optimizer(root)
    if not args.cpu_only:
        with Path("/tmp/vrl-preservation-gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            audit_likelihood(root)


if __name__ == "__main__":
    main()
