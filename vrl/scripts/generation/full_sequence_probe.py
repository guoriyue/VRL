"""Roll out one generation batch of a recipe and replay it the way the trainer does.

The per-family verification tool for model landings and the locator for
rollout/replay drift (no Ray, no trainer). Everything that defines the run
comes from the recipe, loaded with the same ``--config`` + overrides grammar
as ``vrl-train``: model family/path/preset, precision and rollout
quantization, ``model.torch_compile``, sampling, ``rollout.sde`` /
``rollout.denoise_mode``, the generation batch
(``rollout.samples_per_generation_batch``), the replay batch
(``actor.training_microbatch_size``) and the replay gate
(``trainer.replay_parity.max_abs_logprob_diff``).

    python -m vrl.scripts.generation.full_sequence_probe \\
        --config experiment/sana/online_grpo_aesthetic_fullparam \\
        sampling.num_steps=8 --artifact-dir outputs/probes/sana

The probe

- builds the rollout bundle through the registry and prepares the batch with
  the family's own executor hooks, then samples it through the production
  ``run_denoise_loop``;
- decodes and fails on non-finite or ~zero-variance output;
- replays rows ``[0, replay batch)`` at every step from the exported replay
  tensors; a quantized rollout replays through its source master weights, as
  the trainer replays its unquantized policy;
- names the first module whose step-0 output differs between the two sides
  (forward hooks on row 0; skipped under ``torch.compile``, whose guards would
  recompile every hooked block);
- reports ``|Δlogprob|`` per step, which separates an op that is itself
  imprecise from late small-sigma steps amplifying a tiny difference, and fails
  when the maximum exceeds the replay gate.

Inductor switches stay environment variables (``TORCHINDUCTOR_DETERMINISTIC=1``,
``TORCHINDUCTOR_SPLIT_REDUCTIONS=0``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Bundled config name or absolute YAML path.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Ordered preset overlays (+model=...) and OmegaConf dotlist overrides "
            "(sampling.num_steps=8 actor.training_microbatch_size=1)."
        ),
    )
    parser.add_argument("--prompt", default="a photo of a red fox sitting in fresh snow")
    parser.add_argument(
        "--reference-image",
        default=None,
        help="conditioning image for a reference-conditioned family",
    )
    parser.add_argument(
        "--lora-b-std",
        type=float,
        default=0.0,
        help="fill LoRA B with N(0, std) so the adapter path contributes, as after a "
        "first update (a fresh adapter's zero B hides every LoRA GEMM difference)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="new directory for the decoded image/video and result.json",
    )
    return parser


@contextlib.contextmanager
def _dequantized(root: Any) -> Iterator[None]:
    """Route every quantized linear through ``F.linear`` on its master weight."""

    import torch.nn.functional as F

    from vrl.nn.quantization.base import QuantizedLinear

    patched = [m for m in root.modules() if isinstance(m, QuantizedLinear)]
    for module in patched:
        module.forward = lambda x, m=module: F.linear(  # type: ignore[method-assign]
            x, m.weight.to(x.dtype), None if m.bias is None else m.bias.to(x.dtype)
        )
    try:
        yield
    finally:
        for module in patched:
            del module.forward


class _Recorder:
    """Row-0 outputs of every hooked module, in call order, for one forward."""

    def __init__(self, root: Any, *, rows: int, hooks: bool) -> None:
        from vrl.nn.optimization.regional_compile import repeated_block_lists

        self.rows = rows
        self.calls: list[tuple[str, Any]] = []
        self.enabled = False
        blocks = {
            id(block) for block_list in repeated_block_lists(root).values() for block in block_list
        }
        self.block_names: set[str] = set()
        self._handles = []
        for name, module in root.named_modules() if hooks else ():
            is_block = id(module) in blocks
            if is_block:
                self.block_names.add(name)
            if is_block or not any(True for _ in module.children()):
                self._handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        import torch

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.enabled:
                return
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            picked = []
            for tensor in tensors:
                if (
                    isinstance(tensor, torch.Tensor)
                    and tensor.ndim > 0
                    and tensor.is_floating_point()
                ):
                    picked.append(self._row0(tensor).detach().float().clone())
            if picked:
                self.calls.append((name, picked))

        return hook

    def _row0(self, tensor: Any) -> Any:
        # Batched CFG packs [uncond rows, cond rows]: the conditional row 0 sits
        # at index ``rows``. Unbatched or non-CFG calls see row 0 at index 0.
        if tensor.shape[0] == 2 * self.rows:
            return tensor[self.rows]
        return tensor[0]

    def take(self) -> list[tuple[str, Any]]:
        calls, self.calls = self.calls, []
        return calls

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def _compare_calls(
    rollout: list[tuple[str, Any]], replay: list[tuple[str, Any]], block_names: set[str]
) -> dict[str, Any]:
    if [name for name, _ in rollout] != [name for name, _ in replay]:
        return {"error": "module call order differs between rollout and replay"}
    first = None
    blocks = []
    for (name, a_list), (_, b_list) in zip(rollout, replay, strict=True):
        diff = max(
            (a - b).abs().max().item() if a.shape == b.shape else float("inf")
            for a, b in zip(a_list, b_list, strict=True)
        )
        if first is None and diff > 0:
            first = {"module": name, "max_abs": diff}
        if name in block_names:
            blocks.append({"block": name, "max_abs": diff})
    return {"first_divergent_module": first, "blocks": blocks}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=False)

    from dataclasses import replace

    import torch

    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.generation.steps.denoise.config import DenoiseLoopConfig
    from vrl.generation.steps.denoise.loop import run_denoise_loop
    from vrl.generation.types import GenerationInput, GenerationRequest
    from vrl.math.denoise.flow_matching import sde_step_with_logprob
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.families.semantics import task_type_for
    from vrl.rollouts.collector import RolloutCollectorConfig
    from vrl.trainers.core.types import ReplayParityConfig
    from vrl.trainers.online.config import OnlineBatchConfig
    from vrl.utils.config import import_from_path

    root = parse_config(load_config(args.config, overrides=args.overrides))
    entry = get_model_family_entry(str(root.model.family))
    batch_config = OnlineBatchConfig.from_root(root)
    replay_parity = (
        root.trainer.replay_parity if root.trainer is not None else None
    ) or ReplayParityConfig()
    gate = float(replay_parity.max_abs_logprob_diff)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build = entry.resolve_model_build(
        root, device, precision=PrecisionPolicy.from_section(root.precision)
    )
    # Keep a quantized rollout's source masters (the flag only gates dropping
    # them): replay runs through them, as the trainer replays its own
    # unquantized weights.
    build.rollout = replace(build.require_rollout(), base_weight_sync=True)
    quantization = build.precision.quantization
    compiled = build.torch_compile is not None
    print(f"[probe] building {entry.family} bundle from {build.model_name_or_path} ...")
    model = entry.build_rollout(build, expected_identity=None).model
    transformer = model.transformer
    if args.lora_b_std:
        if not any("lora_B" in name for name, _ in transformer.named_parameters()):
            raise SystemExit("[probe] --lora-b-std needs a LoRA model (model.use_lora=true)")
        lora_generator = torch.Generator(device="cpu").manual_seed(0)
        with torch.no_grad():
            for name, parameter in transformer.named_parameters():
                if "lora_B" in name:
                    noise = torch.randn(parameter.shape, generator=lora_generator)
                    parameter.copy_((noise * args.lora_b_std).to(parameter.dtype))

    # The request the collector would send for one prompt; the executor's own
    # hooks resolve sampling defaults, encode, expand and prepare it, so the
    # first generation batch is exactly what a rollout worker would run.
    collector = RolloutCollectorConfig.from_root(root)
    request = GenerationRequest(
        request_id="full-sequence-probe",
        family=entry.family,
        task=entry.task,
        inputs=[
            GenerationInput(
                prompt=args.prompt,
                task_type=task_type_for(entry.task),
                reference_images=[args.reference_image] if args.reference_image else [],
            )
        ],
        samples_per_prompt=batch_config.n_samples_per_prompt,
        sampling=collector.request_sampling,
        samples_per_generation_batch=collector.samples_per_generation_batch,
        denoise=collector.denoise,
        sde_window_seed=0,
    )
    batch = request.sample_batches()[0]
    rollout_rows = batch.sample_count
    replay_width = batch_config.training_microbatch_size
    replay_rows = (
        request.samples_per_prompt
        if replay_width <= 0
        else min(request.samples_per_prompt, replay_width)
    )
    if replay_rows > rollout_rows:
        raise SystemExit(
            f"[probe] replay batch {replay_rows} exceeds the generation batch "
            f"{rollout_rows}; the probe replays rows of one generation batch"
        )
    executor = import_from_path(entry.executor_cls)(model, **entry.executor_kwargs(root))
    params = executor.parse_sampling_params(request)
    loop_config = DenoiseLoopConfig(
        sample_start=batch.sample_start,
        sample_count=rollout_rows,
        seed=params.model_request.seed,
        sde=params.sde,
        sde_window=params.sde_window,
        denoise_mode=params.denoise_mode,
        teacache=params.teacache,
    )
    with torch.no_grad():
        encoded = executor.encode_prompt_for_batch(
            generation_request=request, params=params, batch=batch
        )
        state = executor.prepare_denoise_state(
            request=params.model_request,
            encoded=executor.expand_conditioning_to_batch(
                encoded=encoded, generation_request=request, batch=batch
            ),
            config=loop_config,
            prepare_kwargs=executor.build_prepare_kwargs(
                encoded=encoded, generation_request=request, batch=batch
            ),
        )

    # Per-module hooks are closures dynamo guards on, so every hooked block
    # would recompile until the recompile limit drops it to eager: a compiled
    # run is compared per step only.
    recorder = _Recorder(transformer, rows=rollout_rows, hooks=not compiled)
    rollout_noise: dict[int, Any] = {}

    def forward_step(step_state: Any, step_idx: int) -> dict[str, Any]:
        # Module outputs are compared at step 0 only; later steps would hold
        # every module's activation for every step.
        recorder.enabled = step_idx == 0
        try:
            out = model.forward_step(step_state, step_idx)
        finally:
            recorder.enabled = False
        rollout_noise[step_idx] = out["noise_pred"].detach().float().clone()
        return out

    result = run_denoise_loop(
        model=SimpleNamespace(forward_step=forward_step), state=state, config=loop_config
    )
    step0_calls = recorder.take()
    if not bool(torch.isfinite(result.latents).all()) or not bool(
        torch.isfinite(result.log_probs).all()
    ):
        raise SystemExit("[probe] FAIL: nonfinite latents/log-probs in the rollout")

    with torch.no_grad():
        image = model.decode_latents(result.state.latents)
    finite = bool(torch.isfinite(image).all())
    image_std = image.float().std().item()
    print(
        f"[probe] decoded output shape={tuple(image.shape)} "
        f"range=[{image.min().item():.3f}, {image.max().item():.3f}] "
        f"std={image_std:.4f} finite={finite}",
    )
    if not finite or image_std < 1e-3:
        raise SystemExit("[probe] FAIL: output is not finite or has ~zero variance")

    replay_tensors = {
        key: value[:replay_rows] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
        for key, value in model.export_replay_tensors(result.state).items()
    }
    batch_context = model.export_batch_context(result.state)
    scheduler = result.state.scheduler
    report_steps = []
    dequant: contextlib.AbstractContextManager[Any] = (
        _dequantized(transformer) if quantization is not None else contextlib.nullcontext()
    )
    with torch.no_grad(), dequant:
        for step_idx in range(result.log_probs.shape[1]):
            observation = result.observations[:replay_rows, step_idx]
            replay_tensors["timesteps"] = result.timesteps[:replay_rows, step_idx].clone()
            restored = model.restore_eval_state(
                replay_tensors, batch_context, observation, step_idx
            )
            recorder.rows = replay_rows
            recorder.enabled = step_idx == 0
            # The trainer's own index mapping: most families rebuild a one-step
            # state (index 0), Cosmos keeps the full table.
            out = model.forward_step(restored, model._replay_forward_step_index(step_idx))
            recorder.enabled = False
            sde = sde_step_with_logprob(
                scheduler,
                out["noise_pred"].float(),
                result.timesteps[0, step_idx].unsqueeze(0),
                observation.float(),
                prev_sample=result.actions[:replay_rows, step_idx].float(),
                noise_level=loop_config.sde.noise_level,
                sde_type=loop_config.sde.sde_type,
                step_index=step_idx,
            )
            rollout_log_prob = result.log_probs[:replay_rows, step_idx]
            logprob_diff = (sde.log_prob - rollout_log_prob).abs()
            entry_report: dict[str, Any] = {
                "step": step_idx,
                "rollout_logprob_mean": rollout_log_prob.mean().item(),
                "logprob_max_abs": logprob_diff.max().item(),
                "logprob_mean_abs": logprob_diff.mean().item(),
                "bitwise_logprob": bool(torch.equal(sde.log_prob, rollout_log_prob)),
            }
            if step_idx in rollout_noise:
                entry_report["noise_pred_max_abs"] = (
                    (out["noise_pred"].float() - rollout_noise[step_idx][:replay_rows])
                    .abs()
                    .max()
                    .item()
                )
            if step_idx == 0:
                entry_report.update(
                    _compare_calls(step0_calls, recorder.take(), recorder.block_names)
                )
            report_steps.append(entry_report)
            first = entry_report.get("first_divergent_module")
            print(
                f"[probe] step {step_idx}: |Δlogprob|={entry_report['logprob_max_abs']:.3e}"
                + (f" first={first['module']} ({first['max_abs']:.3e})" if first else "")
            )
    recorder.remove()

    max_logprob_abs = max(step["logprob_max_abs"] for step in report_steps)
    passed = max_logprob_abs <= gate
    print(f"[probe] max |Δlogprob| = {max_logprob_abs:.3e} (gate {gate:.3e})")
    if artifact_dir is not None:
        from vrl.utils.media import write_mp4, write_png

        if image.ndim == 5:
            write_mp4(
                image[0].float().cpu(),
                artifact_dir / "video.mp4",
                fps=params.model_request.fps or 16,
            )
        else:
            write_png(image[0].float().cpu(), artifact_dir / "image.png")
        report = {
            "passed": passed,
            "config": args.config,
            "overrides": args.overrides,
            "family": entry.family,
            "prompt": args.prompt,
            "reference_image": args.reference_image,
            "lora_b_std": args.lora_b_std,
            "rollout_batch": rollout_rows,
            "replay_batch": replay_rows,
            "compiled": compiled,
            "quantization": None if quantization is None else quantization.format,
            "max_abs_logprob_diff_gate": gate,
            "max_logprob_abs": max_logprob_abs,
            "mean_logprob_abs": sum(step["logprob_mean_abs"] for step in report_steps)
            / len(report_steps),
            "steps": report_steps,
            "decoded_shape": list(image.shape),
            "decoded_std": image_std,
            "seconds_including_load": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        (artifact_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("[probe] FAIL: replay |Δlogprob| exceeds the replay gate")
    print(f"[probe] PASS: {entry.family} rollout and replay verified")


if __name__ == "__main__":
    main()
