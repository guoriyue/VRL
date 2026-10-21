"""Locate where a rollout and its replay first stop agreeing.

``full_sequence_probe --check-replay`` answers *whether* step 0 replays within
tolerance. This probe answers *where* a difference starts and *how* it grows:

- the rollout samples ``--rollout-batch`` rows through the production denoise
  loop (``forward_step`` + ``sde_step_with_logprob``) and records every step;
- replay restores rows ``[0, --replay-batch)`` from the exported replay tensors
  and recomputes every step, so the row set, and therefore every kernel's batch
  shape, can differ from the rollout's (the trainer's gen batch != replay batch);
- forward hooks record each module's output for row 0 on both sides; the report
  names the first module, in execution order, whose output differs, and the
  per-block max-abs curve;
- ``|Δlogprob|`` is reported per denoise step, which separates an op that is
  itself imprecise from late small-sigma steps amplifying a tiny difference.

Replay-only switches build the comparisons of
``docs/sprints/planned/SPRINT_rollout_replay_parity_root_cause.md``:
``--replay-compile`` (regional compile on replay only; with
``--rollout-compile`` both sides compile), ``--quantize fp8
--replay-dequantize`` (FP8 rollout, master-weight replay, as the trainer
does), and ``--rollout-sdpa``/``--replay-sdpa`` (attention backend per side).
``--reduced-precision-reduction`` and ``--batch-invariant`` apply to both sides.

    python -m vrl.scripts.generation.parity_divergence_probe --family sana \\
        --path Efficient-Large-Model/Sana_1600M_1024px_diffusers \\
        --dtype fp16 --float32-precision ieee --no-outer-autocast \\
        --rollout-batch 4 --replay-batch 1 --steps 8 --json-out out.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator
from typing import Any

SDPA_BACKENDS = ("default", "flash", "efficient", "cudnn", "math")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--model-preset", default=None)
    parser.add_argument("--prompt", default="a photo of a red fox sitting in fresh snow")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--float32-precision", choices=["ieee", "tf32"], required=True)
    parser.add_argument("--outer-autocast", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--quantize", default=None, choices=["fp8", "nvfp4"])
    parser.add_argument("--rollout-batch", type=int, default=4)
    parser.add_argument("--replay-batch", type=int, default=1)
    parser.add_argument(
        "--replay-dequantize",
        action="store_true",
        help="replay quantized linears through their source-dtype master weight",
    )
    parser.add_argument(
        "--rollout-compile",
        action="store_true",
        help="regional compile before the rollout, so both sides run compiled",
    )
    parser.add_argument("--replay-compile", action="store_true")
    parser.add_argument("--rollout-sdpa", choices=SDPA_BACKENDS, default="default")
    parser.add_argument("--replay-sdpa", choices=SDPA_BACKENDS, default="default")
    parser.add_argument(
        "--reduced-precision-reduction",
        choices=["on", "off"],
        default="off",
        help="cuBLAS fp16/bf16 reduced-precision (split-K) reductions, both sides "
        "(production: off, see apply_float32_precision)",
    )
    parser.add_argument(
        "--batch-invariant",
        action="store_true",
        help="run both sides under thinking-machines batch_invariant_ops (importable)",
    )
    parser.add_argument(
        "--lora-b-std",
        type=float,
        default=0.0,
        help="fill LoRA B with N(0, std) so the adapter path contributes, as after a "
        "first update (a fresh adapter's zero B hides every LoRA GEMM difference)",
    )
    parser.add_argument("--json-out", default=None)
    return parser


def _sdpa_context(name: str) -> contextlib.AbstractContextManager[Any]:
    if name == "default":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }[name]
    return sdpa_kernel(backend)


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


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.replay_batch <= args.rollout_batch:
        raise SystemExit("--replay-batch must be in [1, --rollout-batch]")

    import torch

    from vrl.generation.types import DenoiseRequest
    from vrl.math.denoise.flow_matching import sde_step_with_logprob
    from vrl.models.families.registry import get_model_family_entry
    from vrl.nn.optimization.regional_compile import compile_repeated_blocks
    from vrl.scripts.generation.full_sequence_probe import resolve_probe_model_build

    invariant: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()
    if args.batch_invariant:
        from batch_invariant_ops import set_batch_invariant_mode

        invariant = set_batch_invariant_mode(True)

    entry = get_model_family_entry(args.family)
    device = torch.device("cuda")
    build = resolve_probe_model_build(args, entry, device)
    if args.replay_dequantize:
        from dataclasses import replace

        # Keep the quantized linears' source masters: the probe replays through
        # them, as the trainer replays through its own unquantized weights.
        build.rollout = replace(build.require_rollout(), base_weight_sync=True)
    bundle = entry.build_rollout(build, expected_identity=None)
    model = bundle.model
    transformer = model.transformer
    if args.lora_b_std:
        lora_generator = torch.Generator(device="cpu").manual_seed(args.seed)
        if not any("lora_B" in name for name, _ in transformer.named_parameters()):
            raise SystemExit("--lora-b-std needs a LoRA model: use a model preset with use_lora")
        with torch.no_grad():
            for name, parameter in transformer.named_parameters():
                if "lora_B" in name:
                    noise = (
                        torch.randn(parameter.shape, generator=lora_generator) * args.lora_b_std
                    )
                    parameter.copy_(noise.to(parameter.dtype))
    # After the build: building applies the production matmul policy (reductions
    # off), and this switch exists to measure the alternative.
    rpr = args.reduced_precision_reduction == "on"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = rpr
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = rpr

    encode_kwargs: dict[str, Any] = {"guidance_scale": args.guidance_scale}
    if args.max_sequence_length is not None:
        encode_kwargs["max_sequence_length"] = args.max_sequence_length
    prompts = [args.prompt] * args.rollout_batch
    negatives = [args.negative_prompt] * args.rollout_batch if args.guidance_scale > 1.0 else None
    encoded = model.encode_prompt(prompts, negatives, **encode_kwargs)
    request = DenoiseRequest(
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        frame_count=args.frames,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    state = model.prepare_sampling(request, encoded)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # Per-module hooks are closures dynamo guards on, so every hooked block
    # would recompile until the recompile limit drops it to eager: a compiled
    # replay is compared per step only.
    compiled = args.rollout_compile or args.replay_compile
    recorder = _Recorder(transformer, rows=args.rollout_batch, hooks=not compiled)
    if args.rollout_compile:
        compile_repeated_blocks(transformer, mode="default")
    steps: list[dict[str, Any]] = []
    step0_calls: list[tuple[str, Any]] = []
    with torch.inference_mode(), invariant:
        for step_idx in range(args.steps):
            # Module outputs are compared at step 0 only; later steps would
            # hold every module's activation for every step.
            recorder.rows = args.rollout_batch
            recorder.enabled = step_idx == 0
            with _sdpa_context(args.rollout_sdpa):
                out = model.forward_step(state, step_idx)
            recorder.enabled = False
            if step_idx == 0:
                step0_calls = recorder.take()
            timestep = state.timesteps[step_idx]
            sde = sde_step_with_logprob(
                state.scheduler,
                out["noise_pred"].float(),
                timestep.unsqueeze(0),
                state.latents.float(),
                generator=generator,
                step_index=step_idx,
            )
            steps.append(
                {
                    "latents": state.latents.detach().clone(),
                    "noise_pred": out["noise_pred"].detach().float().clone(),
                    "prev_sample": sde.prev_sample.detach().clone(),
                    "log_prob": sde.log_prob.detach().clone(),
                    "timestep": timestep.detach().clone(),
                }
            )
            state.latents = sde.prev_sample

        replay_tensors = dict(model.export_replay_tensors(state))
        batch_context = model.export_batch_context(state)
        rows = args.replay_batch
        sliced = {
            key: value[:rows] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
            for key, value in replay_tensors.items()
        }
        if args.replay_compile and not args.rollout_compile:
            compile_repeated_blocks(transformer, mode="default")
        dequant: contextlib.AbstractContextManager[Any] = (
            _dequantized(transformer) if args.replay_dequantize else contextlib.nullcontext()
        )
        report_steps = []
        with dequant:
            for step_idx, record in enumerate(steps):
                sliced["timesteps"] = record["timestep"].expand(rows).clone()
                restored = model.restore_eval_state(
                    sliced, batch_context, record["latents"][:rows], step_idx
                )
                recorder.rows = rows
                recorder.enabled = step_idx == 0
                with _sdpa_context(args.replay_sdpa):
                    # The trainer's own index mapping: most families rebuild a
                    # one-step state (index 0), Cosmos keeps the full table.
                    out = model.forward_step(restored, model._replay_forward_step_index(step_idx))
                recorder.enabled = False
                replay_calls = recorder.take()
                sde = sde_step_with_logprob(
                    state.scheduler,
                    out["noise_pred"].float(),
                    record["timestep"].unsqueeze(0),
                    record["latents"][:rows].float(),
                    prev_sample=record["prev_sample"][:rows].float(),
                    step_index=step_idx,
                )
                noise_err = (out["noise_pred"].float() - record["noise_pred"][:rows]).abs().max()
                logprob_diff = (sde.log_prob - record["log_prob"][:rows]).abs()
                logprob_err = logprob_diff.max()
                entry_report = {
                    "step": step_idx,
                    "noise_pred_max_abs": noise_err.item(),
                    "logprob_max_abs": logprob_err.item(),
                    "logprob_mean_abs": logprob_diff.mean().item(),
                    "bitwise_logprob": bool(torch.equal(sde.log_prob, record["log_prob"][:rows])),
                }
                if step_idx == 0:
                    entry_report.update(
                        _compare_calls(step0_calls, replay_calls, recorder.block_names)
                    )
                report_steps.append(entry_report)
                first = entry_report.get("first_divergent_module")
                print(
                    f"[divergence] step {step_idx}: |Δnoise|={noise_err.item():.3e} "
                    f"|Δlogprob|={logprob_err.item():.3e}"
                    + (f" first={first['module']} ({first['max_abs']:.3e})" if first else "")
                )
    recorder.remove()

    report = {
        "arguments": vars(args),
        "max_logprob_abs": max(step["logprob_max_abs"] for step in report_steps),
        "mean_logprob_abs": sum(step["logprob_mean_abs"] for step in report_steps)
        / len(report_steps),
        "steps": report_steps,
    }
    print(f"[divergence] max |Δlogprob| over {args.steps} steps = {report['max_logprob_abs']:.3e}")
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
