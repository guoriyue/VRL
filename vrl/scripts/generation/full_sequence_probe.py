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
  recompile every hooked block, and under ``--no-module-hooks``, for a packed
  sequence whose per-module row-0 outputs would not fit next to the model);
- reports ``|Δlogprob|`` per step, which separates an op that is itself
  imprecise from late small-sigma steps amplifying a tiny difference, and fails
  when the maximum over the steps the trainer puts loss on exceeds the replay
  gate (``actor.timestep_selection: sde_window`` trains only the stochastic
  window; the deterministic steps after it are reported, not gated);
- recomputes every step's transition density in float64 from the replay's
  noise prediction, and tests the rollout's standardized residuals
  ``(x_{t-1} - mean) / scale`` of every SDE-window step against N(0, 1): the
  sampler must draw from the density the log-prob describes;
- with ``--block-audit``, runs the first repeated block on its replay step-0
  inputs once as the policy runs it and once in float64 (float64 attention
  through a reference ``scaled_dot_product_attention``), and compares the
  outputs and the gradient of the trainable tensors against a float64 noise
  floor; every policy attention call is also compared with float64 attention
  over its own inputs, which fails a wrong kernel apart from how sensitive the
  attention is to the policy dtype's rounding of q and k.

Inductor switches stay environment variables (``TORCHINDUCTOR_DETERMINISTIC=1``,
``TORCHINDUCTOR_SPLIT_REDUCTIONS=0``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# A correct low-precision attention kernel differs from exact softmax attention
# over its own (already rounded) inputs by its output rounding: ~2e-3 relative
# L2 in bf16 (unit roundoff 2^-8 = 3.9e-3). Five times the unit roundoff leaves
# that margin and still fails a kernel that computes something else.
ATTENTION_KERNEL_MAX_REL_L2 = 2e-2


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
    parser.add_argument(
        "--no-module-hooks",
        action="store_true",
        help="compare the two sides per step only, without holding every module's "
        "step-0 row-0 output (a long packed sequence times every block does not fit)",
    )
    parser.add_argument(
        "--block-audit",
        action="store_true",
        help="compare the first repeated block's forward and trainable gradient with a "
        "float64 reference (holds a float64 copy of the block and its activations; "
        "run it with a replay batch of 1)",
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


def _residual_stats(reference: Any, action: Any, *, sde_type: str) -> dict[str, Any]:
    """Moments of the standardized residual of one sampled step against N(0, 1).

    ``reference`` is the float64 transition of the rollout's own noise
    prediction, so ``action`` was drawn from exactly this Gaussian. Over ``n``
    iid N(0, 1) draws the sample mean has standard deviation ``1/sqrt(n)`` and
    the mean square ``sqrt(2/n)``; both bounds sit at six of those.
    """

    scale = reference.std_dev_t
    if sde_type != "cps":
        scale = scale * reference.sqrt_neg_dt
    if not bool((scale > 0).all()):
        return {"skipped": "zero noise scale"}
    z = (action.double() - reference.prev_sample_mean) / scale
    n = z.numel()
    mean = z.mean().item()
    mean_square = z.square().mean().item()
    mean_bound = 6.0 / n**0.5
    mean_square_bound = 6.0 * (2.0 / n) ** 0.5
    return {
        "n": n,
        "mean": mean,
        "mean_square": mean_square,
        "mean_bound": mean_bound,
        "mean_square_bound": mean_square_bound,
        "passed": abs(mean) <= mean_bound and abs(mean_square - 1.0) <= mean_square_bound,
    }


def _map_tensors(value: Any, fn: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(item, fn) for item in value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    return value


def _float64(tensor: Any) -> Any:
    import torch

    if tensor.is_complex():
        return tensor.to(torch.complex128)
    return tensor.to(torch.float64) if tensor.is_floating_point() else tensor


def _exact_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Any,
    scale: float | None,
    *,
    query_chunk: int,
    checkpointed: bool,
) -> tuple[Any, Any]:
    """Float64 softmax attention over query chunks; returns ``(output, max |logit|)``.

    Query chunks keep the score matrix of a long packed sequence to
    ``heads x query_chunk x keys``; ``checkpointed`` makes backward recompute
    each chunk instead of holding every chunk's probabilities.
    """

    import torch
    from torch.utils.checkpoint import checkpoint

    def chunk_attention(q: Any, k: Any, v: Any, mask: Any, scale: float) -> tuple[Any, Any]:
        scores = (q @ k.transpose(-2, -1)) * scale
        logit_max = scores.detach().abs().amax()
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask.to(scores.dtype)
        return scores.softmax(dim=-1) @ v, logit_max

    q, k, v = (_float64(t) for t in (query, key, value))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    chunks, logit_max = [], torch.zeros((), dtype=torch.float64, device=q.device)
    for start in range(0, q.shape[-2], query_chunk):
        rows = slice(start, start + query_chunk)
        mask = attn_mask
        if mask is not None and mask.ndim >= 2 and mask.shape[-2] > 1:
            mask = mask[..., rows, :]
        if checkpointed:
            out, chunk_max = checkpoint(
                chunk_attention, q[..., rows, :], k, v, mask, scale, use_reentrant=False
            )
        else:
            out, chunk_max = chunk_attention(q[..., rows, :], k, v, mask, scale)
        chunks.append(out)
        logit_max = torch.maximum(logit_max, chunk_max)
    return torch.cat(chunks, dim=-2), logit_max


def _sdpa_arguments(dropout_p: float, is_causal: bool, enable_gqa: bool) -> None:
    if dropout_p or is_causal or enable_gqa:
        raise NotImplementedError("reference attention covers non-causal, dropout-free SDPA")


@contextlib.contextmanager
def _reference_attention(*, query_chunk: int = 512) -> Iterator[list[float]]:
    """Route ``F.scaled_dot_product_attention`` through float64 softmax attention.

    Yields the max |logit| of every call, so a block whose attention bypasses
    SDPA (a flash library backend) is caught instead of compared against
    itself.
    """

    import torch.nn.functional as F

    native = F.scaled_dot_product_attention
    logit_maxima: list[float] = []

    def reference(
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Any:
        _sdpa_arguments(dropout_p, is_causal, enable_gqa)
        out, logit_max = _exact_attention(
            query, key, value, attn_mask, scale, query_chunk=query_chunk, checkpointed=True
        )
        logit_maxima.append(logit_max.item())
        return out.to(query.dtype)

    F.scaled_dot_product_attention = reference
    try:
        yield logit_maxima
    finally:
        F.scaled_dot_product_attention = native


@contextlib.contextmanager
def _audited_attention(*, query_chunk: int = 512) -> Iterator[list[float]]:
    """Run native SDPA and record its error against float64 on the same inputs.

    The policy's kernel receives inputs already rounded to the policy dtype;
    comparing its output with exact softmax attention over exactly those
    inputs measures the kernel alone, apart from how sensitive the attention
    is to that rounding. Yields the relative L2 error of every call.
    """

    import torch
    import torch.nn.functional as F

    native = F.scaled_dot_product_attention
    errors: list[float] = []

    def audited(
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Any:
        _sdpa_arguments(dropout_p, is_causal, enable_gqa)
        out = native(query, key, value, attn_mask=attn_mask, scale=scale)
        with torch.no_grad():
            exact, _ = _exact_attention(
                query, key, value, attn_mask, scale, query_chunk=query_chunk, checkpointed=False
            )
            errors.append(((out.detach().double() - exact).norm() / exact.norm()).item())
        return out

    F.scaled_dot_product_attention = audited
    try:
        yield errors
    finally:
        F.scaled_dot_product_attention = native


@contextlib.contextmanager
def _unstreamed(block: Any) -> Iterator[None]:
    """Let ``functional_call`` run a host-streamed block over the state it swaps in.

    A streamed block's ``__call__`` (``vrl.nn.offload.block_streaming``) swaps
    in its own device copies unless ``_streaming_active`` says it is already
    inside its streamed call.
    """

    if getattr(block, "_streaming", None) is None:
        yield
        return
    block._streaming_active = True
    try:
        yield
    finally:
        block._streaming_active = False


class _BlockInputs:
    """The inputs of the first repeated block's call while enabled."""

    def __init__(self, root: Any) -> None:
        from vrl.nn.optimization.regional_compile import repeated_block_lists

        lists = repeated_block_lists(root)
        if not lists:
            raise SystemExit(
                f"[probe] --block-audit: {type(root).__name__} has no repeated blocks"
            )
        list_name, blocks = next(iter(lists.items()))
        self.name = f"{list_name}.0"
        self.block = blocks[0]
        self.enabled = False
        self.inputs: tuple[Any, Any] | None = None
        self._handle = self.block.register_forward_pre_hook(self._capture, with_kwargs=True)

    def _capture(self, _module: Any, args: Any, kwargs: Any) -> None:
        if self.enabled and self.inputs is None:
            self.inputs = _map_tensors((args, kwargs), lambda t: t.detach().clone())

    def remove(self) -> None:
        self._handle.remove()


def _audit_block(model: Any, captured: _BlockInputs, *, device: Any) -> dict[str, Any]:
    """One block's forward and trainable gradient, policy numerics against float64.

    Both sides run the block through ``functional_call`` over device copies of
    its own tensors, the policy side under the model's autocast as
    ``forward_step`` runs it, the reference side in float64 with autocast off.
    The scalar whose gradient is compared projects every output on one fixed
    Gaussian direction, so every output element reaches the gradient.

    The errors mean nothing without the problem's own conditioning: a
    saturated attention softmax and the cancellation in a projected sum make
    even an exact float64 gradient move by tens of percent under bf16-sized
    perturbations (measured on Qwen-Image block 0 on CPU: float64 with inputs
    and weights jittered by 2^-9 moved the gradient by 29%, the bf16 policy by
    40%, fp32 by 17%). A third run therefore repeats the float64 reference
    with every floating input and tensor scaled by ``1 + 2^-9 * N(0, 1)``
    (about bf16's unit roundoff), and the report carries the policy error in
    units of that floor: a kernel or semantic bug shows as a ratio far above
    one, bf16 rounding as a ratio near it.
    """

    import torch

    from vrl.models.precision import model_autocast

    block = captured.block
    if captured.inputs is None:
        raise SystemExit(f"[probe] FAIL: {captured.name} was not called at replay step 0")
    if block.training and any(
        isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0
        for module in block.modules()
    ):
        raise SystemExit("[probe] --block-audit: dropout is active, so the two calls would differ")
    args, kwargs = captured.inputs
    tensors = {**dict(block.named_parameters()), **dict(block.named_buffers())}
    trainable = [name for name in tensors if "lora_" in name] or [
        name for name, _ in block.named_parameters()
    ]
    generator = torch.Generator(device="cpu").manual_seed(0)
    projections: list[Any] = []

    def run(cast: Any) -> tuple[list[Any], Any]:
        state = {}
        for name, tensor in tensors.items():
            tensor = cast(tensor.detach().to(device))
            state[name] = tensor.requires_grad_() if name in trainable else tensor
        with _unstreamed(block):
            outputs = torch.func.functional_call(
                block, state, _map_tensors(args, cast), _map_tensors(kwargs, cast)
            )
        outputs = [
            output
            for output in (outputs if isinstance(outputs, (tuple, list)) else (outputs,))
            if isinstance(output, torch.Tensor) and output.is_floating_point()
        ]
        if not projections:
            projections.extend(
                torch.randn(output.shape, generator=generator, dtype=torch.float64).to(device)
                for output in outputs
            )
        scalar = sum(
            (output.double() * direction).sum()
            for output, direction in zip(outputs, projections, strict=True)
        )
        grads = torch.autograd.grad(scalar, [state[name] for name in trainable])
        return [output.detach().double() for output in outputs], torch.cat(
            [grad.double().flatten() for grad in grads]
        )

    with model_autocast(model, device), _audited_attention() as kernel_errors:
        policy_outputs, policy_grad = run(lambda t: t)
    with torch.autocast(device_type=device.type, enabled=False), _reference_attention() as logits:
        reference_outputs, reference_grad = run(_float64)
        jitter = torch.Generator(device=device).manual_seed(1)

        def perturbed(tensor: Any) -> Any:
            tensor = _float64(tensor)
            if not tensor.is_floating_point():
                return tensor
            noise = torch.randn(tensor.shape, generator=jitter, device=device, dtype=tensor.dtype)
            return tensor * (1 + 2**-9 * noise)

        floor_outputs, floor_grad = run(perturbed)
    if not logits:
        raise SystemExit(
            f"[probe] FAIL: {captured.name} attention does not call "
            "F.scaled_dot_product_attention, so the float64 reference would not cover it"
        )

    def relative(a: Any, b: Any) -> float:
        return ((a - b).norm() / b.norm()).item()

    def cosine(a: Any, b: Any) -> float:
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    forward = [relative(a, b) for a, b in zip(policy_outputs, reference_outputs, strict=True)]
    forward_floor = [relative(a, b) for a, b in zip(floor_outputs, reference_outputs, strict=True)]
    grad, grad_floor = relative(policy_grad, reference_grad), relative(floor_grad, reference_grad)
    return {
        "block": captured.name,
        "reference_dtype": "float64",
        # The reference and floor runs both record; the first half is the reference.
        "attention_calls": len(kernel_errors),
        "attention_max_logit": logits[: len(kernel_errors)],
        "attention_kernel_rel_l2": kernel_errors,
        "attention_kernel_passed": max(kernel_errors) <= ATTENTION_KERNEL_MAX_REL_L2,
        "trainable_tensors": len(trainable),
        "trainable_numel": policy_grad.numel(),
        "forward_rel_l2": forward,
        "forward_floor_rel_l2": forward_floor,
        "forward_over_floor": [a / b for a, b in zip(forward, forward_floor, strict=True)],
        "grad_rel_l2": grad,
        "grad_floor_rel_l2": grad_floor,
        "grad_over_floor": grad / grad_floor,
        "grad_cosine": cosine(policy_grad, reference_grad),
        "grad_floor_cosine": cosine(floor_grad, reference_grad),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }


@dataclass(slots=True)
class ProbeInputs:
    """One recipe's prepared rollout, shared by forward and training probes."""

    root: Any
    entry: Any
    model: Any
    request: Any
    params: Any
    loop_config: Any
    state: Any
    replay_rows: int
    quantization: Any
    compiled: bool


def prepare_inputs(args: argparse.Namespace, root: Any = None) -> ProbeInputs:
    """Build and condition the recipe through the production family/executor APIs."""
    from dataclasses import replace

    import torch

    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.generation.steps.denoise.config import DenoiseLoopConfig
    from vrl.generation.types import GenerationInput, GenerationRequest
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.families.semantics import task_type_for
    from vrl.rollouts.collector import RolloutCollectorConfig
    from vrl.trainers.online.config import OnlineBatchConfig
    from vrl.utils.config import import_from_path

    if root is None:
        root = parse_config(load_config(args.config, overrides=args.overrides))
    entry = get_model_family_entry(str(root.model.family))
    batch_config = OnlineBatchConfig.from_root(root)
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

    return ProbeInputs(
        root,
        entry,
        model,
        request,
        params,
        loop_config,
        state,
        replay_rows,
        quantization,
        compiled,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=False)

    import torch

    from vrl.generation.steps.denoise.loop import run_denoise_loop
    from vrl.math.denoise.flow_matching import sde_step_with_logprob
    from vrl.models.families.semantics import VIDEO_TASKS
    from vrl.trainers.core.types import ReplayParityConfig

    inputs = prepare_inputs(args)
    entry, model = inputs.entry, inputs.model
    params, state, loop_config = inputs.params, inputs.state, inputs.loop_config
    quantization, compiled = inputs.quantization, inputs.compiled
    transformer = model.transformer
    rollout_rows, replay_rows = loop_config.sample_count, inputs.replay_rows
    device = next(transformer.parameters()).device
    replay_parity = (
        inputs.root.trainer.replay_parity if inputs.root.trainer is not None else None
    ) or ReplayParityConfig()
    gate = float(replay_parity.max_abs_logprob_diff)
    window_only = (
        inputs.root.actor is not None and inputs.root.actor.timestep_selection == "sde_window"
    )

    # Per-module hooks are closures dynamo guards on, so every hooked block
    # would recompile until the recompile limit drops it to eager: a compiled
    # run is compared per step only, as is a run that opted out of the hooks.
    module_hooks = not compiled and not args.no_module_hooks
    recorder = _Recorder(transformer, rows=rollout_rows, hooks=module_hooks)
    if args.block_audit and compiled:
        raise SystemExit("[probe] --block-audit hooks a block, which a compiled run would retrace")
    block_inputs = _BlockInputs(transformer) if args.block_audit else None
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
            if block_inputs is not None:
                block_inputs.enabled = step_idx == 0
            # The trainer's own index mapping: most families rebuild a one-step
            # state (index 0), Cosmos keeps the full table.
            out = model.forward_step(restored, model._replay_forward_step_index(step_idx))
            recorder.enabled = False
            if block_inputs is not None:
                block_inputs.enabled = False
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
            density = sde_step_with_logprob(
                scheduler,
                out["noise_pred"].float(),
                result.timesteps[0, step_idx].unsqueeze(0),
                observation.float(),
                prev_sample=result.actions[:replay_rows, step_idx].float(),
                noise_level=loop_config.sde.noise_level,
                sde_type=loop_config.sde.sde_type,
                step_index=step_idx,
                math_dtype=torch.float64,
            )
            entry_report["density_fp64_max_abs"] = (
                (sde.log_prob.double() - density.log_prob).abs().max().item()
            )
            window = loop_config.sde_window
            in_window = window is None or window[0] <= step_idx < window[1]
            entry_report["trained"] = in_window or not window_only
            if step_idx in rollout_noise and in_window:
                if getattr(scheduler, "_vrl_edm_sigma_domain", False):
                    entry_report["residual"] = {"skipped": "EDM sigma domain"}
                else:
                    sampled = sde_step_with_logprob(
                        scheduler,
                        rollout_noise[step_idx],
                        result.timesteps[0, step_idx].unsqueeze(0),
                        result.observations[:, step_idx].float(),
                        prev_sample=result.actions[:, step_idx].float(),
                        noise_level=loop_config.sde.noise_level,
                        sde_type=loop_config.sde.sde_type,
                        step_index=step_idx,
                        math_dtype=torch.float64,
                        return_dt=True,
                    )
                    entry_report["residual"] = _residual_stats(
                        sampled, result.actions[:, step_idx], sde_type=loop_config.sde.sde_type
                    )
            if step_idx in rollout_noise:
                entry_report["noise_pred_max_abs"] = (
                    (out["noise_pred"].float() - rollout_noise[step_idx][:replay_rows])
                    .abs()
                    .max()
                    .item()
                )
            if step_idx == 0 and module_hooks:
                entry_report.update(
                    _compare_calls(step0_calls, recorder.take(), recorder.block_names)
                )
            report_steps.append(entry_report)
            first = entry_report.get("first_divergent_module")
            residual = entry_report.get("residual", {})
            print(
                f"[probe] step {step_idx}: |Δlogprob|={entry_report['logprob_max_abs']:.3e}"
                f" fp64 density={entry_report['density_fp64_max_abs']:.3e}"
                + (
                    f" z mean={residual['mean']:+.2e} E[z^2]-1={residual['mean_square'] - 1:+.2e}"
                    if "mean" in residual
                    else ""
                )
                + (f" first={first['module']} ({first['max_abs']:.3e})" if first else "")
            )
    recorder.remove()
    block_audit = None
    if block_inputs is not None:
        block_inputs.remove()
        block_audit = _audit_block(model, block_inputs, device=device)
        print(
            f"[probe] block audit {block_audit['block']}: forward rel l2 "
            + ", ".join(
                f"{value:.3e} ({ratio:.2f}x floor)"
                for value, ratio in zip(
                    block_audit["forward_rel_l2"], block_audit["forward_over_floor"], strict=True
                )
            )
            + f"; grad rel l2 {block_audit['grad_rel_l2']:.3e}"
            f" ({block_audit['grad_over_floor']:.2f}x floor)"
            f" cosine {block_audit['grad_cosine']:.6f}"
            f" (floor {block_audit['grad_floor_cosine']:.6f})"
            f" over {block_audit['trainable_numel']} elements"
        )
        print(
            "[probe] block audit attention: kernel rel l2 vs float64 on its own inputs "
            + ", ".join(f"{value:.3e}" for value in block_audit["attention_kernel_rel_l2"])
            + f" (bound {ATTENTION_KERNEL_MAX_REL_L2:.0e}); max |logit| "
            + ", ".join(f"{value:.4g}" for value in block_audit["attention_max_logit"])
        )

    max_logprob_abs = max(step["logprob_max_abs"] for step in report_steps if step["trained"])
    residuals_passed = all(
        step["residual"].get("passed", True) for step in report_steps if "residual" in step
    )
    kernel_passed = block_audit is None or block_audit["attention_kernel_passed"]
    passed = max_logprob_abs <= gate and residuals_passed and kernel_passed
    print(f"[probe] max |Δlogprob| = {max_logprob_abs:.3e} over trained steps (gate {gate:.3e})")
    print(
        "[probe] max fp64 density |Δlogprob| = "
        f"{max(step['density_fp64_max_abs'] for step in report_steps):.3e}; "
        f"residuals {'N(0, 1)' if residuals_passed else 'NOT N(0, 1)'}"
    )
    if artifact_dir is not None:
        from vrl.utils.media import write_mp4, write_png

        if image.ndim == 5 and entry.task not in VIDEO_TASKS:
            # A frame stack of an image task (Qwen-Image-Layered's RGBA layers).
            for frame, layer in enumerate(image[0].float().cpu()):
                write_png(layer, artifact_dir / f"image_{frame}.png")
        elif image.ndim == 5:
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
            "module_hooks": module_hooks,
            "quantization": None if quantization is None else quantization.format,
            "max_abs_logprob_diff_gate": gate,
            "max_logprob_abs": max_logprob_abs,
            "mean_logprob_abs": sum(step["logprob_mean_abs"] for step in report_steps)
            / len(report_steps),
            "steps": report_steps,
            "block_audit": block_audit,
            "decoded_shape": list(image.shape),
            "decoded_std": image_std,
            "seconds_including_load": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        (artifact_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if not residuals_passed:
        raise SystemExit("[probe] FAIL: sampled residuals are not N(0, 1)")
    if not kernel_passed:
        raise SystemExit(
            "[probe] FAIL: an attention kernel disagrees with float64 on its own inputs"
        )
    if not passed:
        raise SystemExit("[probe] FAIL: replay |Δlogprob| exceeds the replay gate")
    print(f"[probe] PASS: {entry.family} rollout and replay verified")


if __name__ == "__main__":
    main()
