"""Diffusion rollout stage topology — the SGLang-Omni-shape stage graph.

This is the DECLARATION half of the staged pipeline (the StageConfig analog from
``vrl/generation/pipeline/topology.py``). It names the diffusion rollout stages and
their order so they can be run either in-process (SerialPipelineRunner) or as
physical Ray stage actors (RayPipelineRunner). The EXECUTION/wiring of the existing
``DiffusionExecutor`` stage methods (run_prompt_encode_stage / run_prepare_stage /
run_denoise_stage / run_decode_stage) into these stages, and the concurrent
multi-payload pipelining (bounded queues / backpressure), are the follow-on steps
documented in ``docs/sprints/parked/SPRINT_diffusion_rollout_stage_pipeline.md``.

Per-stage ``gpu_ids`` is the load-bearing knob for the multi-GPU win: two stages
on DIFFERENT GPUs run their tensor-core compute truly in parallel (throughput =
1/max(stage), not the sum). On ONE GPU the stages' compute serializes on the
shared tensor cores (measured: compute∥compute is neutral); the single-GPU win is
only the inter-stage copy+CPU overlap (measured ~20% for compute∥(copy+CPU)).
Defaults leave ``gpu_ids`` empty = inherit the rollout GPU (today's single-card
behavior, bit-identical). The full-rollout variant that appends a terminal
``reward`` stage (denoise ∥ reward across GPUs) is part of the parked sprint
above — reward today is scored separately in the collector.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.pipeline.topology import PipelineStage, PipelineTopology

# Stage names — these match the DiffusionExecutor stage methods + the existing
# ``generation.*`` record_function / stage_durations labels, so the pipeline
# stages line up 1:1 with the profiler attribution (P0 nsys NVTX ranges).
PROMPT_ENCODE = "prompt_encode"
PREPARE = "prepare"
DENOISE = "denoise"
DECODE = "decode"


def build_diffusion_pipeline_topology(
    *,
    gpu_ids: dict[str, tuple[int, ...]] | None = None,
) -> PipelineTopology:
    """Build the linear diffusion generation stage graph (decode terminal).

    ``gpu_ids`` optionally pins a stage onto specific GPU ordinals (the multi-GPU
    per-stage placement). Omitted/empty = the stage inherits the rollout GPU, so a
    single-card run is unchanged. The graph is intentionally linear (one
    ``next_stages`` each) — both runners currently support a single successor, and
    diffusion rollout has no fan-out. Reward is scored separately in the collector
    (the ≥2-GPU async-reward path); the single-GPU chunk overlap lives entirely in
    these four generation stages.
    """

    placement = gpu_ids or {}
    stages = [
        PipelineStage(
            name=PROMPT_ENCODE,
            next_stages=(PREPARE,),
            gpu_ids=placement.get(PROMPT_ENCODE, ()),
        ),
        PipelineStage(
            name=PREPARE,
            next_stages=(DENOISE,),
            gpu_ids=placement.get(PREPARE, ()),
        ),
        # denoise: the GPU-bound DiT loop. The largest stage; on multi-GPU it owns
        # its own card(s) so another stage can run on a different card meanwhile.
        PipelineStage(
            name=DENOISE,
            next_stages=(DECODE,),
            gpu_ids=placement.get(DENOISE, ()),
        ),
        # decode: VAE latents -> pixels, terminal for the generation pipeline.
        PipelineStage(
            name=DECODE,
            terminal=True,
            gpu_ids=placement.get(DECODE, ()),
        ),
    ]
    return PipelineTopology(stages=tuple(stages), entry_stage=PROMPT_ENCODE)


def run_chunk_through_pipeline(executor: Any, request: Any, chunk: Any) -> Any:
    """Run ONE sample chunk through the executor generation stages via the pipeline
    contract (SerialPipelineRunner). BEHAVIOR-IDENTICAL to the monolithic
    ``forward_chunk_plan`` — same stage methods, same order, same outputs — but
    structured as named stages so the concurrent pipeline (chunk N+1 denoise ∥
    chunk N decode) can later drive the SAME handlers across Ray stage actors.

    This is the T2 foundation: serial, in-process, bit-exact. It carries no overlap
    on its own; it only re-expresses the existing chain as routable stages.
    """

    from vrl.generation.pipeline import PipelineStagePayload, SerialPipelineRunner
    from vrl.utils.profiling import record_function

    stage_durations: dict[str, float] = {}

    def _encode(_payload: PipelineStagePayload) -> Any:
        prompt_input = executor.build_prompt_stage_input(request, chunk)
        return executor.run_prompt_encode_stage(
            prompt_input,
            stage_durations=stage_durations,
            record_function=record_function,
        )

    def _prepare(payload: PipelineStagePayload) -> Any:
        return executor.run_prepare_stage(payload.data, stage_durations=stage_durations)

    def _denoise(payload: PipelineStagePayload) -> Any:
        return executor.run_denoise_stage(payload.data, stage_durations=stage_durations)

    def _decode(payload: PipelineStagePayload) -> Any:
        return executor.apply_wire_storage_policy(
            request,
            executor.run_decode_stage(payload.data),
        )

    topology = build_diffusion_pipeline_topology()
    runner = SerialPipelineRunner(
        topology,
        {
            PROMPT_ENCODE: _encode,
            PREPARE: _prepare,
            DENOISE: _denoise,
            DECODE: _decode,
        },
    )
    payload = PipelineStagePayload(
        request_id=str(getattr(request, "request_id", "")),
        stage=PROMPT_ENCODE,
        data=None,  # _encode rebuilds from (request, chunk) captured above
    )
    result = runner.run(payload)
    if result.error:
        raise RuntimeError(f"diffusion pipeline stage failed: {result.error}")
    return result.payload.data


def _move_tree_to_cpu_async(value: Any, stream: Any) -> Any:
    """Move every CUDA tensor in a (possibly nested) structure to pinned CPU on
    ``stream`` with a non-blocking copy — NO global sync. Mirrors worker._to_cpu
    but stream-scoped + event-based, so the D2H drains concurrently with the next
    chunk's denoise instead of blocking it (the copy engine ≠ the tensor cores).
    """

    import torch

    def _leaf(t: Any) -> Any:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            with torch.cuda.stream(stream):
                host.copy_(t.detach(), non_blocking=True)
            return host
        return t

    if isinstance(value, torch.Tensor):
        return _leaf(value)
    if isinstance(value, dict):
        return {k: _move_tree_to_cpu_async(v, stream) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        mapped = [_move_tree_to_cpu_async(v, stream) for v in value]
        return type(value)(mapped) if not isinstance(value, tuple) else tuple(mapped)
    # dataclass-like chunk result: copy its tensor-bearing public attributes
    if hasattr(value, "__dict__"):
        for name, attr in list(vars(value).items()):
            if name.startswith("_"):
                continue
            moved = _move_tree_to_cpu_async(attr, stream)
            if moved is not attr:
                setattr(value, name, moved)
        return value
    return value


def forward_chunks_pipelined(
    executor: Any,
    request: Any,
    chunks: Any,
    *,
    produce_fn: Any = None,
    teardown_fn: Any = None,
) -> list:
    """In-process software pipeline over a request's chunks: while chunk N+1's
    PRODUCE (encode->prepare->denoise->decode, GPU compute on the default stream)
    runs, chunk N's TEARDOWN (the GPU->CPU result copy + host packing, on a copy
    stream) drains — hiding the copy+CPU boundary behind the next chunk's denoise.

    BIT-EXACT by construction: the side-stream copy never changes values, and the
    only ordering it introduces (teardown(N) before produce(N+1) is launched) is
    guarded by a default-stream event so the copy never reads tensors a denoise
    kernel is still writing (torn-read safety). Chunks are independent and gather
    re-orders by ordered_chunks, so completion order is irrelevant to the output.

    ``produce_fn``/``teardown_fn`` are injectable for testing; defaults use the T2
    pipeline produce + a stream-scoped GPU->CPU teardown. Returns the per-chunk
    results in CHUNK ORDER (the caller gathers as usual).
    """

    import torch

    produce = produce_fn or (lambda chunk: run_chunk_through_pipeline(executor, request, chunk))
    cuda = torch.cuda.is_available()
    copy_stream = torch.cuda.Stream() if cuda else None

    def _teardown(result: Any) -> Any:
        if teardown_fn is not None:
            return teardown_fn(result)
        if copy_stream is None:
            return result
        return _move_tree_to_cpu_async(result, copy_stream)

    chunk_list = list(chunks)
    results: list = [None] * len(chunk_list)
    prev_idx = -1
    prev_result = None
    prev_done = None  # default-stream event marking prev produce complete
    pending_events: list = []

    for idx, chunk in enumerate(chunk_list):
        # Start the PREVIOUS chunk's teardown on the copy stream BEFORE producing
        # this chunk, so the D2H overlaps this chunk's denoise. Wait on prev_done so
        # the copy never reads a tensor the denoise kernels are still writing.
        if prev_result is not None and copy_stream is not None:
            copy_stream.wait_event(prev_done)
            results[prev_idx] = _teardown(prev_result)
            ev = torch.cuda.Event()
            ev.record(copy_stream)
            pending_events.append(ev)
        elif prev_result is not None:
            results[prev_idx] = _teardown(prev_result)

        prev_result = produce(chunk)
        prev_idx = idx
        if cuda:
            prev_done = torch.cuda.Event()
            prev_done.record()  # default stream: this chunk's produce is enqueued

    # Flush the final chunk's teardown.
    if prev_result is not None:
        if copy_stream is not None:
            copy_stream.wait_event(prev_done)
            results[prev_idx] = _teardown(prev_result)
            ev = torch.cuda.Event()
            ev.record(copy_stream)
            pending_events.append(ev)
        else:
            results[prev_idx] = _teardown(prev_result)

    # Single join point: the copies must complete before the caller reads them.
    for ev in pending_events:
        ev.synchronize()
    return results


__all__ = [
    "DECODE",
    "DENOISE",
    "PREPARE",
    "PROMPT_ENCODE",
    "build_diffusion_pipeline_topology",
    "forward_chunks_pipelined",
    "run_chunk_through_pipeline",
]
