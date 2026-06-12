# sglang-omni — Architecture Reading

> Repo: `/home/mingfeiguo/Desktop/sglang-omni`. All paths below are relative to the repo root unless noted.
>
> **The delta vs vanilla SGLang, read this first.** sglang-omni is **not a fork** of sglang — it is a separate orchestration package that imports upstream `sglang==0.5.8` as a pip dependency (`pyproject.toml:29`), alongside the transports that define the project's character: `pyzmq` (ZMQ control plane), `nixl`, and `mooncake-transfer-engine` (`pyproject.toml:32-33` area). Vanilla sglang gives one AR engine (tokenizer → scheduler → detokenizer processes) serving one model. sglang-omni adds:
>
> 1. **A multi-stage pipeline runtime** for omni models whose stages have heterogeneous compute profiles ("a compute-bound thinker, a memory-bound talker, a latency-sensitive codec" — `README.md`, About). Each stage (preprocessing, image/audio encoder, fan-in aggregator, thinker AR, talker AR, code2wav vocoder, decode) runs **its own scheduler in its own OS process** (or a shared process for non-TP stages), declared in a `PipelineConfig`.
> 2. **A new scheduler family** behind one uniform `inbox`/`outbox` contract: `OmniScheduler` (re-uses sglang's upstream `Scheduler` *methods* via `__getattr__` composition, no inheritance), `SimpleScheduler` / `StreamingSimpleScheduler` (non-AR stages), `DllmScheduler` (diffusion LLM, for LLaDA2.0-Uni), `Code2WavScheduler` (streaming vocoder).
> 3. **A two-plane communication system**: ZMQ PUSH/PULL control plane (msgpack messages) + a pluggable tensor "relay" data plane (`shm`/`nccl`/`nixl`/`mooncake`), with same-process `LOCAL_OBJECT` and same-GPU CUDA-IPC fast paths.
> 4. **New process/actor types**: a `Coordinator` (in the HTTP server process), per-stage worker processes spawned via `multiprocessing` spawn context, TP leader/follower rank processes per stage, and an optional external HTTP router (`sglang_omni_router`) that load-balances across whole server instances.
> 5. **In-process sglang engine embedding**: instead of launching sglang's own tokenizer/scheduler/detokenizer process tree, each AR stage constructs sglang's `ModelRunner`, memory pools, and tree cache directly inside the stage process (`sglang_omni/scheduling/bootstrap.py:9-84`).
>
> **Ray is not used anywhere** — `grep -rn "^import ray|^from ray"` over the repo returns zero hits (verified). Orchestration is `multiprocessing` (spawn) + ZMQ over per-run `ipc://` unix sockets + per-stage NCCL groups for TP + relay backends for tensor transfer.

---

## 1. Repo layout & module organization

Top level: `sglang_omni/` (main package), `sglang_omni_router/` (separate multi-worker HTTP router package, console script `sgl-omni-router`, `pyproject.toml:83-85`), plus `benchmarks/`, `docs/`, `examples/`, `playground/`, `scripts/`, `tests/`.

| Directory | Owns | Key files (read) |
|---|---|---|
| `sglang_omni/pipeline/` | Inter-stage orchestration: process lifecycle, coordinator, stage IO shell, ZMQ control plane, TP fanout | `mp_runner.py` (537 LOC, process topology builder + runner), `stage_workers.py` (717, subprocess entrypoint), `stage/runtime.py` (1257, the `Stage` class), `coordinator.py`, `control_plane.py`, `relay_io.py`, `local_dispatch.py`, `tp_control.py`, `runtime_config.py` |
| `sglang_omni/scheduling/` | Per-stage execution loops behind a uniform inbox/outbox contract | `omni_scheduler.py` (1176, AR via sglang composition), `simple_scheduler.py`, `dllm_scheduler.py`, `streaming_simple_scheduler.py`, `bootstrap.py`, `sglang_backend/` (prefill/decode managers, cache, server-args builder) |
| `sglang_omni/model_runner/` | AR forward path shared bases | `model_worker.py` (wraps sglang `ModelRunner`), `thinker_model_runner.py`, `sglang_model_runner.py`, `base.py`, `_hidden_capture.py` |
| `sglang_omni/models/<model>/` | Model-local pipeline configs, stage factories, payload projections | `qwen3_omni/`, `qwen3_tts/`, `higgs_tts/`, `fishaudio_s2_pro/`, `moss_tts/`, `voxtral_tts/`, `ming_omni/`, `llada2_uni/`, `whisper_asr/`, `qwen3_asr/` |
| `sglang_omni/config/` | Declarative pipeline schema + placement/topology planning | `schema.py` (`PipelineConfig`/`StageConfig`), `placement.py` (GPU plan), `topology.py` (process-grouping plan), `runtime.py`, `manager.py` |
| `sglang_omni/relay/` | Data-plane backends behind a registry | `base.py` (`RELAY_REGISTRY`, `create_relay`), `shm.py`, `nccl.py`, `nixl.py`, `mooncake.py` |
| `sglang_omni/serve/` | OpenAI-compatible FastAPI server + launcher; `/v1/realtime` WebSocket | `launcher.py`, `openai_api.py`, `realtime/` (VAD, session, audio buffer) |
| `sglang_omni/client/` | In-process client bridging API handlers to the Coordinator | `client.py` |
| `sglang_omni/proto/` | Control-plane message types + `StagePayload`/`OmniRequest` | `messages.py`, `request.py`, `stage.py` |
| `sglang_omni/vendor/sglang/` | **Not vendored source** — a static re-export wrapper around installed sglang for IDE navigation + optional patches: "Use static imports to preserve IDE navigation, and apply optional patches below" (`vendor/sglang/core.py:1-21`) | `core.py`, `distributed.py`, `layers.py`, `models.py` |
| `sglang_omni_router/` | Capability-aware HTTP load balancer over multiple sgl-omni servers | `serve.py`, `worker.py`, `selector.py`, `launcher/local.py` |

Model directory convention is documented in `docs/developer_reference/main.md`: "Only model-local behavior belongs here. The framework-owned layers are still `Stage`, `Coordinator`, schedulers, model-runner bases, relay, runtime prep". Nearly every model package carries the same uniform file set: `config.py`, `stages.py`, `request_builders.py`, `payload_types.py`, `model_runner.py`/`*_model_runner.py`, `bootstrap.py` (verified across `models/{qwen3_omni,qwen3_tts,higgs_tts,moss_tts,fishaudio_s2_pro,llada2_uni}/`).

Size norms: infra (non-`models/`, non-`vendor/`) is 97 files / ~22k lines ≈ 227 lines mean; the ceiling is ~1.2k (`pipeline/stage/runtime.py` 1257, `scheduling/omni_scheduler.py` 1176, `cli/serve.py` 1098). Tiny single-purpose files are accepted without shame: `proto/stage.py` is 12 lines holding one 2-field dataclass (`proto/stage.py:7-12`).

---

## 2. Architecture overview

### Layering

`HTTP API -> Client -> Coordinator -> Stage -> Scheduler -> ModelRunner -> model forward` (`docs/developer_reference/main.md:9-11`, verified against code below).

### Process / thread / loop topology

```
┌─ Parent OS process (uvicorn) ───────────────────────────────────────────────┐
│ FastAPI app (openai_api.create_app) ── Client ── Coordinator                │
│ MultiProcessPipelineRunner: spawn + wait_ready + _monitor_children          │
│   Coordinator sockets: completion PULL (bind), abort PUB,                   │
│                        per-stage PUSH (SubmitMessage / Shutdown)            │
└────────────┬───────────────────────────────────┬─────────────────────────---┘
   ZMQ ipc:// SubmitMessage              ZMQ ipc:// CompleteMessage/StreamMessage
             ▼                                    ▲
┌─ Stage process "preprocessing" ─┐   ┌─ Stage process "thinker" (per TP rank) ─┐
│ asyncio loop:                   │   │ asyncio loop: Stage IO shell            │
│  Stage.run(): ctrl recv,        │   │ scheduler thread: OmniScheduler         │
│   abort SUB listener,           │   │   _event_loop_normal/overlap/async      │
│   outbox drain                  │   │   → sglang get_next_batch_to_run /      │
│ scheduler thread:               │   │     run_batch / process_batch_result    │
│  SimpleScheduler inbox→fn→outbox│   │   (ModelWorker wraps sglang ModelRunner)│
└──────────┬──────────────────────┘   └———──────┬────────────────────────────---┘
           │  control: DataReadyMessage (ZMQ PUSH→PULL, msgpack)
           │  data:    relay put_async/get_async (shm | nccl | nixl | mooncake)
           │  fast paths: LOCAL_OBJECT (same process), CUDA IPC (same GPU)
           ▼
   image_encoder / audio_encoder ─→ mm_aggregate (fan-in) ─→ thinker
   thinker ──stream(hidden states)──→ talker_ar ──stream(codes)──→ code2wav ─┐
   thinker ──→ decode (terminal) ────────────────→ Coordinator ◄─(terminal)──┘
```

Concrete pipeline shape (Qwen3-Omni speech): 8 stages — `preprocessing → {image_encoder, audio_encoder, mm_aggregate}` (fan-out with `project_payload` per target), `mm_aggregate` fan-in with `wait_for=["preprocessing", "image_encoder", "audio_encoder"]` + `merge_fn` (`models/qwen3_omni/config.py:78-105`), `thinker` with `stream_to=["talker_ar", "decode"]` (`config.py:108-134`), `talker_ar` with `stream_to=["code2wav"]` and `can_accept_stream_before_payload=True` (`config.py:147-179`), and two terminal stages `decode` and `code2wav` whose results the Coordinator merges (`pipeline/coordinator.py:53-61`).

### Process model

Each OS process can host **multiple stages sharing one asyncio loop**; each stage's scheduler runs in its own thread. From `sglang_omni/pipeline/stage_workers.py:362-410`:

```python
def _run_process(spec, ready_event, log):
    """Construct and drive all stages owned by one OS process.
    - All stages in ``spec.stage_specs`` share this OS process and one asyncio
      event loop. ``asyncio.gather`` runs them concurrently; **if any stage
      raises, the whole process exits** ...
    """
    local_dispatcher = LocalStageDispatcher()
    stages = [_construct_stage(stage_spec, log, local_dispatcher=local_dispatcher) ...]
```

There is an explicit shared-failure-domain caveat: "if any stage raises, the whole process exits ... There is no per-stage failure isolation inside one process group" (`stage_workers.py:367-380`). A hard invariant on the other side: "a TP stage (`tp_size > 1`) must own its OS process exclusively" (`stage_workers.py:118-131`).

### Core loop 1 — the Stage IO shell (`pipeline/stage/runtime.py`)

Each stage process runs `Stage.run()`: one asyncio loop receiving control messages, plus a **dedicated scheduler thread** started in `Stage.start()`:

```python
self._scheduler_thread = threading.Thread(
    target=_run_scheduler, name=f"scheduler-{self.name}", daemon=True,
)
```
(`pipeline/stage/runtime.py:181-186`). The main loop dispatches by message type — `SubmitMessage` (entry payload from coordinator), `DataReadyMessage` with `chunk_id` → stream chunk, with `is_done/error` → stream signal, otherwise full payload (`runtime.py:250-263`). A full payload is read back from relay (`runtime.py:293-295`), pushed through the input handler (fan-in `AggregatedInput` or `DirectInput`), and enqueued:

```python
self.scheduler.inbox.put(
    IncomingMessage(request_id=request_id, type="new_request", data=payload)
)
```
(`runtime.py:652-654`). Two background tasks per stage: `_abort_listener` (ZMQ SUB) and `_drain_outbox` (`runtime.py:203-210`), which converts scheduler `OutgoingMessage`s of type `result`/`stream`/`error` into downstream sends (`runtime.py:666-707`). The invariant "Stage does not branch on scheduler type" holds: every scheduler exposes `inbox`, `outbox`, `start()`, `stop()`, `abort()` (`scheduling/omni_scheduler.py:71-73`, `scheduling/simple_scheduler.py:45-47`, `scheduling/dllm_scheduler.py:28-32`).

Routing on result: dynamic `route_fn` or static `next` (validated against the static topology, `pipeline/stage_workers.py:469-517`); terminal stages send `CompleteMessage` straight to the coordinator (`runtime.py:751-767`); fan-out applies per-target `project_payload` functions before sending (`runtime.py:808-816`).

### Core loop 2 — OmniScheduler: sglang reuse via composition (`scheduling/omni_scheduler.py`)

Detailed in §3 below — the single most distinctive design in the repo.

### In-process sglang engine

The sglang engine itself is built in-process per AR stage by `create_sglang_infrastructure`, which constructs `ModelWorker` (wrapping `sglang.srt.model_executor.model_runner.ModelRunner` via `ModelConfig.from_server_args`, `model_runner/model_worker.py:34-82`), memory pools, tree cache, and Prefill/Decode managers (`scheduling/bootstrap.py:9-84`). Per-architecture sub-config extraction (e.g. `"Qwen3OmniTalker": ("talker_config", "text_config")`) lets one HF checkpoint feed multiple AR stages (`model_worker.py:24-31`).

### Notable design judgments (evidence-backed)

- **Composition-over-inheritance for sglang reuse** (`omni_scheduler.py:1-12`) keeps sglang's scheduling MRO intact while swapping transport — but it hard-pins `sglang==0.5.8` (`pyproject.toml:29`), since `OmniScheduler.__init__` mirrors upstream `Scheduler.__init__`'s attribute layout field by field (`omni_scheduler.py:118-280`); any upstream attribute rename breaks it silently at `__getattr__` time.
- **Single-node control plane by construction** (`ipc://` sockets, `runtime_config.py:160-173`) — cross-node pipelines would require endpoint scheme changes, even though the relay layer (NIXL/Mooncake) is already network-capable.
- **Fail-fast, no per-request recovery**: dead stage process ⇒ fail all in-flight requests and stop the whole runner (`mp_runner.py:439-457`); failure isolation granularity is the OS process group, deliberately documented (`stage_workers.py:367-380`).

---

## 3. Core scheduling & orchestration

### 3.1 Three scheduler families, one contract

Every scheduler exposes the same contract to its `Stage`: `inbox` (queue of `IncomingMessage` with types `new_request` / `stream_chunk` / `stream_done`), `outbox` (queue of `OutgoingMessage` with types `result` / `stream` / `error`), `start()`, `stop()`, `abort(request_id)` (`scheduling/omni_scheduler.py:71-72`, `scheduling/messages.py:10-23`). The `Stage` runs the scheduler in a dedicated daemon thread (`stage/runtime.py:153-186`) and drains its outbox from the asyncio loop (`stage/runtime.py:660-707`). `SimpleScheduler` mirrors it explicitly: "Same inbox/outbox interface as OmniScheduler so Stage doesn't need branching" (`scheduling/simple_scheduler.py:7`).

### 3.2 OmniScheduler — AR scheduling by composition over upstream SGLang

The central design trick (module docstring, `omni_scheduler.py:1-12`, verified verbatim):

```python
"""OmniScheduler — stage-facing AR scheduler using composition.

Uses SGLang's batch selection and result processing logic via **unbound
method calls** on the upstream ``Scheduler`` class.  No inheritance.

When an upstream method (e.g. ``get_next_batch_to_run``) internally calls
``self.get_new_batch_prefill()``, Python finds it through
``OmniScheduler.__getattr__`` → looks it up on the upstream class → binds
it to this instance.  This gives us the full scheduling MRO without
inheriting from ``SGLangScheduler``.
"""
```

Implemented in `__getattr__` (`omni_scheduler.py:360-384`, verified):

```python
def __getattr__(self, name: str):
    """Look up methods on the upstream SGLang Scheduler class. ..."""
    ...
    attr = getattr(_Upstream, name)   # _Upstream = sglang.srt.managers.scheduler.Scheduler
    if callable(attr):
        return types.MethodType(attr, self)
    return attr
```

So **batch selection, chunked prefill, retraction, priority scheduling, and result processing are literally upstream SGLang code** executing against state that `OmniScheduler.__init__` lays out by hand: `waiting_queue` / `running_batch` / `cur_batch` / `last_batch` (`omni_scheduler.py:180-191`), chunked-prefill fields (`:197-204`), the upstream `SchedulePolicy` and new-token-ratio dynamics (`:207-237`), and priority/preemption knobs (`:217-224`):

```python
self.policy = SchedulePolicy(
    self.schedule_policy, self.tree_cache,
    server_args.enable_hierarchical_cache,
    server_args.enable_priority_scheduling,
    server_args.schedule_low_priority_values_first,
)
self.enable_priority_scheduling = server_args.enable_priority_scheduling
self.try_preemption = server_args.enable_priority_scheduling
```

Features intentionally stubbed out so upstream code paths no-op: grammar/constrained decoding (`_NoOpGrammarManager`, `:47-65`), detokenizer ZMQ sender (`_NoOpSender`, `:40-44`, replaced by outbox emission), speculative decoding (`:258-272`), disaggregation (`:286-294`), LoRA/hierarchical cache (`:240-247`). The IO methods are **overridden** instead: `recv_requests`, `process_input_requests`, `run_batch`, `stream_output`, `send_to_tokenizer` (`:79-80`).

**Request intake** (`recv_requests`, `omni_scheduler.py:417-453`): drains the python-queue inbox on the TP entry rank and broadcasts to TP followers via upstream `broadcast_pyobj` — replacing vanilla SGLang's ZMQ recv from the TokenizerManager:

```python
def _recv_scheduler_messages(self) -> list[IncomingMessage]:
    if self.tp_size == 1:
        return self._drain_local_inbox()
    recv_msgs = self._drain_local_inbox() if self.is_entry_rank else []
    return broadcast_pyobj(recv_msgs, self.tp_group.rank, self.tp_cpu_group, src=self.tp_group.ranks[0])
```

`process_input_requests` (`:455-519`) converts a `StagePayload` into an SGLang `Req` via the model-specific `request_builder`, attaches omni state as `req._omni_data` (`:490`), runs an admission KV-capacity check (`_request_kv_capacity_error`, `:567-589`: reject if `input_len + max_new_tokens > max_req_len`), supports **deferred builds** for requests whose upstream stream input hasn't arrived yet (`_is_request_build_ready` → `_deferred_request_payloads`, `:468-472`), then appends to `waiting_queue` (`:519`).

### 3.3 The main scheduling loop, line by line

`start()` selects one of three event loops (`omni_scheduler.py:809-816`):

```python
def start(self) -> None:
    self._running = True
    if getattr(self, "enable_async_decode", False):
        self._event_loop_async_decode()
    elif self.enable_overlap:
        self._event_loop_overlap()
    else:
        self._event_loop_normal()
```

**Normal loop** (`omni_scheduler.py:895-923`, verified verbatim) — same skeleton as vanilla SGLang's `event_loop_normal`, with an omni-specific twist documented in the comment (the GIL note is a real multi-stage co-location constraint that does not exist in vanilla sglang):

```python
def _event_loop_normal(self) -> None:
    # Note (Chenyang): yield the GIL when idle so co-located non-AR stages
    # (encoders, preprocessor) running in sibling threads aren't starved
    # ... the audio_encoder forward pass ... slows ~600x, dropping audio QPS
    # from >10 to <0.5.
    while self._running:
        recv_reqs = self.recv_requests()                       # drain inbox + TP broadcast  (:903)
        recv_reqs.extend(self._take_deferred_request_payloads())  # re-check deferred builds (:904)
        self.process_input_requests(recv_reqs)                 # build Reqs → waiting_queue  (:905)
        if self._engine_paused:
            time.sleep(0.001); continue                        # (:906-908)

        batch = self.get_next_batch_to_run()                   # UPSTREAM batch selection    (:910)
        self.cur_batch = batch
        if batch:
            result = self.run_batch(batch)                     # overridden: omni runner     (:914)
            if result is not _FAILED_BATCH_RESULT:
                self.process_batch_result(batch, result)       # UPSTREAM result processing  (:916)
        else:
            self.self_check_during_idle()                      # reset new_token_ratio       (:918)
            time.sleep(0.001)                                  # GIL yield                   (:919)
        self.last_batch = batch
```

`get_next_batch_to_run` and `process_batch_result` are the **upstream** SGLang methods bound through `__getattr__`; they consume the prefill/decode queues, radix cache, retraction, and chunked-prefill state initialized above. (Upstream source lives in the installed `sglang==0.5.8`, not this checkout.)

**run_batch override** (`omni_scheduler.py:602-677`) bridges SGLang's `ScheduleBatch` to the omni `ModelRunner` and back to upstream's expected `GenerationBatchResult`:

```python
def _run_batch(self, batch, pp_proxy_tensors=None):
    self._emit_prefill_start_for_batch(batch)
    if self._model_runner is not None:
        sched_output = self._build_sched_output(batch)       # wraps batch + per-req _omni_data (:627-636)
        mr_output = self._model_runner.execute(sched_output) # forward + sample (:621)
        self._emit_stream_output(sched_output, mr_output)    # per-token stream chunks → outbox (:622)
        return self._make_batch_result(batch, mr_output)     # → GenerationBatchResult (:664-677)
    return _Upstream.run_batch(self, batch, pp_proxy_tensors)
```

`run_batch` itself is a **6-line error-boundary wrapper** over `_run_batch` (`omni_scheduler.py:602-607`): catch → `_handle_batch_failure` → return the module-level sentinel `_FAILED_BATCH_RESULT = object()` (`omni_scheduler.py:37`), so the event loops never see an exception from a batch.

Two output paths run **per step**: `_emit_stream_output` (`:638-662`) pushes per-token `OutgoingMessage(type="stream")` chunks (built by a model-specific `stream_output_builder`, e.g. thinker token-ids + hidden states for the talker) into the outbox; and `stream_output` (`:736-782`, called by upstream `process_batch_result` on finish) intercepts finished requests, runs `result_adapter`, and puts `OutgoingMessage(type="result")` — replacing vanilla's detokenizer hop entirely (`send_to_tokenizer` is a no-op, `:784-786`).

**Overlap loop** (`:925-968`) is the upstream two-step overlap pattern reproduced on omni state: results are queued (`self.result_queue = deque()`) and processed one iteration later so CPU result processing overlaps the next GPU step.

**Async-decode loop (omni-only "launch-first" one-step lookahead)** (`:1041-1123`) — used by Higgs TTS (`models/higgs_tts/config.py:54` sets `"enable_async_decode": True`):

```python
use_lookahead = (
    batch is not None
    and len(batch.reqs) >= self.async_decode_min_batch_size   # bs=1 bypass (the bs=1 regression, :135-140)
    and self._batch_is_decode(batch)
)
if use_lookahead:
    sched_output, pending_step = self._run_batch_launch(batch)  # enqueue forward+sample+async D2H, no wait (:679-687)
    prev_pending = self._async_pending
    self._async_pending = (batch.copy(), sched_output, pending_step)
    if prev_pending is not None:
        self._resolve_and_process(pb, ps, pstep)              # resolve step N-1 while N runs on GPU (:991-1026)
```

Each iteration *launches* decode step N and then *resolves* step N−1's host-side collect, so "~1.1ms of per-step CPU work overlaps the current step's GPU forward" (`:1041-1050`). Prefill/empty/small batches flush the in-flight step first (`_resolve_pending_async`, `:1028-1039` — exists because the in-flight lookahead step must be flushed from **three** call sites: pause, prefill transition, loop body) and run synchronously. Two correctness guards are scheduling-specific:

- **Lookahead overrun** (`_resolve_and_process`, `:991-1026`): a request finishing at step S is still in step S+1's already-launched batch; its rows are dropped from `next_token_ids` and `batch.reqs` before `process_batch_result`, else upstream would double-free its KV.
- **Stale-batch overrun on the drain transition** (`:1094-1112`): after flushing, finished reqs are filtered from the freshly built batch — the inline comment reads:

```python
# Stale-batch overrun: `batch` was built (get_next_batch_to_run,
# top of loop) BEFORE this drain. The drain can finish reqs that
# are still present in `batch` (the live running batch); running
# them again double-frees their committed KV cache
```

The GPU/host split lives in `ModelRunner.execute_launch` / `execute_resolve` (`model_runner/base.py:126-210`): launch enqueues forward + on-GPU sample + async D2H into a **pinned, ping-ponged host buffer** and records a CUDA event; resolve waits on the event (`query()` fast path counted as `_async_query_hit`, `base.py:185-189`) and runs the per-request collect. The ping-pong rationale is documented (`base.py:69-91`, verified):

```python
"""Return a pinned host buffer mirroring ``device_staging``'s full
shape, ping-ponging between two buffers on each call.

Two buffers are required: resolve(N) reads one on the host while
launch(N+1)'s async D2H writes the other. That CPU-read vs GPU-write
overlap is not protected by single-stream ordering (design.md §1.4).
"""
```

The sync path `execute()` documents that it is "byte-identical to the pre-async implementation: it is a pure extraction over the same shared sub-steps (`_build_forward_batch` / `_prepare_and_forward` / `_finalize`) that `execute_launch` + `execute_resolve` also use, in the same order" (`model_runner/base.py:93-102`). Invariant: "at most one `_PendingStep` is live at a time" (`base.py:29-30`); "The CALLER owns the handle ... launch-first scheduling has two steps momentarily in flight" (`base.py:133-137`).

**Preemption/eviction in the AR loop**: priority preemption is upstream's (`try_preemption`, `:218`); decode retraction under KV pressure is also upstream's via the composed `get_next_batch_to_run`. The standalone re-implementation in `DecodeManager.schedule_next_batch` (`scheduling/sglang_backend/decode.py:27-83`, `retract_decode` on `check_decode_mem()` failure, retracted reqs re-queued through `on_retract` → `PrefillManager.add_one_request`, `scheduling/bootstrap.py:70-74`) is used by the FishAudio custom scheduler (`models/fishaudio_s2_pro/fish_scheduler.py:36-82`), not by `OmniScheduler`.

**Abort** (`omni_scheduler.py:824-893`): removes from waiting queue, marks running reqs `FINISH_ABORT` so upstream finishes them cleanly next step (`_mark_running_request_aborted`, `:855-872`), or — for non-deferred cleanup — directly releases KV via `release_kv_cache(req, self.tree_cache)` (`:890-893`) and strips the rid from `running_batch`/`cur_batch`/`last_batch`/async-pending (`:847-852`).

### 3.4 SimpleScheduler — batching for non-AR stages

Non-AR stages (preprocess, encoders) get a deliberately small scheduler: "No KV cache... Just: inbox.get() → run function → outbox.put()" (`simple_scheduler.py:4-5`), but with **three-dimensional batching**: max batch size, max wait time, and a per-request *cost budget* (`_collect_batch`, `simple_scheduler.py:101-130`, verified):

```python
while len(batch) < self._max_batch_size:
    ... msg = self.inbox.get(timeout=remaining)   # wait up to max_batch_wait_ms
    if msg.type == "new_request":
        if self._max_batch_cost is not None:
            msg_cost = self._message_cost(msg)
            if batch and batch_cost + msg_cost > self._max_batch_cost:
                self._pending_messages.appendleft(msg); break
```

The Qwen3-Omni image encoder uses this with a **bytes-denominated activation-cost model** (`models/qwen3_omni/stages.py:844-851`: `max_batch_size=32, max_batch_wait_ms=50, request_cost_fn=..., max_batch_cost=10GiB×` with a `×5` activation multiplier, `:47-48`), plus a CPU LRU output cache and same-batch dedup of identical media (`_batch_image_encoder_payloads`, `:368-571`; `StageOutputCache`, `scheduling/stage_cache.py`). The audio encoder pads-and-concats variable-length features for batched forward (`:631-737`). `StreamingSimpleScheduler` adds stream-chunk lifecycle for vocoder-type stages (`streaming_simple_scheduler.py:1-40`).

### 3.5 Cross-stage scheduling dependencies: the talker

The talker AR stage consumes the thinker's **per-token stream** (token ids + hidden states), creating scheduling couplings vanilla SGLang doesn't have. `QwenTalkerScheduler` (subclass of OmniScheduler) adds:

1. **Deferred admission / partial start** (`models/qwen3_omni/talker_scheduler.py:70-84`): a talker request only becomes buildable once the thinker is done (`pending_stream_done`) or — with `enable_partial_start` — once ≥ N usable chunks have been prefetched.
2. **Decode-readiness gating with KV rollback** (`:110-143`, verified): if the model runner reports the feedback/text inputs for this decode step aren't ready, the already-`prepare_for_decode`-ed batch is *undone* — the allocated KV slot freed and seq_lens decremented:

```python
def get_next_batch_to_run(self) -> Any | None:
    batch = _Upstream.get_next_batch_to_run(self)
    if batch is not None and not self._is_batch_ready_to_run(batch):
        self._rollback_decode_prep_after_skip(batch)   # free out_cache_loc, seq_lens.sub_(1), ... (:117-143)
        return None
    return batch
```

The rollback also zeroes the `req_to_token_pool` cell that `alloc_for_decode` wrote, and hard-fails with a `TypeError` if upstream's `prepare_for_decode` invariants change ("sglang upstream prepare_for_decode changed; update rollback", `:128-130`).

3. Talker server-args policy: radix cache off, chunked prefill off, overlap off when feedback is enabled (`configure_talker_server_args`, `:29-36`).

### 3.6 Memory / cache management as it interacts with scheduling

**Paged KV + radix cache are upstream's, instantiated per AR stage.** `create_sglang_infrastructure` (`scheduling/bootstrap.py:9-84`) builds an omni `ModelWorker` (which wraps the upstream `ModelRunner` after applying sub-model arch overrides, `model_runner/model_worker.py:34-115`), then pulls the upstream pools straight from it — `model_worker.get_memory_pool()` returns `(model_runner.req_to_token_pool, token_to_kv_pool_allocator)` (`model_worker.py:128-131`) — and wraps them in a tree cache (`scheduling/sglang_backend/cache.py:9-33`):

```python
if server_args.disable_radix_cache:
    return ChunkCache(params)     # plain KV semantics, no prefix matching
return RadixCache(params)
```

So each AR stage (thinker, talker) has its **own independent paged KV pool, radix prefix cache, and mem_fraction budget** on its GPU. Colocation is governed by an explicit per-stage memory contract: `total_gpu_memory_fraction` minus an `encoder_mem_reserve` becomes the SGLang `mem_fraction_static`, with conflict validation (`models/qwen3_omni/stages.py:74-148`); placement planning enforces per-GPU fraction sums (`config/schema.py:104-117`, `config/placement.py`).

Scheduling↔memory interaction points:

- **Admission**: a request that cannot ever fit (`input + max_new_tokens > kv_capacity`) is rejected before entering the waiting queue, with a `--thinker-mem-fraction-static` hint (`omni_scheduler.py:567-589`).
- **Prefill packing**: upstream `PrefillAdder` decides token budgets against the radix-cache hit and the allocator's free pages; the standalone `PrefillManager` mirrors this for custom (Fish) schedulers and adds an omni-specific rule — requests with *projected input embeddings* disable chunked prefill entirely (`scheduling/sglang_backend/prefill.py:63-86`: `rem_chunk_tokens=(None if disable_chunking else self.chunked_prefill_size)`), because a projected-embeds prompt can't be split across rounds.
- **Eviction/retraction**: on `check_decode_mem()` failure the running batch is retracted (`retract_decode`), freed tokens are logged, the `new_token_ratio` is raised to be more conservative, and retracted reqs re-enter the prefill waiting queue (`scheduling/sglang_backend/decode.py:36-78`; for OmniScheduler the equivalent upstream path runs via composition with the `new_token_ratio` state at `omni_scheduler.py:225-237`).
- **Abort / failure**: KV released through upstream `release_kv_cache` (`omni_scheduler.py:890-893`); async-decode adds the double-free guards described in §3.3.
- **Talker rollback** un-allocates one decode token per skipped step (§3.5).
- **Non-KV caches**: encoder-output CPU LRU (`StageOutputCache`, bytes+entries capped, `stages.py:50-52`, `stage_cache.py`) sits in front of the encoder batcher, so the scheduler dedups identical media both across requests (cache hit) and within one batch (`dedup_same_batch`, `stages.py:413-423`).

**Inter-stage data plane is also memory management**: stage hops move tensors out of the payload, pack them into one aligned uint8 buffer (with explicit dtype-alignment padding), and ship via relay while only pickled metadata goes over ZMQ (`pipeline/relay_io.py:93-145`, verified); the default `shm` relay does a single memcpy into POSIX shared memory (`relay/shm.py:18-35`), and same-GPU stream chunks bypass the relay with CUDA IPC handles (`relay_io.py:315-353`, `stage/runtime.py:392-411`).

---

## 4. Distributed orchestration (Ray or alternative)

### Not Ray: multiprocessing(spawn) + ZMQ-over-IPC + per-stage NCCL

**Ray is not used anywhere** — `grep -rn "^import ray|^from ray"` over the repo returns zero hits (verified in this checkout); the only "ray" substrings are words like "arrays".

**Process spawning.** `MultiProcessPipelineRunner.start()` uses `multiprocessing.get_context("spawn")` (`pipeline/mp_runner.py:366`), builds `StageGroup`s, and spawns each `StageWorkerProcessSpec` as a daemon process with entrypoint `stage_process_main`:

```python
proc = ctx.Process(
    target=stage_process_main,
    args=(spec, event, startup_error_channel),
    name=proc_name, daemon=True,
)
```
(`pipeline/stage_workers.py:230-235`). Readiness is a per-process `ctx.Event()` plus a `ctx.Queue()` startup-error channel carrying the child traceback (`stage_workers.py:253-292`). After startup, `_monitor_children` polls every 5 s and on any dead child fails all pending coordinator requests and stops the runner — fail-fast, no respawn (`mp_runner.py:439-457`, verified: `await self._fail_runtime(error)` → `coordinator.fail_pending_requests` → `stop()`).

**Placement & topology planning** is two-phase and declarative. `prepare_pipeline_runtime` runs `config.apply_fusion()` (optional merging of adjacent non-TP stages, validated linear/same-GPU, `config/schema.py:368-426`), then `build_stage_placement_plan` (GPU assignment + memory-fraction budgets per GPU, `config/placement.py:16-50`) and `build_process_topology_plan` ("which non-TP stages should run in the same OS process?", `config/topology.py:3-58`), then allocates ZMQ endpoints (`pipeline/runtime_config.py:74-113`). Multiple `role="single"` stages may share one OS process and one asyncio event loop (shared failure domain, §2). Placement-derived fast paths (`same_gpu_targets`, `same_process_targets`) are resolved at build time (`mp_runner.py:75-84, 168-195`).

**Control plane = ZMQ over per-run unix sockets.** All endpoints are `ipc://` paths under a per-run temp dir — i.e. **the pipeline runtime is single-node by construction** (`pipeline/runtime_config.py:160-173`, verified):

```python
endpoints = {
    "completion": f"ipc://{base_dir}/completion.sock",
    "abort": f"ipc://{base_dir}/abort.sock",
}
for stage in stages:
    endpoints[f"stage_{stage.name}"] = f"ipc://{base_dir}/stage_{stage.name}.sock"
```

Transport is PUSH/PULL with msgpack serialization (`pipeline/control_plane.py:26-39, 80-137`); abort uses PUB/SUB. Wire messages are hand-written `to_dict`/`from_dict` dataclasses, not pickle/pydantic (`proto/messages.py:28-124`).

**Data plane = pluggable relay.** `PipelineConfig.relay_backend: Literal["shm", "nccl", "nixl", "mooncake"] = "shm"` (`config/schema.py:212`), behind `RELAY_REGISTRY`/`create_relay` (`relay/base.py:14-60` — ABC + `@register_relay` decorator registry, with a factory that introspects `__init__` signatures to filter kwargs, `relay/base.py:14-75`). Senders extract tensors out of a `StagePayload`, concatenate them into one aligned `uint8` buffer, `relay.put_async()` it, and ship pickled tensor-free payload + tensor offsets inside the `DataReadyMessage` (`pipeline/relay_io.py:93-147`); the control message is deliberately sent **before** waiting for put completion so credit-based backends (NIXL) don't deadlock (`runtime.py:852-868`; rationale in `docs/developer_reference/communication.md`). Backends: `shm` copies through POSIX shared memory (`relay/shm.py:18-34`); `nccl` builds its own `dist.init_process_group("nccl", rank=..., world_size=...)` with explicit send/recv rank topology (`relay/nccl.py:25-66`). Fast paths bypass relay: same-process edges pass Python references via `LocalStageDispatcher` (`stage_workers.py:381-386`, dispatch gated by mutable-container isolation checks at `runtime.py:870-896`), same-GPU stream edges use CUDA IPC via `ForkingPickler` (`runtime.py:392-412`).

**Tensor parallelism is per-stage, one process per rank.** Each TP stage gets a freshly probed NCCL port (`mp_runner.py:66-73, 305-321`) and per-rank `StageLaunchConfig`s with roles leader/follower; followers receive work through `ctx.Queue()`s created before spawn (`mp_runner.py:235-285`). A hard invariant: "a TP stage (`tp_size > 1`) must own its OS process exclusively" (`stage_workers.py:118-131`). Spawn-time env pins each rank to a single device:

```python
return {
    "CUDA_VISIBLE_DEVICES": mapped_gpu,
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
}
```
(`stage_workers.py:654-658`). Only TP rank 0 owns external ZMQ IO; the leader replicates work/abort/shutdown to followers through `TPLeaderFanout` over mp queues so "NCCL collectives in TP-parallel forward passes do not deadlock" (`pipeline/tp_control.py:1-71`), and followers run a queue-backed `TPFollowerControlPlane` (`tp_control.py:74-132`). Inside `OmniScheduler`, TP request distribution reuses sglang's `broadcast_pyobj` over the TP CPU group: rank 0 drains the inbox, then broadcasts to all ranks (`omni_scheduler.py:434-444`). The coordinator "only talks to rank 0. Peer ranks stay internal to the stage group" (`docs/developer_reference/pipeline.md`, matching `StageGroup.stage_control_endpoints` filtering on `owns_external_io`, `stage_workers.py:213-218`).

**Weight sync:** none — this is an inference-only serving system; each stage loads its own weights at startup (serialized per GPU by `gpu_startup_lock`, `stage_workers.py:608-621`). There is no trainer↔engine weight transfer path.

**Multi-node / scale-out** is handled one level up by `sglang_omni_router`: a capability-aware HTTP proxy with `round_robin` / `least_request` / `random` policies (`sglang_omni_router/selector.py:42-55`), worker health tracking (`worker.py:34-101`), and an optional `LocalLauncher` that spawns whole `sgl-omni serve` server processes as workers (`sglang_omni_router/launcher/local.py:37-57`). So the topology is: router → N independent single-node pipeline servers → per-server stage process trees.

---

## 5. Code organization style: function granularity

The codebase has a clear two-speed pattern: **control loops and compat shims stay monolithic with heavy inline commentary; anything called from ≥2 sites or crossing a thread/TP boundary gets extracted.**

**Deliberately monolithic #1 — `OmniScheduler.__init__`, ~243 lines** (`scheduling/omni_scheduler.py:83-325`). One flat sequential block initializing ~80 attributes (batch state, chunked prefill, schedule policy, feature flags, stubs), grouped only by banner comments (`# --- Core scheduling state (read/written by upstream methods) -----`, `# Subsystem stubs`, `# Disaggregation / hybrid (disabled)`). The justification is structural: every field exists because an *upstream* method bound via `__getattr__` will read it, so the init is effectively a field-for-field mirror of upstream `Scheduler.__init__` — splitting it would hide the mirroring. Only two pieces are extracted: `_init_upstream_compat_flags` (`:327-345`) and `_init_parallel_state` (`:386-415`), both because they are self-contained upstream-compat sub-protocols, not for line count.

**Deliberately monolithic #2 — `_event_loop_async_decode`, ~83 lines** (`scheduling/omni_scheduler.py:1041-1123`). The one-step-lookahead decode loop keeps launch, resolve-previous, flush-on-prefill, and the stale-batch double-free guard in a single function, because the ordering invariants only make sense read top-to-bottom. Roughly a third of the body is WHY-comments explaining failure modes (the stale-batch comment quoted in §3.3, `:1094-1103`).

**Extracted small helpers — and what justified each:**
- `recv_requests` → `_recv_scheduler_messages` → `_drain_local_inbox` (`omni_scheduler.py:417-453`): three layers, each ≤18 lines, where each boundary is real — message-type dispatch / TP rank-0 `broadcast_pyobj` fanout / raw queue drain.
- `run_batch` is a 6-line error-boundary wrapper over `_run_batch` with the `_FAILED_BATCH_RESULT` sentinel (`omni_scheduler.py:37, 602-607`).
- `_resolve_pending_async` (`:1028-1039`) exists because the in-flight lookahead step must be flushed from **three** call sites (pause, prefill transition, loop body) — "flush before prefill / pause / shutdown so a launched step is never stranded".
- The sync/async split in `ModelRunner` is the cleanest example of disciplined extraction: `execute()` documents it is "byte-identical to the pre-async implementation: a pure extraction over the same shared sub-steps (`_build_forward_batch` / `_prepare_and_forward` / `_finalize`) that `execute_launch` + `execute_resolve` also use, in the same order" (`model_runner/base.py:93-102`). They extracted shared sub-steps **only when the async variant forced it**, and recorded the equivalence claim in the docstring.

`pipeline/stage/runtime.py` (1257 lines, single `Stage` class, ~50 methods) shows the complementary norm: a big class is fine, but each method is a small message handler (`_on_submit`, `_on_data_ready`, `_on_stream_chunk`, `_on_stream_signal` — `runtime.py:250-264` dispatch) routed from one `run()` loop (`runtime.py:200-248`).

**Types and protocols.** Plain `@dataclass` everywhere for internal state and wire messages (121 `@dataclass` uses outside vendor), with hand-written `to_dict`/`from_dict` on every proto message instead of pickle/pydantic (`proto/messages.py:28-124` — including explicit `_type` markers and import-guarded fallbacks for metadata variants). Pydantic `BaseModel` is reserved for user-facing config schema (`config/schema.py`: `PipelineConfig`, `StageConfig`, `RelayConfig`, ...). Enums are rare (5 in the whole non-vendor tree, e.g. `SchedulerStatus(Enum): WAITING/RUNNING/FINISHED/ABORTED`, `scheduling/types.py:14-18`). `Protocol` appears exactly once (`PlacementPolicy`, `config/placement.py:40-41`); plugin boundaries that need a registry use ABC + decorator instead — `@register_relay("nccl")` filling `RELAY_REGISTRY`, with a `create_relay` factory that introspects `__init__` signatures to filter kwargs (`relay/base.py:14-75`).

**Comment style — the most distinctive trait of this repo:**
1. Every file opens with `# SPDX-License-Identifier: Apache-2.0` plus a one-line role docstring of the form "Name — role" ("Stage — IO shell for pipeline processing.", `pipeline/stage/runtime.py:2`).
2. **Attributed comments**: `# Note (Chenyang):`, `# Note (Xuesong):`, `# FIXME (Ratish):` — 11 occurrences, always carrying a measured fact or a dated decision, e.g. the GIL note on the event loop: "the audio_encoder forward pass ... slows ~600x, dropping audio QPS from >10 to <0.5" (`omni_scheduler.py:896-901`, verified), or the seq-len bump rationale "bumped 8192 → 32768 because the V1 talker prefill replays the full thinker prompt ... ~22K positions" (`models/qwen3_omni/config.py:160-164`).
3. Docstrings state **ownership and invariants**, not behavior: "The CALLER owns the handle ... launch-first scheduling has two steps momentarily in flight" (`model_runner/base.py:133-137`), "Invariant: at most one `_PendingStep` is live at a time" (`model_runner/base.py:29-30`).
4. Comments cross-reference design docs by section (`design.md §1.3/§1.4`, `model_runner/base.py:33` and verified at `base.py:75`, `omni_scheduler.py:1049`) — though note `design.md`/`benchmark_results.md` are **not present in this checkout** (`find` returns nothing; `docs/developer_reference/` has `pipeline.md`, `communication.md` etc.) — a dangling-reference anti-pattern worth not copying.

---

## 6. Naming conventions

- **Module names are plain role nouns**, no decoration: `omni_scheduler.py`, `simple_scheduler.py`, `streaming_simple_scheduler.py`, `threaded_simple_scheduler.py` (capability prefixes stack left of the base noun), `stage_workers.py`, `mp_runner.py`, `control_plane.py`, `relay_io.py`, `stage_cache.py` (all under `sglang_omni/scheduling/` and `sglang_omni/pipeline/`).
- **Class suffixes encode runtime role**: `*Scheduler` (compute loop owner), `*ModelRunner`/`SGLModelRunner` (`model_runner/base.py:46`, `model_runner/sglang_model_runner.py`), `ModelWorker` + `ModelWorkerConfig` (`model_runner/model_worker.py`), `*ControlPlane` (`pipeline/control_plane.py`: `StageControlPlane`, `CoordinatorControlPlane`), `*Relay` per transport (`relay/base.py:45-52`: `NcclRelay`, `ShmRelay`, `NixlRelay`, `MooncakeRelay`), `*Message` for every wire type (`proto/messages.py`: `DataReadyMessage`, `CompleteMessage`, `StreamMessage`, `SubmitMessage`, `ShutdownMessage` — event-named, not verb-imperative), `*Config`/`*Plan`/`*Planner` in config land (`config/placement.py:44`: `StagePlacementPlanner`).
- **Private upstream-compat artifacts get a leading underscore even at module level**: `_NoOpSender`, `_NoOpGrammarManager` (`omni_scheduler.py:40-65`), `from ... import Scheduler as _Upstream` (`omni_scheduler.py:27`), `_FAILED_BATCH_RESULT`, `_PendingStep` (`model_runner/base.py:21`).
- **A routing vocabulary of snake_case verb functions referenced by dotted strings in config**: factories `create_*_executor` / `create_*_scheduler`, projections `project_<src>_to_<dst>` (`project_thinker_to_decode`, `project_talker_to_code2wav`), routers `resolve_*_next_stages`, mergers `merge_for_thinker` — all visible in one stage declaration, `models/qwen3_omni/config.py:108-134` (`factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config"`, `project_payload={"decode": f"{_PKG}.request_builders.project_thinker_to_decode"}`). Dotted paths are resolved at spawn by `StageLaunchConfig` (`pipeline/stage_workers.py:30-62`).
- **Uniform per-model file set** (cross-family grepability): `config.py`, `stages.py`, `request_builders.py`, `payload_types.py`, `model_runner.py`, `bootstrap.py` across 8+ model packages. Each `config.py` ends with the same registration idiom: `EntryClass = Qwen3OmniSpeechPipelineConfig` + `Variants = {"text": ..., "speech": ..., "speech-colocated": ...}` (`models/qwen3_omni/config.py:345-351`).
- **The inbox/outbox mailbox metaphor is the cross-scheduler contract name**: "Public contract (used by Stage): `inbox`, `outbox`, `start()`, `stop()`, `abort(request_id)`" (`omni_scheduler.py:71-72`), mirrored by `SimpleScheduler` ("Same inbox/outbox interface as OmniScheduler so Stage doesn't need branching", `scheduling/simple_scheduler.py:7`), with `IncomingMessage`/`OutgoingMessage` as the queue item types (`scheduling/messages.py:10-23`).

---

## 7. End-to-end flow trace

Streaming `POST /v1/chat/completions` on Qwen3-Omni (text + speech out). Pipeline topology (declared, not coded): `preprocessing → {image_encoder, audio_encoder, mm_aggregate}`; `mm_aggregate` fan-in (`wait_for=["preprocessing","image_encoder","audio_encoder"]`) `→ {thinker, talker_ar}`; thinker `next="decode"`, `stream_to=["talker_ar","decode"]`; `talker_ar → code2wav` (also `stream_to`); terminals = `decode` (text) and `code2wav` (audio) (`models/qwen3_omni/config.py:26-189`).

1. **Server boot**: `launch_server` → `_run_server` starts `MultiProcessPipelineRunner` (spawns stage processes per the placement/topology plan) and gets the in-process `Coordinator`; uvicorn runs in the same parent process with a failure watcher racing the server against `mp_runner.wait_failed()` — `serve/launcher.py:304-355, 358-390`. Each stage process constructs its stages and schedulers and starts serving — `pipeline/stage_workers.py:362-410`.
2. **HTTP entry**: `POST /v1/chat/completions` → `chat_completions` builds a `GenerateRequest`; `stream=True` returns a `StreamingResponse(_chat_stream(...))` — `serve/openai_api.py:159-201, 275-291`.
3. **Client → Coordinator**: `Client.completion_stream` → `Client.generate` → `coordinator.stream(req_id, omni_request)` — `client/client.py:52-67, 132-144`.
4. **Admission**: `Coordinator.stream` opens a per-request `asyncio.Queue` and calls `_submit_request`, which creates the completion future, wraps inputs in a `StagePayload`, resolves the request's terminal set, tracks state PENDING→RUNNING, and sends `SubmitMessage` over ZMQ to the entry stage (`preprocessing`) — `pipeline/coordinator.py:149-179, 182-239`.
5. **Entry stage receives**: `Stage.run` loop `await self.control_plane.recv()` → `_handle_message` → `_on_submit` → `_execute` puts `IncomingMessage(type="new_request")` into the preprocessing `SimpleScheduler.inbox` — `pipeline/stage/runtime.py:213-229, 265-281, 639-654`.
6. **Preprocess + encoders**: `SimpleScheduler._start_serial` pops the message, runs the compute fn (tokenize, fetch media) — `scheduling/simple_scheduler.py:214-242`; result goes to outbox; `Stage._drain_outbox_external` → `_route_result` → `_send_to_stage` for each of `next=["image_encoder","audio_encoder","mm_aggregate"]`: tensors written to the relay (`relay_io.write_payload`, `pipeline/relay_io.py:93-145`) and a `DataReadyMessage` sent via ZMQ, control-before-wait — `stage/runtime.py:666-707, 728-788, 852-868`. Encoder stages batch arrivals under size/wait/cost limits (`simple_scheduler.py:101-130`; `models/qwen3_omni/stages.py:844-851`) and run a single fused `model(**batched_inputs)` forward (`stages.py:497-498`).
7. **Fan-in**: `mm_aggregate`'s `InputHandler.receive` merges the three payloads; when complete, `_receive_payload_from_stage` dispatches the merged payload — `stage/runtime.py:353-382` — then routes to both `thinker` and `talker_ar` (`config.py:89`). The talker copy sits **deferred** in its scheduler until thinker chunks arrive (`talker_scheduler.py:70-84`).
8. **Thinker admission**: thinker `Stage._execute` → `OmniScheduler.inbox`; the scheduler thread's `_event_loop_normal` picks it up: `recv_requests` (`omni_scheduler.py:417-432`) → `process_input_requests` builds the SGLang `Req` via `make_thinker_scheduler_adapters` and appends to `waiting_queue` (`:455-519`; builder wired in `models/qwen3_omni/bootstrap.py:82-101`).
9. **Batch selection**: `self.get_next_batch_to_run()` — upstream SGLang `Scheduler` method bound via `__getattr__` (`omni_scheduler.py:360-384`, called at `:910`) — performs radix-prefix matching, chunked-prefill packing, and decode/prefill interleaving against this stage's KV pool.
10. **Model forward**: `run_batch` → `ThinkerModelRunner.execute` → `_build_forward_batch` (upstream `ForwardBatch.init_new`, `model_runner/base.py:212-248`) → `custom_prefill_forward` injects image/video/audio embeddings + deepstack embeds in place of placeholder tokens (`model_runner/thinker_model_runner.py:65-76, 102-116`) → otherwise upstream `tp_worker.forward_batch_generation` (`base.py:277`) → sampling with repetition-penalty/suppress-token hooks (`base.py:476-545`).
11. **Per-token streaming out**: same loop iteration, `_emit_stream_output` runs `make_thinker_stream_output_builder`, emitting the token id (text path) and token+hidden-state chunks (talker path) to the outbox (`omni_scheduler.py:638-662`; `models/qwen3_omni/request_builders.py:804-864`). `Stage._drain_outbox_external` sends each chunk to `stream_to=["talker_ar","decode"]` — CUDA-IPC if same-GPU, relay blob + `DataReadyMessage` otherwise (`stage/runtime.py:966-1030`; `relay_io.py:315-353`).
12. **Talker decode**: talker stage routes incoming chunks into `QwenTalkerScheduler.inbox` as `stream_chunk` messages (`stage/runtime.py:634-637`); the scheduler appends them to the request's `pending_text_queue` (`talker_scheduler.py:152-158`), un-defers the request once enough chunks exist (`omni_scheduler.py:535-544` + `talker_scheduler.py:70-84`), and generates codec tokens, gating each decode step on feedback/text availability with KV rollback when not ready (`talker_scheduler.py:110-143`). Codec chunks stream to `code2wav`.
13. **Terminal emission**: `decode` (streaming detokenizer, `stages.py:922-923`) and `code2wav` are `terminal=True`; their stages forward untargeted stream chunks to the coordinator as `StreamMessage`s (`stage/runtime.py:1071-1124`) and, on finish, send `CompleteMessage` via `control_plane.send_complete` (`stage/runtime.py:751-767`).
14. **Fan-in at coordinator, response**: `run_completion_loop` routes `StreamMessage`s into the request's queue (`coordinator.py:289-305, 398-423`); `Coordinator.stream` yields them and returns only when **all** expected terminal stages (`{decode, code2wav}` here) have completed (`:160-179`); multi-terminal results are merged in `_handle_completion` (`:374-396`). `Client.generate` converts each to a `GenerateChunk` (`client/client.py:58-64`) and `_chat_stream` serializes SSE deltas (text + base64 PCM audio) back to the HTTP caller (`serve/openai_api.py:275-291`).

Abort path: client disconnect / explicit abort → `Coordinator.abort` broadcasts `AbortMessage` on the abort ZMQ channel (`coordinator.py:241-287`); every stage's `_abort_listener` (`stage/runtime.py:1176-1187`) calls `scheduler.abort(rid)`, which in the AR scheduler releases KV and scrubs all queues/batches (`omni_scheduler.py:824-853`).

---

## 8. Ideas worth borrowing for wm-infra

1. **Adopt the 5-member mailbox contract between EngineLoop and per-phase executors.** sglang-omni runs wildly heterogeneous compute (SGLang AR scheduler, encoder batcher, streaming vocoder) behind one shape — `inbox: Queue[IncomingMessage]`, `outbox: Queue[OutgoingMessage]`, `start/stop/abort` (`omni_scheduler.py:68-81`, `simple_scheduler.py:25-46`) — so `Stage` (the IO shell) never branches on scheduler type. wm-infra's `VideoIterationRunner` phases and a future GRPO-rollout consumer could share exactly this contract; it is smaller and more decoupled than a method-per-phase Protocol, and makes thread/process placement a deployment detail.
2. **Launch/resolve split for denoise-step lookahead.** The `execute_launch`/`execute_resolve` + `_PendingStep` pattern (`model_runner/base.py:20-43, 126-160`; consumed at `omni_scheduler.py:1041-1123`) is directly transplantable to continuous-batched diffusion: launch step N's UNet/DiT forward + async D2H of whatever the host needs, resolve step N-1's host-side bookkeeping while N runs. Three details to copy verbatim: (i) keep the sync path as a *pure extraction over the same sub-steps* so the fallback is byte-identical (`base.py:97-102`); (ii) ping-pong **two** pinned host buffers because resolve(N) reads while launch(N+1) writes — single-stream ordering does not protect CPU-read vs GPU-write (`base.py:69-91`); (iii) gate lookahead on a `min_batch_size` because at bs=1 the fixed overhead is a net loss (`omni_scheduler.py:133-140`).
3. **Declare model-family pipeline topology as data, with dotted-path factories.** A new model in sglang-omni is a `config.py` returning `StageConfig(name=..., factory="pkg.stages.create_x_executor", next=..., route_fn=..., project_payload={...})` (`models/qwen3_omni/config.py:108-134`) plus an `EntryClass`/`Variants` registration (`config.py:345-351`) — the engine never changes. wm-infra currently hardcodes the ENCODE→DENOISE→DECODE phase chain in the runner; moving per-family phase graphs (and per-edge payload projections like `project_thinker_to_decode`) into the family package would make I2V/AR/cosmos variants additive instead of branchy.
4. **Error boundaries via a sentinel object, not exceptions through the loop.** `_FAILED_BATCH_RESULT = object()` + a 6-line `run_batch` wrapper that converts any batch exception into per-request abort+error emission (`omni_scheduler.py:37, 602-607, 710-716`) keeps the event loop body linear. wm-infra's `_advance()` per-phase execution would benefit from the same single choke point instead of try/except per phase.
5. **Composition-over-inheritance to reuse a scheduler engine without forking it.** `__getattr__` + `types.MethodType` binding upstream `Scheduler` methods onto a hand-mirrored state surface (`omni_scheduler.py:360-384, 83-325`) buys the full upstream scheduling MRO (radix cache, chunked prefill, retraction, priority) while replacing only the IO edges — at the explicit cost of a hard version pin (`sglang==0.5.8`) and a defensive `TypeError` tripwire where upstream invariants are assumed (`talker_scheduler.py:128-130`). If wm-infra ever wraps sglang/vLLM internals, this pattern (plus the tripwire) is the disciplined way to do it.
6. **Copy the comment culture, not just the code**: attributed `Note (name):` comments with measured numbers (600x GIL, bs=1 regression, 22K-position overflow) turn tuning constants and threading hacks into auditable decisions (`omni_scheduler.py:896-901`, `models/qwen3_omni/config.py:158-164`). For wm-infra's batch-size/compile-mode constants (the torch.compile-vs-LoRA tradeoffs already in project memory), this is the cheapest possible provenance system. But keep design-doc references pointing at files that exist in-repo (sglang-omni's `design.md` references dangle).
7. **Uniform per-family file vocabulary.** sglang-omni's `config.py / stages.py / request_builders.py / payload_types.py / model_runner.py / bootstrap.py` set, repeated across 8 model packages, matches wm-infra's `models/families/<name>/` instinct (e.g. `wan/state.py`) — worth formalizing the canonical filename set so cross-family grep stays trivial.
8. **Cost-budget batching for non-AR stages.** `SimpleScheduler._collect_batch`'s three-dimensional batching (max size, max wait, per-request byte-cost budget with a measured activation multiplier, `simple_scheduler.py:101-130`, `stages.py:844-851`) is exactly the shape wm-infra's VAE-encode/decode phases need: resolution-heterogeneous requests have wildly different activation footprints, and a `request_cost_fn` + budget beats a fixed batch size.
9. **Talker-style decode gating with KV rollback** (`talker_scheduler.py:110-143`) is prior art for any cross-stage data dependency inside an AR loop — for wm-infra, an AR world-model step that must wait on an external conditioning stream (action tokens, reward feedback) can adopt "build the batch optimistically, roll back the one-token allocation if inputs aren't ready" instead of stalling the whole loop.

---

## 9. Source-of-truth index

| Claim | Evidence |
|---|---|
| Layer on top of sglang (pip dep), not a fork | `pyproject.toml:29` (`sglang==0.5.8`); `sglang_omni/vendor/sglang/core.py:1-21` (static re-export wrapper) |
| nixl / mooncake / pyzmq dependencies | `pyproject.toml:13, 30-31` |
| Computation-centric multi-stage design, per-stage scheduler, shared inbox/outbox, zero-copy shm | `README.md` (About, lines 17-21) |
| Package map / directory ownership | `docs/developer_reference/main.md` (Directory Layout); verified via `ls` of `sglang_omni/*` |
| No Ray anywhere | grep `^import ray|^from ray` over repo: 0 hits (verified) |
| Runner spawns stage processes via mp spawn ctx, daemon procs, readiness events, error channels | `pipeline/mp_runner.py:361-433`; `pipeline/stage_workers.py:224-292` |
| Child monitor: poll 5 s, fail-all on dead child, no respawn | `pipeline/mp_runner.py:439-457` (verified) |
| One process hosts many stages, one asyncio loop, fail-together | `pipeline/stage_workers.py:362-410, 367-380` |
| Coordinator: entry-stage submit, terminal collection, abort broadcast, multi-terminal merge | `pipeline/coordinator.py:26-91, 149-239, 241-305, 374-423` |
| ZMQ PUSH/PULL + msgpack control plane; abort PUB/SUB | `pipeline/control_plane.py:26-39, 80-137` |
| `ipc://` per-run endpoints ⇒ single-node control plane | `pipeline/runtime_config.py:160-173` (verified) |
| Fusion → placement plan → process topology plan pipeline | `pipeline/runtime_config.py:74-113`; `config/schema.py:368-428`; `config/placement.py:16-50`; `config/topology.py:3-58` |
| Stage = IO shell; scheduler in dedicated thread; inbox/outbox bridging | `pipeline/stage/runtime.py:146-263, 639-654, 666-707` |
| Relay payload format (tensor extraction, one aligned uint8 buffer, metadata in control msg) | `pipeline/relay_io.py:93-147` (verified) |
| Control-before-wait send ordering (NIXL credits) | `pipeline/stage/runtime.py:852-868`; `docs/developer_reference/communication.md` |
| Relay backends + registry; default `shm` | `relay/base.py:14-75`; `config/schema.py:212`; `relay/shm.py:18-35`; `relay/nccl.py:25-66` |
| LOCAL_OBJECT same-process fast path + isolation checks | `pipeline/stage_workers.py:381-386`; `pipeline/stage/runtime.py:810-850, 870-896` |
| Same-GPU CUDA-IPC stream fast path | `pipeline/stage/runtime.py:392-412, 617-632`; `pipeline/relay_io.py:315-353` |
| OmniScheduler composition via `__getattr__` unbound binding | `scheduling/omni_scheduler.py:1-12, 360-384` (verified verbatim) |
| Upstream scheduling state hand-initialized (queues, policy, ratios, priority/preemption) | `scheduling/omni_scheduler.py:83-325 (180-237)` |
| Upstream features stubbed (grammar, detok, spec, disagg, LoRA) | `scheduling/omni_scheduler.py:40-65, 240-294` |
| TP intake via `broadcast_pyobj` from entry rank | `scheduling/omni_scheduler.py:417-453` |
| Request build, deferral, KV-capacity admission, waiting_queue | `scheduling/omni_scheduler.py:455-519, 535-544, 567-589` |
| Normal event loop (incl. GIL-yield 600x co-location note) | `scheduling/omni_scheduler.py:895-923` (verified verbatim) |
| Overlap loop | `scheduling/omni_scheduler.py:925-968` |
| run_batch bridge → GenerationBatchResult; per-step stream emit; finish intercept; no-op detok | `scheduling/omni_scheduler.py:602-677, 638-662, 736-786` |
| Error-boundary sentinel `_FAILED_BATCH_RESULT` | `scheduling/omni_scheduler.py:37, 602-607` |
| Async-decode lookahead loop + overrun/double-free guards + bs=1 gate | `scheduling/omni_scheduler.py:1041-1123, 991-1039, 126-140` |
| Launch/resolve split, pinned ping-pong buffers, CUDA event, query-hit counters | `model_runner/base.py:20-44, 60-91 (verified), 126-210` |
| "byte-identical pure extraction" sync/async split | `model_runner/base.py:93-124` |
| Sampling penalties / hooks | `model_runner/base.py:250-294, 476-545` |
| Abort: FINISH_ABORT marking + KV release + batch scrub | `scheduling/omni_scheduler.py:824-893` |
| SimpleScheduler size/wait/cost batching | `scheduling/simple_scheduler.py:33-66, 101-130 (verified), 214-242` |
| StreamingSimpleScheduler for vocoder-type stages | `scheduling/streaming_simple_scheduler.py:1-40` |
| Image-encoder cost model, LRU cache, same-batch dedup | `models/qwen3_omni/stages.py:47-52, 230-252, 368-571, 844-851` |
| Talker partial-start admission + decode gating + KV rollback + upstream tripwire | `models/qwen3_omni/talker_scheduler.py:18-36, 70-84, 110-143 (verified)` |
| Per-stage SGLang infra (worker, pools, tree cache, managers); sub-config extraction | `scheduling/bootstrap.py:9-84`; `model_runner/model_worker.py:24-131` |
| RadixCache vs ChunkCache factory | `scheduling/sglang_backend/cache.py:9-33` |
| PrefillManager: chunking disabled for projected embeds | `scheduling/sglang_backend/prefill.py:49-147 (63-86)` |
| DecodeManager: retraction on KV full → re-queue to prefill; used by Fish scheduler | `scheduling/sglang_backend/decode.py:27-83`; `scheduling/bootstrap.py:70-74`; `models/fishaudio_s2_pro/fish_scheduler.py:36-82` |
| Colocated memory contract (fractions, encoder reserve) | `models/qwen3_omni/stages.py:74-148`; `config/schema.py:104-117` |
| TP: one process per rank, exclusive-process invariant, leader-only external IO | `pipeline/mp_runner.py:124-144, 224-287`; `pipeline/stage_workers.py:118-131, 213-218` |
| CUDA_VISIBLE_DEVICES remap per TP rank | `pipeline/stage_workers.py:631-658` |
| TP leader→follower fanout over mp queues (NCCL deadlock avoidance) | `pipeline/tp_control.py:1-132`; `pipeline/stage/runtime.py:646-651` |
| NCCL port allocation per TP stage | `pipeline/mp_runner.py:66-73, 305-321` |
| Per-GPU serialized weight loading; no weight-sync path | `pipeline/stage_workers.py:608-621` |
| Qwen3-Omni 8-stage topology (fan-out, fan-in, streams, dual terminals) | `models/qwen3_omni/config.py:24-238`; topology fields `config/schema.py:120-237` |
| HTTP server + Coordinator colocated in parent process; failure watcher | `serve/launcher.py:288-355, 358-390` |
| Scheduler family contract (inbox/outbox/start/stop/abort) | `scheduling/omni_scheduler.py:71-73`; `scheduling/simple_scheduler.py:25-47`; `scheduling/dllm_scheduler.py:27-32`; `scheduling/messages.py:10-23` |
| API entry → Client → Coordinator chain | `serve/openai_api.py:159-201, 275-291`; `client/client.py:52-67, 132-144` |
| Thinker/talker wiring (builders, output processors) | `models/qwen3_omni/bootstrap.py:39-231` |
| Thinker multimodal embedding injection forward | `model_runner/thinker_model_runner.py:65-76, 102-116` |
| Thinker per-token stream builder (token + hidden to talker) | `models/qwen3_omni/request_builders.py:804-864` |
| Higgs enables async decode | `models/higgs_tts/config.py:54` |
| External router: policies, health, LocalLauncher of `sgl-omni serve` workers | `sglang_omni_router/selector.py:42-55`; `sglang_omni_router/worker.py:34-101`; `sglang_omni_router/launcher/local.py:37-57` |
| `*Message` wire dataclasses, hand-written to_dict/from_dict | `proto/messages.py:28-124, 286-317` |
| Single `Protocol` use; 5 enums; 121 dataclasses | `config/placement.py:40-41`; `scheduling/types.py:14-18`; grep counts |
| `EntryClass` + `Variants` registration idiom | `models/qwen3_omni/config.py:345-351` |
| Dotted-path factory resolution at spawn | `pipeline/stage_workers.py:30-62` |
| Attributed Note/FIXME comments (counts + GIL 600x, 8192→32768) | grep "Note (" = 9, "FIXME (" = 2; `omni_scheduler.py:896-901` (verified); `models/qwen3_omni/config.py:155-170` |
| Infra size norms (97 files, ~22k lines, max 1257) | `wc -l` over `sglang_omni/` excl. `vendor/`, `models/` |
| 12-line single-dataclass file accepted | `proto/stage.py:1-12` |
| design.md referenced but absent from checkout (anti-pattern) | `model_runner/base.py:75` (verified), `omni_scheduler.py:1049`; `find . -name design.md` empty |

---

# Part II — Deep Dive

> Scope note on citations. sglang-omni is not a fork; the executor is split across two codebases. Paths like `sglang_omni/...` are this repo. Paths prefixed `sglang@v0.5.8:` are the **pinned upstream dependency** (`sglang==0.5.8`, `pyproject.toml:29`), read from the local clone `~/Desktop/sglang` at tag `v0.5.8` via `git show v0.5.8:<path>` — i.e., the exact source that ships in this deployment, not memory of upstream.
>
> The division of labor, in one sentence: **upstream owns batch→tensor materialization, CUDA graphs, attention backends, the loader, the KV/radix pools, and the sampler kernel; sglang-omni owns a hook-based wrapper (`sglang_omni/model_runner/base.py`) that injects multimodal/projected embeddings, omni-specific sampling masks, hidden-state capture, and a colocated-GPU memory-profiling override — plus one genuinely novel executor design: the talker's fully in-graph sample+RVQ-predict decode step (§10.2.3).**

## 10. Model executor / worker internals

Part I §3.3 covered `run_batch` and the launch/resolve split at the *loop* level. This section goes one level down: how a scheduled batch becomes GPU tensors, what is inside vs outside the CUDA graphs, the attention-backend contract, and the sampling/output path.

### 10.1 (a) How a scheduled batch becomes GPU tensors

#### Three batch representations

A step passes through three objects: `ScheduleBatch` (scheduler-owned, mixed host/device state) → `ModelWorkerBatch` (a flat dataclass of tensor references, `sglang@v0.5.8: python/sglang/srt/managers/schedule_batch.py:2340`) → `ForwardBatch` (device-only view + pool/backend handles). The omni wrapper drives this exact pipeline in `_build_forward_batch`:

```python
model_worker_batch = schedule_batch.get_model_worker_batch()
is_prefill = bool(schedule_batch.forward_mode.is_extend())
...
forward_batch = ForwardBatch.init_new(model_worker_batch, self.tp_worker.model_runner)
```
(`sglang_omni/model_runner/base.py:212-248`; `get_model_worker_batch` is `sglang@v0.5.8: schedule_batch.py:2168-2244`, `ForwardBatch.init_new` is `sglang@v0.5.8: python/sglang/srt/model_executor/forward_batch_info.py:379-528`.)

#### Prefill packing: ragged concatenation, no padding

`prepare_for_extend` builds inputs by **flattening the per-request suffixes into one ragged 1-D tensor** — there is no per-request padding in non-graph forwards:

```python
input_ids = [r.fill_ids[len(r.prefix_indices):] for r in reqs]
extend_num_tokens = sum(len(ids) for ids in input_ids)
...
input_ids_tensor = torch.tensor(
    list(chain.from_iterable(input_ids)), dtype=torch.int64
).to(self.device, non_blocking=True)
```
(`sglang@v0.5.8: schedule_batch.py:1471-1509`). All host→device transfers in this function are `torch.tensor(...).to(device, non_blocking=True)` from Python lists (`seq_lens`, `orig_seq_lens`, `token_type_ids`, same file 1502-1516). KV slots are then allocated and the per-request token→slot map written by `alloc_for_extend` (called at `schedule_batch.py:1525-1528`). Multimodal pixel features ride along on the same batch and are moved (or CUDA-IPC-reconstructed) per item at `schedule_batch.py:1631-1645` (`mm_item.feature = pixel_values.to(self.device, non_blocking=True)` / `reconstruct_on_target_device`).

#### Decode: zero host→device traffic for input ids

`prepare_for_decode` never touches the host for input ids — the previous step's sampled tokens are already a GPU tensor:

```python
self.input_ids = self.output_ids
self.output_ids = None
...
self.out_cache_loc = alloc_for_decode(self, token_per_req=1)
...
self.seq_lens.add_(1); self.seq_lens_cpu.add_(1); self.orig_seq_lens.add_(1)
```
(`sglang@v0.5.8: schedule_batch.py:1951-2014`; in overlap mode the in-place `add_` is replaced by out-of-place `+ 1` to avoid racing the in-flight forward, lines 2005-2010). `alloc_for_decode` allocates one slot per request and **writes it into the persistent `req_to_token` page table**: `batch.req_to_token_pool.write((batch.req_pool_indices, locs), out_cache_loc.to(torch.int32))` (`sglang@v0.5.8: python/sglang/srt/mem_cache/common.py:425-464`).

`ForwardBatch.init_new` then computes positions on device (`clamp_position(batch.seq_lens)` for decode, `forward_batch_info.py:486-489,1086`; `compute_position` over extend prefix/seq lens for prefill, lines 490-509) and uploads the small CPU-side lists (`extend_seq_lens`, `extend_prefix_lens`) non-blocking (lines 491-498).

#### Persistent buffers

Three classes of persistent device state exist per AR stage:

1. **Upstream pools** — `req_to_token_pool` (request → token-slot page table) and `token_to_kv_pool` (paged KV), created in `init_memory_pool` and handed to the omni scheduler verbatim via `model_worker.get_memory_pool()` (`sglang_omni/model_runner/model_worker.py:128-132`). Struct-level detail in §11.1.
2. **CUDA-graph static input buffers** — `GraphInputBuffers.create` allocates `input_ids[max_num_token]`, `input_embeds[max_num_token, hidden]`, `req_pool_indices`, `seq_lens` (pre-filled with the backend's `seq_len_fill_value`), `out_cache_loc`, `positions`, `mrope_positions[3, max_num_token]`, plus a CPU `seq_lens_cpu` mirror (`sglang@v0.5.8: python/sglang/srt/model_executor/input_buffers.py:16-131`).
3. **Omni model-private static buffers** — the talker allocates a dozen `max_running_requests`-sized tensors at construction (`_feedback_buffer`, `_feedback_mask`, `_predictor_input_buffer`, `_predictor_k_cache`/`_predictor_v_cache`, `_sampled_token_ids`, `_repetition_mask`, `_suppress_mask`, per-batch sampling-param vectors, `_output_codes`, `_output_embeds`) — `sglang_omni/models/qwen3_omni/components/talker.py:395-405, 795-890`. Plus the wrapper's two lazily-allocated **pinned ping-pong host staging buffers** for async decode D2H (`sglang_omni/model_runner/base.py:69-91`, already analyzed in Part I §3.3).

#### Omni-specific input building (the wrapper's actual job)

Part I §7 step 10 mentioned the thinker's embedding injection; the tensor-level mechanics: the thinker's prefill replaces multimodal placeholder tokens with externally-computed encoder embeddings *after* the embedding lookup, by boolean-mask scatter into the packed rows, tracking a per-request `_omni_consumed` offset so chunked prefill consumes the embedding tensor incrementally:

```python
mask = req_input_ids == match_id
...
chunk_embeds = embeds[offset : offset + n_tokens].to(device=device, dtype=input_embeds.dtype)
input_embeds[torch.where(mask)[0] + start] = chunk_embeds
consumed[modality] = offset + n_tokens
```
(`sglang_omni/model_runner/thinker_model_runner.py:102-162`; the deepstack visual embeds are concatenated layer-wise and scattered through a global visual mask, lines 164-259; the custom forward path then bypasses `forward_extend` entirely, lines 265-312).

The talker's prefill input is not tokens at all but **projected thinker hidden states**, re-assembled per request from `data.prefill_input_embeds` plus a replayed "decode input history" (so a retract→re-prefill can reconstruct embeddings for already-generated codec tokens, raising if the history is missing: "Cannot replay retracted talker decode tokens", `sglang_omni/models/qwen3_omni/talker_model_runner.py:158-336`). Its decode-step input is host-combined (`feedback + next_text` embedding rows) and written into the static GPU `_feedback_buffer` + `_feedback_mask` before the forward (`talker_model_runner.py:338-364`); the model swaps them in **inside** the forward with a graph-safe `torch.where`:

```python
hidden_states = torch.where(
    feedback_mask.unsqueeze(-1),
    self._feedback_buffer[:bs].to(hidden_states.dtype),
    hidden_states,
)
```
(`sglang_omni/models/qwen3_omni/components/talker.py:425-434`).

### 10.2 (b) CUDA graph capture / replay

#### 10.2.1 Upstream mechanics (what actually runs here)

`init_device_graphs` constructs a `CudaGraphRunner` unless `disable_cuda_graph` / non-generation model (`sglang@v0.5.8: python/sglang/srt/model_executor/model_runner.py:1979-2017`; graph memory cost is logged as a before/after free-memory delta, lines 1995-2016). Key facts, all verified at the tag:

- **Decode-only**: `self.capture_forward_mode = ForwardMode.DECODE`, `num_tokens_per_bs = 1` (changes only for spec-decode TARGET_VERIFY and dLLM block extend) — `cuda_graph_runner.py:270-290`.
- **Capture sizes** come from server args: `[1, 2, 4, 8, 12] + range(16, 257, 8) + range(272, 512, 16) + range(512, max+1, 32)` clipped to `cuda_graph_max_bs` (`sglang@v0.5.8: python/sglang/srt/server_args.py:1039-1068`), then filtered to ≤ `req_to_token_pool.size` (`cuda_graph_runner.py:189-212`).
- **Capture order is reversed (largest bs first)** "to enable better memory sharing across cuda graphs", each bs warm-runs twice with a TP barrier before recording, and all graphs share one global graph memory pool (`cuda_graph_runner.py:472-534, 725-738`).
- **What is inside the graph**: `run_once()` calls the model forward over slices of the static buffers (`forward(input_ids, forward_batch.positions, forward_batch)`) after a one-time `attn_backend.init_forward_metadata_capture_cuda_graph(...)` (`cuda_graph_runner.py:686-720`). Sampling is *not* captured in the standard path — except in the talker, see §10.2.3.

**Replay eligibility** (`can_run`, `cuda_graph_runner.py:381-446`): batch must be in DECODE-graph mode (checked in `_forward_raw`, `model_runner.py:2277-2300`), bs ≤ `max_bs` (or exact-match if `disable_cuda_graph_padding`), encoder-decoder batches must have no `encoder_len == 0` rows, and the **requested `capture_hidden_mode` must be NULL or equal to the captured one** — the dynamic-shape escape hatches. **Replay padding** (`replay_prepare`, lines 772-840): `bs = self.capture_bs[bisect.bisect_left(self.capture_bs, raw_bs)]`, padded rows get `seq_len_fill_value` and zeroed `out_cache_loc`, real rows are `copy_`-ed into the static buffers (`input_buffers.py:133-207`), the attention backend rebuilds its graph metadata for the padded bs, then `self.graphs[bs].replay()` and the output is sliced back: `next_token_logits[: self.raw_num_token]` (lines 842-885). A mode change triggers a full **recapture** (`recapture_if_needed`, lines 740-771) — expensive, which is why omni pins the mode (below).

#### 10.2.2 The omni "deferred capture" idiom

Every AR stage in this repo constructs the upstream `ModelRunner` with CUDA graphs *forced off*, mutates the model, then re-enables and captures manually:

```python
want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
defer_cuda_graph_capture = want_cuda_graph and capture_hidden
if defer_cuda_graph_capture:
    server_args.enable_return_hidden_states = True
    server_args.disable_cuda_graph = True
... create_sglang_infrastructure(...)            # builds worker, installs hidden hooks
if defer_cuda_graph_capture:
    server_args.disable_cuda_graph = False
    model_worker.model_runner.init_device_graphs()
```
(thinker: `sglang_omni/models/qwen3_omni/bootstrap.py:31-59`; talker: same file 133-159, where the deferred step also runs *after* `model._sampler = model_runner.sampler` at lines 155-156 — the sampler must exist before capture because the talker samples inside the graph; the same pattern repeats for qwen3_asr (`models/qwen3_asr/stages.py:84-105`, which also sets `cuda_graph_max_bs: max_running_requests` at line 67), qwen3_tts, fishaudio, voxtral. Qwen3-TTS additionally `torch.compile`s only the decoder layers before capture and then un-sets `enable_torch_compile` so sglang doesn't compile a second time — `models/qwen3_tts/stages.py:109-124, 202-204, 220-242`).

Why the thinker needs it: hidden capture is hook-based — `install_hidden_capture_hooks` monkey-wraps the text model's `forward` to siphon the aux-hidden tuple into `model._captured_aux_hidden_states` (`sglang_omni/model_runner/_hidden_capture.py:26-76`), and `create_sglang_infrastructure` installs that wrapper after worker construction (`sglang_omni/scheduling/bootstrap.py:40-46`) — so capture must happen after the wrapper exists. Setting `enable_return_hidden_states=True` makes the runner's *initial* capture mode FULL (`sglang@v0.5.8: cuda_graph_runner.py:300-302`: "set initial capture hidden mode to full to avoid double-capture on startup"), and the wrapper then pins per-batch requests to NULL to keep `can_run` true forever:

```python
# Hidden capture for thinker streaming comes from our local forward hooks,
# not from SGLang's logits-output hidden-state path. Requesting LAST here
# causes CUDA-graph mode mismatches and can silently disable replay.
return CaptureHiddenMode.NULL
```
(`sglang_omni/model_runner/thinker_model_runner.py:78-96`, verified verbatim; the base wrapper would otherwise have requested LAST whenever `output_processor._capture_hidden` is set, `base.py:240-243`).

Replay-killers are explicit in this codebase: every custom forward returns `GenerationBatchResult(..., can_run_cuda_graph=False)` (`thinker_model_runner.py:310-312`, `talker_model_runner.py:523-526`, `ming_thinker_model_runner.py:251`) — acceptable because all custom forwards are prefill-only; decode always goes through `tp_worker.forward_batch_generation` → upstream `_forward_raw` → graph replay. A subtler one: DeepGEMM would JIT-compile for the request-dependent projected-prefill GEMM shapes, so the worker policy forces `fp8_gemm_runner_backend = "triton"` for FP8 talkers ("Projected talker prefill has request-dependent FP8 dense GEMM shapes outside decode CUDA graph replay; DeepGEMM can otherwise JIT there", `sglang_omni/model_runner/model_worker.py:313-323`).

#### 10.2.3 The talker: sampling + RVQ code predictor *inside* the graph

The most aggressive executor design in the repo. The talker's `forward` in decode mode does, in one capturable region: backbone → `codec_head` logits (a hand-rolled `_manual_decode_logits` that skips SGLang's `LogitsProcessor`) → repetition-penalty/suppression masking → upstream `Sampler` invocation → 31 sequential residual-codebook predictions — all writing results to static buffers:

```python
logits_output = self._manual_decode_logits(hidden_states)
if forward_batch.forward_mode.is_decode():
    sampled_token_ids = self._sample_decode_tokens(logits_output.next_token_logits, forward_batch)
    self._sampled_token_ids[:batch_size].copy_(sampled_token_ids)
    self.code_predictor_forward(sampled_token_ids.unsqueeze(1), hidden_states.unsqueeze(1))
```
(`sglang_omni/models/qwen3_omni/components/talker.py:1083-1095`, verified). The pieces that make this graph-safe:

- **Per-step dynamic sampling params become static-buffer writes done on the host before launch**: `prepare_decode_buffers` (called from `before_decode`, `talker_model_runner.py:63`) packs temperature/top-p/top-k/seeds and scatter-writes the repetition and suppress masks into `[max_bs, vocab]` boolean buffers (`talker.py:911-1000`).
- **Sampler control flow is frozen**: `_build_static_sampling_info` constructs a `SamplingBatchInfo` with `is_all_greedy=False, need_top_p_sampling=True, need_top_k_sampling=True` regardless of the actual params — "Keep sampler control flow static during graph capture while preserving SGLang's actual sampling kernel semantics" (`talker.py:1167-1190`).
- **The RVQ code predictor avoids the paged KV allocator entirely**, using a private static `[layers, max_bs, kv_heads, predictor_len, head_dim]` K/V cache plus plain `scaled_dot_product_attention` over the cached prefix (`talker.py:818-828, 1349-1414`) — fixed shapes, fixed addresses, capturable.
- **Outputs are mirrored to static buffers** `_output_codes` / `_output_embeds` when `seq_len == 1` ("runtime_single_token", `talker.py:1244-1249`), which the host-side runner reads after replay: `result.next_token_ids = self.model._sampled_token_ids[:batch_size].clone()` and per-request `code_chunk = self.model._output_codes[idx].detach().clone()` (`sglang_omni/models/qwen3_omni/talker_model_runner.py:93-136`).

The padded-replay corner is also handled: `_extend_last_index` switches from `cumsum(extend_seq_lens)-1` to `idx * padded_static_len + extend_seq_lens - 1` when `forward_batch.padded_static_len >= 0` (`talker.py:1192-1216`).

### 10.3 (c) Attention backend abstraction

**Selection.** Backends live in a string-keyed registry filled by decorators — `ATTENTION_BACKENDS = {}` + `@register_attention_backend("flashinfer")` etc. (`sglang@v0.5.8: python/sglang/srt/layers/attention/attention_registry.py:12-45`). `ModelRunner.init_attention_backend` → `_get_attention_backend` reads `server_args.get_attention_backends()` (separate prefill/decode strings, `server_args.py:4768-4779`), wraps differing prefill/decode choices in a `HybridAttnBackend`, and instantiates via `ATTENTION_BACKENDS[backend_str](self)` (`sglang@v0.5.8: model_runner.py:1596-1668`). When unset, the default is hardware-derived: `fa3` on Hopper+CUDA12.3, `trtllm_mha` on SM100, `aiter` on ROCm, else `flashinfer`/`triton` for MHA; `fa3`/`flashinfer`/`triton` for MLA (`sglang@v0.5.8: server_args.py:1655-1700`). sglang-omni does not register new attention backends; its backend policy layer only touches MoE/FP8-GEMM backends (`sglang_omni/model_runner/model_worker.py:242-333`).

**Per-batch metadata building.** The contract is: once per batch, before any layer runs, `attn_backend.init_forward_metadata(forward_batch)` precomputes everything layers will share — `forward_decode`/`forward_extend` call it unless replaying a graph (`sglang@v0.5.8: model_runner.py:2128-2178`). For FlashAttention-3, normal decode metadata is four tensors derived from the batch + the persistent page table:

```python
metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
metadata.cu_seqlens_q = torch.arange(0, batch_size + 1, ...)
metadata.cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, ...), (1, 0))
metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
    forward_batch.req_pool_indices, : metadata.max_seq_len_k
]
```
(`sglang@v0.5.8: python/sglang/srt/layers/attention/flashattention_backend.py:389-481`). Note `max_seq_len_k` comes from `seq_lens_cpu` — the reason `ScheduleBatch` maintains a CPU mirror of seq lens at all. Graph capture/replay have dedicated entry points (`init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph`, called at `cuda_graph_runner.py:686-694` and `829-840`) that build metadata into backend-owned static state sized by `init_cuda_graph_state(max_bs, max_num_token)` (`cuda_graph_runner.py:304-307`).

**Omni interaction**: because the custom prefill forwards bypass `forward_extend`, they must invoke the metadata step themselves — both do, identically: `model_runner.attn_backend.init_forward_metadata(forward_batch)` (`sglang_omni/model_runner/thinker_model_runner.py:275`, `sglang_omni/models/qwen3_omni/talker_model_runner.py:498`). Forgetting this line would mean attention layers read the *previous* batch's metadata.

### 10.4 (d) Weight loading

(Memory profiling and KV sizing, which the loading order feeds, are in §11.1.)

`ModelRunner.__init__` snapshots free memory **before** weights via `min_per_gpu_memory = self.init_torch_distributed()` (`sglang@v0.5.8: model_runner.py:365`; internally the distributed-min of `get_available_gpu_memory`, lines 779-805) and then runs `initialize(min_per_gpu_memory)` (line 383), which sequences: `create_sampler()` + `load_model()` (459-460) → `configure_kv_cache_dtype` → `init_memory_pool(min_per_gpu_memory)` (555) → `init_cublas` / `init_attention_backend` / `kernel_warmup` / `init_device_graphs` (571-575).

`load_model` builds a `LoadConfig`, takes the loader from `get_model_loader`, and measures the cost as a free-memory delta: `self.weight_load_mem_usage = before_avail_memory - after_avail_memory` (`sglang@v0.5.8: model_runner.py:805-948`); a TP `monitored_barrier` catches ranks that fail to load (lines 976-991). `DefaultModelLoader.load_model` instantiates the architecture *on the target device* under the model's dtype, then streams weights: `model.load_weights(self._get_all_weights(...))` followed by per-module `quant_method.process_weights_after_loading` (`sglang@v0.5.8: python/sglang/srt/model_loader/loader.py:640-683`); the weight iterator is safetensors-file streaming (lines 471-528). Architecture resolution goes through `get_model_architecture(model_config)` → `ModelRegistry` (`loader.py:229-234`).

sglang-omni hooks this path at two points, with zero loader changes:

1. **Registry injection** — before calling `super().__init__`, `SGLModelRunner._register_omni_model` writes 10 omni model classes straight into upstream's table: `ModelRegistry.models[arch] = getattr(importlib.import_module(module_path), attr)` (`sglang_omni/model_runner/sglang_model_runner.py:81-117`).
2. **Architecture override** — `ModelWorker._apply_arch_override` rewrites `model_config.hf_config.architectures = [arch]` and swaps `hf_text_config` / head counts / layer counts to the *sub-model* config (e.g. `Qwen3OmniTalker` → `talker_config.text_config`), so the loader, KV sizing, and attention setup all see the talker as the model (`sglang_omni/model_runner/model_worker.py:85-114`, table at 24-31; Part I §2 noted this mechanism — the executor-level consequence is that one HF checkpoint feeds multiple AR engines with correct per-sub-model KV cell sizes).

Prefix handling for shared checkpoints is done **inside the model's `load_weights`**, not in the loader: the talker accepts both monolithic and split checkpoints — `if name.startswith("talker."): name = name[len("talker."):]` and skips `thinker.`/`code2wav.` keys (`sglang_omni/models/qwen3_omni/components/talker.py:1448-1453`). (The `filter_weights_by_prefix` helper and the `weight_prefix` plumbed through `ModelWorkerConfig` have **no call sites** in the repo — `grep -rn filter_weights_by_prefix` finds only the definition at `sglang_omni/model_runner/sglang_model_runner.py:25-35`; the prefix logic that actually runs is the in-model strip above.)

Startup is serialized per GPU with an flock-based `gpu_startup_lock` ("Serialize heavyweight scheduler construction on one visible GPU", `sglang_omni/utils/gpu_memory.py:289-302`), taken around every scheduler factory in the stage worker (`sglang_omni/pipeline/stage_workers.py:608-621`) — which is also what makes the load-delta profiling fallback in §11.1 sound.

### 10.5 (e) Sampling and the output path back to the scheduler

#### Logits → token ids

The default path is upstream's: `ModelRunner.sample` runs `_preprocess_logits` (regex vocab mask + logit bias) and the `Sampler`, passing decode positions for deterministic seeding (`sglang@v0.5.8: model_runner.py:2352-2399`). `Sampler.forward` is argmax when all-greedy; otherwise `logits.div_(temperatures)` → in-place softmax → flashinfer `top_k_top_p_sampling_from_probs` or the pytorch fallback (`sglang@v0.5.8: python/sglang/srt/layers/sampler.py:67-160`).

Part I §3.3 noted the omni wrapper's repetition-penalty/suppress-token hooks (`sglang_omni/model_runner/base.py:476-545`); the executor-level detail is *when* sampling runs: subclasses control it via `sample_before_post_{prefill,decode}` — the talker samples prefill eagerly so `post_prefill` can run the code predictor on the first token, but returns `False` for decode because decode sampling already happened inside the graph (`talker_model_runner.py:138-148`); anything still unsampled is finalized in `_finalize` (`base.py:296-334`); the default delegate is `return self.tp_worker.model_runner.sample(logits_output, forward_batch)` (`base.py:485`). One omni-found upstream bug shapes a default here: the talker forces `sampling_backend: "pytorch"` because "Sampler.forward doesn't forward seed to flashinfer, so under cuda graph the captured RNG is boot-dependent and ~5% of prompts trigger degenerate AR loops (see #408)" (`sglang_omni/models/qwen3_omni/stages.py:1049-1056`, verified).

#### Token travel: GPU tensor → next step → host → scheduler → stream

1. **Stay-on-GPU feedback loop**: `schedule_batch.output_ids = batch_result.next_token_ids` (`base.py:293, 333-334`) feeds the *next* `prepare_for_decode`'s `self.input_ids = self.output_ids` (`sglang@v0.5.8: schedule_batch.py:1989`) — token ids never round-trip to the host between decode steps.
2. **Host materialization** happens exactly once per step, in the output processor: `token_list = model_output.next_token_ids.tolist()` (`sglang_omni/scheduling/sglang_backend/output_processor.py:34-38`) — this `.tolist()` is the implicit per-step GPU sync of the synchronous path (the async path replaces it with the event + pinned-buffer read of `execute_launch`/`execute_resolve`, `base.py:126-210`, covered in Part I §3.3). Each request gets a `RequestOutput(request_id, data=token_id, extra=hidden_extras)` (lines 52-62).
3. **Hidden-state extras** ride the same `RequestOutput.extra`: the hook side channel is consumed-and-cleared per forward (`captured_aux_hidden_states = self._model._captured_aux_hidden_states; self._model._captured_aux_hidden_states = None`, `output_processor.py:82-97`), sliced per request either by batch row or by `extend_input_len` offsets (`_slice_per_request_tensor`, lines 185-212), and `.clone()`d (line 149) — necessary because under graph replay those tensors are graph-owned buffers the next replay will overwrite; when the hook channel is empty (graph-replay decode steps, where the wrapped Python forward doesn't execute), it falls back to `logits_output.hidden_states`, which is the FULL-mode graph output slice (lines 99-128).
4. **Back into upstream scheduling**: the omni scheduler bridges the wrapper's `ModelRunnerOutput` to upstream's `GenerationBatchResult(logits_output=None, next_token_ids=batch.output_ids, ...)` (`sglang_omni/scheduling/omni_scheduler.py:665-677`, with `batch.input_ids = next_token_ids.to(torch.int64)` priming the next step), and upstream `process_batch_result_decode` does `next_token_ids = next_token_ids.tolist()` and `req.output_ids.append(next_token_id)` plus KV release on finish (`sglang@v0.5.8: python/sglang/srt/managers/scheduler_output_processor_mixin.py:359-405`).
5. **Per-token streaming to other stages** bypasses upstream's detokenizer entirely (Part I §3.3 covered the outbox side): the talker is even more direct, pushing `OutgoingMessage(type="stream", data=code_chunk, target="code2wav")` from inside its `post_decode` hook and queueing the feedback embedding row for its own next step (`talker_model_runner.py:111-136`).

---

## 11. Memory & cache subsystem

Part I §3.6 covered *which* caches exist and how they interact with scheduling. This section is the struct level: the actual pool/tree data structures in the pinned upstream, how requests map to physical slots, and what invalidation guarantees hold.

### 11.1 The block/page pool: structs, allocation, free list, sizing

Each AR stage owns **three layered objects**, all created inside the stage process by `create_sglang_infrastructure` (`sglang_omni/scheduling/bootstrap.py:48-55`; pools forwarded via `get_memory_pool`, `sglang_omni/model_runner/model_worker.py:128-132`).

**Struct 1 — `ReqToTokenPool`: the request→token indirection table.** Literally one int32 matrix plus a Python free list:

```python
self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int32, device=device)
self.free_slots = list(range(size))
```
Allocation is list slicing, free is `list.extend`, and `clear()` re-creates the full range (`sglang@v0.5.8: python/sglang/srt/mem_cache/memory_pool.py:117-164`). Row = one running request (`size = max_running_requests`, defaulted from profiled token count if unset), column = token position, cell value = a slot index into the KV pool. `max_context_len = context_len + 4` ("temporary fix for the context length issue when using speculative decoding", `model_runner_kv_cache_mixin.py:333-390`).

**Struct 2 — `MHATokenToKVPool`: the actual KV storage.** Per layer, two dense tensors indexed by token slot:

```python
# [size, head_num, head_dim] for each layer
# The padded slot 0 is used for writing dummy outputs from padded tokens.
self.k_buffer = [torch.zeros((self.size + self.page_size, self.head_num, self.head_dim), ...)
                 for _ in range(self.layer_num)]
```
(`sglang@v0.5.8: python/sglang/srt/mem_cache/memory_pool.py:754-778`). Note the `+ page_size` over-allocation: slot/page 0 is a sacrificial dummy target, which is why every allocator hands out indices starting at 1.

**Struct 3 — the allocator (free-list manager).** Selected in `init_memory_pool`: `page_size == 1` → `TokenToKVPoolAllocator`, else `PagedTokenToKVPoolAllocator` (`sglang@v0.5.8: python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:617-633`). Omni's server-args builder never sets `page_size`, and upstream defaults it to 1 (`_handle_page_size`: `if self.page_size is None: self.page_size = 1`, `sglang@v0.5.8: python/sglang/srt/server_args.py:1917-1919`), so **omni AR stages run the token-granular allocator** in practice. The free list is a device tensor, and alloc/free are tensor slicing/concat:

```python
def clear(self):
    # The padded slot 0 is used for writing dummy outputs from padded tokens.
    self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, device=self.device)
def alloc(self, need_size: int):
    ...
    select_index = self.free_pages[:need_size]
    self.free_pages = self.free_pages[need_size:]
def free(self, free_index: torch.Tensor):
    ...
    self.free_pages = torch.cat((self.free_pages, free_index))
```
(`sglang@v0.5.8: python/sglang/srt/mem_cache/allocator.py:131-165`). Two refinements in the base class: a **free-group** mechanism batches many small frees into one `torch.cat` (`free_group_begin/free_group_end`, `allocator.py:73-80`), and a deferred-sort `release_pages` staging list merged only when allocation would fail (`merge_and_sort_free`, `allocator.py:82-88`) — `need_sort` is only true in PD-disaggregation mode (`model_runner_kv_cache_mixin.py:587`), which omni doesn't use. The paged variant adds Triton kernels `alloc_extend_kernel`/`alloc_decode_kernel` that carve page-aligned slots and reuse the partial last page of the prefix (`allocator.py:338-428`); its `free` collapses token indices to pages with `torch.unique(free_index // self.page_size)` (`allocator.py:430-441`).

**There is no per-page reference count in the pool itself.** A KV slot has exactly one owner at a time: either a request row (via `req_to_token`) or a radix-tree node (`TreeNode.value` holds slot indices). Sharing and pinning are implemented one level up by tree-node `lock_ref` (§11.2); when a finished request's prefix already exists in the tree, the duplicate slots are *freed*, not refcounted (`radix_cache.py:463-466`, quoted in §11.3).

**Free memory → KV cache size.** Upstream formula (`sglang@v0.5.8: python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:111-144`):

```python
rest_memory = available_gpu_memory - total_gpu_memory * (1 - self.mem_fraction_static)
return int(rest_memory * (1 << 30)) // cell_size
```

where `available_gpu_memory` is measured **after** load (distributed-min over ranks), `total_gpu_memory` is the *pre-load* snapshot, and `cell_size` is bytes-per-token-per-pool: `num_kv_heads(tp) * (head_dim + v_head_dim) * num_layers * dtype_size` for MHA, `(kv_lora_rank + qk_rope_head_dim) * num_layers * dtype_size` for MLA (`mixin:47-110`). `init_memory_pool` turns the token count into pools — page-aligning it, deriving `max_num_reqs` from `max_total_num_tokens / context_len * 512` clamped to [2048, 4096] when unset, and erroring with the `--mem-fraction-static` hint if ≤ 0 (`mixin:252-330`). `get_available_gpu_memory` itself is `torch.cuda.mem_get_info` after `empty_cache()` (`sglang@v0.5.8: python/sglang/srt/utils/common.py:494-521`).

**The omni colocation override.** That free-memory-delta math is wrong when several stage processes share one GPU and load concurrently-ish — another process shrinks global free memory mid-load, so the delta under-counts or goes negative. `SGLModelRunner.profile_max_num_token` (`sglang_omni/model_runner/sglang_model_runner.py:221-237`) keeps upstream semantics when no stage budget is set, but with a `total_gpu_memory_fraction` budget it switches to **process-scoped accounting** (`_profile_available_bytes`, docstring verified verbatim at `sglang_model_runner.py:119-161`):

```python
"""...colocated Omni stages can load multiple SGLang engines in separate
processes on the same GPU. In that case another process can change global
free memory while this process is loading weights, making the global delta
too small or negative.

When a stage total-memory budget is provided, compute cache headroom as
total GPU memory times that budget minus this stage's measured memory.
NVML process accounting is preferred. ..."""
```

The mechanics: NVML `nvmlDeviceGetComputeRunningProcesses` filtered to `os.getpid()` gives this process's bytes (with CUDA_VISIBLE_DEVICES→NVML index/UUID remapping, `sglang_omni/utils/gpu_memory.py:84-135`); headroom is `int(total * fraction) - process_used`, raising "Colocated GPU memory budget leaves no KV-cache headroom" if non-positive (`gpu_memory.py:223-248`). If NVML can't see the process, the fallback uses the stage's own load delta — valid only because the flock serializes loads (`sglang_model_runner.py:163-198`; §10.4). The budget itself comes from the per-stage memory contract Part I §3.6 described: pipeline configs hand `total_gpu_memory_fraction` down, and `apply_encoder_mem_reserve` subtracts external-encoder headroom from SGLang's auto `mem_fraction_static` with a 0.1 safety floor (`sglang_omni/scheduling/sglang_backend/server_args_builder.py:40-62`).

### 11.2 Prefix cache internals: radix tree, lookup, insertion, eviction, ref protection

Part I §3.6 showed the two-way factory (`ChunkCache` vs `RadixCache`, `sglang_omni/scheduling/sglang_backend/cache.py:20-33`) — never the hierarchical/SWA/Mamba variants. Omni passes only the 5 basic `CacheInitParams` fields, so `eviction_policy` stays at its default `"lru"` and `enable_kv_cache_events=False` (`cache_init_params.py:13-35`).

**Node struct.** Not hash-bucketed — a true radix tree of `TreeNode`s:

```python
class TreeNode:
    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)   # child_key -> TreeNode
        self.parent: TreeNode = None
        self.key: RadixKey = None                # token-id segment
        self.value: Optional[torch.Tensor] = None  # int64 KV slot indices, same length as key
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        ...
        self.hit_count = 0
```
(`sglang@v0.5.8: python/sglang/srt/mem_cache/radix_cache.py:92-117`). `RadixKey` couples the token-id list with an `extra_key` namespace (lora id / cache salt); keys with different `extra_key` "are intentionally kept disjoint and never share prefix nodes" — the child-key function returns `(extra_key, first_token)` (`radix_cache.py:62-89, 185-193, 344-356`). The root is pinned forever: `self.root_node.lock_ref = 1` (`radix_cache.py:322-328`).

**Lookup** (`match_prefix` → `_match_prefix_helper`): walk children by first-token key, compare segments element-wise (`_key_match_page_size1`, `radix_cache.py:162-169`), accumulate each fully-matched node's `value` tensor; on a partial match inside a node, **split** it and stop:

```python
prefix_len = self.key_match_fn(child.key, key)
if prefix_len < len(child.key):
    new_node = self._split_node(child.key, child, prefix_len)
    value.append(new_node.value)
    node = new_node
    break
```
(`radix_cache.py:627-651`). The split clones the slot-index tensors (`new_node.value = child.value[:split_len].clone()`) and inherits `lock_ref` so a split can never unpin a protected path (`radix_cache.py:653-672`). Matched values are `torch.cat`-ed into the request's `prefix_indices` (`radix_cache.py:405-409`). The walk refreshes `last_access_time` on every touched node — that *is* the LRU bookkeeping.

**Insertion** (`_insert_helper`): same walk; consumes matched prefix, splits on partial overlap, and hangs one new leaf with the unmatched suffix, growing `evictable_size_` (`radix_cache.py:674-715`). It returns `total_prefix_length` = how many leading tokens were already present — the caller uses this to free duplicate KV slots (§11.3).

**Eviction — the actual code.** Eviction is *synchronous and allocation-driven*, not a background thread. Every allocation funnels through `alloc_token_slots` / `alloc_paged_token_slots_extend/decode`, which first call:

```python
def evict_from_tree_cache(tree_cache, num_tokens):
    ...
    if allocator.available_size() < num_tokens:
        tree_cache.evict(num_tokens)
```
(`sglang@v0.5.8: python/sglang/srt/mem_cache/common.py:229-251`, called at `common.py:207, 267, 407`). `evict` is a leaf-heap pop loop (verified verbatim at the tag):

```python
def evict(self, num_tokens: int):
    leaves = self._collect_leaves()
    eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
    heapq.heapify(eviction_heap)
    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))
```
(`radix_cache.py:548-573`). Two properties to note: (1) only **leaves** are candidates, and a parent becomes a candidate only after all children die — interior shared prefixes survive longest; (2) `_collect_leaves` filters `if cur_node.lock_ref == 0` (`radix_cache.py:754-766`), so locked nodes are unconditionally exempt. With omni's default policy the heap priority is plain `node.last_access_time` (`LRUStrategy`, `sglang@v0.5.8: python/sglang/srt/mem_cache/evict_policy.py:16-18`; strategy table selected at `radix_cache.py:284-299`).

**Ref protection against eviction races.** There is **no mutex anywhere in the tree** — correctness relies on the scheduler being single-threaded per stage (each omni stage runs its scheduler in one dedicated thread, Part I §2; eviction happens inline inside that thread's alloc calls). What `lock_ref` protects against is *cross-batch* races: request A is mid-decode while request B's prefill triggers eviction. Pinning is path-based:

```python
def inc_lock_ref(self, node: TreeNode):
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        node = node.parent
```
(`radix_cache.py:575-587`; `dec_lock_ref` is the mirror, with a tree-membership assert "This request holds the node from another tree", `radix_cache.py:589-605`). The lock is taken when a request is admitted to a prefill batch — `PrefillAdder._req_inc_lock_ref(req)` locks `req.last_node` (`sglang@v0.5.8: python/sglang/srt/managers/schedule_policy.py:552-557, 753`), and `PrefillAdder` also uses a temporary `_lock_node` contextmanager while sizing a candidate (`schedule_policy.py:586-598`) — and released in `cache_finished_req`/`cache_unfinished_req` (§11.3). `evictable_size_`/`protected_size_` are the running counters the scheduler's memory checks read (`radix_cache.py:607-612`).

### 11.3 How requests map to physical memory

The full indirection chain is `Req.req_pool_idx → req_to_token[row, 0..seq_len) → k_buffer[layer][slot]`. The per-request bookkeeping fields live on upstream `Req`:

```python
# For req-level memory management
self.kv_committed_len = 0
self.kv_allocated_len = 0
...
self.req_pool_idx: Optional[int] = None
...
self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
self.last_node: Any = None
# The prefix length that is inserted into the tree cache
self.cache_protected_len: int = 0
```
(`sglang@v0.5.8: python/sglang/srt/managers/schedule_batch.py:543-546, 590, 635, 640, 646`).

**Prefill.** `req.init_next_round_input(tree_cache)` runs `match_prefix` (capped at `input_len - 1` so at least one token is computed for logprobs) and stores `prefix_indices` / `last_node` / `cache_protected_len = len(prefix_indices)` (`schedule_batch.py:861-907`). `ScheduleBatch.prepare_for_extend` → `alloc_for_extend`: allocate a row (`alloc_req_slots`), allocate `extend_num_tokens` KV slots, then write both the cached prefix indices and the new slots into the row in one Triton kernel, `write_req_to_token_pool_triton` (`sglang@v0.5.8: python/sglang/srt/mem_cache/common.py:329-393, 78-107`). Omni reaches this path two ways: via composition through upstream `get_next_batch_to_run` (OmniScheduler, Part I §3.2), or through its own standalone `PrefillManager`, which rebuilds the same upstream `PrefillAdder` + `ScheduleBatch.init_new(...).prepare_for_extend()` pipeline (`sglang_omni/scheduling/sglang_backend/prefill.py:78-147`) — the projected-embeds no-chunking rule was covered in Part I §3.6.

**Decode.** One slot per request per step: `alloc_for_decode` writes the new slot at column `seq_lens` and `prepare_for_decode` then increments the request's ledger — `req.kv_committed_len += 1; req.kv_allocated_len += 1` (`common.py:425-464`; `schedule_batch.py:1999-2002`).

**Release.** Both finish and abort go through `release_kv_cache(req, tree_cache)` (`common.py:467-497`), which omni calls from `_release_request_kv_cache` (`sglang_omni/scheduling/omni_scheduler.py:890-893`) and the dllm/fish schedulers (`sglang_omni/scheduling/dllm_scheduler.py:134-136`, `sglang_omni/models/fishaudio_s2_pro/fish_scheduler.py:232-235`). Inside `cache_finished_req`, the radix cache *adopts* the request's KV instead of freeing it:

```python
# Radix Cache takes one ref in memory pool
if is_insert:
    new_prefix_len = self.insert(radix_key, values, priority=priority)
    # Free the duplicates that were already in the tree
    self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])
...
self.req_to_token_pool.free(req.req_pool_idx)
self.dec_lock_ref(req.last_node)
```
(`radix_cache.py:459-477`). Chunked requests do the same dance mid-flight via `cache_unfinished_req`, which re-matches, rewrites the row with canonical tree indices, and migrates the lock (`dec_lock_ref(req.last_node); inc_lock_ref(new_last_node)`) (`radix_cache.py:479-539`). Double-free is detected, not tolerated: `pop_committed_kv_cache` asserts `not self.kv_committed_freed` ("Committed KV cache already freed") (`schedule_batch.py:801-807`).

**The talker rollback, at the ledger level.** Part I §3.5 quoted the rollback's allocator/seq_lens inversion; the memory-ledger half is that it also decrements the per-request accounting fields so the next real decode step's commit stays consistent:

```python
for req in batch.reqs:
    req.decode_batch_idx -= 1
    req.kv_committed_len -= 1
    req.kv_allocated_len -= 1
```
(`sglang_omni/models/qwen3_omni/talker_scheduler.py:110-143`, verified — guarded by the explicit drift tripwire `raise TypeError(... "sglang upstream prepare_for_decode changed; update rollback.")` and scoped by the comment "This is talker-only. It does not fully invert prepare_for_decode"). The async-decode lookahead adds the symmetric guard on the free side: a stale batch drained after the in-flight step completes would "double-free their committed KV cache (process_batch_result_decode -> release_kv_cache -> pop_committed_kv_cache asserts 'already freed')", so finished reqs are filtered out before re-running (`sglang_omni/scheduling/omni_scheduler.py:1092-1112`; Part I §3.3).

**Retraction.** On `check_decode_mem()` failure, `DecodeManager` runs upstream `retract_decode` and re-queues victims into the `PrefillManager` (`sglang_omni/scheduling/sglang_backend/decode.py:36-78`). Upstream retraction picks victims by most-output/shortest-input and — important for invalidation semantics — "release memory and don't insert into the tree because we need the space instantly" (`sglang@v0.5.8: python/sglang/srt/managers/schedule_batch.py:1846-1906`).

### 11.4 Cache invalidation paths and guarantees

The findings here are mostly **negative space**, which matters for operating this system:

- **No flush endpoint, no weight-update path.** Upstream ships `mem_cache/flush_cache.py` and weight-update RPCs, but grep over `sglang_omni/` finds zero calls to `flush_cache`, `update_weights`, or `tree_cache.reset()` outside construction (verified: `grep -rn "flush_cache\|update_weights\|reset_prefix" sglang_omni/` → no scheduler/serve hits; `RadixCache.reset()` is invoked only from its own `__init__`, `radix_cache.py:300, 322-332`). **Guarantee:** prefix-cache contents are valid for the entire stage-process lifetime because nothing can mutate the weights they were computed under — weights are loaded once at stage construction and the only teardown is process shutdown (per Part I §2, a stage crash kills the whole process group, taking pools with it).
- **Per-request invalidation = abort.** `_release_immediate_request_resources` scans every batch a live req can inhabit (`running_batch, cur_batch, last_batch, _async_pending_batch()`) and releases KV once per req identity (`sglang_omni/scheduling/omni_scheduler.py:874-893`). Since abort goes through `release_kv_cache` with default `is_insert=True`, an aborted request's computed prefix is *kept* in the radix tree — abort invalidates the request, not the cache.
- **Retraction guarantee:** retracted decode reqs free their KV without polluting the tree (`schedule_batch.py:1884`, quoted above) and are guaranteed to re-prefill from whatever prefix is still cached (`decode.py:73-78` re-queues via `on_retract → prefill_mgr.add_one_request`).
- **`disable_radix_cache` semantics:** `ChunkCache` returns an empty match for every lookup and frees everything at request end — "Chunk cache's eviction is the same with request's lifecycle" (`sglang@v0.5.8: python/sglang/srt/mem_cache/chunk_cache.py:50-84`); its `evict()` and `inc/dec_lock_ref` are no-ops, and `evict_from_tree_cache` short-circuits on `is_chunk_cache()` (`common.py:233-234`).

### 11.5 Second-tier memory: present upstream, deliberately disabled here

The pinned upstream tree contains a full hierarchical stack (`hiradix_cache.py`, `memory_pool_host.py`, `hicache_storage.py`; `TreeNode` even carries the hooks: `host_ref_counter`, `host_value`, `protect_host()/release_host()`, `radix_cache.py:106-136`). **None of it is reachable in this checkout's omni schedulers**: `OmniScheduler.__init__` hard-codes

```python
self.enable_hierarchical_cache = False
self.enable_hicache_storage = False
self.enable_kv_cache_events = False
```
(`sglang_omni/scheduling/omni_scheduler.py:245-247`), and the cache factory never constructs `HiRadixCache` (`sglang_omni/scheduling/sglang_backend/cache.py:28-33`). The CPU-side memories that *do* exist:

1. **`StageOutputCache`** — omni's own LRU for non-AR stage outputs (encoder results, introduced in Part I §3.4/§3.6): an `OrderedDict` of `_CacheEntry(data, size_bytes)` with dual budgets (`max_size` entries + `max_bytes`), tensor detach + optional device move on insert, oversized values silently not cached (`if size_bytes > self.max_bytes: return`), and FIFO-from-LRU-end eviction in `_evict_over_budget` (`sglang_omni/scheduling/stage_cache.py:40-98`). Keyed by media hash — *content* caching, not KV caching.
2. **`cpu_offload_gb`** — plumbed by the CLI only to the thinker stage's ServerArgs (`sglang_omni/cli/serve.py:461-485, 851-856`); upstream-side it configures the **weight** offloader (`OffloaderV1(cpu_offload_max_bytes=...)`, `sglang@v0.5.8: python/sglang/srt/utils/offloader.py:64-68`), i.e. parameters in host RAM — KV cache never leaves the GPU.

So the honest one-line summary: **no second-tier KV memory is active in sglang-omni; each AR stage's prefix cache is GPU-resident, sized by the stage's `total_gpu_memory_fraction` contract, and evaporates with the stage process.**

---

## 12. Extension surfaces: adding a model/reward/backend

Part I §8.3 recommended the "pipeline as data" idea at the concept level; this section is the full mechanical trace of what adding a model actually requires, plus the config plumbing underneath it.

### 12.1 The registry: filesystem-scan + `EntryClass`, keyed by HF architecture string

There is exactly one model registry, `PIPELINE_CONFIG_REGISTRY`, populated at import time by scanning `sglang_omni/models/*` for subpackages that contain a `config` module exporting `EntryClass` (`models/registry.py:135-136`):

```python
PIPELINE_CONFIG_REGISTRY = _PipelineConfigRegistry()
PIPELINE_CONFIG_REGISTRY.register_config("sglang_omni.models", "config")
```

`import_pipeline_configs` walks `pkgutil.iter_modules` over the package, imports each subpackage, then imports `<pkg>.config`; a missing `config` submodule is skipped, but a package whose `config` module exists **must** export `EntryClass` or registration hard-fails (`models/registry.py:67-70`, verified):

```python
if not hasattr(config_module, "EntryClass"):
    raise AssertionError(
        f"Config module {name}.{config_path} must have an EntryClass"
    )
```

Three robustness details worth knowing before you add a package:

- **Lenient by default, strict on demand**: any import error in a model package is logged and skipped unless `strict=True` (`models/registry.py:38-44`) — so a model whose optional dependency (e.g. `qwen-tts`) isn't installed silently disappears from the registry rather than breaking every other model. The whole scan is `@lru_cache()`d (`models/registry.py:27-30`).
- **Registry key = HF architecture string(s)**: each config class declares `architecture: ClassVar[str]` plus optional `architecture_aliases` (`models/registry.py:13-24`; `PipelineConfig.architecture_aliases` default at `config/schema.py:206`). Duplicate architecture claims across two different classes raise `ValueError` at import (`models/registry.py:73-83`, verified).
- **Variants are looked up lazily by class name**: `get_config_cls_by_name` searches both `EntryClass`es and each config module's `Variants` dict, so a YAML config file can name a variant class directly via its `config_cls` field (`models/registry.py:121-132`, consumed at `config/manager.py:138-139`).

### 12.2 Architecture detection: three fallbacks from `--model-path` to a config class

`resolve_config_cls_for_model_path` (`config/manager.py:16-31`) tries, in order: (1) `AutoConfig.from_pretrained` → `architectures`/`architecture`/`model_type` (`utils/hf.py:34-47`); (2) raw `config.json` parsed as plain JSON — the documented reason is models requiring `trust_remote_code=True` whose custom config module isn't installed (`utils/hf.py:81-88`); (3) Mistral-format `params.json`, because "Official Voxtral TTS checkpoints ship without `config.json`" (`utils/hf.py:50-54, 72-78`). Fallbacks (2)/(3) map `model_type` through a small hand-maintained table (`utils/hf.py:26-31`):

```python
_CONFIG_MODEL_TYPE_TO_ARCH = {
    "moss_tts_delay": "MossTTSDelayModel",
    ...
    "qwen3_tts": "Qwen3TTSForConditionalGeneration",
}
```

So **a new model whose checkpoint carries a proper `architectures` list needs zero detection code**; only config-less checkpoints need a `_CONFIG_MODEL_TYPE_TO_ARCH` entry.

### 12.3 Concrete trace: Qwen3-TTS (the smallest complete AR model package)

`models/qwen3_tts/` is 7 files (`config.py` 45 LOC, `stages.py` 356, `request_builders.py` 832, `model_runner.py` 271, `payload_types.py` 106, `sglang_model.py` 1272, `__init__.py`). The `__init__.py` only does `from . import config` (`models/qwen3_tts/__init__.py:4`) so the registry scan finds `EntryClass`.

**Step 1 — declare the pipeline as data.** The whole "model registration" is one Pydantic subclass with a `ClassVar` architecture and a default `stages` list of dotted-path factories (`models/qwen3_tts/config.py:13-45`):

```python
class Qwen3TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3TTSForConditionalGeneration"
    stages: list[StageConfig] = [
        StageConfig(name="preprocessing", process="pipeline",
                    factory=f"{_PKG}.stages.create_preprocessing_executor",
                    next="tts_engine"),
        StageConfig(name="tts_engine", ..., factory_args={"gpu_id": 0, "dtype": "bfloat16"},
                    gpu=0, next="vocoder"),
        StageConfig(name="vocoder", ..., gpu=0, terminal=True),
    ]
EntryClass = Qwen3TTSPipelineConfig
```

`PipelineConfig.model_post_init` validates the graph (unique names, exactly one of `next`/`terminal`, `wait_for` ⇒ `merge_fn`, all routing targets exist, non-TP stages must declare `process`) at construction time (`config/schema.py:222-227, 282-364`).

**Step 2 — the stage factory is the only required "interface".** Each `factory` is a plain function returning *any object satisfying the 5-member scheduler contract* (`inbox`, `outbox`, `start()`, `stop()`, `abort(rid)` — `scheduling/omni_scheduler.py:71-72`, Part I §3.1). For non-GPU work that's one line — `create_preprocessing_executor` returns `SimpleScheduler(preprocess_qwen3_tts_payload, abort_callback=cleanup_prepared_qwen3_tts_request)` (`models/qwen3_tts/stages.py:154-159`). The dotted path is resolved in the **child process** by `import_string` inside `_construct_scheduler` (`pipeline/stage_workers.py:608-621`), with kwargs coming from resolved `factory_args` (§12.4.3).

**Step 3 — for an AR stage, wire an `OmniScheduler` with model-specific adapters.** `create_sglang_tts_engine_executor` (`models/qwen3_tts/stages.py:162-267`) is the canonical recipe:

1. Build sglang `ServerArgs` from shared defaults + per-model overrides (`stages.py:187-200` → `scheduling/sglang_backend/server_args_builder.py:10-37`).
2. `create_sglang_infrastructure(server_args, gpu_id, model_arch_override="Qwen3TTSTalker")` (`stages.py:206-218`) — builds the `ModelWorker`, memory pools, tree cache, and prefill/decode managers (`scheduling/bootstrap.py:9-84`). The `model_arch_override` is the mechanism that lets one HF checkpoint feed a sub-model AR engine (§10.4): adding a new sub-model arch means adding one tuple to `_ARCH_CONFIG_MAP` (`model_runner/model_worker.py:24-31, 86-114`) plus the model class itself (in `sglang_model.py`, registered with sglang's model loader).
3. Construct the `OmniScheduler` with the model-local adapters (`stages.py:254-267`). The adapter contract is visible in `OmniScheduler.__init__`'s keyword signature (`scheduling/omni_scheduler.py:83-104`): `request_builder` ("StagePayload → SGLangARRequestData", `omni_scheduler.py:109-110`), `result_adapter` (request data → output `StagePayload`), and optional `stream_output_builder` / `stream_chunk_handler` / `stream_done_handler` / `abort_callback`. For Qwen3-TTS the pair is a 14-line closure factory (`models/qwen3_tts/request_builders.py:819-832`):

```python
def make_qwen3_tts_scheduler_adapters(*, model, wrapper):
    def request_builder(payload): return build_sglang_qwen3_tts_request(payload, model=model, wrapper=wrapper)
    def result_adapter(data):     return apply_sglang_qwen3_tts_result(data.stage_payload, data)
    return request_builder, result_adapter
```

4. Optionally subclass `ModelRunner` for per-step hooks — `Qwen3TTSModelRunner.before_prefill` calls `model.prepare_decode_buffers(requests)` (`models/qwen3_tts/model_runner.py:16-30`).

**Engineering detail — cross-stage prepared-request handoff.** Preprocessing runs heavyweight prompt/audio prep in the preprocessing stage but parks the result in a process-local `_PREPARED_REQUESTS` dict keyed by request id (`request_builders.py:677-691`); the AR `request_builder` pops it (`request_builders.py:694-710`). This only works because both stages declare `process="pipeline"` — they share one OS process — and it is why both stages register the same `abort_callback=cleanup_prepared_qwen3_tts_request` (`stages.py:158, 266`), or aborted requests would leak prepared tensors. This is a model-local pattern, not a framework facility.

**Checklist to add a model** (everything the framework requires, verified against the above):
1. `sglang_omni/models/<name>/__init__.py` importing `config`.
2. `config.py`: `PipelineConfig` subclass with `architecture` ClassVar + `stages` data; `EntryClass = ...`; optional `Variants = {...}` dict (looked up by `ConfigManager.from_model_path(variant=...)`, `config/manager.py:106-123`) and optional role-map classmethod overrides (`mem_fraction_role_to_stage` etc., `config/schema.py:239-272`) if you want the generic CLI flags (`--mem-fraction-static`, `--talker-gpu`, …) to work for your pipeline.
3. `stages.py`: one factory per stage returning a scheduler-contract object.
4. For AR stages: adapters in `request_builders.py`, a `ModelRunner` subclass if hooks are needed, and (for sub-model checkpoints) a `_ARCH_CONFIG_MAP` entry.
5. If the checkpoint has no `architectures` in `config.json`: a `_CONFIG_MODEL_TYPE_TO_ARCH` entry (`utils/hf.py:26-31`).

### 12.4 The config/args system

#### 12.4.1 CLI: typed typer flags + untyped dotted-path passthrough

The console script is `sgl-omni = "sglang_omni.cli:app"` (`pyproject.toml:75-76`). The `serve` command is registered with `allow_extra_args + ignore_unknown_options` (`cli/__init__.py:10-12`), which creates a **two-tier argument system**:

- **Tier 1 — typed flags** declared as `typer.Option`s on `serve()` (`cli/serve.py:775-1009`): placement (`--thinker-tp-size`, `--thinker-gpus`, `--talker-gpu`), memory (`--mem-fraction-static` + per-role variants, `--encoder-mem-reserve`), perf toggles as tri-states (`--thinker-cuda-graph default|on|off`, `--thinker-torch-compile`, `--async-decode`), and server basics.
- **Tier 2 — arbitrary dotted-path overrides** captured from `ctx.args` (`cli/serve.py:1036-1037`). `ConfigManager.parse_extra_args` accepts `--a.b.c=v` or `--a.b.c v` pairs (`config/manager.py:45-70`); `merge_config` then walks `model_dump()` output by dotted key — **integer path segments index into lists** (`if k.isdigit(): k = int(k)`, `config/manager.py:91-94`) — mutates the dict, and re-validates by reconstructing the config class: `merged_config = config_cls(**cfg_copy)` (`config/manager.py:102`). So Pydantic `extra="forbid"` on every schema model (`config/schema.py:14, 138, 204`) is the validation backstop for free-form CLI input: an override of a nonexistent field fails at reconstruction. Values are type-inferred (`true/false/none/int/float/str`, `config/manager.py:225-245`). The docstring credits TorchTitan for this design (`config/manager.py:35-40`).

A representative subtlety: because `tp_size` and `parallelism.tp` are aliases, `_sync_stage_parallelism_aliases` patches the *other* one when a dotted override sets only one of them (`config/manager.py:198-222`), and `StageConfig.model_post_init` rejects contradictory explicit settings (`config/schema.py:188-198`).

#### 12.4.2 Config sources and precedence

Resolution order in `serve()` (`cli/serve.py:1022-1086`): `--config` YAML file (must carry `config_cls`, supports a `stage_overrides: {stage: {runtime: ...}}` block deep-merged per stage, `config/manager.py:126-185`) **or** `--model-path` via the registry (with `--text-only` selecting the `"text"` variant, `cli/serve.py:1026-1028`); then dotted extra-args merge; then ~7 typed-flag override passes applied in a fixed order (mem-fraction → encoder-reserve → thinker server-args → parallelism → cuda-graph → torch-compile → async-decode → partial-start, `cli/serve.py:1042-1086`).

Two disciplined patterns in the override passes:

- **Atomic validation before mutation** — documented in `apply_mem_fraction_cli_overrides`: "out-of-range values raise typer.BadParameter atomically, before any stage mutation, so a partially-applied config cannot leak into the launch path" (`cli/serve.py:225-228`), with explicit per-role precedence (per-role flag > global flag, `cli/serve.py:217-224, 260-270`).
- **Capability discovery via role maps, not stage-name guessing** — generic flags resolve their target stage through classmethods the pipeline config opts into (`talker_role_to_stage()`, `code2wav_stage()`, base impls return empty at `config/schema.py:239-272`), and factory-pinned flags verify the stage's factory string before applying (`--talker-partial-start` rejects non-Qwen talkers, `cli/serve.py:650-655`; `--async-decode` is pinned to the Higgs factory, `cli/serve.py:726-733`).

#### 12.4.3 Plumbing down to workers: three namespaces with ownership validation

A stage's runtime knobs live in three places, merged by `resolve_stage_factory_args` (`config/runtime.py:15-55`) at spec-build time (`pipeline/mp_runner.py:87`):

1. `stage.factory_args` (model-package defaults, e.g. `{"gpu_id": 0, "dtype": "bfloat16"}`),
2. `pipeline.runtime_overrides[stage_name]` (file/CLI-level), with `server_args_overrides` dict-merged rather than replaced (`config/runtime.py:125-139`),
3. typed `stage.runtime` fields (`max_seq_len`, `video_fps`, `resources.total_gpu_memory_fraction`, `sglang_server_args.mem_fraction_static`, `config/schema.py:80-101`), translated into factory kwargs via the stage's `runtime_arg_map` (`config/runtime.py:142-161`).

Double-setting the same knob through two namespaces is a hard error, not last-writer-wins (`_validate_runtime_sources`, `config/runtime.py:75-108`; untyped `total_gpu_memory_fraction` is rejected outright with a "set runtime.resources… instead" message, `config/runtime.py:58-72`). Cross-cutting injections are **signature-gated**: `model_path`, `gpu_id`, and `total_gpu_memory_fraction` are added only if `inspect.signature(factory)` declares the parameter and the caller didn't set it (`config/runtime.py:34-53`) — so factories opt into framework-provided values by naming the parameter.

The end of the pipe is sglang itself: factory kwargs like `server_args_overrides` flow into `build_sglang_server_args`, which sets shared defaults (`trust_remote_code=True`, `random_seed=123`, `max_prefill_tokens=16384`) and applies overrides as plain `ServerArgs(**kwargs)` fields (`scheduling/sglang_backend/server_args_builder.py:10-37`). Everything a `StageConfig` resolves to is frozen into a picklable `StageLaunchConfig` ("All string references (factory, merge_fn) are dotted import paths resolved by the child via `import_string`", `pipeline/stage_workers.py:29-39`) — config classes themselves never cross the process boundary.

Finally, `PipelineConfig.env_defaults` becomes spawn-time environment for every stage process, applied only for keys not already in the parent env, with conflicting defaults across co-located stages being an `AssertionError` (`pipeline/stage_workers.py:136-167`).

---

## 13. Startup sequence & failure handling

Part I §4 covered the process topology; this section is the exact boot order and the complete error/abort ladders.

### 13.1 Startup order of operations

Numbered trace for `sgl-omni serve --model-path Qwen/Qwen3-TTS...` (single-GPU Qwen3-TTS shape; TP differences noted inline):

1. **CLI parse + logging**: typer invokes `serve()`; `logging.basicConfig` at the chosen `--log-level` (`cli/serve.py:1011-1014`).
2. **Config resolution**: registry lookup from model path (or YAML), dotted-args merge, typed override passes (`cli/serve.py:1023-1086`; §12.4.2). Merged config printed when `--colocate` or debug (`cli/serve.py:1088-1089`).
3. **Enter async serving path**: `launch_server` → `asyncio.run(_run_server(...))` (`serve/launcher.py:395-429`).
4. **Port probe before any model load**: `_find_available_port` binds/releases the socket; falls back to a free port with a warning (`serve/launcher.py:301-302, 58-71`).
5. **Runner start with global startup deadline**: `MultiProcessPipelineRunner(pipeline_config)`; `await mp_runner.start(timeout=float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600")))` (`serve/launcher.py:304-306`).
6. **Plan resolution (parent, no GPU touched)**: `prepare_pipeline_runtime` → `config.apply_fusion()` → `build_stage_placement_plan` → `build_process_topology_plan` → `allocate_endpoints` under a fresh per-run IPC tempdir (`pipeline/mp_runner.py:369-374`; `pipeline/runtime_config.py:74-113, 162-174`).
7. **Spec construction**: `_build_stage_groups` resolves factory args (§12.4.3), same-GPU/same-process fast-path target sets, NCCL port per TP stage, and builds `StageLaunchConfig`/`StageWorkerProcessSpec` per process (`pipeline/mp_runner.py:375-383, 41-165`).
8. **Coordinator up first**: constructed with completion/abort endpoints + terminal-stage set, `await coordinator.start()` binds the ZMQ sockets, and the completion loop task starts **before any child exists** (`pipeline/mp_runner.py:385-400`) — so no completion can ever race the listener.
9. **Spawn**: per group, `ctx.Process(target=stage_process_main, ..., daemon=True)` with a `ctx.Event()` readiness flag and `ctx.Queue()` startup-error channel; `_patched_spawn_env` temporarily injects `env_defaults` + TP CUDA remap into the parent env around `proc.start()` (`pipeline/stage_workers.py:224-244, 135-167`; spawn ctx at `mp_runner.py:366`).
10. **Child boot**: `stage_process_main` → `logging.basicConfig(INFO)` → `_prepare_cuda_environment` (TP ranks see exactly one device, `gpu_id` normalized to local 0) → `_run_process` (`pipeline/stage_workers.py:338-359, 661-699`). Any exception puts `traceback.format_exc()` on the startup-error channel and `sys.exit(1)` (`stage_workers.py:353-359`).
11. **Model load, serialized per GPU**: `_construct_stage` sets `torch.cuda.set_device` (`stage_workers.py:418-427`), then `_construct_scheduler` takes an `fcntl.flock` file lock per visible GPU before calling the factory (`stage_workers.py:608-621`; lock impl `utils/gpu_memory.py:290-302`). Inside the factory (AR case): `ServerArgs` → `create_sglang_infrastructure` (`scheduling/bootstrap.py:9-84`) → `ModelWorker.__init__` runs `_init_model_config` (arch override) → `_configure_backend_policy` (MoE/FP8 backend selection with hard validation errors, `model_runner/model_worker.py:50-52, 242-333`) → `_init_model_runner` constructs `SGLModelRunner` = **weights load + KV pool allocation here** (`model_worker.py:166-185`; loading order detail in §10.4, KV sizing in §11.1); then tree cache + prefill/decode managers (`bootstrap.py:48-74`).
12. **Stage shell wiring**: routing closures (static `next` / validated `route_fn`), fan-in `AggregatedInput` vs `DirectInput`, `StageControlPlane` (or `TPFollowerControlPlane`), `TPLeaderFanout` for leaders, then `Stage(...)` (`stage_workers.py:493-605`).
13. **Stage start + ready signal**: `await stage.start()` for all stages in the process — ZMQ control plane up, scheduler thread spawned (`pipeline/stage/runtime.py:146-188`) — then `ready_event.set()` and `asyncio.gather(*(stage.run() ...))` (`stage_workers.py:388-400`).
14. **Parent readiness gate**: `wait_ready` polls each event with the shared deadline; a dead child or timeout raises with the child traceback from the error channel (`stage_workers.py:253-292`); a post-gather liveness re-check catches late deaths (`mp_runner.py:411-416`).
15. **Stage registration + monitoring**: leader endpoints registered on the Coordinator (`mp_runner.py:418-420`); `_monitor_children` task starts (5 s poll, `mp_runner.py:423, 439-449`).
16. **Serving ready**: placement/topology summary logged with per-GPU NVML hardware info (`serve/launcher.py:312-328, 112-148`); `Client` + `create_app` (routes: health/models/chat/speech/transcriptions/realtime, `serve/openai_api.py:80-122`); profiler routes mounted (`launcher.py:338-340`); uvicorn `server.serve()` raced against `mp_runner.wait_failed()` via `_serve_with_failure_watch` (`launcher.py:342-350, 357-392`). The `finally` guarantees `mp_runner.stop()` on any exit (`launcher.py:351-354`).

### 13.2 Explicit abort (the designed path)

Part I §7 sketched the abort hop; the complete chain: `Client.abort` → `Coordinator.abort` (`client/client.py:228-234`; `pipeline/coordinator.py:241-287`): broadcast `AbortMessage` on the PUB socket to **all** stages, set `asyncio.CancelledError` on the completion future, push a synthetic failed `CompleteMessage(error="aborted")` into the stream queue, drop request tracking. On every stage, `_abort_listener` receives the SUB message (leaders also fan it to TP followers) and runs `_on_abort`: record the rid in the bounded `_aborted` set (capped at 10k, trimmed to 5k — `stage/runtime.py:1189-1195`), `relay.cleanup(request_id)`, clear stream/fan-in state, `scheduler.abort(request_id)` (`stage/runtime.py:1176-1201`). Every subsequent message handler early-returns on `request_id in self._aborted` and *drains* the already-relayed payload bytes instead of leaking relay slots (`_discard_payload_data` / `_discard_stream_chunk_data`, `stage/runtime.py:267, 284, 526-556`).

Inside `OmniScheduler.abort` the cleanup is two-mode (`scheduling/omni_scheduler.py:824-853`; Part I §3.3 covered the FINISH_ABORT half): by default a request found in a live batch is *deferred* — marked `req.to_finish = FINISH_ABORT()` so upstream `process_batch_result` finishes it cleanly next step (`:855-872`); with `defer_running_cleanup=False` (the batch-failure path) KV is released immediately via `release_kv_cache(req, self.tree_cache)` and the rid is scrubbed from `running_batch`/`cur_batch`/`last_batch`/async-pending (`:847-852, 874-893`). Model-local `abort_callback`s clean per-model side tables (Qwen3-TTS prepared requests, `models/qwen3_tts/stages.py:158, 266`).

### 13.3 Client disconnect: only the realtime WebSocket aborts; plain HTTP does not

The sharpest operational finding in Part II. The realtime path wires disconnect → abort explicitly: `RealtimeSession.teardown` and `handle_response_cancel` call `await self.client.abort(request_id)` then cancel the task (`serve/realtime/session.py:218-223, 434-453`, verified — these are the only two `client.abort` call sites under `serve/`), with a documented reason for using `gather(..., return_exceptions=True)` over `.exception()` ("turning a normal disconnect into a handler exception", `session.py:438-442`).

The plain HTTP SSE path has **no such wiring**. `_chat_stream` is a bare async generator with no `try/finally` abort (`serve/openai_api.py:275-385`); a repo-wide grep for `.abort(` under `serve/` and `client/` hits only the two realtime sites plus the `Client.abort` definition itself (verified). When the HTTP client disconnects, starlette cancels the generator; the cancellation propagates into `Coordinator.stream`, whose `finally` merely unregisters the queues (`pipeline/coordinator.py:178-180`):

```python
finally:
    self._stream_queues.pop(request_id, None)
    self._completion_futures.pop(request_id, None)
```

…but `self._requests[request_id]` survives and **no `AbortMessage` is broadcast**, so the pipeline keeps decoding until natural completion; the eventual `CompleteMessage` is then absorbed by `_handle_completion` (request still tracked, futures gone — `coordinator.py:326-349, 363-372`). Disconnected streaming requests therefore burn GPU until finish — abandoned-request KV is reclaimed only at terminal completion, not at disconnect.

### 13.4 Forward-pass failure: sentinel at the scheduler, fail-fast at the coordinator

Part I §3.3 covered the `_FAILED_BATCH_RESULT` sentinel; the full chain when a model forward throws:

1. `run_batch` catches everything, calls `_handle_batch_failure`, returns the sentinel so the event loop skips result processing (`scheduling/omni_scheduler.py:602-607`).
2. `_handle_batch_failure` per request: emit `OutgoingMessage(type="error")` then `self.abort(req.rid, defer_running_cleanup=False)` — immediate KV release (`omni_scheduler.py:710-716, 591-600`). Error emission is entry-rank-only so TP followers stay silent (`:592-593`).
3. `Stage._drain_outbox_external` maps `type=="error"` → `_send_failure` (`stage/runtime.py:705-706`), which marks the rid aborted locally and sends `CompleteMessage(success=False, error=...)` straight to the coordinator, then clears all per-request stage state (`stage/runtime.py:1126-1153`). On a TP **follower**, `_send_failure` instead raises — deliberately killing the rank process, since a follower cannot reach the coordinator (`:1128-1130`).
4. Coordinator `_handle_completion` fail-fast branch: mark FAILED, **broadcast abort to all stages** (cleaning the request out of every other stage of the DAG), set the exception on the future / push the failed message to the stream queue (`pipeline/coordinator.py:334-349`).
5. HTTP layer converts the raised error to 400 vs 500 by substring-matching context-length markers (`_is_bad_request_error` + `_BAD_REQUEST_MARKERS`, `serve/openai_api.py:69-77, 221-229`).

Related per-request rejections never reach the GPU: malformed builds and KV-capacity violations are caught in `process_input_requests` and emitted as errors pre-queue, the capacity message embedding an actionable `--thinker-mem-fraction-static` hint (`omni_scheduler.py:483, 500-508, 567-589`).

### 13.5 Scheduler-thread crash and process death: escalation ladder

- **Scheduler thread dies**: the thread wrapper catches, flips `_running`, and schedules `_handle_scheduler_crash` onto the asyncio loop via `run_coroutine_threadsafe` (`stage/runtime.py:170-179`). That coroutine fails every active request individually (`scheduler.abort` + `_send_failure` + `relay.cleanup`) before closing the control plane (`:1155-1174`); `run()`'s `finally` then re-raises a `RuntimeError(... crashed) from exc` (`:245-248`), which collapses the process (`asyncio.gather` semantics documented at `stage_workers.py:367-380`, Part I §2).
- **Background task (abort listener / outbox drain) dies**: `_on_background_task_done` stores the exception, stops the loop, closes the control plane; `run()`'s `finally` re-raises it (`stage/runtime.py:1236-1251, 243-244`).
- **Process dies**: `_monitor_children` (5 s poll) → `_fail_runtime` → `coordinator.fail_pending_requests(error)` (every tracked request: future exception + failed stream message, `pipeline/coordinator.py:105-127`) → `_fatal_event.set()` → `self.stop()` (`pipeline/mp_runner.py:439-457`; Part I §4 covered the no-respawn policy). The launcher's failure watcher sees `wait_failed()` complete, sets `server.should_exit = True`, and re-raises the fatal error out of `_run_server` (`serve/launcher.py:357-392`). Shutdown itself is graceful-then-violent: ZMQ `Shutdown` to each stage, `join(30s)` → `terminate()` → `kill()` (`mp_runner.py:494-504`; `stage_workers.py:316-335`), then the IPC tempdir is removed (`mp_runner.py:477-481, 512`).

### 13.6 Observability surfaces

**Request-level event tracing (the primary built-in instrument).** A process-local JSONL recorder (`profiler/event_recorder.py`) is the system's request-flow tracer: each process appends `RequestEvent{request_id, stage, event_name, timestamp_ns, run_id, pid, metadata}` to `<dir>/events_<stage>_<pid>.jsonl` (`event_recorder.py:2-7, 60-73, 100-138`). Notable engineering details:

- **Active-stage binding** for emit sites that can't plumb a stage name: `Stage._run_scheduler` binds the stage on the scheduler thread (`stage/runtime.py:157-159`), via a contextvar + thread-local pair documented to cover both `asyncio.to_thread` and plain `threading.Thread` (`event_recorder.py:24-34, 194-197`).
- **Tensor-safe serialization**: the JSON fallback summarizes tensors as `{__tensor_summary__, shape, dtype, device}` and "never materialise[s]" them (`event_recorder.py:221-249`).
- Emission is a no-op when inactive and write errors are swallowed after the first warning (`event_recorder.py:186-218`) — tracing can stay compiled-in on hot paths.

The event taxonomy spans the whole request path (grep-verified emit sites): coordinator `request_admission` / `terminal_response` / `coordinator_stream_received` (`pipeline/coordinator.py:216-221, 316-324, 403-422`); stage shell `stage_input_received`, `stage_aggregate_ready`, `stage_dispatch`, `stage_hop_sent`, `stage_complete`, `stage_first_stream_chunk_sent`/`stage_stream_chunk_sent`/`_received` (`stage/runtime.py:275, 379, 644, 841, 757, 989-995`); scheduler `scheduler_queue_enter`, `scheduler_request_build_start/end`, `scheduler_prefill_start`, `scheduler_first_emit` (`scheduling/omni_scheduler.py:477, 495, 517, 659, 732`). The analysis layer (`profiler/views.py`) merges the per-process files by request id, reconstructs per-request timelines, computes stage/hop interval breakdowns with p50/p95 percentiles (`views.py:111, 200-250, 311, 186-198`), and explicitly defines latency pairs — `("scheduler_prefill_start", "scheduler_first_emit")` is the thinker-TTFT interval (`views.py:135-142`).

**Torch profiling, controlled over the same ZMQ control plane.** Profiling is runtime-toggleable over HTTP without restart. The serving app mounts `/start_profile`, `/stop_profile`, `/start_request_profile`, `/stop_request_profile` (`serve/launcher.py:172-284`); these (a) start/stop the coordinator-side event recorder and (b) broadcast `ProfilerStartMessage`/`ProfilerStopMessage` to every stage through `ProfilerControlClient` — plain PUSH sockets to the *same* per-stage control endpoints used for request submission (`profiler/profiler_control.py:14-87`). Stages handle the messages in their main loop (`stage/runtime.py:260-263`), TP leaders fan them to followers (`:215-223`), and `_on_profiler_start` starts a `TorchProfiler` with a per-pid trace template plus the local event recorder (`:1203-1222`).

`TorchProfiler` itself (adapted from vLLM-Omni, attribution at `profiler/torch_profiler.py:2-4`) records **continuously** between start/stop (no schedule), exports a chrome trace on stop, and offloads gzip to a `subprocess.Popen` "to avoid blocking the worker loop" (`torch_profiler.py:23-28, 108-126, 164-180`). Expensive capture flags are env-var opt-in with a documented size rationale ("default off keeps the trace tens of MB; all on can hit multi-GB"): `SGLANG_TORCH_PROFILER_RECORD_SHAPES` / `_PROFILE_MEMORY` / `_WITH_STACK` / `_WITH_FLOPS` (`torch_profiler.py:108-120`); `SGLANG_TORCH_PROFILER_DIR` provides default output roots (`serve/launcher.py:338, 181-195`). Stop with `run_id=None` is a wildcard, and mismatched run_ids are ignored with a warning on both recorder and profiler (`event_recorder.py:140-155`; `torch_profiler.py:143-150`).

**Health, logs, and what's *not* there.**

- `/health` returns 200/503 keyed on coordinator liveness plus live request-state counts (`serve/openai_api.py:125-139`; `Coordinator.health` aggregates per-state counters and pending completions, `pipeline/coordinator.py:462-476`).
- Startup logging doubles as a placement audit: the resolved plan log includes process groups, TP rank→process maps, per-stage memory budgets, and per-GPU hardware (name/total memory via pynvml, best-effort) (`serve/launcher.py:112-148, 320-323`; `utils/gpu_memory.py:23-28, 305+`).
- Logging setup is plain `logging.basicConfig` twice — once in the CLI parent (`cli/serve.py:1011-1014`), once per stage process to stdout (`stage_workers.py:344`). The env-var framework in `environ.py` defines exactly one variable, `SGLOMNI_LOG_LEVEL = EnvStr("INFO")` (`environ.py:93-98`), and a repo-wide grep finds **no consumer** of `OMNIENV`/`SGLOMNI_LOG_LEVEL` outside `environ.py` (verified) — scaffolding that never got wired.
- **No metrics endpoint**: grep for `prometheus`/`Histogram` over the non-vendor tree returns nothing (the only `Counter` is `collections.Counter` in topology planning, `config/topology.py:184`). Operational visibility is logs + JSONL events + torch traces + `/health`; there is no time-series metrics surface at all.

---

## 14. Engineering details worth copying (and anti-patterns)

Part I §8 listed borrowable *architecture* ideas (mailbox contract, launch/resolve, pipeline-as-data, sentinel error boundary, composition-over-inheritance, comment culture, file vocabulary, cost-budget batching, decode gating). Part II surfaces a second tier of *executor/operations* patterns:

**Worth copying:**

1. **Deferred CUDA-graph capture as a first-class idiom.** Construct the engine with graphs off, mutate the model (hooks, compiled submodules, attached tokenizers, sampler references), then call `init_device_graphs()` yourself (§10.2.2; `models/qwen3_omni/bootstrap.py:31-59, 133-159`). The companion rule: **pin the capture-hidden mode** so `can_run` never silently falls off the replay path (`thinker_model_runner.py:78-96`). For wm-infra, any post-construction model surgery (LoRA fusion, compile, hook installation) before graph capture should follow exactly this order, with the same explicit comment explaining the replay-killer being avoided.
2. **In-graph sampling + auxiliary-head decoding via static buffers** (§10.2.3). The recipe — host pre-writes dynamic params into `[max_bs, ...]` static buffers, freezes sampler control flow, gives auxiliary modules a private fixed-shape KV cache, mirrors outputs to static buffers read after replay — is the maximal-capture playbook for any AR loop with per-step auxiliary computation (e.g. a world-model step that samples and then runs a reward/value head): `talker.py:911-1000, 1167-1190, 1349-1414, 1244-1249`.
3. **Drift tripwires where you depend on un-versioned upstream internals.** The talker rollback hard-fails with `TypeError("...sglang upstream prepare_for_decode changed; update rollback.")` (`talker_scheduler.py:128-130`) instead of silently corrupting the KV ledger; the double-free side is an upstream assert (`schedule_batch.py:801-807`). Pair every "manual inverse of someone else's mutation" with a cheap invariant check that names the assumed contract.
4. **Process-scoped GPU memory accounting for colocated engines.** NVML per-process bytes (`nvmlDeviceGetComputeRunningProcesses` filtered to `os.getpid()`, with CUDA_VISIBLE_DEVICES remap) + a per-stage `total_gpu_memory_fraction` budget, falling back to a load delta that is only sound because an flock serializes loads (§11.1; `sglang_model_runner.py:119-237`; `utils/gpu_memory.py:84-135, 223-248, 289-302`). Directly relevant to wm-infra's colocated trainer+engine GPU topologies: global `mem_get_info` deltas are wrong the moment two processes load on one device.
5. **Config ownership validation instead of precedence rules.** Double-setting one knob through two namespaces is a hard error (`config/runtime.py:75-108`), framework-injected kwargs are signature-gated (`config/runtime.py:34-53`), CLI override passes validate atomically before mutating (`cli/serve.py:225-228`), and free-form dotted overrides are backstopped by reconstructing the `extra="forbid"` Pydantic model (`config/manager.py:102`; `config/schema.py:14, 138, 204`). This combination makes "where did this value come from" always answerable — the same property wm-infra's config-resolve aims at.
6. **A JSONL request-event recorder beats a metrics framework for pipeline debugging.** Per-process append-only files keyed by request id, contextvar/thread-local stage binding, tensor-safe serialization, swallowed write errors, merged offline into per-request timelines with named latency pairs (TTFT = `("scheduler_prefill_start", "scheduler_first_emit")`) — §13.6; `profiler/event_recorder.py`, `profiler/views.py:135-142`. The whole instrument is ~2 files and would map directly onto wm-infra's phase-transition events.
7. **Reuse the control plane for operations traffic.** Profiler start/stop rides the same per-stage ZMQ endpoints as request submission (`profiler/profiler_control.py:14-87`) — no second RPC system to keep alive.
8. **Coordinator-before-spawn ordering** (§13.1 step 8): bind every listener before the first child exists, so readiness/completion can never race the listener (`mp_runner.py:385-400`). Trivial to copy, painful to retrofit.

**Anti-patterns observed (don't copy):**

1. **HTTP SSE disconnect does not abort** (§13.3) — disconnected streaming clients burn GPU until natural completion because only the realtime WebSocket wires teardown → `client.abort`. Any streaming API in wm-infra should put the abort in the generator's `finally` from day one (`serve/openai_api.py:275-385`; `pipeline/coordinator.py:178-180`).
2. **Dead plumbing left in place**: `filter_weights_by_prefix` + the `weight_prefix` field on `ModelWorkerConfig` have zero call sites (`sglang_model_runner.py:25-35`); the real prefix handling lives inside the model's `load_weights` (`talker.py:1448-1453`). Two mechanisms for one job, one of them dead.
3. **Unwired scaffolding**: `environ.py` defines an env-var framework with exactly one variable that nothing reads (`environ.py:93-98`, grep-verified). Either wire it or delete it.
4. **No metrics surface at all** (§13.6) — fine for a research serving stack, but the JSONL recorder is request-scoped and opt-in; there is nothing to alert on. A production deployment would need at least queue-depth/KV-utilization gauges.
5. (Carried from Part I §5: design-doc references that dangle — `design.md §1.4` cited from code but absent from the checkout.)

---

## 15. Part II source-of-truth index

One merged table for §10–§14. Upstream rows are read from the local sglang clone at tag `v0.5.8` (`git -C ~/Desktop/sglang show v0.5.8:<path>`), the exact pinned version (`pyproject.toml:29`).

| Claim | Evidence |
|---|---|
| **§10 executor** | |
| Wrapper batch build: get_model_worker_batch + capture-mode override + ForwardBatch.init_new | `sglang_omni/model_runner/base.py:212-248` |
| Prefill ragged packing, list→tensor H2D non_blocking, mm feature moves/CUDA-IPC | `sglang@v0.5.8: srt/managers/schedule_batch.py:1471-1516, 1631-1645` |
| Decode reuses GPU output_ids; in-place seq_lens.add_(1) (out-of-place under overlap) | `sglang@v0.5.8: schedule_batch.py:1951-2014, 1989` |
| alloc_for_decode writes req_to_token page table | `sglang@v0.5.8: srt/mem_cache/common.py:425-464` |
| ModelWorkerBatch dataclass; ForwardBatch.init_new positions/extend-lens H2D | `sglang@v0.5.8: schedule_batch.py:2168-2244, 2340`; `srt/model_executor/forward_batch_info.py:379-528, 1086` |
| Pools handed to omni scheduler; pinned ping-pong host staging | `sglang_omni/model_runner/model_worker.py:128-132`; `sglang_omni/model_runner/base.py:69-91` |
| Thinker mm-embedding scatter + _omni_consumed chunking; manual init_forward_metadata | `sglang_omni/model_runner/thinker_model_runner.py:102-162, 265-312, 275` |
| Talker projected prefill reassembly + retract replay | `sglang_omni/models/qwen3_omni/talker_model_runner.py:158-336` |
| Feedback buffers: host write + in-graph torch.where | `talker_model_runner.py:338-364`; `sglang_omni/models/qwen3_omni/components/talker.py:395-405, 425-434` |
| init_device_graphs gating + graph mem delta | `sglang@v0.5.8: srt/model_executor/model_runner.py:1979-2017` |
| Capture bs list; DECODE-only; reversed capture; warmups; shared pool | `sglang@v0.5.8: srt/server_args.py:1039-1068`; `srt/model_executor/cuda_graph_runner.py:189-212, 270-302, 472-534, 686-738` |
| GraphInputBuffers create/populate (padding fills) | `sglang@v0.5.8: srt/model_executor/input_buffers.py:16-131, 133-207` |
| can_run, replay padding via bisect, output slicing, recapture on hidden-mode change | `sglang@v0.5.8: cuda_graph_runner.py:381-446, 772-840, 842-885, 740-771`; `model_runner.py:2277-2300` |
| Deferred capture idiom (thinker/talker/asr/tts); sampler-before-capture; enable_return_hidden_states FULL | `sglang_omni/models/qwen3_omni/bootstrap.py:31-59, 133-159`; `models/qwen3_asr/stages.py:63-105`; `models/qwen3_tts/stages.py:109-124, 202-242`; `sglang@v0.5.8: cuda_graph_runner.py:300-302` |
| Hidden hooks installed pre-capture; NULL pin to keep replay | `sglang_omni/model_runner/_hidden_capture.py:26-76`; `scheduling/bootstrap.py:40-46`; `thinker_model_runner.py:78-96` (verified verbatim) |
| Custom forwards set can_run_cuda_graph=False; DeepGEMM JIT avoidance | `thinker_model_runner.py:310-312`; `talker_model_runner.py:523-526`; `model_worker.py:313-323` |
| Talker in-graph sample+RVQ: static sampling info, masks, private static KV, output mirrors, padded-replay index | `talker.py:1083-1095 (verified), 911-1000, 1167-1190, 818-828, 1349-1414, 1244-1249, 1192-1216`; host reads `talker_model_runner.py:93-136` |
| Attention registry + selection + HW defaults | `sglang@v0.5.8: srt/layers/attention/attention_registry.py:12-45`; `model_runner.py:1596-1668`; `server_args.py:1655-1700, 4768` |
| Per-batch metadata contract + FA3 decode metadata (page_table gather, seq_lens_cpu max) | `sglang@v0.5.8: model_runner.py:2128-2178`; `srt/layers/attention/flashattention_backend.py:389-481` |
| Startup order: pre-load mem snapshot → load → pools → attn → graphs | `sglang@v0.5.8: model_runner.py:263-399 (365, 383), 414-595 (459-460, 555, 571-575)` |
| load_model delta accounting; loader stream + post-load quant; arch via registry | `sglang@v0.5.8: model_runner.py:805-948`; `srt/model_loader/loader.py:640-683, 229-234, 471-528` |
| Omni registry injection + sub-arch config override; in-model "talker." strip; dead filter helper | `sglang_omni/model_runner/sglang_model_runner.py:81-117, 25-35 (no call sites)`; `model_worker.py:24-31, 85-114`; `talker.py:1448-1453` |
| Upstream sample path + Sampler kernel | `sglang@v0.5.8: model_runner.py:2352-2399`; `srt/layers/sampler.py:67-160` |
| Sample-before-post hooks; pytorch sampling backend (#408 RNG bug) | `sglang_omni/model_runner/base.py:296-334, 485`; `talker_model_runner.py:138-148`; `models/qwen3_omni/stages.py:1049-1056` (verified) |
| Token travel: GPU loop, tolist sync, GenerationBatchResult bridge, req.output_ids append, stream emit | `base.py:293, 333-334`; `sglang@v0.5.8: schedule_batch.py:1989`; `sglang_omni/scheduling/sglang_backend/output_processor.py:34-38, 82-128, 149, 185-212`; `scheduling/omni_scheduler.py:665-677`; `sglang@v0.5.8: srt/managers/scheduler_output_processor_mixin.py:359-405`; `talker_model_runner.py:111-136` |
| **§11 memory** | |
| Pools built in-process per stage | `sglang_omni/scheduling/bootstrap.py:48-55` |
| `ReqToTokenPool` struct (int32 matrix + Python free list); max_context_len +4 | `sglang@v0.5.8: srt/mem_cache/memory_pool.py:117-164`; `model_runner_kv_cache_mixin.py:333-390` |
| KV buffers `[size+page_size, heads, dim]` per layer; slot 0 dummy | `sglang@v0.5.8: srt/mem_cache/memory_pool.py:754-778` |
| Token allocator free list, free-group, release_pages | `sglang@v0.5.8: srt/mem_cache/allocator.py:35-165` |
| Paged allocator (Triton alloc kernels, page collapse on free) | `sglang@v0.5.8: srt/mem_cache/allocator.py:290-453` |
| Allocator selection; page_size default 1 ⇒ token-granular in omni | `sglang@v0.5.8: model_runner_kv_cache_mixin.py:586-633`; `server_args.py:1917-1919`; `sglang_omni/scheduling/sglang_backend/server_args_builder.py:21-31` |
| KV sizing formula + cell size + init_memory_pool + mem-fraction error | `sglang@v0.5.8: model_runner_kv_cache_mixin.py:47-144, 252-330`; `srt/utils/common.py:494-521` |
| Colocated NVML/process profiling override (docstring on `_profile_available_bytes`, entry `profile_max_num_token`) + budget math + flock | `sglang_omni/model_runner/sglang_model_runner.py:119-161 (verified verbatim), 163-198, 221-259`; `utils/gpu_memory.py:84-135, 223-248, 289-302`; `pipeline/stage_workers.py:608-621` |
| Encoder mem reserve + 0.1 floor | `sglang_omni/scheduling/sglang_backend/server_args_builder.py:40-62` |
| `TreeNode` / `RadixKey` structs, extra_key namespacing, root pinned | `sglang@v0.5.8: srt/mem_cache/radix_cache.py:62-117, 185-193, 322-328, 344-356` |
| match / split / insert walkthrough | `sglang@v0.5.8: radix_cache.py:627-672, 674-715` |
| Eviction loop (leaf heap), allocation-driven trigger | `sglang@v0.5.8: radix_cache.py:548-573 (verified verbatim), 754-766`; `common.py:201-251, 407` |
| LRU strategy default; CacheInitParams defaults | `sglang@v0.5.8: srt/mem_cache/evict_policy.py:16-18`; `radix_cache.py:284-299`; `cache_init_params.py:13-35` |
| `inc/dec_lock_ref` path pinning; lock at prefill admission | `sglang@v0.5.8: radix_cache.py:575-612`; `srt/managers/schedule_policy.py:552-557, 586-598, 753` |
| Req indirection fields; `init_next_round_input` | `sglang@v0.5.8: srt/managers/schedule_batch.py:543-546, 590, 635-646, 861-907` |
| alloc_for_extend / write_cache_indices / alloc_for_decode | `sglang@v0.5.8: srt/mem_cache/common.py:78-107, 329-393, 425-464` |
| Decode ledger increments; double-free asserts | `sglang@v0.5.8: schedule_batch.py:795-820, 1999-2002` |
| `cache_finished_req` adopts KV, frees duplicates, dec_lock_ref; `cache_unfinished_req` lock migration | `sglang@v0.5.8: radix_cache.py:433-477, 479-539` |
| `release_kv_cache` callers in omni | `sglang@v0.5.8: srt/mem_cache/common.py:467-497`; `sglang_omni/scheduling/omni_scheduler.py:890-893`; `scheduling/dllm_scheduler.py:134-136`; `models/fishaudio_s2_pro/fish_scheduler.py:232-235` |
| Talker rollback ledger decrements + tripwire (verified) | `sglang_omni/models/qwen3_omni/talker_scheduler.py:110-143` |
| Async-decode stale-batch double-free guard | `sglang_omni/scheduling/omni_scheduler.py:1092-1112` |
| Retraction: no tree insert, re-queue to prefill | `sglang@v0.5.8: schedule_batch.py:1846-1906`; `sglang_omni/scheduling/sglang_backend/decode.py:36-78` |
| Abort keeps prefix in tree (is_insert=True); releases across all live batches | `sglang_omni/scheduling/omni_scheduler.py:874-893` |
| No flush/weight-update/reset path (negative grep) | `grep -rn "flush_cache\|update_weights\|reset_prefix" sglang_omni/` → 0 scheduler hits; `radix_cache.py:300, 322-332` |
| ChunkCache = request-lifetime semantics; evict short-circuit | `sglang@v0.5.8: srt/mem_cache/chunk_cache.py:25-84`; `common.py:233-234` |
| Hierarchical cache hard-disabled; TreeNode host hooks unused | `sglang_omni/scheduling/omni_scheduler.py:245-247`; `scheduling/sglang_backend/cache.py:28-33`; `sglang@v0.5.8: radix_cache.py:106-136` |
| StageOutputCache LRU struct (dual budgets, oversize skip) | `sglang_omni/scheduling/stage_cache.py:11-98` |
| `cpu_offload_gb` = weights-only offload, thinker CLI | `sglang_omni/cli/serve.py:461-485, 851-856`; `sglang@v0.5.8: srt/utils/offloader.py:64-68` |
| **§12 extension** | |
| Registry scans `models/*` for `config.EntryClass`; arch-keyed; lru_cached; lenient imports | `sglang_omni/models/registry.py:27-44, 67-84 (verified), 135-136` |
| Duplicate-arch registration hard error; `Variants` lookup by class name | `sglang_omni/models/registry.py:73-83, 121-132`; `config/manager.py:138-139` |
| Arch detection: AutoConfig → raw config.json → mistral params.json; model_type map | `sglang_omni/config/manager.py:16-31`; `sglang_omni/utils/hf.py:26-47, 50-54, 72-125` |
| Qwen3-TTS pipeline = 3 StageConfigs + `EntryClass`, arch ClassVar | `sglang_omni/models/qwen3_tts/config.py:13-45` |
| Pipeline graph validation at construction | `sglang_omni/config/schema.py:222-227, 282-364` |
| Factory = dotted path resolved in child; returns scheduler-contract object | `sglang_omni/pipeline/stage_workers.py:29-39, 608-621`; `scheduling/omni_scheduler.py:71-72` |
| AR factory recipe (ServerArgs → infra → adapters → OmniScheduler) | `sglang_omni/models/qwen3_tts/stages.py:162-267` |
| Adapter contract + 14-line closure factory | `sglang_omni/scheduling/omni_scheduler.py:83-116`; `models/qwen3_tts/request_builders.py:819-832` |
| Per-step hook subclass example | `sglang_omni/models/qwen3_tts/model_runner.py:16-30` |
| Prepared-request handoff across co-located stages + abort cleanup | `sglang_omni/models/qwen3_tts/request_builders.py:666-710`; `stages.py:154-159, 266` |
| Two-tier CLI (`allow_extra_args`); dotted merge with list indexing; reconstruct-revalidate; type inference; TorchTitan credit | `sglang_omni/cli/__init__.py:10-12`; `cli/serve.py:775-1009, 1036-1037`; `config/manager.py:35-40, 45-70, 78-103, 225-245` |
| `extra="forbid"` as override backstop | `sglang_omni/config/schema.py:14, 138, 204` |
| tp_size ↔ parallelism.tp alias coherence | `sglang_omni/config/manager.py:198-222`; `config/schema.py:178-198` |
| Atomic CLI override validation; per-role precedence | `sglang_omni/cli/serve.py:210-271 (217-228)` |
| Role-map classmethods gate generic flags; factory-pinned flags | `sglang_omni/config/schema.py:239-272`; `cli/serve.py:650-655, 726-733` |
| Three config namespaces + ownership validation + signature-gated injection | `sglang_omni/config/runtime.py:15-55, 58-122, 142-161` |
| Shared sglang ServerArgs defaults + overrides | `sglang_omni/scheduling/sglang_backend/server_args_builder.py:10-37` |
| `env_defaults` injected at spawn, conflict-checked | `sglang_omni/pipeline/stage_workers.py:136-167` |
| Console scripts; `sgl-omni config view/export` | `pyproject.toml:75-77`; `sglang_omni/cli/config.py:13-73` |
| **§13 startup & failure** | |
| Startup steps 1–5 (CLI → launch → port probe → runner timeout env) | `cli/serve.py:1011-1098`; `serve/launcher.py:301-306, 395-429` |
| Startup steps 6–9 (plans, coordinator-before-spawn, spawn) | `pipeline/mp_runner.py:366-407`; `pipeline/runtime_config.py:74-113, 162-174`; `stage_workers.py:224-244` |
| Startup steps 10–14 (child boot, GPU flock, model load, ready gate) | `stage_workers.py:338-410, 413-427, 608-621, 253-292`; `utils/gpu_memory.py:290-302`; `scheduling/bootstrap.py:9-84`; `model_runner/model_worker.py:50-53, 166-185, 242-333` |
| Startup steps 15–16 (register, monitor, uvicorn + failure watch) | `pipeline/mp_runner.py:411-423, 439-449`; `serve/launcher.py:312-355, 357-392` |
| Coordinator.abort: PUB broadcast, CancelledError, synthetic fail message | `sglang_omni/pipeline/coordinator.py:241-287` |
| Stage abort listener, bounded aborted set, payload draining | `sglang_omni/pipeline/stage/runtime.py:1176-1201, 1189-1195, 526-556` |
| OmniScheduler.abort dual mode (FINISH_ABORT vs immediate KV release) | `sglang_omni/scheduling/omni_scheduler.py:824-893` |
| Realtime WS abort wiring (only disconnect→abort sites, verified) | `sglang_omni/serve/realtime/session.py:218-223, 434-453`; `client/client.py:228-234` |
| **HTTP SSE disconnect does NOT abort** — coordinator only pops queues | `sglang_omni/serve/openai_api.py:275-385`; `pipeline/coordinator.py:178-180, 326-349`; grep `.abort(` over `serve/`+`client/` (verified: only realtime + Client.abort def) |
| Batch-failure chain: sentinel → error msg → abort → CompleteMessage(fail) → coordinator broadcast-abort | `scheduling/omni_scheduler.py:602-607, 710-716, 591-600`; `stage/runtime.py:705-706, 1126-1139`; `coordinator.py:334-349` |
| Follower `_send_failure` raises (kills rank) | `stage/runtime.py:1128-1130` |
| HTTP 400-vs-500 by message markers | `serve/openai_api.py:69-77, 221-229` |
| Pre-GPU rejections (malformed build, KV capacity with CLI hint) | `scheduling/omni_scheduler.py:483, 500-508, 567-589` |
| Scheduler-thread crash escalation (threadsafe handoff, fail-all, re-raise) | `stage/runtime.py:156-179, 1155-1174, 243-248, 1236-1251` |
| Dead-process escalation; graceful-then-violent shutdown | `pipeline/mp_runner.py:439-457, 477-512`; `coordinator.py:105-127`; `serve/launcher.py:357-392`; `stage_workers.py:316-335` |
| JSONL event recorder design (stage binding, tensor-safe JSON, swallowed errors) | `profiler/event_recorder.py:2-7, 24-34, 100-138, 186-249`; `stage/runtime.py:157-159` |
| Event taxonomy emit sites | `coordinator.py:216-221, 316-324, 403-422`; `stage/runtime.py:275, 379, 644, 757, 841, 989-995`; `omni_scheduler.py:477, 495, 517, 659, 732` |
| views.py timelines, percentiles, TTFT interval pairs | `profiler/views.py:111-142, 186-250, 311` |
| Profiler HTTP routes + ZMQ broadcast + per-stage handling + TP fanout | `serve/launcher.py:172-284, 338-340`; `profiler/profiler_control.py:14-87`; `stage/runtime.py:215-223, 260-263, 1203-1234` |
| TorchProfiler: continuous, gzip subprocess, env opt-in flags, wildcard stop | `profiler/torch_profiler.py:2-4, 23-28, 108-126, 143-188` |
| /health 200/503 + request-state counts | `serve/openai_api.py:125-139`; `coordinator.py:462-476` |
| `SGLOMNI_LOG_LEVEL` declared, never consumed | `sglang_omni/environ.py:93-98`; grep `OMNIENV|SGLOMNI_LOG_LEVEL` (verified, no hits outside environ.py) |
| No Prometheus/metrics surface | grep `prometheus|Histogram|Counter(` non-vendor (only `collections.Counter` at `config/topology.py:184`) |
