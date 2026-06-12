# sglang — Architecture Reading

*Read at local checkout `/home/mingfeiguo/Desktop/sglang`, commit `492883c8c` (2026-04). Paths are
relative to the repo root; `srt/...` is shorthand for `python/sglang/srt/...`. Citations were
spot-checked against the checkout (Engine docstring, event loops, batch-flow docstring, FutureMap,
`rem_total_tokens`, Ray actor, running-batch merge all verified verbatim).*

## 1. Repo layout & module organization

Top level: `python/` (the main `sglang` Python package), `sgl-kernel/` (CUDA/C++ kernels: `csrc/`,
`CMakeLists.txt`, `cmake/`), `sgl-model-gateway/` (a Rust crate: `Cargo.toml`, `build.rs`,
`benches/`, `bindings/`), `rust/` (contains only `sglang-grpc`), plus `docs/`, `benchmark/`,
`test/`, `examples/`, `docker/`, `scripts/`, `proto/`, `3rdparty/`.

Package metadata: `python/pyproject.toml` — `name = "sglang"`, console scripts
`sglang = "sglang.cli.main:main"` and `killall_sglang` (`python/pyproject.toml:181-183`); described
as "a fast serving framework for large language models and vision language models"
(`python/pyproject.toml:8`).

Inside `python/sglang/`: `lang/` (frontend language), `srt/` (SGLang RunTime — the serving
engine), `cli/`, `launch_server.py`, plus bench scripts (`bench_serving.py`, `bench_one_batch.py`).

`python/sglang/srt/` ownership map:

| Directory | Owns |
|---|---|
| `entrypoints/` | `engine.py` (Engine API + subprocess launch), `http_server.py` (FastAPI), `grpc_server.py`, `openai/`, `anthropic/`, `ollama/` API adapters |
| `managers/` | The 3-process core: `tokenizer_manager.py`, `scheduler.py` (+ `schedule_batch.py`, `schedule_policy.py`, many `scheduler_*_mixin.py`), `detokenizer_manager.py`, `tp_worker.py`, `data_parallel_controller.py`, `io_struct.py` (IPC message dataclasses) |
| `model_executor/` | `model_runner.py`, `forward_batch_info.py`, `cuda_graph_runner.py`, `piecewise_cuda_graph_runner.py` |
| `mem_cache/` | `memory_pool.py` (`ReqToTokenPool:127`, `KVCache:668`, `MHATokenToKVPool:764`), `allocator.py` (`TokenToKVPoolAllocator:117`, `PagedTokenToKVPoolAllocator:356`), `radix_cache.py` + variants (`swa_radix_cache.py`, `mamba_radix_cache.py`, `hiradix_cache.py`, `radix_cache_cpp.py`, `unified_radix_cache.py`), `chunk_cache.py` |
| `distributed/` | `parallel_state.py` (vLLM-style process groups), `device_communicators/`, `communication_op.py` |
| `ray/` | Optional Ray actor backend: `engine.py` (RayEngine), `scheduler_actor.py`, `data_parallel_controller.py`, `http_server.py` |
| `layers/`, `models/`, `model_loader/` | NN layers, model definitions, weight loading |
| `disaggregation/` | Prefill/decode disaggregation (PD), encoder disaggregation servers |
| `speculative/`, `lora/`, `constrained/`, `sampling/` | Spec-decoding workers, LoRA, grammar backends, sampling |
| `weight_sync/` | `tensor_bucket.py` (FlattenedTensorBucket), `utils.py` — RLHF-style weight update helpers |
| `checkpoint_engine/`, `elastic_ep/`, `eplb/`, `multiplex/` | Weight update via IPC, elastic expert parallel, expert load balancing, pd-multiplexing |

File-size philosophy: hot-path files are huge and tolerated (`scheduler.py` 3823 lines,
`model_runner.py` 3272, `schedule_batch.py` 2792, `tokenizer_manager.py` 2757, `io_struct.py`
2100, `memory_pool.py` 2067), while single-purpose peripherals are tiny
(`scheduler_recv_skipper.py` 38 lines, `scheduler_input_blocker.py` 106, `prefill_delayer.py`
304). The split criterion is coupling to scheduler hot state, not line count.

## 2. Architecture overview

The engine is explicitly a 3-component, multi-process design. From the `Engine` docstring
(`srt/entrypoints/engine.py:146-158`, verified):

```python
- The engine consists of three components:
    1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
    2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
    3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.
Note:
1. The HTTP server, Engine, and TokenizerManager all run in the main process.
2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
```

`launch_server` (HTTP mode) reuses the exact same launcher: `Engine._launch_subprocesses(...)`
then runs uvicorn/FastAPI in the main process (`srt/entrypoints/http_server.py:2313-2350`).

```
                         main process                    subprocesses (mp.Process, spawn)
┌──────────────────────────────────────────┐
│ FastAPI HTTP server (uvicorn)            │
│   └─> TokenizerManager (asyncio)         │   PUSH scheduler_input_ipc_name
│         send_to_scheduler ───────────────┼──────────────────────────────┐
│         recv_from_detokenizer <──────────┼───────────────┐              ▼
└──────────────────────────────────────────┘               │   ┌────────────────────────────┐
                                                           │   │ Scheduler ×(tp_size·pp_size)│
        ┌──────────────────────────┐  PUSH tokenizer_ipc   │   │  event_loop_normal/overlap  │
        │ DetokenizerManager       │───────────────────────┘   │  ├ TpModelWorker            │
        │  recv_from_scheduler <───┼───────────────────────────│  │  └ ModelRunner (1 GPU)   │
        └──────────────────────────┘  PUSH detokenizer_ipc     │  ├ ScheduleBatch/Policy     │
                                                               │  ├ RadixCache (tree_cache)  │
   Engine.send_to_rpc (DEALER) ── rpc_ipc_name ── DEALER ──>   │  └ req_to_token_pool /      │
                                                               │    token_to_kv_pool_alloc   │
   [dp_size>1: TokenizerManager ─> DataParallelController ─┐  └────────────────────────────┘
    (round-robin/min-load) ── per-dp-rank PUSH sockets ─────┘     ▲ NCCL/gloo collectives
                                                                  │ between scheduler ranks
```

**Process/socket wiring.** `PortArgs` defines one ZMQ IPC endpoint per edge: `tokenizer_ipc_name`
(tokenizer receives outputs from detokenizer), `scheduler_input_ipc_name`,
`detokenizer_ipc_name`, plus `nccl_port` (torch.dist init), `rpc_ipc_name`, `metrics_ipc_name` —
all `ipc://`-tempfile endpoints in the single-node case, switched to TCP when DP-attention /
multi-node is used (`srt/server_args.py:7112-7172`).

- TokenizerManager: `zmq.asyncio` context, `recv_from_detokenizer = PULL(tokenizer_ipc_name)`,
  `send_to_scheduler = PUSH(scheduler_input_ipc_name)` (`srt/managers/tokenizer_manager.py:342-360`);
  its async `handle_loop` awaits `recv_from_detokenizer.recv_pyobj()` and dispatches
  `BatchStrOutput/BatchTokenIDOutput/...` to per-request asyncio futures
  (`tokenizer_manager.py:1623-1636`).
- Scheduler (only attn-TP/CP rank 0 of PP rank 0 owns sockets):
  `recv_from_tokenizer = PULL(scheduler_input_ipc_name)`, `recv_from_rpc = DEALER(rpc_ipc_name)`,
  `send_to_tokenizer = PUSH(tokenizer_ipc_name)`, `send_to_detokenizer =
  PUSH(detokenizer_ipc_name)`; non-rank-0 workers get `None` sockets
  (`srt/managers/scheduler.py:498-538`).
- DetokenizerManager: `recv_from_scheduler = PULL(detokenizer_ipc_name)`, `send_to_tokenizer =
  PUSH(tokenizer_ipc_name)`; its loop is a plain `while True: recv_pyobj → dispatch → send_pyobj`
  (`srt/managers/detokenizer_manager.py:94-99,137-145`).

**Process launch.** `Engine._launch_scheduler_processes` spawns one
`mp.Process(target=run_scheduler_process, ...)` per `(pp_rank, tp_rank)` on this node, computing
`gpu_id = base_gpu_id + (pp_rank % pp_size_per_node)*tp_size_per_node + (tp_rank %
tp_size_per_node)*gpu_id_step`, with an `mp.Pipe` per child for the init handshake
(`srt/entrypoints/engine.py:546-598`). The detokenizer is one more `mp.Process`
(`engine.py:741-748`). Each scheduler subprocess constructs `Scheduler(...)`, sends
`scheduler.get_init_info()` back over the pipe, then blocks in `scheduler.run_event_loop()`; on
exception it SIGQUITs the parent (`srt/managers/scheduler.py:3764-3823`). The parent installs a
`SubprocessWatchdog` over all scheduler + detokenizer processes (`engine.py:769-778`).
`mp.set_start_method("spawn", force=True)` is enforced (`engine.py:1206`).

**Worker stack.** `Scheduler.init_tp_model_worker` instantiates `TpModelWorker` in-process
(`scheduler.py:615-636`); `TpModelWorker.__init__` builds the `ModelRunner`
(`srt/managers/tp_worker.py:217-260`). So one "scheduler process" = Scheduler + TpModelWorker +
ModelRunner on one GPU, optionally plus a draft worker for speculative decoding
(`scheduler.py:638-666`).

**Batch data flow.** Three batch layers with ownership declared in module docstrings
(`schedule_batch.py:20-36`, repeated verbatim in `model_executor/forward_batch_info.py:14-28`;
verified):

```
ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`. ... Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`. It is a subset of `ScheduleBatch` ...
- ForwardBatch is managed by `model_runner.py::ModelRunner`. It contains low-level tensor data.
```

Classes: `Req` (`schedule_batch.py:574`), `ScheduleBatch` (`schedule_batch.py:1352`),
`ModelWorkerBatch` (`schedule_batch.py:2691`).

**Memory/cache layer.** Memory pools are created by the worker and handed up:
`self.req_to_token_pool, self.token_to_kv_pool_allocator = self.tp_worker.get_memory_pool()`
(`scheduler.py:778-780`), then wrapped in a prefix cache chosen by config — `RadixCache` by
default, with `ChunkCache` (radix disabled), `RadixCacheCpp`, `HiRadixCache`, `SWARadixCache`,
`MambaRadixCache`, `UnifiedRadixCache`, `LMCRadixCache` variants (`scheduler.py:795-888`).

## 3. Core scheduling & orchestration

### 3.1 Scheduler state

Initialized in `init_running_status` (`scheduler.py:918-933`):

```python
self.waiting_queue: List[Req] = []
# The running decoding batch for continuous batching
self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
self.cur_batch: Optional[ScheduleBatch] = None   # batch being forwarded this iter
self.last_batch: Optional[ScheduleBatch] = None  # batch of previous iter
```

Plus `self.chunked_req = None` for the single in-flight chunked-prefill request
(`scheduler.py:953`) and `self.policy = SchedulePolicy(...)` (`scheduler.py:975-981`).

`dispatch_event_loop` selects the loop variant: pdmux / pipeline-parallel / **overlap** (default)
/ normal, with disagg prefill/decode variants (`scheduler.py:3678-3704`). Overlap is the default:
`self.enable_overlap = not server_args.disable_overlap_schedule` (`scheduler.py:376`, default
`False` at `server_args.py:644`).

### 3.2 The normal event loop

`event_loop_normal` (`scheduler.py:1389-1415`, verified verbatim):

```python
while True:
    recv_reqs = self.recv_requests()              # drain ZMQ, broadcast to TP peers
    self.process_input_requests(recv_reqs)        # dispatch -> waiting_queue
    ...
    batch = self.get_next_batch_to_run()          # continuous batching decision
    self.cur_batch = batch
    if batch:
        result = self.run_batch(batch)            # GPU forward + sample
        self.process_batch_result(batch, result)  # finish checks, stream out
    else:
        self.on_idle()
    self.last_batch = batch
```

- `recv_requests` (`scheduler.py:1510-1553`): only `pp_rank==0 && attn_tp_rank==0 &&
  attn_cp_rank==0` actually reads ZMQ — it drains both the tokenizer socket and the RPC socket
  with `recv_pyobj(zmq.NOBLOCK)` until `ZMQError`, capped by `max_recv_per_poll`
  (`scheduler.py:1526-1542`); the result is then broadcast so all TP ranks process identical
  inputs (PP uses `point_to_point_pyobj` at `scheduler.py:1546-1549`).
- `process_input_requests` (`scheduler.py:1696-1719`) routes each object through a
  `TypeBasedDispatcher`; generate inputs land in `handle_generate_request`
  (`scheduler.py:1833-2029`), which constructs a `Req` from the tokenized input
  (`scheduler.py:1854-1890`), validates length / logprob args / sessions, and ends with
  `self._add_request_to_queue(req)` — a FIFO `waiting_queue.append` (after priority validation
  and queued-limit abort) in non-disaggregated mode (`scheduler.py:2064-2072`).

### 3.3 The overlap event loop (default)

`event_loop_overlap` (`scheduler.py:1418-1470`) pipelines CPU scheduling of step *N+1* with GPU
execution of step *N* using a `result_queue: Deque[(ScheduleBatch, GenerationBatchResult)]`:

```python
batch = self.get_next_batch_to_run()
...
if batch:
    batch_result = self.run_batch(batch)              # launches GPU work, returns futures
    self.result_queue.append((batch.copy(), batch_result))
...
if self.last_batch:
    if not disable_overlap_for_batch:
        pop_and_process()                             # process_batch_result of LAST batch
...
if self.is_generation:
    self.launch_batch_sample_if_needed(batch_result)  # grammar-delayed sampling
self.last_batch = batch
```

Result processing always lags one iteration behind the forward launch. Overlap is selectively
disabled — two consecutive prefills (TTFT optimization, "might slightly hurt the throughput") or
spec-v2+grammar decode — via the extracted predicate `is_disable_overlap_for_batch`
(`scheduler.py:1472-1503`), in which case the queued result is popped *before* launching the new
batch (`scheduler.py:1443-1444`).

**Future tokens.** In overlap mode the next batch must be built before the previous batch's
sampled tokens exist on CPU. `run_batch` allocates *future indices* and stores
`batch.output_ids = -future_indices.indices` (negative ids as placeholders,
`scheduler.py:2841,2878`). `FutureMap` (`srt/managers/overlap_utils.py:45-166`, verified) is a
circular GPU buffer of size `max_running_requests * (3 + max_num_chunks)` (layout: running decode
batch → prefill chunk 1 → ... → prefill chunk N); `store_to_map` writes sampled token ids at the
future slots (`overlap_utils.py:161-166`) and `resolve_future(model_worker_batch)` patches the
negative placeholders in the next batch's `input_ids` from the buffer
(`overlap_utils.py:130-132`).

**Dual CUDA streams.** A `schedule_stream` is created in `run_event_loop`
(`scheduler.py:1383-1386`); `run_batch` does
`self.forward_stream.wait_stream(self.schedule_stream)` then resolves futures and runs the model
inside the forward-stream context (`scheduler.py:2824-2836`). A CUDA event
`batch_result.copy_done` (`scheduler.py:2833`) is synchronized at the start of result processing
(`scheduler_output_processor_mixin.py:136-137,397-398`) so the CPU only blocks when it actually
consumes the tokens.

### 3.4 `get_next_batch_to_run` — the continuous-batching core

`scheduler.py:2308-2417` (~110 lines):

1. Abort timed-out waiting/running requests (`2309-2310`).
2. **Stash the chunked request** so unfinished chunked prefill is not merged into the decode
   batch: `chunked_req_to_exclude.add(self.chunked_req); self.stash_chunked_request(...)`
   (`2322-2326`). Stashing temporarily releases its tree-cache lock.
3. **Merge the last prefill batch into `running_batch`** — the prefill→decode transition of
   continuous batching (`2341-2368`, verified):

```python
if ... self.last_batch and self.last_batch.forward_mode.is_extend():
    ...
    self.last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
    ...
    if not self.last_batch.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch
        else:
            self.running_batch.merge_batch(self.last_batch)
```

4. **Prefill-first policy**: try `get_new_batch_prefill()` (`2383`); if it returns a batch, run it
   (`2394-2396`); otherwise run decode:
   `self.running_batch = self.update_running_batch(self.running_batch)` (`2399-2404`).
5. DP-attention MLP-sync / idle-batch padding via `maybe_prepare_mlp_sync_batch`
   (`2385-2392`, `2409`).

### 3.5 Building a prefill batch — policy + admission

`get_new_batch_prefill` (`scheduler.py:2425-2441`, 17 lines) is a thin wrapper that handles
`PrefillDelayerSinglePassExecutor` setup/finalize around the ~240-line monolith
`_get_new_batch_prefill_raw` (`scheduler.py:2443-2681`) — cross-cutting concerns become a
wrapper, core policy stays a single linear function with a `_raw` suffix:

- Requests whose grammar (structured output) finished compiling are re-queued (`2446-2450`).
- Fast exits: `if (self.running_batch.batch_is_full or len(self.waiting_queue) == 0) and
  self.chunked_req is None: return None` (`2459-2462`); request-slot exhaustion sets
  `batch_is_full` (`2471-2477`).
- **Sort the waiting queue**: `self.policy.calc_priority(self.waiting_queue, self.running_batch)`
  (`2480`).
- **Admission** through a `PrefillAdder` (`2497-2512`), continuing the chunked request first
  (`2514-2516`), then iterating the sorted queue calling `adder.add_one_req(req, ...)`
  (`2522-2598`), with LoRA-set validation and priority-preemption hooks. The loop breaks on
  `NO_TOKEN` (sets `batch_is_full=True`, `2575-2583`).
- Admitted requests leave the waiting queue (`2605-2606`); preempted running requests are
  re-queued (`2607-2609`); a partially-admitted request becomes the new `self.chunked_req`
  (`2611-2614`).
- Materialization: `ScheduleBatch.init_new(can_run_list, self.req_to_token_pool,
  self.token_to_kv_pool_allocator, self.tree_cache, ...)` then `new_batch.prepare_for_extend()`
  (`2627-2644`).
- **Mixed chunk mode** (`--enable-mixed-chunk`): the running decode batch is folded into the
  prefill batch (`new_batch.mix_with_running(self.running_batch)`, `2660-2676`), so chunked
  prefill and decode share one forward.

**SchedulePolicy** (`srt/managers/schedule_policy.py`): cache-aware policies `LPM`
(longest-prefix-match, default) and `DFS_WEIGHT`; cache-agnostic `FCFS`, `LOF`, `RANDOM`,
`ROUTING_KEY` (`schedule_policy.py:80-93`). `calc_priority` (`117-159`) runs
`tree_cache.match_prefix` for each waiting request, stores `r.prefix_indices` / `r.last_node`
(`195-214`), then sorts by `-len(r.prefix_indices)` for LPM (`250-256`). Notable mechanisms:

- **LPM degradation**: `if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
  return CacheAgnosticPolicy.FCFS` — prefix matching is too expensive for long queues
  (`161-165`).
- **In-batch prefix-caching dedup**: requests with a small existing-cache match are matched
  against a *simulated* radix tree of the waiting queue itself; if ≥32 tokens are shared with an
  earlier queued request, the request is deprioritized (`float("inf")` sort key) so only one of
  the duplicates prefills and the rest hit its cache later (`185-243`, thresholds at `61-74`).
- Priority scheduling (`--enable-priority-scheduling`) sorts by
  `(priority * sign, wait_queue_entry_time)` (`302-312`).

**PrefillAdder — token-budget admission control** (`schedule_policy.py:375-899`). The central
quantity (`459-476`, verified; default branch shown):

```python
@property
def rem_total_tokens(self):
    ...
    available_and_evictable = (
        self.token_to_kv_pool_allocator.available_size()
        + self.tree_cache.evictable_size()
    )
    return available_and_evictable - self.rem_total_token_offset
```

The offset pre-reserves, for each running request,
`min(max_new_tokens - len(output_ids), CLIP_MAX_NEW_TOKENS) * new_token_ratio` (`418-425`,
`450-457`) — decode headroom is reserved using a *decaying estimate* `new_token_ratio` of how
many of its `max_new_tokens` a request will actually use (clipped at 4096 via
`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`, `53-59`). `add_one_req` (`767-899`) checks
`total_tokens = extend_input_len + max_new + page_size >= rem_total_tokens → NO_TOKEN`
(`798-810`), locks the matched radix node (`_lock_node`, `820`), and either admits whole
(`858-870`) or **truncates into a chunked prefill** when the chunk budget `rem_chunk_tokens`
(from `--chunked-prefill-size`) is exceeded (`871-897`):

```python
trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
...
req.set_extend_input_len(trunc_len)
req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
self.can_run_list.append(req)
self.new_chunked_req = req
```

`budget_state()` returns `NO_TOKEN`/`OTHER`/`CONTINUE` after each add (`526-543`).

**Priority preemption** (`preempt_to_schedule`, `schedule_policy.py:901-969`): when
`--enable-priority-scheduling` with preemption is on and the batch is full, running requests with
priority worse than the candidate by more than `priority_scheduling_preemption_threshold` are
released (`release_req`) until enough tokens are freed; preempted requests return to the waiting
queue (`scheduler.py:2548-2553,2607-2609`).

### 3.6 Decode step & retraction (memory-pressure preemption)

`update_running_batch` (`scheduler.py:2682-2769`): filters finished requests, then

```python
if (kv_full_retract_flag := not batch.check_decode_mem()) or (TEST_RETRACT and ...):
    ...
    retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(self.server_args)
```

`check_decode_mem` first evicts from the radix tree, then checks the allocator:
`evict_from_tree_cache(self.tree_cache, num_tokens);
return self.token_to_kv_pool_allocator.available_size() >= num_tokens`
(`schedule_batch.py:2121-2124`). If still short, `retract_decode`
(`schedule_batch.py:2134-2203`) kicks requests out **newest-output-first** (sort key
`(len(output_ids), -len(origin_input_ids))` reversed, `2145-2152`), frees their KV *without*
inserting into the tree ("we need the space instantly", `2167-2168`), keeps at least one request
(`2159-2161`) and aborts even that one rather than crash if it cannot fit (`2170-2187`). It
returns a raised `new_token_ratio` estimate; on healthy steps the ratio decays back:
`self.new_token_ratio = max(self.new_token_ratio - self.new_token_ratio_decay,
self.min_new_token_ratio)` (`scheduler.py:2756-2759`). Retracted requests are re-queued
(`scheduler.py:2753-2754`) and will re-prefill later (their prefix may still be radix-cached).
Finally `batch.prepare_for_decode()` allocates one KV slot per request and advances `seq_lens`
(`scheduler.py:2768`, `schedule_batch.py:2249-2320`).

### 3.7 `run_batch` and result processing

`run_batch` (`scheduler.py:2780-2936`): converts to a `ModelWorkerBatch`
(`batch.get_model_worker_batch()`, `2806`), then either the overlap path (forward stream +
future map, `2812-2857`) or direct `self.model_worker.forward_batch_generation(...)`
(`2867-2870`). Sampling can be deferred (`delay_sample_func`) when grammars need last-batch
results — executed later by `launch_batch_sample_if_needed` (`2938-2961`; the closure is dropped
afterwards to avoid a VRAM leak, `2953-2961`).

`process_batch_result` dispatches on forward mode (`scheduler.py:2963-2980`) into
`SchedulerOutputProcessorMixin`:

- **Prefill** (`srt/managers/scheduler_output_processor_mixin.py:128-287`): per request, append
  the first sampled token, `req.check_finished()`; if finished →
  `release_kv_cache(req, self.tree_cache)` (inserts into radix cache + frees duplicates), else →
  `self.tree_cache.cache_unfinished_req(req)` (`196-204`). A still-chunked request just
  decrements `req.is_chunked` and is *not* streamed (`258-264`).
- **Decode** (`392-541`): appends `next_token_ids[i]`, `req.check_finished(...)` (stop
  conditions: max_new_tokens, EOS, stop strings, grammar termination —
  `schedule_batch.py:1194-1223`), `_handle_finished_req` → `release_kv_cache` (`551-575`),
  logprob/hidden-state bookkeeping, grammar `accept_token`. Ends with
  `self.stream_output(batch.reqs, ...)` (`541`).
- `stream_output_generation` packs everything into one `BatchTokenIDOutput` and pushes it to the
  detokenizer (`912-922`, `1172-1217`).

### 3.8 Memory & cache management as it meets scheduling

**Two-level pools.**

- `ReqToTokenPool` (`srt/mem_cache/memory_pool.py:127-185`): a
  `(max_running_requests, max_context_len)` int32 tensor `req_to_token` mapping
  (request slot, position) → KV-pool index, plus a Python free list of slots. A chunked request
  keeps its slot across chunks (`alloc` reuses `req_pool_idx`, `156-180`).
- Token-to-KV allocator over the KV cache tensors: `TokenToKVPoolAllocator` for `page_size=1`
  keeps a flat `free_pages` index tensor — `alloc` slices the head, `free` concatenates back
  (`srt/mem_cache/allocator.py:144-165`); `PagedTokenToKVPoolAllocator` implements
  `alloc_extend`/`alloc_decode` with Triton kernels that fill partial pages before claiming new
  pages (`allocator.py:174-232+`). Pools are created by the model runner
  (`model_executor/model_runner_kv_cache_mixin.py:272-278,649`).

**RadixCache (prefix cache)** (`srt/mem_cache/radix_cache.py`):

- `match_prefix` walks the tree for the longest cached prefix, splitting nodes at partial
  matches, keyed by `(token_ids, extra_key)` so LoRA ids/salts get disjoint namespaces
  (`398-466`).
- `cache_finished_req` re-inserts the request's KV indices into the tree and **frees the
  duplicated span** that was already cached:
  `self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])`,
  then `dec_lock_ref(req.last_node)` (`488-533`).
- `cache_unfinished_req` does the same for chunked prefill between chunks, then re-matches and
  repoints `req.prefix_indices`/`req.last_node`, moving the lock to the new deeper node
  (`535-599`).
- **Eviction** is a heap over evictable leaves ordered by a pluggable `EvictionStrategy` — LRU
  (`node.last_access_time`), LFU, FIFO, priority-aware, SLRU
  (`srt/mem_cache/evict_policy.py:16-60`); popping a leaf frees its KV indices and may make its
  parent a new evictable leaf (`radix_cache.py:608-635`).
- **Lock refs are the scheduling↔cache contract**: `inc_lock_ref` moves a path's bytes from
  `evictable_size_` to `protected_size_` (`637-650`); the `PrefillAdder` takes this lock at
  admission (`schedule_policy.py:596-598,820`) so a scheduled request's prefix can never be
  evicted underneath it.

**Allocation is eviction-aware, scheduling is eviction-aware.** Every allocation helper first
evicts exactly the shortfall: `evict_from_tree_cache` does
`if allocator.available_size() < num_tokens: tree_cache.evict(EvictParams(num_tokens=num_tokens))`
(`srt/mem_cache/common.py:229-252`), called by `alloc_token_slots` (`201-226`),
`alloc_paged_token_slots_extend` (over-estimates by one page per request, `255-294`) and
`alloc_paged_token_slots_decode` (`394-420`). `prepare_for_extend` → `alloc_for_extend`
allocates request slots + KV and writes `req_to_token` rows (`common.py:328-391`, called from
`schedule_batch.py:1718`); `prepare_for_decode` → `alloc_for_decode` allocates 1 token/req
(`common.py:423-462`, called from `schedule_batch.py:2301`). Conversely, the scheduler treats
evictable cache as schedulable budget (`rem_total_tokens` includes `tree_cache.evictable_size()`,
`schedule_policy.py:472-475`). Allocator failure after eviction raises
`RuntimeError("...Out of memory...")` (`common.py:215-224`) — admission control plus retraction
is supposed to make that unreachable. Request teardown:
`release_kv_cache(req, tree_cache, is_insert)` → `tree_cache.cache_finished_req` plus freeing
over-allocated tail pages (`common.py:465-510`).

## 4. Distributed orchestration (Ray or alternative)

**Default: no Ray — multiprocessing + ZMQ + torch.distributed.** The only `import ray` sites in
`srt/` are the optional `srt/ray/` backend and a guarded import in `parallel_state.py`. The
default topology:

1. **Within a node**: `mp.Process` per scheduler rank, ZMQ IPC for request/response, `mp.Pipe`
   for the one-shot init handshake (section 2).
2. **Across nodes**: no torchrun. The user starts `sglang.launch_server` on every node with
   `--node-rank`; rank ranges per node come from
   `_calculate_rank_ranges(nnodes, pp_size, tp_size, node_rank)` (`engine.py:555-562`).
   Non-zero-rank nodes launch only their scheduler processes, then block — they run a
   `launch_dummy_health_check_server` and wait for completion (`engine.py:712-731`).
3. **Collectives**: each scheduler's `ModelRunner.init_torch_distributed` calls
   `torch.get_device_module().set_device(gpu_id)`, derives `dist_init_method` from
   `--dist-init-addr` (TCP) or an env override (`SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE`,
   documented for external orchestrators), then (`srt/model_executor/model_runner.py:1015-1071`):

```python
init_distributed_environment(
    backend=backend,
    world_size=self.tp_size * self.pp_size,
    rank=self.tp_size * self.pp_rank + self.tp_rank, ...)
initialize_model_parallel(
    tensor_model_parallel_size=self.tp_size,
    attention_data_parallel_size=self.dp_size,
    pipeline_model_parallel_size=self.pp_size,
    expert_model_parallel_size=self.moe_ep_size, ...)
```

   `init_distributed_environment` does `torch.distributed.init_process_group(...)` for WORLD
   (`srt/distributed/parallel_state.py:1688-1695`), and group creation builds *both* a device
   group (NCCL) and a `backend="gloo"` CPU group per parallel group
   (`parallel_state.py:288-311`). `initialize_model_parallel` creates TP/PP/EP/attention-CP/
   MoE-DP groups, Megatron-style (`parallel_state.py:1721-1769`).
4. **Control-plane fan-out across TP ranks**: only attn-TP-rank-0 polls ZMQ; requests are then
   broadcast as Python objects over the **gloo CPU groups** —
   `broadcast_pyobj(recv_reqs, self.tp_group.rank, self.tp_cpu_group, src=...)`, and PP stages
   forward via `point_to_point_pyobj(...)` on `world_group.cpu_group`
   (`scheduler.py:1522-1619`). DP-attention mode splits work vs control messages and broadcasts
   each over the appropriate sub-group to avoid an "expensive all-ranks gloo sync"
   (`scheduler.py:1561-1612`).

**Data parallelism (`dp_size > 1`)**: an extra `DataParallelController` process sits between
tokenizer and schedulers — it PULLs from `scheduler_input_ipc_name`, keeps one PUSH socket per
DP rank (`self.workers: List[zmq.Socket]`), and dispatches via `ROUND_ROBIN` or load-based
`DPBudget` policies; control messages are replicated to all workers
(`srt/managers/data_parallel_controller.py:121-238`). It launches each DP replica's TP group via
threads calling `launch_tensor_parallel_group`, spawning the same
`mp.Process(run_scheduler_process)` per `(pp_rank, tp_rank)` with per-dp-rank `PortArgs` but a
**shared `nccl_port`** ("Data parallelism reuses the tensor parallelism group",
`data_parallel_controller.py:240-266,421-480`). For multi-node DP-attention it brokers worker
ZMQ ports over a REQ/REP socket pair (`_broadcast_ports_as_server` / `_receive_ports_as_client`,
`data_parallel_controller.py:338-382`).

**Optional Ray backend (`--use-ray`)**: `launch_server.py` switches to
`sglang.srt.ray.http_server.launch_server` (`python/sglang/launch_server.py:36-40`).
`RayEngine(Engine)` overrides only `_launch_scheduler_processes`: it gets/creates a placement
group (`[{"CPU":1,"GPU":gpus_per_node}] * nnodes`, `STRICT_PACK` single-node / `SPREAD`
multi-node), pins the rank-0 bundle to the Engine's node via a `num_cpus=0` probe task that
compares node IPs, and launches one
`SchedulerActor.options(num_gpus=1, scheduling_strategy=PlacementGroupSchedulingStrategy(...)).remote(...)`
per `(pp_rank, tp_rank)` with `dist_init_addr = f"{rank0_node_ip}:{port_args.nccl_port}"`
(`srt/ray/engine.py:104-195`). Init handshake is `ray.get([actor.get_info.remote()])`, then
`run_event_loop.remote()` per actor (`ray/engine.py:197-215`). The actor docstring states the
division of labor (`srt/ray/scheduler_actor.py:30-37`, verified):

```python
"""Ray actor wrapper for SGLang Scheduler.

Each actor manages one GPU and runs the Scheduler + TpModelWorker stack.
Ray is used for process lifecycle; ZMQ handles request/response communication.
"""
```

Inside the actor it reads `ray.get_runtime_context().get_accelerator_ids()` for the real GPU id,
binds NUMA in-process, and constructs the same `Scheduler`
(`srt/ray/scheduler_actor.py:30-114`). The detokenizer still runs as a local `mp.Process` even
under Ray (`engine.py:741-748,770-771`).

**Weight sync (RLHF/online updates)**: the Scheduler handles
`UpdateWeightsFrom{Disk,Distributed,Tensor,IPC}ReqInput` control messages (via
`scheduler_update_weights_mixin.py` and `tp_worker.py:96-170`).
`ModelRunner.update_weights_from_distributed` receives new weights by
`torch.distributed.broadcast(weight, src=0, group=self._model_update_group[group_name],
async_op=True)` over a dedicated process group created by `init_weights_update_group` (an
external trainer is rank 0), with a `flattened_bucket` fast path using
`srt/weight_sync/tensor_bucket.py`'s `FlattenedTensorBucket`
(`srt/model_executor/model_runner.py:1521-1546,1670-1710`). The `Engine` exposes all of these
plus `release/resume_memory_occupation` as Python APIs routed through the tokenizer→scheduler
ZMQ path (`srt/entrypoints/engine.py:880-1083`).

## 5. Code organization style: function granularity

SGLang's granularity philosophy is consistent: **short control-flow skeletons, deliberately
large policy functions, all predicates extracted.**

**1. Orchestration functions kept deliberately short.** `event_loop_normal` is ~26 lines; every
step is a verb-named call, reading like pseudocode (`scheduler.py:1389-1415`, quoted in §3.2).
The overlap loop (`scheduler.py:1418-1470`) is ~50 lines; its complexity is pushed into one
inline closure `pop_and_process()` (`1424-1427`) and one extracted predicate
`is_disable_overlap_for_batch` (`1472-1503`) — the predicate carries the WHY comments
(consecutive-prefill TTFT trade-off; grammar+spec unsupported), moving "why" out of the loop
body.

**2. Policy functions kept deliberately huge.** `_get_new_batch_prefill_raw`
(`scheduler.py:2443-2681`, ~240 lines) is one linear admission pipeline — grammar queue check,
hicache events, priority preemption, chunk-size prediction, `PrefillAdder` construction,
per-request LoRA/capacity admission — every step reading/writing the same scheduler state
(`waiting_queue`, `running_batch`, `chunked_req`, `batch_is_full`); splitting it would only
create 10-parameter pseudo-functions. Same pattern: `get_next_batch_to_run`
(`scheduler.py:2308-2417`, ~110 lines) and `handle_generate_request`
(`scheduler.py:1833-2030`, ~198 lines). A refined splitting move: `get_new_batch_prefill`
(`2425-2441`, 17 lines) is only a thin wrapper handling the `PrefillDelayerSinglePassExecutor`
cross-cutting concern; the core stays in the `_raw`-suffixed monolith.

**3. Extracted small functions are almost all 3-8 line boolean predicates or single
computations**: `recv_limit_reached` (`scheduler.py:1505-1508`), `get_num_allocatable_reqs`
(`scheduler.py:2419-2423`), module-level `is_health_check_generate_req` / `is_work_request`
(`scheduler.py:3638-3653`), and ~8 `maybe_*` conditional-execution functions
(`maybe_init_draft_worker`, `_maybe_prepare_ngram_embedding`,
`maybe_send_health_check_signal`, ...).

**4. Giant classes split by Mixin files, not by objects.** `Scheduler` (3823 lines) is assembled
from 11 mixins in sibling files (`scheduler.py:317-330`):

```python
class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    ...
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
    SchedulerDllmMixin,
):
```

Files like `scheduler_output_processor_mixin.py` (1282 lines), `scheduler_pp_mixin.py` (1490),
`scheduler_profiler_mixin.py` (410). Effect: each file is readable, all mixins share the same
hot-path state on `self` with no indirection layer — but state ownership is purely by
convention.

**5. `__init__` as a table of contents.** `Scheduler.__init__` (`scheduler.py:332-475`) is flat
server_args assignment followed by a sequence of named, group-commented `init_*` phase calls
(`scheduler.py:402-473`): `init_model_config()` → `init_ipc_channels()` → `init_tokenizer()` →
`init_model_worker()` → `init_cache_with_memory_pool()` → `init_schedule_policy()` →
`init_disaggregation()` → `init_overlap()` → `init_request_dispatcher()`. The constructor reads
as startup documentation.

**6. Structured data conventions.** The three batch layers are `@dataclass` with ownership
declared in duplicate module docstrings (§2). `ScheduleBatch` fields use group comments + tensor
shape comments (`schedule_batch.py:1351-1459`): under `# Batched arguments to model runner`,
each line like `input_ids: torch.Tensor = None  # shape: [b], int64`. `io_struct.py` (2100
lines) is purely a dataclass message catalog with `BaseReq(ABC)`/`BaseBatchReq(ABC)` bases
(`io_struct.py:50-83`). Enums carry behavior: `ForwardMode(IntEnum)` has per-member comments and
`is_prefill()/is_extend()/is_decode()` query methods (`forward_batch_info.py:81-120`) — callers
write `batch.forward_mode.is_extend()` instead of comparing raw values. Policies are dual enums
`CacheAwarePolicy` / `CacheAgnosticPolicy` (`schedule_policy.py:80-94`); admission result is
`AddReqResult(Enum)` (`:369`).

**7. ABC over Protocol.** Only ~7 `Protocol` uses in all of `srt/`; interface boundaries use
ABC: `BaseTpWorker(ABC)` (`tp_worker.py:62`), `KVCache(abc.ABC)` (`memory_pool.py:668`), mixed
`BasePrefixCache(ABC, PrefixCacheTrait)` where the trait is a Protocol
(`mem_cache/base_prefix_cache.py:28,150`).

**8. Comment style: WHY + attributed TODO + paragraph comments on tuning constants.** Docstrings
are mostly one line ("A tensor parallel worker.", "A normal scheduler loop."); inline comments
explain trade-offs, e.g. `scheduler.py:1473-1476`: "we disable overlap to improve the TTFT of
the first batch. This might slightly hurt the throughput". TODOs carry owners:
`TODO(lsyin): support overlap + spec + grammar` (`scheduler.py:1494`),
`TODO(lmzheng): ModelWorkerBatch seems a bit redundant` (`schedule_batch.py:35`). Module-level
tuning constants each get 3-5 line comments explaining semantics and disable values
(`schedule_policy.py:54-75`: `CLIP_MAX_NEW_TOKENS`, `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`).

**9. Control-message dispatch via a type table, not if/elif.** 40+ control message types are
registered in one `(type, handler)` table (`scheduler.py:1283-1345`); the `TypeBasedDispatcher`
itself is ~20 lines with MRO caching (`python/sglang/utils.py:621-641`). Adding a control
command = one dataclass pair in `io_struct.py` + one table row.

## 6. Naming conventions

| Pattern | Examples | Path |
|---|---|---|
| `*Manager` = service with its own lifecycle/process | `TokenizerManager` ("is a process that tokenizes the text"), `DetokenizerManager`, `GrammarManager`, `lora_manager` | `managers/tokenizer_manager.py:14`; `managers/detokenizer_manager.py:14`; `scheduler.py:473` |
| `*Worker` = GPU/model-holding executor | `BaseTpWorker(ABC)`, `TpModelWorker` | `managers/tp_worker.py:62,217` |
| `*Runner` = the layer that actually runs forward | `ModelRunner` | `model_executor/model_runner.py:295` |
| `*Pool` / `*Allocator` / `*Cache` = memory layer | `ReqToTokenPool`, `MHATokenToKVPool(KVCache)`, `BaseTokenToKVPoolAllocator`, `RadixCache(BasePrefixCache)` | `mem_cache/memory_pool.py:127,668,764`; `mem_cache/radix_cache.py:321` |
| `run_*_process` = module-level subprocess entry | `run_scheduler_process`, `run_detokenizer_process` | `entrypoints/engine.py:61,84` |
| `*ReqInput` / `*ReqOutput` paired IPC messages; `Batch*` prefix = batched | `FlushCacheReqInput`/`FlushCacheReqOutput`, `TokenizedGenerateReqInput` | `managers/io_struct.py:702,793,1232-1237` |
| `handle_*` = dispatcher target | `handle_generate_request`, `handle_rpc_request` | `scheduler.py:1286,1323` |
| `*_wrapped` = adapts an internal method to the dispatcher signature | `flush_cache_wrapped` (wraps `flush_cache`) | `scheduler.py:3053,3229` |
| `maybe_*` = conditional execution | `maybe_init_draft_worker`, `maybe_sleep_on_idle`, `maybe_evict_swa` | `scheduler.py:638,3569`; `schedule_batch.py:2613` |
| `init_*` = construction phases | see §5.5 | `scheduler.py:403-464` |
| `_*_helper` = recursive kernel | `_match_prefix_helper`, `_insert_helper`, `_total_size_helper` | `radix_cache.py:693,749,846` |
| `_raw` suffix = core stripped of cross-cutting wrapper | `_get_new_batch_prefill_raw` | `scheduler.py:2443` |

A fixed abbreviation vocabulary runs through the codebase: `req`/`rid`/`bs`/`recv`/`mm`/`kv`/
`tp`/`pp`/`dp`/`ep`/`cp`/`spec`. One deliberate anti-PEP8 exception: finish-reason classes are
ALL_CAPS so they read like enum constants in logs/comparisons —
`class FINISH_MATCHED_TOKEN(BaseFinishReason)`, `FINISH_LENGTH`, `FINISH_ABORT`
(`schedule_batch.py:128-198`).

## 7. End-to-end flow trace

One `/generate` request, with file:line at each hop:

1. **HTTP entry** — `POST /generate` → `generate_request` →
   `tokenizer_manager.generate_request(obj, request)` (streaming SSE or single await) —
   `srt/entrypoints/http_server.py:701-735`.
2. **TokenizerManager** — `generate_request`: normalize args, register `rid_to_state`, tokenize
   → `_tokenize_one_request` builds `TokenizedGenerateReqInput` —
   `srt/managers/tokenizer_manager.py:507-552,954-1053` (`TokenizedGenerateReqInput(...)` at
   `:989`).
3. **ZMQ hop 1** — `_send_one_request`: `self.send_to_scheduler.send_pyobj(tokenized_obj)` over
   `scheduler_input_ipc_name` — `tokenizer_manager.py:1165-1172`; caller parks in
   `_wait_one_response` on `state.event` (`1267-1309`).
4. **Scheduler recv** — event loop iteration: `recv_requests` drains the PULL socket
   (`scheduler.py:1392-1395,1530`); broadcast to TP peers over the gloo cpu_group.
5. **Queueing** — dispatcher → `handle_generate_request` constructs `Req`
   (`scheduler.py:1833-1890`) → `_add_request_to_queue` → `self.waiting_queue.append(req)`
   (`scheduler.py:2064-2072`).
6. **Schedule** — next loop iteration: `get_next_batch_to_run` (`scheduler.py:2308`) →
   `get_new_batch_prefill` (`2425`): `policy.calc_priority` sorts the queue with radix
   `match_prefix` (`2480`; `schedule_policy.py:195-214`), `adder.add_one_req(req, ...)` admits
   within token budget and locks the prefix node (`2566-2570`; `schedule_policy.py:767-899`).
7. **Batch materialization** — `ScheduleBatch.init_new(...)` + `new_batch.prepare_for_extend()`
   (`scheduler.py:2627-2644`): builds input tensors and calls `alloc_for_extend` →
   `alloc_req_slots` + `alloc_extend` + `write_cache_indices`
   (`schedule_batch.py:1657-1743`; `mem_cache/common.py:328-391`).
8. **Forward** — `run_batch` (`scheduler.py:2780`): `batch.get_model_worker_batch()` (`2806`) →
   (overlap: forward stream + `future_map.resolve_future`, `2824-2836`) →
   `TpModelWorker.forward_batch_generation` builds `ForwardBatch.init_new` and runs
   `self.model_runner.forward(forward_batch, ...)` then
   `self.model_runner.sample(logits_output, forward_batch)` —
   `srt/managers/tp_worker.py:443-522`; `ModelRunner.forward` → `_forward_raw` at
   `model_executor/model_runner.py:2904-2972`, `sample` at `:3078`.
9. **Result processing** — (overlap: one iteration later, after `copy_done.synchronize()`)
   `process_batch_result` (`scheduler.py:2963`) → `process_batch_result_prefill`:
   `req.output_ids.append(next_token_id)`, `req.check_finished()`,
   `cache_unfinished_req`/`release_kv_cache` —
   `scheduler_output_processor_mixin.py:128-204`; subsequent decode steps loop through
   §3.4/§3.6 (`update_running_batch` → `prepare_for_decode` → `run_batch` →
   `process_batch_result_decode`, mixin `:392-541`) until finished; on finish
   `release_kv_cache` inserts the sequence into the radix cache and frees duplicates
   (mixin `:551-575`; `radix_cache.py:488-533`).
10. **ZMQ hop 2** — `stream_output_generation` sends `BatchTokenIDOutput` to the detokenizer
    (`scheduler_output_processor_mixin.py:1172-1217`).
11. **Detokenizer** — `event_loop`: `recv_pyobj` → `handle_batch_token_id_out` does incremental
    `batch_decode` + stop-string trimming and returns a `BatchStrOutput` →
    `send_to_tokenizer.send_pyobj(output)` —
    `srt/managers/detokenizer_manager.py:137-145,322-367`.
12. **ZMQ hop 3 → response** — TokenizerManager `handle_loop` receives it
    (`tokenizer_manager.py:1623-1636`), `_handle_batch_output` fills `meta_info`, appends text
    to `ReqState`, sets `state.finished`, fires `state.event` (`1638-1727+`); the parked
    `_wait_one_response` wakes and yields the dict (`1276-1334`), and the FastAPI handler
    streams `data: {...}` / returns JSON (`http_server.py:712-732`).

## 8. Ideas worth borrowing for wm-infra

1. **Declare the three batch layers in module docstrings.** vrl has `ScheduleBatch`/phase
   execution state, but no authoritative declaration of "which layer owns what, CPU or GPU" like
   `schedule_batch.py:20-36`. Write the same flow diagram at the top of the
   `vrl/engine/model_executor/` execution-state module: scheduler-layer batch (CPU metadata) →
   worker batch (forward subset) → tensor batch (GPU), and annotate each dataclass field with
   shape/dtype (`schedule_batch.py:1378-1390` style — especially useful for denoise
   latent/timestep tensors).

2. **Keep EngineLoop at 26-line discipline.** SGLang compresses the `while True` loop to five
   verbs — recv → process_input → get_next_batch → run_batch → process_result
   (`scheduler.py:1390-1415`) — and sinks all policy into 100-240-line monoliths like
   `get_next_batch_to_run`. vrl's EngineLoop→Scheduler→IterationRunner is isomorphic; resist
   stuffing conditionals into the loop body, tolerate long batch-assembly functions, and do NOT
   split the admission pipeline into single-caller helpers (consistent with the existing
   no-single-caller-helpers consensus).

3. **The overlap loop's `result_queue` pattern maps directly onto the denoise-step pipeline.**
   `event_loop_overlap` (`scheduler.py:1418-1470`): launch batch N without waiting,
   `result_queue.append((batch.copy(), result))`, then `pop_and_process` batch N-1's CPU
   post-processing in the next iteration — plus a `is_disable_overlap_for_batch` predicate for
   the corner cases that cannot overlap. CPU scheduling between diffusion denoise steps (batch
   regrouping, sampler-state advance) overlapping with GPU forward can copy this deque +
   predicate shape exactly. The `FutureMap` negative-placeholder trick
   (`overlap_utils.py:45-166`) is the companion mechanism when step N+1's inputs depend on step
   N's GPU outputs.

4. **Unify control-plane messages as `*ReqInput`/`*ReqOutput` dataclasses + a type-dispatch
   table.** vrl's weight_sync / GRPO control commands (update weights, pause generation, flush
   cache) can mirror `io_struct.py`'s paired naming and `TypeBasedDispatcher`
   (`python/sglang/utils.py:621`): each RPC's request/response is an explicit schema, handler
   registration is one greppable table (`scheduler.py:1283-1345`). Far cheaper to extend than
   scattered `if msg.type == ...` when adding GRPO control commands.

5. **Phased `init_*` construction + `maybe_*`/`is_*`/`*_wrapped`/`_raw` naming discipline.**
   `Scheduler.__init__` is startup documentation (`scheduler.py:402-473`). vrl engine startup
   (IPC, model load, cache pools, policy) deserves the same table-of-contents style. Adopt:
   conditional execution → `maybe_*` prefix; dispatcher adapter layer → `*_wrapped` suffix with
   the inner method still directly callable (`flush_cache_wrapped`/`flush_cache`,
   `scheduler.py:3053,3229`); core policy under a cross-cutting wrapper → `_raw` suffix
   (`scheduler.py:2425/2443`).

6. **Phase enums with behavior methods.** vrl's ENCODE_TEXT→DENOISE_STEP→DECODE_VAE phase enum
   can learn from `ForwardMode` (`forward_batch_info.py:81-120`): per-member comments +
   `is_denoise()/is_encode()` query methods, so the batch planner writes `phase.is_denoise()`
   instead of comparing enum values everywhere — `is_extend()` collapsing 6 modes into one
   semantic predicate is exactly the value.

7. **Scheduling-budget ideas for KV-analogue memory**: (a) treat evictable cache as schedulable
   budget (`rem_total_tokens` includes `tree_cache.evictable_size()`); (b) reserve decode
   headroom with a decaying `new_token_ratio` estimate instead of worst-case `max_new_tokens`;
   (c) on memory pressure, retract newest-output-first and re-queue (work loss is minimal and
   prefix may still be cached); (d) lock-ref the prefix path at admission so eviction can never
   pull cache out from under a scheduled request. All four generalize to latent-cache management
   for video world-model rollouts.

8. **Do NOT borrow: mixin multiple inheritance.** The 11-mixin `Scheduler`
   (`scheduler.py:317-330`) is damage control for an established 3800-line class; state
   ownership is purely conventional (mixins freely read/write `self.running_batch`). vrl already
   has cleaner Protocol+implementation separation (`vrl/engine/interfaces.py`) and should not
   regress. The transferable lesson: when an optional feature (PP, disaggregation, profiler)
   starts invading the main scheduler file, split it out immediately rather than letting it grow
   to a thousand lines.

## 9. Source-of-truth index

| Claim | Source |
|---|---|
| 3-component engine, main process = HTTP+TokenizerManager, ZMQ IPC | `srt/entrypoints/engine.py:146-158` ✓verified |
| Scheduler launch: one mp.Process per (pp,tp) rank, gpu_id formula, mp.Pipe handshake | `srt/entrypoints/engine.py:533-613` |
| Multi-node: node_rank≥1 runs only schedulers + dummy health server | `srt/entrypoints/engine.py:712-731` |
| Detokenizer mp.Process; SubprocessWatchdog; spawn start method | `srt/entrypoints/engine.py:741-748,769-778,1206` |
| run_*_process subprocess entries | `srt/entrypoints/engine.py:61,84,577-612,741` |
| Engine weight-update / memory-occupation Python APIs | `srt/entrypoints/engine.py:880-1083` |
| HTTP server reuses Engine._launch_subprocesses | `srt/entrypoints/http_server.py:2313-2350` |
| HTTP /generate endpoint | `srt/entrypoints/http_server.py:701-735` |
| PortArgs: per-edge ipc:// endpoints, nccl_port, TCP for DP-attention | `srt/server_args.py:7112-7172` |
| Overlap schedule default on | `srt/server_args.py:644`; `scheduler.py:376` |
| Scheduler ZMQ sockets only on attn-TP/CP rank 0 | `srt/managers/scheduler.py:498-538` |
| Scheduler 11-mixin class head | `srt/managers/scheduler.py:317-330` |
| `init_*` phased `__init__` | `srt/managers/scheduler.py:332-475` |
| Scheduler state (waiting_queue, running_batch, chunked_req, policy) | `srt/managers/scheduler.py:918-933,953,975-981` |
| Scheduler owns TpModelWorker in-process; draft worker | `srt/managers/scheduler.py:615-636,638-666` |
| Memory pools from worker; radix cache variant selection | `srt/managers/scheduler.py:753-888` |
| Type-dispatch table (40+ messages) | `srt/managers/scheduler.py:1283-1345`; `python/sglang/utils.py:621-641` |
| event_loop_normal (26-line skeleton) | `srt/managers/scheduler.py:1389-1415` ✓verified |
| event_loop_overlap: result_queue, one-iteration lag, pop_and_process | `srt/managers/scheduler.py:1418-1470` |
| Overlap-disable predicate + TTFT comment + TODO(lsyin) | `srt/managers/scheduler.py:1472-1503` |
| recv_requests: NOBLOCK drain on entry rank → broadcast_pyobj over gloo; PP point_to_point_pyobj; DP-attn split broadcast | `srt/managers/scheduler.py:1505-1553,1561-1619` |
| Request intake → Req → waiting_queue (FIFO append) | `srt/managers/scheduler.py:1696-1719,1833-2030,2064-2072` |
| get_next_batch_to_run: stash chunked, merge last prefill into running batch, prefill-first | `srt/managers/scheduler.py:2308-2417` ✓verified (merge block 2341-2368) |
| get_new_batch_prefill wrapper / `_raw` monolith | `srt/managers/scheduler.py:2425-2441,2443-2681` |
| Small predicate helpers | `srt/managers/scheduler.py:1505-1508,2419-2423,3638-3653` |
| Priority preemption requeue | `srt/managers/scheduler.py:2548-2553,2607-2609` |
| Decode OOM retraction + new_token_ratio adaptation | `srt/managers/scheduler.py:2682-2769`; `srt/managers/schedule_batch.py:2121-2221` |
| run_batch: forward stream, FutureMap future indices, delayed sampling | `srt/managers/scheduler.py:2780-2936,2938-2961` |
| process_batch_result dispatch | `srt/managers/scheduler.py:2963-2980` |
| `*_wrapped` dispatcher adapters | `srt/managers/scheduler.py:3053,3229` |
| maybe_* naming (8 sites in scheduler) | `srt/managers/scheduler.py:638,1215,1223,1805,1822,2986,3569` |
| Event-loop dispatch (pdmux/pp/overlap/normal/disagg) | `srt/managers/scheduler.py:3678-3704` |
| run_scheduler_process: ctor, pipe init info, SIGQUIT parent on crash | `srt/managers/scheduler.py:3764-3823` |
| Dual CUDA streams + copy_done event sync | `srt/managers/scheduler.py:1383-1386,2824-2836`; `srt/managers/scheduler_output_processor_mixin.py:136-137,397-398` |
| Result processing: check_finished, cache_unfinished/finished, stream_output | `srt/managers/scheduler_output_processor_mixin.py:128-287,392-541,551-575,912-922,1172-1217`; `schedule_batch.py:1194-1223` |
| Three-layer batch docstring (duplicated) | `srt/managers/schedule_batch.py:20-36`; `srt/model_executor/forward_batch_info.py:14-28` ✓verified |
| Req / ScheduleBatch / ModelWorkerBatch classes; grouped fields + shape comments | `srt/managers/schedule_batch.py:574,1352,1351-1459,2691` |
| FINISH_* ALL_CAPS finish-reason classes | `srt/managers/schedule_batch.py:128-198` |
| prepare_for_extend / prepare_for_decode | `srt/managers/schedule_batch.py:1657-1743,2249-2320` |
| FutureMap circular buffer, negative placeholders, resolve/store | `srt/managers/overlap_utils.py:45-166` ✓verified; `scheduler.py:2822-2841,2878` |
| Schedule policies, LPM>128 FCFS fallback, in-batch prefix dedup, priority sort | `srt/managers/schedule_policy.py:80-94,117-165,185-256,302-312` |
| Token-budget admission: rem_total_tokens, new_token_ratio reservation, CLIP constant | `srt/managers/schedule_policy.py:53-75,375-476,526-566,767-899` ✓verified (rem_total_tokens 459-476) |
| Priority preemption (preempt_to_schedule) | `srt/managers/schedule_policy.py:901-969` |
| Prefix lock at admission | `srt/managers/schedule_policy.py:596-598,820` |
| TokenizerManager ZMQ wiring + async handle_loop + response wake | `srt/managers/tokenizer_manager.py:342-360,507-552,954-1053,1165-1172,1267-1334,1598-1727` |
| Detokenizer PULL/PUSH + sync loop + incremental decode | `srt/managers/detokenizer_manager.py:94-99,137-145,322-367` |
| BaseTpWorker(ABC)/TpModelWorker; ModelRunner built by worker; forward+sample | `srt/managers/tp_worker.py:62,96-170,217-260,443-522` |
| io_struct ReqInput/ReqOutput pairs + BaseReq(ABC) | `srt/managers/io_struct.py:50-83,702,793,1232-1237` |
| DP controller: dispatch policies, per-dp PUSH sockets, shared nccl_port, port brokering | `srt/managers/data_parallel_controller.py:121-238,240-271,338-382,421-480` |
| init_torch_distributed: world = tp×pp, rank formula, init-method override | `srt/model_executor/model_runner.py:987-1106` (call block 1015-1071) |
| ModelRunner.forward/_forward_raw/sample | `srt/model_executor/model_runner.py:2904-2972,3078` |
| Weight sync: broadcast over _model_update_group, flattened buckets | `srt/model_executor/model_runner.py:1521-1546,1670-1710`; `srt/weight_sync/tensor_bucket.py` |
| KV pool creation in model runner | `srt/model_executor/model_runner_kv_cache_mixin.py:272-278,649` |
| ForwardMode enum with behavior methods | `srt/model_executor/forward_batch_info.py:81-120` |
| init_process_group + gloo cpu groups + model-parallel groups | `srt/distributed/parallel_state.py:288-311,1646-1714,1721-1769` |
| ReqToTokenPool structure & slot reuse for chunked reqs | `srt/mem_cache/memory_pool.py:127-185` |
| Memory pool / KVCache classes | `srt/mem_cache/memory_pool.py:127,668,764` |
| Token allocator (page_size=1 free list; paged alloc_extend) | `srt/mem_cache/allocator.py:117,144-232,356` |
| RadixCache match/insert/cache_finished/cache_unfinished/evict/lock_ref; `_*_helper` recursion | `srt/mem_cache/radix_cache.py:321,398-671,693,749,846` |
| Eviction strategies (LRU/LFU/FIFO/priority/SLRU) | `srt/mem_cache/evict_policy.py:16-60` |
| Eviction-aware allocation helpers; OOM RuntimeError; release_kv_cache | `srt/mem_cache/common.py:201-294,328-462,465-510` |
| ABC+Protocol hybrid (only ~7 Protocols in srt) | `srt/mem_cache/base_prefix_cache.py:28,150` |
| --use-ray switch | `python/sglang/launch_server.py:36-40` |
| RayEngine: placement group, bundle pinning, SchedulerActor per rank, ray.get handshake | `srt/ray/engine.py:79-234` (launch 104-195, handshake 197-215) |
| "Ray for lifecycle; ZMQ for communication"; actor GPU id from runtime ctx | `srt/ray/scheduler_actor.py:30-114,123-133` ✓verified (docstring 30-37) |
| Package metadata, console scripts | `python/pyproject.toml:6-8,181-183` |
| File sizes (scheduler 3823 / model_runner 3272 / recv_skipper 38 etc.) | `wc -l` over `srt/managers/*.py`, `srt/mem_cache/`, `srt/model_executor/` |

---

# Part II — Deep Dive

*Same checkout (`492883c8c`), same path convention (`srt/...` = `python/sglang/srt/...`). Part I
covered orchestration: event loops, batching policy, distributed topology. Part II goes below
that — worker tensor mechanics, CUDA graphs, the memory subsystem's exact structs and ownership
rules, extension surfaces, boot/failure machinery. Citations were re-verified by opening files;
spot-checked snippets are quoted verbatim.*

## 10. Model executor & worker internals

*Part I (§3.7, §7) traced `run_batch → forward_batch_generation → forward → sample` at the
orchestration level; this section explains what actually happens inside the worker: tensor
materialization, CUDA graphs, attention metadata, and the sampling/output return path.*

### 10.1 From `ScheduleBatch` to GPU tensors

**Where host→device transfer actually happens.** Earlier than the name `ForwardBatch` suggests:
the GPU tensors are created by the *scheduler* inside `ScheduleBatch.prepare_for_extend()` /
`prepare_for_decode()`, not by the model runner. `ForwardBatch.init_new` mostly re-wraps
already-on-GPU tensors.

`prepare_for_extend` flattens per-request token lists, builds **pinned CPU tensors**, and ships
them with async copies (`srt/managers/schedule_batch.py:1692-1702`):

```python
_pin = is_pin_memory_available(self.device)
input_ids_tensor = torch.tensor(
    list(chain.from_iterable(input_ids)), dtype=torch.int64, pin_memory=_pin
).to(self.device, non_blocking=True)
seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
    self.device, non_blocking=True
)
seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
```

Note the deliberate **CPU shadow copy** `seq_lens_cpu` — many downstream consumers (attention
metadata `max_seq_len_k`, CUDA-graph bucket selection) need sequence lengths on the host without
a sync. KV slots for the new tokens are then allocated (`alloc_for_extend(self)` at
`schedule_batch.py:1718-1720`), producing `out_cache_loc` (the KV-pool indices where this
forward's K/V will be written).

The packing layout is **ragged concatenation, no padding**: `input_ids` is shape
`[sum(extend_lens)]`, and per-token `positions` are computed on GPU by a Triton kernel that
writes `prefix_len + offset` per request segment plus the `extend_start_loc` cumsum
(`srt/model_executor/forward_batch_info.py:1134-1187`; the kernel comment admits the naive
per-program cumsum "can be slow for large bs").

`prepare_for_decode` is the cheap path — no new input tensor is built; last step's sampled ids
become this step's inputs, and lengths are bumped in place (or out-of-place under overlap, where
the old tensor may still be in flight on the forward stream)
(`schedule_batch.py:2294-2319`):

```python
self.input_ids = self.output_ids
self.output_ids = None
...
self.out_cache_loc = alloc_for_decode(self, token_per_req=1)
...
if self.enable_overlap:
    # Do not use in-place operations in the overlap mode
    self.seq_lens = self.seq_lens + 1
    self.seq_lens_cpu = self.seq_lens_cpu + 1
else:
    # A faster in-place version
    self.seq_lens.add_(1)
    self.seq_lens_cpu.add_(1)
```

Even `filter_batch` (removing finished requests) does its index_select with a pinned,
non-blocking index tensor (`schedule_batch.py:2398-2402`).

**`ModelWorkerBatch` → `ForwardBatch`.** `get_model_worker_batch` is a pure field-projection —
it copies ~40 fields (GPU tensor handles + CPU metadata) into the `ModelWorkerBatch` dataclass
without touching tensor data (`schedule_batch.py:2501-2582`). `TpModelWorker.forward_batch_generation`
then calls `ForwardBatch.init_new(model_worker_batch, self.model_runner)`
(`srt/managers/tp_worker.py:455-459`). `ForwardBatch.init_new`
(`srt/model_executor/forward_batch_info.py:443-611`) does the *remaining* host→device work and
attaches runner-owned state:

- passes through the big tensors (`input_ids=batch.input_ids`, `seq_lens`, `out_cache_loc`, ...
  at `:449-493`) and binds the pools/backend: `req_to_token_pool=model_runner.req_to_token_pool,
  token_to_kv_pool=model_runner.token_to_kv_pool, attn_backend=model_runner.attn_backend`
  (`:478-480`);
- uploads small extend-only tensors: `extend_seq_lens`/`extend_prefix_lens` via
  `torch.tensor(...).to(device, non_blocking=True)` and computes positions with the Triton
  kernel (`:561-580`);
- decode positions are just `clamp_position(batch.seq_lens)` = `seq_lens - 1` via a JIT CUDA
  kernel (`:557-559`, `:1207-1216`);
- precomputes SWA cache locations once per batch for hybrid sliding-window models (`:594-600`),
  translates DP-attention global token counts to GPU tensors (`:509-531`), and fetches LoRA
  adapters into the pool right before the forward (`:602-609`).

**Persistent/shared input buffers.** For non-graph forwards, input tensors are freshly allocated
per step (caching-allocator-recycled). Persistent buffers exist for CUDA graphs (§10.2) and are
deduplicated across runners by a module-level pool: `ForwardInputBuffers.share_buffers()` checks
`_forward_input_buffer_pool` by field name and reuses the larger existing buffer via
`as_strided`, asserting dtype/device match — so e.g. a draft worker's graph buffers alias the
target worker's (`srt/model_executor/input_buffers.py:11-65`, disabled on NPU "due to accuracy
issue" at `:37-39`).

### 10.2 CUDA graph capture and replay

**What is captured, for which shapes.** Only token-per-request-constant modes are graphable:
`ForwardMode.is_cuda_graph()` returns true for `DECODE`, `TARGET_VERIFY`, `IDLE`, `DLLM_EXTEND`
(`forward_batch_info.py:166-172` ✓verified). Each graph fixes `num_tokens = bs *
num_tokens_per_bs`, where `num_tokens_per_bs` is 1 for decode, `speculative_num_draft_tokens`
for verify, or the dLLM block size (`srt/model_executor/cuda_graph_runner.py:620-632`).

Capture batch sizes come from server args: by GPU memory tier, `cuda_graph_max_bs` defaults to
8 (T4) / 24-80 (4090) / 32-160 (A100-40G) / 256-512 (H100/H200) / 512 (B200)
(`srt/server_args.py:1298-1355`), expanded into the bucket list (✓verified at
`server_args.py:1485-1492`):

```python
capture_bs = ([1, 2, 4, 8, 12]
    + list(range(16, 257, 8))
    + list(range(272, 512, 16))
    + list(range(512, self.cuda_graph_max_bs + 1, 32)))
```

then filtered in `get_batch_sizes_to_capture`: every captured bs must satisfy
`bs * num_tokens_per_bs % mul_base == 0` where `mul_base` folds in two-batch-overlap (×2),
attention-TP gather, and attention-CP divisibility, and `bs <= req_to_token_pool.size` (padded
up so the true max-requests bs is still captured on tiny GPUs)
(`cuda_graph_runner.py:495-529`).

**Static input buffers and the capture loop.** `CudaGraphRunner.__init__` allocates one
`DecodeInputBuffers` set at `max_bs`/`max_num_token`: `input_ids`, `req_pool_indices`,
`seq_lens` (pre-filled with the backend's `seq_len_fill_value`), `out_cache_loc`, `positions`,
`mrope_positions`, plus PP proxy hidden/residual buffers and DP-attention global-token
counters — and a **true CPU tensor** `seq_lens_cpu` kept in lockstep
(`cuda_graph_runner.py:132-272`, CPU tensor at `:244-250`). Buffers are then deduplicated across
runners (`:706`).

`capture()` iterates **largest bs first** — "Capture the large shapes first so that the smaller
shapes can reuse the memory pool allocated for the large shapes" (✓verified comment at
`:888-890`; reversed iteration at `:848-852`) — inside `freeze_gc` (GC frozen during capture to
keep Python-object churn out of the graph window, `:417-432`) and the distributed
`graph_capture()` context. All graphs share one global memory pool
(`global_graph_memory_pool`, `:532-543`, `:1165-1168`). Per batch size,
`capture_one_batch_size`:

1. slices the static buffers to `[:num_tokens]`/`[:bs]` and builds a `ForwardBatch` directly on
   them (`:959-1089`);
2. asks the attention backend to set up *graph-resident* metadata:
   `attn_backend.init_forward_metadata_capture_cuda_graph(bs, num_tokens, req_pool_indices,
   seq_lens, ...)` (`:1104-1113`);
3. runs the model **twice eagerly** for warmup, with a TP barrier each time
   (`for _ in range(2): synchronize(); tp_group.barrier(); run_once()`, `:1160-1163`), then
   records the graph (`:1169-1171`).

`torch.compile` is layered orthogonally: for `bs <= torch_compile_max_bs` the captured callable
is `torch.compile(model.forward, mode="max-autotune-no-cudagraphs")` — inductor kernels inside
an SGLang-managed CUDA graph, not inductor's own cudagraphs (`patch_model`, `:446-476`).

**Replay path and skip conditions.** The decision is made per step in `_forward_raw`:
`can_run_graph = mode_check() and self.graph_runner and self.graph_runner.can_run(forward_batch)`;
if true it short-circuits straight to `self.graph_runner.replay(...)` before any of the
eager-path DP-padding logic (`srt/model_executor/model_runner.py:2980-3003`).

`can_run` (`cuda_graph_runner.py:742-811`) **skips replay** when: per-request embedding
overrides exist (`replace_embeds`, dynamic); the (global, for DP-attention) batch size exceeds
`max_bs` — or, with `--disable-cuda-graph-padding`, isn't an exactly-captured key; DP-attention
requires MLP sync but the batch can't run a DP graph; an encoder-decoder batch contains
`encoder_len == 0`; a higher `capture_hidden_mode` is requested than captured; TBO/ngram
constraints fail. The hidden-mode mismatch path is notable: `recapture_if_needed`
**re-captures the whole graph set at runtime** when a request suddenly needs hidden states
(`:1175-1205`).

`replay_prepare` handles dynamic shapes by **padding up to the next captured bucket**:

```python
index = bisect.bisect_left(self.capture_bs, raw_bs)
bs = self.capture_bs[index]
```

(`:1228-1231`), then `populate_from_forward_batch` copies the real batch into the static
buffers — slack slots are reset to `seq_len_fill_value`/zero only when `bs != raw_bs` — using a
single dtype-grouped `torch._foreach_copy_` for all GPU tensors and one separate CPU copy for
`seq_lens_cpu` (`:274-383`, foreach grouping at `:111-129`). The attention backend then rebuilds
its graph-resident metadata in place via
`init_forward_metadata_replay_cuda_graph(bs, ..., seq_lens_cpu=...)` (`:1270-1279`). Finally
`self.graphs[graph_key].replay()` and the output is **sliced back to the real size**:
`next_token_logits[: self.raw_num_token]` (`:1313-1344`). Graph keys encode bs + pdmux stream
index + LoRA/no-LoRA variant (each bs can be captured twice when dual LoRA graphs are enabled,
`:854-886`).

Prefill has a separate, optional **piecewise CUDA graph** path: `forward_extend` first checks
`self.piecewise_cuda_graph_runner.can_run(forward_batch)` and replays it instead of eager extend
(`model_runner.py:2836-2845`; runner at `srt/model_executor/piecewise_cuda_graph_runner.py`,
token buckets at `server_args.py:1530-1546`).

### 10.3 The attention backend abstraction

**Selection.** `AttentionBackend(ABC)` defines the contract: `init_forward_metadata` (eager,
per batch), `init_cuda_graph_state` (once, max-size buffers),
`init_forward_metadata_capture/replay_cuda_graph`, `get_cuda_graph_seq_len_fill_value`, and a
`forward` that dispatches to `forward_decode/forward_extend` per layer
(`srt/layers/attention/base_attn_backend.py:18-120`).

Backends register into a string-keyed table via decorator —
`@register_attention_backend("flashinfer") ... ATTENTION_BACKENDS[name] = fn` — with ~20 entries
(flashinfer, fa3/fa4, triton, trtllm_mha/mla, flashmla, cutlass_mla, nsa, aiter, ascend, ...)
(`srt/layers/attention/attention_registry.py:20-28` and onwards). A factory can itself branch:
`"flashinfer"` returns `FlashInferMLAAttnBackend` for MLA models
(`attention_registry.py:31-53`).

Defaults are hardware/architecture heuristics in `ServerArgs._get_default_attn_backend`: MHA →
`fa3` on Hopper+CUDA≥12.3 (with a comment citing a flashinfer-0.6.1 perf regression, issue
#17411), `trtllm_mha` on SM100, `aiter` on HIP, else `flashinfer`/`triton`; MLA → `fa3` on
Hopper, `flashinfer` on SM100, `triton` otherwise (`srt/server_args.py:2428-2469`). Some
backends force constraints back into server args: flashmla forces `page_size=64`, cutlass_mla
`page_size=128` (`server_args.py:2517-2534`).

`ModelRunner._get_attention_backend` adds composition: different prefill/decode backend strings
get wrapped in a `HybridAttnBackend`; pdmux creates one decode backend per SM group;
two-batch-overlap wraps in `TboAttnBackend` (`srt/model_executor/model_runner.py:2086-2151`).

**What metadata building does per batch (FlashAttention backend).** The point of
`init_forward_metadata` is to compute **once per forward pass** everything the per-layer kernel
calls share. `FlashAttentionMetadata` holds `cache_seqlens_int32`, `max_seq_len_q/k`,
`cu_seqlens_q/k`, the `page_table`, and precomputed FA3 `scheduler_metadata`
(`srt/layers/attention/flashattention_backend.py:37-84`). For normal decode (`:333-353`):

```python
metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
metadata.cu_seqlens_q = torch.arange(0, batch_size + 1, ...)
metadata.cu_seqlens_k = torch.nn.functional.pad(
    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
    forward_batch.req_pool_indices, : metadata.max_seq_len_k]
```

i.e. the **page table is a gather of `req_to_token` rows** truncated to the batch max length —
this is where the `ReqToTokenPool` (Part I §3.8) meets the kernels. The
`seq_lens_cpu.max().item()` is exactly why the scheduler maintains a CPU shadow of seq_lens: no
D2H sync on the hot path. `scheduler_metadata` pre-runs FA3's `prepare_varlen_num_blocks` once
"to avoid per-layer kernel calls" (`:346-353`).

For CUDA graphs the same quantities live in **pre-allocated max-size buffers** —
`init_cuda_graph_state` allocates `cache_seqlens[max_bs]`, `cu_seqlens_k[max_bs+1]`,
`page_table[max_bs, max_num_pages]` (`:1294-1332`) — and
`init_forward_metadata_replay_cuda_graph` refreshes them in place each replay (e.g. fused
`normal_decode_set_metadata` writes seqlens/cumsums/page-table rows into the captured buffers,
`:1857-1935`). Spec-decode topk>1 builds a second "expand" metadata set describing the
draft-token tail (`:303-332`).

### 10.4 Sampling and the output path back to the scheduler

**Pruning logits before sampling.** The model's `LogitsProcessor` never materializes prefill
logits for all tokens: for extend without input-logprobs it gathers only each sequence's last
hidden state — `last_index = torch.cumsum(extend_seq_lens, dim=0) - 1;
pruned_states = hidden_states[last_index]` (`srt/layers/logits_processor.py:427-453`) — so the
sampler always sees `[bs, vocab]`. (Prefill *with* input logprobs builds four index sets,
documented with a worked example at `:454-476`.)

**Sampler.** `ModelRunner.sample` first applies grammar masks/logit bias (`_preprocess_logits`
→ `sampling_info.update_regex_vocab_mask(); apply_logits_bias(...)`, then drops `vocab_mask`
immediately to avoid a documented overlap-mode VRAM leak, `model_runner.py:3061-3076`), then
calls the `Sampler` module with positions = `positions` for decode or `seq_lens - 1` for prefill
(`:3101-3113`).

`Sampler.forward` (`srt/layers/sampler.py:83-193`): all-greedy batches short-circuit to
`torch.argmax(logits, -1)` (`:111-117`); otherwise temperature-divide in place, in-place softmax
(`logits[:] = torch.softmax(...)`, `:161-165`), then either plain multinomial (no top-k/p/min-p)
or the flashinfer fused kernels `top_k_top_p_sampling_from_probs` /
`top_k_renorm_prob + min_p_sampling_from_probs` (`:213-232`), with a pytorch fallback backend.
Two RL-specific paths exist: deterministic Gumbel sampling seeded per `(seed, position)`
(`_sample_from_logprobs`, `:248-265`) and log_softmax-based logprobs "to match the trainer" when
`rl_on_policy_target` is set (`:129-139`). Requested logprobs/top-k logprobs are attached in
place onto `LogitsProcessorOutput` (`:178-189`).

**Return to the scheduler.** Part I §3.7 covered the FutureMap/delayed-sampling orchestration;
the exact mechanics: `forward_batch_generation` wraps everything in a
`GenerationBatchResult(logits_output, next_token_ids, can_run_cuda_graph, ...)`
(`srt/managers/tp_worker.py:467-522`; dataclass at `srt/managers/utils.py:25-54`). Three return
variants: normal (sample now), grammar+overlap (return a `delay_sample_func` closure instead,
`:485-498`), prefill-only (dummy zero token ids, optional `compute_logprobs_only`, `:500-520`).

In overlap mode the scheduler, still on the **forward stream**, immediately (a) writes the
sampled ids into the `FutureMap` GPU ring buffer —
`self.token_ids_buf[intv] = batch_result.next_token_ids` (`srt/managers/overlap_utils.py:161-166`)
— so the *next* batch can resolve its negative placeholder inputs without CPU involvement, and
(b) launches the async D2H: `batch_result.copy_to_cpu(...)` does
`self.next_token_ids.to("cpu", non_blocking=True)` for token ids plus any logprob/hidden
tensors, then `self.copy_done.record()` (`utils.py:56-100`; call site
`srt/managers/scheduler.py:2824-2841`). One iteration later, `process_batch_result_decode`
blocks only on that event — `result.copy_done.synchronize()` — and converts to Python:
`next_token_ids = next_token_ids.tolist()` before `check_finished`/streaming
(`srt/managers/scheduler_output_processor_mixin.py:397-413`). So the GPU→scheduler handoff is:
sample on forward stream → GPU ring buffer (for next inputs) + pinned async copy (for output
processing) → CUDA event → `.tolist()` → detokenizer.

## 11. Memory & cache subsystem

*Part I §3.8 described the pools as the scheduler sees them (eviction-aware allocation,
lock-ref contract, cache_finished/unfinished protocol). This section gives the exact structs,
the ownership rules, and the tiers below the GPU.*

### 11.1 Three layers: indirection table, index allocator, physical KV

The module docstring states the layering contract (`srt/mem_cache/memory_pool.py:18-25`,
verbatim):

```python
"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""
```

So there are really three objects: an **indirection table** (`ReqToTokenPool`), an **index
allocator** (`TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator`), and the **physical
tensors** (`KVCache` subclasses). The allocator hands out *row indices into the KVCache
tensors*; nothing in this layer is a "block object" — a page is purely an integer range
convention (`page * page_size + offset`).

**`ReqToTokenPool` — one int32 matrix + a Python free list.** Part I described its role; the
struct itself (`srt/mem_cache/memory_pool.py:127-188`):

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""
    def __init__(self, size, max_context_len, device, enable_memory_saver):
        ...
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))
```

The entire request→token mapping is one dense `(max_running_requests, max_context_len)` int32
GPU tensor; the free list is a plain Python `list`. Slot reuse for chunked prefill (Part I) is
guarded by `assert ... "reusing request must be chunked or have committed KV"`
(`memory_pool.py:156-180`); `free(req)` appends the slot back and nulls `req.req_pool_idx`
(`:182-185`).

**`KVCache` ABC and the MHA/MLA physical layouts.** `KVCache(abc.ABC)`
(`memory_pool.py:668-762`) holds `size`, `page_size`, `dtype` and one notable trick: FP8 caches
are **stored as uint8** because `Tensor.index_put` isn't implemented for fp8 — `# NOTE: Store as
torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2`
(`memory_pool.py:685-689`); readers re-`view(self.dtype)` on access (`:996-998`).

`MHATokenToKVPool._create_buffers` (`memory_pool.py:872-916` ✓verified) — the actual data:

```python
# [size, head_num, head_dim] for each layer
# The padded slot 0 is used for writing dummy outputs from padded tokens.
self.k_buffer = [
    torch.zeros((self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype, device=self.device)
    for _ in range(self.layer_num)
]
self.v_buffer = [...]
```

Per layer one K and one V tensor of `size + page_size` rows; **slot/page 0 is a sacrificial
dummy row** for padded tokens (the allocators below never hand it out). It also precomputes
`data_ptrs`/`data_strides` uint64 tensors over all K+V buffers (`:898-915`) so Triton kernels
(KV copy for spec-decode, `copy_all_layer_kv_cache_tiled`) can address all layers from one
kernel launch (`move_kv_cache`, `:1061-1108`).

`MLATokenToKVPool` collapses K and V into a single per-layer buffer of width
`kv_lora_rank + qk_rope_head_dim`: `self.kv_buffer = [torch.zeros((self.size + self.page_size,
1, self.kv_cache_dim), ...)]` (`memory_pool.py:1537-1552`).

The write path `set_kv_buffer(layer, loc, cache_k, cache_v)` (`memory_pool.py:1022-1059`)
optionally rescales/casts to cache dtype, then dispatches to `_set_kv_buffer_impl` (`:90-124`),
which prefers a JIT `store_cache` kernel and otherwise falls back to `k_cache[indices] = k` —
during CUDA-graph capture it splits K and V writes across an alternate stream (`:115-121`).

**Allocator: tensor free list, page arithmetic, and where "ref-counting" actually lives.**
`BaseTokenToKVPoolAllocator` (`srt/mem_cache/allocator.py:35-114`) has exactly this state:

```python
self.free_pages = None        # 1-D int64 tensor of free page ids
self.release_pages = None     # freed-but-unsorted pages (need_sort mode)
self.is_not_in_free_group = True
self.free_group = []          # batched frees
```

Beyond Part I's head-slice/concat description:

- **Index 0 is never allocatable**: `clear()` initializes `free_pages = torch.arange(1, size+1)`
  for page_size 1 (`allocator.py:131-138`) and starts at page 1 for the paged allocator
  (`:506-510`) — that's the dummy slot above. No compaction, no buddy system — order of
  `free_pages` *is* the allocation order.
- **`need_sort`** is enabled only for PD disaggregation:
  `need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")`
  (`srt/model_executor/model_runner_kv_cache_mixin.py:572`). In that mode frees go to
  `release_pages` and are merged + **sorted** lazily when the free list runs short
  (`merge_and_sort_free`, `allocator.py:82-88`) — keeping transferred KV ranges as contiguous as
  possible.
- **Free batching**: `free_group_begin()/free_group_end()` accumulate frees and apply one
  `torch.cat` (`allocator.py:73-80`). The decode output processor brackets its per-request
  finish loop with exactly this pair
  (`srt/managers/scheduler_output_processor_mixin.py:441,542`) so a batch with many finished
  requests does one tensor concat instead of N.
- **`PagedTokenToKVPoolAllocator`** (`allocator.py:356-513`): `free_pages` holds **page ids**,
  `alloc()` expands them to token indices via
  `out_pages[:, None] * page_size + arange(page_size)` (`:393-399`). `free()` collapses token
  indices back to pages with `torch.unique(free_index // self.page_size)` (`:490-499`).
- **`alloc_extend` is a 3-part Triton kernel** (`alloc_extend_kernel`, `allocator.py:234-317`):
  per request, Part 1 fills the tail of the request's last partial page starting at
  `last_loc + 1`; Part 2 claims whole new pages from `free_page_ptr` in a *runtime-bounded
  blocked loop* (the comment explains: the loop bound is derived from a runtime value "so Triton
  generates a real loop instead of unrolling — ... only one kernel compilation", `:282-290`);
  Part 3 fills the new trailing partial page. The host wrapper then pops `num_new_pages`
  (computed on CPU from `seq_lens_cpu`) off `free_pages` and returns `None` on shortfall
  (`:440-448`) — note the kernel has already written `out_indices` *before* the shortfall check;
  the `None` return makes the caller raise, so those writes are abandoned.
  `alloc_decode_kernel` is the 1-token special case: either `last_loc + 1` within the current
  page or the first slot of one new page (`:320-353`).
- **There is no per-page reference counter in the allocator.** Ownership is single-writer by
  convention: a token index is owned either by a running request (recorded in its
  `req_to_token` row) or by the radix tree (recorded in `TreeNode.value`). Sharing and pinning
  are implemented one level up by the radix tree's `lock_ref` (§11.2), and the handoff is
  explicit: on insert, the request's freshly-written duplicates of already-cached tokens are
  freed immediately — `# Radix Cache takes one ref in memory pool` followed by
  `self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])`
  (`srt/mem_cache/radix_cache.py:513-522` ✓verified).

### 11.2 RadixCache internals: keys, nodes, split/insert, eviction

**`RadixKey` — namespaced, page-quantized keys** (`srt/mem_cache/radix_cache.py:71-208`). A key
is `(token_ids, extra_key, is_bigram)` with `__slots__`. Three load-bearing methods:

- `child_key(page_size)` returns the **hashable dict key for the first page**: `t[0]` for
  page_size 1, else `tuple(t[:page_size])`, and namespaced as `(self.extra_key, plain)` when
  `extra_key` is set (`:183-193`). Children of a node are a `dict` keyed by this — so descending
  one edge is an O(1) hash lookup on the next page's tokens, and different `extra_key`s (LoRA
  id, cache salt) can never collide (`match_prefix` docstring spells this out, `:399-433`).
- `match(other, page_size)` compares token-by-token but **rounds down to page multiples** for
  page_size > 1 (`:152-181`); `page_aligned(page_size)` truncates the key length to a page
  multiple before any tree operation (`:126-130`).
- The `is_bigram` flag re-reads the same `token_ids` as overlapping `(t_i, t_{i+1})` pairs for
  EAGLE spec-decode, flipped in O(1) by `maybe_to_bigram_view` (`:132-143`).

**`TreeNode` — the actual node struct** (`radix_cache.py:211-271`, fields in full):

```python
self.children = defaultdict(TreeNode)
self.parent: TreeNode = None
self.key: RadixKey = None
self.value: Optional[torch.Tensor] = None   # KV-pool indices for this segment
self.lock_ref = 0
self.last_access_time = time.monotonic()
self.creation_time = time.monotonic()
self.hit_count = 0
self.host_ref_counter = 0       # pin count for the host (CPU) copy
self.host_value: Optional[torch.Tensor] = None   # host-pool indices
self.hash_value: Optional[List[str]] = None      # per-page SHA256 chain
self.priority = priority
```

Two derived states define the tier of a node: `evicted` ⇔ `value is None` (no GPU copy) and
`backuped` ⇔ `host_value is not None` (`:238-244`). `__lt__` orders by `last_access_time` so
nodes can sit directly in a heap (`:270-271`). `hash_value` is a per-page SHA256 chain (each
page hash feeds the next, and the first page consumes the parent's last hash —
`compute_node_hash_values`, `:274-291`) used for KV events and the L3 storage keys; on node
split the list is sliced at the page boundary (`split_node_hash_value`, `:294-318`).

**Lookup, splitting, insertion.** `match_prefix` page-aligns the key, then
`_match_prefix_helper` (`radix_cache.py:693-717`) loops: hash-lookup the child by first-page
key, compare with `child.key.match(key)`, and either (a) full edge match → append
`child.value`, descend, re-derive `child_key`; or (b) **partial match → split the node in
place**:

```python
prefix_len = child.key.match(key, page_size=self.page_size)
if prefix_len < len(child.key):
    new_node = self._split_node(child.key, child, prefix_len)
    value.append(new_node.value)
    node = new_node
    break
```

`_split_node` (`:719-739`) inserts `new_node` between parent and child; it **inherits
`lock_ref`, `hit_count`, `priority`** and slices both `key` and `value`
(`new_node.value = child.value[:split_len].clone()`), re-registering both halves in their
parents' child dicts. So a lookup can mutate tree shape — the docstring calls this "structural
refinement ... does not duplicate data" (`:428-433`).

`_insert_helper` (`:749-801`) walks the same way, propagating `priority = max(...)` and
`hit_count` along the matched path, and creates at most one new leaf for the unmatched suffix
(`new_node.value = value.clone(); self.evictable_size_ += len(key)`). The return value
`total_prefix_length` (how much was already present) is exactly what `cache_finished_req` uses
to free the request's duplicated KV (§11.1).

**Eviction: heap over a maintained leaf set; lock_ref is the race protection.** Part I gave the
heap-pop sketch; the structural detail is that the tree maintains `self.evictable_leaves: set`
*incrementally*. `_update_leaf_status` (`radix_cache.py:831-844`) is called on every lock
change, insert, and delete; a node is in the set iff it is **non-evicted, `lock_ref == 0`, and
has no non-evicted children**. `evict` (`:608-635`) then heapifies that set under a pluggable
strategy and pops:

```python
eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
heapq.heapify(eviction_heap)
while num_evicted < num_tokens and len(eviction_heap):
    _priority, x = heapq.heappop(eviction_heap)
    self.token_to_kv_pool_allocator.free(x.value)
    num_evicted += len(x.value)
    self._delete_leaf(x)
    if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
        heapq.heappush(eviction_heap, (self.eviction_strategy.get_priority(x.parent), x.parent))
```

The strategy set is wider than Part I listed: LRU = `last_access_time`, LFU =
`(hit_count, last_access_time)`, FIFO/FILO = ±`creation_time`, MRU = `-last_access_time`,
Priority = `(node.priority, last_access_time)`, SLRU = two segments split at `hit_count >= 2` —
all one-liner priority functions (`srt/mem_cache/evict_policy.py:16-65`).

**Eviction-race protection is logical, not mutex-based.** The whole RadixCache runs inside the
single scheduler thread (no `threading.Lock` exists anywhere in `radix_cache.py`); the race it
must prevent is *temporal*: eviction (triggered by any later allocation,
`evict_from_tree_cache` at `srt/mem_cache/common.py:229-252`) must not free a prefix some
admitted-but-running request points at. That is what `inc_lock_ref` / `dec_lock_ref` do
(`radix_cache.py:637-671`): walk node→root, increment/decrement `lock_ref`, move the byte count
between the two accounting buckets:

```python
if node.lock_ref == 0:
    self.evictable_size_ -= len(node.key)
    self.protected_size_ += len(node.key)
node.lock_ref += 1
self._update_leaf_status(node)   # removes it from evictable_leaves
```

Because `_update_leaf_status` removes locked nodes from `evictable_leaves` immediately, a locked
path can never even enter the eviction heap. `dec_lock_ref` carries a tree-mixup assertion:
`assert node is self.root_node, "This request holds the node from another tree"` (`:666-670`).
The lock lifecycle (taken at admission by `PrefillAdder`, dropped in `cache_finished_req`,
migrated to the deeper node by `cache_unfinished_req`:
`self.dec_lock_ref(req.last_node); self.inc_lock_ref(new_last_node)` at `:586-587`) was covered
in Part I §3.8.

**`cache_unfinished_req` deepened.** Beyond Part I's description, it **re-matches and rewrites
the `req_to_token` row** so the request points at the canonical tree-owned indices:

```python
match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
...
self.req_to_token_pool.write(
    (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
    new_indices[req.cache_protected_len :],
)
req.cache_protected_len = len(new_indices)
```

(`radix_cache.py:535-599`). The long comment at `:580-584` explains `cache_protected_len`'s
reason for existing: with `page_size > 1` the trailing partial page is in `req.prefix_indices`
but *not* in the tree, so a plain `len(prefix_indices)` would double-free or leak it across
chunks. `cache_finished_req` (`:488-533`) frees three disjoint ranges of the request's own
indices — the newly-discovered duplicates `[cache_protected_len, new_prefix_len)`, (when
insertion is disabled) the whole uncached span, and always the unaligned tail
`kv_indices[key_len:]` — and finally drops the lock.

### 11.3 How a request maps to physical memory

The full indirection chain is
`Req.req_pool_idx → req_to_token[slot, pos] → kv_index → k_buffer[layer][kv_index]`.
Request-side bookkeeping lives in `Req`: `kv_committed_len` (KV actually written) initialized at
`srt/managers/schedule_batch.py:637`, `cache_protected_len` at `:745` (with a comment block at
`:647-648` describing which ranges are freed by whom), and pop-style accessors that enforce
single-free: `pop_committed_kv_cache` asserts `"Committed KV cache already freed"`
(`:931-935`), `pop_overallocated_kv_cache` (`:939-947`).

**Prefill wiring** — Part I covered `alloc_for_extend` at the helper level
(`srt/mem_cache/common.py:328-391`); the kernel-level detail: paged extend over-reserves one
page per request before evicting
(`num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size`, `:265-268`), then
`write_cache_indices` fills each request's `req_to_token` row in one Triton launch
(`write_req_to_token_pool_triton`, `common.py:27-75`): block-copy the matched prefix's indices
from `req.prefix_indices` (passed as raw data pointers, `:92-96`), then copy this request's
slice of the newly allocated `out_cache_loc` at offset `pre_len`. Paged allocation needs each
request's `last_loc` (last physical index of its prefix) to continue partial pages; that is
fetched from the table itself by `get_last_loc_kernel` (`common.py:155-198`).

**Decode wiring** — `alloc_for_decode` (`common.py:423-462`): one token per request,
`last_loc = req_to_token[req_pool_indices, seq_lens - 1]`, then a single row-write
`req_to_token_pool.write((req_pool_indices, locs), out_cache_loc.to(torch.int32))`.

**Consumption** — attention backends read the table directly as the page table (FlashAttention
gather quoted in §10.3; the Triton backend caches
`self.req_to_token = model_runner.req_to_token_pool.req_to_token` at init,
`srt/layers/attention/triton_backend.py:129`).

**Teardown** — `release_kv_cache(req, tree_cache, is_insert)` (`common.py:465-515`):
`cache_finished_req` (insert + dedup + unlock, §11.2), then free the over-allocated tail
`[ceil_align(start_p, page_size), end_p)` from the request's row, then
`req_to_token_pool.free(req)`. The non-spec invariant is asserted:
`assert start_p == end_p, f"Unexpected overallocated KV cache..."` (`:494-497`). The
memory-pressure path deliberately bypasses the tree: `retract_decode` evicts
newest-output-first and "release[s] memory and do[es]n't insert into the tree because we need
the space instantly" (`srt/managers/schedule_batch.py:2134-2168`), aborting (not crashing) even
the last request if it still cannot fit (`:2170-2187`).

### 11.4 Cache invalidation and its guarantees

**`flush_cache`** (`srt/managers/scheduler.py:3229-3254` ✓verified) is the single
full-invalidation primitive:

```python
def flush_cache(self):
    """Flush the memory pool and cache."""
    if self.is_fully_idle():
        self.cur_batch = None
        self.last_batch = None
        self.tree_cache.reset()
        self.req_to_token_pool.clear()
        self.token_to_kv_pool_allocator.clear()
        self.grammar_manager.clear()
        ...
        torch.cuda.empty_cache()
```

Its guarantee is **refusal, not draining**: it only runs when `is_fully_idle()` holds — running
batch empty, no chunked request, last/cur batch empty, overlap result queue empty, waiting queue
empty, PP microbatches empty (`scheduler.py:3082-3100`) — otherwise it logs and returns `False`
(`:3247-3253`). The RPC wrapper adds an optional deferred mode: with `timeout_s > 0` the request
parks in `self._pending_flush` and `_check_pending_flush` retries each loop iteration until idle
or deadline (`scheduler.py:3053-3070`, `2997-3021`). Note what reset means physically:
`tree_cache.reset()` rebuilds an empty root (`radix_cache.py:385-396`, emitting
`AllBlocksCleared` to KV-event subscribers) and the allocators rebuild their `arange` free
lists — **the KV tensors themselves are not zeroed**; stale data is unreachable because no index
maps to it.

**Weight updates**: every update flavor (`disk`, `distributed`, `tensor`, `IPC`) flushes when
the caller asks, and *asserts* the flush succeeded rather than serving stale KV against new
weights:

```python
if tp_success and recv_req.flush_cache:
    flush_cache_success = self.flush_cache()
    assert flush_cache_success, "Cache flush failed after updating weights"
```

(`srt/managers/scheduler_update_weights_mixin.py:54-56`; repeated at `:82-84`, `:100-102`,
`:116-118`). Since `flush_cache` fails when not idle, the contract is on the RLHF caller to
drain first — the assert turns a stale-cache hazard into a crash. `release_memory_occupation`
(sleep mode) likewise requires idleness (`assert self.is_fully_idle()`, `:131-133`) and flushes
before pausing the KV-cache memory region via the memory-saver adapter (`:143-145`). The
hierarchical L3 has a separate invalidation RPC: `clear_hicache_storage_wrapped →
tree_cache.clear_storage_backend()` (`scheduler.py:3072-3080`).

### 11.5 Tiered memory: `HiRadixCache`, host pool, L3 storage

This checkout has a full three-tier hierarchy (GPU → pinned host RAM → pluggable storage), all
keyed off the *same* radix tree: a `TreeNode` with `value=None, host_value!=None` *is* the
host-tier entry.

**Host pool** — `HostKVCache` (`srt/mem_cache/memory_pool_host.py:155-288`): sized by
`--hicache-size` GB or `device_pool.size * host_to_device_ratio`, page-aligned, with
`assert self.size > device_pool.size` ("host memory should be larger than the device memory
with the current protocol", `:187-189`) and a `psutil.virtual_memory()` admission check
(`:191-201`). Its allocator is the same head-slice free list as the GPU one
(`free_slots = torch.arange(self.size)`; `alloc`/`free` at `:272-288`) but — unlike anything on
the GPU side — guarded by a real `threading.RLock` (`self.lock = threading.RLock()` +
`@synchronized`, `:209-211`), because the `HiCacheController` moves data on background threads.
`MHATokenToKVPoolHost` allocates one big pinned buffer with four selectable layouts
(`layer_first`, `page_first`, `page_first_direct`, `page_head`, `:351-375`).

**Write-down (GPU→CPU)** — `HiRadixCache._inc_hit_count` triggers `write_backup(node)` once
`hit_count >= write_through_threshold` (1 for `write_through`, 2 for
`write_through_selective`; `srt/mem_cache/hiradix_cache.py:708-717`, threshold set at
`:178-180`). `write_backup` enforces a contiguity invariant — "backed-up nodes must form a
contiguous prefix from root — no gaps. Skip if parent isn't backed up yet" (`:656-663`) —
allocates host indices (evicting host on shortfall), records the node in
`ongoing_write_through`, and **locks the node** (`inc_lock_ref`) until the async DMA completes
(`:665-689`). Completion is consumed by `writing_check` (`:719-766`), which contains the
cross-rank correctness trick: each rank counts finished CUDA events, then `all_reduce(MIN)` over
the TP group so **all ranks apply the identical set of tree updates** (`# synchronize TP workers
to make the same update to radix cache`, `:747-753`), then per-acked-node emits the CPU store
event, `dec_lock_ref`, and optionally chains `write_backup_storage` to L3 (`:759-765`).

**Eviction becomes demotion** — `HiRadixCache.evict` (`:843-889`) pops the same
evictable-leaves heap but branches: backuped node → `_evict_backuped` (free GPU indices via
`cache_controller.evict_device`, set `node.value = None`, node stays in tree as host-tier,
`:891-904`); non-backuped node under `write_back` policy → synchronously write it down first,
then demote (`:860-866`, `:882-886`); otherwise → `_evict_regular`, a true delete
(`:906-914`). Unlike base `RadixCache.evict`, it re-checks `if x.lock_ref > 0: continue` after
popping (`:857-858`). A second heap over `evictable_host_leaves` implements host-tier eviction
with `host_ref_counter` as the pin (`evict_host`, `:916-949`; pins via
`protect_host()/release_host()`, `radix_cache.py:246-255`).

**Lookup and promotion** — `HiRadixCache.match_prefix` walks the unified tree, then splits the
result into a device part and a host part by walking back over evicted nodes:
`while last_node.evicted: host_hit_length += len(last_node.host_value); last_node =
last_node.parent`, returning `MatchResult(device_indices, last_device_node, last_host_node,
host_hit_length)` (`hiradix_cache.py:1221-1255`). The scheduler can then call `init_load_back`
→ `load_back` (`:951-1021`): collect the evicted chain, `inc_lock_ref` the on-device ancestor
so it can't vanish mid-copy, apply an all-or-nothing policy (skip if
`< load_back_threshold = 10` tokens or over `mem_quota`, `:971-978`), allocate device indices
via `cache_controller.load` (GPU-evicting on shortfall, `:985-991`), re-point each node's
`value` slice, and lock the target node until `loading_check` sees the CUDA event finish and
unlocks (`:768-781`). Layer-wise overlap of load-with-compute is wired through the pool:
`get_key_buffer` blocks per layer on `layer_transfer_counter.wait_until(layer_id -
start_layer)` when a transfer is in flight (`memory_pool.py:1000-1006`).

**L3 storage** — `HiCacheStorage(ABC)` is a get/set/exists key-value interface over page-hash
keys (`srt/mem_cache/hicache_storage.py:98-249`) with a reference `HiCacheFile` file-per-page
backend (`:277+`); selected by `--hicache-storage-backend` (`hiradix_cache.py:106`, controller
wiring `:135-155`). Prefetch into the host tier is asynchronous and rate-limited:
`prefetch_from_storage` page-aligns the uncached suffix, requires
`prefetch_length >= prefetch_threshold`, pins `last_host_node.protect_host()`, allocates host
indices (evicting host if needed), and registers the operation in `ongoing_prefetch`
(`hiradix_cache.py:1257-1305`). Storage hits are inserted as host-only nodes
(`new_node.value = None; new_node.host_value = host_value.clone()`, `_insert_helper_host`,
`:1333-1340`).

## 12. Extension surfaces: adding a model, backend, or config knob

### 12.1 Adding a new model end-to-end

**Registry mechanism: `EntryClass` + package scan.** There is no decorator and no
hand-maintained list. The registry is a single dataclass keyed by HF architecture string
(`srt/models/registry.py:17-20`):

```python
@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)
```

Population is a `pkgutil` scan over `sglang.srt.models` that imports every module and picks up
the module-level `EntryClass` attribute; the architecture key is literally the **class name**
(`srt/models/registry.py:92-125`):

```python
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        ...
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(entry, list):  # To support multiple model classes in one module
                    for tmp in entry:
                        ...
                        model_arch_name_to_cls[tmp.__name__] = tmp
```

Import errors are swallowed with a warning unless `strict` (`registry.py:102-108`), and
individual archs can be excluded via `SGLANG_DISABLED_MODEL_ARCHS` (`registry.py:98-100`). The
module-level singleton registers the built-in package at import time and allows an out-of-tree
model package to overwrite entries (`registry.py:128-132` ✓verified):

```python
ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```

So the three extension points for a new model are: (1) drop a file in `srt/models/` with
`EntryClass = <YourClass>`; (2) ship an external package and set
`SGLANG_EXTERNAL_MODEL_PACKAGE`; (3) do nothing and fall back to the Transformers backend.

**Architecture resolution & config detection.** `ModelConfig.__init__` loads the HF config and
derives capability flags purely from `hf_config.architectures` and config attributes:
`self.is_generation = is_generation_model(self.hf_config.architectures, is_embedding)`
(`srt/configs/model_config.py:180-182`), plus `is_multimodal`, `is_encoder_decoder`, dtype
verification via `_get_and_verify_dtype` (`model_config.py:184-224`), context length derivation
and quantization checks (`model_config.py:227-239`). `ModelConfig.from_server_args` is the
single constructor used by workers (`model_config.py:263-300`).

Architecture → class resolution happens in `get_model_architecture`
(`srt/model_loader/utils.py:193-230`): it reads
`getattr(model_config.hf_config, "architectures", [])`, special-cases quantized Mixtral
(`utils.py:212`), then:

```python
is_native_supported = any(arch in supported_archs for arch in architectures)
...
elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
    architectures = resolve_transformers_arch(model_config, architectures)
model_cls, resolved_arch = ModelRegistry.resolve_model_cls(architectures)
```
(`model_loader/utils.py:215-221`)

The Transformers fallback synthesizes an arch name from detected capabilities —
pooling/multimodal/MoE — e.g. `TransformersMoEForCausalLM` (`_get_transformers_backend_arch`,
`model_loader/utils.py:74-99`; MoE detection probes `num_local_experts`, `n_routed_experts`
etc. on the text config, `utils.py:30-63`). Independently, `_normalize_archs` in the registry
appends `"TransformersForCausalLM"` as last-resort whenever any requested arch is unknown
(`srt/models/registry.py:73-76`).

**The model interface contract, traced through `ExaoneForCausalLM`.** `srt/models/exaone.py`
(377 lines ✓verified) is a clean minimal example. The contract has exactly three parts:

*1. Constructor* — must accept the kwargs that `_initialize_model` passes
(`srt/model_loader/loader.py:261-281`):

```python
model_class, _ = get_model_architecture(model_config)
kwargs = {
    "config": model_config.hf_config,
    "quant_config": quant_config,
}
...
return model_class(**kwargs)
```

Exaone's matches: `def __init__(self, config, quant_config=None, prefix="")`
(`exaone.py:297-303`). Inside, the model is assembled from SGLang's parallel layer library, not
raw `nn.Linear`: `QKVParallelLinear`/`RowParallelLinear`/`MergedColumnParallelLinear`
(`exaone.py:128-143`, `56-69`), `VocabParallelEmbedding` + `ParallelLMHead`
(`exaone.py:255-258`, `313-317`), and — the key serving hook — `RadixAttention` carrying
`layer_id` so the attention layer can index the paged KV pool (`exaone.py:153-160`). TP
sharding decisions (head partition vs. KV-head replication) live in the model file itself
(`exaone.py:101-114`).

*2. `forward`* — signature is `(input_ids, positions, forward_batch: ForwardBatch,
input_embeds=None)` and the CausalLM wrapper ends in `LogitsProcessor`, returning a
`LogitsProcessorOutput` rather than logits (`exaone.py:320-333`):

```python
@torch.no_grad()
def forward(self, input_ids, positions, forward_batch, input_embeds=None) -> LogitsProcessorOutput:
    hidden_states = self.transformer(input_ids, positions, forward_batch, input_embeds)
    return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
```

The attention sub-layer just calls `self.attn(q, k, v, forward_batch)` (`exaone.py:171`) — all
KV-cache/backend logic is behind `RadixAttention`.

*3. `load_weights`* — receives an iterator of `(hf_name, tensor)` and must map HF checkpoint
names onto the fused parallel layers via a `stacked_params_mapping`, calling each parameter's
attached `weight_loader` (`exaone.py:335-374`):

```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "c_fc_0", 0),
    ("gate_up_proj", "c_fc_1", 1),
]
```

Finally `EntryClass = ExaoneForCausalLM` (`exaone.py:377`) — matching the HF config's
`architectures: ["ExaoneForCausalLM"]`.

The loader drives this contract in `DefaultModelLoader.load_model`
(`srt/model_loader/loader.py:675-719`): instantiate under
`set_default_torch_dtype(model_config.dtype)` and the target device context
(`loader.py:691-697`), then

```python
@staticmethod
def load_weights_and_postprocess(model, weights, target_device):
    model.load_weights(weights)
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            ...
            quant_method.process_weights_after_loading(module)
```
(`loader.py:707-719`)

Optional class attributes the loader consults: `packed_modules_mapping`, `remap_prefix`,
`hf_to_sglang_mapper` (for quant config name remapping, `loader.py:197-199`, `253-256`),
`post_load_weights` (`model_loader/utils.py:251-258`), and `load_weights_to_module` if you want
`--load-format layered` support (`loader.py:753-760`). Loader selection itself is a
`load_format` switch in `get_model_loader` (`loader.py:3148+`).

**Adding an attention backend** is the other big extension surface — covered in §10.3:
implement the `AttentionBackend` ABC, register with `@register_attention_backend("name")`, and
optionally add a default-selection heuristic in `ServerArgs._get_default_attn_backend`.

### 12.2 The config/args system

`ServerArgs` is a single flat dataclass — 7310 lines of file (✓verified `wc -l`), fields
grouped by comment headers (`srt/server_args.py:292-316`), with an explicit consistency
convention in the docstring:

```python
@dataclasses.dataclass
class ServerArgs:
    """...
    NOTE: When you add new arguments, please make sure the order
    in this class definition the same as the order in the function
    `ServerArgs.add_cli_args`.
    """
    model_path: str
    tokenizer_path: Optional[str] = None
```

**Declaration → CLI.** `add_cli_args` mirrors every field manually, pulling defaults from the
class attributes so the dataclass stays the single source of default values:
`default=ServerArgs.tokenizer_path` (`server_args.py:4056-4086`). The reverse direction,
`from_cli_args`, constructs the dataclass generically from `dataclasses.fields`, after aliasing
long CLI spellings (`args.tp_size = args.tensor_parallel_size`, `server_args.py:6508-6517`).

**Entry & file config.** `prepare_server_args` builds the parser, and if `--config` is present,
merges a YAML/config file into argv via `ConfigArgumentMerger` before parsing
(`server_args.py:7070-7104`; merger lives in `srt/server_args_config_parser.py`).

**Validation in two stages.** (1) `__post_init__` is a long normalization pipeline —
deprecated-arg handling, device-specific backend selection, GPU-memory-derived defaults
(`mem_fraction_static`, chunked prefill, CUDA-graph batch sizes), attention/sampling/grammar
backend choices (`server_args.py:763-840`). (2) `check_server_args()` (`server_args.py:6605`)
runs later, inside `_launch_subprocesses` (`srt/entrypoints/engine.py:673`), as the launch-time
assertion pass.

**Plumbing to workers.** The entire `ServerArgs` object is pickled into each scheduler
subprocess as a positional arg of `mp.Process(target=run_scheduler_process, args=(server_args,
port_args, gpu_id, tp_rank, ...))` (`engine.py:577-591`). Inside the worker,
`ModelRunner.__init__` installs it as a process-global:
`set_global_server_args_for_scheduler(server_args)` (`srt/model_executor/model_runner.py:447`),
with the accessor raising if unset (`server_args.py:7055-7067`). Deep code (e.g.
`LayeredModelLoader`) then reaches it via `get_global_server_args()` (`loader.py:740`). A
sibling dataclass `PortArgs` carries the ZMQ IPC names + NCCL port and is created by
`PortArgs.init_new(server_args)` (`server_args.py:7111-7134`; `engine.py:677-678`).

**Engine API parity.** `Engine(**kwargs)` accepts the same field names and just does
`server_args = self.server_args_class(**kwargs)` (defaulting `log_level="error"`,
`engine.py:178-187`).

**Env vars** are not raw `os.environ` reads in new code: `srt/environ.py` defines typed fields
(`EnvStr`/`EnvBool`/`EnvInt`, `environ.py:114-129`) collected in a registry class `Envs`
(`environ.py:159`), e.g. `SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")` (`environ.py:531`),
accessed as `envs.X.get()` everywhere.

## 13. Startup sequence & failure handling

### 13.1 Boot: numbered order of operations

Part I §2/§4 covered the process topology; this is the precise order with the in-process init
phases.

1. **CLI entry.** `sglang serve` / `python -m sglang.launch_server` → `load_plugins()`,
   `prepare_server_args(sys.argv[1:])`, `run_server()` which dispatches on
   `grpc_mode`/`encoder_only`/`use_ray` to `srt/entrypoints/http_server.py::launch_server`
   (`python/sglang/launch_server.py:15-71`).
2. **Subprocess launch.** `launch_server` → `Engine._launch_subprocesses(...)`
   (`srt/entrypoints/http_server.py:2336-2348`). That classmethod: `configure_logger`,
   `_set_envs_and_config` (`engine.py:666-667`) — which sets NCCL/CUDA env vars, registers the
   **launch-phase SIGQUIT handler**, and forces `mp.set_start_method("spawn", force=True)`
   (`engine.py:1121-1206`) — then `check_server_args()`, `PortArgs.init_new`
   (`engine.py:673-678`).
3. **Scheduler process spawn.** For `dp_size==1`, one `mp.Process(target=run_scheduler_process)`
   per `(pp_rank, tp_rank)` with a `mp.Pipe` writer for the readiness handshake and computed
   `gpu_id` (`engine.py:546-598`); for `dp_size>1` a single
   `run_data_parallel_controller_process` which fans out (`engine.py:599-613`).
4. **Child process setup.** `run_scheduler_process`: `load_plugins()`,
   `configure_scheduler_process` (proctitle `sglang::scheduler_TP...`, `faulthandler.enable()`,
   logger prefix, CPU/NUMA affinity) (`srt/managers/scheduler.py:3764-3787`, `3707-3761`),
   optional OTLP tracing init (`3791-3798`), then `Scheduler(...)` inside a `try`.
5. **Scheduler.__init__ ordered phases** (`scheduler.py:402-449`): `init_model_config()` (:403)
   → `init_metrics()` (:406) → `init_ipc_channels(port_args)` (:409, ZMQ sockets) →
   `init_tokenizer()` (:416) → `init_model_worker()` (:425) → `init_cache_with_memory_pool()`
   (:431) → `init_running_status()` / `init_chunked_prefill()` / `init_schedule_policy()`
   (:434-443) → watchdog + profiler init (:446-449).
6. **Model worker.** `init_model_worker` constructs `TpModelWorker` (`scheduler.py:634-636`),
   whose `__init__` runs `_init_model_config()` (→ `ModelConfig.from_server_args`) then
   `_init_model_runner()`, loads the tokenizer/processor, and derives serving limits from the
   runner: `max_total_num_tokens`, `max_running_requests`,
   `max_req_input_len = max_req_len - 5` (`srt/managers/tp_worker.py:220-307`).
7. **ModelRunner pre-init.** `ModelRunner.__init__`: `model_specific_adjustment()`,
   `set_global_server_args_for_scheduler` (`model_runner.py:444-447`), then
   `pre_model_load_memory = self.init_torch_distributed()` (`model_runner.py:458`) — which sets
   the device, picks the backend, and calls
   `init_distributed_environment(world_size=tp*pp, rank=tp*pp_rank+tp_rank, ...)`
   (`model_runner.py:987-1061`; Part I §4 has the group details) — then
   `self.initialize(pre_model_load_memory)` (`model_runner.py:485`).
8. **Model load.** `initialize()` → `self.load_model()` (`model_runner.py:575`); details in
   §13.2.
9. **Memory pool + backends.** Still in `initialize()`: `configure_kv_cache_dtype()` (:680) →
   `init_memory_pool(pre_model_load_memory)` (:683; profiles free memory and builds
   `ReqToTokenPool` + KV allocator,
   `srt/model_executor/model_runner_kv_cache_mixin.py:756-771`, §13.2) → for CUDA:
   `init_cublas(); init_attention_backend(); kernel_warmup(); ...; init_device_graphs()` (CUDA
   graph capture) (`model_runner.py:716-721`). I.e. *profile after weights, capture graphs
   after pools*, so graph memory comes out of the non-static fraction.
10. **Radix cache.** Back in the scheduler, `init_cache_with_memory_pool` pulls the pools from
    the worker (`self.req_to_token_pool, self.token_to_kv_pool_allocator =
    self.tp_worker.get_memory_pool()`, `scheduler.py:778-780`) and selects the tree-cache class
    from a decision ladder: `ChunkCache`/`SWAChunkCache` if radix disabled, else
    `RadixCacheCpp` / `HiRadixCache` / mamba/SWA/unified variants by flags
    (`scheduler.py:795-852`).
11. **Readiness handshake.** Child sends `pipe_writer.send(scheduler.get_init_info())` —
    `{"status": "ready", "max_total_num_tokens": ..., "max_req_input_len": ...}`
    (`scheduler.py:3815`, `1363-1376`); parent's `wait_for_ready` collects these from all pipes
    (`engine.py:618-625`, called at :762) and copies `max_req_input_len` onto the tokenizer
    manager (`engine.py:765-767`).
12. **Detokenizer + tokenizer + watchdog.** Parent spawns the detokenizer process
    (`engine.py:741-748`), builds `TokenizerManager` (or `MultiTokenizerRouter`) in-process
    (`engine.py:752-758`), and starts a `SubprocessWatchdog` over all children
    (`engine.py:771-778`).
13. **HTTP serving + warmup.** `launch_server` → `_setup_and_run_http_server` → `uvicorn`
    (`http_server.py:2240`, `2260`, `2291`). The FastAPI `lifespan` hook starts a
    `_wait_and_warmup` thread (`http_server.py:287-400`) which sends a real `/generate` warmup
    request via `_execute_server_warmup` (`http_server.py:1856+`) and finally logs **"The
    server is fired up and ready to roll!"** (`http_server.py:2031`).
14. **Steady state.** The child meanwhile blocks in `scheduler.run_event_loop()` →
    `dispatch_event_loop` (`scheduler.py:3818`, `3678-3704`; Part I §3.2-3.3).

### 13.2 Weight loading and memory profiling

**The measurement protocol.** The KV cache size is derived from **three free-memory snapshots**
around model load. `get_available_gpu_memory` reads `torch.cuda.mem_get_info(gpu_id)` (true
driver free memory, after `empty_cache()`), and in distributed mode all-reduces with `MIN` over
the gloo CPU group so every rank agrees on the most-constrained GPU
(`srt/utils/common.py:498-613`, all-reduce at `:606-611`).

1. **Snapshot A** — after `init_torch_distributed` (NCCL buffers already allocated), before
   weights: `pre_model_load_memory` (`srt/model_executor/model_runner.py:1098-1103`, returned
   at `:1123` and threaded into `initialize(pre_model_load_memory)` at `:458,485`).
2. **Snapshot B** — after `load_model()`, inside `_profile_available_bytes`, which computes
   (✓verified `srt/model_executor/model_runner_kv_cache_mixin.py:56-70`):

```python
rest_memory = post_model_load_memory - pre_model_load_memory * (
    1 - self.mem_fraction_static
)
...
return int(rest_memory * (1 << 30))  # bytes
```

So the reservation for activations/graphs is `(1 - mem_fraction_static) × pre-load free
memory` — `mem_fraction_static` itself is auto-derived as
`(GPU capacity - reserved_mem) / capacity` with
`reserved_mem ≈ chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2` GB (heuristic, documented
at `srt/server_args.py:1285-1296`).

3. **Snapshot C** — around CUDA graph capture, purely for logging `graph_mem_usage`
   (`model_runner.py:2586-2618`).

**Bytes → tokens: the pool configurator.** `init_memory_pool` → `_resolve_memory_pool_config`
feeds the profiled bytes into an architecture-specific configurator under an explicit linear
model — "`available_bytes = max_tokens * coeff + bias`"
(`srt/model_executor/pool_configurator.py:1-12`). `DefaultPoolConfigurator._compute_cell_size`
is the per-token cost: MHA = `num_kv_heads(tp) * (head_dim + v_head_dim) * num_layers *
kv_dtype_size`; MLA = `(kv_lora_rank + qk_rope_head_dim) * num_layers * kv_size`, plus
FP4-scale and NSA-indexer overheads (`pool_configurator.py:113-168`). Then
`max_total_num_tokens = available_bytes // cell_size`, page-aligned (`:170-175`).
`MemoryPoolConfig.__post_init__` raises "Not enough memory. Please try to increase
--mem-fraction-static." if it goes non-positive (`:39-44`). Hybrid-SWA models get a two-pool
solver where `cell_size = F*nf + ratio*S*ns` (`:184-273`); Mamba models pre-subtract state
memory by solving a ratio equation (`model_runner_kv_cache_mixin.py:72-131`).

Constraints are applied after profiling: user `--max-total-tokens` caps it, and under PP the
capacity is **all-reduced MIN across ranks** since layer counts differ per stage
(`model_runner_kv_cache_mixin.py:669-696`). `max_running_requests` is then derived:
`estimated = clamp(token_capacity / context_len * 512, 2048, 4096)`,
`min(estimated, token_capacity // 2)` unless user-set (`:698-717`). `_init_pools` finally
allocates `ReqToTokenPool(size=max_num_reqs, max_context_len=context_len+4(+draft_tokens))`
(`:199-278`), the KV cache itself (§11.1 layouts; e.g. `MHATokenToKVPool` construction at
`:551-569`), and the allocator (`TokenToKVPoolAllocator` if `page_size==1` else
`PagedTokenToKVPoolAllocator`, `model_runner_kv_cache_mixin.py:640-656`).

**Weight loading path.** `load_model` (`model_runner.py:1170-1397`): downgrades dtype to fp16
below SM80 (`:1180-1188`), builds a `LoadConfig` and resolves a loader via `get_model_loader`
(default `DefaultModelLoader`), loads inside a memory-saver region tagged
`GPU_MEMORY_TYPE_WEIGHTS` (so RLHF can release/resume weights) (`:1255-1266`), measures
`weight_load_mem_usage` from before/after snapshots (`:1344-1356`), pre-expands the RoPE cache
before graph capture (`:1372-1378`), and ends with a `monitored_barrier` with a custom error
naming the slow/OOM rank (`:1384-1396`).

`DefaultModelLoader.load_model` is the vLLM-style two-phase pattern
(`srt/model_loader/loader.py:675-704`):

```python
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = _initialize_model(model_config, self.load_config, quant_config)
    self.load_weights_and_postprocess(
        model, self._get_all_weights(model_config, model), target_device)
```

— architecture class instantiated **directly on the GPU** with empty weights, then
`model.load_weights(weights_iterator)` streams `(name, tensor)` pairs from disk (the §12.1
contract). The iterator is multi-threaded safetensors by default
(`buffered_multi_thread_safetensors_weights_iterator`, 8 threads, with optional mmap-disable
and prefetch knobs, `:481-545`).

### 13.3 Abort/cancel paths

**Client disconnect → abort.** The maintainers keep a literal decision table as a comment
(✓verified `srt/managers/tokenizer_manager.py:2746-2757`):

```python
# | entrypoint | is_streaming | status          | abort engine    | cancel asyncio task   | rid_to_state                |
# | http       | yes          | waiting queue   | background task | fast api              | del in _handle_abort_req    |
# | http       | no           | running         | type 3          | type 3 exception      | del in _handle_batch_output |
```

**Streaming**: `/generate` returns `StreamingResponse(...,
background=_global_state.tokenizer_manager.create_abort_task(obj))`
(`srt/entrypoints/http_server.py:722-726`). FastAPI runs the background task when the response
ends *for any reason including disconnect*; the task sleeps 2 s then calls
`self.abort_request(rid)` (`tokenizer_manager.py:1584-1596`).

**Non-streaming**: `_wait_one_response` waits on the per-request event with a timeout, and on
each timeout (request still queued — "type 1") or after each non-stream chunk (running — "type
3") polls `await request.is_disconnected()`; if so it calls `self.abort_request(obj.rid)` and
raises `ValueError` to tear down the asyncio task (`tokenizer_manager.py:1276-1293`,
`1360-1371`). `TokenizerManager.abort_request` just ships an `AbortReq` over ZMQ to the
scheduler (`tokenizer_manager.py:1471-1480`).

**Scheduler-side abort: three escalation levels.** `Scheduler.abort_request`
(`srt/managers/scheduler.py:3344-3444`) handles the request depending on how far it has
progressed:

- **Method 1 — still waiting**: pop from `waiting_queue` and *echo an `AbortReq` back* so the
  tokenizer cleans its state: `self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)`
  (`scheduler.py:3353-3361`). PD-disaggregation extras: release KV/metadata buffers and abort
  `disagg_kv_sender`/`kv_receiver` queues (`scheduler.py:3362-3428`).
- **Method 2 — in grammar queue**: `self.grammar_manager.abort_requests(recv_req)`; the request
  "will still run one prefill forward pass... we change the input_ids to be only one token to
  make this prefill cheap" (`scheduler.py:3381-3385`).
- **Method 3 — running**: set `req.to_finish = FINISH_ABORT()` (`scheduler.py:3436-3444`). No
  special cleanup path exists: `check_finished` promotes `to_finish` to `finished_reason`
  (`srt/managers/schedule_batch.py:1194-1201`), so the request exits through the *normal*
  finish path on the next decode step, reusing all KV-cache freeing code. Matching by
  `req.rid.startswith(recv_req.rid)` also gives prefix-abort of whole batch requests.

The tokenizer's `_handle_abort_req` then marks the state finished and synthesizes a final
output with `finish_reason = {"type": "abort", ...}`, including logprobs accumulated so far
(`tokenizer_manager.py:2345-2384`).

### 13.4 Crashes, watchdogs, graceful exit

`Scheduler.run_batch` has **no** try/except (`scheduler.py:2779+`); a CUDA error propagates up
through the event loop to the single catch in `run_scheduler_process`
(`scheduler.py:3820-3823`):

```python
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
```

The parent's response depends on phase: during launch, the SIGQUIT handler is
`kill_process_tree(os.getpid())` (`engine.py:1185-1191`, replaceable via
`--custom-sigquit-handler`, :1192-1197); once the tokenizer manager's loop runs, it swaps in
`running_phase_sigquit_handler`, which stops the subprocess watchdog, **dumps in-flight
requests to the crash-dump folder**, then kills the tree (`tokenizer_manager.py:1614-1617`,
`2734-2743`). SIGTERM is the graceful path: `sigterm_handler` only sets
`gracefully_exit = True` and a `sigterm_watchdog` task drains (`tokenizer_manager.py:2728-2732`,
`2303`).

Two watchdogs cover the non-exception failure modes:

- **Hang detection** — `WatchdogRaw` polls a counter (`forward_ct` for the scheduler) while
  `is_active` (a batch in flight); if it stalls past `--watchdog-timeout`, it dumps batch info
  + py-spy stacks of all scheduler processes, then
  `self.parent_process.send_signal(signal.SIGQUIT)` (`srt/utils/watchdog.py:133-163`; factory
  `srt/managers/scheduler_runtime_checker_mixin.py:563-585`; created at `scheduler.py:1036`,
  plus an optional soft variant at :1029-1031).
- **Hard-crash detection** — `SubprocessWatchdog` polls child liveness, because "NCCL timeout
  causing C++ `std::terminate()`" never runs Python handlers and would leave a zombie service
  (`watchdog.py:166-175`).

Idle-time invariant checking complements this: `on_idle` runs full memory-pool leak checks and
radix-tree `sanity_check()` whenever the scheduler is fully idle
(`scheduler_runtime_checker_mixin.py:537-562`, `_check_all_pools` :468-489).

### 13.5 Observability hooks

**Prometheus metrics.** `--enable-metrics` sets `PROMETHEUS_MULTIPROC_DIR` before any
`prometheus_client` import (`engine.py:1153-1155`) — necessary because scheduler/detokenizer
are separate processes. The FastAPI app mounts `/metrics` with a `MultiProcessCollector`
(`srt/utils/common.py:1389-1399`) via `add_prometheus_middleware(app)`
(`http_server.py:302-304`). Collectors live in `srt/observability/metrics_collector.py`:
`SchedulerStats` (dataclass of the loggable snapshot, :78) and `SchedulerMetricsCollector` with
gauges `sglang:num_running_reqs`, `token_usage`, `gen_throughput`, `cache_hit_rate`, etc.
(:179-280); `TokenizerMetricsCollector` (:1146) covers per-request latency/abort counters
(e.g. `observe_one_aborted_request`, used at `tokenizer_manager.py:1476-1480`). The scheduler
builds its label set (`model_name`, `engine_type` unified/prefill/decode, tp/pp/dp/ep ranks,
`--extra-metric-labels`) in `init_metrics` (`srt/observability/scheduler_metrics_mixin.py:90-151`);
throughput stats are emitted by `report_prefill_stats` (:329) and `report_decode_stats`, gated
by `--decode-log-interval` (:463, :495-517); idle metrics flush every 30 s
(`scheduler_runtime_checker_mixin.py:491-527`). Optional MFU metrics (`enable_mfu_metrics`)
estimate flops/bytes per step (`scheduler_metrics_mixin.py:152-157`).

**Profiling.** HTTP `/start_profile` and `/stop_profile` (`http_server.py:948`, `973`) reach
`SchedulerProfilerMixin`: `init_profiler` sets up state incl. target forward/prefill/decode
counts (`srt/managers/scheduler_profiler_mixin.py:38-65`), `start_profile` builds a
`torch.profiler.profile(...)` (CPU/CUDA/XPU activities, `with_stack`, `record_shapes`) or a
ROCm RPD tracer (:138-200). Profiling can be armed by batch count: `run_batch` calls
`self._profile_batch_predicate(batch)` on every step (`scheduler.py:2787-2788`; predicate at
`scheduler_profiler_mixin.py:339`).

**Tracing.** `--enable-trace` + `--otlp-traces-endpoint` initialize OpenTelemetry per process:
`process_tracing_init` (`srt/observability/trace.py:160`) is called both in `Engine.__init__`
(thread labeled "Tokenizer"/"Prefill Tokenizer"/..., `engine.py:231-239`) and in
`run_scheduler_process` (`scheduler.py:3790-3798`); request-scoped spans use
`trace_req_start`/`trace_req_finish`/`trace_slice` (`trace.py:380`, `415`, `545`).

**Request logging & crash dumps.** `RequestLogger` is reconfigurable at runtime
(`log_requests`, level, format) along with `dump_requests_folder`/`crash_dump_folder`
(`tokenizer_manager.py:1565-1576`); every finished request goes through
`request_logger.log_finished_request` and optionally an async `request_metrics_exporter` record
write (`tokenizer_manager.py:1329-1338`; exporter in
`srt/observability/request_metrics_exporter.py`). On SIGQUIT, `dump_requests_before_crash`
snapshots in-flight requests (`tokenizer_manager.py:2742`, dedup guard at :2246).

**Misc surfaces.** Per-process `setproctitle` names (`sglang::scheduler_TP0`,
`scheduler.py:3744`) and `faulthandler.enable()` (:3745); `--show-time-cost`
(`model_runner.py:440-441`); memory accounting exposed via `get_server_info` (weight/graph
memory usage, `scheduler.py:3260-3265`); KV-cache event publishing on idle for external
cache-aware routers (`_publish_kv_events`, `scheduler_runtime_checker_mixin.py:556`); the
dedicated `metrics_ipc_name` ZMQ channel in `PortArgs` for scheduler→tokenizer KV metrics
(`server_args.py:7126-7127`, `scheduler_metrics_mixin.py:686-687`); and `srt/observability/`
also houses `cpu_monitor.py`, `func_timer.py`, `startup_func_log_and_timer.py`,
`req_time_stats.py`.

## 14. Engineering details worth copying (and anti-patterns)

Part I §8 listed architecture-level borrowables for visual-rl; these are the lower-level
patterns Part II surfaced.

**Worth copying.**

1. **CPU shadow tensors for hot-path metadata.** `seq_lens_cpu` is maintained in lockstep with
   the GPU tensor from creation (`schedule_batch.py:1692-1702`) through CUDA-graph buffers
   (`cuda_graph_runner.py:244-250`), so `seq_lens_cpu.max().item()`
   (`flashattention_backend.py:333-353`) and graph-bucket bisect never trigger a D2H sync. Any
   GPU scheduler (including a denoise-step scheduler) needs this discipline.
2. **Pin + `non_blocking` at the producer, sync only at the consumer.** All H2D in
   `prepare_for_*` is pinned/async; the only sync on the output path is one CUDA event
   (`copy_done.synchronize()`, `scheduler_output_processor_mixin.py:397-413`). One event per
   batch, not per tensor.
3. **A sacrificial dummy slot for padded writes.** KV buffers are `size + page_size` rows and
   "padded slot 0 is used for writing dummy outputs from padded tokens"
   (`memory_pool.py:872-893`); allocators simply never hand out index 0
   (`allocator.py:131-138`). This deletes masking branches from every kernel write path.
4. **Largest-first CUDA-graph capture into one shared pool** ("Capture the large shapes first
   so that the smaller shapes can reuse the memory pool", `cuda_graph_runner.py:888-890`), plus
   `freeze_gc` during the capture window (`:417-432`), bucket-padding via `bisect` and slicing
   outputs back to `raw_num_token` (`:1228-1246,1313-1344`).
5. **Ownership by convention + lock counters instead of mutexes.** The allocator has no
   refcounts; sharing/pinning live in the radix tree's `lock_ref`, valid because everything
   runs on one scheduler thread (§11.1-11.2). The *only* real lock in the subsystem is the
   `threading.RLock` on the host pool — exactly where background DMA threads exist
   (`memory_pool_host.py:209-211`). Locks appear precisely at thread boundaries, nowhere else.
6. **`all_reduce(MIN)` as the cross-rank consensus primitive**: free-memory profiling
   (`utils/common.py:606-611`), PP token capacity
   (`model_runner_kv_cache_mixin.py:669-696`), and HiRadix write-ack counts so all TP ranks
   mutate the tree identically (`hiradix_cache.py:747-753`).
7. **Refusal-not-draining invalidation + assert-on-failure.** `flush_cache` runs only when
   fully idle and returns `False` otherwise (`scheduler.py:3229-3254`); weight updates `assert
   flush_cache_success` (`scheduler_update_weights_mixin.py:54-56`). For an RLHF/online-RL
   serving engine this turns "stale KV against new weights" from a silent corruption into a
   crash. Directly relevant to vrl's weight-sync path.
8. **Pop-once accessors for single-free invariants**: `pop_committed_kv_cache` asserts
   `"Committed KV cache already freed"` (`schedule_batch.py:931-935`) — double-free becomes an
   assert, not a heisenbug. Complemented by idle-time leak checks + tree `sanity_check()`
   (`scheduler_runtime_checker_mixin.py:468-562`).
9. **Free-group batching**: bracket a finish loop with `free_group_begin()/end()` to turn N
   tensor concats into one (`allocator.py:73-80`;
   `scheduler_output_processor_mixin.py:441,542`).
10. **The two-watchdog pair**: a counter-stall watchdog that dumps py-spy stacks before
    SIGQUIT (`watchdog.py:133-163`) plus a subprocess-liveness watchdog justified by "NCCL
    timeout causing C++ `std::terminate()`" never running Python handlers
    (`watchdog.py:166-175`). Any multi-process GPU service needs both.
11. **Decision tables and explicit models as comments**: the abort-handling matrix
    (`tokenizer_manager.py:2746-2757`) and the pool configurator's stated linear model
    "`available_bytes = max_tokens * coeff + bias`" (`pool_configurator.py:1-12`) make
    intent reviewable.
12. **Zero-ceremony plugin registries**: `EntryClass` + pkgutil scan with class-name-as-key and
    env-var external-package override (`registry.py:92-132`); attention backends via one
    decorator into a dict (`attention_registry.py:20-53`). No central list to forget to edit.
13. **Dataclass as the single source of CLI defaults** (`default=ServerArgs.tokenizer_path`,
    `server_args.py:4056-4086`) and generic `from_cli_args` over `dataclasses.fields`
    (`:6508-6517`) — defaults can't diverge between class and parser.

**Anti-patterns / warts (acknowledged in-tree or visible).**

- The Triton positions kernel does a naive per-program cumsum and the comment admits it "can be
  slow for large bs" (`forward_batch_info.py:1134-1187`).
- `alloc_extend` writes `out_indices` *before* the free-list shortfall check; correctness
  depends on the caller raising on `None` and abandoning the writes (`allocator.py:440-448`) —
  an implicit contract that would bite anyone reusing the kernel standalone.
- `ServerArgs` is a 7310-line flat file whose dataclass/`add_cli_args` ordering is kept in sync
  by a docstring plea ("please make sure the order ... the same", `server_args.py:292-316`) —
  manual mirroring at this scale is rot-prone.
- Magic constants without derivation: `max_req_input_len = max_req_len - 5`
  (`tp_worker.py:220-307`), the 2 s abort sleep (`tokenizer_manager.py:1584-1596`), the
  load-back threshold of 10 tokens (`hiradix_cache.py:181`).
- Model-registry import errors are swallowed with a warning unless `strict`
  (`registry.py:102-108`) — a broken model file silently disappears from the registry.
- The `vocab_mask` is dropped immediately after use to plug a documented overlap-mode VRAM leak
  (`model_runner.py:3061-3076`) — a workaround for an object-lifetime hazard inherent to the
  overlap design, worth remembering when building overlap schedulers.

## 15. Part II source-of-truth index

| Claim | Source |
|---|---|
| **Executor (§10)** | |
| H2D in prepare_for_extend: pinned + non_blocking, seq_lens_cpu shadow | `srt/managers/schedule_batch.py:1692-1702` ✓verified |
| KV alloc for extend; alloc_for_decode + in/out-of-place seq_lens bump under overlap | `srt/managers/schedule_batch.py:1718-1720,2294-2320` |
| filter_batch pinned index tensor | `srt/managers/schedule_batch.py:2398-2402` |
| ModelWorkerBatch = field projection (~40 fields) | `srt/managers/schedule_batch.py:2501-2582` |
| Triton positions kernel + extend_start_loc; per-program cumsum comment | `srt/model_executor/forward_batch_info.py:1134-1187` |
| ForwardBatch.init_new: passthrough, pool/backend binding, extend tensors, clamp_position, SWA precompute, LoRA fetch | `srt/model_executor/forward_batch_info.py:443-611` (pools `:478-480`, decode pos `:557-559`, extend `:561-580`, SWA `:594-600`, LoRA `:602-609`) |
| Shared input-buffer pool (as_strided reuse; NPU disabled) | `srt/model_executor/input_buffers.py:11-65` |
| ForwardMode.is_cuda_graph (DECODE/TARGET_VERIFY/IDLE/DLLM_EXTEND) | `srt/model_executor/forward_batch_info.py:166-172` ✓verified |
| num_tokens_per_bs per mode (verify/dLLM) | `srt/model_executor/cuda_graph_runner.py:620-632` |
| capture_bs generation + GPU-tier cuda_graph_max_bs defaults | `srt/server_args.py:1298-1361,1477-1508` ✓verified (list 1485-1492) |
| capture bs filtering (mul_base: TBO ×2, attn-TP, CP) | `srt/model_executor/cuda_graph_runner.py:495-529` |
| DecodeInputBuffers static tensors + CPU seq_lens_cpu | `srt/model_executor/cuda_graph_runner.py:132-272` |
| Largest-first capture comment, shared graph pool, freeze_gc, double warmup + barrier | `srt/model_executor/cuda_graph_runner.py:417-432,532-543,848-852,888-890,1160-1171` ✓verified (comment 888-890) |
| torch.compile mode `max-autotune-no-cudagraphs` inside capture | `srt/model_executor/cuda_graph_runner.py:446-476` |
| Replay gate in _forward_raw | `srt/model_executor/model_runner.py:2980-3003` |
| can_run skip conditions; runtime recapture on hidden-mode change | `srt/model_executor/cuda_graph_runner.py:742-811,1175-1205` |
| bisect to next bucket; foreach-grouped buffer fill; output slicing `[:raw_num_token]` | `srt/model_executor/cuda_graph_runner.py:111-129,274-383,1228-1246,1313-1344` |
| Piecewise CUDA graph for extend | `srt/model_executor/model_runner.py:2836-2845`; `srt/server_args.py:1530-1546` |
| AttentionBackend ABC contract | `srt/layers/attention/base_attn_backend.py:18-120` |
| Backend registry decorator; flashinfer MLA branch | `srt/layers/attention/attention_registry.py:20-53` |
| Default backend heuristics (fa3 Hopper, trtllm_mha SM100, ...; flashinfer regression note) | `srt/server_args.py:2428-2469` |
| Backend-forced page sizes (flashmla 64, cutlass_mla 128) | `srt/server_args.py:2517-2534` |
| Hybrid prefill/decode backend, pdmux groups, TBO wrapper | `srt/model_executor/model_runner.py:2086-2151` |
| FA decode metadata: page_table gather from req_to_token, cu_seqlens, CPU max | `srt/layers/attention/flashattention_backend.py:37-84,333-353` |
| FA graph metadata buffers + in-place replay update | `srt/layers/attention/flashattention_backend.py:1294-1332,1857-1935` |
| Logits pruning to last token per sequence | `srt/layers/logits_processor.py:427-476` |
| sample(): grammar mask, vocab_mask leak fix, positions arg | `srt/model_executor/model_runner.py:3061-3113` |
| Sampler: greedy argmax, in-place softmax, flashinfer kernels, RL paths | `srt/layers/sampler.py:83-265` |
| GenerationBatchResult + copy_to_cpu (non_blocking + event record) | `srt/managers/utils.py:25-100` |
| Worker return variants (delay_sample_func, prefill-only dummies) | `srt/managers/tp_worker.py:443-522` |
| Overlap: FutureMap store, copy_to_cpu on forward stream | `srt/managers/scheduler.py:2812-2841`; `srt/managers/overlap_utils.py:121-166` |
| Output processing: copy_done.synchronize → tolist | `srt/managers/scheduler_output_processor_mixin.py:392-413` |
| **Memory (§11)** | |
| Three-layer pool contract (docstring) | `srt/mem_cache/memory_pool.py:18-25` |
| `ReqToTokenPool` struct, slot reuse assert, free/clear | `srt/mem_cache/memory_pool.py:127-188` |
| FP8 stored as uint8 (`index_put` limitation) | `srt/mem_cache/memory_pool.py:685-689,996-998` |
| MHA K/V buffers per layer, dummy slot 0, `data_ptrs/strides` | `srt/mem_cache/memory_pool.py:872-916` ✓verified (872-893) |
| MLA single `kv_buffer` of width `kv_lora_rank+qk_rope_head_dim` | `srt/mem_cache/memory_pool.py:1537-1552` |
| `set_kv_buffer` write path + capture-mode alt stream | `srt/mem_cache/memory_pool.py:1022-1059,90-124` |
| Allocator state (`free_pages/release_pages/free_group`) | `srt/mem_cache/allocator.py:35-114` |
| page_size=1 alloc/free, index 0 reserved | `srt/mem_cache/allocator.py:131-165` |
| `need_sort` only for PD disaggregation; lazy merge+sort | `srt/model_executor/model_runner_kv_cache_mixin.py:572`; `srt/mem_cache/allocator.py:82-88` |
| 3-part `alloc_extend_kernel`, runtime-bounded loop comment, shortfall after writes | `srt/mem_cache/allocator.py:234-317,440-448` |
| Paged alloc/decode/free/clear (`unique(idx // page_size)`) | `srt/mem_cache/allocator.py:380-513` |
| Free-group bracketing in decode result processing | `srt/managers/scheduler_output_processor_mixin.py:441,542` |
| "Radix Cache takes one ref"; dedup free on insert | `srt/mem_cache/radix_cache.py:513-522` ✓verified |
| `RadixKey` (`child_key`, `match`, page align, bigram, extra_key namespace) | `srt/mem_cache/radix_cache.py:71-208,399-433` |
| `TreeNode` fields; `evicted`/`backuped`; heap ordering | `srt/mem_cache/radix_cache.py:211-271` |
| Page-hash chaining and split | `srt/mem_cache/radix_cache.py:274-318` |
| Lookup walk + in-place node split | `srt/mem_cache/radix_cache.py:693-739` |
| Insert walk, `evictable_size_` accounting | `srt/mem_cache/radix_cache.py:749-801` |
| Eviction heap over maintained `evictable_leaves`, parent cascade, `_update_leaf_status` | `srt/mem_cache/radix_cache.py:608-635,831-844` |
| Eviction strategies incl. MRU/FILO, exact priority tuples | `srt/mem_cache/evict_policy.py:16-65` |
| `inc/dec_lock_ref` bucket accounting, cross-tree assert; lock migration between chunks | `srt/mem_cache/radix_cache.py:637-671,586-587` |
| `cache_finished_req` 3-range free; `cache_unfinished_req` req_to_token rewrite; `cache_protected_len` rationale comment | `srt/mem_cache/radix_cache.py:488-599` (comment 580-584) |
| Paged extend over-reserves one page per request | `srt/mem_cache/common.py:255-294` (formula 265-268) |
| `write_req_to_token_pool_triton`, `get_last_loc_kernel` | `srt/mem_cache/common.py:27-198` |
| `alloc_for_extend` / `alloc_for_decode` wiring | `srt/mem_cache/common.py:328-462` |
| Triton backend caches req_to_token at init | `srt/layers/attention/triton_backend.py:129` |
| `Req` KV fields and pop-once asserts | `srt/managers/schedule_batch.py:637,647-648,745,928-947` |
| `release_kv_cache` teardown + overalloc assert | `srt/mem_cache/common.py:465-515` |
| Retraction frees without tree insert; aborts last req | `srt/managers/schedule_batch.py:2134-2187` |
| `flush_cache` resets tree+pools only when fully idle; deferred flush | `srt/managers/scheduler.py:3229-3254,3082-3100,2997-3021,3053-3070` ✓verified (3229+) |
| Tree reset emits `AllBlocksCleared` | `srt/mem_cache/radix_cache.py:385-396` |
| Weight-update flush + assert (all 4 paths); sleep-mode flush | `srt/managers/scheduler_update_weights_mixin.py:54-56,82-84,100-102,116-118,131-145` |
| `HostKVCache`: sizing, host>device assert, psutil check, RLock'd free list | `srt/mem_cache/memory_pool_host.py:155-288` |
| Host buffer layouts | `srt/mem_cache/memory_pool_host.py:351-375` |
| HiRadix init: host pool + controller + write-through threshold | `srt/mem_cache/hiradix_cache.py:66-188` |
| `write_backup` contiguous-prefix invariant + lock until DMA ack | `srt/mem_cache/hiradix_cache.py:656-689` |
| `writing_check` TP `all_reduce(MIN)` for identical tree updates | `srt/mem_cache/hiradix_cache.py:719-766` |
| Demotion vs deletion in `evict`; lock_ref recheck; `evict_host` | `srt/mem_cache/hiradix_cache.py:843-949` |
| Two-tier `match_prefix` (`host_hit_length`) | `srt/mem_cache/hiradix_cache.py:1221-1255` |
| `load_back` all-or-nothing promotion, threshold 10, lock protocol | `srt/mem_cache/hiradix_cache.py:951-1021,768-781,181` |
| Layer-wise transfer gating in `get_key_buffer` | `srt/mem_cache/memory_pool.py:1000-1006` |
| L3 interface + file backend; prefetch gating; host-only insert | `srt/mem_cache/hicache_storage.py:98-249,277`; `srt/mem_cache/hiradix_cache.py:1257-1305,1333-1340` |
| **Extension (§12)** | |
| Registry dataclass keyed by arch; `EntryClass` pkgutil scan; external package overwrite | `srt/models/registry.py:17-34,92-132` ✓verified (128-132) |
| Transformers fallback appended in `_normalize_archs` | `srt/models/registry.py:73-76` |
| `get_model_architecture` resolution + Mixtral hack | `srt/model_loader/utils.py:193-230` |
| Transformers backend arch synthesis (pooling/MM/MoE) | `srt/model_loader/utils.py:30-99,106-191` |
| Model constructor contract (`config`, `quant_config` kwargs) | `srt/model_loader/loader.py:261-281` |
| `DefaultModelLoader.load_model` two-phase + `load_weights_and_postprocess` | `srt/model_loader/loader.py:675-719` |
| Loader-consulted class attrs (`packed_modules_mapping`, `hf_to_sglang_mapper`, `post_load_weights`, layered) | `srt/model_loader/loader.py:197-199,253-256,753-760`; `srt/model_loader/utils.py:251-258` |
| Multithread safetensors weights iterator (8 threads) | `srt/model_loader/loader.py:481-545` |
| Example model: ctor / RadixAttention(layer_id) / forward / load_weights / EntryClass | `srt/models/exaone.py:297-303,153-160,320-333,335-374,377` ✓verified (377 lines) |
| ModelConfig capability detection from `hf_config.architectures` | `srt/configs/model_config.py:139-251,263-300` |
| ServerArgs dataclass + ordering convention; 7310 lines | `srt/server_args.py:292-316` ✓verified (`wc -l` = 7310) |
| `add_cli_args` defaults from class; `from_cli_args` generic; `--config` merge | `srt/server_args.py:4056-4113,6508-6517,7070-7104` |
| `__post_init__` normalization; `check_server_args` call site | `srt/server_args.py:763-840,6605`; `srt/entrypoints/engine.py:673` |
| Global server args accessor; PortArgs | `srt/server_args.py:7055-7067,7111-7134` |
| Typed env registry | `srt/environ.py:114-159,531` |
| **Startup & failure (§13)** | |
| CLI entry & dispatch | `python/sglang/launch_server.py:15-71` |
| `_set_envs_and_config`: SIGQUIT handler, prometheus dir, spawn method | `srt/entrypoints/engine.py:1121-1206` |
| `run_scheduler_process` + process config + exception→SIGQUIT | `srt/managers/scheduler.py:3707-3823` |
| Scheduler.__init__ ordered init phases | `srt/managers/scheduler.py:402-449` |
| TpModelWorker init & serving limits (`max_req_input_len = max_req_len - 5`) | `srt/managers/tp_worker.py:220-320` |
| ModelRunner init order (dist → load → pools → backend → graphs) | `srt/model_executor/model_runner.py:444-503,573-741` |
| get_available_gpu_memory: mem_get_info + MIN all-reduce | `srt/utils/common.py:498-613` |
| pre_model_load_memory snapshot + threading into initialize | `srt/model_executor/model_runner.py:458,485,1098-1123` |
| rest-memory formula (`post - pre*(1-frac)`) | `srt/model_executor/model_runner_kv_cache_mixin.py:56-70` ✓verified |
| mem_fraction_static heuristic docstring | `srt/server_args.py:1277-1296` |
| coeff+bias model; MHA/MLA cell_size; SWA two-pool solver; OOM message | `srt/model_executor/pool_configurator.py:1-12,113-181,184-273,39-44` |
| Token constraints (user cap, PP MIN all-reduce); max_running_requests derivation | `srt/model_executor/model_runner_kv_cache_mixin.py:669-717` |
| Pool construction (ReqToTokenPool, MHATokenToKVPool, allocator by page_size) | `srt/model_executor/model_runner_kv_cache_mixin.py:194-278,551-569,640-656,756-771` |
| load_model: sm80 dtype, memory-saver region, mem usage log, RoPE pre-expand, monitored_barrier | `srt/model_executor/model_runner.py:1170-1397` |
| Graph memory logging snapshot | `srt/model_executor/model_runner.py:2586-2618` |
| `get_init_info` handshake payload | `srt/managers/scheduler.py:1363-1376,3815` |
| Warmup thread + "fired up" log | `srt/entrypoints/http_server.py:287-400,2015-2031,2313-2359` |
| Streaming abort via background task; `/generate` endpoint | `srt/entrypoints/http_server.py:700-735`; `srt/managers/tokenizer_manager.py:1584-1596` |
| Disconnect polling (type 1 / type 3) | `srt/managers/tokenizer_manager.py:1267-1371` |
| Abort decision table comment | `srt/managers/tokenizer_manager.py:2746-2757` ✓verified |
| Scheduler abort methods 1/2/3; `to_finish=FINISH_ABORT()` | `srt/managers/scheduler.py:3344-3444` |
| `check_finished` consumes `to_finish` | `srt/managers/schedule_batch.py:1194-1201` |
| Tokenizer `_handle_abort_req` final output synthesis | `srt/managers/tokenizer_manager.py:2345-2384` |
| SIGQUIT/SIGTERM handlers (running phase, crash dump) | `srt/managers/tokenizer_manager.py:1609-1621,2724-2743` |
| WatchdogRaw hang detection; SubprocessWatchdog NCCL-crash rationale | `srt/utils/watchdog.py:103-192`; `srt/managers/scheduler_runtime_checker_mixin.py:563-585` |
| Idle leak checks + 30 s idle metrics | `srt/managers/scheduler_runtime_checker_mixin.py:468-562` |
| Prometheus multiproc mount; collectors; scheduler labels; decode interval | `srt/utils/common.py:1389-1399`; `srt/observability/metrics_collector.py:78,179-280,1146`; `srt/observability/scheduler_metrics_mixin.py:90-151,329,463-517` |
| Profiler endpoints + torch.profiler/RPD; batch-count arming | `srt/entrypoints/http_server.py:948,973`; `srt/managers/scheduler_profiler_mixin.py:38-200,339` |
| OTLP tracing init in both processes | `srt/entrypoints/engine.py:231-239`; `srt/managers/scheduler.py:3790-3798`; `srt/observability/trace.py:160,380-545` |
| Request logging/metrics exporter on finish | `srt/managers/tokenizer_manager.py:1329-1338,1565-1576` |
