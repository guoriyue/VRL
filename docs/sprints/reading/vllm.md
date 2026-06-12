# vllm — Architecture Reading

Repo: `/home/mingfeiguo/Desktop/vllm`, commit `9b17c5746` ("[ModelBash][DSR1 NVFp4] Removed Bf16 Bias Cast"). All paths relative to repo root. Scope: the V1 engine (`vllm/v1/`). Citations spot-checked against this checkout.

## 1. Repo layout & module organization

Top-level `vllm/` is a single Python package; the V1 engine lives entirely under `vllm/v1/` while cross-cutting infrastructure (distributed groups, config, platforms, entrypoints) stays outside it:

| Path | Owns |
|---|---|
| `vllm/entrypoints/`, `vllm/engine/` | OpenAI API server / legacy engine args (front-of-house) |
| `vllm/distributed/` | NCCL/process-group plumbing (`init_distributed_environment`, `GroupCoordinator`), shared-memory broadcast (`shm_broadcast.MessageQueue`), KV-transfer connectors |
| `vllm/config/` | `VllmConfig` and sub-configs threaded through every layer |
| `vllm/v1/engine/` | Front-end ↔ engine-core boundary: `async_llm.py` (AsyncLLM front-end), `llm_engine.py` (sync), `core.py` (`EngineCore`/`EngineCoreProc`), `core_client.py` (ZMQ clients), `coordinator.py` (DP coordinator process), `utils.py` (process/actor launchers, handshake metadata) |
| `vllm/v1/core/` | Scheduler-side state: `sched/scheduler.py` (continuous batching), `kv_cache_manager.py`, `kv_cache_coordinator.py`, `block_pool.py`, `single_type_kv_cache_manager.py`, `encoder_cache_manager.py` |
| `vllm/v1/executor/` | `abstract.py` (Executor ABC + backend selection), `multiproc_executor.py`, `ray_executor.py`, `uniproc_executor.py`, `ray_utils.py` |
| `vllm/v1/worker/` | Per-device workers: `worker_base.py` (`WorkerWrapperBase`), `gpu_worker.py`, `gpu_model_runner.py` |
| `vllm/v1/{attention,sample,spec_decode,structured_output,kv_offload,metrics}/` | Kernel backends, sampler, speculative decoding, grammar-constrained decoding, KV offload, stats |

The module split mirrors the process topology: `v1/engine` = front-end process + engine-core process + the IPC between them; `v1/core` = code that runs **inside** the engine-core process only; `v1/executor` + `v1/worker` = code that runs in (or drives) GPU worker processes.

**File-size norms**: core orchestrators are big and flat — `scheduler.py` 2178 lines, `engine/core.py` 1681, `core_client.py` 1458, `async_llm.py` 1099, `gpu_model_runner.py` 6278 (verified by `wc -l`). Leaf components stay small: `kv_cache_manager.py` 494, `block_pool.py` 490, `sched/interface.py` 214, `sched/utils.py` 64. The norm is roughly "one process role or one resource = one file, however long that gets".

## 2. Architecture overview

vLLM V1 is a pipeline of **three process tiers** connected by ZMQ (front-end ↔ engine core) and shared-memory message queues or Ray channels (engine core ↔ workers):

```
┌─ Front-end process (API server) ─────────────────────────────┐
│ AsyncLLM                                                     │
│  ├─ InputProcessor (tokenize)        OutputProcessor (detok) │
│  ├─ output_handler asyncio task  <───────────┐               │
│  └─ AsyncMPClient (EngineCoreClient)         │               │
│       input_socket ROUTER ──┐   output_socket PULL           │
└─────────────────────────────┼────────────────┼───────────────┘
              ZMQ (msgpack)   │ requests       │ EngineCoreOutputs
┌─ EngineCore process ────────▼────────────────┴───────────────┐
│ EngineCoreProc                                               │
│  ├─ input thread  (ZMQ DEALER → input_queue)                 │
│  ├─ output thread (output_queue → ZMQ PUSH)                  │
│  └─ run_busy_loop:  Scheduler.schedule()                     │
│        │   ├─ KVCacheManager ── KVCacheCoordinator           │
│        │   │        └─ BlockPool (prefix cache, free queue)  │
│        │   └─ chunked prefill / preemption / spec decode     │
│        ├─ Executor.execute_model(SchedulerOutput)            │
│        └─ Scheduler.update_from_output() → EngineCoreOutputs │
└────────────┼─────────────────────────────────────────────────┘
   MultiprocExecutor: shm MessageQueue broadcast / response MQs
   RayDistributedExecutor: Ray actors + compiled DAG (NCCL/shm)
┌─ Worker processes (one per GPU rank) ────────────────────────┐
│ WorkerProc → WorkerWrapperBase → gpu Worker → GPUModelRunner │
│   NCCL groups: TP × PP × (D/P)CP via torch.distributed       │
└──────────────────────────────────────────────────────────────┘
```

### Front-end: AsyncLLM

`AsyncLLM.__init__` wires tokenizer-side processors and spawns the engine via the client: "EngineCore (starts the engine in background process). `self.engine_core = EngineCoreClient.make_async_mp_client(...)`" (`vllm/v1/engine/async_llm.py:147-155`). A background asyncio task pulls outputs:

```python
async def output_handler():
    while True:
        # 1) Pull EngineCoreOutputs from the EngineCore.
        outputs = await engine_core.get_output_async()
        ...
        processed_outputs = output_processor.process_outputs(...)
```
(`vllm/v1/engine/async_llm.py:662-683`), chunked by `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` to avoid blocking the event loop (`async_llm.py:673-689`).

### Engine core

`EngineCore` owns executor + scheduler (`vllm/v1/engine/core.py:106, 139-146`). KV cache sizing happens at init by **profiling workers**: `available_gpu_memory = self.model_executor.determine_available_memory()` then `get_kv_cache_configs(...)` and `self.model_executor.initialize_from_config(kv_cache_configs)` (`core.py:226-283`).

`EngineCoreProc` wraps `EngineCore` for a background process ("ZMQ-wrapper for running EngineCore in background process", `core.py:684-685`). It runs **three threads**: an input-socket thread and an output-socket thread feeding `queue.Queue`s, explicitly to "overlap ZMQ socket IO with GPU since they release the GIL" (`core.py:748-775`), plus the main busy loop (section 3).

### Worker tier

Each worker process hosts `WorkerProc → WorkerWrapperBase → Worker → GPUModelRunner`. Each GPU worker calls `init_distributed_environment(world_size, rank, init_method, local_rank, backend="nccl")` then `ensure_model_parallel_initialized(tp, pp, pcp, dcp)` (`vllm/v1/worker/gpu_worker.py:1030-1056`) — collectives are vLLM's own NCCL groups via torch.distributed, regardless of which executor orchestrates the processes.

## 3. Core scheduling & orchestration

### 3.1 Engine-core event loop

The busy loop is three lines (`vllm/v1/engine/core.py:1018-1026`):

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while True:
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
```

- `_process_input_queue` blocks on `self.input_queue.get()` only while the scheduler is empty; otherwise it drains pending client messages non-blockingly (`core.py:1028-1054`). Messages are dispatched by type — `ADD` → `self.add_request`, `ABORT`, `UTILITY` (generic RPC, e.g. `reset_prefix_cache`), `EXECUTOR_FAILED` (`core.py:1076-1106`). Aborts are *also* pushed into a separate `aborts_queue` from the input IO thread so they can be applied eagerly while a forward pass is in flight (`core.py:1210-1218`, drained at `core.py:415-417`).
- `_process_engine_step` calls `self.step_fn()` and puts each `(client_index, EngineCoreOutputs)` on `output_queue` (`core.py:1056-1065`).

`step_fn` is chosen at construction (`core.py:206-208`):

```python
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
```

`batch_queue` exists only when `model_executor.max_concurrent_batches > 1` (`core.py:177-187`); the base executor returns 1 (`vllm/v1/executor/abstract.py:237-239`), while the Ray executor returns `pp_size`, or 2 with async scheduling when `pp_size <= 1` (`vllm/v1/executor/ray_executor.py:111-117`).

**Synchronous step** (`core.py:389-422`) — schedule → dispatch → sample → update:

```python
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
...
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

Note the split of one model step into `execute_model` (forward pass up to logits) and `sample_tokens`: the structured-output grammar bitmask is computed on the CPU *while the GPU runs the forward pass*, then applied at sampling time. The worker-side counterpart is the `ExecuteModelState` stash: on the last PP rank `execute_model` returns `None` after saving logits (`vllm/v1/worker/gpu_model_runner.py:3605-3618`), and `sample_tokens` unpacks it (`gpu_model_runner.py:3643-3657`).

**Pipelined step** `step_with_batch_queue` (`core.py:434-552`) keeps up to `batch_queue_size` in-flight batches: it first tries to schedule a *new* batch and returns immediately unless the queue is full or the oldest result is already done; only then does it block on `batch_queue.pop()` and run `update_from_output`. The docstring states the policy: "fulfilling the batch queue has a higher priority than getting model outputs" (`core.py:441-447`). This is what removes PP bubbles — the scheduler can issue batch N+1 before batch N's tokens are sampled (the two batches contain disjoint requests, since scheduled requests have `num_computed_tokens` advanced eagerly, see 3.4).

### 3.2 Scheduler data structures and policy

- `self.waiting = create_request_queue(self.policy)`; `self.running: list[Request]` (`vllm/v1/core/sched/scheduler.py:157-159`).
- Two policies (`vllm/v1/core/sched/request_queue.py:13-17`): `FCFS` → a `deque` subclass (`request_queue.py:75-128`), `PRIORITY` → a heap ordered by `(priority, arrival_time)` (`request_queue.py:131-198`; "requests with a smaller value of `priority` are processed first", `request_queue.py:135-139`).
- Budgets: `max_num_running_reqs = scheduler_config.max_num_seqs`, `max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens` (`scheduler.py:100-102`).

The core design point is stated in the comment at the top of `schedule()` (`scheduler.py:321-331`, verified verbatim):

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. ...
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec. This is general enough to cover
# chunked prefills, prefix caching, speculative decoding,
# and the "jump decoding" optimization in the future.
```

So a "decode" is just a request whose deficit is 1 token; a chunked prefill is a request whose deficit is large and gets clipped by the token budget.

### 3.3 The scheduling loop, line by line

`Scheduler.schedule()` (`scheduler.py:321-898`) is a single token-budgeted pass: **first RUNNING, then WAITING**.

**Phase 1 — RUNNING requests** (`scheduler.py:353-518`):

1. For each running request compute the deficit (`scheduler.py:373-377`):
   ```python
   num_new_tokens = (
       request.num_tokens_with_spec
       + request.num_output_placeholders
       - request.num_computed_tokens
   )
   ```
   (`num_output_placeholders` is the async-scheduling mechanism: tokens that were scheduled but whose values aren't known yet — see 3.6.)
2. Clip by `long_prefill_token_threshold`, the token budget, and `max_model_len - 1` (`scheduler.py:378-386`).
3. `num_new_tokens == 0` → `continue` rather than `break`, explicitly trading strict FCFS for utilization: "by doing `continue` instead of `break`, we do not strictly follow the FCFS scheduling policy and allow the lower-priority requests to be scheduled" (`scheduler.py:423-427`).
4. **Allocate-or-preempt loop** (`scheduler.py:430-479`, verified): call `kv_cache_manager.allocate_slots(...)`; if it returns `None` (not enough free blocks), preempt a victim and retry:
   ```python
   if self.policy == SchedulingPolicy.PRIORITY:
       preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
       self.running.remove(preempted_req)
       ...
   else:
       preempted_req = self.running.pop()
   self._preempt_request(preempted_req, scheduled_timestamp)
   ...
   if preempted_req == request:
       # No more request to preempt. Cannot schedule this request.
       break
   ```
   FCFS preempts the *newest* running request (`self.running.pop()`); priority mode preempts the worst `(priority, arrival_time)` — which may even be a request already scheduled in this very step, in which case its token budget, blocks, and encoder budget are refunded (`scheduler.py:450-467`). `_preempt_request` (`scheduler.py:900-920`, verified) frees **all** of the victim's KV blocks, resets `num_computed_tokens = 0`, sets status `PREEMPTED`, and prepends it to the waiting queue — i.e. V1 uses **recompute-style preemption** (no swap-to-CPU; the prefix cache may still soften the recompute cost).
5. On success, record the request and decrement the budget (`scheduler.py:482-487`); slice scheduled speculative tokens to fit the budget (`scheduler.py:489-505`).

**Phase 2 — WAITING requests** (`scheduler.py:530-804`), only entered `if not preempted_reqs:` (`scheduler.py:535`) — if anyone was preempted this step, memory is tight and admitting new work is pointless:

1. Loop `while self.waiting and token_budget > 0`, capped by `len(self.running) == self.max_num_running_reqs` (`scheduler.py:536-538`).
2. `peek_request()` then a gauntlet of *skip* checks, each of which pops the request into a temporary `skipped_waiting_requests` queue that is prepended back afterwards (`scheduler.py:530-532`, `802-804`): waiting for remote KVs (P/D disaggregation, `scheduler.py:544-560`), waiting for structured-output FSM compilation (`scheduler.py:564-571`), waiting for streaming input (`scheduler.py:574-578`), or would exceed `max_loras` (`scheduler.py:582-593`).
3. **Prefix-cache lookup** for fresh requests (`scheduler.py:600-633`): `new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)`, plus an optional external KV-connector match.
4. Deficit = `request.num_tokens - num_computed_tokens` (uses `num_tokens` not `num_prompt_tokens` "to consider the resumed requests, which have output tokens", `scheduler.py:650-654`). **Chunked prefill** is just `num_new_tokens = min(num_new_tokens, token_budget)` (`scheduler.py:669`); if chunked prefill is disabled and the prompt doesn't fit the remaining budget, scheduling stops (`scheduler.py:661-667`).
5. `allocate_slots(...)` with the prefix-cache hit blocks (`scheduler.py:715-724`). `None` → `break` (no preemption is triggered on behalf of waiting requests).
6. Admission (`scheduler.py:757-788`): pop from waiting, append to `running`, classify into `scheduled_new_reqs` (status WAITING) vs `scheduled_resumed_reqs` (status PREEMPTED), set `status = RUNNING`, `num_computed_tokens` = the cached amount.

**Epilogue** (`scheduler.py:806-898`): assert invariants (`total_num_scheduled_tokens <= max_num_scheduled_tokens`, `scheduler.py:807-811`); compute `num_common_prefix_blocks` for cascade attention (`scheduler.py:819-827`); build the `SchedulerOutput` — full `NewRequestData` for new requests vs delta-only `CachedRequestData` (new block IDs, counts) for already-known requests (`scheduler.py:829-877`, `_make_cached_request_data` at `scheduler.py:999-1057`); attach KV-connector metadata (`scheduler.py:883-887`).

### 3.4 Eager state advance + closing the loop

`_update_after_schedule` (`scheduler.py:922-954`) advances `request.num_computed_tokens += num_scheduled_token` **immediately**, before the model runs — "allowing us to schedule the prefill request again immediately in the next scheduling step" (`scheduler.py:927-929`); rejected spec tokens are rolled back later.

`Scheduler.update_from_output(scheduler_output, model_runner_output)` (`scheduler.py:1246-1499`) runs once per step after sampling:

- Per request: fetch sampled tokens via `req_index = model_runner_output.req_id_to_index[req_id]` (`scheduler.py:1304-1307`); for spec decode, roll back `num_computed_tokens -= num_rejected` (`scheduler.py:1312-1326`).
- Append tokens and check stop conditions: `_update_request_with_output` (`scheduler.py:1543-1559`) calls `check_stop` per token — EOS, stop-token-ids, `max_tokens`, `max_model_len` (`vllm/v1/core/sched/utils.py:40-64`) — and trims tokens after the stop. Finished requests get `kv_transfer_params = self._free_request(request)` (`scheduler.py:1362`), which frees blocks and records the ID in `finished_req_ids` (`scheduler.py:1723-1744`) so the *next* `SchedulerOutput` tells workers to drop their cached state (`scheduler.py:871-875`).
- Emit one `EngineCoreOutput` per request with new tokens, bucketed per front-end client: `outputs[request.client_index].append(EngineCoreOutput(...))` (`scheduler.py:1400-1418`), returned as `dict[client_index, EngineCoreOutputs]` (`scheduler.py:1468-1471`).
- Stopped requests are batch-removed from `running` (`scheduler.py:1424-1428`).

### 3.5 KV-cache management as it interacts with scheduling

Layering:

```
Scheduler ──> KVCacheManager (vllm/v1/core/kv_cache_manager.py:94)
                └─> KVCacheCoordinator (Unitary / Hybrid / NoPrefixCache)
                      (vllm/v1/core/kv_cache_coordinator.py:28,256,302,368)
                      └─> per-group SingleTypeKVCacheManager (full attn / sliding window / mamba ...)
                            (vllm/v1/core/single_type_kv_cache_manager.py)
                            └─> BlockPool (vllm/v1/core/block_pool.py:128)
```

`KVCacheBlocks` is explicitly "the interface between Scheduler and KVCacheManager, to hide KVCacheManager's internal data structure from the Scheduler" (`kv_cache_manager.py:21-27`, verified). The scheduler only ever sees opaque block lists/IDs.

**`allocate_slots`** (`kv_cache_manager.py:206-376`) is the single entry point the scheduler calls for both running and waiting requests. Its docstring contains the token-layout diagram (`kv_cache_manager.py:238-259`):

```
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
```

Three stages (`kv_cache_manager.py:276-283`): (1) free skipped blocks (e.g. outside a sliding window) *before* the capacity check "to reduce the number of evicted blocks" (`kv_cache_manager.py:316-324`); (2) capacity check — **this is the scheduler's admission/preemption signal** (verified):

```python
if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
    # Cannot allocate new blocks
    return None
```
(`kv_cache_manager.py:336-338`) — `None` is what makes `schedule()` preempt (running) or stop admitting (waiting); (3) pin prefix-hit blocks (`allocate_new_computed_blocks`, which `touch()`es them so they leave the free queue, `single_type_kv_cache_manager.py:190`) then allocate fresh blocks (`coordinator.allocate_new_blocks` → `block_pool.get_new_blocks`, `kv_cache_coordinator.py:143-176`, `single_type_kv_cache_manager.py:235`). The capacity math even counts prefix-hit blocks that are currently eviction candidates (`num_evictable_blocks`, `single_type_kv_cache_manager.py:133-139`).

Newly-full blocks are committed to the prefix cache, capped at `request.num_tokens` so unverified draft tokens are never cached: "we cap the number at `request.num_tokens`, ensuring only 'finalized' tokens are cached" (`kv_cache_manager.py:365-374`).

**Prefix caching and eviction:**

- **Lookup**: `get_computed_blocks` walks the request's pre-computed `block_hashes` chain via `coordinator.find_longest_cache_hit` (`kv_cache_manager.py:164-204`); full-attention groups walk hash-by-hash and stop at the first miss ("If a block hash is not in the cached_block_hash_to_id, the following block hashes are not computed yet for sure", `single_type_kv_cache_manager.py:435-445`). A full hit is intentionally truncated to `num_tokens - 1` so the last token is recomputed for logits (`kv_cache_manager.py:183-189`).
- **Ref-counting**: `touch` increments `ref_cnt` and removes ref_cnt==0 blocks from the free queue (`block_pool.py:372-387`); `free_blocks` decrements and re-appends ref_cnt==0 blocks (`block_pool.py:389-403`).
- **Eviction is lazy LRU**: freed cached blocks stay in `cached_block_hash_to_block` *and* in the `FreeKVCacheBlockQueue` — a hand-rolled doubly-linked list ("The least recent used block is at the front (LRU). If two blocks have the same last accessed time ... the one with more hash tokens (the tail of a block chain) is at the front", `vllm/v1/core/kv_cache_utils.py:157-173`). Tail-first ordering is produced by freeing a request's blocks in reverse: `ordered_blocks = reversed(req_blocks)` (`single_type_kv_cache_manager.py:265-279`). Actual eviction happens only when a block is reallocated: `get_new_blocks` → `_maybe_evict_cached_block`, which removes the hash mapping and emits a `BlockRemoved` event (`block_pool.py:300-330`, `332-370`).
- **Block-hash map**: `BlockHashToBlockMap` allows duplicate blocks per hash and never de-duplicates, "to make sure the allocated block IDs won't change so that block tables are append-only" (`block_pool.py:46-50`). Block 0 is reserved as a null placeholder block (`block_pool.py:160-175`).
- **RLHF hook**: `reset_prefix_cache` exists explicitly "to invalidate prefix caching after the weights are updated" and only succeeds when every block is free (`block_pool.py:424-457`); it is exposed end-to-end as an engine utility RPC (`AsyncLLM.reset_prefix_cache` → `call_utility_async("reset_prefix_cache", ...)`, `core_client.py:995-1000`).

Scheduling × KV interactions in one table:

| Scheduler event | KV-cache effect |
|---|---|
| Admit waiting request | `get_computed_blocks` (prefix hit) → `allocate_slots` (touch + allocate) — `scheduler.py:600-604`, `715-724` |
| Running request grows | `allocate_slots` per step; lookahead slots for spec decode — `scheduler.py:432-436` |
| `allocate_slots` returns `None` | Preempt victim → `kv_cache_manager.free(request)` + `num_computed_tokens = 0` — `scheduler.py:438-479`, `909-912` |
| Request finishes | `_free_request` → `_free_blocks` → reverse-order free (tail evicted first) — `scheduler.py:1723-1744`, `single_type_kv_cache_manager.py:275-279` |
| Async sched decode step | `cache_blocks` called incrementally as placeholders resolve — `async_scheduler.py:56-59` |

### 3.6 Async scheduling

`AsyncScheduler` (`vllm/v1/core/sched/async_scheduler.py:12-60`) lets step N+1 be scheduled before step N's tokens are known: after scheduling, each decode gets `request.num_output_placeholders += 1 + cur_num_spec_tokens` (`async_scheduler.py:32`), which Phase 1 of `schedule()` adds to the deficit (`scheduler.py:373-377`); when real tokens arrive, `_update_request_with_output` decrements placeholders (`async_scheduler.py:52`). On the worker side, `execute_model` returns `None` early and an `async_output_busy_loop` thread enqueues `AsyncModelRunnerOutput`s (`multiproc_executor.py:829-843`).

## 4. Distributed orchestration (Ray or alternative)

### 4.1 Two orthogonal axes, two backend choices

vLLM V1 splits orchestration into **(1) engine-core launching** (one engine process per DP rank) and **(2) intra-engine worker execution** (TP×PP×CP ranks under one engine). Each axis independently supports plain multiprocessing or Ray. Ray is strictly optional; the default path is multiprocessing + ZMQ + shared-memory MQ + NCCL.

**Executor backend selection** is config-driven in `Executor.get_class`: `"ray"` → `RayDistributedExecutor`, `"mp"` → `MultiprocExecutor`, `"uni"` → `UniProcExecutor`, `"external_launcher"` (e.g. torchrun-style SPMD) → `ExecutorWithExternalLauncher`, or any qualname (`vllm/v1/executor/abstract.py:46-86`). The base class is one RPC surface: `execute_model` / `sample_tokens` / everything else funnels through `collective_rpc` (`abstract.py:133-228`), which `EngineCore.collective_rpc` simply forwards (`core.py:650-657`) — this is also the path RLHF-style weight updates / `sleep`/`wake_up` use (utility calls travel front-end → ZMQ UTILITY message → `_handle_client_request` dispatch by method name, `core.py:1086-1100`).

### 4.2 Default path: multiprocessing + ZMQ + shared-memory MQ + NCCL

**Front-end ↔ EngineCore (ZMQ + msgpack).** `MPClient` "pushes EngineCoreRequests via input_socket, pulls EngineCoreOutputs via output_socket" (`vllm/v1/engine/core_client.py:442-453`); it binds a `zmq.ROUTER` input socket and a `zmq.PULL` output socket (`core_client.py:506-512`), addresses engines by 2-byte ZMQ identity = DP rank (`core_client.py:530-532`), and serializes with `MsgpackEncoder/Decoder` (`core_client.py:465-466`) using zero-copy multipart frames with `MessageTracker`-guarded tensor buffers (`core_client.py:925-948`). On the engine side, the input thread connects `zmq.DEALER` sockets (identity created at `core.py:708`) and the output thread `zmq.PUSH` sockets (`core.py:1152-1160, 1240-1245`). The client side runs an asyncio `process_outputs_socket` task feeding `outputs_queue` (`core_client.py:873-900`).

**Engine process launch + handshake.** `launch_core_engines` (`vllm/v1/engine/utils.py:777-934`) spawns one `EngineCore_DP{i}` process per local DP rank via `CoreEngineProcManager` (`context.Process(target=EngineCoreProc.run_engine_core, name=f"EngineCore_DP{global_index}", ...)`, `utils.py:119-128`). Startup is a ZMQ DEALER↔ROUTER handshake: engine sends `HELLO`, front-end replies with `EngineHandshakeMetadata` ("addresses of the front-end ZMQ queues that they should connect to" + parallel-config overrides, `utils.py:69-77`; `core.py:891-927`), engine sends `READY` with `num_gpu_blocks` (`core.py:869-889`). Liveness is monitored by daemon threads waiting on process sentinels on both sides (`core_client.py:589-627`; `multiproc_executor.py:232-262`).

**EngineCore ↔ Workers (MultiprocExecutor).** Not ZMQ: control-plane RPC uses a **shared-memory broadcast MessageQueue** — the executor creates `self.rpc_broadcast_mq = MessageQueue(self.world_size, self.local_world_size, ...)` and passes its handle to every worker (`vllm/v1/executor/multiproc_executor.py:127-137`); each `WorkerProc` (forked via `get_mp_context().Process`, named `VllmWorker-{rank}`, `multiproc_executor.py:596-626`) attaches to it plus a 1→1 response MQ (`multiproc_executor.py:499-529`). The worker busy loop is a pure RPC dispatcher (verified):

```python
method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(...)
func = getattr(self.worker, method)   # or cloudpickle.loads(method)
output = func(*args, **kwargs)
...
if output_rank is None or self.rank == output_rank:
    self.handle_output(output)
```
(`multiproc_executor.py:845-871`). `execute_model` broadcasts to all ranks but collects only from `output_rank` (`multiproc_executor.py:270-301`); non-blocking calls return a `FutureWrapper` that drains an ordered futures queue (`multiproc_executor.py:64-90, 365-375`). Multi-node-per-DP-rank is supported without Ray: follower nodes have no broadcast MQ and remote response MQs are bridged through the stateless inner-DP group (`multiproc_executor.py:511-529, 172-185`).

**Data plane / collectives = NCCL via torch.distributed**, not the orchestrator: each GPU worker calls `init_distributed_environment(...)` then `ensure_model_parallel_initialized(tp, pp, pcp, dcp)` (`vllm/v1/worker/gpu_worker.py:1030-1056`); the rendezvous address is a loopback TCP `distributed_init_method` minted by the executor (`multiproc_executor.py:120-122`).

### 4.3 Ray path

Ray appears in two places:

1. **`RayDistributedExecutor`** (`distributed_executor_backend="ray"`): workers are `RayWorkerWrapper` **actors** pinned to placement-group bundles — `PlacementGroupSchedulingStrategy(placement_group=..., placement_group_bundle_index=bundle_id)` then `ray.remote(num_cpus=0, num_gpus=num_gpus, scheduling_strategy=...)(RayWorkerWrapper).remote(rpc_rank=rank)` (`vllm/v1/executor/ray_executor.py:211-233`, verified). Ranks are re-sorted so driver-node workers come first and same-node workers are adjacent (`ray_executor.py:254-279`). Steady-state execution uses **Ray Compiled Graphs**: a DAG where "All workers in the first TP group take in the ExecuteModelRequest... Each PP worker takes in the output of the previous PP worker, and the TP group executes in SPMD fashion", with inter-stage tensors carried over NCCL or shm channels (`with_tensor_transport(transport=...)`), compiled once via `forward_dag.experimental_compile(...)` (`ray_executor.py:578-638`) and executed per step by `self.forward_dag.execute((scheduler_output, grammar_output))` (`ray_executor.py:457-467`); `execute_model` merely stashes the scheduler output and defers actual execution to `sample_tokens` so both are submitted as one DAG invocation (`ray_executor.py:413-455`). Control RPCs still go through per-actor `worker.execute_method.remote(...)` + `ray.get` (`ray_executor.py:488-513`). PP concurrency is exposed as `max_concurrent_batches = pp_size` (or 2 with async scheduling when pp_size <= 1) (`ray_executor.py:111-117`), which feeds `EngineCore.batch_queue`.
2. **`CoreEngineActorManager`** (`data_parallel_backend="ray"`): the engine-core processes themselves become Ray actors (`EngineCoreActor` / `DPMoEEngineCoreActor`), one per DP rank, each scheduled into its own DP placement group at `placement_group_bundle_index=world_size`, with env vars copied via `RuntimeEnv` (`vllm/v1/engine/utils.py:227-357`; selected in `launch_core_engines` at `utils.py:847-858`). Otherwise DP engine cores are plain `multiprocessing` processes (`utils.py:905-918`).

Even under Ray, the front-end↔engine IPC remains ZMQ (the actors receive the same `EngineZmqAddresses`, `utils.py:342`), and intra-worker collectives remain vLLM's own NCCL groups (optionally wrapping PP comm for the compiled DAG via `RayPPCommunicator`, `ray_executor.py:613-629`).

### 4.4 Data-parallel coordination

For DP>1 online serving, a separate **`DPCoordinator` process** "intermediates between multiple DP engine rank processes and one or more front-end API server processes": it collects waiting/running queue lengths for load balancing, tracks the global "request wave" number (paused↔running transitions synchronized by an all-reduce in `DPEngineCoreProc._has_global_unfinished_reqs`), and broadcasts `START_DP_WAVE` (`vllm/v1/engine/coordinator.py:22-56`); it is itself a daemon `multiprocessing.Process` (`coordinator.py:77-90`), launched only by DP rank 0 in online mode (`utils.py:822-845`). Engines subscribe to it via XSUB/PUSH sockets in their IO threads (`core.py:1161-1188, 1246-1254`). Client-side routing: `DPAsyncMPClient` (external LB, client per rank) vs `DPLBAsyncMPClient` (internal LB, one client balancing across ranks) (`core_client.py:99-124`). Non-MoE DP ranks are deliberately decoupled — each is "completely independent, so treat like DP=1" (`core.py:990-996`); only MoE models get the wave-coordinated `DPEngineCoreProc` (`core.py:985-988`).

## 5. Code organization style: function granularity

vLLM V1 is conspicuously **not** decomposed into small functions. The rule that emerges from reading the scheduler and engine core: **a function is as long as one pass of its algorithm; helpers are extracted only when there is a second caller, a subclass override point, or a reusable pure predicate** — never for cosmetic length reduction.

### Deliberately monolithic: `Scheduler.schedule()` — ~579 lines

`vllm/v1/core/sched/scheduler.py:321-899`. One method covers the entire per-iteration decision: running-queue token assignment, chunked prefill capping, encoder budget, KV slot allocation with inline preemption loop, then the waiting queue, then `SchedulerOutput` assembly. Because prefill/decode are unified into "catch num_computed_tokens up to num_tokens_with_spec" (`scheduler.py:321-331`), there is no natural seam to split on — chunked prefill is literally a budget `min()` (`scheduler.py:669`). Even the preempt-until-allocatable loop is inlined as a `while True:` around `kv_cache_manager.allocate_slots(...)` (`scheduler.py:430-450`). Mutated local state (`token_budget`, `req_index`, four scheduled-request lists, `scheduler.py:333-345`) is shared across the whole pass; splitting it would mean threading ~10 accumulators through helper signatures.

### Second monolith: `Scheduler.update_from_output()` — ~254 lines

`scheduler.py:1246-1499`. The post-step state update is one flat loop, and the comment says why it isn't decomposed — **per-request call overhead matters at this frequency**:

```python
# NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
# the below loop can be a performance bottleneck. We should do our best
# to avoid expensive operations inside the loop.
```
(`scheduler.py:1283-1285`). Similarly `EngineCoreProc.run_busy_loop()` is 8 lines of two numbered calls (`core.py:1018-1026`) while the work lives in `_process_input_queue` / `_process_engine_step` — split there because `DPEngineCoreProc` overrides `run_busy_loop` (`core.py:1409`) but reuses the two halves.

### Helpers they DID extract, and the justification for each

1. **Multiple call sites**: `_preempt_request()` (`scheduler.py:900-921`) — called from the in-schedule preemption loop (`scheduler.py:471`) and from `reset_prefix_cache`'s force-preempt path (`scheduler.py:1773`). Its docstring even documents the contract it does *not* own: `"NOTE: The request should be popped from the running queue outside of this method."` (`scheduler.py:903-904`, verified).
2. **Subclass override points**: `_update_after_schedule()` (`scheduler.py:922-954`) and `_update_request_with_output()` (`scheduler.py:1543`) each have exactly one caller — they exist so `AsyncScheduler` (the whole file is 61 lines, `vllm/v1/core/sched/async_scheduler.py:12-60`) can implement async scheduling by overriding just these two hooks (`async_scheduler.py:18`, `:37`) instead of copying the 579-line `schedule()`. Single-caller helpers are fine **iff the caller graph includes a subclass**.
3. **Pure reusable predicates in a separate `utils.py`**: `check_stop(request, max_model_len)` and `remove_all(lst, items)` (`vllm/v1/core/sched/utils.py:8-64`). `remove_all` carries a perf-justified docstring ("optimizes for the common case of removing a single item", `utils.py:9-26`) — even a 10-line helper gets a why-comment.

By contrast `EngineCore.step()` is short (~34 lines, `core.py:389-422`) because it is pure orchestration — each line a call into an owning component.

### Comment style — three signatures

1. **Attributed notes**: `NOTE(woosuk)`, `NOTE(Kuntai)`, `NOTE(zhuohan)`, `TODO(rob)` — the author signs design rationale (`scheduler.py:322,879,1298,1774`; `async_llm.py:700`). These explain *why a non-obvious decision holds*, e.g. why `continue` instead of `break` deliberately relaxes FCFS (`scheduler.py:423-425`).
2. **Numbered step comments inside long loops**: `# 1) Pull EngineCoreOutputs... # 2) Process...` in AsyncLLM's `output_handler` (`async_llm.py:665-699`), `# 1) Poll the input queue... # 2) Step the engine core` (`core.py:1023-1025`, verified).
3. **ASCII diagrams for memory layout math**: `allocate_slots()` carries a block-layout diagram plus an abbreviation legend directly in the docstring (`kv_cache_manager.py:238-270`).

Also notable: closures over `self`-fields to break GC cycles in long-lived asyncio tasks — `output_handler()` deliberately captures `engine_core`, `output_processor` etc. as locals: "Ensure that the task doesn't have a circular ref back to the AsyncLLM object" (`async_llm.py:653-660`); and `BackgroundResources` dataclass + `weakref.finalize` so ZMQ sockets close even on mid-`__init__` exceptions (`core_client.py:472-476`, `:356-379`).

### Type-system choices

**Serialization-tiered data types** — the most distinctive structural choice:
- **Cross-process (ZMQ) types are `msgspec.Struct` with `array_like=True, omit_defaults=True, gc=False`**: `EngineCoreRequest` (`vllm/v1/engine/__init__.py:55-60`, verified), `EngineCoreOutputs` (`:176-181`). `gc=False` because these are high-churn message objects.
- **In-process step boundary types are plain `@dataclass`**: `SchedulerOutput` / `NewRequestData` / `CachedRequestData` (`vllm/v1/core/sched/output.py:182-253`), documented with per-field `#` comments. `CachedRequestData` is diff-only: "we only send the diff to minimize the communication cost" (`output.py:185-192`).
- **Enums chosen per wire format**: `FinishReason(enum.IntEnum)` — "Int rather than Str for more compact serialization" (`engine/__init__.py:32-36`); `EngineCoreRequestType(enum.Enum)` with **hex-byte values** `ADD = b"\x00"` "so it can be sent over sockets without separate encoding step" (`:207-218`, verified); `RequestStatus(IntEnum)` exploits ordering: `is_finished = status > RequestStatus.PREEMPTED` (`vllm/v1/request.py:296-318`), with the status→reason mapping as a module-level dict `_FINISHED_REASON_MAP` (`request.py:329-336`) instead of an if-chain.

**ABC everywhere, `typing.Protocol` nowhere** in the engine core. `SchedulerInterface(ABC)` (`sched/interface.py:21`), `Executor(ABC)` with `get_class()` static factory (`executor/abstract.py:36-60`), `EngineCoreClient(ABC)` with `make_client` factory (`core_client.py:63-75`), `RequestQueue(ABC)` → `FCFSRequestQueue(deque[Request], RequestQueue)` / `PriorityRequestQueue` (`request_queue.py:20,75,131`). A grep for `Protocol` in `vllm/v1/{core,engine,executor}` returns nothing — they prefer nominal inheritance plus factory methods over structural typing.

**Facade pattern with explicit hiding rationale.** `KVCacheManager` is a thin delegating facade over `coordinator` + `block_pool` (`kv_cache_manager.py:119-131`; e.g. `free`, `remove_skipped_blocks`, `take_events` are one-line delegations, `:378-399, 459-465`). Note the GC-aware micro-optimizations on the boundary type: a pre-constructed immutable `empty_kv_cache_blocks` shared instance (`kv_cache_manager.py:134-141`) and `create_kv_cache_blocks` returning it for empty allocations (`:486-490`).

## 6. Naming conventions

**Module file = snake_case of the file's main class; one concept per file.** `kv_cache_manager.py` → `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py:94`), `block_pool.py` → `BlockPool` (`vllm/v1/core/block_pool.py:128`), `kv_cache_coordinator.py` → `KVCacheCoordinator` (`vllm/v1/core/kv_cache_coordinator.py:28`). Exception: device variants put the device in the *filename*, not the class — `vllm/v1/worker/gpu_worker.py:71` is `class Worker(WorkerBase)`, and `gpu_model_runner.py:329` is `GPUModelRunner`.

**Role-suffix taxonomy** (each suffix means a specific responsibility):
- `*Manager` = owns the lifecycle of one resource type: `KVCacheManager`, `EncoderCacheManager`, `SingleTypeKVCacheManager` with subclasses `FullAttentionManager` / `SlidingWindowManager` / `MambaManager` / `CrossAttentionManager` (`vllm/v1/core/single_type_kv_cache_manager.py:408,469,752,1010`).
- `*Coordinator` = multiplexes several managers: `UnitaryKVCacheCoordinator` / `HybridKVCacheCoordinator` (`kv_cache_coordinator.py:302,368`).
- `*Pool` = flat free-list resource: `BlockPool` (`block_pool.py:128`).
- `Executor` variants encode topology in the prefix: `UniProcExecutor`, `ExecutorWithExternalLauncher` (`vllm/v1/executor/uniproc_executor.py:26,140`), `MultiprocExecutor` (`multiproc_executor.py:93`), `RayDistributedExecutor` (`ray_executor.py:62`).
- Process-boundary wrappers add `Proc` / `ProcHandle`: `EngineCore` (in-process, `core.py:79`) vs `EngineCoreProc(EngineCore)` (ZMQ wrapper, `core.py:684`) vs `DPEngineCoreProc` (`core.py:1315`); `WorkerProc` / `WorkerProcHandle` / `UnreadyWorkerProcHandle` (`multiproc_executor.py:492,465,455`).
- Client side mirrors with `*Client`: `EngineCoreClient` → `MPClient` → `SyncMPClient` / `AsyncMPClient` (`core_client.py:63,442,652,822`).

**Function-verb vocabulary** (consistent semantics across the codebase):
- `get_*` cheap accessor; `make_*` constructs fresh (`make_stats` `scheduler.py:1821`, `make_empty` `output.py:242`, `make_client` `core_client.py:75`); `take_*` = drain-and-clear ownership transfer (`take_events` `block_pool.py:480`, `take_draft_token_ids` `executor/abstract.py:233`).
- `_try_*` = may fail/partially succeed (`_try_schedule_encoder_inputs` `scheduler.py:1059`); `_maybe_*` = conditional no-op (`_maybe_evict_cached_block` `block_pool.py:332`, `_maybe_publish_request_counts` `core.py:1396`).
- `update_from_*` = state sync from another component's output (`update_from_output` `scheduler.py:1246`).
- `*_busy_loop` for blocking process loops (`run_busy_loop` `core.py:1018`, `worker_busy_loop` / `async_output_busy_loop` `multiproc_executor.py:845,839`); `process_*_socket(s)` for IO threads (`core.py:1139,1220`).
- Async twins get an explicit `_async` suffix instead of overloading: `add_request_async`, `get_output_async`, `reset_prefix_cache_async` (`core_client.py:204-251`).
- `touch(blocks)` for ref-count increment (`block_pool.py:372`) — terse Unix-style verb for a hot-path op.

**Abbreviations are standardized, not ad hoc**: `req`/`reqs`, `mm` (multimodal), `spec` (speculative), `kv`, `dp`/`pp`/`tp`, and the directory `vllm/v1/core/sched/`. They appear in public field names (`scheduled_new_reqs`, `num_scheduled_tokens`, `scheduled_spec_decode_tokens`, `output.py:188-203`) — once an abbreviation is adopted it is used everywhere.

## 7. End-to-end flow trace

One completion request, OpenAI API → GPU forward → token back to the client:

1. **HTTP entry**: `POST /v1/completions` → `create_completion` (`vllm/entrypoints/openai/completion/api_router.py:46`) → `OpenAIServingCompletion.create_completion` → `generator = self.engine_client.generate(...)` (`vllm/entrypoints/openai/completion/serving.py:226`).
2. **AsyncLLM.generate** → `await self.add_request(...)` (`vllm/v1/engine/async_llm.py:537-581`).
3. **Tokenize/process**: `self.input_processor.process_inputs(...)` builds an `EngineCoreRequest` (`async_llm.py:364-375`; `InputProcessor.process_inputs` at `vllm/v1/engine/input_processor.py:523`); `n>1` fans out child requests (`async_llm.py:402-411`).
4. **Register output side, then send**: `_add_request` adds the request to the in-process `OutputProcessor` and ships it to the engine core: `await self.engine_core.add_request_async(request)` (`async_llm.py:414-426`).
5. **ZMQ send**: `AsyncMPClient.add_request_async` → `_send_input(EngineCoreRequestType.ADD, request)` → msgpack-encoded `input_socket.send_multipart((engine_id, type, payload))` on the ROUTER socket (`vllm/v1/engine/core_client.py:970-973`, `913-948`).
6. **Engine-core input thread**: `process_input_sockets` polls the DEALER socket, decodes (`add_request_decoder.decode`), pre-processes, and `self.input_queue.put_nowait((request_type, request))` (`vllm/v1/engine/core.py:1139-1218`).
7. **Busy loop picks it up**: `run_busy_loop` → `_process_input_queue` → `_handle_client_request` → `EngineCore.add_request` → `self.scheduler.add_request(request)` (`core.py:1018-1046`, `1081-1083`, `288-319`) → request enters `self.waiting` with status WAITING (`vllm/v1/core/sched/scheduler.py:1649-1669`).
8. **Schedule**: next `step()` → `scheduler.schedule()` — prefix-cache lookup (`scheduler.py:600-604`), `allocate_slots` (`scheduler.py:715-724`), admission to `running` (`scheduler.py:765-784`), `SchedulerOutput` built (`scheduler.py:862-877`).
9. **Dispatch to workers**: `self.model_executor.execute_model(scheduler_output, non_block=True)` (`core.py:405`) → `MultiprocExecutor.collective_rpc` enqueues onto the shared-memory `rpc_broadcast_mq` (`vllm/v1/executor/multiproc_executor.py:270-280`, `337`).
10. **Worker process**: `WorkerProc.worker_busy_loop` dequeues, calls `worker.execute_model(scheduler_output)` (`multiproc_executor.py:845-858`) → `Worker.execute_model` (PP recv if not first rank) → `self.model_runner.execute_model(...)` (`vllm/v1/worker/gpu_worker.py:606-656`).
11. **Model forward**: `GPUModelRunner.execute_model` — `_update_states` (persistent batch update, `vllm/v1/worker/gpu_model_runner.py:3341`), `_prepare_inputs` (`gpu_model_runner.py:3383`), cudagraph/padding decision (`gpu_model_runner.py:3398-3411`), `model_output = self._model_forward(...)` (`gpu_model_runner.py:3538-3544`), `logits = self.model.compute_logits(...)` (`gpu_model_runner.py:3574`), then stash `ExecuteModelState` and `return None` (`gpu_model_runner.py:3605-3618`).
12. **Sample**: engine core sees `model_output is None` → `model_executor.sample_tokens(grammar_output)` (`core.py:412-413`) → `GPUModelRunner.sample_tokens`: apply grammar bitmask, `sampler_output = self._sample(logits, spec_decode_metadata)` (`gpu_model_runner.py:3660-3666`), build `ModelRunnerOutput` with `sampled_token_ids` (`gpu_model_runner.py:3775`), returned via the worker response MQ (`multiproc_executor.py:814-827`).
13. **Scheduler update**: `scheduler.update_from_output(...)` — append token, `check_stop`, build `EngineCoreOutput` per request, keyed by client (`scheduler.py:1246-1499`).
14. **Engine-core output thread**: `_process_engine_step` puts each `(client_index, EngineCoreOutputs)` on `output_queue` (`core.py:1060-1063`); `process_output_sockets` msgpack-encodes with reusable buffers and zero-copy `send_multipart` on the PUSH socket (`core.py:1220-1288`).
15. **Front-end output task**: `AsyncMPClient.process_outputs_socket` receives frames on the PULL socket, decodes, `outputs_queue.put_nowait(outputs)` (`core_client.py:873-892`); `AsyncLLM.output_handler` awaits `engine_core.get_output_async()` and calls `output_processor.process_outputs(...)` in chunks (`async_llm.py:662-695`).
16. **Detokenize & deliver**: `OutputProcessor.process_outputs` — detokenize, stop-string check, `req_state.queue.put(request_output)` into the per-request `RequestOutputCollector` (`vllm/v1/engine/output_processor.py:582-692`, queue put at 660-662); stop-string finishes trigger an abort RPC back to the core (`async_llm.py:691-695`).
17. **Yield to caller**: the per-request `generate()` task wakes on `out = q.get_nowait() or await q.get()` and yields the `RequestOutput` to the API handler (`async_llm.py:583-596`). Loop 8→17 repeats once per engine step until `check_stop` fires; the request's blocks are freed (step 13 → `_free_request`) and its ID is propagated in `finished_req_ids` of the next `SchedulerOutput` so workers clean up (`scheduler.py:871-875`).

## 8. Ideas worth borrowing for wm-infra

1. **Keep `Scheduler.schedule()`-style phase orchestration monolithic; extract only override hooks.** vRL's `VideoIterationRunner._advance()` per-phase dispatch should follow the AsyncScheduler pattern: if a model family needs different post-step behavior, give the base scheduler/runner named `_update_after_*` hooks (cf. `scheduler.py:922`, `async_scheduler.py:18`) instead of splitting the main loop into per-phase micro-functions. A 61-line subclass overriding two hooks (`async_scheduler.py`) is the payoff.
2. **Unify "prefill vs decode" the way vLLM unified it — by counters, not phases, where possible.** The insight at `scheduler.py:321-331` (each request just catches `num_computed_tokens` up to a target) is directly transplantable to diffusion: a denoise request is "current_step catching up to total_steps", and chunking/preemption fall out of `min(work, budget)` (`scheduler.py:669`) rather than a phase enum branch. wm-infra's `DenoiseLoopState(current_step, total_steps)` is already shaped for this; the scheduler should budget *steps* the way vLLM budgets *tokens*.
3. **Adopt the diff-based step output split**: `scheduled_new_reqs: list[NewRequestData]` (full payload, sent once) vs `scheduled_cached_reqs: CachedRequestData` ("we only send the diff to minimize the communication cost", `output.py:185-192`). For video requests with large conditioning (text embeds, reference frames), send conditioning once at first schedule and only step indices afterwards — this matters as soon as the engine loop and model executor are in different processes.
4. **Allocation-result-as-admission-signal**: `allocate_slots` returning `None` is the single signal that drives both preemption (running) and admission stop (waiting) (`kv_cache_manager.py:336-338`, `scheduler.py:430-479`). A future `LatentCacheManager.allocate(...) -> handle | None` gives the vRL planner the same one-bit backpressure interface, with recompute-style preemption (free everything, `num_computed_tokens = 0`, prepend to waiting; `scheduler.py:900-920`) as the simplest correct eviction policy.
5. **Tier your message types**: msgspec `gc=False, array_like` structs for anything crossing the gateway↔engine IPC boundary, plain dataclasses inside the engine step (`engine/__init__.py:55-60` vs `output.py:182`). Also steal the hex-byte request-type enum (`EngineCoreRequestType`, `engine/__init__.py:207-218`) for vRL's IPC framing, and `IntEnum` with ordering-based `is_finished` (`request.py:316-318`) for request status.
6. **Name by the vLLM taxonomy**: resource lifecycle = `*Manager`, multiplexer over managers = `*Coordinator`, free-list = `*Pool`, process wrapper = `*Proc` + `*ProcHandle`, client mirror = `Sync*/Async*Client`, drain semantics = `take_*`, conditional = `_maybe_*`, fallible = `_try_*`, explicit `_async` suffix twins. wm-infra's `FeedbackMailbox`/`ContinuousBatchPlanner` already fit; future KV-like latent-cache work should be `LatentCacheManager` over a `LatentBlockPool`, not a "PagedLatentEngine".
7. **The facade-with-stated-purpose pattern for the planner↔cache boundary**: a `KVCacheBlocks`-style result object whose docstring states it exists "to hide KVCacheManager's internal data structure from the Scheduler" (`kv_cache_manager.py:21-27`), plus a shared pre-built empty instance to avoid per-step allocation (`:134-141`). wm-infra's scheduler should never see latent-pool internals, only an opaque allocation handle.
8. **CPU/GPU overlap via split execute/sample**: compute CPU-side per-step work (grammar bitmask in vLLM; for vRL, e.g. guidance schedules, reward pre-staging, next-batch planning) between issuing `execute_model(non_block=True)` and `future.result()` (`core.py:404-413`); plus the batch-queue pattern (`step_with_batch_queue`, `core.py:434-552`) for keeping multiple in-flight batches when the executor supports it.
9. **Keep orchestration backend pluggable the way `Executor.get_class` does** (`executor/abstract.py:46-86`): one ABC whose entire surface is `collective_rpc`, with mp/Ray/uniproc/external-launcher as interchangeable backends — and note that even under Ray, vLLM keeps ZMQ for front-end IPC and its own NCCL groups for collectives; Ray is only process placement + compiled-DAG dataflow. The shm `MessageQueue` broadcast (`multiproc_executor.py:127-137`) is the prior art for vRL's same-node engine→executor dispatch.
10. **Comment discipline worth copying verbatim**: signed `NOTE(name)` for load-bearing decisions, numbered `# 1) ... # 2)` steps in every busy loop, and ASCII layout diagrams for any allocation math (`kv_cache_manager.py:238-259`). Also the "skip extra step when max_tokens surely reached" guard with its arithmetic spelled out in a comment (`scheduler.py:357-371`) — exactly the style needed for denoise-step accounting with CFG/spec-style lookahead.
11. **Engine-loop hygiene specifics**: yield-GIL sleep when nothing executed but requests wait on background threads (`core.py:1067-1072`); chunked output processing with `await asyncio.sleep(0)` between chunks so the front-end event loop never starves (`async_llm.py:673-689`); `weakref.finalize` + resources dataclass for ZMQ teardown (`core_client.py:472-476`); dual-queue eager aborts so cancellations apply during a forward pass (`core.py:1210-1218, 415-417`). All map one-to-one onto vRL's gateway/engine split.
12. **`reset_prefix_cache` as a first-class RLHF hook**: cache invalidation after weight updates is a named utility RPC reachable from the front-end (`block_pool.py:424-457`, `core_client.py:995-1000`) — vRL's weight-sync path needs the identical hook for any latent/embedding cache.

## 9. Source-of-truth index

| Claim | Source |
|---|---|
| v1 package split (engine/core/executor/worker) | `vllm/v1/` directory tree |
| File line counts (scheduler 2178, core 1681, core_client 1458, async_llm 1099, kv_cache_manager 494, block_pool 490, gpu_model_runner 6278) | `wc -l` on this checkout (verified) |
| AsyncLLM spawns engine via `make_async_mp_client`; output_handler task; chunking; anti-cycle closure | `vllm/v1/engine/async_llm.py:147-155, 647-713` |
| AsyncLLM add_request / generate loop | `vllm/v1/engine/async_llm.py:286-429, 537-606` |
| EngineCore owns executor+scheduler; KV profiling at init | `vllm/v1/engine/core.py:106, 139-146, 226-283` |
| step(): schedule → non-block execute → bitmask overlap → sample → update (verified) | `vllm/v1/engine/core.py:389-422` |
| step_with_batch_queue, PP bubble elimination, deferred sampling | `vllm/v1/engine/core.py:434-552` |
| batch_queue sized by executor `max_concurrent_batches`; step_fn selection | `vllm/v1/engine/core.py:177-208`; `vllm/v1/executor/abstract.py:237-239`; `vllm/v1/executor/ray_executor.py:111-117` (verified) |
| EngineCoreProc ZMQ-wrapper docstring; IO threads; identity | `vllm/v1/engine/core.py:684-783` |
| `run_busy_loop` numbered 2-step body (verified) | `vllm/v1/engine/core.py:1018-1026` |
| Input-queue drain + flat if/elif request dispatch | `vllm/v1/engine/core.py:1028-1106` |
| GIL-yield sleep in step loop | `vllm/v1/engine/core.py:1067-1072` |
| Eager abort dual-queue | `vllm/v1/engine/core.py:1210-1218, 415-417, 554-562` |
| ZMQ IO threads (DEALER input, PUSH output, reusable buffers) | `vllm/v1/engine/core.py:1139-1288` |
| HELLO/READY handshake protocol | `vllm/v1/engine/core.py:844-927`; `vllm/v1/engine/utils.py:54-77` |
| Engine process spawn (`EngineCore_DP{i}`) / launch flow | `vllm/v1/engine/utils.py:80-131, 777-934` |
| Ray engine-core actors for DP backend | `vllm/v1/engine/utils.py:227-357, 847-858, 905-918` |
| DPCoordinator process, wave coordination | `vllm/v1/engine/coordinator.py:22-90`; `vllm/v1/engine/utils.py:822-845`; `vllm/v1/engine/core.py:1161-1188, 1246-1254` |
| Non-MoE DP ranks independent (treated as DP=1) | `vllm/v1/engine/core.py:984-996` |
| Utility RPC path (weight updates, sleep/wake) | `vllm/v1/engine/core.py:650-657, 1086-1100`; `vllm/v1/executor/abstract.py:303-341` |
| No-phase token-deficit scheduling design comment (verified verbatim) | `vllm/v1/core/sched/scheduler.py:321-331` |
| FCFS/PRIORITY queues (deque / heap by (priority, arrival_time)) | `vllm/v1/core/sched/request_queue.py:13-17, 75-128, 131-198` |
| Budgets: max_num_seqs / max_num_batched_tokens | `vllm/v1/core/sched/scheduler.py:100-102` |
| Phase 1 running loop: deficit calc, clipping, continue-not-break | `vllm/v1/core/sched/scheduler.py:352-427` |
| Preemption loop (FCFS pop newest; priority max(priority,arrival); refund; recompute-style) (verified) | `vllm/v1/core/sched/scheduler.py:429-479, 900-920` |
| `_preempt_request` contract note + 2 call sites (verified) | `vllm/v1/core/sched/scheduler.py:900-921, 471, 1773` |
| Phase 2 waiting loop: gate on no-preemption, skip gauntlet, prefix lookup, admission | `vllm/v1/core/sched/scheduler.py:530-804` |
| Chunked prefill = min(deficit, budget); disabled → break | `vllm/v1/core/sched/scheduler.py:654-669` |
| SchedulerOutput build: NewRequestData vs CachedRequestData deltas | `vllm/v1/core/sched/scheduler.py:829-877, 999-1057`; `vllm/v1/core/sched/output.py:182-253` |
| Eager `num_computed_tokens` advance after scheduling | `vllm/v1/core/sched/scheduler.py:922-954` |
| update_from_output: spec rollback, stop check, per-client outputs, freeing; perf comment | `vllm/v1/core/sched/scheduler.py:1246-1499, 1283-1285, 1543-1559, 1723-1744` |
| Stop conditions; `remove_all` perf docstring | `vllm/v1/core/sched/utils.py:8-64` |
| Async scheduling placeholders; 61-line subclass over two hooks | `vllm/v1/core/sched/async_scheduler.py:12-60`; `scheduler.py:373-377, 922, 1543` |
| KVCacheBlocks boundary-hiding docstring + empty-instance reuse (verified) | `vllm/v1/core/kv_cache_manager.py:21-27, 134-141, 486-490` |
| KVCacheManager delegating facade | `vllm/v1/core/kv_cache_manager.py:119-131, 378-399, 459-465` |
| Prefix lookup; full-hit minus last token | `vllm/v1/core/kv_cache_manager.py:164-204`; `vllm/v1/core/single_type_kv_cache_manager.py:410-445` |
| allocate_slots: ASCII layout, 3 stages, None on shortfall (verified), draft-token cache cap | `vllm/v1/core/kv_cache_manager.py:206-376` (check at 336-338, cap at 365-374) |
| Coordinator fan-out to per-group managers | `vllm/v1/core/kv_cache_coordinator.py:71-199` |
| BlockPool: ref counts, touch, lazy LRU eviction, append-only hash map, null block | `vllm/v1/core/block_pool.py:46-50, 128-180, 300-403` |
| FreeKVCacheBlockQueue LRU + tail-first order; reverse-order free | `vllm/v1/core/kv_cache_utils.py:157-187`; `vllm/v1/core/single_type_kv_cache_manager.py:265-279` |
| reset_prefix_cache (RLHF weight-update hook) | `vllm/v1/core/block_pool.py:424-457`; `vllm/v1/engine/core_client.py:995-1000` |
| Executor backend selection (ray/mp/uni/external_launcher) | `vllm/v1/executor/abstract.py:46-86` |
| Executor RPC surface (`collective_rpc`, execute/sample) | `vllm/v1/executor/abstract.py:133-228` |
| Shm MessageQueue broadcast + per-worker response MQs; multi-node without Ray | `vllm/v1/executor/multiproc_executor.py:123-137, 499-529, 172-185` |
| WorkerProc spawn, busy loop, output_rank filter (verified) | `vllm/v1/executor/multiproc_executor.py:586-686, 845-871` |
| MP collective_rpc + FutureWrapper ordering | `vllm/v1/executor/multiproc_executor.py:64-90, 303-375` |
| Worker monitor / failure callback | `vllm/v1/executor/multiproc_executor.py:232-268` |
| Async-output worker thread | `vllm/v1/executor/multiproc_executor.py:829-843` |
| NCCL init per worker (TP/PP/CP groups); rendezvous addr | `vllm/v1/worker/gpu_worker.py:1030-1056`; `multiproc_executor.py:120-122` |
| Ray actors + placement groups (verified) + rank sort | `vllm/v1/executor/ray_executor.py:163-279` |
| Ray compiled DAG (PP×TP SPMD, NCCL/shm channels), deferred to sample_tokens | `vllm/v1/executor/ray_executor.py:413-486, 545-638` |
| MPClient ZMQ ROUTER/PULL, engine identities, BackgroundResources finalizer | `vllm/v1/engine/core_client.py:442-562, 356-379` |
| AsyncMPClient output task + zero-copy send tracking | `vllm/v1/engine/core_client.py:822-948` |
| DP client variants (external vs internal LB) | `vllm/v1/engine/core_client.py:99-124` |
| Worker execute_model / split sampling (ExecuteModelState stash, return None) | `vllm/v1/worker/gpu_worker.py:599-660`; `vllm/v1/worker/gpu_model_runner.py:3312-3618, 3621-3666, 3775` |
| OutputProcessor → per-request queue → generate() yield | `vllm/v1/engine/output_processor.py:582-692`; `vllm/v1/engine/async_llm.py:583-596` |
| API entry call site | `vllm/entrypoints/openai/completion/serving.py:226` |
| msgspec `gc=False` IPC structs (verified) | `vllm/v1/engine/__init__.py:55-60, 176-181` |
| Hex-byte request-type enum (verified) | `vllm/v1/engine/__init__.py:207-218` |
| IntEnum FinishReason rationale | `vllm/v1/engine/__init__.py:32-49` |
| `RequestStatus` ordering trick + reason map | `vllm/v1/request.py:296-336` |
| ABC interfaces, no Protocol (grep empty) | `vllm/v1/core/sched/interface.py:21`; `vllm/v1/executor/abstract.py:36-60`; `vllm/v1/engine/core_client.py:63-75`; grep over `vllm/v1/{core,engine,executor}` |
| Naming taxonomy (Manager/Coordinator/Pool/Proc/Client; verb vocabulary) | `vllm/v1/core/single_type_kv_cache_manager.py:408,469,752,1010`; `kv_cache_coordinator.py:302,368`; `multiproc_executor.py:455,465,492`; `core_client.py:63,442,652,822, 204-251`; `block_pool.py:332,372,480` |
| Signed NOTE comments | `vllm/v1/core/sched/scheduler.py:322, 423-425, 879, 1298, 1774`; `vllm/v1/engine/async_llm.py:700` |
| Async-step guard arithmetic comment | `vllm/v1/core/sched/scheduler.py:357-371` |

---
# Part II — Deep Dive

Same checkout (`9b17c5746`); all paths repo-relative; citations re-verified against this checkout. Part I stopped at the executor boundary and the scheduler's *policy* view of the KV cache. Part II descends three levels: how `GPUModelRunner` turns a `SchedulerOutput` into a forward pass (§10), the memory subsystem's actual data structures (§11), the extension surfaces (§12), the full startup/failure story (§13), and a distilled worth-copying/anti-pattern list (§14).

## 10. Below the executor: GPUModelRunner internals

Part I's flow trace (steps 10–12) covered `execute_model`/`sample_tokens` at the call level. This section is the tensor level: `gpu_model_runner.py` is 6278 lines, and almost all of it serves one goal — **zero per-step allocation** on the hot path.

### 10.1 The buffer substrate: `CpuGpuBuffer` and the persistent batch

Almost every per-step tensor in the runner is a **pre-allocated triple** (pinned CPU tensor, GPU mirror, numpy view) so the hot loop never allocates:

```python
class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""
    def __init__(self, ...):
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = torch.zeros_like(self.cpu, device=device)
        ...
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        ...
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
```
(`vllm/v1/utils.py:105-136`, verified)

`GPUModelRunner.__init__` allocates the per-step buffers once, sized `max_num_tokens = max_num_batched_tokens` and `max_num_reqs = max_num_seqs` (`gpu_model_runner.py:377-378`), under an explicit banner:

```python
# Persistent buffers for CUDA graphs.
self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)
self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
```
(`gpu_model_runner.py:564-570`; more buffers — `inputs_embeds`, `is_token_ids`, `discard_request_mask`, `num_decode_draft_tokens`, `num_accepted_tokens` — at 578-590). A cached `arange_np` avoids re-creating index ramps each step: `# OPTIMIZATION: Cache the tensors rather than creating them every step.` … `self.arange_np = np.arange(max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens), dtype=np.int64)` (`gpu_model_runner.py:629-634`).

Request state is split in two:

- `self.requests: dict[str, CachedRequestState]` — Python-side record per request (prompt ids, output ids, block ids, sampling params) (`gpu_model_runner.py:493`; dataclass at `vllm/v1/worker/gpu_input_batch.py:30-78`).
- `InputBatch` — the **persistent batch**: SoA numpy/torch arrays indexed by a dense row index, including a `(max_num_reqs, max_model_len)` int32 token matrix that lives unpinned on CPU (`# TODO(woosuk): This buffer could be too large if max_model_len is big.` `gpu_input_batch.py:111-121`), `num_computed_tokens_cpu`, a `MultiGroupBlockTable` (`gpu_input_batch.py:141-151`), and per-row sampling-param arrays each with a GPU tensor + pinned CPU twin + numpy view (`temperature`/`top_p`/`top_k`/penalties, `gpu_input_batch.py:154-207`), plus `set`s like `greedy_reqs`, `top_p_reqs` used to compute `no_top_p`-style batch flags.

### 10.2 `_update_states`: diffing the persistent batch

Each step starts by reconciling the persistent batch against the scheduler's delta (`gpu_model_runner.py:874-1120`): pop finished requests, then evict rows whose request was *not* scheduled this step:

```python
unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
# NOTE(woosuk): The persistent batch optimization assumes that
# consecutive batches contain mostly the same requests. If batches
# have low request overlap (e.g., alternating between two distinct
# sets of requests), this optimization becomes very inefficient.
```
(`gpu_model_runner.py:915-920`, verified). New requests build a `CachedRequestState` (per-request seeded `torch.Generator` if `RANDOM_SEED`, `gpu_model_runner.py:937-943`); cached requests get `num_computed_tokens` updated and `new_block_ids` appended to `req_state.block_ids` or replaced wholesale on resume-from-preemption (`gpu_model_runner.py:1063-1073`). The epilogue is the row-compaction dance:

```python
self.input_batch.condense()
# Allow attention backend to reorder the batch, potentially
self._may_reorder_batch(scheduler_output)
# Refresh batch metadata with any pending updates.
self.input_batch.refresh_metadata()
```
(`gpu_model_runner.py:1115-1119`). `condense()` slides the highest-index live rows down into removed-row holes, copying token rows and swapping spec-token lists so rows `0..num_reqs-1` are dense (`gpu_input_batch.py:626-756`). `_may_reorder_batch` lets attention backends pull decodes to the front of the batch when they have a split decode/prefill kernel path (threshold from `reorder_batch_threshold`, computed *after* builders exist — `gpu_model_runner.py:5429-5436, 5597`).

### 10.3 `_prepare_inputs`: numpy gather → pinned copy → async H2D

`_prepare_inputs` (`gpu_model_runner.py:1454-1673`) is where packing happens. The batch is **packed, not padded, at token level**: all scheduled tokens of all requests are concatenated into one flat dim (padding only enters later for CUDA graphs). Steps, all on CPU numpy first:

1. Kick off the block-table H2D copy *first* to overlap with CPU work: `# OPTIMIZATION: Start copying the block table first.` (`gpu_model_runner.py:1472-1475`).
2. Build `req_indices = np.repeat(arange[:num_reqs], num_scheduled_tokens)` and the batched arange via `_get_cumsum_and_arange` — a vectorized `[2,5,3] -> [0,1, 0,1,2,3,4, 0,1,2]` using `np.repeat` of cumsum offsets (`gpu_model_runner.py:1268-1286`).
3. Positions: `np.add(num_computed_tokens_cpu[req_indices], arange, out=positions_np)` (`gpu_model_runner.py:1485-1490`).
4. Token gather from the 2-D token matrix using flattened indices `positions + req_index * max_model_len`, with a deliberate kernel choice: `# NOTE(woosuk): We use torch.index_select instead of np.take here because torch.index_select is much faster than np.take for large tensors.` writing straight into the pinned `input_ids.cpu` (`gpu_model_runner.py:1505-1520`).
5. Slot mapping: `block_table.compute_slot_mapping(req_indices, positions_np)` computes `block_numbers * block_size + offset` in numpy on the CPU side of the block-table buffer, then `commit_slot_mapping` copies it up (`gpu_model_runner.py:1568-1570`; kernel math in §11.3).
6. Attention prerequisites with **CUDA-graph-safe padding semantics**: `query_start_loc` beyond the batch is filled with the last cumsum (`# Note: pad query_start_loc to be non-decreasing, as kernels like FlashAttention requires that`, `gpu_model_runner.py:1573-1576`) and `seq_lens.np[num_reqs:].fill(0)` (`# Fill unused with 0 for full cuda graph mode.`, `gpu_model_runner.py:1584-1586`).
7. `discard_request_mask` marks chunked-prefill rows whose `seq_len < num_tokens` — they get sampled anyway and discarded later: `# NOTE(woosuk): Due to chunked prefills, the batch may contain partial requests. While we should not sample any token from these partial requests, we do so for simplicity.` (`gpu_model_runner.py:1591-1597, 1620-1626`).
8. `logits_indices = query_start_loc[1:] - 1` — sample only the last token position of each request (`gpu_model_runner.py:1626`), unless spec-decode metadata expands it.

`_prepare_input_ids` (`gpu_model_runner.py:1288-1409`) handles the async-scheduling wrinkle: the previous step's sampled tokens may still be GPU-only. If the batch is unchanged and unreordered it does a single sliced device-to-device copy from `prev_sampled_token_ids`; otherwise it uploads index tensors and `scatter_`s sampled (and draft) tokens into `input_ids.gpu` (`gpu_model_runner.py:1361-1409`), falling back to the full pinned-CPU upload when not all requests were decodes last step.

### 10.4 `_preprocess` and the forward call

`_preprocess` (`gpu_model_runner.py:2735-2851`) picks the model's input modality: text-only models pass `input_ids` directly (`# While it is possible to use embeddings ... it is not desirable for performance since then the embedding layer is not included in the CUDA graph.`, `gpu_model_runner.py:2811-2817`); multimodal models always run `embed_input_ids` with gathered mm embeddings and feed `inputs_embeds` instead (`gpu_model_runner.py:2756-2781`). Note the slicing convention: model inputs are sliced to `num_input_tokens` (**padded** length), while embedding work uses `num_scheduled_tokens` (unpadded). The actual call is the trivially small `_model_forward` (`gpu_model_runner.py:3023-3052`), wrapped in `set_forward_context(attn_metadata, ..., cudagraph_runtime_mode=cudagraph_mode, batch_descriptor=batch_desc, ...)` (`gpu_model_runner.py:3523-3540`) — the forward context is the covert channel through which attention layers and CUDA-graph wrappers learn what this batch is.

Part I described the `execute_model`/`sample_tokens` split from the scheduler side (grammar bitmask overlapped with the forward pass); the worker-side mechanics: on the last PP rank `execute_model` computes `logits = self.model.compute_logits(hidden_states[logits_indices])` (`gpu_model_runner.py:3577-3578`), stashes everything in `ExecuteModelState` (a NamedTuple, `gpu_model_runner.py:313-326`) and `return None` (`gpu_model_runner.py:3605-3618`).

### 10.5 CUDA graphs: modes, dispatcher, wrapper, capture

**Two graph flavors, a 5-valued mode enum.**

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)
```
(`vllm/config/compilation.py:51-61`, verified). Tuple values encode `(decode_mode, mixed_mode)`: a config like `FULL_AND_PIECEWISE` means *uniform decode batches* run a FULL graph (attention kernel inside the graph) while *mixed prefill-decode batches* run PIECEWISE graphs (torch.compile splits the FX graph at attention ops; each non-attention segment is its own graph, attention runs eagerly between them).

What gets wrapped where:
- **FULL**: the entire model is wrapped once at the end of `load_model`: `self.model = CUDAGraphWrapper(self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)` (`gpu_model_runner.py:4271-4280`).
- **PIECEWISE**: each compiled subgraph piece is wrapped by the compiler backend with `runtime_mode=CUDAGraphMode.PIECEWISE` and per-piece options (`gc_disable` for all but the first piece, `weak_ref_output` only for the last) (`vllm/compilation/backends.py:491-510`).

**What is captured, and what "shape" means.** `CUDAGraphWrapper.__call__` reads the forward context and **blindly trusts** the dispatched key:

```python
forward_context = get_forward_context()
batch_descriptor = forward_context.batch_descriptor
cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
if (cudagraph_runtime_mode == CUDAGraphMode.NONE
        or cudagraph_runtime_mode != self.runtime_mode):
    return self.runnable(*args, **kwargs)
```
(`vllm/compilation/cuda_graph.py:207-223`, verified). On first sight of a `BatchDescriptor` it captures into a shared global graph pool; afterwards `entry.cudagraph.replay()` returns the cached weak-ref'd output (`cuda_graph.py:225-310`). Crucially the wrapper holds **no input buffers** — `Note: CUDAGraphWrapper does not store persistent buffers or copy any runtime inputs into that buffers for replay. We assume implementing them is done outside of the wrapper.` (`cuda_graph.py:155-159`). The "outside" is exactly the runner's persistent `CpuGpuBuffer`s from §10.1: replay is only valid because `input_ids.gpu[:n]`, `positions.gpu[:n]` etc. are always slices of the same allocation (asserted by data_ptr comparison when `VLLM_LOGGING_LEVEL=DEBUG`, `cuda_graph.py:296-305`).

The dispatch key is `BatchDescriptor(num_tokens, num_reqs, uniform, has_lora, num_active_loras)` (`vllm/forward_context.py:29-57`) — note **no sequence lengths**: dynamic context lengths are handled inside attention kernels via the metadata tensors, so only the padded token count (and request count, for FULL graphs) defines a graph.

**Capture sizes and runtime padding.** Default capture sizes are generated, not hand-listed: `max_cudagraph_capture_size = min(max_num_seqs * decode_query_len * 2, 512)`, then `[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)`, filtered by `max_num_batched_tokens` and TP divisibility for sequence parallelism (`vllm/config/vllm.py:1171-1280`, esp. 1226-1268). At runtime `CudagraphDispatcher` pre-computes a dense `bs -> padded size` table (`vllm/v1/cudagraph_dispatcher.py:70-84`) and `dispatch()` resolves `(num_tokens, uniform_decode, has_lora, ...)` to `(runtime_mode, padded BatchDescriptor)`:

```python
if (not self.keys_initialized
        or self.cudagraph_mode == CUDAGraphMode.NONE
        or num_tokens > self.compilation_config.max_cudagraph_capture_size):
    return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```
(`cudagraph_dispatcher.py:259-264`) — i.e. **replay is skipped** (eager fall-through) when the padded batch exceeds the largest captured size, when no exact/relaxed key exists, and case-by-case when features force it: cascade attention or encoder inputs disable FULL via `disable_full` (`gpu_model_runner.py:3132-3134`, callsite passes `use_cascade_attn or has_encoder_output`), and KV-scale calculation forces a wholesale `cudagraph_mode = CUDAGraphMode.NONE` for the first step (`gpu_model_runner.py:3510-3516`). Uniform-decode batches first try the exact FULL key, then a `relax_for_mixed_batch_cudagraphs()` key (drops `num_reqs`/`uniform`) for FULL, then PIECEWISE, then eager (`cudagraph_dispatcher.py:285-302`). The dispatcher's docstring states the contract: keys stored in it are "the only source of truth for valid cudagraphs", and wrappers "blindly trust" what it puts in the forward context (`cudagraph_dispatcher.py:13-30`).

Padding is two-stage in `_determine_batch_execution_and_padding` (`gpu_model_runner.py:3076-3199`): SP padding first, then graph-size padding via dispatch; under DP, `coordinate_batch_across_dp` synchronizes padded token counts across ranks and re-dispatches (`gpu_model_runner.py:3148-3181`). Everything downstream (`num_tokens_padded`, `num_reqs_padded`) flows into attention metadata only when `pad_attn = cudagraph_mode == CUDAGraphMode.FULL` (`gpu_model_runner.py:3451`) — piecewise graphs don't need padded attention because attention runs outside them.

**Capture procedure.** `capture_model` (`gpu_model_runner.py:5185-5254`) freezes the GC (`gc.freeze()` to keep capture fast), enables capture globally, and walks `cudagraph_dispatcher.get_capture_descs()` — PIECEWISE first, then FULL, each list sorted **largest-first** so smaller graphs reuse the big graphs' pool memory (`# Capture the large shapes first so that the smaller shapes can reuse the memory pool allocated for the large shapes.`, `gpu_model_runner.py:5214-5216`; sort at `cudagraph_dispatcher.py:316-321`). Each descriptor is exercised via `_dummy_run` with warmup passes at mode NONE followed by one capture pass (`gpu_model_runner.py:5310-5331`). `_dummy_run` fabricates a fake batch shape — uniform decode (`[max_query_len] * num_reqs`) or an even token split — and for FULL capture builds real attention metadata with `for_cudagraph_capture=True`, which forces `max_seq_len = self.max_model_len` so sliding-window kernels select the right code path (`gpu_model_runner.py:4673-4706, 1705-1709`). Graph memory cost is measured as the `mem_get_info` delta and logged ("usually takes 5~20 seconds", `gpu_model_runner.py:5218-5252`).

**Backend capability negotiation.** Whether FULL graphs are even legal depends on the attention backend. Each metadata-builder class declares a support level:

```python
class AttentionCGSupport(Enum):
    ALWAYS = 3                       # supports mixed-prefill-decode
    UNIFORM_BATCH = 2                # uniform query lens (spec-decode ok)
    UNIFORM_SINGLE_TOKEN_DECODE = 1  # query_len==1 decodes only
    NEVER = 0
```
(`vllm/v1/attention/backend.py:416-430`). `_check_and_update_cudagraph_mode` takes the **min** support across all backends in all KV-cache groups and downgrades the configured mode with logged warnings: FULL mixed-batch + non-ALWAYS backend → `FULL_AND_PIECEWISE` or `FULL_DECODE_ONLY`; NEVER → `PIECEWISE` (if attention was compiled piecewise) or `NONE`; spec-decode with support < UNIFORM_BATCH → `PIECEWISE`/`NONE` (`gpu_model_runner.py:5437-5557`). Concrete example: FlashAttention is `ALWAYS` on FA3 but `UNIFORM_BATCH` on FA2 because of FA2's special `max_query_len=1` packed-GQA kernel (long NOTE at `vllm/v1/attention/backends/flash_attn.py:234-257`). Only after this resolution are the dispatcher keys generated (`gpu_model_runner.py:5589-5592`).

### 10.6 Attention backend abstraction

**Selection: per-layer, validated priority list.** Selection happens at **model-build time**, inside each `Attention` layer's constructor: `self.attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype, block_size, use_mla=False, has_sink=..., attn_type=...)` (`vllm/model_executor/layers/attention/attention.py:266-276`). `get_attn_backend` reads any user-forced backend from `attention_config.backend` and delegates to a `@cache`d resolver that asks the platform (`vllm/v1/attention/selector.py:46-119`). On CUDA the platform keeps explicit priority lists — e.g. non-MLA on Blackwell (`major == 10`): `FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION`; on older parts FLASH_ATTN first (`vllm/platforms/cuda.py:45-82`). Each candidate self-vets via `backend_class.validate_configuration(device_capability, head_size, dtype, kv_cache_dtype, ...)` (classmethod stack on the `AttentionBackend` ABC: `supports_head_size` / `supports_dtype` / `supports_block_size` / `supports_compute_capability` etc., `vllm/v1/attention/backend.py:121-273`); the first valid one by priority wins, and a forced-but-invalid backend raises instead of silently falling back (`vllm/platforms/cuda.py:306-380`). A selected backend can also retroactively force a KV-cache layout: `set_kv_cache_layout(required_layout)` (`selector.py:106-117`).

**Grouping: `(backend class, KVCacheSpec)` → `AttentionGroup`.** The runner doesn't deal with layers individually. `initialize_attn_backend` buckets every layer of every KV-cache group by `(full_cls_name, layer_kv_cache_spec)` — deduping on the class *name* because dynamically-created backend subclasses would otherwise not compare equal (`# Dedupe based on full class name; this is a bit safer than using the class itself as the key...`, `gpu_model_runner.py:5343-5390`) — and creates one `AttentionGroup` per bucket, indexed `attn_groups[kv_cache_group_id][attn_group_id]` (`gpu_model_runner.py:5380-5414`). Each group later gets one `AttentionMetadataBuilder` per micro-batch (`initialize_metadata_builders`, `gpu_model_runner.py:5414-5432`).

**The metadata-building step per batch.** Per step, the runner builds **one `CommonAttentionMetadata`** (`vllm/v1/attention/backend.py:287-333`: `query_start_loc` GPU+CPU, `seq_lens`, `num_reqs/num_actual_tokens/max_query_len/max_seq_len`, `block_table_tensor`, `slot_mapping`) from its persistent buffers:

```python
cm_base = CommonAttentionMetadata(
    query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
    seq_lens=self.seq_lens.gpu[:num_reqs_padded], ...
    block_table_tensor=block_table_gid_0,
    slot_mapping=slot_mapping_gid_0, causal=True)
```
(`gpu_model_runner.py:1744-1760`). Then `_build_attention_metadata` (`gpu_model_runner.py:1673-1915`) shallow-copies it per KV-cache group (only block table / slot mapping / encoder lens differ), and per attention group calls `builder.build(common_prefix_len=cascade_len, common_attn_metadata=cm)` — with two fast paths: `builder.build_for_cudagraph_capture` during capture, and a cross-group cache that reuses a built metadata when only the block table changed (`# Cache attention metadata builds across hybrid KV-cache groups ...`, `gpu_model_runner.py:1786-1842`). The result is a per-layer-name dict (`attn_metadata_dict[layer_name] = attn_metadata_i`, `gpu_model_runner.py:1850-1852`) shipped through `set_forward_context`. Padding hygiene for FULL graphs lives here too: unused block-table rows are filled with -1 (`# Fill unused with -1. Needed for reshape_and_cache in full cuda graph mode.`, `gpu_model_runner.py:1733-1736`).

What a concrete `build` does — FlashAttention (`vllm/v1/attention/backends/flash_attn.py:329-512`): slices/derives `max_seq_len`, optionally computes the FA3 **AOT scheduler metadata** (`get_scheduler_metadata(...)`, the precomputed tile schedule), handles cascade-attention splits (separate prefix/suffix schedules), DCP interleaving, and for full-CUDA-graph mode copies the schedule into a **persistent** `self.scheduler_metadata` buffer and zeroes the tail (`# NOTE(woosuk): We should zero out the rest of the scheduler metadata to guarantee the correctness. Otherwise, some thread blocks may use the invalid scheduler metadata and overwrite the output buffer.`, `flash_attn.py:482-490`); under full graphs it also bounds `max_num_splits` so intermediate buffers captured by the graph are large enough (`flash_attn.py:313-326`). Its `update_block_table` is a `copy.copy` + two field swaps (`flash_attn.py:514-523`) — which is exactly why the runner's cross-group cache is worth having.

### 10.7 Sampling and the output path

**Sampler: one fused pass over `[num_logits, vocab]`.** `Sampler.forward` (`vllm/v1/sample/sampler.py:67-130`) snapshots raw logprobs first if requested (top-k logprobs are computed on **pre-penalty** logits — `# NOTE(woosuk): Use the original logits (before any penalties or temperature scaling) for the top-k logprobs. This is different from the V0 sampler...`, `sampler.py:76-79`), casts to fp32, applies allowed-token masks / bad words / non-argmax-invariant logitsprocs / penalties (`apply_logits_processors`, `sampler.py:266-301`), then samples. `sample()` exploits the batch flags from `InputBatch` (§10.1): pure-greedy batches return `argmax` immediately; mixed batches compute both greedy and random results and merge per-row:

```python
sampled = torch.where(
    sampling_metadata.temperature < _SAMPLING_EPS,
    greedy_sampled, random_sampled,
    out=greedy_sampled,  # Reuse tensor
)
```
(`sampler.py:147-205`). Temperature is an in-place batched division with zero-guard (`sampler.py:132-141`); top-k/top-p go through `TopKTopPSampler` (FlashInfer-capable, hence the later `.long()` cast comment at `sampler.py:101-106`). `gather_logprobs` concatenates the sampled token's logprob+rank with the top-k rows into `LogprobsTensors` (`sampler.py:210-264`). With spec decode, `RejectionSampler` replaces the plain sampler (`gpu_model_runner.py:2873-2879`).

**`sample_tokens`: grammar → sample → bookkeeping → output.** `sample_tokens(grammar_output)` unpacks `ExecuteModelState`, applies the structured-output bitmask **now** (`apply_grammar_bitmask(...)`, `gpu_model_runner.py:3659-3663` — the worker half of the bitmask/forward overlap Part I described), runs `_sample`, then `_bookkeeping_sync` (`gpu_model_runner.py:2881-3009`), which:

- discards tokens for partial-prefill rows via `discard_request_mask` and rewinds their generator offsets (`gen.set_offset(gen.get_offset() - 4)`, `gpu_model_runner.py:2903-2910`);
- in **sync** mode materializes Python lists from the GPU tensor via `_to_list`, which copies into a pre-pinned buffer and synchronizes on a dedicated `transfer_event` instead of `.tolist()` — `# `tolist` would trigger a cuda wise stream sync, which would block other copy ops from other cuda streams.` (`gpu_model_runner.py:6189-6203`);
- in **async** mode skips the sync entirely: token ids stay on GPU (`input_batch.prev_sampled_token_ids = sampled_token_ids`, consumed next step by `_prepare_input_ids`'s scatter), and per-request lists get `-1` placeholders (`gpu_model_runner.py:2944-2974`);
- writes accepted tokens back into `token_ids_cpu` / `req_state.output_token_ids` so the *worker* is the source of truth for output history — `# Cache the sampled tokens in the model runner, so that the scheduler doesn't need to send them back. NOTE(woosuk): As an exception, when using PP, the scheduler sends the sampled tokens back...` (`gpu_model_runner.py:2962-2993`).

The return value is a `ModelRunnerOutput` of plain Python lists (`req_ids`, `sampled_token_ids`, logprobs lists, prompt-logprobs dict — `gpu_model_runner.py:3775-3788`). Under async scheduling it is instead wrapped in `AsyncGPUModelRunnerOutput`, whose constructor immediately launches a non-blocking D2H copy **on a separate stream** and records an event; `get_output()` (called one step later by the executor's async-output thread, Part I §4.2) synchronizes the event, drops the GPU references, and only then builds the lists (`gpu_model_runner.py:203-270`).

The PP wrinkle: non-last ranks return `IntermediateTensors` from `execute_model` and `Worker.execute_model` ships them via `send_tensor_dict` to the next rank (`vllm/v1/worker/gpu_worker.py:645-677`); with async scheduling the last rank broadcasts sampled ids back to rank 0 over the GPU (`_pp_broadcast_prev_sampled_token_ids`, `gpu_model_runner.py:3671-3679, 3815-3853`) so the first-stage worker can build the next step's inputs without a scheduler round-trip.

## 11. Memory & cache subsystem: blocks, hashes, and the two block tables

Part I §3.5 covered this subsystem at the policy level (lazy LRU, touch/free, reverse-order frees, the `allocate_slots` admission signal). This section walks the actual data structures, the hash machinery, the scheduler→worker indirection tables, and the CPU offload tier.

### 11.1 The block pool: structs, free list, ref-counting

**`KVCacheBlock` — the page descriptor.** One Python dataclass per physical GPU block, created once at startup (`self.blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]`, `vllm/v1/core/block_pool.py:160-162`). The full struct (`vllm/v1/core/kv_cache_utils.py:108-126`):

```python
@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None
    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
    # Whether the block is a null block that should never be cached.
    is_null: bool = False
```

Three details beyond Part I:

- The free-list links are **intrusive** — they live inside the block struct, so queue manipulation allocates zero Python objects.
- `block_hash` is a guarded property: the setter asserts `self.block_hash is None, "The block already has a hash."` (`kv_cache_utils.py:132-137`) — a block can only acquire a hash once between evictions; `reset_hash()` (`:139-141`) is the only way to clear it.
- The null block (block 0, Part I) is special-cased on every mutation path (`touch` at `block_pool.py:383`, `free_blocks` at `:402`, `cache_full_blocks` at `:256-261`), with the comment "The ref_cnt of null_block is not maintained, needs special care to avoid freeing it" (`block_pool.py:171-175`). Null blocks are placeholders for token ranges whose KV is dead (outside a sliding window, pre-window Mamba state).

**`FreeKVCacheBlockQueue` — the free list / eviction queue.** A hand-rolled doubly-linked list rather than a `deque`, for one reason stated in the docstring: "to support removing a block in the middle of the queue in O(1) time" (`kv_cache_utils.py:157-164`) — required because a prefix-cache hit must yank an evictable block out of the middle of the LRU order (`touch`, §11.2). Sentinel head/tail nodes (`block_id=-1`) "reduce branching in the code" and guarantee every real queued block has both neighbors (`kv_cache_utils.py:189-207`). The five operations (`kv_cache_utils.py:209-345`): `popleft()` (raises `ValueError("No free blocks available")` on empty), `popleft_n(n)` (batched, one head-reconnect at the end — used by `get_new_blocks`), `remove(block)` (the O(1) middle removal), `append(block)` / `append_n(blocks)`. `num_free_blocks` is an eagerly-maintained counter, so the scheduler's capacity check (`get_num_free_blocks()`, `block_pool.py:459-465`) is O(1).

**Allocation IS eviction.** `BlockPool.get_new_blocks(num_blocks)` is the only allocation primitive (`block_pool.py:300-330`, verified):

```python
ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
if self.enable_caching:
    for block in ret:
        self._maybe_evict_cached_block(block)
        assert block.ref_cnt == 0
        block.ref_cnt += 1
```

A freed-but-cached block sits simultaneously in the free queue and in the prefix-cache map; only when it is physically reallocated does `_maybe_evict_cached_block` pop its hash from `cached_block_hash_to_block`, call `block.reset_hash()`, and emit a `BlockRemoved` event (`block_pool.py:332-370`). Note it deliberately does **not** check the cache for an existing block with the same content — "Note that we do not check block cache in this function" (`:303`); the cache lookup happened earlier in `get_computed_blocks`.

Ref-count transitions, complete list:

| Operation | ref_cnt | free-queue effect | source |
|---|---|---|---|
| `get_new_blocks` | 0 → 1 | `popleft_n` (already removed) | `block_pool.py:314-327` |
| `touch` (prefix hit pinned by a 2nd request) | +1 | if was 0 and not null: `remove()` from middle | `block_pool.py:372-387` |
| `free_blocks` | −1 | re-`append_n` blocks reaching 0 (tail of queue) | `block_pool.py:389-403` |

`touch`'s comment is the ref-protection contract in one line: "ref_cnt=0 means this block is in the free list (i.e. eviction candidate), so remove it" (`block_pool.py:380-384`).

**`BlockHashToBlockMap` — the cache index.** Not a radix tree: a flat dict keyed by `BlockHashWithGroupId` with a space-optimized value union (`block_pool.py:32-58`):

```python
self._cache: dict[
    BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
] = {}
```

"Mostly block_hash maps to a single KVCacheBlock... Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}" — the union exists "to reduce GC costs from the inner dict" (`block_pool.py:37-52`). Part I noted duplicates per hash are never de-duplicated (append-only block tables); `insert` promotes single→dict on first collision (`:73-89`); `pop(key, block_id)` removes one specific physical block and puts survivors back (`:91-119`).

### 11.2 Prefix cache internals

**The structure is a hash *chain*, not a radix tree.** Each block's hash commits to the entire prefix because the parent hash is folded in (`vllm/v1/core/kv_cache_utils.py:526-553`):

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

So "longest prefix match" degenerates to walking a list of hashes and probing a flat dict — no tree traversal. `NONE_HASH` (the chain seed) is `os.urandom(32)` unless `PYTHONHASHSEED` is set, in which case it's derived from the seed for cross-process reproducibility (`kv_cache_utils.py:90-105`). The default hash algorithm is `"sha256"` (`vllm/config/cache.py:78`); the keys are raw bytes given nominal typing via `BlockHash = NewType("BlockHash", bytes)` (`kv_cache_utils.py:35`), and the group id is packed into the same byte string — `block_hash + group_id.to_bytes(4, "big")` — "avoids creating tuples" (`kv_cache_utils.py:48-57`).

**Hash inputs beyond token ids**: `generate_block_hash_extra_keys` mixes in (a) multimodal feature identifiers overlapping the block's token range, (b) LoRA adapter name, (c) `cache_salt` (first block only — sufficient because of the chain), (d) raw prompt-embedding bytes (`kv_cache_utils.py:488-523`, mm-overlap walk at `:388-449`). This is what makes it safe for two requests with identical token ids but different images/LoRA to *not* share blocks.

**Who computes hashes, and when.** The `Request` object hashes itself incrementally. `EngineCore.__init__` builds one closure per engine: `self.request_block_hasher = get_request_block_hasher(scheduler_block_size, caching_hash_fn)` — created whenever prefix caching **or** a KV connector is enabled (`vllm/v1/engine/core.py:195-204`). The closure hashes only *new full blocks* since the last call, chaining from `request.block_hashes[-1]` (`kv_cache_utils.py:556-607`, early-out at `:568-570` "Early stop when there no new full blocks created"). It runs at request construction and again on every `append_output_token_ids` (`vllm/v1/request.py:165-170, 204-220`). The hasher is stored unbound — "Store the block hasher without binding self to avoid creating a reference cycle" (`request.py:166-169`).

**Lookup — per-attention-type `find_longest_cache_hit`.** Part I covered the full-attention scan; the per-type algorithms differ fundamentally. Three coordinator strategies (`vllm/v1/core/kv_cache_coordinator.py:547-591`): `NoPrefixCache` returns empty (`:291-299`), `Unitary` calls its single manager directly (`:349-365`), `Hybrid` runs a fixed-point loop:

- **Full attention — left-to-right, stop at first miss** (`vllm/v1/core/single_type_kv_cache_manager.py:435-445`; the chain property — "downward-closed" — is what makes the early break sound).
- **Sliding window — right-to-left, looking for a contiguous run** (`single_type_kv_cache_manager.py:493-549`): it only needs the last `cdiv(sliding_window - 1, block_size)` blocks to be cached; it scans from the rightmost block backwards, counts `num_contiguous_blocks`, and on success **truncates everything after the run and nulls everything before it** — the result shape is e.g. `[NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]` (docstring `:340-342`). Eagle spec-decode needs the last block recomputed, implemented by requiring one extra contiguous block then popping it (`:498-503, 556-561`).
- **Mamba — only the single last state block matters** (`single_type_kv_cache_manager.py:791-811`): right-to-left scan, take the **first** (rightmost) hit, pad everything before it with null blocks, `break  # we just need the last match - early stopping`.
- **Cross-attention — no caching at all**; `find_longest_cache_hit` raises (`single_type_kv_cache_manager.py:1034-1056`, rationale: encoder states are unique per request).
- **Hybrid models** (`HybridKVCacheCoordinator.find_longest_cache_hit`, `kv_cache_coordinator.py:453-544`): groups with identical specs are batched (`verify_and_split_kv_cache_groups`, `:410-451`), sorted full-attention-first because "its efficient left-to-right scan provides a tighter initial bound" (`:438-443`), and then the hit length is shrunk to a **fixed point**: "Each attention type either accepts the current candidate length or reduces it. If any type reduces the length, restart checks over all types. This converges because length monotonically decreases" (`:458-464`). Hit length must be a multiple of `lcm(block_sizes)` since partial-block hits aren't supported (`:445-451`). When groups have different block sizes, hashes computed at `hash_block_size` granularity are converted lazily by concatenating consecutive hashes (`BlockHashListWithBlockSize`, `kv_cache_utils.py:1607-1677`).

**Insertion.** After `allocate_slots` commits, `BlockPool.cache_full_blocks` stamps each newly-full block (`block_pool.py:256-272`):

```python
for i, blk in enumerate(new_full_blocks):
    if blk.is_null:
        continue
    assert blk.block_hash is None
    block_hash = new_block_hashes[i]
    block_hash_with_group_id = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
    blk.block_hash = block_hash_with_group_id
    self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
```

It never recomputes hashes — "The block hashes values are computed by the Request object immediately when it is created and when new tokens are appended" (`block_pool.py:223-225`). The per-request high-water mark `num_cached_block[req_id]` makes insertion idempotent and incremental (`single_type_kv_cache_manager.py:239-263`). Mamba adds a same-step poison check: a block cached by another request *in the current step* can't be trusted yet, so `get_num_blocks_to_allocate` returns `num_gpu_blocks + 1` to force the scheduler to defer the request one step (`single_type_kv_cache_manager.py:852-860`, cleared by `new_step_starts`, `:1006-1007`).

**Protection against eviction races.** There are no locks because there is no concurrency: all of `BlockPool` is mutated only from the engine-core busy loop thread (Part I §3.1). The races that *do* exist are logical, within one `schedule()` call, and are closed by ordering and accounting:

1. **Hit-then-evict race**: between `get_computed_blocks` (lookup) and `allocate_slots` (commit), the hit blocks may have `ref_cnt == 0` and thus still be in the free queue. Two mechanisms protect them:
   - The **capacity check counts them as allocations**. `get_num_blocks_to_allocate` adds `num_evictable_blocks = self._get_num_evictable_blocks(new_computed_blocks[...])` where evictable means `blk.ref_cnt == 0 and not blk.is_null` (`single_type_kv_cache_manager.py:73-75, 133-139`), with the comment: "If a computed block is an eviction candidate (in the free queue and ref_cnt == 0), it will be removed from the free queue when touched by the allocated request, so we must count it in the free-capacity check."
   - **Touch before allocate**. `allocate_slots` first runs `allocate_new_computed_blocks` (which `touch`es the hit blocks — removing them from the free queue) and only then `allocate_new_blocks` → `get_new_blocks` (`kv_cache_manager.py:340-358`); so a fresh allocation in the same call can never pop-and-evict a block the request just hit. The ordering comment: "Append the new computed blocks to the request blocks until now to avoid the case where the new blocks cannot be allocated" (`:344-345`).
2. **Stale-hash race**: a block evicted and reallocated keeps its old data until overwritten, but its `_block_hash` was reset at eviction (`block_pool.py:357`) so it is unreachable through the cache index; the `assert blk.block_hash is None` at re-caching (`:262`) enforces the lifecycle.

### 11.3 Request → token → physical block: the indirection tables

There are **two copies of the mapping**, one per process tier:

**Scheduler side (source of truth)**: `SingleTypeKVCacheManager.req_to_blocks: defaultdict[str, list[KVCacheBlock]]` — one ordered list of block objects per request per KV-cache group (`single_type_kv_cache_manager.py:59-62`). The scheduler itself never sees this; it gets the opaque `KVCacheBlocks` facade whose `get_block_ids()` flattens to `tuple[list[int], ...]` (outer = group, inner = block ids in token order) (`kv_cache_manager.py:21-91`). Those integer lists are what travel in `NewRequestData` / `CachedRequestData` to the workers.

**Worker side (GPU-consumable)**: `BlockTable` per group, owned by `InputBatch` via `MultiGroupBlockTable` (`vllm/v1/worker/gpu_input_batch.py:141`, `vllm/v1/worker/block_table.py:253-304`). The core arrays (`block_table.py:68-75`):

```python
self.block_table = self._make_buffer(
    self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
)
self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
self.slot_mapping = self._make_buffer(
    self.max_num_batched_tokens, dtype=torch.int64
)
```

— a dense 2D `[max_num_reqs, max_num_blocks_per_req]` int32 matrix kept as a pinned-CPU/GPU `CpuGpuBuffer` pair (§10.1); rows are appended in place (`append_row`, `:100-116`), moved/swapped when the persistent batch condenses (`:118-131`), and uploaded with `commit_block_table` → `copy_to_gpu` (`:193-194`).

The **two-level translation** per step:

1. *Request row update*: `GPUModelRunner._update_states` extends `req_state.block_ids` with the scheduler's new ids (or replaces them wholesale on resume-from-preemption) and calls `self.input_batch.block_table.append_row(new_block_ids, req_index)` (`vllm/v1/worker/gpu_model_runner.py:1062-1092`); new requests enter via `input_batch.add_request` → `block_table.add_row(request.block_ids, req_index)` (`gpu_input_batch.py:304, 342`).
2. *Token→slot flattening*: `compute_slot_mapping(req_indices, positions)` produces, for every scheduled token, the physical cache slot (`block_table.py:181-191`, verified):

```python
block_table_indices = (
    req_indices * self.max_num_blocks_per_req + positions // self.block_size
)
block_numbers = self.block_table.np.ravel()[block_table_indices]
block_offsets = positions % self.block_size
np.add(block_numbers * self.block_size, block_offsets,
       out=self.slot_mapping.np[: req_indices.shape[0]])
```

so `slot = block_table[req, pos // bs] * bs + pos % bs`, vectorized in numpy. Under decode/prefill context parallelism the same function uses a "virtual block" of `block_size * cp_world_size`, an interleave mask (`virtual_block_offsets // cp_kv_cache_interleave_size % world == rank`), and writes `-1` for non-local tokens (`block_table.py:142-179`). It is invoked from `_prepare_inputs` (`gpu_model_runner.py:1567`).

**Kernel-block splitting**: when the manager block size isn't supported by the attention kernel, each memory block is subdivided — `self.blocks_per_kv_block = block_size // kernel_block_size` and IDs are expanded as `kv_manager_block_ids.reshape(-1,1) * blocks_per_kv_block + arange` (e.g. manager blocks `[0,1,2]` → kernel blocks `[0,1,2,3,4,5]`) (`block_table.py:51-66, 203-231`). The scheduler never knows about kernel blocks.

**Physical memory itself**: one flat int8 `torch.zeros(kv_cache_tensor.size)` buffer per `KVCacheTensor`, possibly shared by multiple layers (`shared_by`), then reinterpreted per layer via `attn_backend.get_kv_cache_shape(kernel_num_blocks, kernel_block_size, num_kv_heads, head_size, ...)` permuted by `get_kv_cache_stride_order()` then inverse-permuted to keep the canonical view (`gpu_model_runner.py:5749-5779, 5833-5905`); Mamba layers instead carve strided state tensors out of the same raw buffer (`gpu_model_runner.py:5906-5930`). So a "block id" is literally an index into dim 0 (or the backend's block dim) of these reshaped views.

### 11.4 Cache invalidation paths and their guarantees

Part I introduced `BlockPool.reset_prefix_cache()` as the RLHF hook; the precise guarantee is all-or-nothing (`block_pool.py:424-457`):

```python
num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
if num_used_blocks != 1:  # The null block is always marked as used
    logger.warning("Failed to reset prefix cache because some blocks (%d) are not freed yet", ...)
    return False
# Remove all hashes so that no new blocks will hit.
self.cached_block_hash_to_block = BlockHashToBlockMap()
# Remove all hashes from all blocks.
for block in self.blocks:
    block.reset_hash()
```

Note what it does **not** do: it doesn't touch GPU memory or the free queue — stale KV bytes remain in VRAM but are unreachable because every hash is gone, and any request holding blocks would have made the reset refuse. It emits `AllBlocksCleared` for external subscribers (`:454-455`).

**`Scheduler.reset_prefix_cache(reset_running_requests=...)`** (`vllm/v1/core/sched/scheduler.py:1753-1797`) — the forced variant for weight updates with traffic in flight, new to this Part. It drains `self.running` through `_preempt_request` (freeing every block, ref counts → 0), zeroes async-scheduling placeholders and sets `discard_latest_async_tokens = True` ("we need to discard the latest output token on the fly to avoid a redundant repetitive output token"), and clears `prev_step_scheduled_req_ids` so the model runner flushes them from the persistent batch ("we must act as if these requests were not scheduled in the prior step"). After force-preemption a failed pool reset is a hard error (`raise RuntimeError(...)`, `:1786-1792`) — the only legitimate blocker is requests pinned by remote KV transfer. The preempted requests recompute from token 0 under the new weights, which is exactly the correctness guarantee RLHF needs. Weight updates for multimodal models additionally have `reset_encoder_cache` — "stale vision embeddings computed with old weights are not reused" (`vllm/v1/engine/core.py:596-606`).

**Targeted invalidation**: `BlockPool.evict_blocks(block_ids)` removes specific blocks from the hash map *without* freeing them ("blocks with ref_cnt > 0 are not freed from the block pool, only evicted from the prefix cache hash table", `block_pool.py:405-422`) — used when a KV connector reports blocks whose contents it clobbered.

**Sleep/wake (RLHF colocate)**: `Worker.sleep(level)` uses `CuMemAllocator.sleep(offload_tags=("weights",) if level == 1 else tuple())` — level 1 offloads weights to CPU and *discards* the kv_cache pool's physical pages; the KV cache was allocated inside `allocator.use_memory_pool(tag="kv_cache")` (`vllm/v1/worker/gpu_worker.py:123-145, 429-430`). `wake_up` re-maps and, for fp8 KV caches, re-initializes scaling factors (`:147-169`). After a wake the prefix cache contents are garbage, which is why the serving-layer contract pairs wake_up with the reset path above.

### 11.5 Second tier: the `kv_offload` CPU cache

This checkout has a full scheduler+worker CPU offload subsystem under `vllm/v1/kv_offload/`, integrated as a KV connector (`OffloadingConnector`, `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:116`), i.e. it rides the same `num_external_computed_tokens` path in `allocate_slots` as P/D disaggregation.

**Scheduler-side bookkeeping** is hash-addressed, mirroring the GPU prefix cache. The `OffloadingManager` ABC defines the primitives — `lookup` (longest offloaded prefix), `prepare_load`/`complete_load` (with eviction protection), `prepare_store`/`complete_store`, `touch` (`vllm/v1/kv_offload/abstract.py:69-163`; the module docstring at `:9-27` states "prepare_load() ... The given blocks will be protected from eviction"). The default `LRUOffloadingManager` is an `OrderedDict[BlockHash, BlockStatus]` (`vllm/v1/kv_offload/lru_manager.py:16-25`): `touch` = `move_to_end` (`:46-49`), eviction happens inside `prepare_store` by scanning from the LRU end for `ref_cnt == 0` victims and returning `None` if not enough can be evicted (`:57-109`). An ARC (Adaptive Replacement Cache: T1/T2 + B1/B2 ghost lists) manager is selectable via `eviction_policy: "arc"` (`vllm/v1/kv_offload/arc_manager.py:16-28`, wiring in `vllm/v1/kv_offload/cpu.py:60, 73-85`).

**The offloaded block descriptor is a ctypes struct**, not a dataclass — `BlockStatus(ctypes.Structure)` with `_fields_ = [("ref_cnt", ctypes.c_int32)]`, where `ref_cnt == -1` means "stored but not yet readable" and `is_ready = ref_cnt >= 0` (`vllm/v1/kv_offload/backend.py:11-34`, verified); `CPUBlockStatus` adds an `int64 block_id` (`vllm/v1/kv_offload/backends/cpu.py:12-17`). The CPU backend's allocator is a trivial bump-pointer + free list (`backends/cpu.py:20-54`).

**Granularity and capacity**: offloaded blocks are a multiple of GPU blocks (`block_size_factor = offloaded_block_size // gpu_block_size`, `offloading_connector.py:249-250`); CPU capacity is computed from `cpu_bytes_to_use / (page_size_bytes × num_kv_cache_tensors × world_size × factor)` (`cpu.py:25-52`). Offload-tier hashes **reuse the request's existing GPU-granularity hash chain** by striding it: `islice(req.block_hashes, factor - 1, ..., factor)` — i.e. the hash of every factor-th GPU block identifies the containing CPU block (`offloading_connector.py:271-282`).

**Lookup/load flow**: `get_num_new_matched_tokens` touches the LRU, then `manager.lookup(...)` beyond the GPU-computed prefix and reports hits as asynchronously-loadable external tokens (`offloading_connector.py:284-356`); returning `None` (lookup-retry or a hit block still mid-load) makes the scheduler skip the request this step (`:321-325, 341-355`, matching the ABC contract at `abstract.py:80-84`). After the scheduler allocates GPU blocks, `update_state_after_alloc` calls `manager.prepare_load(...)` (ref_cnt++ on the CPU blocks) and builds a `(CPULoadStoreSpec, GPULoadStoreSpec)` transfer of block-id arrays for the worker side (`offloading_connector.py:357-400`); actual copies are performed by `CpuGpuOffloadingHandlers` registered per medium pair (`cpu.py:88-109`).

There is no disk/third tier in this checkout — `vllm/v1/kv_offload/mediums.py` + `backends/` only define GPU and CPU media (`cpu.py:13-15` imports `CPULoadStoreSpec, GPULoadStoreSpec` only).

## 12. Extension surfaces: adding a model/reward/backend

vLLM has no "reward" concept; its extension surfaces are models, attention backends (§10.6), worker implementations, stat loggers, and plugins. For wm-infra the model surface is the analogue of registering a new model family, and the stat-logger/plugin surfaces are the analogue of reward/metric plugins.

### 12.1 The model registry: a dict of lazy import specs

The whole model zoo is a set of plain module-level dicts mapping HF architecture string → `(module_relname, class_name)`:

```python
_TEXT_GENERATION_MODELS = {
    ...
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
```
(`vllm/model_executor/models/registry.py:70`, llama entry at `:145`, verified). Category dicts (`_TEXT_GENERATION_MODELS`, `_EMBEDDING_MODELS`, `_MULTIMODAL_MODELS`, `_SPECULATIVE_DECODING_MODELS`, transformers-backend fallbacks…) are merged into `_VLLM_MODELS` (`registry.py:572-580`) and materialized into a singleton:

```python
ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
    }
)
```
(`registry.py:1216-1224`). Two registered-model flavors behind one ABC (`_BaseRegisteredModel`, `registry.py:659-666`): `_RegisteredModel` (already-imported class, `:669-689`) and `_LazyRegisteredModel` (`:692-798`), whose `load_model_cls` is just `importlib.import_module(self.module_name)` + `getattr` (`:796-798`).

Two engineering details worth noting:

1. **Capability inspection runs in a subprocess.** Resolving "what can this architecture do" must not import CUDA-touching model code into the front-end process. `_LazyRegisteredModel.inspect_model_cls` first tries a JSON cache under `VLLM_CACHE_ROOT/modelinfos`, keyed by a hash of the model's `.py` file (`registry.py:759-781`), and on miss: `# Performed in another process to avoid initializing CUDA` → `_run_in_subprocess(lambda: _ModelInfo.from_model_cls(self.load_model_cls()))` (`registry.py:782-785`, `_run_in_subprocess` at `:1229`, spawning `_SUBPROCESS_COMMAND = [sys.executable, "-m", "vllm.model_executor.models.registry"]`, `:586`). The result is a frozen `_ModelInfo` dataclass of capability booleans — `is_text_generation_model`, `supports_multimodal`, `supports_pp`, `has_inner_state`, `is_attention_free`, … (`registry.py:604-620`).
2. **Removed models fail with a version pointer**: `_PREVIOUSLY_SUPPORTED_MODELS = {"Phi3SmallForCausalLM": "0.9.2", ...}` and `_raise_for_unsupported` tells you the last vLLM version that had it (`registry.py:588-602, 882-905`).

**Out-of-tree registration** is the same dict: `ModelRegistry.register_model(arch, model_cls_or_str)` accepts a class or a lazy `"<module>:<class>"` string — the docstring recommends the string form precisely to avoid `RuntimeError: Cannot re-initialize CUDA in forked subprocess` (`registry.py:836-880`). Plugins do this at import time: every process calls `load_general_plugins()` which runs all `importlib.metadata.entry_points(group=...)` functions (`vllm/plugins/__init__.py:29-33, 68-80`); workers invoke it inside `WorkerWrapperBase.init_worker` before resolving the worker class (`vllm/v1/worker/worker_base.py:242-244`), and the CLI invokes it in `AsyncEngineArgs.add_cli_args` so plugins can even extend the argparse choices (`vllm/engine/arg_utils.py:2089-2096`).

### 12.2 Config detection: HF `architectures` → registry → runner type

`ModelConfig.__post_init__` loads the HF config and immediately resolves the architecture against the registry:

```python
hf_config = get_config(self.hf_config_path or self.model, self.trust_remote_code, ...)
...
model_info, arch = registry.inspect_model_cls(architectures, self)
self._model_info = model_info
self._architecture = arch
logger.info("Resolved architecture: %s", arch)
```
(`vllm/config/model.py:473-531`; `architectures` comes from a per-`model_type` convertor over `hf_config` — `get_model_arch_config` at `model.py:623-630`; `registry` property is the `ModelRegistry` singleton, `model.py:701-703`). The same block derives `runner_type` (generate vs pooling) and `convert_type` from the resolved `_ModelInfo`, raising if e.g. `--runner pooling` is requested for a model with no pooling support (`model.py:506-525`).

Resolution order in `_ModelRegistry.inspect_model_cls`/`resolve_model_cls` (the two are mirror images, `registry.py:1013-1063, 1065-1119`): explicit `model_impl="transformers"` → vLLM-native lookup per architecture (with `_normalize_arch`, `:987`) → fallback to the Transformers backend if `model_impl="auto"` and nothing matched. The loader logs the fallback: "has no vLLM implementation, falling back to Transformers implementation" (`vllm/model_executor/model_loader/utils.py:178-186`).

### 12.3 The model interface contract

The minimum contract is a `runtime_checkable` Protocol — notably Protocols *are* used here, unlike in the engine core (Part I §5 found zero `Protocol` in `vllm/v1/{core,engine,executor}`):

```python
@runtime_checkable
class VllmModel(Protocol[T_co]):
    """The interface required for all models in vLLM."""
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...
```
(`vllm/model_executor/models/interfaces_base.py:45-55`), checked structurally by `is_vllm_model` via `supports_kw(model_init, "vllm_config")` etc. (`interfaces_base.py:58-110`). Text-generation models additionally implement `compute_logits` (`VllmModelForTextGeneration`, `:112-116`). Optional capabilities are `ClassVar[Literal[True]]` tag Protocols in `interfaces.py` — e.g. `class SupportsLoRA(Protocol): supports_lora: ClassVar[Literal[True]] = True` (`vllm/model_executor/models/interfaces.py:515-518`) and `SupportsPP` (`:593-596`) — which the registry's `_ModelInfo.from_model_cls` sniffs without instantiating the model.

### 12.4 The loader pipeline

`GPUModelRunner.load_model` is the runtime consumer: `model_loader = get_model_loader(self.load_config); self.model = model_loader.load_model(...)` inside a `DeviceMemoryProfiler`, recording `self.model_memory_usage = m.consumed_memory` (`vllm/v1/worker/gpu_model_runner.py:4119-4146, 4140-4211`). `get_model_loader` is a format→loader table (`default`, `gguf`, `tensorizer`, `sharded_state`, `runai_streamer`, `dummy`, …; `vllm/model_executor/model_loader/__init__.py:118-123`). All loaders share the template method:

```python
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
    ...
    self.load_weights(model, model_config)        # loader-specific
    ...
    process_weights_after_loading(model, model_config, target_device)
return model.eval()
```
(`vllm/model_executor/model_loader/base_loader.py:42-76`). Three phases:

1. **`initialize_model`** introspects the constructor: if it takes `vllm_config` and `prefix`, it's a "new-style" model and is built under `set_current_vllm_config(...)`; otherwise a `DeprecationWarning` fires and vLLM *guesses* legacy kwargs (`config`, `quant_config`, `lora_config`…) (`vllm/model_executor/model_loader/utils.py:33-89`). This signature check is the de-facto enforcement of the `VllmModel` contract.
2. **`load_weights`** streams `(name, tensor)` pairs into the model's own `load_weights` method. `DefaultModelLoader.load_weights` then enforces completeness: `weights_not_loaded = weights_to_load - loaded_weights` → raise (`vllm/model_executor/model_loader/default_loader.py:279-306`).
3. **`process_weights_after_loading`** walks modules and calls `quant_method.process_weights_after_loading` (repack/quantize), then `Attention`/`MLAAttention` post-load hooks (`utils.py:92-115`) — quantization deliberately happens *after* raw weight load (`# Quantization does not happen in `load_weights` but after it`, `base_loader.py:59-61`).

After loading, the model gets wrapped for full CUDA graphs as described in §10.5 (`gpu_model_runner.py:4270-4290`).

### 12.5 Concrete trace: `LlamaForCausalLM`

- **Registry entry**: `"LlamaForCausalLM": ("llama", "LlamaForCausalLM")` (`registry.py:145`) → lazy import of `vllm.model_executor.models.llama`.
- **Class declaration** mixes in the capability tags and declares LoRA fusion metadata as class attributes:
  ```python
  class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3):
      packed_modules_mapping = {
          "qkv_proj": ["q_proj", "k_proj", "v_proj"],
          "gate_up_proj": ["gate_proj", "up_proj"],
      }
  ```
  (`vllm/model_executor/models/llama.py:506-512`). `packed_modules_mapping` tells the LoRA layer and the weight loader that three HF projections are fused into one vLLM `qkv_proj` parallel linear.
- **Constructor** is the new-style signature `def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", ...)`, reads `vllm_config.model_config.hf_config` / `vllm_config.quant_config`, and is PP-aware: only `get_pp_group().is_last_rank` builds `lm_head` + `LogitsProcessor`, other ranks get a `PPMissingLayer()` (`llama.py:520-556`).
- **Inner model is the compile boundary**: `@support_torch_compile(shape_invariants=llama_model_invariants)` decorates `LlamaModel`, not the ForCausalLM wrapper (`llama.py:349-354`), with the invariants function expressing `torch._check(positions.size()[0] == input_ids.size()[0])` for unbacked dynamic shapes (`llama.py:339-346`).
- **Contract methods**: `forward(input_ids, positions, intermediate_tensors, inputs_embeds)` returns hidden states or `IntermediateTensors` for PP (`llama.py:582-592`); `compute_logits(hidden_states)` delegates to the logits processor (`:594-599`); `load_weights` is one line of policy — `AutoWeightsLoader(self, skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None)).load_weights(weights)` (`:601-606`), while the inner `LlamaModel.load_weights` owns the `stacked_params_mapping` that folds `q_proj/k_proj/v_proj` shards into `qkv_proj` (`llama.py:441-476`).
- **Reuse pattern**: other architectures alias the same file — `"AquilaForCausalLM": ("llama", "LlamaForCausalLM")` (`registry.py:75`), and `as_seq_cls_model(LlamaForCausalLM)` / `as_embedding_model(...)` adapters derive pooling variants (`llama.py:613-621`).

So "add a model" = (1) write `vllm/model_executor/models/foo.py` with the 4-method contract and capability mixins, (2) add one line to the right `_*_MODELS` dict (plus `tests/models/registry.py`, per the module docstring at `registry.py:3-6`), or for out-of-tree: `ModelRegistry.register_model("FooForCausalLM", "your_pkg.foo:FooForCausalLM")` from a plugin entry point.

### 12.6 The config/args system

**Config classes: pydantic dataclasses with `extra="forbid"`.** Every config class uses a single decorator that wraps `pydantic.dataclasses.dataclass`:

```python
@dataclass_transform(field_specifiers=(PydanticField,))
def config(cls=None, *, config: ConfigDict | None = None, **kwargs):
    """Decorator to create a pydantic dataclass with default config. The default
    config for the dataclass forbids extra fields. All config classes in vLLM
    should use this decorator."""
    merged_config = ConfigDict(extra="forbid")
```
(`vllm/config/utils.py:37-67`, verified). So unknown keys are hard errors at construction, and field validation is pydantic-native (`@field_validator` e.g. for lower-casing `tokenizer_mode`, `vllm/config/model.py:640-645`). `VllmConfig` itself is one of these (`@config(config=ConfigDict(arbitrary_types_allowed=True))`, `vllm/config/vllm.py:203-204`), an aggregate of ~20 sub-configs each a `Field(default_factory=...)` (`vllm.py:209-277`), with three validation layers: per-field pydantic validators, a large `__post_init__` doing *cross-config* consistency (`model_config.verify_with_parallel_config`, deriving `quant_config`, async-scheduling × executor-backend compatibility checks that `raise ValueError`, `vllm.py:587-640`), and `@model_validator(mode="after")` for late checks (`validate_mamba_block_size`, `vllm.py:1474-1486`). It also carries `compute_hash()` — the torch.compile cache key over "all the configs that affect the structure of the computation graph" (`vllm.py:278-290`).

**CLI args are *generated* from the config dataclasses.** `EngineArgs` is a flat dataclass whose defaults are literally read off the config classes — `model: str = ModelConfig.model`, and nested/factory fields via `get_field(ModelConfig, "hf_overrides")` (`vllm/engine/arg_utils.py:351-380, 451`; `get_field` helper at `vllm/config/utils.py:69-75`). There is no hand-maintained argparse table. `_compute_kwargs(cls)` introspects each dataclass field — type hints (unwrapping `Annotated`/unions), pydantic `FieldInfo` defaults, and the field's *attribute docstring* as help text — and emits argparse kwargs: nested dataclass fields get a `TypeAdapter(cls).validate_json` parser ("Should either be a valid JSON string or JSON keys passed individually"), bools get `BooleanOptionalAction` (auto `--no-x`), `Literal`s become `choices`, `max_model_len` gets a human-readable-int parser (`arg_utils.py:236-336`). Help-text extraction is skipped unless `--help`/mkdocs is detected (`NEEDS_HELP`, `:229-234`), and results are `lru_cache`d (`get_kwargs`, `:338-349`). `EngineArgs.add_cli_args` then groups them per config class: `model_kwargs = get_kwargs(ModelConfig); model_group = parser.add_argument_group(title="ModelConfig", description=ModelConfig.__doc__)` (`:640-648`). Nested configs are also reachable as dotted shorthand, e.g. `-cc.mode=3` for `compilation_config` (documented on the field, `vllm/config/vllm.py:244-250`).

Round trip: `vllm serve` registers the parser via `make_arg_parser` (which delegates to `AsyncEngineArgs.add_cli_args`, `vllm/entrypoints/openai/cli_args.py:273-280`; `vllm/entrypoints/cli/serve.py:117-127`) → `AsyncEngineArgs.from_cli_args(args)` reconstructs the dataclass by field-name intersection (`arg_utils.py:1255-1262`) → `create_engine_config()` builds each sub-config and finally `config = VllmConfig(...)` (`arg_utils.py:1395-1463, 1809`). `AsyncEngineArgs` only adds front-end-ish flags like `--enable-log-requests` (`arg_utils.py:2083-2105`).

**Plumbing down: the whole `VllmConfig` is pickled into every process.** There is no per-field re-marshalling. The engine-core process gets it as a spawn kwarg: `common_kwargs = {"vllm_config": vllm_config, ...}` → `context.Process(target=..., kwargs=common_kwargs | {"dp_rank": ...})` (`vllm/v1/engine/utils.py:100-128`). Worker processes likewise: `process_kwargs = {"vllm_config": vllm_config, "local_rank": ..., "rank": ..., ...}` → `context.Process(target=WorkerProc.worker_main, ...)` (`vllm/v1/executor/multiproc_executor.py:603-622`). Inside the worker, `WorkerWrapperBase.init_worker` reads `vllm_config` from the kwargs, loads plugins, and resolves the worker class **by qualified name string** (`parallel_config.worker_cls` — passing a class object is explicitly rejected: "passing worker_cls is no longer supported", `vllm/v1/worker/worker_base.py:227-256`), so the config is also the extension point for swapping the worker implementation.

Late mutation is allowed at two defined points: the front-end handshake can override `ParallelConfig` fields on the engine (`for key, value in init_message.parallel_config.items(): setattr(parallel_config, key, value)`, `vllm/v1/engine/core.py:923-925`), and KV auto-fit can shrink `max_model_len` after profiling, synced to workers via `collective_rpc("update_max_model_len", ...)` because "workers were spawned before memory profiling and have the original (larger) max_model_len cached" (`core.py:262-269`).

### 12.7 Other pluggable surfaces, in one list

- **Attention backends** — per-layer selection with platform priority lists and `validate_configuration` self-vetting (§10.6); a forced-but-invalid backend raises instead of silently falling back (`vllm/platforms/cuda.py:306-380`).
- **Worker class** — `parallel_config.worker_cls` qualified-name string (§12.6).
- **Executor backend** — `Executor.get_class` accepts any qualname (Part I §4.1).
- **Stat loggers** — `AsyncLLM.__init__` accepts `custom_stat_loggers` factories (`vllm/v1/engine/async_llm.py:157-168`; see §14.1).
- **General plugins** — `importlib.metadata.entry_points` groups run in *every* process (`vllm/plugins/__init__.py:29-33, 68-80`).

## 13. Startup sequence & failure handling

### 13.1 Startup order of operations

For `vllm serve <model>` with the default mp executor, single API server (Part I covered steps 4–6 at the IPC level; the full ordering):

1. **CLI dispatch** — `ServeSubcommand.cmd` → `uvloop.run(run_server(args))` (`vllm/entrypoints/cli/serve.py:48-112`).
2. **Parse & validate config** — `AsyncEngineArgs.from_cli_args(args)` (`vllm/entrypoints/openai/api_server.py:88`) → `engine_args.create_engine_config(...)` builds `VllmConfig` incl. `ModelConfig.__post_init__`'s HF-config download + architecture resolution (`api_server.py:122`; `vllm/engine/arg_utils.py:1395-1463`; `vllm/config/model.py:473-531`).
3. **Front-end objects** — `AsyncLLM.from_vllm_config` picks the executor class (`Executor.get_class(vllm_config)`) (`api_server.py:137-145`; `vllm/v1/engine/async_llm.py:208-235`); `AsyncLLM.__init__` builds tokenizer/Input/Output processors then `EngineCoreClient.make_async_mp_client(...)` (`async_llm.py:147-155`).
4. **Spawn engine-core process(es)** — `MPClient.__init__` enters `launch_core_engines(...)` (`vllm/v1/engine/core_client.py:490-496`), which mints ZMQ addresses and has `CoreEngineProcManager` spawn `EngineCore_DP{i}` with the pickled `VllmConfig` (`vllm/v1/engine/utils.py:777-934, 100-128`).
5. **Engine process bootstrap** — `EngineCoreProc.run_engine_core` installs SIGTERM/SIGINT handlers, sets the process title, constructs `EngineCoreProc` (`vllm/v1/engine/core.py:930-997`). Its `__init__` first performs the HELLO→init-metadata handshake with the front-end (`core.py:688-718, 892-927`), then calls `super().__init__` = `EngineCore.__init__` (`core.py:739-746`).
6. **Executor + worker spawn** — `EngineCore.__init__`: `self.model_executor = executor_class(vllm_config)` (`core.py:106`) → `MultiprocExecutor._init_executor`: loopback `distributed_init_method`, shared-memory broadcast `MessageQueue`, then `WorkerProc.make_worker_process(...)` per rank and `WorkerProc.wait_for_ready(...)` on the ready pipes (`vllm/v1/executor/multiproc_executor.py:100-166, 586-626, 649-686`).
7. **Worker bootstrap (per `VllmWorker-{rank}` process)** — `worker_main` → `WorkerProc.__init__`: `wrapper.init_worker(...)` (resolve worker class, load plugins) → `self.worker.init_device()` → attach MQs → `self.worker.load_model()` (`multiproc_executor.py:696-774, 530-584`). `Worker.init_device` sets `cuda:{local_rank}`, **initializes NCCL before the memory snapshot** (`# Initialize the distributed environment BEFORE taking memory snapshot... This ensures NCCL buffers are allocated before we measure available memory`, `vllm/v1/worker/gpu_worker.py:229-239`), takes `MemorySnapshot`, and constructs `GPUModelRunner` (`gpu_worker.py:189-284`). `load_model` runs the §12.4 loader pipeline under an optional sleep-mode CuMem memory pool (`gpu_worker.py:288-295`; `gpu_model_runner.py:4119-4146`). The worker then sends `{"status": "READY", "handle": <response MQ handle>, ...}` up its pipe (`multiproc_executor.py:757-771`).
8. **KV-cache sizing by profiling** — back in `EngineCore.__init__`: `_initialize_kv_caches` (`core.py:113, 225-283`): `get_kv_cache_specs()` → `determine_available_memory()` RPC (each worker runs the §13.2 profiling pass) → `get_kv_cache_configs(...)` (possible `max_model_len` auto-fit sync, `core.py:259-269`).
9. **KV alloc + warmup + CUDA-graph capture** — `model_executor.initialize_from_config(kv_cache_configs)` = two collective RPCs: `initialize_from_config` (allocate KV tensors, init KV connector; `vllm/v1/worker/gpu_worker.py:415-433`) then `compile_or_warm_up_model` (`vllm/v1/executor/abstract.py:112-118`), which dummy-runs compile/warmup sizes, runs `kernel_warmup`, and `capture_model()` for CUDA graphs unless `enforce_eager` (`gpu_worker.py:435-535`).
10. **Scheduler + ancillary state** — `collective_rpc("initialize_cache", ...)`, `StructuredOutputManager`, scheduler construction with the now-final `kv_cache_config`, batch queue sizing, block hasher, then `freeze_gc_heap()` (`core.py:119-222`).
11. **READY handshake** — `_perform_handshakes` exits its context manager by sending `{"status": "READY", "num_gpu_blocks": ..., ...}` (`core.py:869-889`); the front-end's `wait_for_engine_startup` poller has been counting HELLO/READY messages and watching process sentinels the whole time (`vllm/v1/engine/utils.py:937-1010`); the engine enters `run_busy_loop()` (`core.py:998-999`).
12. **HTTP serving ready** — control returns through `build_async_engine_client`; `run_server_worker` awaits `engine_client.get_supported_tasks()` (a utility RPC — proof the loop is alive), builds the FastAPI app, `init_app_state`, then `serve_http(...)` (`vllm/entrypoints/openai/api_server.py:490-505`). The per-request `output_handler` task is created lazily on the first request (`async_llm.py:649-655, 662`).

### 13.2 Free VRAM → KV blocks: the profiling math

**Memory snapshot before anything else.** After NCCL init (step 7 above):

```python
self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
```
(`gpu_worker.py:252-253`). `request_memory = ceil(total_memory * gpu_memory_utilization)`, erroring if current free memory is already below it (`vllm/v1/worker/utils.py:92-113`). `MemorySnapshot.measure` decomposes usage into torch vs non-torch: `free/total` from `mem_get_info`, `torch_memory = memory_reserved()`, `non_torch_memory = (total - free) - torch_memory`, with `torch_peak` from `allocated_bytes.all.peak` (`vllm/utils/mem_utils.py:92-134`).

**Profiling run → available KV memory.** Each worker runs `profile_run()` — a worst-case `_dummy_run(self.max_num_tokens, is_profile=True)` plus, for MM models, a max-size encoder batch held in the encoder cache, plus a dummy sampler run (`gpu_model_runner.py:5109-5182`) — inside the `memory_profiling` context manager, which is documented with a worked 3-category example (other-processes / torch / non-torch) and computes:

```python
result.torch_peak_increase = diff_profile.torch_peak
result.non_torch_increase = diff_from_create.non_torch_memory
...
result.non_kv_cache_memory = (
    non_torch_memory + peak_activation_memory + result.weights_memory)
```
(`vllm/utils/mem_utils.py:196-281`). The worker's answer is simply:

```python
self.available_kv_cache_memory_bytes = (
    self.requested_memory - profile_result.non_kv_cache_memory)
```
(`gpu_worker.py:358-360`) — i.e. *budget minus (weights + peak activations + non-torch growth)*, with an explicit sanity assert that no co-tenant process freed memory mid-profile (`gpu_worker.py:348-356`). A user-set `kv_cache_memory_bytes` short-circuits the whole computation but still runs `profile_run` to trigger compilation (`gpu_worker.py:316-333`).

**Bytes → blocks → tensors.** Engine-side, the byte count is converted to a block count: `num_blocks = int(available_memory // page_size // num_layers)` where `num_layers` is the max group size for hybrid models — the layout comment shows how sliding-window layers share full-attention layers' per-layer tensors (`vllm/v1/core/kv_cache_utils.py:830-846, 1070-1143`). Back on the worker, `_allocate_kv_cache_tensors` allocates each `KVCacheTensor.size` as a flat **int8** buffer shared by its `shared_by` layers (`gpu_model_runner.py:5749-5779`), reshaped per backend as described in §11.3. Finally, because profiling never saw CUDA-graph memory, `compile_or_warm_up_model` logs a suggested `--kv-cache-memory` value computed from `weights + peak_activation + non_torch + cuda_graph_bytes + 150MiB buffer` (`gpu_worker.py:435-530`, the 150 MiB fudge at 489-491).

### 13.3 Client disconnect → request abort

HTTP layer: every OpenAI route is wrapped in `@with_cancellation` (e.g. `create_completion`, `vllm/entrypoints/openai/completion/api_router.py:44-46`), which races the handler task against a `listen_for_disconnect` task that awaits `request.receive()` until `"http.disconnect"`, then cancels the loser (`vllm/entrypoints/utils.py:42-99`). That cancellation lands in `AsyncLLM.generate`, whose except-ladder is the cleanup contract:

```python
# If the request is disconnected by the client, generate()
# is cancelled or the generator is garbage collected. So,
# we abort the request if we end up here.
except (asyncio.CancelledError, GeneratorExit):
    if q is not None:
        await self.abort(q.request_id, internal=True)
```
(`vllm/v1/engine/async_llm.py:597-606`, verified; the same `await self.abort(...)` runs for `InputStreamError` and generic `Exception`, `:621-643`, while `EngineDeadError` deliberately does **not** abort — "Engine is dead. Do not abort since we shut down", `:608-612`).

`AsyncLLM.abort` cleans both sides: `all_request_ids = self.output_processor.abort_requests(request_ids, internal)` then `await self.engine_core.abort_requests_async(all_request_ids)` (`async_llm.py:715-724`). The front-end half resolves external→internal ID fan-out (parallel sampling children), pops `request_states`, and pushes a final `FinishReason.ABORT` output into the per-request queue (`vllm/v1/engine/output_processor.py:452-503`). The engine half sends an `EngineCoreRequestType.ABORT` message (`core_client.py:975-977`), which the engine applies twice for latency: eagerly from the IO thread via `aborts_queue` mid-forward-pass (Part I §3.1), and authoritatively in `_handle_client_request` → `EngineCore.abort_requests` → `scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)` (`core.py:1084-1085, 321-327`).

`Scheduler.finish_requests` (whose docstring names exactly this scenario: "the API server can abort a request when the client disconnects") does a two-pass cleanup — batch-remove from `running`/`waiting`, then per request set the finished status and `_free_request`, with block-freeing deferred when a remote-KV transfer is still in flight (`vllm/v1/core/sched/scheduler.py:1671-1721`). `_free_request` frees encoder cache, records the ID in `finished_req_ids`, and `_free_blocks` returns KV blocks and deletes the `Request` object (`scheduler.py:1723-1745`) — the Part I flow-trace cleanup path. A second abort source is the front-end itself: stop-string matches detected during detokenization are batched into `processed_outputs.reqs_to_abort` and sent back from the output handler (`async_llm.py:691-695`).

### 13.4 Forward pass throws → request-level or engine-level failure

**Worker level.** `worker_busy_loop` catches everything per-RPC, attaches the traceback as a note, and ships the exception object as the response instead of crashing:

```python
except Exception as e:
    if hasattr(e, "add_note"):
        e.add_note(traceback.format_exc())
    logger.exception("WorkerProc hit an exception.")
    if output_rank is None or self.rank == output_rank:
        self.handle_output(e)
    continue
```
(`vllm/v1/executor/multiproc_executor.py:859-868`). `enqueue_output` converts it to `(ResponseStatus.FAILURE, str(output))` on the response MQ ("exception might not be serializable, so we convert it to string", `:814-827`).

**Executor level.** The executor-side `get_response` turns FAILURE into `RuntimeError("Worker failed with error '...', please check the stack trace above for the root cause")` (`multiproc_executor.py:345-362`), surfaced either synchronously or through the ordered `FutureWrapper` queue, whose `wait_for_response` does `set_exception(e)` (`:64-90`). If the worker *process* dies rather than raising, the monitor thread Part I mentioned sets `is_failed`, shuts down the executor, and fires the failure callback (`multiproc_executor.py:232-262`); subsequent RPCs short-circuit with `raise RuntimeError("Executor failed.")` (`:318-319`).

**Engine level.** The failure callback was registered at `EngineCore.__init__` as `executor_fail_callback = lambda: self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))` (`core.py:703-704, 107-108`); the busy loop dispatches it to `raise RuntimeError("Executor failed.")` (`core.py:1101-1102`). Any exception escaping the busy loop reaches `run_engine_core`'s catch-all: `logger.exception("EngineCore encountered a fatal error."); engine_core._send_engine_dead(); raise` with `engine_core.shutdown()` in the `finally` (`core.py:1004-1015`). `_send_engine_dead` enqueues the sentinel bytes `ENGINE_CORE_DEAD` and joins the output thread for up to 5s so the message actually leaves the process (`core.py:687, 1125-1137`).

**Client level.** The client's output path checks every batch of frames: `validate_alive(frames)` → if the single frame equals `ENGINE_CORE_DEAD`, set `engine_dead` and `raise EngineDeadError()` (`core_client.py:436-439`). A parallel sentinel-watcher thread `monitor_engine_cores` catches silent process death (`core_client.py:589-627`), and `_format_exception` rewrites any subsequent client error into `EngineDeadError` "so root cause is clear" (`:568-572`). The exception is pushed into `outputs_queue` (`:893-896`), re-raised in `AsyncLLM.output_handler`, which fans it out to **every** in-flight request: `logger.exception("AsyncLLM output_handler failed."); output_processor.propagate_error(e)` (`async_llm.py:709-711`), where `propagate_error` puts the exception on each request's `RequestOutputCollector` queue (`output_processor.py:445-450`) — so every open `generate()` call raises, and the API returns errors rather than hanging. Utility-RPC errors are scoped, not fatal: caught per-call and returned as `UtilityOutput.failure_message` (`core.py:1086-1100`).

**Process-tree hygiene.** Parent→child death propagation uses a dedicated `death_pipe` (child gets EOF when the parent exits, `multiproc_executor.py:601-602, 727-741`); child shutdown escalates politely — wait 4s, SIGTERM, wait 4s, SIGKILL (`_ensure_worker_termination`, `multiproc_executor.py:377-406`).

## 14. Engineering details worth copying (and anti-patterns)

### 14.1 Observability hooks (the model to copy wholesale)

**Stats pipeline: engine → msgpack → front-end loggers.** Per step, the scheduler attaches a `SchedulerStats` snapshot (running/waiting counts, `kv_cache_usage`, prefix-cache hit stats, spec-decode stats, KV-eviction events) to exactly one client's `EngineCoreOutputs` — even if no request produced tokens that step (`scheduler.py:1487-1499`, `make_stats` at `:1821-1859`, gated by `log_stats`). Token-level timing is computed front-end-side: `output_handler` creates one `IterationStats` per output batch (`async_llm.py:669-671`), the output processor fills it during detokenization, and `logger_manager.record(engine_idx=..., scheduler_stats=..., iteration_stats=..., mm_cache_stats=...)` publishes it (`async_llm.py:700-708`). `StatLoggerManager` multiplexes per-engine loggers (`vllm/v1/metrics/loggers.py:1235-1324`); `custom_stat_loggers` factories are a public extension point (`async_llm.py:157-168`). Two built-in sinks:

- **`LoggingStatLogger`** — the familiar periodic log line, assembled from parts: "Avg prompt throughput / Avg generation throughput / Running / Waiting [/ Preemptions] / GPU KV cache usage / Prefix cache hit rate", with `logger.debug` instead of `info` when idle ("Avoid log noise on an idle production system") and optional spec-decode/KV-connector/CUDA-graph/MFU sub-loggers (`loggers.py:95-275`).
- **`PrometheusStatLogger`** — declares the full `vllm:*` metric family (`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:engine_sleep_state`, `vllm:prompt_tokens_total`, TTFT/e2e histograms, …) labeled by `["model_name", "engine"]`, with deprecated metrics behind `show_hidden_metrics` (`loggers.py:389-470, 586-617`). The `/metrics` endpoint is mounted via `prometheus_client.make_asgi_app` plus a `prometheus_fastapi_instrumentator.Instrumentator` for HTTP-level metrics (`vllm/entrypoints/serve/instrumentator/metrics.py:18-44`).

**Profiling: a worker-side profiler driven by front-end RPC.** `ProfilerConfig` selects the backend at worker construction: `profiler == "torch"` → `TorchProfilerWrapper(profiler_config, worker_name=f"{vllm_config.instance_id}-rank-{self.rank}", activities=["CPU","CUDA"])`, `"cuda"` → `CudaProfilerWrapper` (`vllm/v1/worker/gpu_worker.py:104-119`). Control is an end-to-end RPC chain: `AsyncLLM.start_profile/stop_profile` → `engine_core.profile_async(True/False)` (plus an optional *front-end-process* `torch.profiler.profile` with its own tensorboard trace handler, `async_llm.py:187-205, 913-923`) → `EngineCore.profile` → `model_executor.profile(is_start)` (`core.py:571-572`) → `Worker.profile`, which errors helpfully if no profiler was configured (`gpu_worker.py:680-691`). Each iteration is annotated for trace readability — `annotate_profile` builds a string like `execute_context_3(8192)_generation_25(25)` from the scheduler output and wraps the step in it (`gpu_worker.py:573-597`).

**Tracing: span-per-startup-phase + span-per-request.** A backend-pluggable `@instrument` decorator (OTel today; no-op when unavailable) marks the startup hot spots from §13.1: `@instrument(span_name="Prepare model")` on `_initialize_kv_caches` (`core.py:225`), `"EngineCoreProc init"` (`core.py:688`), `"Executor init"` (`abstract.py:87`), `"Worker init"` (`multiproc_executor.py:529`), `"Warmup (GPU)"` (`gpu_worker.py:434`), `"Initialize model"` / `"Load model"` in the loader (`model_loader/utils.py:30`, `base_loader.py:41`). Machinery in `vllm/tracing/__init__.py:66-118` (`init_tracer`, `maybe_init_worker_tracer` — called per spawned process, `core.py:958-969`). Per-request OTLP spans are emitted by `OutputProcessor.do_tracing` at request finish, with gen-AI semconv attributes (`GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN`, `GEN_AI_LATENCY_E2E`, `GEN_AI_LATENCY_TIME_IN_QUEUE`, prefill/decode split) (`output_processor.py:714-751`), enabled by `ObservabilityConfig.otlp_traces_endpoint` (`vllm/config/observability.py:36`), which also gates opt-in metric families (`kv_cache_metrics`, `cudagraph_metrics`, `enable_mfu_metrics`, `observability.py:48-65`).

### 14.2 Details worth copying into vrl

Part I §8 covered the scheduler/IPC-level borrowings; Part II adds the executor/memory-level list:

1. **The `CpuGpuBuffer` triple (pinned CPU + GPU + numpy view) as the universal per-step tensor substrate** (`vllm/v1/utils.py:105-136`) — every hot-loop tensor in vrl's `VideoIterationRunner` that crosses CPU→GPU each step (timesteps, guidance scales, latents indices) should be a pre-allocated pair with `non_blocking=True` copies, not per-step `torch.tensor(...)` calls.
2. **Numpy-first CPU prep, single async H2D commit** (§10.3): all index math (`np.repeat` cumsum-aranges, slot mapping) happens in numpy views of pinned memory, then one `copy_to_gpu(n)` per buffer. Includes the micro-decision to start the largest copy first (`# OPTIMIZATION: Start copying the block table first.`, `gpu_model_runner.py:1472-1475`) and `torch.index_select`-over-`np.take` for big gathers (`:1505-1520`).
3. **CUDA-graph dispatch as a single-source-of-truth key table** (`cudagraph_dispatcher.py:13-30`): one dispatcher owns valid `(mode, BatchDescriptor)` keys; wrappers blindly trust the forward context; eager fall-through is always legal. For vrl's denoise loop, the `BatchDescriptor` analogue is `(num_tokens=latent_tokens, uniform=same_step_count)` — sequence/sigma values stay out of the key, handled by metadata tensors, exactly like vLLM excludes seq_lens.
4. **Capability negotiation by enum-min** (`AttentionCGSupport`, §10.5): backends declare what they can do, the runner takes the min across all of them and *downgrades with a logged warning* instead of failing or silently misbehaving. The same pattern fits vrl's model families declaring compile/CUDA-graph support levels.
5. **Allocation-is-eviction + intrusive free list** (§11.1): no eviction loop, no background GC thread — a freed cached block stays indexed until physically reallocated. A future vrl latent-block pool should copy `FreeKVCacheBlockQueue` verbatim (O(1) middle removal is what makes touch-on-hit possible).
6. **Hash chain instead of radix tree** (§11.2): parent-hash folding turns longest-prefix-match into a flat dict probe, and `extra_keys` (LoRA name, mm hashes, salt) solve cross-request safety without any tree structure. For vrl, conditioning embeddings hashed into the chain's first block = `cache_salt` precedent.
7. **CLI generated from config dataclasses** (§12.6): `EngineArgs` defaults read off `ModelConfig.model`, help text from attribute docstrings, parsers derived from type hints — zero duplicated argparse tables. Directly applicable to `vrl/config.py`; this is the derivation-over-duplication rule applied to a whole CLI.
8. **Exception-as-response RPC** (§13.4): worker exceptions become response payloads with `add_note(traceback)`, the process never dies on a request-level error, and `EngineDeadError` fan-out guarantees no caller ever hangs. vrl's engine↔gateway error path should adopt the trio: per-RPC catch, sentinel dead-message with a flush-join, `propagate_error` to every in-flight request.
9. **The profiling-based memory budget** (§13.2): `requested − (weights + peak_activation + non_torch)` with a worked example in the contextmanager docstring, a co-tenant assert, and a logged suggestion for the manual override flag. vrl's latent-cache sizing for video models (where activation peaks dwarf LLMs) needs exactly this discipline rather than hand-tuned fractions.
10. **`annotate_profile` per-iteration trace labels** (`gpu_worker.py:573-597`) and `@instrument` spans on every startup phase — trivial to add, transformative for reading traces of a multi-phase video pipeline.

### 14.3 Anti-patterns and sharp edges (documented, but don't copy blindly)

- **The O(max_num_reqs × max_model_len) token matrix** carries its own `# TODO(woosuk): This buffer could be too large if max_model_len is big.` (`gpu_input_batch.py:111-121`) — a known scalability tradeoff accepted for gather speed. For video (huge per-request state) the equivalent dense matrix would be unacceptable; keep per-request ragged storage.
- **Persistent-batch diffing degrades under low overlap**, and the code says so (`# If batches have low request overlap (e.g., alternating between two distinct sets of requests), this optimization becomes very inefficient.`, `gpu_model_runner.py:915-920`). vrl's resolution-homogeneous `PhaseGroupKey` batching can produce exactly this alternation — if a persistent batch is adopted, it must be per phase-group, not global.
- **The CUDA-graph replay contract is invisible at the call site**: correctness rests entirely on the discipline that all graph inputs are stable slices of `__init__`-time buffers, checked only by a debug-level data_ptr assert (`cuda_graph.py:296-305`). Copy the pattern only together with the banner-comment convention (`# Persistent buffers for CUDA graphs.`) that makes the contract greppable.
- **Sampling partial-prefill rows and discarding** (`gpu_model_runner.py:1591-1597`) plus generator-offset rewind (`gen.set_offset(gen.get_offset() - 4)`, `:2903-2910`) is a simplicity-over-purity tradeoff that costs sampler FLOPs and couples to cuRAND offset internals — fine at vocab-size logits, worth rethinking for video decoders where the "sampler" is a VAE.
- **Class-name string deduplication** for attention groups (`# Dedupe based on full class name; this is a bit safer than using the class itself as the key...`, `gpu_model_runner.py:5343-5390`) is a workaround for dynamically-created subclasses breaking identity — a smell that the backend-subclassing machinery leaks into consumers.
- **The 150 MiB fudge factor** in the suggested `--kv-cache-memory` value (`gpu_worker.py:489-491`) is honest about being a buffer, but it is still a magic number papering over unprofiled allocations (CUDA-graph pools) — the suggestion is advisory output, never silently applied, which is the right way to use a fudge.
- **`-1` placeholder tokens in async mode** (`gpu_model_runner.py:2944-2974`) leak a sentinel into `sampled_token_ids` lists that downstream code must know to skip — acceptable only because the scheduler's `num_output_placeholders` accounting (Part I §3.6) is the single consumer.

## 15. Part II source-of-truth index

| Claim | Source |
|---|---|
| **§10 executor internals** | |
| `CpuGpuBuffer` pinned-CPU/GPU/numpy triple, non-blocking copies (verified) | `vllm/v1/utils.py:105-141` |
| Persistent per-step buffers sized max_num_tokens/max_num_reqs | `vllm/v1/worker/gpu_model_runner.py:377-378, 564-590`; `_make_buffer` 794-804 |
| Cached `arange_np` optimization | `vllm/v1/worker/gpu_model_runner.py:629-634` |
| `InputBatch` SoA: token matrix (unpinned, TODO note), block table, sampling-param triples | `vllm/v1/worker/gpu_input_batch.py:111-151, 154-207` |
| `_update_states`: finished/unscheduled eviction, low-overlap warning (verified), resume path | `vllm/v1/worker/gpu_model_runner.py:874-921, 1063-1073` |
| condense → reorder → refresh epilogue; `condense()` hole-filling | `gpu_model_runner.py:1115-1119`; `gpu_input_batch.py:626-756` |
| `_get_cumsum_and_arange` vectorized batched arange | `gpu_model_runner.py:1268-1286` |
| `_prepare_inputs`: block-table copy first, positions np.add, index_select gather, qsl/seq_lens padding semantics, discard mask, logits_indices | `gpu_model_runner.py:1472-1475, 1485-1490, 1505-1520, 1573-1597, 1620-1626` |
| Async-scheduling `prev_sampled_token_ids` scatter into input_ids | `gpu_model_runner.py:1288-1409` (fast path 1361-1372, scatter 1374-1409) |
| `_preprocess` input modality choice (ids vs embeds, CUDA-graph rationale) | `gpu_model_runner.py:2735-2825` |
| execute_model/sample_tokens split via `ExecuteModelState`, return None | `gpu_model_runner.py:313-326, 3605-3618, 3621-3657` |
| `CUDAGraphMode` tuple encoding (decode/mixed) (verified) | `vllm/config/compilation.py:51-93` |
| FULL wrapper around whole model at load; PIECEWISE wrappers per compiled piece | `gpu_model_runner.py:4270-4290`; `vllm/compilation/backends.py:480-510` |
| `CUDAGraphWrapper`: trust-the-context dispatch (verified), capture into shared pool, weak-ref outputs, no internal buffers, debug addr check | `vllm/compilation/cuda_graph.py:139-310` |
| `BatchDescriptor` fields + `relax_for_mixed_batch_cudagraphs` | `vllm/forward_context.py:29-71` |
| Default capture-size generation `[1,2,4]+range(8,256,8)+range(256,max,16)`, max = min(2·max_num_seqs·q_len, 512) | `vllm/config/vllm.py:1171-1268`; doc at `vllm/config/compilation.py:571-586` |
| Dispatcher keys = source of truth; bs→padded table; skip conditions; FULL→relaxed→PIECEWISE→NONE order | `vllm/v1/cudagraph_dispatcher.py:13-30, 70-104, 259-302` |
| Padding/dispatch per batch incl. DP coordination & re-dispatch | `gpu_model_runner.py:3076-3199`; callsite 3398-3451 |
| KV-scale calc forces NONE; cascade/encoder disable FULL | `gpu_model_runner.py:3510-3516, 3132-3134` |
| Capture: gc.freeze, largest-first, warmup-then-capture, memory delta | `gpu_model_runner.py:5185-5331`; `cudagraph_dispatcher.py:305-321` |
| `_dummy_run` fake batch shapes; capture forces `max_seq_len = max_model_len` | `gpu_model_runner.py:4610-4706, 1705-1709` |
| `AttentionCGSupport` levels; mode downgrade negotiation | `vllm/v1/attention/backend.py:416-430`; `gpu_model_runner.py:5437-5592` |
| FA2 vs FA3 cudagraph support note | `vllm/v1/attention/backends/flash_attn.py:234-257` |
| Backend selected per Attention layer at model build | `vllm/model_executor/layers/attention/attention.py:266-276` |
| Selector: forced backend or platform priority list, `validate_configuration` vetting, layout override | `vllm/v1/attention/selector.py:46-119`; `vllm/platforms/cuda.py:45-82, 275-380`; ABC checks `vllm/v1/attention/backend.py:121-273` |
| AttentionGroup bucketing by (full_cls_name, spec); dedupe rationale | `gpu_model_runner.py:5333-5414` |
| `CommonAttentionMetadata` fields; per-group shallow copy; build cache via `update_block_table`; -1 block-table fill | `vllm/v1/attention/backend.py:287-333`; `gpu_model_runner.py:1744-1760, 1786-1852, 1733-1736` |
| FA build: AOT scheduler metadata, persistent buffer zeroing, max_num_splits bound | `vllm/v1/attention/backends/flash_attn.py:313-326, 329-512` (zeroing 482-490), `514-523` |
| Sampler pipeline (raw-logprobs-first NOTE, greedy/random merge, in-place temp) | `vllm/v1/sample/sampler.py:20-141, 147-205, 266-322` |
| Grammar bitmask applied in sample_tokens | `gpu_model_runner.py:3659-3663` |
| `_bookkeeping_sync`: discard+generator rewind, async −1 placeholders, worker-side token history NOTE | `gpu_model_runner.py:2881-2993` |
| `_to_list` pinned-buffer + event sync instead of tolist | `gpu_model_runner.py:6189-6203` |
| `AsyncGPUModelRunnerOutput` copy-stream + event, deferred get_output | `gpu_model_runner.py:203-270` |
| PP: IntermediateTensors send; async PP token broadcast | `vllm/v1/worker/gpu_worker.py:645-677`; `gpu_model_runner.py:3671-3679, 3815-3853` |
| **§11 memory & cache** | |
| `KVCacheBlock` struct, intrusive links, hash setter assert | `vllm/v1/core/kv_cache_utils.py:108-141` |
| Free queue: O(1) middle removal rationale, sentinels, popleft/popleft_n/remove/append_n | `vllm/v1/core/kv_cache_utils.py:157-345` |
| Null block special-casing on every mutation path | `vllm/v1/core/block_pool.py:171-175, 256-261, 383, 402` |
| `get_new_blocks` = pop + evict-on-reuse + ref_cnt 0→1 (verified) | `vllm/v1/core/block_pool.py:300-330` |
| `_maybe_evict_cached_block` (hash unmap + `BlockRemoved`) | `vllm/v1/core/block_pool.py:332-370` |
| `touch` / `free_blocks` ref-count transitions | `vllm/v1/core/block_pool.py:372-403` |
| `BlockHashToBlockMap` union value, GC-cost rationale | `vllm/v1/core/block_pool.py:32-119` |
| Chained hash function, `NONE_HASH` seed | `vllm/v1/core/kv_cache_utils.py:90-105, 526-553` |
| Extra hash keys: mm/LoRA/salt/prompt-embeds | `vllm/v1/core/kv_cache_utils.py:388-523` |
| Incremental request-side hashing, hasher creation in EngineCore | `vllm/v1/request.py:165-220`; `vllm/v1/engine/core.py:195-204` |
| Default hash algo sha256; `BlockHash` NewType + packed group id | `vllm/config/cache.py:78`; `kv_cache_utils.py:35, 48-57` |
| Full-attn forward scan / sliding-window backward run / Mamba last-match / cross-attn raises | `vllm/v1/core/single_type_kv_cache_manager.py:435-456, 493-562, 791-813, 1034-1056` |
| Hybrid fixed-point hit search, full-attn-first sort, lcm alignment | `vllm/v1/core/kv_cache_coordinator.py:410-544`; strategies `:291-299, 349-365, 547-591` |
| Lazy hash-granularity conversion | `vllm/v1/core/kv_cache_utils.py:1607-1677` |
| Insertion (`cache_full_blocks`), no recompute, idempotent high-water mark | `vllm/v1/core/block_pool.py:209-298`; `single_type_kv_cache_manager.py:239-263` |
| Hit-block eviction-race accounting + touch-before-allocate ordering | `vllm/v1/core/single_type_kv_cache_manager.py:73-75, 128-139`; `vllm/v1/core/kv_cache_manager.py:336-358` |
| Mamba same-step block poison check | `vllm/v1/core/single_type_kv_cache_manager.py:852-860, 995-1007` |
| `req_to_blocks` map; `KVCacheBlocks` facade / `get_block_ids` | `vllm/v1/core/single_type_kv_cache_manager.py:59-62`; `vllm/v1/core/kv_cache_manager.py:21-91` |
| Worker block table arrays, append/move/swap, GPU commit | `vllm/v1/worker/block_table.py:68-131, 193-197` |
| Slot mapping math (verified) incl. CP interleave | `vllm/v1/worker/block_table.py:133-191` |
| Kernel-block subdivision | `vllm/v1/worker/block_table.py:45-66, 203-231` |
| Block-id wiring into persistent batch | `vllm/v1/worker/gpu_model_runner.py:1062-1092`; `vllm/v1/worker/gpu_input_batch.py:141, 304-342` |
| Physical KV tensor allocation + per-backend reshape; Mamba strided states | `vllm/v1/worker/gpu_model_runner.py:5749-5779, 5833-5930` |
| `reset_prefix_cache` all-free precondition, hash wipe only | `vllm/v1/core/block_pool.py:424-457` |
| Forced reset: preempt-all, async-token discard, hard error | `vllm/v1/core/sched/scheduler.py:1753-1797` |
| Targeted `evict_blocks` for connectors | `vllm/v1/core/block_pool.py:405-422` |
| Encoder-cache reset on weight update | `vllm/v1/engine/core.py:589-606` |
| Sleep/wake: kv_cache memory pool, fp8 scale reinit | `vllm/v1/worker/gpu_worker.py:123-169, 429-430` |
| Offload manager primitives + eviction protection contract | `vllm/v1/kv_offload/abstract.py:9-27, 69-163` |
| LRU offload manager (OrderedDict, evict in prepare_store) | `vllm/v1/kv_offload/lru_manager.py:16-109` |
| ARC alternative + policy selection | `vllm/v1/kv_offload/arc_manager.py:16-28`; `vllm/v1/kv_offload/cpu.py:60-85` |
| ctypes `BlockStatus` (ref_cnt=-1 = not ready) (verified) | `vllm/v1/kv_offload/backend.py:11-34`; `vllm/v1/kv_offload/backends/cpu.py:12-54` |
| CPU capacity math, GPU/CPU media only | `vllm/v1/kv_offload/cpu.py:25-52, 13-15` |
| Connector lookup/load flow, strided hash reuse, delayed scheduling | `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:244-400` |
| **§12 extension surfaces** | |
| Registry tables; llama entry (verified); `_VLLM_MODELS` merge; singleton init | `vllm/model_executor/models/registry.py:70, 145, 572-580, 1216-1224` |
| `_ModelInfo` capability dataclass; lazy vs eager registered model | `vllm/model_executor/models/registry.py:604-657, 659-689, 692-798` |
| Subprocess inspection + modelinfo JSON cache | `vllm/model_executor/models/registry.py:758-792, 586, 1229` |
| `register_model` OOT API, lazy-string rationale; previously-supported error | `vllm/model_executor/models/registry.py:836-880, 588-602, 882-905` |
| Plugin loading (entry points), worker/CLI hooks | `vllm/plugins/__init__.py:29-33, 68-80`; `vllm/v1/worker/worker_base.py:242-244`; `vllm/engine/arg_utils.py:2089-2096` |
| HF config load → architecture resolution → runner/convert detection | `vllm/config/model.py:473-531, 623-630, 701-707` |
| `inspect_model_cls`/`resolve_model_cls` resolution order, transformers fallback | `vllm/model_executor/models/registry.py:1013-1119`; `vllm/model_executor/model_loader/utils.py:166-199` |
| `VllmModel` protocol contract; structural checks | `vllm/model_executor/models/interfaces_base.py:45-116` |
| Capability tag protocols (`SupportsLoRA`, `SupportsPP`) | `vllm/model_executor/models/interfaces.py:515-518, 593-596` |
| Loader template method (init → load_weights → post-process); quantize-after comment; completeness check | `vllm/model_executor/model_loader/base_loader.py:42-76`; `model_loader/utils.py:33-115`; `model_loader/__init__.py:118-138`; `default_loader.py:279-306` |
| Runtime call site `GPUModelRunner.load_model` + memory accounting | `vllm/v1/worker/gpu_model_runner.py:4119-4146, 4140-4211` |
| Llama trace: class/mapping/init/PP-aware lm_head/forward/compute_logits/load_weights/compile boundary | `vllm/model_executor/models/llama.py:506-606, 339-354, 441-476` |
| `@config` pydantic dataclass, `extra="forbid"` (verified) | `vllm/config/utils.py:37-67` |
| `VllmConfig` aggregate + `__post_init__` cross-validation + model_validator + compute_hash | `vllm/config/vllm.py:203-290, 587-640, 1474-1486` |
| EngineArgs mirrors config defaults; argparse generation from dataclass introspection | `vllm/engine/arg_utils.py:351-380, 236-349, 640-648` |
| `from_cli_args` / `create_engine_config` / `VllmConfig(...)` | `vllm/engine/arg_utils.py:1255-1262, 1395-1463, 1809` |
| Config pickled into engine/worker spawn kwargs | `vllm/v1/engine/utils.py:100-128`; `vllm/v1/executor/multiproc_executor.py:548-560, 603-622` |
| Handshake parallel-config override; max_model_len auto-fit sync | `vllm/v1/engine/core.py:923-925, 259-269` |
| `init_worker` resolves worker_cls by qualname (string-only) | `vllm/v1/worker/worker_base.py:227-256` |
| **§13 startup & failure** | |
| Startup steps 1-4 (CLI → config → AsyncLLM → spawn) | `vllm/entrypoints/cli/serve.py:48-127`; `vllm/entrypoints/openai/api_server.py:88-145, 490-505`; `vllm/v1/engine/async_llm.py:147-155, 208-235`; `vllm/v1/engine/core_client.py:490-496` |
| Startup steps 5-7 (engine proc, executor, worker init/load, READY pipe) | `vllm/v1/engine/core.py:930-999, 688-746, 892-927, 106`; `vllm/v1/executor/multiproc_executor.py:100-166, 530-584, 649-686, 696-774` |
| init_device: NCCL-before-snapshot comment, MemorySnapshot, runner ctor | `vllm/v1/worker/gpu_worker.py:189-284` (NCCL comment 229-239, snapshot 252-253) |
| `request_memory` = ceil(total × utilization); `MemorySnapshot` torch/non-torch decomposition | `vllm/v1/worker/utils.py:92-113`; `vllm/utils/mem_utils.py:67-134` |
| `memory_profiling` 3-category accounting; `non_kv_cache_memory` formula | `vllm/utils/mem_utils.py:196-281` |
| `available_kv = requested − non_kv_cache`; co-tenant assert; manual override path | `vllm/v1/worker/gpu_worker.py:303-362` (assert 348-356, override 316-333) |
| `profile_run` worst-case dummy batch + encoder + sampler | `gpu_model_runner.py:5109-5182` |
| bytes → blocks (`// page_size // num_layers`), hybrid sharing layout | `vllm/v1/core/kv_cache_utils.py:830-846, 1070-1143`; engine call `vllm/v1/engine/core.py:226-283` |
| Warmup: compile sizes, capture, `--kv-cache-memory` suggestion (+150MiB) | `vllm/v1/worker/gpu_worker.py:435-530`; `vllm/v1/executor/abstract.py:112-118` |
| READY msg with num_gpu_blocks; front-end startup poller | `vllm/v1/engine/core.py:869-889`; `vllm/v1/engine/utils.py:937-1010` |
| `with_cancellation` disconnect race | `vllm/entrypoints/utils.py:42-99`; `vllm/entrypoints/openai/completion/api_router.py:44-46` |
| generate() except-ladder (verified snippet); abort fan-out | `vllm/v1/engine/async_llm.py:570-648, 715-728` |
| OutputProcessor.abort_requests / propagate_error | `vllm/v1/engine/output_processor.py:445-503` |
| Engine-side abort → `finish_requests` two-pass cleanup → `_free_request` | `vllm/v1/engine/core.py:321-327, 1084-1085`; `vllm/v1/core/sched/scheduler.py:1671-1745` |
| Worker exception → FAILURE response; string conversion rationale | `vllm/v1/executor/multiproc_executor.py:845-871, 814-827` |
| Executor FAILURE → RuntimeError; FutureWrapper set_exception; is_failed gate | `vllm/v1/executor/multiproc_executor.py:303-362, 64-90, 232-268, 318-319` |
| EXECUTOR_FAILED input; fatal path `_send_engine_dead` | `vllm/v1/engine/core.py:703-704, 1101-1102, 1004-1015, 1125-1137` |
| Client dead-engine detection, monitor, error fan-out to all requests | `vllm/v1/engine/core_client.py:436-439, 568-627, 893-896`; `vllm/v1/engine/async_llm.py:709-711` |
| Worker termination escalation; death_pipe | `vllm/v1/executor/multiproc_executor.py:377-406, 601-602, 727-741` |
| **§14 observability** | |
| SchedulerStats attach + make_stats; IterationStats in output_handler; StatLoggerManager; custom loggers | `vllm/v1/core/sched/scheduler.py:1487-1499, 1821-1859`; `vllm/v1/engine/async_llm.py:157-168, 669-708`; `vllm/v1/metrics/loggers.py:1235-1324` |
| LoggingStatLogger / PrometheusStatLogger; /metrics mount | `vllm/v1/metrics/loggers.py:95-275, 389-470, 586-617`; `vllm/entrypoints/serve/instrumentator/metrics.py:18-44` |
| Profiler wrappers, RPC chain, iteration annotation | `vllm/v1/worker/gpu_worker.py:104-119, 573-597, 680-691`; `vllm/v1/engine/async_llm.py:187-205, 913-923`; `vllm/v1/engine/core.py:571-572` |
| `@instrument` tracing + per-request OTLP spans + config gates | `vllm/tracing/__init__.py:66-118`; `vllm/v1/engine/output_processor.py:714-751`; `vllm/config/observability.py:36-65`; span sites `core.py:225,688`, `abstract.py:87`, `multiproc_executor.py:529`, `gpu_worker.py:434`, `model_loader/utils.py:30`, `base_loader.py:41` |
