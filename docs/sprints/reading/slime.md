# slime — Architecture Reading

Repo: `/home/mingfeiguo/Desktop/slime` (Megatron + SGLang RL post-training framework). All `file:line` references are relative to that repo root and were spot-checked against the source.

## 1. Repo layout & module organization

Top level: the pip package `slime/`, a plugin namespace `slime_plugins/`, two driver entrypoints `train.py` / `train_async.py`, plus `scripts/`, `tests/`, `tools/` (ckpt converters, profilers), `docs/`, `examples/`, `docker/`. `pyproject.toml` declares `slime` and `slime_plugins` as the first-party packages (`pyproject.toml:17-19`: `known_first_party = ["slime", "slime_plugins"]`).

The `slime/` package has four subpackages plus utils, each with a clear ownership boundary:

| Path | Owns |
|---|---|
| `slime/ray/` | All Ray orchestration: `placement_group.py` (GPU allocation + driver-side factory functions), `actor_group.py` (`RayTrainGroup`), `rollout.py` (`RolloutManager` actor + `RolloutServer`/`ServerGroup`), `train_actor.py` (`TrainRayActor` base), `ray_actor.py` (tiny shared base), `utils.py` (`Lock` actor) |
| `slime/backends/megatron_utils/` | Training backend: `actor.py` (`MegatronTrainRayActor`), `model.py`/`loss.py`/`data.py`, `update_weight/` (weight-sync implementations + Megatron→HF weight iterators), `megatron_to_hf/` |
| `slime/backends/sglang_utils/` | Inference backend: `sglang_engine.py` (`SGLangEngine` actor wrapping an SGLang HTTP server subprocess), `sglang_config.py` (multi-model / PD-disaggregation server-group config) |
| `slime/rollout/` | Data-generation logic: `sglang_rollout.py` (asyncio generate loop), `data_source.py` (dataset + replay buffer), `rm_hub/` (reward models), `filter_hub/` (dynamic sampling filters), plus SFT/OPD variants |
| `slime/utils/` | Cross-cutting: `arguments.py`, `http_utils.py`, `health_monitor.py`, `tensor_backper.py`, `seqlen_balancing.py`, `reloadable_process_group.py`, etc. |

`slime_plugins/` holds optional integrations: `mbridge`, `megatron_bridge`, `models`, `rollout_buffer` (a standalone FastAPI buffer service for agentic/external generation).

Notably, rollout *policy* (what to generate) is injected by dotted path, not hard-wired: `RolloutManager.__init__` does `self.generate_rollout = load_function(self.args.rollout_function_path)` and `data_source_cls = load_function(self.args.data_source_path)` (`slime/ray/rollout.py:359-363`), so `slime/rollout/sglang_rollout.py:generate_rollout` is just the default plugin. The split is deliberate: `slime/ray/` knows nothing about Megatron math; the only backend coupling is `actor_group.py:79-81` importing `MegatronTrainRayActor` as the actor implementation class.

## 2. Architecture overview

slime is a **Ray-driven synchronous (or one-step-async) train↔rollout loop**. One driver process runs `train()`; everything else is Ray actors. There is **no centralized request scheduler of slime's own** — token-level batching/scheduling (continuous batching, paged KV, radix prefix cache) is delegated entirely to the SGLang servers behind an `sglang_router`; slime schedules at the level of *HTTP requests per sample* and *Ray RPCs per step*. The full per-step control flow is in `train.py:67-99`:

```python
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())
...
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
...
if args.offload_rollout:
    ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
```

Component/process topology:

```
 driver (train.py)
   │  ray.remote calls
   ├──────────────────────────────────────────────────────────────┐
   ▼                                                              ▼
 RolloutManager (CPU Ray actor)                       RayTrainGroup (driver-side handle list)
   │  owns: data_source (dataset+buffer),               │  one MegatronTrainRayActor per GPU
   │  generate_rollout fn, Lock actor,                  │  (world_size = nodes × gpus/node)
   │  RolloutHealthMonitor threads                      │  torch.distributed NCCL group inside
   │                                                    │
   │ HTTP POST /generate (asyncio, via router)          │
   ▼                                                    │
 sglang_router (daemon multiprocessing.Process          │
   on the RolloutManager node)                          │
   │ load-balances over registered workers              │
   ▼                                                    │
 SGLangEngine Ray actors (one per engine)               │
   └─ each spawns an SGLang HTTP server as a            │
      multiprocessing child process (spawn)             │
            ▲                                           │
            └────────── weight sync ────────────────────┘
        colocated: CUDA-IPC handles via Ray RPC (UpdateWeightFromTensor)
        disaggregated: dedicated NCCL broadcast groups (UpdateWeightFromDistributed)

 data path back:  samples → RolloutManager._convert_samples_to_train_data
                  → split by DP rank → ray.put → Box(ObjectRef) list
                  → each train actor ray.get()s its own DP shard
```

Key connections, with evidence:

- **Generation** is plain HTTP through the router: `url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"; ... output = await post(url, payload)` (`slime/rollout/sglang_rollout.py:158,201`). Concurrency is an asyncio semaphore sized `sglang_server_concurrency × num_engines` (`sglang_rollout.py:94-96`); prompts are submitted as one task per *group* (`n_samples_per_prompt` samples sharing a prompt, `sglang_rollout.py:136-149`), with dynamic over-sampling + filtering until `rollout_batch_size` valid groups exist, then leftover in-flight requests are aborted and (under `partial_rollout`) the partial samples are pushed back into the buffer (`sglang_rollout.py:422-461`, `345-386`, `620-624`).
- **Rollout→train data hand-off** is via the Ray object store: `RolloutManager.generate` converts samples to a train dict, then `_split_train_data_by_dp` builds one dict per DP rank and returns `Box(ray.put(rollout_data))` refs (`slime/ray/rollout.py:478-491`, `754-805`). On the training side each actor fetches only its shard: `rollout_data = ray.get(rollout_data_ref[dp_rank].inner)` (`slime/utils/data.py:299-301`). The DP partition can be sequence-length balanced (`rollout.py:764-767` calling `get_seqlen_balanced_partitions`).
- **Training step**: `MegatronTrainRayActor.train` wakes from offload, fetches its shard, optionally runs ref/teacher/actor log-prob forward passes by hot-swapping CPU-backed weight snapshots (`self._switch_model("ref")` etc., backed by `TensorBackuper`), computes advantages, then runs Megatron `train()` (`slime/backends/megatron_utils/actor.py:355-374`, `401-509`, `95-105`).
- **Colocation memory juggling**: with `--offload-train`, train actors pause/resume their entire CUDA allocation via `torch_memory_saver.pause()/resume()` plus destroying/reloading NCCL process groups (`actor.py:156-177`); rollout engines symmetrically release/resume sglang memory by tag (WEIGHTS vs KV_CACHE+CUDA_GRAPH) (`slime/ray/rollout.py:326-346`).
- **Async variant**: `train_async.py` pipelines exactly one rollout ahead — it launches `generate.remote(rollout_id + 1)` before `ray.get`-ing training on rollout `rollout_id`, and forces a sync before weight updates "to prevent update weight in the middle of generation" (`train_async.py:35-43`, `69-73`). Colocation is asserted off (`train_async.py:11`: `assert not args.colocate, "Colocation is not supported for async training."`).
- **Fault tolerance**: a `RolloutHealthMonitor` (plain `threading.Thread` inside the `RolloutManager` actor, `slime/utils/health_monitor.py:10-21,52`) health-checks engines; `update_weights` on rank 0 first calls `rollout_manager.recover_updatable_engines.remote()` to restart dead engines, and reconnects sync groups when `num_new_engines > 0` (`actor.py:543-564`, `rollout.py:527-548`, `RolloutServer.recover` at `rollout.py:269-310`).

## 3. Core scheduling & orchestration

### 3.1 Top-level structure: a synchronous driver loop over Ray actors

The unit of scheduling is one *rollout step* (`rollout_id`). The driver (`train.py`) is a plain Python loop on the Ray driver process that alternates between two actor groups:

- **`RolloutManager`** (CPU-only Ray actor, `slime/ray/rollout.py:349-351`) — owns the SGLang server fleet + router, the prompt `DataSource`, and the sample→train-data conversion.
- **`RayTrainGroup`** (`slime/ray/actor_group.py:10`) — one `MegatronTrainRayActor` per training GPU.

The main loop, `train.py:66-99`:

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    ...
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
    ...
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
    ...
    offload_train(actor_trains_this_step)
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
```

So one step = **generate → (offload rollout GPUs) → train → (offload train / onload rollout weights) → update_weights → onload KV**. Every phase is a blocking `ray.get`, i.e. the sync trainer is strictly phase-serialized — that is precisely what makes GPU colocation safe (rollout and Megatron time-share the same GPUs; placement is built that way in `slime/ray/placement_group.py:89-94`, where `--colocate` makes `num_gpus = actor GPUs` and `rollout_offset = 0`, vs. disaggregated where rollout bundles start after actor bundles, `placement_group.py:93-94`).

`train_async.py:35-43` is the one-step-pipelined variant (disaggregated only — `assert not args.colocate`, `train_async.py:11`): it launches `generate(rollout_id+1)` **before** training on step `rollout_id`, overlapping rollout with training:

```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
```

and before each weight sync it drains the in-flight generation first ("sync generate before update weights to prevent update weight in the middle of generation", `train_async.py:69-73`). Weight staleness is bounded by `--update-weights-interval` (`train_async.py:69`).

### 3.2 The rollout scheduling loop (the real event loop)

`RolloutManager.generate` (`slime/ray/rollout.py:478-491`) calls the configurable rollout function (default `slime/rollout/sglang_rollout.py:generate_rollout`, loaded via `load_function(self.args.rollout_function_path)` at `rollout.py:362`), which runs an **asyncio event loop inside the RolloutManager actor** (`sglang_rollout.py:621`: `run(generate_rollout_async(...))`).

The core loop, `slime/rollout/sglang_rollout.py:422-452`, is an **over-sample / first-completed-wins / dynamic-filter** scheduler:

```python
while len(data) < target_data_size:
    while state.remaining_batch_size < target_data_size:
        # get samples from the buffer and submit the generation requests.
        samples = data_source(args.over_sampling_batch_size)
        state.submit_generate_tasks(samples)

    # wait for the generation to finish
    done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        group: list[Sample] = task.result()
        ...
        dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
        if not dynamic_filter_output.keep:
            metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
            state.remaining_batch_size -= 1
            continue
        if len(data) < target_data_size:
            data.append(group)
```

Line-by-line semantics:

1. **Target & granularity** — `target_data_size = args.rollout_batch_size` (`sglang_rollout.py:416`), counted in *prompt groups* (each group = `n_samples_per_prompt` responses to one prompt, asserted at `sglang_rollout.py:440`). The submission granularity is `--over-sampling-batch-size`, which defaults to `rollout_batch_size` (`slime/utils/arguments.py:1782-1783`) and must be ≥ it (`arguments.py:1785-1787`). With a dynamic filter (e.g. GRPO zero-std group dropping), each drop decrements `remaining_batch_size` (`sglang_rollout.py:445`), which re-triggers the inner `while` to submit another over-sampling batch — i.e. **demand-driven over-provisioning**.
2. **Task tree** — `submit_generate_tasks` creates one asyncio task per *group* (`GenerateState.submit_generate_tasks`, `sglang_rollout.py:136-149`); `generate_and_rm_group` fans out one task per *sample* (`sglang_rollout.py:323-333`), then optionally runs a group-level reward model (`sglang_rollout.py:336-341`).
3. **Admission control / queueing** — there is no explicit queue; backpressure is a single semaphore sized to the whole fleet, `sglang_rollout.py:94-96`:
   ```python
   self.semaphore = asyncio.Semaphore(
       args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
   )
   ```
   (`sglang_server_concurrency` defaults to 512 per engine, `slime/backends/sglang_utils/arguments.py:39`). Each per-sample `generate_and_rm` holds the semaphore only for the generation HTTP call (`sglang_rollout.py:260-277`); reward-model HTTP calls happen outside it. Excess tasks just wait on the semaphore inside the actor's event loop — the queue *is* the set of pending coroutines (`state.pendings`).
4. **Load balancing** — two layers: the `sglang_router` process started per model (`rollout.py:911-962`, slime disables router health checks at `rollout.py:948`), plus an in-client least-loaded DP-rank picker `dp_rank_context` (`sglang_rollout.py:119-129`) and optional consistent-hashing by `session_id` header (`sglang_rollout.py:196-198`).
5. **Priority** — none. FIFO submission, completion-order collection (`asyncio.wait(..., FIRST_COMPLETED)`), and `data` is later re-sorted by sample index for determinism (`sglang_rollout.py:464`).
6. **Preemption (the long-tail killer)** — once `target_data_size` groups are collected, all still-running requests are **aborted**: `abort()` (`sglang_rollout.py:345-386`) sets `state.aborted = True`, queries the router for all worker URLs, and posts `/abort_request {"abort_all": True}` to every SGLang server (`sglang_rollout.py:360`). Tasks that were still waiting on the semaphore see `state.aborted` and return `Sample.Status.ABORTED` (`sglang_rollout.py:261-263`). With `--partial-rollout`, partially generated samples are *not discarded*: the abort drain collects them (`sglang_rollout.py:374-381`), and `generate_rollout` pushes them into the buffered data source (`sglang_rollout.py:622-623` → `RolloutDataSourceWithBuffer.add_samples`, `slime/rollout/data_source.py:198-211`). On the next rollout, `get_samples` pops the buffer first before touching the dataset (`data_source.py:177-189`, default `pop_first` policy at `data_source.py:225-229`), and `generate()` resumes from the existing tokens because a `COMPLETED/TRUNCATED` sample short-circuits (`sglang_rollout.py:251-255`) while an `ABORTED` one re-enters generation with its accumulated `sample.tokens` as the prompt (`sglang_rollout.py:191-192`, `211-218`); off-policy prefix tokens can be loss-masked via `mask_offpolicy_in_partial_rollout` (`sglang_rollout.py:247-248`).
7. **Batch trimming** — back in `RolloutManager._get_rollout_data` (`rollout.py:590-605`), the flattened sample list is trimmed to a multiple of `global_batch_size` (or a `dynamic_global_batch_size` is computed to force exactly one optimizer step, `rollout.py:609-634`).

### 3.3 Per-request path into SGLang

A single sample is one HTTP `POST /generate` to the router with token IDs and `return_logprob: True` (`sglang_rollout.py:173-201`); the response's `output_token_logprobs` is appended to `sample.tokens` / `sample.rollout_log_probs` (`sglang_rollout.py:204-222`) — no re-tokenization, which is what makes resumed partial rollouts token-exact. MoE expert routing traces can also be returned for routing replay (`sglang_rollout.py:224-232`).

### 3.4 Training-side micro-batch scheduling

After generation, `RolloutManager._split_train_data_by_dp` (`rollout.py:754-805`) shards samples across DP ranks — either round-robin (`partitions = [range(i, len, dp_size)]`, `rollout.py:767`) or **sequence-length-balanced** with `--balance-data` (`get_seqlen_balanced_partitions`, `rollout.py:764-765`) — and puts one dict per DP rank into the Ray object store (`Box(ray.put(rollout_data))`, `rollout.py:804`).

Each Megatron worker then builds its micro-batch schedule in `get_data_iterator` (`slime/backends/megatron_utils/data.py:290-385`). With `--use-dynamic-batch-size`, the number of micro-batches per optimizer step is computed from `max_tokens_per_gpu` via first-fit packing (`slime/utils/data.py:283-296`), **all-reduced with MAX over the DP group** so every rank executes the same pipeline schedule, then samples are re-partitioned to equalize tokens per micro-batch:

```python
num_microbatches = torch.tensor(num_microbatches, ...)
dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)
...
partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
```
(`data.py:353-376`; fixed `micro_batch_size` path at `data.py:338-340`; VPP divisibility clamp at `data.py:356-361`.)

### 3.5 Memory & cache management as it interacts with scheduling

Paged KV / radix prefix cache live inside SGLang; slime's job is to **time-share GPU memory between Megatron and SGLang** around the phase boundaries of the loop.

- **Rollout-side offload** is staged by SGLang memory-saver tags (imported at `rollout.py:15`): `GPU_MEMORY_TYPE_WEIGHTS`, `GPU_MEMORY_TYPE_KV_CACHE`, `GPU_MEMORY_TYPE_CUDA_GRAPH`. After generation, `rollout_manager.offload()` calls `release_memory_occupation` on every engine (`rollout.py:177-185`), which **flushes the radix/KV cache first** (`SGLangEngine.release_memory_occupation`, `slime/backends/sglang_utils/sglang_engine.py:357-359`: `self.flush_cache(); return self._make_request("release_memory_occupation")`). After training, the driver does a **two-stage resume**: `onload_weights()` (WEIGHTS only, `rollout.py:326-339`) → `update_weights()` overwrites them → `onload_kv()` (KV_CACHE + CUDA_GRAPH, `rollout.py:341-346`, driven from `train.py:91-97`). Re-allocating KV only *after* weight sync keeps peak memory during the IPC/NCCL transfer low.
- **Per-group offload decisions**: only server groups whose GPU range overlaps Megatron's get `needs_offload=True` (`rollout.py:1026-1027`: `needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus`); non-overlapping groups even get `enable_memory_saver=False` (`rollout.py:1032-1033`). Frozen (non-weight-updated) models that must offload reload from disk instead of CPU backup to save host RAM (`onload_weights_from_disk`, `rollout.py:197-207`).
- **Train-side offload** uses `torch_memory_saver` with an `LD_PRELOAD` hook (`slime/ray/actor_group.py:62-73`): `sleep()` destroys process groups and `torch_memory_saver.pause()`; `wake_up()` resumes and reloads process groups (`actor.py:156-177`).
- **Cache coherence at weight sync**: before pushing new weights the updater pauses generation and flushes the prefix/KV cache on every engine — stale KV computed under old weights must not be reused: `ray.get([engine.pause_generation...]); ray.get([engine.flush_cache...])` (`update_weight_from_tensor.py:144-147`; same in the distributed path, `update_weight_from_distributed.py:88-90`), then `continue_generation` after the last bucket (`update_weight_from_tensor.py:180`). `flush_cache` retries up to 60× because SGLang refuses to flush while requests are pending (`sglang_engine.py:291-308`).
- **Transfer memory hygiene**: colocated IPC sends weights in size-bounded buckets; after each bucket the trainer frees the flattened tensors and runs `torch.cuda.ipc_collect()` (`update_weight_from_tensor.py:158-170`). Distributed NCCL broadcast buckets are bounded by `--update-weight-buffer-size` (`update_weight_from_distributed.py:159-164`), and a global Ray `Lock` actor serializes bucket broadcasts to avoid NCCL deadlock with in-flight engine work (`update_weight_from_distributed.py:234-248`; lock impl `slime/ray/utils.py:38-57`).
- **Prefix-cache observability**: slime aggregates per-sample `cached_tokens / total_prompt_tokens` into `prefix_cache_hit_rate` (`rollout.py:1265-1273`).

### 3.6 Fault tolerance hooks in the schedule

`RolloutHealthMonitor` threads watch each server group (`rollout.py:386-391`); before each `update_weights`, rank 0 calls `recover_updatable_engines` (`slime/backends/megatron_utils/actor.py:543-546` → `rollout.py:527-548` → `RolloutServer.recover`, `rollout.py:269-310`), which restarts dead engines and reports `num_new_engines` so the weight updater re-creates its NCCL/IPC groups (`actor.py:555-564`).

## 4. Distributed orchestration (Ray or alternative)

**Ray is the single orchestration layer** — there is no torchrun; `torch.distributed` is initialized *inside* Ray actors from env vars the actors set themselves.

### 4.1 Placement: one PACK placement group, index-sliced

`create_placement_groups` builds **one** placement group of `{"GPU": 1, "CPU": 1}` bundles covering all GPUs, sized by mode (`slime/ray/placement_group.py:79-108`):

```python
elif args.colocate:
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    rollout_offset = 0
else:
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
    rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
```

So **colocated** mode shares the same bundles between training and rollout, while **disaggregated** mode gives rollout the tail slice of the same PG. Because Ray does not guarantee bundle ordering, a throwaway `InfoActor` (`@ray.remote(num_gpus=1)`, `placement_group.py:14-17`) is scheduled on every bundle to report `(node_ip, gpu_id)`, then bundles are re-sorted by IP+GPU id to get deterministic, NCCL-friendly rank→GPU mapping (`placement_group.py:41-76`).

Fractional GPUs make colocation work: train actors take `num_gpus=0.4` per bundle (`placement_group.py:111-119` passing `num_gpus_per_actor=0.4`, applied at `actor_group.py:89-96`) and each `SGLangEngine` actor takes `num_gpus = 0.2` (`rollout.py:99`), so both kinds of actor fit on the same 1-GPU bundle.

### 4.2 Actor topology

- **Training**: `RayTrainGroup` is *not* an actor — it's a driver-side handle list. It wraps `MegatronTrainRayActor` with `ray.remote(num_gpus=1, runtime_env={"env_vars": ...})` (env includes `NCCL_CUMEM_ENABLE=0` "because sglang will always set NCCL_CUMEM_ENABLE to 0", and an `LD_PRELOAD` of the torch_memory_saver hook for offload; `actor_group.py:53-83`) and creates `world_size` actors, one per bundle. Rank 0 picks a free port and all others receive it (`actor_group.py:88-99`). Each actor then sets `MASTER_ADDR/MASTER_PORT/WORLD_SIZE/RANK/LOCAL_RANK` env vars and calls `dist.init_process_group(...)` itself (`train_actor.py:41-48`, `63-66`) — Megatron TP/PP/EP/DP groups live entirely inside this actor set.
- **Rollout**: `RolloutManager` is a CPU-only actor (`num_cpus=1, num_gpus=0`, `placement_group.py:184-187`). Inside it, `start_rollout_servers` builds per-model `RolloutServer`s, each = one `sglang_router` + a list of `ServerGroup`s (regular, or prefill/decode for PD disaggregation, or encoder for EPD; `rollout.py:981-1115`). Each `ServerGroup.start_engines` creates `ray.remote(SGLangEngine)` actors pinned to specific bundles (`rollout.py:91-143`), pre-allocates server/nccl/dist-init ports via per-node port cursors so concurrent groups don't race (`rollout.py:822-908`), and `engine.init` launches the actual SGLang HTTP server as a **spawned multiprocessing child** and waits for `/health_generate` (`sglang_engine.py:53-79`, `196-198`). Node-0 engines self-register with the router via `POST /workers` (`sglang_engine.py:203-220`). All subsequent control (flush_cache, pause/continue generation, update_weights_from_tensor metadata) goes over HTTP to that child server (`sglang_engine.py:222-242`).
- The router itself is a **daemon `multiprocessing.Process`** running `sglang_router` inside the `RolloutManager` actor's process tree, not a Ray actor (`rollout.py:952-957`).
- A tiny `Lock` Ray actor provides cross-actor mutual exclusion for NCCL broadcasts (`slime/ray/utils.py:39-56`, created at `rollout.py:382`).

### 4.3 Weight sync: two strategies, selected by placement

`update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed` (`actor.py:126-127`). Every step (or every `update_weights_interval`), the driver calls `actor_model.update_weights()` which fans out to all train actors (`actor_group.py:135-137`); each actor gets the current engine handles from the `RolloutManager` (`actor.py:548-550`) and runs the updater under `pause_generation → flush_cache → ... → continue_generation` bracketing on rank 0 (`update_weight_from_tensor.py:144-181`).

**Colocated path (CUDA IPC via Ray):** Each train rank chunks its Megatron weights into HF-format named tensors, packs them into a `FlattenedTensorBucket`, serializes with sglang's `MultiprocessingSerializer` (which carries CUDA IPC handles, since the engine is on the *same GPU*), then `dist.gather_object`s the serialized strings over a per-engine **Gloo** group to the engine's first GPU rank, which makes a single Ray call `ipc_engine.update_weights_from_tensor.remote(...)` (`update_weight_from_tensor.py:122-135`, `209-267`). The class docstring states it plainly: "Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine" (`update_weight_from_tensor.py:24-30`). The HTTP server only receives metadata; "the real weights will be copied directly from GPUs" (`sglang_engine.py:266-278`). After each chunk, `torch.cuda.ipc_collect()` reclaims IPC blocks (`update_weight_from_tensor.py:158-170`). Mixed setups are handled: engines whose GPU range falls past `actor_num_nodes × actor_num_gpus_per_node` are treated as remote and get the NCCL path (`update_weight_from_tensor.py:84-96`).

**Disaggregated path (NCCL broadcast):** Per PP rank, the `DP=0,TP=0` rank becomes broadcast source for group `f"slime-pp_{pp_rank}"` (`update_weight_from_distributed.py:62-67`). `connect_rollout_engines_from_distributed` builds a fresh NCCL group of `sum(engine_gpu_counts) + 1` ranks — training source at rank 0, each engine's GPUs at cumulative offsets — by calling `engine.init_weights_update_group.remote(...)` and `init_process_group(backend="nccl", init_method=f"tcp://{master_address}:{master_port}", ...)` (`update_weight_from_distributed.py:252-298`). Updates then run param-by-param: TP `all_gather_param` → Megatron→HF conversion → bucketed `dist.broadcast(param.data, 0, group=group)` with names/dtypes/shapes sent ahead via Ray RPC (`update_weight_from_distributed.py:106-128`, `310-337`); expert (EP) params take a separate path with an extra EP all-gather (`update_weight_from_distributed.py:190-226`). Each bucket broadcast is wrapped in the Ray `Lock` actor "to prevent dead lock on broadcast" when multiple PP sources broadcast to shared engines (`update_weight_from_distributed.py:234-248`).

### 4.4 Rollout data source / buffer

The dataset lives in the `RolloutManager` process as a `DataSource` ABC (`get_samples/add_samples/save/load`, `slime/rollout/data_source.py:17-46`). `RolloutDataSource` walks a shuffled prompt dataset with epoch wraparound and stamps `group_index`/`index` on `n_samples_per_prompt` deep copies per prompt (`data_source.py:90-118`); `RolloutDataSourceWithBuffer` adds a FIFO (or custom `buffer_filter_path`) replay buffer drained before the dataset — this is where partial-rollout aborted groups are re-queued (`data_source.py:168-211`, fed by `sglang_rollout.py:622-624`). Its cursor state is checkpointed alongside model checkpoints via `rollout_manager.save/load` (`data_source.py:123-160`, driver calls at `train.py:63-64`).

## 5. Code organization style: function granularity

slime's rule of thumb is visible everywhere: **orchestration stays in one long, linear function with inline comments; extraction only happens for (1) reuse, (2) a pluggable seam, or (3) a rank/process boundary.** They do not extract for "tidiness" — single-use steps stay inline.

**Monolithic example 1 — the top-level driver `train(args)` (~94 lines, `train.py:9-102`).** The entire RL loop — placement groups, rollout manager creation, weight push, eval, train, save, offload/onload choreography — is one flat function the reader can scan top to bottom (loop quoted in §3.1). The only "extractions" are two *nested closures*, `offload_train` (`train.py:42-49`) and `save` (`train.py:51-64`) — extracted not for abstraction but because the closure keeps the loop body short while staying local. Justification: the loop is the product; hiding steps behind helpers would obscure the colocate offload/onload ordering, which is the trickiest invariant here.

**Monolithic example 2 — `MegatronTrainRayActor.train_actor` (~109 lines, `slime/backends/megatron_utils/actor.py:401-509`).** One function sequences ref-model logprobs → teacher logprobs → actor logprobs → advantages → train → backup, with model-swapping side effects (`self._switch_model("ref")`, `actor.py:413`) and env-var staging for routing replay (`os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"`, `actor.py:439`). Each block is guarded by flags and ordered comments ("Calculate adv and returns. Need to performed before training … because we may need normalize the whole rollout", `actor.py:462-463`). Splitting this would scatter an order-sensitive state machine. Note they DID split `train` / `train_critic` / `train_actor` (`actor.py:355-401`) — the split axis is *role*, not step.

**Monolithic example 3 — `generate_rollout_async` (~92 lines, `slime/rollout/sglang_rollout.py:389-480`)**: the whole over-sampling + dynamic-filter + abort continuous loop in one function, with the inner `while state.remaining_batch_size < target_data_size:` submission loop inline (`sglang_rollout.py:422-426`).

**Helpers they DID extract, and why:**

- `should_run_periodic_action(rollout_id, interval, num_rollout_per_epoch, num_rollout)` (`slime/utils/misc.py:73-94`) — extracted because it is called twice in the driver (save at `train.py:87`, eval at `train.py:98`) and encodes non-obvious boundary logic (last-rollout force, epoch boundary).
- `_prepare_prompt_ids(sample, tokenizer, processor)` (`slime/rollout/sglang_rollout.py:42-61`) — isolates the multimodal/token-reuse decision tree out of the async `generate` hot path; ~20 lines, pure function.
- `_send_to_colocated_engine(...)` (`slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py:209-267`) — module-level (not a method) because it is a self-contained collective: serialize → `dist.gather_object` → fire Ray IPC; it owns the "placeholder ranks have no gather group" edge case (`:218-220`).
- Tiny `_compute_rollout_offset` / `_compute_megatron_num_gpus` (`slime/ray/rollout.py:965-979`, 8 lines each) — extracted because colocate-vs-disaggregated GPU math is reused and easy to get wrong.
- Interesting middle ground: inside the ~135-line `start_rollout_servers` (`slime/ray/rollout.py:981-1115`), the per-group construction is a **nested closure `_make_group` with `nonlocal engine_offset, gpu_offset`** (`:1020-1056`) — reused across the EPD/non-EPD branches but kept local because it mutates loop-scoped offsets. Closures over helpers when state is loop-local is a recurring slime pattern (same as `train.py`).

**File/class size norms, value objects, comments:**

- **File sizes**: most files are 50–650 lines; the outliers are `slime/ray/rollout.py` (1283 — manager + server dataclasses + port allocation + metrics in one file), `slime/backends/megatron_utils/loss.py` (1212), and `slime/utils/arguments.py` (1841 — one giant argparse builder organized as ~20 *nested* functions `add_cluster_arguments`, `add_rollout_arguments`, `add_debug_arguments` inside `add_slime_arguments`, `arguments.py:36-1348`). They tolerate big files when the content is one cohesive concern; they don't split into a `config/` package. The whole framework threads a single argparse `Namespace` named `args` through every constructor (e.g. `RolloutManager.__init__(self, args, pg)`, `rollout.py:353`).
- **Dataclasses as the default value-object tool**, including for infra topology, not just data: `ServerGroup` / `RolloutServer` are dataclasses with behavior (`start_engines`, `recover`) (`rollout.py:38-70, 210-228`); `Sample` is a dataclass with **nested dataclasses as metric accumulators** (`Sample.SpecInfo.add(meta_info)`, `slime/utils/types.py:53-91`) and a `from_dict` that derives valid kwargs from the source of truth — `field_names = set(Sample.__dataclass_fields__.keys())` (`types.py:136-138`); `ParamInfo` is `@dataclass(frozen=True)` (`types.py:177-185`).
- **Enums are rare; strings carry most discriminators.** The only enum in the package is `Sample.Status` (`types.py:31-41`). `worker_type` is a commented string field — `worker_type: str = "regular"  # "regular", "prefill", "decode", or "placeholder"` (`rollout.py:52`) — and `role: str = "actor"` (`actor_group.py:36`). **No `typing.Protocol` anywhere** (grep over `slime/` returns zero); the one interface is a classic `abc.ABC` `DataSource` with 5 abstract methods (`slime/rollout/data_source.py:17-46`). Type aliases instead of classes for batches: `RolloutBatch = dict[str, list[torch.Tensor] | ...]` (`types.py:190`).
- **Comment style**: docstrings document *contracts and return shapes* for cross-process calls ("Returns ``(init_handles, port_cursors)`` … The caller should ``ray.get()`` on the handles", `rollout.py:71-80`; `async_train` docstring spelling out what critic refs resolve to, `actor_group.py:112-119`); inline comments explain WHY, especially around CUDA/NCCL hazards: "because sglang will always set NCCL_CUMEM_ENABLE to 0 we need also set it to 0 to prevent nccl error" (`actor_group.py:54-56`), "Free GPU tensors so the caching allocator can reuse the blocks, then release CUDA IPC cache entries whose consumers … have already closed their IPC handles" (`update_weight_from_tensor.py:161-165`). Enum variants get rationale comments (`Status.FAILED`, `types.py:36-39`). **`# TODO` is left liberally in shipped code** as honest debt markers — `data_source.py:49,59,65,91,213,217`, `rollout.py:594,637,643`, even in `pyproject.toml` (`line-length = 320  # TODO`). Formatting: black line length 119, ruff with E501 ignored (`pyproject.toml:23-37`).
- **Backward-compat adapters instead of breaking changes**: rollout functions may return legacy bare lists, normalized by `call_rollout_fn` ("# compatibility for legacy version", `slime/rollout/base_types.py:19-26`); same shape for filters via `call_dynamic_filter` (`slime/rollout/filter_hub/base_types.py:11-21`).

## 6. Naming conventions

- **`*_utils` is the dominant module suffix**, used even for whole backends: `slime/backends/megatron_utils/`, `slime/backends/sglang_utils/`, and ~20 files in `slime/utils/` (`http_utils.py`, `ppo_utils.py`, `mask_utils.py`, `trace_utils.py`, `wandb_utils.py`, …). It's a junk-drawer convention they accept openly.
- **`*_hub` for plugin registries**: `slime/rollout/rm_hub/` (reward models) and `slime/rollout/filter_hub/` (sample filters), each with a `base_types.py` defining the output dataclass + a `call_*` adapter.
- **Class-name suffixes encode the Ray topology**: `*Manager` = a singleton Ray actor coordinating others (`RolloutManager`, `slime/ray/rollout.py:349-350`, decorated `@ray.remote`); `*Group` = a handle-holding fan-out wrapper that is NOT itself an actor (`RayTrainGroup` "A group of ray actors", `slime/ray/actor_group.py:10-13`; `ServerGroup` dataclass, `slime/ray/rollout.py:38-45`); `*Actor` = the per-GPU Ray actor class (`RayActor` → `TrainRayActor` → `MegatronTrainRayActor`, `slime/ray/ray_actor.py:4`, `slime/ray/train_actor.py:28`, `slime/backends/megatron_utils/actor.py:44`), including local rebinding at the remote-ification site: `RolloutRayActor = ray.remote(SGLangEngine)` (`slime/ray/rollout.py:91`).
- **`async_` prefix means "returns Ray ObjectRefs, caller must ray.get"** — a documented protocol, not asyncio: "Functions start with 'async' should return list of object refs" (`slime/ray/actor_group.py:13`); `async_init` / `async_train` vs blocking `save_model` / `update_weights` which `ray.get` internally (`actor_group.py:101-137`).
- **A uniform verb vocabulary repeated identically at every layer**: `offload` / `onload` / `onload_weights` / `onload_kv` exist with the same names on `ServerGroup` (`rollout.py:177-207`), `RolloutServer` (`rollout.py:312-346`), `RolloutManager` (`rollout.py:510-525`), and are driven from `train.py:73-96`. Grep one verb, see the whole pipe. (The one inconsistency: the train side uses `sleep`/`wake_up`, mapped at `actor_group.py:139-143`.)
- **Plugin seams are `*_path` argparse strings + `load_function`**: `rollout_function_path`, `data_source_path`, `custom_reward_post_process_path` resolved via `load_function(self.args.rollout_function_path)` (`slime/ray/rollout.py:359-371`); `load_function` itself is 9 lines of `importlib` (`slime/utils/misc.py:9-17`). Same pattern for `dynamic_sampling_filter_path` (`sglang_rollout.py:409-411`) and `buffer_filter_path` (`slime/rollout/data_source.py:172-175`).
- **snake_case verb_noun for free functions**, often chaining domain verbs: `generate_and_rm`, `generate_and_rm_group` (`sglang_rollout.py:240,310`), `start_rollout_servers`, `connect_rollout_engines`, `update_weights_from_distributed`. Private module helpers get `_` even when long: `_allocate_rollout_engine_addr_and_ports_normal` (`rollout.py:822`) — they prefer explicit ugliness over abbreviation.
- **Public-wrapper-over-private pattern for remote APIs**: `get_metrics_router_addr` is a one-line "Public wrapper for remote calls from the driver process" around `_get_metrics_router_addr` (`rollout.py:394-409`).

## 7. End-to-end flow trace

One RL step (sync, colocated):

1. Driver enters step: `train.py:70` — `rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))`.
2. `RolloutManager.generate` → `_get_rollout_data` → `call_rollout_fn(self.generate_rollout, ...)` — `slime/ray/rollout.py:478-484, 583` (dispatch shim `slime/rollout/base_types.py:19-26`).
3. `generate_rollout` starts the asyncio loop — `slime/rollout/sglang_rollout.py:621`: `run(generate_rollout_async(args, rollout_id, data_source.get_samples))`.
4. Prompt fetch: `RolloutDataSourceWithBuffer.get_samples` — buffer first, then dataset, replicating each prompt `n_samples_per_prompt` times — `slime/rollout/data_source.py:177-189, 107-118`.
5. Group tasks submitted: `GenerateState.submit_generate_tasks` → `generate_and_rm_group` → per-sample `generate_and_rm` — `sglang_rollout.py:136-149, 323-333`.
6. Semaphore + DP pick, then HTTP generate: `async with state.semaphore` → `with state.dp_rank_context()` → `generate()` → `POST http://{router_ip}:{router_port}/generate` with `input_ids` + `return_logprob` — `sglang_rollout.py:260-277, 158, 174-201`. (The router forwards to an SGLang server launched as a subprocess of an `SGLangEngine` Ray actor — registration at `slime/backends/sglang_utils/sglang_engine.py:203-220`; engine actors created in `ServerGroup.start_engines`, `slime/ray/rollout.py:91-145`.)
7. Tokens/logprobs appended to `Sample`; reward computed via `async_rm`/`batched_async_rm` — `sglang_rollout.py:204-234, 288-301`.
8. Collection loop accepts/drops groups until `rollout_batch_size`, then aborts stragglers (partial samples → buffer) — `sglang_rollout.py:422-452, 461, 622-623`.
9. Back in `RolloutManager`: trim to global-batch multiple → GRPO group-norm rewards → tensor-ish train dict → DP split → `Box(ray.put(...))` per DP rank — `rollout.py:590-605, 655-680, 682-749, 754-805`.
10. Driver trains: `train.py:86` → `RayTrainGroup.async_train` fans out `actor.train.remote(...)` to every Megatron worker — `slime/ray/actor_group.py:111-129`.
11. `MegatronTrainRayActor.train`: `wake_up()` (if offloaded) → `_get_rollout_data` (`process_rollout_data` pulls this rank's partition from the object store, `slime/utils/data.py:299-310`) → `train_actor` — `slime/backends/megatron_utils/actor.py:355-374`.
12. `train_actor`: `get_data_iterator` builds the (dynamic) micro-batch schedule (`megatron_utils/data.py:290-385`) → ref/teacher/old-actor log-prob forwards via `weights_backuper` model switching (`actor.py:408-460`) → `compute_advantages_and_returns` (`actor.py:464`) → Megatron `train(...)` (`actor.py:478-486`) → CPU backup of new weights `weights_backuper.backup("actor")` (`actor.py:496`).
13. Driver syncs weights: `train.py:93` → `actor_group.update_weights()` (`actor_group.py:135-137`) → `MegatronTrainRayActor.update_weights` (`actor.py:539-591`) fetches engine handles + lock from `RolloutManager` (`actor.py:548-550` → `rollout.py:460-472`).
14. Colocated transfer (`UpdateWeightFromTensor.update_weights`, `update_weight_from_tensor.py:138-181`): pause + flush caches → per HF-weight chunk: serialize CUDA-IPC handles via `MultiprocessingSerializer` into a `FlattenedTensorBucket`, Gloo `gather_object` to the engine's first co-resident rank, then one Ray call `ipc_engine.update_weights_from_tensor.remote(...)` per dtype bucket — `update_weight_from_tensor.py:209-267` — landing in `SGLangEngine.update_weights_from_tensor` → `POST /update_weights_from_tensor` (`sglang_engine.py:266-289`). (Disaggregated engines instead get per-PP-rank NCCL groups `slime-pp_{pp_rank}` and `dist.broadcast` from the PP source rank — `update_weight_from_distributed.py:62-79, 252-298, 310-337`.)
15. `continue_generation` resumes serving (`update_weight_from_tensor.py:180`), driver does `onload_kv` (`train.py:96-97`), loop returns to step 1 with the new policy.

## 8. Ideas worth borrowing for wm-infra

1. **Keep the driver loop flat and sacred.** `train.py:67-102` proves a ~35-line visible loop of named remote calls is the best documentation of train↔rollout↔weight-sync ordering. wm-infra's `EngineLoop` should read the same way: phase ordering (`ENCODE_TEXT → … → DECODE_VAE`) inline in one function, not dispatched through layers. Resist extracting steps that run once.
2. **Adopt the `async_*` = "returns futures" naming contract.** One documented prefix (`actor_group.py:13`) removes all ambiguity about which calls block. wm-infra's mixed sync/async engine/trainer surface (e.g. `submit_async` in the Wan job queue) would benefit from making this a hard convention rather than per-method memory.
3. **One verb vocabulary, repeated verbatim at every layer.** slime's `offload`/`onload_weights`/`onload_kv` chain from `train.py` down to `ServerGroup` is instantly greppable and makes the GPU-memory choreography auditable. For wm-infra's colocated rollout+GRPO future, copy this exactly: pick the verbs once, never synonym them (`sleep`/`wake_up` on the train side is slime's one inconsistency — `actor_group.py:139-143` maps `onload→wake_up` — and it costs readability).
4. **`*_path` string args + 9-line `load_function` is a surprisingly load-bearing plugin system** (`misc.py:9-17`, consumed at `rollout.py:359-371`). For wm-infra rewards/filters/postprocessors this beats a decorator registry: zero import-time coupling, works across Ray workers, and the config file shows the fully-qualified implementation. Pair it with slime's `call_*` output-normalizing adapters (`base_types.py:19-26`) so the seam can evolve without breaking user functions.
5. **Dataclass-with-behavior for topology objects.** `ServerGroup`/`RolloutServer` (`rollout.py:38-247`) show that placement math (gpu_offset, rank_offset, `needs_offload`) belongs in a dataclass whose `@property`s derive everything else — not in dicts threaded through functions. wm-infra's GPU-topology-derived release flags (cosmos run) are the same shape of problem.
6. **Derive, don't duplicate, field sets** — `Sample.from_dict` using `Sample.__dataclass_fields__` (`types.py:136-142`) is exactly the AGENTS.md "derive validation sets from the typed structure" rule, in the wild.
7. **Nested metric accumulators on the sample object.** `Sample.SpecInfo.add(meta_info)` / `PrefixCacheInfo.add` (`types.py:53-120`) accumulate per-request engine stats across partial rollouts and serialize with the sample. wm-infra's per-request denoise stats (steps, cache hits, per-phase latency) should live on the request object the same way, not in a side dict in the scheduler.
8. **The data source is an injected callable, and the buffer is a subclass.** `generate_rollout_async(args, rollout_id, data_source: Callable)` (`sglang_rollout.py:389-390`) plus `RolloutDataSourceWithBuffer.get_samples` = buffer-first-then-dataset (`data_source.py:177-189`), with the eviction policy itself pluggable (`buffer_filter_path`, default `pop_first`, `data_source.py:172-175, 225-229`). Clean blueprint for a wm-infra replay/partial-rollout buffer.
9. **Over-sample / first-completed-wins / abort-the-tail is the long-tail answer.** The combination of demand-driven over-provisioning (`sglang_rollout.py:422-426`), `asyncio.wait(FIRST_COMPLETED)` collection, `/abort_request abort_all` preemption (`sglang_rollout.py:345-386`), and partial-rollout resume-from-tokens (`sglang_rollout.py:191-218`) is directly transferable to video rollout: long denoise jobs are the same straggler problem as long generations.
10. **Staged memory resume around weight sync.** Onload weights → push new weights → only then re-allocate KV/cache (`train.py:91-97`, `rollout.py:326-346`) bounds peak memory during transfer; wm-infra's colocated trainer+denoiser should stage VAE/latent-cache re-allocation the same way.
11. **Don't import the enum religion.** slime ships a production system with one enum and string `worker_type`/`role` discriminators (`rollout.py:52`). The lesson isn't "use strings" — it's that they spend their rigor budget on *protocol docstrings* (what refs resolve to, who must `ray.get`) rather than on type ceremony. For wm-infra: keep the existing typed phases, but copy the habit of documenting cross-process return shapes in docstrings.
12. **Anti-pattern to skip**: `GenerateState(metaclass=SingletonMeta)` (`sglang_rollout.py:83`, `misc.py:20-34`) — a process-global singleton holding the semaphore, pending tasks, and `aborted` flag, manually `reset()` between rollouts (`sglang_rollout.py:470`). It works because exactly one rollout runs per process, but it's fragile state-by-convention; wm-infra's explicit `VideoExecutionState` objects are already the better design.

## 9. Source-of-truth index

| Claim | Evidence |
|---|---|
| Driver train loop: generate → train → update_weights, all via blocking `ray.get` | `train.py:66-99` |
| Driver nested closures `offload_train` / `save` | `train.py:42-49, 51-64` |
| One-step-ahead async loop; drain before weight sync; colocate asserted off | `train_async.py:11, 35-43, 69-73` |
| Single PACK PG; colocate shares bundles, disagg appends rollout GPUs | `slime/ray/placement_group.py:41-47, 79-108` |
| InfoActor probes + IP/GPU-sorted bundle reordering | `slime/ray/placement_group.py:14-17, 49-76` |
| Train actors 0.4 GPU/bundle; engines 0.2 GPU/bundle | `slime/ray/placement_group.py:111-119`; `slime/ray/actor_group.py:89-96`; `slime/ray/rollout.py:99` |
| RayTrainGroup = driver-side handle list; rank-0 master addr propagation; NCCL_CUMEM_ENABLE/LD_PRELOAD env | `slime/ray/actor_group.py:53-99` |
| Actors self-init torch.distributed from env vars | `slime/ray/train_actor.py:41-48, 63-66` |
| RolloutManager CPU actor; pluggable rollout fn + data source via `load_function`; Lock + health monitors | `slime/ray/placement_group.py:184-187`; `slime/ray/rollout.py:349-392` |
| `generate()` step: rollout fn → trim → reward norm → convert → DP split | `slime/ray/rollout.py:478-491, 590-634, 655-805` |
| ServerGroup/RolloutServer; engine creation, port cursors, PD/EPD groups | `slime/ray/rollout.py:38-207, 822-908, 981-1115` |
| SGLangEngine spawns SGLang HTTP server child; registers with router; HTTP control plane | `slime/backends/sglang_utils/sglang_engine.py:53-79, 196-220, 222-242` |
| Router = daemon multiprocessing.Process; health check disabled | `slime/ray/rollout.py:911-962` (launch `:952-957`, health-check off `:948`) |
| Main rollout scheduling loop (over-sample, FIRST_COMPLETED, dynamic filter) | `slime/rollout/sglang_rollout.py:416-452` |
| `over_sampling_batch_size` default = rollout_batch_size, must be ≥ | `slime/utils/arguments.py:1782-1787` |
| Fleet-wide semaphore admission control; 512/engine default | `slime/rollout/sglang_rollout.py:94-96`; `slime/backends/sglang_utils/arguments.py:39` |
| Group→sample task fan-out; group RM | `slime/rollout/sglang_rollout.py:136-149, 310-342` |
| Per-sample HTTP `/generate`, token append, logprobs, routed experts | `slime/rollout/sglang_rollout.py:152-236` |
| Abort stragglers via `/abort_request abort_all`; partial samples collected | `slime/rollout/sglang_rollout.py:345-386, 461, 622-623` |
| Partial-rollout buffer (pop_first) and resume-from-tokens; off-policy masking | `slime/rollout/data_source.py:168-229`; `sglang_rollout.py:191-192, 211-218, 247-255` |
| DP rank balancing + session consistent hashing | `slime/rollout/sglang_rollout.py:119-129, 196-198` |
| Seqlen-balanced DP split, `Box(ray.put(...))` per rank; per-rank fetch | `slime/ray/rollout.py:754-805`; `slime/utils/data.py:299-311` |
| Dynamic micro-batching: max_tokens_per_gpu packing, DP-MAX all-reduce, balanced partitions | `slime/backends/megatron_utils/data.py:290-385`; `slime/utils/data.py:283-296` |
| Train actor step: wake_up → data fetch → logprob forwards → adv → train → backup | `slime/backends/megatron_utils/actor.py:355-509` |
| Updater selection by colocate flag | `slime/backends/megatron_utils/actor.py:126-127` |
| update_weights orchestration + engine recovery + group reconnect | `slime/backends/megatron_utils/actor.py:539-591`; `slime/ray/rollout.py:269-310, 527-548` |
| Colocated sync: FlattenedTensorBucket + MultiprocessingSerializer + Gloo gather + Ray IPC; `ipc_collect`; mixed remote engines | `update_weight/update_weight_from_tensor.py:24-30, 84-96, 122-181, 209-267`; `sglang_engine.py:266-289` |
| Distributed sync: per-PP NCCL groups `slime-pp_{rank}`, world_size = engines+1, bucketed broadcast, EP path, Lock | `update_weight/update_weight_from_distributed.py:62-79, 106-128, 190-226, 228-298, 310-337` |
| Lock = single-flag Ray actor | `slime/ray/utils.py:38-57` |
| Train-side offload: torch_memory_saver pause/resume + process-group destroy/reload | `slime/ray/actor_group.py:62-73`; `actor.py:156-177` |
| Staged rollout memory: release (flush KV first) / onload weights then KV; per-group `needs_offload`; disk reload for frozen models | `slime/backends/sglang_utils/sglang_engine.py:291-308, 357-368`; `slime/ray/rollout.py:177-207, 326-346, 1026-1033`; `train.py:91-97` |
| Weight sync pauses generation + flushes cache; resumes after; flush retries 60× | `update_weight_from_tensor.py:138-181`; `update_weight_from_distributed.py:81-140`; `sglang_engine.py:291-308` |
| Fault tolerance: health monitor thread; recovery before weight sync | `slime/utils/health_monitor.py:10-21, 52`; `slime/ray/rollout.py:386-391, 527-548`; `actor.py:543-564` |
| Prefix cache hit-rate metrics | `slime/ray/rollout.py:1265-1273` |
| DataSource ABC, epoch cursor, buffer-first sampling, state checkpoint | `slime/rollout/data_source.py:17-46, 90-118, 123-160, 168-211`; `train.py:63-64` |
| Monolithic `train_actor` (~109 lines) + role-based split | `slime/backends/megatron_utils/actor.py:355-509` |
| `generate_rollout_async` monolithic loop ~92 lines | `slime/rollout/sglang_rollout.py:389-480` |
| Extracted helpers: `should_run_periodic_action`, `_prepare_prompt_ids`, `_send_to_colocated_engine`, offset math, `_make_group` closure | `slime/utils/misc.py:73-94`; `sglang_rollout.py:42-61`; `update_weight_from_tensor.py:209-267`; `slime/ray/rollout.py:965-979, 1020-1056` |
| `*_hub` plugin dirs + base_types adapters | `slime/rollout/rm_hub/`, `slime/rollout/filter_hub/base_types.py:11-21`; `slime/rollout/base_types.py:19-26` |
| `*Manager`/`*Group`/`*Actor` suffix semantics; `RolloutRayActor = ray.remote(SGLangEngine)` | `slime/ray/rollout.py:349-350, 91`; `slime/ray/actor_group.py:10-13`; `slime/ray/train_actor.py:28`; `slime/ray/ray_actor.py:4` |
| `async_*` returns-refs convention | `slime/ray/actor_group.py:13, 101-137` |
| offload/onload verb vocabulary across 3 layers (+ sleep/wake_up exception) | `slime/ray/rollout.py:177-207, 312-346, 510-525`; `train.py:73-96`; `actor_group.py:139-143` |
| `*_path` + `load_function` plugin seam | `slime/utils/misc.py:9-17`; `slime/ray/rollout.py:359-371`; `slime/rollout/data_source.py:172-175`; `sglang_rollout.py:409-411` |
| Public wrapper over private for remote API | `slime/ray/rollout.py:394-409` |
| `arguments.py` nested `add_*_arguments` structure, 1841 lines | `slime/utils/arguments.py:36-1348` |
| Dataclasses with behavior; Sample nested accumulators; derived field set; frozen ParamInfo | `slime/ray/rollout.py:38-70, 210-247`; `slime/utils/types.py:53-120, 136-142, 177-185` |
| One Enum only; string `worker_type`/`role`; no Protocol; `RolloutBatch` alias | `slime/utils/types.py:31-41, 190`; `slime/ray/rollout.py:52`; `slime/ray/actor_group.py:36`; `slime/rollout/data_source.py:17-46` |
| WHY comments (NCCL env, ipc_collect); contract docstrings; liberal TODOs; black 119 | `slime/ray/actor_group.py:54-56, 112-119`; `update_weight_from_tensor.py:161-170`; `slime/ray/rollout.py:71-80`; `slime/rollout/data_source.py:49-91`; `pyproject.toml:23-37` |
| GenerateState singleton anti-pattern | `slime/rollout/sglang_rollout.py:83-149, 470`; `slime/utils/misc.py:20-34` |

---

# Part II — Deep Dive

Part I covered orchestration (driver loop, Ray topology, rollout scheduling, weight-sync transport). Part II goes below those boundaries: the math and mechanics inside one training step, the byte-level life of a sample, every extension seam, startup/failure semantics, and the engineering tricks worth stealing. All citations were re-checked against the checkout. One framing fact up front: **this checkout has no FSDP backend** — `--train-backend` accepts only `"megatron"` (`slime/utils/arguments.py:1413`: `choices=["megatron"], default="megatron"`), and `slime/backends/` contains only `megatron_utils` + `sglang_utils`. "Parallelism" in slime means Megatron TP/PP/VPP/CP/EP plus Megatron's ZeRO-style distributed optimizer.

## 10. Training engine internals: below `MegatronTrainRayActor.train()`

Part I §3.4 and §4.3 covered DP sharding and the weight-sync *transport*; this section covers what happens inside one training step on one Megatron actor.

### 10.1 The recomputation passes: one model, four weight sets

A single Megatron model instance serves up to four logical models (actor, ref, teacher, old_actor) by hot-swapping weights from pinned-CPU backups. `TensorBackuper.backup(tag)` copies every named param/buffer into `torch.empty_like(param, device="cpu", pin_memory=True)`; `restore(tag)` copies back with `non_blocking=True` + `torch.cuda.synchronize()` (`slime/utils/tensor_backper.py:54-74`). `_switch_model(tag)` is just `restore` + bookkeeping (`slime/backends/megatron_utils/actor.py:254-258`). With `--disable-weights-backuper` a `_TensorBackuperNoop` replaces real backups with a uint32-sum hash sanity check (`tensor_backper.py:77-115`), saving host RAM when only the single "actor" tag exists.

`train_actor` (`actor.py:401-509`) sequences the passes:

1. ref logprobs — `_switch_model("ref")` then `forward_only(get_log_probs_and_entropy, ..., store_prefix="ref_")` (`actor.py:413-420`); the ref model exists only when KL is on (`with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss`, `slime/ray/placement_group.py:158`).
2. teacher logprobs for Megatron-side on-policy distillation (`store_prefix="teacher_"`, `actor.py:423-433`).
3. **old-policy logprobs** — `self._switch_model("old_actor" if self.args.keep_old_actor else "actor")` then a prefix-less `compute_log_prob` (`actor.py:435-448`). So π_old in the PPO ratio is *recomputed by the training engine*, not taken from SGLang; the pass is skipped only when `--use-rollout-logprobs` is on and mismatch metrics are off (`if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics`, `actor.py:436`). Rollout logprobs, when shipped, are CP-sliced and moved to GPU float32 in `_get_rollout_data` (`actor.py:224-247`).
4. `_switch_model("actor")` back, `compute_advantages_and_returns` (must run before training "because we may need normalize the whole rollout", `actor.py:459-464`), then `train(...)`.
5. After the optimizer steps: `self.weights_backuper.backup("actor")` refreshes the CPU copy that the weight updater reads (`actor.py:496`), and every `--ref-update-interval` rollouts the ref backup is overwritten from the current actor (`actor.py:499-507`).

`forward_only` flips modules to `.eval()`, runs Megatron's `forward_backward_func(..., forward_only=True)` once per optimizer-step's microbatch count, and collects the per-sample tensors that the post-forward callback returns; under dynamic batch size it un-shuffles results back to original sample order via `data_iterator[0].micro_batch_indices` (`slime/backends/megatron_utils/model.py:309-357`).

### 10.2 Logprob math: fused vocab-parallel CE on the packed stream

`get_log_probs_and_entropy` (`slime/backends/megatron_utils/loss.py:384-468`) operates on the full packed logits `[T, V]` at once "so backward traverses [T, V] only once". Steps: squeeze the `thd` batch dim, **divide logits by `rollout_temperature` so train-side logprobs match sampling-time logprobs** (`loss.py:417-420`), build a shifted-target token tensor for the packed/CP-sliced layout (`_build_shifted_tokens`, `loss.py:231-289`), then:

```python
log_prob_full, entropy_full = calculate_log_probs_and_entropy(
    logits, full_tokens, tp_group, with_entropy=with_entropy, chunk_size=chunk_size,
)
```

`calculate_log_probs_and_entropy` optionally chunks the `[T, V]` tensor (`--log-probs-chunk-size`, default -1, `slime/utils/arguments.py:156-157`) and computes log-probs as the negative of Megatron's `fused_vocab_parallel_cross_entropy(logits, tokens, tp_group)` (`slime/utils/ppo_utils.py:151-158`) — i.e. TP ranks never materialize full-vocab logits. Entropy uses a verl-derived custom autograd function `_VocabParallelEntropy` with three `all_reduce`s over the TP group (max, sum-exp, softmax·logits) and an in-place backward that reuses `softmax_logits` as the grad buffer (`ppo_utils.py:162-198`). Per-sample response slices are then extracted with layout-specific offset math (`_extract_per_sample`, `loss.py:292-381`): `log_prob_full[start-1 : end-1]` against `tokens[-response_length:]` — the standard one-token shift.

### 10.3 Advantages: GRPO normalization happens in *two* places

**Place 1 — RolloutManager (CPU, before DP split).** `_post_process_rewards` group-normalizes raw rewards (`slime/ray/rollout.py:655-680`), gated by `--disable-rewards-normalization` / `--disable-grpo-std-normalization` (the Dr.GRPO option; `store_false` dests at `arguments.py:871-879`; std-normalization auto-disabled when `n_samples_per_prompt == 1`, `arguments.py:1779-1780`):

```python
if (
    self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
    and self.args.rewards_normalization
):
    # group norm
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
        rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
    ...
    mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean
    if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)
```

**Place 2 — train actor (GPU, last PP stage only).** `compute_advantages_and_returns` (`loss.py:571-741`) first computes the reward-shaping KL — or zeros when `kl_coef == 0` (`loss.py:609-622`) — then for GRPO simply broadcasts the scalar normalized reward over response tokens:

```python
elif args.advantage_estimator in ["grpo", "gspo"]:
    rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    returns = get_grpo_returns(rewards, kl)
    advantages = [r for r in returns]
```

with `get_grpo_returns` literally `torch.ones_like(kl[i]) * rewards[i]` (`ppo_utils.py:201-208`) — **the `kl` argument is used only for shape**; for GRPO the KL penalty never enters returns, it only enters the loss via `--use-kl-loss` (the two are mutually exclusive: `assert not (args.kl_coef != 0 and args.kl_loss_coef != 0)`, `arguments.py:1676`). PPO instead folds `-kl_coef·kl` into token rewards with the scalar reward added at the last token, then runs GAE — including a FlashLinearAttention-style `chunked_gae` (intra-chunk matmul scan `S_local = Δ @ M` with `M[i,j] = w^(j-i)`, recurrent state across 128-token chunks) reducing the sequential scan from O(T) to O(T/128) (`ppo_utils.py:506-646`). REINFORCE++ does a per-token discounted return after a CP all-gather (`ppo_utils.py:211-278`). Optional OPD adds `-opd_kl_coef·(student_logp - teacher_logp)` to advantages in-place (`loss.py:530-568`). `--normalize-advantages` whitens advantages across the DP group with masked statistics via a 3-element `all_reduce` of `[sum, sum_sq, mask_sum]` (`loss.py:685-738`; `slime/utils/distributed_utils.py:94-134`).

### 10.4 The policy loss, quoted

The PPO ratio core (`ppo_utils.py:124-148`, `@torch.compile(dynamic=True)`):

```python
ratio = (-ppo_kl).exp()
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
clipfrac = torch.gt(pg_losses2, pg_losses1).float()
```

where `ppo_kl = old_log_probs - log_probs` per token (`loss.py:883-885`), so `ratio = exp(logp_new − logp_old)`; defaults `eps_clip=0.2`, `eps_clip_high=eps_clip` (`arguments.py:765`, `1702-1703`), optional dual-clip PPO via `eps_clip_c` (`ppo_utils.py:138-144`). For **GSPO** the per-token KL is replaced by a sequence-level mean KL expanded back to token shape, computed on CP-allgathered full sequences (`compute_gspo_kl`, `ppo_utils.py:95-121`; gather at `loss.py:848-860`). **OPSM** zeroes whole sequences where `advantage < 0 and seq_kl > opsm_delta` (`ppo_utils.py:54-92`, applied `loss.py:889-890`).

Assembly in `policy_loss_function` (`loss.py:794-1012`):

```python
pg_loss = pg_loss_reducer(pg_loss)            # sum of per-sample masked means
...
loss = pg_loss - args.entropy_coef * entropy_loss
if args.use_kl_loss:
    kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type,
                           importance_ratio=importance_ratio)   # k1/k2/k3/low_var_kl
    loss = loss + args.kl_loss_coef * kl_loss
if log_probs.numel() == 0:
    loss += 0 * logits.sum()                  # keep autograd graph alive on empty CP chunks
```

(`loss.py:945-974`). `compute_approx_kl` implements Schulman's k1/k2/k3 estimators with optional DeepSeek-V3.2-style unbiased IS weighting `exp(logp_new − logp_old)·kl` (`--use-unbiased-kl`) and a ±10 clamp for `low_var_kl` (`ppo_utils.py:11-51`). **TIS** (truncated importance sampling against rollout logprobs) multiplies `pg_loss` by `clamp(exp(train_old_logp − rollout_logp), tis_clip_low, tis_clip)` — `vanilla_tis_function` (`loss.py:744-765`); the icepop variant zeroes out-of-range tokens instead of clamping (`loss.py:768-791`); when the TIS function returns modified masks, the loss denominator is rebuilt with them while mismatch *metrics* keep the pre-rejection masks to avoid being "artificially driven to 0" (`loss.py:893-932, 995-1002`).

**Loss masking** is per-sample-mean by construction: `get_sum_of_sample_mean` returns `Σ_samples (x_i · mask_i).sum() / clamp_min(mask_i.sum(), 1)` — or plain masked token-sum under `--calculate-per-token-loss` — with CP-aware chunked masks when `cp_size > 1` (`slime/backends/megatron_utils/cp_utils.py:53-120`). The masks themselves come from rollout (`sample.loss_mask`, zeroed entirely for removed samples, `slime/ray/rollout.py:707-719`) and are re-aligned to the shifted-token stream by `F.pad(loss_mask, (prompt_length - 1, 1), value=0)` (`slime/backends/megatron_utils/data.py:139-145`).

### 10.5 Loss scaling: cancelling Megatron's own averaging

`loss_function` (`loss.py:1124-1212`) dispatches by `--loss-type` (policy/value/sft/custom), optionally wraps the whole loss fn in `torch.utils.checkpoint` (`--recompute-loss-function`, `loss.py:1177-1178`), then rescales:

```python
if not args.calculate_per_token_loss:
    loss = (loss * num_microbatches / global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True))
else:
    loss = loss * mpu.get_context_parallel_world_size()
```

(`loss.py:1190-1197`). Megatron's pipeline engine divides each microbatch loss by `num_microbatches` and DDP averages grads over DP×CP; multiplying by `num_microbatches × dp_cp_size / global_batch_size` makes the effective objective `Σ sample_means / global_batch_size` exactly, independent of the (dynamic) microbatch count. In per-token mode the returned normalizer is `num_tokens` (clamped ≥1 per sample, `loss.py:1153, 1199-1201`) and Megatron's `finalize_model_grads` (installed at `model.py:603`) does global token normalization; the `× cp_size` cancels Megatron's CP scaling (comment at `loss.py:1190`). A second graph-liveness hack: under allgather-CP a rank whose chunk has zero loss tokens would skip the CP-gather backward (reduce-scatter) and **deadlock the other CP ranks**, so `loss = loss + 0 * logits.sum()` forces full-graph traversal (`loss.py:1182-1188`). Metric reporting piggybacks the same path: each microbatch returns `{"keys": [...], "values": tensor([count, m1, m2, ...])}`, summed across microbatches and all-reduced over DP-with-CP, then divided by the count (`loss.py:1199-1212`; `model.py:527-544`).

### 10.6 Parallelism mechanics

**Configuration.** Each Ray train actor calls `init(args)` → `mpu.initialize_model_parallel(tp, pp, vpp, context_parallel_size=..., expert_model_parallel_size=..., expert_tensor_parallel_size=..., order="tp-cp-ep-dp-pp")` (`slime/backends/megatron_utils/initialize.py:33-53`), seeds with PP-rank-offset seeds (`initialize.py:14-30`), and initializes Megatron's microbatch calculator only "to pass some validation" — slime drives its own microbatching (`initialize.py:78-86`). slime forces two Megatron settings: **`args.use_distributed_optimizer = True` ("always use zero optimizer")** and `variable_seq_lengths = True` (`slime/backends/megatron_utils/arguments.py:80-84, 17-24`). The model is built by a standard Megatron `model_provider` (TE layer specs, MoE decoder block spec, optional MTP block, optional `fp8_model_init`; `slime/backends/megatron_utils/model_provider.py:122-237`), wrapped to apply regex-based parameter freezing (`--only-train-params-name-list` / `--freeze-params-name-list`, `model_provider.py:242-283`). The critic is the same GPTModel with `output_layer` replaced by a plain `torch.nn.Linear(hidden, 1)` that gathers from sequence-parallel regions and returns float32 (`LinearForLastLayer`, `model_provider.py:25-55`, installed at `:234-235`).

**Sequence packing + context parallelism (the hairiest subsystem).** Training batches are **packed `thd` varlen streams**: per microbatch, each sample's tokens are CP-sliced, concatenated, padded to a multiple of `tp_size × data_pad_size_multiplier`, and described by `PackedSeqParams(cu_seqlens, qkv_format="thd")` — note `cu_seqlens * cp_size` because thd wants original lengths (`data.py:77-124`). A `bshd` mode pads every sample to a rollout-wide `max_seq_len` instead (`actor.py:214-222`; `data.py:70-75`). Two CP layouts coexist:

- **Zigzag ring-attention CP (default)**: each sample is split into `2·cp_size` chunks; rank r owns chunks `r` and `2·cp_size−1−r` (`slice_with_cp`, `cp_utils.py:175-216`). All response-extraction code must then compute, per sample, which slice of the local logits corresponds to which token range — `get_logits_and_tokens_offset_with_cp` returns `(chunk_size, chunks_offset, logits_offset, tokens_offset)` including the off-by-one logit shift (`cp_utils.py:9-50`); the author's own comment: "TODO: this is super ugly... do better abstraction" (`loss.py:120`). Reconstructing a full response (needed by GSPO/GAE/REINFORCE++) is done *differentiably*: each rank zero-pads its chunks into a full-length tensor and the group runs `dist.nn.all_reduce` (`all_gather_with_cp`, `cp_utils.py:123-172`).
- **Allgather-CP ("DSA mode")**: concatenate *all* sequences globally, pad so chunks are equal, and give each rank one contiguous `1/cp_size` slice (`data.py:78-96`). Response logprobs are then redistributed back to the zigzag layout with one differentiable all-reduce per result key (`_allgather_cp_redistribute`, `loss.py:145-228`), so downstream loss code is layout-agnostic.

**PP/VPP and the microbatch schedule.** slime reuses Megatron's `get_forward_backward_func()` pipeline schedules verbatim for both training and forward-only passes (`model.py:319, 482-492`), passing a list of `DataIterator`s, one per VPP stage, that share the same `rollout_data` dict and index schedule (`data.py:332-336`). Under dynamic batch size the per-step microbatch count is forced divisible by `microbatch_group_size_per_vp_stage` (`data.py:357-361`). Advantage computation and metric logging run only on `mpu.is_pipeline_last_stage()` (`loss.py:606`, `data.py:402`); intermediate PP stages early-return from `compute_advantages_and_returns`.

**Resharding train→rollout layout (parameter level).** Part I §4.3 covered the transport; the actual reshard is: Megatron's TP/PP/EP-sharded params → full HF-format tensors, recomputed per bucket on the fly. `HfWeightIteratorDirect` pre-partitions local params into buckets ≤ `update_weight_buffer_size` accounting for TP replication (`update_weight/hf_weight_iterator_direct.py:108-120`), then per bucket (`:43-105`): (1) the owning rank uploads its CPU-backup tensor to GPU, others allocate empty buffers; (2) async **broadcast over the PP group** from `info.src_rank`; (3) async **broadcast over the EP group** for `.experts.` params; (4) batched async **TP all-gather** (`all_gather_params_async`). The TP gather in `all_gather_param` (`update_weight/common.py:15-50`) handles two layout quirks worth quoting:

```python
if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
    param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
    param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
# this is bug in megatron's grouped moe.
if "linear_fc2.weight" in name:
    if partition_dim == 0:
        partition_dim = 1
```

i.e. GLU gate/up halves are interleaved per TP shard and must be re-grouped, and grouped-MoE fc2 reports the wrong partition dim. Expert params gather over the *expert*-TP group, not regular TP (`common.py:27-32`). Names are globalized first — VPP layer offsets via `get_transformer_layer_offset`, EP expert-index offsets `ep_rank * num_experts // ep_size`, regex-rewritten for decoder/MTP layers (`_named_params_and_buffers_global`, `common.py:160-237`). The full tensors are then converted to HF naming/layout by per-architecture modules under `megatron_to_hf/` (dispatch via `convert_to_hf` at `hf_weight_iterator_direct.py:36-40`; see §12.1). The "bridge" alternative (`--megatron-to-hf-mode bridge`) delegates the same job to NVIDIA Megatron-Bridge.

### 10.7 Optimizer, LR schedule, grad-clip, gradient accumulation

**Optimizer construction** harvests `OptimizerConfig` fields straight off the argparse namespace — the AGENTS.md "derive from the typed structure" pattern in the wild (`model.py:172-184`):

```python
for f in dataclasses.fields(OptimizerConfig):
    if hasattr(args, f.name):
        kwargs[f.name] = getattr(args, f.name)
config = OptimizerConfig(**kwargs)
optimizer = get_megatron_optimizer(config=config, model_chunks=model, ...)
```

so `--lr` (default **1e-6**, `arguments.py:746`), `--clip-grad` (default **1.0**, `arguments.py:744`), bf16/fp16 master-weight handling, and distributed-optimizer sharding are all Megatron's; gradient clipping happens inside `optimizer.step()`, which returns `(update_successful, grad_norm, num_zeros_in_grad)` (`model.py:516`).

**LR schedule** is iteration-based but counted in *samples*: `train_iters = num_rollout × rollout_batch_size × n_samples_per_prompt // global_batch_size`, `lr_decay_steps = lr_decay_iters × global_batch_size` (`model.py:111-123`), and after each optimizer step `opt_param_scheduler.step(increment=args.global_batch_size)` (`model.py:520`). Per-param-group LR is logged each step (`model.py:711-712`).

**Gradient accumulation structure.** One rollout = `num_steps_per_rollout = num_local_samples // (global_batch_size / dp_size)` optimizer steps (`data.py:321-324`); each step accumulates over `num_microbatches[step_id]` microbatches inside Megatron's `forward_backward_func` (`model.py:482-492`) with `no_sync`/`start_grad_sync` hooks wired for overlap (`model.py:587-602`). The microbatch count is fixed (`num_local_gbs // micro_batch_size`) or dynamic, as covered in Part I §3.4 (first-fit packing into `max_tokens_per_gpu × cp_size` bins, DP-MAX all-reduce, Karmarkar-Karp-balanced partitions; `slime/utils/data.py:285-296`; `data.py:338-380`). (Side observation: `--log-probs-max-tokens-per-gpu` is parsed and defaulted at `arguments.py:635, 1699-1700` but consumed nowhere in this checkout — the logprob passes reuse the training schedule.)

**Step validity & resets.** `train_one_step` zeroes grad buffers before *and* after stepping (`model.py:392-395, 522-525`), can pre-check `found_inf`/NaN-or-Inf grad norm and skip the step (`model.py:494-505`), and asserts `update_successful` (`model.py:516-519`). `--reset-optimizer-states` zeroes Adam `step/exp_avg/exp_avg_sq` at the start of every rollout's `train()` — fully on-policy optimizer memory (`model.py:607-627`). With distributed optimizer + `--overlap-param-gather`, forward pre-hooks are disabled for the first step of each rollout so a bad load doesn't propagate through the first all-gather, then re-enabled (`model.py:548-550, 636-645, 664-671`).

### 10.8 Checkpointing & resume semantics

**What is saved.** `save(rollout_id, ...)` calls Megatron's `save_checkpoint(iteration, model, optimizer, opt_param_scheduler, ...)` with forward pre-hooks disabled around it (`model.py:751-776`) — **the Megatron "iteration" number IS the rollout_id**. Saved state = model + distributed-optimizer shards + LR scheduler + RNG (suppress optimizer with `--no-save-optim`, which "disables training resumption", `arguments.py:725-733`). `--async-save` uses Megatron's async dist-ckpt machinery; `save_model` first finalizes any pending async save, and the driver forces a blocking finalize on the last rollout (`actor.py:511-536`; `train.py:51-64`). Optionally a parallel HF-format export goes through Megatron-Bridge (`save_hf_model`, `model.py:779-814`). The rollout data-source cursor is checkpointed separately by the RolloutManager (`train.py:63-64`, Part I §4.4). On ROCm, the dist-ckpt async writer is monkey-patched (`model.py:831-837`); on all platforms `checkpoint.py` patches out `validate_non_overlapping_shards_metadata` because it "is really slow for large models with many shards" (`slime/backends/megatron_utils/checkpoint.py:13-90`).

**What is loaded.** slime's `load_checkpoint` sniffs the format: a `latest_checkpointed_iteration.txt` (or `iter_NNNNNNN` dir) means Megatron dist-ckpt → delegate to Megatron; otherwise it's a HF checkpoint loaded via Megatron-Bridge (`bridge.load_hf_weights(ddp_model)`) with `iteration = 0` and `optimizer.reload_model_params()` to refresh fp16/bf16 master weights (`checkpoint.py:97-152`). A critic whose checkpoint lacks (or shape-mismatches) the `output_layer` gets it re-randomized after load, again followed by `reload_model_params()` (`model.py:43-95, 841-854`).

**Resume semantics.** `initialize_model_and_optimizer` returns the loaded iteration; `start_rollout_id = loaded_rollout_id + 1` (`actor.py:84-88`). All actors' start ids are asserted identical and become `args.start_rollout_id`, and the data source is restored to `start_rollout_id - 1` (`placement_group.py:154-178`); the driver loop is simply `for rollout_id in range(args.start_rollout_id, args.num_rollout)` (`train.py:67`). Cold-start fallback lives in arg validation: if `--load` isn't a Megatron checkpoint, set `no_load_optim/no_load_rng/finetune`, repoint `load` to `--ref-load` (or HF checkpoint in bridge mode) and force `start_rollout_id = 0` (`arguments.py:1643-1668`). Ref/teacher checkpoints are loaded by *temporarily mutating* `args.load/no_load_optim/no_load_rng/finetune`, running the same `load_checkpoint` into the live model, snapshotting to a CPU tag, then restoring args (`load_other_checkpoint`, `actor.py:593-621`) — the model on GPU ends up holding ref weights until the next `_switch_model("actor")`.

### 10.9 Training-side GPU memory management

Part I §3.5 covered the offload choreography from the scheduler's view; the train-actor internals:

- **Whole-process VRAM pause/resume**: `sleep()` = `clear_memory(clear_host_memory=True)` (which calls `torch._C._host_emptyCache()`, `slime/utils/memory_utils.py:11-16`) → `destroy_process_groups()` → `torch_memory_saver.pause()`; `wake_up()` reverses (`actor.py:156-177`). A safety margin is reserved via `torch_memory_saver.memory_margin_bytes` (default **1 GiB**, `--train-memory-margin-bytes`, `actor.py:79-82`; `arguments.py:120-125`). During weight sync while offloaded, the updater runs under `torch_memory_saver.disable()` so transfer allocations aren't tracked (`actor.py:566-569`), and the weights-getter can read tensors **directly from the saver's CPU backup** without resuming GPU memory (`_maybe_get_cpu_backup` via `torch_memory_saver.get_cpu_backup(x)`, `update_weight/common.py:129-141`).
- **Host-RAM model copies**: every extra logical model (ref/teacher/old_actor/rollout_actor) costs one pinned-CPU full copy in `TensorBackuper` (`tensor_backper.py:54-61`); `--disable-weights-backuper` swaps in the hash-only noop when no extra tags are needed (`actor.py:95-104`).
- **Activation memory**: slime adds no recompute machinery of its own — Megatron's flags pass straight through, and every shipped recipe enables full recompute (`scripts/run-qwen3-4B.sh:82`: `--recompute-granularity full`). slime-specific knobs sit *above* the transformer: `--recompute-loss-function` wraps the entire loss computation (which holds float32 `[T, V]`-adjacent intermediates) in `torch.utils.checkpoint(..., use_reentrant=False)` (`loss.py:1177-1178`), and `--log-probs-chunk-size` chunks the logprob/entropy kernels over T (`ppo_utils.py:653-669`).
- **Fragmentation control**: packed token streams are always padded to `tp_size × data_pad_size_multiplier` (default 128, `arguments.py:1272`) "to reduce memory fragmentation and maybe make the computation faster" (`data.py:106-110`); bshd mode pads the whole rollout to one rounded `max_seq_len` (`actor.py:214-222`). `clear_memory()` (sync + `gc.collect()` + `empty_cache`) brackets init and non-offload steps (`model.py:842, 854`; `train.py:42-49`). `--manual-gc` disables the Python GC during the train loop "to align the timing of garbage collection across ranks" (`model.py:629-634`).

## 11. Sample & data lifecycle

Part I §3.2–3.3 covered the rollout loop from the scheduler's view; this section follows the *data*: what a sample physically is, every transformation from prompt file to optimizer step, and the exact drop/retry/carry-over policies. The lifecycle spans three address spaces: the **RolloutManager actor** (CPU, owns `Sample` objects), the **SGLang server subprocess** (sees only HTTP payloads), and the **Megatron train actors** (see only a `dict` of lists — `Sample` objects never cross into training). The conversion point is `RolloutManager._convert_samples_to_train_data` (`slime/ray/rollout.py:682`).

### 11.1 The full life of one sample (numbered call chain)

**Phase A — birth from the prompt file (once at startup)**

1. `RolloutManager.__init__` builds the data source by dotted path: `data_source_cls = load_function(self.args.data_source_path); self.data_source = data_source_cls(args)` (`slime/ray/rollout.py:359-360`). Default class `RolloutDataSourceWithBuffer` constructs a `Dataset` from `args.prompt_data` (`slime/rollout/data_source.py:71-84`).
2. `Dataset.__init__` streams the file via `read_file` — `.jsonl` line-by-line or `.parquet` via `pyarrow.iter_batches`, with an optional row-slice suffix `path@[start:end]` parsed by regex (`slime/utils/data.py:25-78`). For each row it builds the prompt (`_build_messages` handles `<image>`/`<video>`/`<audio>` placeholder splitting, `slime/utils/data.py:130-192`), optionally applies the chat template (`data.py:229-236`), and materializes a prototype `Sample(prompt=..., label=..., metadata=..., multimodal_inputs=...)` (`data.py:250-257`).
3. Prompts longer than `rollout_max_prompt_len` are dropped **once, at load time**, by batch-tokenizing all prompts (`filter_long_prompt`, `data.py:81-127`). This is the first drop policy in the pipeline.

**Phase B — selection for a rollout step**

4. Driver: `rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))` (`train.py:71`) → `RolloutManager.generate` → `_get_rollout_data` → `call_rollout_fn(self.generate_rollout, ...)` (`slime/ray/rollout.py:484, 583`).
5. The default rollout fn enters the asyncio loop and pulls prompts on demand: `samples = data_source(args.over_sampling_batch_size)` inside `while state.remaining_batch_size < target_data_size` (`slime/rollout/sglang_rollout.py:423-426`).
6. `RolloutDataSourceWithBuffer.get_samples` drains the replay buffer first, then the dataset (`slime/rollout/data_source.py:177-189`). `RolloutDataSource.get_samples` advances a cursor `sample_offset` with epoch wraparound, then **deep-copies each prompt `n_samples_per_prompt` times** and stamps identity (`data_source.py:107-118`):

   ```python
   for _ in range(self.args.n_samples_per_prompt):
       sample = copy.deepcopy(prompt_sample)
       sample.group_index = self.sample_group_index
       sample.index = self.sample_index
       self.sample_index += 1
   ```
   `index` is a *monotonic global counter across the whole run* (saved/restored in checkpoints, `data_source.py:127-156`), not a position in the batch.

**Phase C — generation request**

7. One asyncio task per group (`GenerateState.submit_generate_tasks`, `sglang_rollout.py:136-149`) → `generate_and_rm_group` assigns each sample a `session_id = str(uuid.uuid4())` (`sglang_rollout.py:319-321`) and fans out one `generate_and_rm` task per sample, optionally pinning per-sample deterministic seeds `sampling_seed = rollout_seed + idx` (`sglang_rollout.py:324-331`).
8. `generate_and_rm` acquires the fleet semaphore, then `generate()` (`sglang_rollout.py:260-277`). `_prepare_prompt_ids` decides the token source: reuse `sample.tokens` if present (partial-rollout resume), processor for multimodal, else `tokenizer.encode(sample.prompt, add_special_tokens=False)` (`sglang_rollout.py:42-61`).
9. The HTTP payload is token-native: `payload = {"sampling_params": ..., "return_logprob": True}` plus `input_ids` (or `image_data`+`text` for multimodal) (`sglang_rollout.py:174-189`). On first generation, `sample.tokens` is seeded with the prompt ids (`sglang_rollout.py:191-192`).

**Phase D — tokens & logprobs captured**

10. The response is consumed without re-tokenization (`sglang_rollout.py:204-222`):

    ```python
    new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
    new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]
    ...
    sample.rollout_log_probs += new_response_log_probs
    ```
    Note the `+=` accumulation everywhere: a resumed partial sample keeps extending the same lists. MoE routing traces arrive base64-encoded and are reshaped to `[len(tokens)-1, num_layers, topk]` (`sglang_rollout.py:224-232`).
11. `sample.update_from_meta_info` maps SGLang's `finish_reason` to status — `length→TRUNCATED`, `abort→ABORTED`, `stop→COMPLETED` — and accumulates `SpecInfo`/`PrefixCacheInfo`/`weight_versions` (`slime/utils/types.py:153-174`).

**Phase E — reward** (plumbing detailed in §11.4 / §12.2)

12. Per-sample: `sample.reward = await async_rm(args, sample)` unless already set or `group_rm` (`sglang_rollout.py:294-301`); group-mode: `batched_async_rm(args, group)` after `asyncio.gather` of the group (`sglang_rollout.py:336-340`).

**Phase F — collection, filtering, abort**

13. The collection loop accepts groups in completion order; a dynamic filter can drop a whole group (decrementing `remaining_batch_size`, which re-triggers over-sampling) (`sglang_rollout.py:429-452`). Once `rollout_batch_size` groups are accepted, stragglers are aborted; with `--partial-rollout` the drained partial groups are stamped `sample.metadata["start_rollout_id"] = rollout_id` and returned (`sglang_rollout.py:374-381`), then pushed into the buffer by `generate_rollout`: `data_source.add_samples(aborted_samples)` (`sglang_rollout.py:622-623`). Accepted data is re-sorted by `group[0].index` for determinism (`sglang_rollout.py:464`).

**Phase G — Sample → train dict**

14. Back in the manager: flatten groups, trim `len(data)` to a multiple of `global_batch_size` (tail samples silently discarded; or compute `dynamic_global_batch_size` to force exactly one optimizer step) (`slime/ray/rollout.py:586-605, 609-634`).
15. `_post_process_rewards` performs GRPO group normalization *here, on the CPU actor, before any tensor exists* (§10.3 place 1, `rollout.py:655-680`). Both `raw_rewards` and normalized `rewards` are kept.
16. `_convert_samples_to_train_data` builds the `RolloutBatch` dict of plain lists (`rollout.py:694-748`) — this is the moment `Sample` objects die; only selected fields survive (full key map in §11.3). Loss masks are instantiated as `[1] * response_length` if absent, and `sample.remove_sample` zeroes them (`rollout.py:708-718`).
17. `_split_train_data_by_dp` computes `total_lengths = [len(t) for t in data["tokens"]]`, partitions sample indices per DP rank (round-robin or seqlen-balanced), and ships one dict per rank: `rollout_data_refs.append(Box(ray.put(rollout_data)))` (`rollout.py:754-805`). `Box` is a 7-line wrapper whose only job is to stop Ray auto-dereferencing the ObjectRef when passed through another remote call (`slime/utils/misc.py:97-103`).

**Phase H — into training**

18. Each Megatron actor fetches **only its own shard**: `rollout_data = ray.get(rollout_data_ref[dp_rank].inner)`, pops `partition`, and reorders `total_lengths` to match it (`slime/utils/data.py:299-310`).
19. `MegatronTrainRayActor._get_rollout_data` tensorizes: `tokens` → `torch.long` CUDA tensors, `loss_masks` → `torch.int` CUDA tensors (`slime/backends/megatron_utils/actor.py:190-195`); `rollout_log_probs`/`teacher_log_probs` are CP-sliced **per sample on the CPU lists** via `slice_log_prob_with_cp` before becoming float32 CUDA tensors (`actor.py:224-247`) so they line up with the context-parallel shard each rank will compute.
20. `get_data_iterator` builds the micro-batch schedule (Part I §3.4; `slime/backends/megatron_utils/data.py:290-385`). The iterator is index-based; the **same iterators are `reset()` and replayed** for the ref forward, the actor forward, and the train pass (`slime/backends/megatron_utils/model.py:244-246`; `actor.py:414-448`).
21. Per micro-batch, `get_batch` fetches the requested keys, CP-slices and concatenates tokens into a packed `thd` stream with `cu_seqlens`/`PackedSeqParams`, and aligns loss masks to the *shifted* token stream: `loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)` — left-pad covers prompt positions (minus the one-token shift), right-pad the last position (`data.py:131-146`).
22. Reference / actor log-prob forwards: `forward_only(get_log_probs_and_entropy, ...)` runs eval-mode pipeline passes and stores results back **into the same `rollout_data` dict** under `store_prefix + key` — producing `ref_log_probs`, `teacher_log_probs`, and `log_probs` as lists of per-sample response tensors; with dynamic batching the per-microbatch outputs are scattered back to original sample order via `micro_batch_indices` (`model.py:339-357`; called at `actor.py:413-448`).
23. `compute_advantages_and_returns(args, rollout_data)` mutates the dict in place (§10.3 place 2), writing `rollout_data["advantages"]`/`["returns"]` (`slime/backends/megatron_utils/loss.py:596-741`). The old-policy logprob source is chosen here and in the loss: `"rollout_log_probs" if args.use_rollout_logprobs else "log_probs"` (`loss.py:596`, `loss.py:825`).
24. Training consumes the dict through `get_batch` with the full key list `["tokens", ..., "log_probs", "ref_log_probs", "values", "advantages", "returns", "rollout_log_probs", ..., "teacher_log_probs"]` (`model.py:420-441`); `policy_loss_function` recomputes current log-probs from logits and forms the PPO ratio `ppo_kl = old_log_probs - log_probs` (`loss.py:831-887`), with optional TIS off-policy correction using both `train_log_probs` and `rollout_log_probs` (`loss.py:893-932`).

**Phase I — death**

25. The per-rank `rollout_data` dict is a local variable of `MegatronTrainRayActor.train`; nothing persists it past the step (the only outliving artifacts are CPU weight backups and optional debug dumps, `actor.py:490, 496`). On the driver, `rollout_data_ref` is rebound every loop iteration (`train.py:71`), dropping the previous step's `Box`ed ObjectRefs and letting Ray GC the object-store copies. Samples that were generated but not selected exist only in the rollout function's local `all_data`/`data` lists and vanish when it returns — except partial-rollout groups explicitly re-buffered (hop 13).

### 11.2 `Sample`: every field's producer and consumer

The single rollout-side currency (`slime/utils/types.py:8-174`), quoted minus the nested accumulators:

```python
@dataclass
class Sample:
    group_index: int | None = None
    index: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] | None = None
    multimodal_train_inputs: dict[str, Any] | None = None
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None
    rollout_routed_experts: list[list[int]] | None = None
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    train_metadata: dict | None = None
    session_id: str | None = None
    non_generation_time: float = 0.0
    spec_info: SpecInfo = ...
    prefix_cache_info: PrefixCacheInfo = ...
```

| Field | Producer (writes) | Consumer (reads) |
|---|---|---|
| `group_index` | `RolloutDataSource.get_samples` (`data_source.py:112`) | nothing in core train path (metrics/debug only); grouping is positional |
| `index` | `data_source.py:113` (global counter); eval re-stamps per dataset (`sglang_rollout.py:551`) | deterministic re-sort (`sglang_rollout.py:464, 590`); shipped as `sample_indices` (`rollout.py:702`) |
| `prompt` | `Dataset.__init__` (`utils/data.py:250-257`) | tokenization (`sglang_rollout.py:61`), multimodal payload text (`:187`), RM payload (`rm_hub/__init__.py:35-39`) |
| `tokens` | seeded with prompt ids then appended per generation call (`sglang_rollout.py:191-192, 211`) | resume-from-tokens (`sglang_rollout.py:58-59`); becomes `train_data["tokens"]` (`rollout.py:695`) → model `input_ids` (`model.py:288`) and shifted-label targets in `get_log_probs_and_entropy` (`loss.py:428-440`) |
| `response`, `response_length` | appended per call (`sglang_rollout.py:212-213`) | RM input (`rm_hub/__init__.py:62-65`); `response_lengths` drives prompt/response split everywhere: loss-mask alignment (`data.py:139-141`), per-sample logprob extraction (`loss.py:442-452`) |
| `label` | dataset `label_key` (`utils/data.py:253`) | rule-based RMs (`rm_hub/__init__.py:63-85`) |
| `reward` | `async_rm`/`batched_async_rm` (`sglang_rollout.py:291-300, 338-340`); custom generate fns may pre-fill (`:297-298`) | dynamic filter (`filter_hub/dynamic_sampling_filters.py:10`); group norm (`rollout.py:655-680`); `get_reward_value` selects by `args.reward_key` when reward is a dict (`types.py:146-147`) |
| `loss_mask` | `generate_and_rm` zeroes the off-policy prefix under partial rollout (`sglang_rollout.py:247-248`); extended `+[1]*new` per call (`:216-218`); default-instantiated at conversion (`rollout.py:710-711`) | `train_data["loss_masks"]` → CP-aligned `full_loss_masks` (`data.py:131-161`), loss denominators (`loss.py:1153`), advantage whitening masks (`loss.py:689`) |
| `weight_versions` | appended from engine `meta_info` (`types.py:165-166`) | **no consumer in `slime/`** (grep: only producer) — serialized diagnostic for multi-version partial rollouts |
| `rollout_log_probs` | `sglang_rollout.py:220-222` | TIS/ratio when `use_rollout_logprobs` (`loss.py:596, 825, 911`), mismatch metric `train_rollout_logprob_abs_diff` (`loss.py:977-979`) |
| `rollout_routed_experts` | `sglang_rollout.py:224-232` | `fill_routing_replay` records per-layer expert ids into `RoutingReplay` then **deletes the key** before training (`actor.py:289-333`) |
| `remove_sample` | custom filters only (e.g. `rollout_sample_filter_path`); never set in core | zeroes the loss mask at conversion (`rollout.py:716-717`) — soft-drop that keeps batch geometry |
| `teacher_log_probs` | OPD rollout variants; or Megatron teacher forward writes dict key (`actor.py:423-433`) | OPD KL on advantages (`loss.py:554-568, 676-682`) |
| `status` | `update_from_meta_info` (`types.py:168-174`); abort path (`sglang_rollout.py:262`) | short-circuit on resume (`sglang_rollout.py:251-255`); `truncated` flag in train data (`rollout.py:701`); eval truncation metric (`sglang_rollout.py:596`) |
| `metadata` | dataset `metadata_key` (`utils/data.py:254`); abort stamps `start_rollout_id` (`sglang_rollout.py:378-379`) | per-sample `rm_type` override (`rm_hub/__init__.py:60-61`), `raw_reward`/`round_number` overrides at conversion (`rollout.py:723-731`) |
| `train_metadata` | custom rollout fns only | shipped as `train_data["metadata"]` (`rollout.py:740-741`) |
| `session_id` | `generate_and_rm_group` (`sglang_rollout.py:319-321`) | consistent-hashing router header (`sglang_rollout.py:196-198`) |
| `spec_info` / `prefix_cache_info` | accumulated per engine call (`types.py:160-163`) — accumulators because partial rollout spans multiple calls (`types.py:159` comment) | rollout metric aggregation (`rollout.py` metrics block) |
| `multimodal_train_inputs` | processor output cached on first tokenization (`sglang_rollout.py:52-55`) | `train_data["multimodal_train_inputs"]` → GPU (`actor.py:196-212`) → concatenated per micro-batch and passed as extra `forward_kwargs` (`data.py:164-174`, `model.py:295-296`) |

Serialization: `to_dict`/`from_dict` round-trip through plain dicts (used for debug dump/load at `rollout.py:573, 646-653`); `from_dict` derives valid kwargs from `Sample.__dataclass_fields__` and `setattr`s unknown keys (`types.py:129-144`).

### 11.3 The train-side batch: `RolloutBatch`

Not a class — a documented dict alias: `RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]` (`types.py:190`). Key lineage:

| Key | Created at | Mutated at | Final consumer |
|---|---|---|---|
| `tokens` | `rollout.py:695` (list[list[int]]) | GPU long tensors `actor.py:190-192`; packed/CP-sliced per micro-batch `data.py:77-128` | model `input_ids` + shifted labels |
| `loss_masks` | `rollout.py:707-719` | GPU int tensors `actor.py:193-195`; padded/aligned `data.py:131-161` | loss reducers, whitening |
| `rewards` (normalized) / `raw_reward` | `rollout.py:689-700` | — | `compute_advantages_and_returns` (`loss.py:598`) / pass-rate logging (`data.py:508, 589-593`) |
| `truncated`, `sample_indices`, `response_lengths` | `rollout.py:700-702` | — | metrics; response/prompt split |
| `rollout_log_probs` | `rollout.py:734-735` | CP-sliced float32 GPU `actor.py:224-247` | ratio/TIS (`loss.py:825, 911`) |
| `partition`, `total_lengths` | `rollout.py:761-774, 794-800` | `partition` popped + `total_lengths` reordered in `process_rollout_data` (`utils/data.py:303-308`) | micro-batch packing (`data.py:344-376`) |
| `dynamic_global_batch_size` | `rollout.py:801-803` | forwarded into each batch (`data.py:56-57`) | step count `data.py:322-324` |
| `log_probs`, `ref_log_probs`, `teacher_log_probs`(megatron), `entropy` | `forward_only` store (`model.py:339-357`) | — | loss + KL (`loss.py:825-970`) |
| `kl`, `advantages`, `returns`, `values`, `opd_reverse_kl` | `compute_advantages_and_returns` (`loss.py:622, 740-741`), critic `get_values` | — | `policy_loss_function` / `value_loss_function` (`loss.py:824, 1038-1050`) |

The hand-off wrapper: `RolloutFnTrainOutput(samples, metrics)` with a legacy-compat shim `call_rollout_fn` that wraps bare lists (`slime/rollout/base_types.py:7-26`).

### 11.4 Reward computation placement

Rewards run inside the RolloutManager's asyncio event loop, interleaved with generation — *not* a separate service phase. Crucially, the RM call happens **outside the generation semaphore** (`async with state.semaphore:` ends at `sglang_rollout.py:277`; `async_rm` awaited at `:300`), so slow rewards don't block GPU admission. Skip conditions: completed samples re-entering from the buffer assert `sample.reward is not None` and never re-score (`sglang_rollout.py:251-255`); aborted samples skip RM entirely (`:285-286, 295-296`); custom generate functions may pre-fill (`:297-298` comment). Group-level RM (`--group-rm`, `arguments.py:1201`) defers all rewards to after the whole group finishes (`sglang_rollout.py:279-281, 336-340`), for RMs that rank within a group. Dispatch, retries, and the override layers are in §12.2; post-processing (group normalization, `reward_key` selection, `metadata["raw_reward"]` per-sample override) is centralized in `_post_process_rewards` (§10.3; `rollout.py:655-680, 723-727`; `types.py:146-147`), overridable wholesale by `custom_reward_post_process_path` (`rollout.py:656-657`, loaded at `:364-366`).

After 10 remote-RM failures the exception propagates up through `generate_and_rm` → the group task → `task.result()` at `sglang_rollout.py:431`, crashing the rollout step — there is **no per-sample reward-failure drop policy**; the retry loop *is* the policy (see §12.2 for the retry code).

### 11.5 Drop points, carry-over, and staleness policies

**What gets dropped (5 distinct drop points, in pipeline order):**

1. *Load time*: over-length prompts (`utils/data.py:81-127`).
2. *Dynamic filter*: whole groups failing `dynamic_sampling_filter_path` (default example `check_reward_nonzero_std`: GRPO zero-variance groups, `filter_hub/dynamic_sampling_filters.py:9-15`); drop reasons are counted into `rollout/dynamic_filter/drop_*` metrics (`filter_hub/base_types.py:24-37`). Dropped groups are *not* re-buffered.
3. *Surplus completions*: groups that finish after `target_data_size` is met but pass the filter are simply not appended — the code is explicit: `# NOTE: here we have not stored all the unused samples back to the data buffer.` (`sglang_rollout.py:448-451`). Without `--partial-rollout` the abort drain also discards everything (`sglang_rollout.py:371-372`).
4. *Trim*: tail samples beyond a `global_batch_size` multiple (`rollout.py:598-604`), or `wasted = num_samples - dynamic_gbs` under dynamic GBS (`rollout.py:627`).
5. *Soft drop*: `remove_sample=True` → all-zero loss mask, sample still occupies batch space (`rollout.py:716-717`).

**What gets carried over:** only `--partial-rollout` aborted groups. They re-enter through `RolloutDataSourceWithBuffer.add_samples` (group-shape asserted, `data_source.py:198-211`) and are drained buffer-first next step via the pluggable `buffer_filter` (default FIFO `pop_first`, `data_source.py:225-229`; override via `--buffer-filter-path`, `arguments.py:407-415`). On resume, `_prepare_prompt_ids` reuses `sample.tokens` (`sglang_rollout.py:58-59`) and `generate` asserts status `PENDING|ABORTED` (`:160-162`), appending new tokens to the old ones — the request is token-exact across weight versions.

**Staleness accounting for carried samples:** three mechanisms — (a) `metadata["start_rollout_id"]` stamps the birth step (`sglang_rollout.py:378-379`); (b) `weight_versions` lists every engine weight version that contributed tokens (`types.py:165-166`); (c) `--mask-offpolicy-in-partial-rollout` zeroes the loss mask on tokens generated under old weights: `sample.loss_mask = [0] * sample.response_length` before resuming (`sglang_rollout.py:247-248`), then `+[1]` for fresh tokens (`:216-218`). The help text states the intent: "only on-policy generated tokens will be used in training" (`arguments.py:368-374`).

**Off-policy correction instead of dropping:** `--use-rollout-logprobs` makes the *engine's* logprobs the old-policy term (`loss.py:596, 825`); TIS (`--use-tis`) computes truncated importance weights from the train↔rollout logprob mismatch and can apply rejection-sampling masks, with denominators rebuilt from the modified masks (`loss.py:893-932`).

**Trainer-side staleness machinery:** `--keep-old-actor` maintains CPU snapshots and, when `update_weights_interval == 1`, a two-deep queue rotated after every weight sync: `rollout_actor → old_actor; actor → rollout_actor` (`actor.py:114-119, 579-588`), so the "old policy" forward (`self._switch_model("old_actor")`, `actor.py:435`) matches what the engines actually generated with. In `train_async.py` staleness is bounded by syncing weights only every `update_weights_interval` steps, draining in-flight generation first (`train_async.py:69-73`).

**Epoch-boundary leakage:** the buffer is never cleared between epochs or checkpoints — `save()` persists only the dataset cursor (`data_source.py:127-133`), so buffered partial groups are **lost on restart** (they are not in the state dict), a real but undocumented drop point.

### 11.6 Epochs, shuffling, resume

**Epoch arithmetic** lives in the cursor, not in a DataLoader. `get_samples` wraps: take the dataset tail, increment `epoch_id`, reshuffle, take the head (`data_source.py:93-103`). Therefore one logical "epoch" can straddle a wrap inside a single rollout step. `num_rollout_per_epoch = len(data_source) // rollout_batch_size` (`rollout.py:474-476`) sizes the run when `--num-epoch` is given (`placement_group.py:190-194`) — note it counts *prompt groups* fetched per step, ignoring over-sampling churn, so heavy dynamic filtering makes real epochs shorter than nominal.

**Shuffling** is a deterministic per-epoch permutation of the *original* order — `random.seed(self.seed + new_epoch_id); ... self.samples = [self.origin_samples[i] for i in permutation]` (`utils/data.py:268-276`) — idempotent per epoch (`if self.epoch_id == new_epoch_id: return`), which is what makes resume reproducible: replaying `shuffle(epoch_id)` regenerates the exact order.

**Resume** is a 5-integer state dict checkpointed alongside model checkpoints: `{sample_offset, epoch_id, sample_group_index, sample_index, metadata}` saved to `{save}/rollout/global_dataset_state_dict_{rollout_id}.pt` (`data_source.py:123-136`). The driver saves it on the same cadence as model checkpoints (`train.py:63-64`); on startup, `args.start_rollout_id` is derived from the loaded Megatron checkpoint iteration (all ranks must agree, `placement_group.py:154-171`) and the manager loads the matching cursor: `rollout_manager.load.remote(args.start_rollout_id - 1)` (`placement_group.py:177-178`), re-applying `dataset.shuffle(self.epoch_id)` (`data_source.py:159-160`). Missing state file degrades to a warning + fresh cursor (`data_source.py:146-148`). Eval datasets are separate, cached per `(path-config, checkpoint, chat-template)` key in a module-global `EVAL_PROMPT_DATASET`, with no cursor at all — every eval scores the full dataset (`sglang_rollout.py:483, 511-530`).

A second resume path bypasses the dataset entirely: `--load-debug-rollout-data` deserializes dumped samples (`Sample.from_dict`) and feeds them straight into conversion, letting training be replayed/regression-tested without engines (`rollout.py:568-581`).

## 12. Extension surfaces: adding a model/reward/backend

### 12.1 Adding a new model — there is no registry; there are three contracts

slime has **no model registry class**. "Supporting a model" means satisfying three independent contracts, each with its own extension mechanism. Tracing Qwen2 (the simplest fully-wired model) through all three:

**Contract 1 — Megatron-side architecture definition.** The architecture is *pure argparse*: each model ships a shell fragment of Megatron CLI flags, e.g. `scripts/models/qwen3-4B.sh:1-17`:

```bash
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   ...
   --qk-layernorm
)
```

The default `model_provider` builds a stock mcore `GPTModel` from these args (`slime/backends/megatron_utils/model_provider.py:122-237`). Three escape hatches, in increasing invasiveness:

1. `--spec module function` — a custom transformer-layer spec, resolved by Megatron's `import_module(args.spec)`; slime additionally allows the spec to be a *callable* `spec(args, config, vp_stage)`, and even allows the result to be a whole model provider ("e.g. glm-omni VL model") (`model_provider.py:140-154`). This is how non-standard attention models plug in: `scripts/models/qwen3-next-80B-A3B.sh:17` → `--spec "slime_plugins.models.qwen3_next" "get_qwen3_next_spec"`, implemented in `slime_plugins/models/qwen3_next.py`. So `slime_plugins/models/` is not auto-discovered — it is reached only via the `--spec` dotted path.
2. `--custom-model-provider-path` — replaces the provider entirely; loaded with `load_function`, signature documented in the help string (`slime/utils/arguments.py:140-147`), wrapped so a critic role still gets `LinearForLastLayer` output (`model_provider.py:62-82`).
3. `--megatron-to-hf-mode bridge` — bypasses hand-written specs altogether: `AutoBridge.from_hf_pretrained(args.hf_checkpoint)` builds the provider, with `import slime_plugins.megatron_bridge  # noqa: F401  # register custom bridges` as the one import-side-effect registry in the codebase (`model_provider.py:84-105`).

Every provider is finally wrapped by `wrap_model_provider_with_freeze` to apply `--only-train-params-name-list` / `--freeze-params-name-list` regex freezing (`model_provider.py:242-266, 269-283`).

**Contract 2 — Megatron→HF weight-name conversion (for weight sync into SGLang).** The "registry" is a hard-coded substring-match if/elif chain in `_convert_to_hf_core` (`slime/backends/megatron_utils/megatron_to_hf/__init__.py:36-60`):

```python
elif "qwen2" in model_name or "qwen3" in model_name:
    converted_named_tensors = convert_qwen2_to_hf(args, name, param)
...
else:
    raise ValueError(f"Unsupported model: {model_name}")
```

`model_name` is **derived from the HF config class name**, not declared: `model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name` (`slime/backends/megatron_utils/actor.py:132`; `--model-name` exists as an override, `arguments.py:220-228`). So `Qwen2Config` → `"qwen2config"` → matches `"qwen2"`. A converter is one pure function mapping a full (already TP-gathered, per §10.6) Megatron param to ≥1 HF named tensors — `convert_qwen2_to_hf` (`megatron_to_hf/qwen2.py:5-71`) shows the canonical work: regex on `module.module.decoder.layers.(\d+).(...)`, interleaved-QKV split via `param.view(args.num_query_groups, -1, head_dim, hidden).split([value_num_per_group, 1, 1], dim=1)` (`qwen2.py:27-36`), and `mlp.linear_fc1.weight` → `gate_proj`/`up_proj` chunk (`qwen2.py:52-57`). **Adding a model = write `convert_<model>_to_hf`, import it in `megatron_to_hf/__init__.py:1-12`, add an elif branch.** The chain is only consulted in `"raw"` mode; the strategy is selected by a 2-entry dict keyed on `args.megatron_to_hf_mode` in `HfWeightIteratorBase.create` (`update_weight/hf_weight_iterator_base.py:10-15`), where `"bridge"` mode delegates naming to megatron-bridge instead. The iterator contract is a single method with a memorable docstring: `get_hf_weight_chunks` = "megatron_model.to_hf_magically().named_parameters()" (`hf_weight_iterator_base.py:24-29`). The direct implementation also shows the cross-rank machinery a new model inherits for free: param-info exchange across PP/EP with min-src-rank dedup and an all-rank consistency assert (`update_weight/hf_weight_iterator_direct.py:138-211`), and bucketing to `--update-weight-buffer-size` with TP-replication accounting (`hf_weight_iterator_direct.py:108-135`).

**Contract 3 — the HF checkpoint itself.** SGLang loads `model_path=args.hf_checkpoint` (`slime/backends/sglang_utils/sglang_engine.py:520`), and at parse time slime cross-validates the Megatron args against the HF config — `hidden_size`, `num_layers`, `tie_word_embeddings` vs `untie_embeddings_and_output_weights` (inverted compare), `rms_norm_eps`, `rope_theta` — raising a single aggregated `AssertionError` (`slime/backends/megatron_utils/arguments.py:33-77`, invoked at `:116-118`). The help text for `--hf-checkpoint` makes the design explicit: weights are always overwritten by Megatron before training, so the HF ckpt only needs the right *architecture* (`arguments.py:209-218`).

### 12.2 Adding a new reward — string dispatch with three override layers

The reward "hub" is one async function with an if/elif chain on `rm_type` (`slime/rollout/rm_hub/__init__.py:55-91`):

```python
async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    ...
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_"):]
```

Resolution priority: (1) `--custom-rm-path` dotted function (signature `def custom_rm(args, sample) -> float`, `arguments.py:1209-1218`) short-circuits everything, including the batch path where the custom fn receives the whole list (`rm_hub/__init__.py:99-102`); (2) **per-sample** `sample.metadata["rm_type"]` — datasets can mix reward types within one run (`:61`); (3) global `--rm-type`. The `boxed_` prefix is a composable answer-extraction modifier (`:64-66`). Built-ins: `remote_rm`, `deepscaler`, `dapo`, `math`, `f1`, `gpqa`, `ifbench` (lazily imported, `:83-85`), `random`; unknown non-empty types raise `NotImplementedError` (`:88-91`) — so adding a rule-based RM = add a module in `rm_hub/`, import it, add an elif.

**Batching** is in-name-only by default: `batched_async_rm` fans out to per-sample `async_rm` via `asyncio.gather` (`rm_hub/__init__.py:103-105`); a real batched implementation requires `custom_rm_path` ("Ensure the custom reward function is implemented in batch mode", `:99-102`).

**Remote RM failure handling** (`rm_hub/__init__.py:34-52`): a module-global shared `aiohttp.ClientSession` (64-connection TCP pool, 120 s total timeout, `:22-31`) and exponential backoff with jitter:

```python
for attempt in range(max_retries):   # max_retries=10
    try:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as e:
        if attempt + 1 >= max_retries: ... raise
        backoff = min(2**attempt, 30) + random.random()
```

Downstream, two more pluggable hooks complete the reward pipeline: `--custom-reward-post-process-path` replaces the default GRPO group mean/std normalization (`slime/ray/rollout.py:655-680`), and `--custom-convert-samples-to-train-data-path` replaces sample→tensor-dict conversion (`rollout.py:367-371, 686-687`; both declared `arguments.py:1219-1236`). Related but separate: dynamic-sampling filters return a `DynamicFilterOutput(keep, reason)` dataclass, with a legacy-bool adapter `call_dynamic_filter` and a `MetricGatherer` that turns drop reasons into `rollout/dynamic_filter/drop_{reason}` counters (`slime/rollout/filter_hub/base_types.py:5-37`; reference impl `check_reward_nonzero_std`, `filter_hub/dynamic_sampling_filters.py:9-15`).

### 12.3 Adding a new rollout backend — four seams of increasing depth

1. **Replace one sample's generation**: `--custom-generate-function-path` substitutes only `def generate(args, sample, sampling_params)` inside the stock rollout fn (`arguments.py:376-384`); it is consulted *per sample* (a sample can carry its own `generate_function_path`, e.g. from eval dataset config) and may or may not accept `evaluation=` — checked via `inspect.signature` (`sglang_rollout.py:265-277`).
2. **Replace the whole rollout function**: `--rollout-function-path` (default `slime.rollout.sglang_rollout.generate_rollout`) with the documented contract `def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`, minimum sample fields `tokens`, `response_length`, `reward`, `status` (`arguments.py:230-243`). Eval defaults to the same fn (`arguments.py:1765-1766`). The data source is independently swappable via `--data-source-path` (default `slime.rollout.data_source.RolloutDataSourceWithBuffer`, `arguments.py:523-528`); both are materialized with `load_function` in `RolloutManager.__init__` (`slime/ray/rollout.py:359-363`).
3. **Keep slime's rollout fn, but bring your own servers**: `--rollout-external` + `--rollout-external-engine-addrs` (`arguments.py:447-459`). The `SGLangEngine` actor then becomes a *validator-proxy*: instead of spawning a server it polls the external server's `/health_generate`, fetches `/get_server_info`, and asserts field-by-field equality of the ServerArgs it *would* have used — minus a skip-list of legitimately differing fields (`model_path`, ports, seeds, `mem_fraction_static`; `slime/backends/sglang_utils/sglang_engine.py:172-194, 565, 606-616`).
4. **A genuinely different training backend** is the one closed seam: `--train-backend` is pre-parsed with `choices=["megatron"]` (`arguments.py:1413`), the abstract surface exists (`TrainRayActor` with abstract `sleep/wake_up/train/save_model/update_weights`, `slime/ray/train_actor.py:101-123`), but `actor_group.py` hard-imports the only implementation: `from slime.backends.megatron_utils.actor import MegatronTrainRayActor; actor_impl = MegatronTrainRayActor` (`slime/ray/actor_group.py:79-81`). A new backend means a new package under `slime/backends/` mirroring `megatron_utils`' five obligations: `init`, checkpointed model/optimizer, log-prob forward, `train`, and a weight-updater pair.

The deployment topology of the SGLang fleet itself is also config-extensible without code: `--sglang-config` YAML declares per-model `server_groups` with `worker_type` regular/prefill/decode/encoder/placeholder and per-group `overrides` of raw `ServerArgs` fields (`slime/backends/sglang_utils/arguments.py:122-134`), resolved by `_resolve_sglang_config` (falls back to one regular group of `rollout_num_gpus`; `slime/ray/rollout.py:1118-1139`) and applied last-wins inside `_compute_server_args` with dash→underscore normalization warnings (`sglang_engine.py:577-595`).

### 12.4 The config/args system that carries all of the above

**One `argparse.Namespace`, assembled in three phases, then mutated by validators, then copied into every actor.** There is no config dataclass, no pydantic, no hydra (OmegaConf appears only for `--eval-config`, `arguments.py:1555-1559`).

**Phase 0 — pre-parse.** `_pre_parse_mode()` runs a throwaway parser for the four flags that change *how parsing itself proceeds* (`--train-backend`, `--debug-rollout-only`, `--debug-train-only`, `--load-debug-rollout-data`), explicitly so they aren't registered twice (`arguments.py:1405-1418`). `debug_train_only` ⇒ skip SGLang arg parsing entirely (`:1428`).

**Phase 1 — SGLang args, namespaced by parser hijack.** `sglang_parse_args` builds a *separate* parser and uses `parse_known_args` so it only consumes its own flags (`slime/backends/sglang_utils/arguments.py:174-197`). The trick is monkey-patching `parser.add_argument` before handing the parser to SGLang's own `ServerArgs.add_cli_args(parser)`: every flag gets rewritten `--foo-bar` → `--sglang-foo-bar` and every `dest` → `sglang_*`, while a `skipped_args` list (model_path, tp_size, ports, nnodes…) suppresses flags slime must control itself (`sglang_utils/arguments.py:41-113`). Result: the *entire* upstream SGLang server CLI is exposed under a prefix with zero maintenance, and slime-managed fields are injected later in `_compute_server_args`, which reflects over `dataclasses.fields(ServerArgs)` to pick up every `args.sglang_<field>` — deriving the mapping from the typed source of truth rather than a hand-written list (`sglang_engine.py:567-575`).

**Phase 2 — Megatron + slime args in one parser.** `megatron_parse_args(extra_args_provider=add_slime_arguments, ignore_unknown_args=True)` reuses Megatron's parser; slime's ~250 args are added through Megatron's `extra_args_provider` hook, organized as ~20 nested `add_*_arguments` closures (`arguments.py:36-1390`), and `ignore_unknown_args=True` lets the `--sglang-*` flags pass through unparsed (`megatron_utils/arguments.py:112-114`, comment at `utils/arguments.py:1436-1438`). `reset_arg` mutates Megatron's existing `action.default`s in place (e.g. `--distributed-timeout-minutes` default 10, `arguments.py:19-32, 101-102`). Then both side-namespaces are merged by brute `setattr` loop (`arguments.py:1447-1454`), and three validators run in order: `slime_validate_args`, `megatron_validate_args` (Megatron's own + slime extras like forcing `variable_seq_lengths=True` and alltoall dispatcher, `megatron_utils/arguments.py:12-30`), `sglang_validate_args` (`utils/arguments.py:1456-1462`).

**Validation is normalization.** `slime_validate_args` (250 lines, `arguments.py:1591-1841`) rewrites the namespace heavily: checkpoint fallback logic that decides between resuming Megatron ckpt vs finetune-from-`ref_load` and sets `start_rollout_id = 0` (`:1656-1668`); derives `use_critic = advantage_estimator == "ppo"` (`:1719`); expands `--offload` into `offload_train`+`offload_rollout` and then **`del args.offload`** so stale reads crash (`:1724-1727`); colocate forcing of offload flags and `rollout_num_gpus` (`:1746-1758`); `over_sampling_batch_size` default+invariant (`:1782-1788`); derived defaults (`eval_max_context_len`, `rollout_max_prompt_len = max_context_len - 1` "so that there is at least one generated token to compute loss", `:1818-1832`).

**YAML enters at four bounded points**, all override-style on top of CLI: (1) `--custom-config-path` — flat key/value `setattr` onto args, logging when it overrides (`:1810-1816`); (2) `--megatron-config-path` — per-role (`actor`/`critic`) override lists; unknown keys warn but are set anyway, `num_nodes`/`num_gpus_per_node` are refused ("GPU allocation always follows CLI args"), and string values are type-coerced to the existing attribute's type because "YAML safe_load doesn't parse scientific notation (e.g. 1e-5) as float" (`:1467-1540`); the critic role also force-disables actor-only features (`:1490-1495`); (3) `--sglang-config` server-group topology (§12.3); (4) `--eval-config` per-dataset eval configs via OmegaConf (`:1548-1588`).

**Plumbing to workers.** The whole Namespace is passed *by value* through Ray into each actor: `RolloutManager.options(...).remote(args, pg)` (`slime/ray/placement_group.py:184-187`) and `actor.init.remote(args, role, ...)` (`actor_group.py:101-109`) — every actor gets a pickled copy, so post-hoc driver mutations don't propagate; the driver therefore finishes all arg mutation that actors need *before* those calls (e.g. `args.start_rollout_id` is set from the actors' returned checkpoint iteration at `placement_group.py:170-171`, and `num_rollout` is computed from the RolloutManager's epoch length before training actors exist, `placement_group.py:190-194`). Inside actors, args keeps being mutated locally (`args.rank = dist.get_rank()`, `train_actor.py:69`; `self.args.vocab_size` from HF config, `actor.py:121-125`). Two values ride the namespace as a cross-process side channel: `args.wandb_run_id` (set by primary wandb init, consumed by every secondary, `wandb_utils.py:79, 152-154`) and `args.sglang_model_routers` written back after server startup "for custom rollout functions" (`rollout.py:1112-1113`).

## 13. Startup sequence & failure handling

### 13.1 Order-of-operations: launch → first step

1. **Launch**: `python train.py --...` (no torchrun) → `parse_args()` (3-phase, §12.4) → `train(args)` — `train.py:105-107`.
2. **GPU claim & deterministic ordering**: `create_placement_groups(args)` sizes one PACK PG by mode (`placement_group.py:79-99`), `ray.get(pg.ready())` blocks for resources, then a throwaway 1-GPU `InfoActor` per bundle reports `(node_ip, gpu_id)` and bundles are re-sorted IP-then-GPU (`placement_group.py:41-76`; Part I §4.1).
3. **Primary W&B init** on the driver: `init_tracking(args)` → `init_wandb_primary` sets `args.wandb_run_id` — `train.py:13`; `wandb_utils.py:22-79`.
4. **Rollout fleet up first** ("need to initialize rollout manager first to calculate num_rollout", `train.py:15-17`): `create_rollout_manager` → `RolloutManager.remote(args, pg)` (`placement_group.py:183-187`). Inside the actor `__init__`: `load_function` the rollout fn / data source / custom hooks (`rollout.py:359-371`) → `start_rollout_servers` (`rollout.py:379`), which per model: launches the `sglang_router` daemon process and sleeps 3s + `assert process.is_alive()` (`rollout.py:952-960`), builds `ServerGroup`s via the `_make_group` closure computing `needs_offload` from GPU overlap (`rollout.py:1020-1056`), fires all `engine.init.remote(...)` non-blocking, then one `ray.get(all_init_handles)` barrier (`rollout.py:1094-1102`).
5. **Each `SGLangEngine.init`** (inside step 4's barrier): compute `ServerArgs` (§12.4) → spawn the SGLang HTTP server as a `multiprocessing.Process(target=launch_server)` with spawn start method (`sglang_engine.py:63-68`) → node-rank-0 polls `/health_generate` every 2s, raising if the child dies (`sglang_engine.py:73-77, 82-100`) → registers with the router via `POST /workers` (`sglang_engine.py:203-220`).
6. **Driver-side rollout finalization**: `num_rollout = num_rollout_per_epoch * num_epoch` from `len(data_source) // rollout_batch_size` (`placement_group.py:190-194`; `rollout.py:474-476`); if `--offload-rollout`, immediately `offload()` the fresh engines to make room for Megatron init (`placement_group.py:200-201`).
7. **W&B reinit with engine metrics**: driver fetches the router address and re-creates the primary run with `x_stats_open_metrics_endpoints={"sgl_engine": f"{router_addr}/engine_metrics"}` — `train.py:19-21`; `wandb_utils.py:82-135`.
8. **Train actor creation**: `create_training_models` → `RayTrainGroup(..., num_gpus_per_actor=0.4)` (`placement_group.py:111-134`) → per rank, `TrainRayActor.options(...).remote(world_size, rank, master_addr, master_port)` on its sorted bundle, rank 0 electing the master port that all later ranks receive (`actor_group.py:86-99`). The remote-class env is set here: `NCCL_CUMEM_ENABLE=0`, and for offload the `torch_memory_saver` `LD_PRELOAD` + `TMS_INIT_ENABLE(_CPU_BACKUP)=1` (`actor_group.py:53-73`). The constructor only stamps `MASTER_ADDR/PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK` env vars (`train_actor.py:29-48`).
9. **Distributed + Megatron init**, all inside `ray.get(actor_model.async_init(...))` (`placement_group.py:154-161`): `TrainRayActor.init` does `torch.cuda.set_device` → `dist.init_process_group(backend, timeout=distributed_timeout_minutes)` → gloo group → pynvml NUMA affinity (`train_actor.py:58-92`); then `MegatronTrainRayActor.init`: `monkey_patch_torch_dist()` (reloadable PGs for offload) → megatron `init(args)` = `set_args` + `mpu.initialize_model_parallel(...)` + seeds + tokenizer + a microbatch calculator initialized only "to pass some validation in megatron" + optional `--custom-megatron-init-path` hook (`actor.py:57-60`; `initialize.py:33-104`); HF config/tokenizer read **serialized one local rank at a time** behind gloo barriers "to prevent concurrent writing bug" (`actor.py:67-72`).
10. **Model load**: `initialize_model_and_optimizer` = `get_model(provider)` + `OptimizerConfig` built by reflecting `dataclasses.fields(OptimizerConfig)` against args + `load_checkpoint(...)` returning `iteration` (`model.py:146-186, 817-856`); back in `actor.init`, `start_rollout_id = loaded_rollout_id + 1` (`actor.py:84-88`), CPU weight backup tags created (`weights_backuper.backup("actor")`, `actor.py:95-105`), ref/teacher/old-actor checkpoints loaded into extra tags (`actor.py:107-119`), updater class chosen by `colocate` (`actor.py:127-134`), and if offloading, the actor immediately `sleep()`s (`actor.py:139-142`).
11. **Cross-wiring**: driver asserts all actors agree on `start_rollout_id` and adopts it (`placement_group.py:168-171`); `set_rollout_manager` pushes the manager handle + the DP-size parallel config to the RolloutManager (`placement_group.py:173-175`; `train_actor.py:125-128`); dataset cursor restored via `rollout_manager.load(start_rollout_id - 1)` (`placement_group.py:177-178`).
12. **First weight push before any rollout** — the HF ckpt in the engines is never trusted: `onload_weights` (if offloaded) → `actor_model.update_weights()` → optional bitwise check (`--check-weight-update-equal` snapshot/compare via the engines' `weights_checker`, `train.py:26-36`; `placement_group.py:196-198`; `sglang_engine.py:370-371`) → `onload_kv` (`train.py:35-36`). The loop then enters step `rollout_id = start_rollout_id` (`train.py:67-71`).

### 13.2 Failure handling: elastic rollout, fail-fast training

Part I §3.6 named the hooks; here is the full machinery. **The only elastic component is the SGLang fleet.** All of it is gated by `--use-fault-tolerance` (default off, `arguments.py:462-468`).

*Detection* — `RolloutHealthMonitor`, one daemon thread per `ServerGroup` inside the RolloutManager process (`rollout.py:386-391`). It is started paused, and pause/resume is slaved to the offload cycle because "cannot health check when offloaded" (`health_monitor.py:10-21, 51`): `RolloutManager.generate/eval` resume it (`rollout.py:481, 497`), `offload()`/`recover_updatable_engines()` pause it (`rollout.py:511, 533`). Each cycle calls `ray.get(engine.health_generate.remote(timeout=...))` per engine (`health_monitor.py:151`), with a configurable first-wait grace period after each resume "for large MoE models to be ready" / deepgemm compile (`health_monitor.py:117-127`; `arguments.py:481-486`). There is no degraded state: any exception ⇒ kill, and the kill takes out **all node-shards of that multi-node engine** (`_kill_engine` iterates `rollout_engine_id * nodes_per_engine ..`, calling `engine.shutdown.remote()` then `ray.kill(engine)` and setting the slot to `None`, `health_monitor.py:160-177`). `shutdown` first deregisters from the router (3 API versions handled) then `kill_process_tree(self.process.pid)` (`sglang_engine.py:315-347`).

*What happens to in-flight requests* — nothing engine-specific: the rollout loop's HTTP layer retries any failed `POST /generate` up to 60× with 1s sleeps against the **router** URL (`http_utils.py:165-198`), so requests that were on the dead engine get re-routed to surviving workers (the router's own health check is disabled, `rollout.py:948`; removal happens via slime's explicit deregistration). Only after 60 failures does the exception propagate — and `generate_rollout_async` has **no try/except around `task.result()`** (`sglang_rollout.py:430-431`), so an exhausted retry kills the whole rollout step, hence the driver.

*Repair point* — recovery is deliberately deferred to the next weight sync, not done at detection time: `MegatronTrainRayActor.update_weights` on rank 0 calls `rollout_manager.recover_updatable_engines.remote()` (`actor.py:543-546`). `RolloutServer.recover` re-runs `start_engines` for `None` slots across all groups concurrently (it reuses the exact creation path; `start_engines` skips non-None slots, `rollout.py:94-96, 269-281`), then puts the new engines into the canonical memory state — release, then resume WEIGHTS only — with frozen models reloading from disk (`rollout.py:283-310`). Back on the training side, `num_new_engines > 0` triggers `connect_rollout_engines` to rebuild IPC/NCCL sync groups before pushing weights, after which the counter is cleared via Ray RPC (`actor.py:548-564`). The freshly pushed weights are what actually "heal" the new engine; `onload_kv` later restores its cache. A `weight_version` check can verify the sync end-to-end in CI (`actor.py:571-577`). The whole path is integration-tested by built-in fault injection: in CI, `generate` at `rollout_id >= 2` calls `simulate_crash` on engine 0 and sleeps `interval + timeout + 5` for the monitor to notice (`rollout.py:411-430, 482-483`; `sglang_engine.py:489-498`).

**Training failures are fail-fast by design.** There is no try/except anywhere in `train.py`'s loop, no actor restart policy (`max_restarts` is never set), and a thrown training step propagates: actor exception → `ray.get(actor_model.async_train(...))` raises on the driver (`train.py:85`) → process exit. Hangs convert to failures via the 10-minute default `--distributed-timeout-minutes` on the NCCL process group (`arguments.py:102`; `train_actor.py:63-66`). Recovery is restart-from-checkpoint, and the state needed is exactly what `save(rollout_id)` persists: Megatron ckpt (+ optional async-save finalization, `actor.py:511-536`) and the dataset cursor `rollout_manager.save(rollout_id)` (`train.py:51-64`; §11.6) — on relaunch `load_checkpoint` yields the iteration that becomes `start_rollout_id` (§13.1 steps 10-11). Sample-level soft failures have their own channel: `Sample.Status.FAILED` is documented as "recoverable or non-critical failure during generation (e.g., tool call failure, external API error, parsing error)… may still contain partial valid output" (`types.py:31-39`) — for custom rollout functions to use, distinct from scheduler-initiated `ABORTED`.

## 14. Engineering details worth copying (and anti-patterns)

### 14.1 Observability machinery (all of it copyable)

- **One W&B run shared by many processes.** The driver creates the primary run (`mode="shared", x_primary=True`, `wandb_utils.py:65`); the RolloutManager and the Megatron "main rank" (DP0/TP0/last-PP — `initialize.py:108-113`) attach as secondaries with the same `wandb_run_id` and `x_update_finish_state=False` (`rollout.py:381`; `actor.py:62-63`; `wandb_utils.py:151-192`). All metric families get explicit step axes: `train/*`→`train/step`, `rollout/*`, `multi_turn/*`, `passrate/*`, `perf/*`→`rollout/step`, `eval/*`→`eval/step` (`wandb_utils.py:195-204`). The single `log()` chokepoint also tees to TensorBoard (`logging_utils.py:49-55`).
- **Engine metrics are scraped, not pushed**: every SGLang server is launched with `enable_metrics=True` "regardless of --sglang-enable-metrics" so `/engine_metrics` exists (`sglang_engine.py:543-545`); after servers start, the primary W&B run is *finished and re-created* with `x_stats_open_metrics_endpoints={"sgl_engine": f"{router_addr}/engine_metrics"}` so W&B's system-stats monitor scrapes router-aggregated Prometheus metrics — guarded by a check that the router is the slime fork ("Only customized sglang_router … supports uploading metrics", `wandb_utils.py:82-135`; wiring at `train.py:19-21`, `rollout.py:394-405`).
- **Step timing = a singleton Timer with two idioms.** `Timer()` accumulates named durations and asserts on double-start (`timer.py:15-52`); `@timer`/`with timer(...)` wrap `sleep`, `wake_up`, `update_weights`, `save_model`, `data_preprocess`, `(ref_)log_probs`, `actor_train` (`actor.py:156, 168, 345, 362, 478, 511, 538`). The clever bit is how *wait* time is measured without instrumenting the driver: `MegatronTrainRayActor.init` is decorated `@with_defer(lambda: Timer().start("train_wait"))` so the wait clock starts the moment init returns (`actor.py:45`; `timer.py:92-103`), and `train_actor` runs under `inverse_timer("train_wait"), timer("train")` — `inverse_timer` *stops* the wait clock on entry and restarts it on exit (`actor.py:408`; `timer.py:83-89`). `log_perf_data_raw` then derives `perf/step_time = train_wait + train` and `perf/wait_time_ratio` (`train_metric_utils.py:38-42`).
- **Throughput accounting, train side**: `process_rollout_data` stashes this rank's `total_lengths` on the singleton (`Timer().seq_lens = total_lengths`, `utils/data.py:307`); per rollout, `log_perf_data` (PP-last/TP0/DP0 only) computes `perf/log_probs_tflops` and `perf/ref_log_probs_tflops` as analytic forward FLOPs ÷ time, `perf/actor_train_tflops = 3 × fwd_flops / actor_train_time`, and `perf/actor_train_tok_per_s = sum(seq_lens)/time` (`train_metric_utils.py:25-36`; FLOPs model divided across world size at `megatron_utils/data.py:598-610`). The timer dict is deep-copied and reset every rollout so all `perf/*_time` values are per-step (`train_metric_utils.py:16-18`).
- **Throughput accounting, rollout side**: `RolloutManager.generate` wall-clocks the whole step and hands it to `_log_rollout_data` (`rollout.py:478-486, 1175-1190`), which logs `perf/rollout_time`, `perf/tokens_per_gpu_per_sec = Σ response_len / time / rollout_num_gpus`, and a straggler diagnostic pair — `longest_sample_tokens_per_sec` plus the same with reward/tool time subtracted (`longest_sample_tokens_per_sec_without_non_generation`), computed from per-sample `non_generation_time` (`rollout.py:1205-1235`). Quality-side metrics from the same hook: response-length stats, `truncated_ratio`, `repetition_frac`, GRPO `zero_std/count_{reward}` group histogram (`rollout.py:1193-1252`), speculative-decoding accept rate/length and `prefix_cache_hit_rate` aggregated from the per-sample `SpecInfo`/`PrefixCacheInfo` accumulators (`rollout.py:1255-1273`). Megatron-side `log_rollout_data` separately logs the *training view* of the same batch — per-sample-mean of every tensor field with CP-correct masked means, on TP0/PP-last only, plus optional `multi_turn`, `passrate` (pass@k from reward groups), and correct-sample length-percentile metrics (`megatron_utils/data.py:388-595`). Both custom-loggable via `--custom-rollout-log-function-path` / `--custom-eval-rollout-log-function-path` returning bool to suppress defaults (`arguments.py:385-404`; `rollout.py:1142-1146, 1176-1179`).
- **Per-request tracing**: an in-house span system (`trace_utils.py`) attaches a `trace` carrier dict to each `Sample` (`bind_trace`, `trace_utils.py:155-171`) and appends typed events with span/parent ids (`_append_event`, `:429-457`); spans wrap `generate_and_rm`, the SGLang call (capturing PD-disaggregation phase durations like `pd_prefill_forward_duration`, `pd_transfer_speed_gb_s` from `meta_info`, `trace_utils.py:16-32`; `sglang_rollout.py:200-202`), and reward calls — all failure-isolated (`_log_trace_error` downgrades to debug, `:112-113`). Since the carrier lives on the sample, traces ride along into `--save-debug-rollout-data` dumps / `--dump-details` (`rollout.py:636-653`; `arguments.py:1708-1710`). Heavier profiling: `TrainProfiler` (torch profiler + CUDA memory-history snapshots keyed by `--profile-target`, `profile_utils.py:13-42`, stepped at `actor.py:65, 152, 488`) and engine-side `start_profile`/`stop_profile` HTTP passthroughs (`sglang_engine.py:455-487`).

### 14.2 Patterns worth copying for wm-infra

1. **Reflect over the upstream dataclass instead of mirroring its fields.** Twice, at two different boundaries: `OptimizerConfig` harvested via `dataclasses.fields` (`model.py:172-184`, quoted in §10.7) and `ServerArgs` populated by reflecting `dataclasses.fields(ServerArgs)` against `args.sglang_<field>` (`sglang_engine.py:567-575`). Zero drift when Megatron/SGLang add fields. Same spirit as the parser hijack (§12.4) that re-exposes the entire SGLang CLI under `--sglang-*` with no maintained list.
2. **The 0-loss graph-liveness idiom.** `loss = loss + 0 * logits.sum()` appears twice for two different deadlocks — empty per-rank loss token sets under allgather-CP (`loss.py:1182-1188`) and empty CP chunks in the policy loss (`loss.py:971-974`). Any wm-infra CP/SP denoise-loss path that can produce empty shards needs exactly this guard, with exactly this comment style explaining the reduce-scatter deadlock.
3. **Temperature-divide logits before computing train-side logprobs** (`loss.py:417-420`) — without it, rollout (sampled at T≠1) and train logprobs are systematically mismatched and every ratio/TIS metric lies. Directly relevant to wm-infra's GRPO logprob parity work (the predict2 sigma-domain bug was the same *class* of error: compute logprobs in the same domain the sampler used).
4. **Time-derived metrics without touching the caller**: `with_defer` + `inverse_timer` (§14.1) measures driver-side wait from inside the actor. Cleaner than threading timestamps through RPC signatures.
5. **CI fault injection in production code paths**: `simulate_crash` lives on the engine and the rollout manager triggers it under a debug flag (`rollout.py:411-430`; `sglang_engine.py:489-498`) — the recovery path is exercised by the same code users run, not a mocked twin.
6. **Layout-quirk knowledge lives in one function with a comment naming the upstream bug** — `all_gather_param`'s GLU re-grouping and "this is bug in megatron's grouped moe" fc2 dim fix (`update_weight/common.py:15-50`, quoted in §10.6). wm-infra's wan/cosmos weight-export paths should concentrate per-family layout quirks the same way instead of spreading them across converters.
7. **`metadata` as the schema escape valve.** Per-sample `rm_type`, `raw_reward`, `start_rollout_id` all ride `Sample.metadata` (§11.2) — new per-sample behavior ships without touching the dataclass. Paired with `from_dict` deriving fields from `__dataclass_fields__`, the schema evolves in both directions safely.
8. **Buffer + cursor instead of DataLoader.** The 5-integer resume state (`data_source.py:123-136`) plus idempotent per-epoch shuffle (`utils/data.py:268-276`) gives exact resume semantics with no torch DataLoader machinery — the right shape for wm-infra's prompt/video-condition datasets too.

### 14.3 Anti-patterns and sharp edges (verified in this checkout)

1. **The args Namespace as a mutable, copied-by-value god object** (§12.4). It works because the driver finishes mutations before actor creation, but `args.wandb_run_id` / `args.sglang_model_routers` as cross-process side channels and `del args.offload` as a tombstone are conventions enforced by nothing. wm-infra's typed config is the better foundation — don't import this.
2. **Substring dispatch on a derived class name**: `"qwen2" in model_name` where `model_name = type(hf_config).__name__.lower()` (§12.1) — `Qwen2Config` matching `"qwen2"` is clever until a config class name embeds another's substring. A `--model-name` override exists precisely because this breaks (`arguments.py:220-228`).
3. **Dead freight ships in production**: `Sample.weight_versions` has a producer but no consumer anywhere in `slime/` (`types.py:165-166`; repo-wide grep), and `--log-probs-max-tokens-per-gpu` is parsed and defaulted but consumed nowhere (`arguments.py:635, 1699-1700`) — the logprob passes silently reuse the training schedule. Both are exactly the "duplicated constant rots" failure mode AGENTS.md warns about, in flag form.
4. **Undocumented data loss on restart**: partial-rollout buffer contents are not in the data-source state dict, so buffered groups vanish on resume (§11.5). If wm-infra adopts a partial-rollout buffer, its contents must be part of the checkpoint contract from day one.
5. **Surplus-completion discard** is acknowledged in a NOTE comment but still wastes finished, reward-scored samples (`sglang_rollout.py:448-451`) — combined with trim-to-global-batch-multiple (`rollout.py:598-604`), a heavily-filtered run can throw away a meaningful fraction of paid GPU tokens. Worth budgeting explicitly (a `wasted_samples` metric exists only for the dynamic-GBS path, `rollout.py:627`).
6. **"TODO: this is super ugly"** (`loss.py:120`) on the zigzag-CP offset math is honest: every consumer of per-sample response slices must reimplement chunk-offset arithmetic. The allgather-CP path's answer — redistribute once, keep downstream layout-agnostic (`loss.py:145-228`) — is the better abstraction and the one to copy.

## 15. Part II source-of-truth index

| Claim | Evidence |
|---|---|
| **§10 training internals** | |
| Megatron is the only train backend (no FSDP) | `slime/utils/arguments.py:1413`; `slime/backends/` contents |
| CPU-pinned weight backups + restore; noop hash variant | `slime/utils/tensor_backper.py:42-115` |
| train_actor pass ordering: ref → teacher → old-actor logprobs → adv → train → backup | `slime/backends/megatron_utils/actor.py:401-509` |
| π_old recomputed by trainer unless use-rollout-logprobs (and no mismatch metrics); ref model only when KL on | `actor.py:435-448`; `slime/ray/placement_group.py:158`; `loss.py:596, 825` |
| Rollout logprobs CP-sliced to GPU fp32 in data preprocessing | `actor.py:224-247` |
| forward_only: eval mode, pipeline forward-only, dynamic-order restore | `slime/backends/megatron_utils/model.py:309-357` |
| Full-[T,V] logprob compute; temperature scaling; shifted targets | `slime/backends/megatron_utils/loss.py:384-468, 231-289` |
| Fused vocab-parallel CE; TP-parallel entropy autograd; chunking | `slime/utils/ppo_utils.py:151-198, 649-681`; `arguments.py:156-157` |
| GRPO group mean/std normalization in RolloutManager; disable flags; auto-off at n=1 | `slime/ray/rollout.py:655-680`; `arguments.py:871-879, 1779-1780` |
| GRPO returns = broadcast scalar reward; kl arg shape-only; kl_coef vs kl_loss_coef exclusive | `ppo_utils.py:201-208`; `loss.py:629-633`; `arguments.py:1676` |
| KL estimators k1/k2/k3/low_var_kl + unbiased IS + clamp | `ppo_utils.py:11-51` |
| PPO clipped loss + dual-clip; defaults eps_clip 0.2, high=low | `ppo_utils.py:124-148`; `arguments.py:765, 1702-1703` |
| GSPO sequence-KL; OPSM masking; chunked GAE; REINFORCE++ | `ppo_utils.py:95-121, 54-92, 211-278, 506-646`; `loss.py:843-890` |
| Loss assembly: entropy term, KL loss term, empty-chunk graph hack | `loss.py:945-974` |
| TIS vanilla/icepop; denominator vs metric masks | `loss.py:744-791, 893-932, 995-1002` |
| Per-sample-mean masked reduction (CP-aware); per-token mode | `slime/backends/megatron_utils/cp_utils.py:53-120` |
| Loss-mask creation (zeroed removed samples) + stream alignment pad | `slime/ray/rollout.py:707-719`; `slime/backends/megatron_utils/data.py:139-145` |
| Loss rescale cancelling Megatron averaging; allgather-CP deadlock guard; metric protocol | `loss.py:1182-1212`; `model.py:527-544` |
| mpu.initialize_model_parallel config + order; forced distributed optimizer + varlen | `slime/backends/megatron_utils/initialize.py:33-53`; `slime/backends/megatron_utils/arguments.py:17-24, 80-84` |
| Model provider, critic value head, regex freezing | `slime/backends/megatron_utils/model_provider.py:25-55, 122-237, 242-283` |
| Packed thd batches + PackedSeqParams; bshd alternative | `data.py:70-126`; `actor.py:214-222` |
| Zigzag CP offsets / slice / differentiable gather; allgather-CP + redistribute | `cp_utils.py:9-50, 123-216`; `data.py:78-96`; `loss.py:145-228` |
| VPP iterators + divisibility clamp; last-stage-only advantage/metrics | `data.py:332-336, 357-361`; `loss.py:606` |
| Reshard: bucket → PP/EP broadcast → async TP all-gather → HF convert; GLU rechunk + fc2 dim fix; EP name offsets | `update_weight/hf_weight_iterator_direct.py:23-120`; `update_weight/common.py:15-50, 160-237` |
| OptimizerConfig harvested from dataclass fields; lr 1e-6, clip_grad 1.0 | `model.py:172-184`; `arguments.py:744-746` |
| train_iters & sample-unit scheduler; step(increment=global_batch_size) | `model.py:98-143, 516-520` |
| Grad-accum structure; dynamic first-fit + DP-MAX + balanced partitions | `data.py:290-385`; `slime/utils/data.py:285-296`; `model.py:482-492` |
| NaN/inf step skip; optimizer-state reset; pre-hook gating | `model.py:494-525, 548-550, 607-627, 636-671` |
| Save = Megatron ckpt with iteration=rollout_id; async finalize; HF export | `model.py:751-814`; `actor.py:511-536`; `train.py:51-64` |
| Load: format sniff, HF-via-bridge with iteration=0; shard-validation patch; critic head reinit | `slime/backends/megatron_utils/checkpoint.py:13-152`; `model.py:43-95, 841-854` |
| Resume: start_rollout_id = iteration+1; data-source restore; cold-start fallback | `actor.py:84-88`; `placement_group.py:154-178`; `arguments.py:1643-1668`; `train.py:67` |
| Ref/teacher load by args mutation + tag backup | `actor.py:593-621` |
| Offload: pause/resume + PG destroy/reload + 1GiB margin; saver-disabled weight sync; CPU-backup read | `actor.py:79-82, 156-177, 566-569`; `update_weight/common.py:129-141`; `arguments.py:120-125` |
| Recompute: Megatron flags pass-through (scripts); loss-fn checkpointing | `scripts/run-qwen3-4B.sh:82`; `loss.py:1177-1178`; `arguments.py:151-154` |
| Pad-to-multiple fragmentation control; clear_memory host cache; manual GC | `data.py:106-110`; `arguments.py:1272`; `slime/utils/memory_utils.py:11-16`; `model.py:629-634` |
| **§11 sample & data lifecycle** | |
| Sample dataclass: all fields, Status enum, accumulators, dict round-trip, reward_key select | `slime/utils/types.py:8-174` |
| `RolloutBatch` alias; ParamInfo | `slime/utils/types.py:177-190` |
| Dataset file readers, `path@[a:b]` slice, message building, chat template, prototype Sample | `slime/utils/data.py:25-78, 130-192, 195-257` |
| Load-time long-prompt drop | `slime/utils/data.py:81-127, 259-262` |
| Deterministic per-epoch shuffle permutation | `slime/utils/data.py:268-276` |
| Cursor + epoch wraparound; group/index stamping; deep-copy per response | `slime/rollout/data_source.py:90-118` |
| Data source state dict save/load; reshuffle on load | `slime/rollout/data_source.py:120-160` |
| Buffer-first sampling; pluggable `pop_first`; group-shape assertion on add | `slime/rollout/data_source.py:168-229`; `arguments.py:314-415` |
| Prompt-id reuse vs processor vs tokenizer | `slime/rollout/sglang_rollout.py:42-61` |
| Generate payload (`input_ids`, `return_logprob`), token/logprob append, loss-mask extend, routed experts decode | `slime/rollout/sglang_rollout.py:174-236` |
| Off-policy prefix masking; completed short-circuit; semaphore scope; per-sample vs group RM call sites | `slime/rollout/sglang_rollout.py:240-302, 310-342` |
| Abort drain; `start_rollout_id` stamp; partial-sample collection | `slime/rollout/sglang_rollout.py:345-386` |
| Surplus-group discard NOTE; sort by index; buffer re-add | `slime/rollout/sglang_rollout.py:389-480, 448-451, 464, 602-624` |
| Eval dataset module cache, no cursor | `slime/rollout/sglang_rollout.py:483-599` |
| Manager plugin seams (data source, rollout fn, reward post-process, convert override) | `slime/ray/rollout.py:359-371` |
| generate(): rollout fn → trim/dynamic GBS → convert → DP split | `slime/ray/rollout.py:478-491, 567-634` |
| Debug rollout dump/load via Sample.to_dict/from_dict | `slime/ray/rollout.py:568-581, 636-653` |
| `_convert_samples_to_train_data` full key map; loss-mask instantiation; remove_sample zeroing | `slime/ray/rollout.py:682-749` |
| DP split key lists, partition, `Box(ray.put(...))` | `slime/ray/rollout.py:754-805`; `slime/utils/misc.py:97-103` |
| Per-rank shard fetch + partition reorder | `slime/utils/data.py:299-310` |
| Tensorization + CP slicing of rollout/teacher logprobs | `slime/backends/megatron_utils/actor.py:179-252` |
| Routing replay record + key deletion | `slime/backends/megatron_utils/actor.py:260-333` |
| old_actor/rollout_actor snapshot queue | `slime/backends/megatron_utils/actor.py:95-119, 579-588` |
| get_batch packing, cu_seqlens, loss-mask `(prompt_length-1, 1)` pad, multimodal concat | `slime/backends/megatron_utils/data.py:25-176` |
| forward_only store_prefix write-back, dynamic-batch reordering; train forward key list | `slime/backends/megatron_utils/model.py:213-358, 420-441` |
| compute_advantages_and_returns inputs/outputs; logprob source selection; whitening | `slime/backends/megatron_utils/loss.py:571-741` |
| Driver loop hops; data-source save cadence; ref rebinding per step | `train.py:51-99` |
| Async staleness: update_weights_interval + drain | `train_async.py:69-73` |
| `weight_versions` has no consumer (producer only) | grep over `slime/`, `examples/`, `slime_plugins/`; producer `slime/utils/types.py:165-166` |
| `--log-probs-max-tokens-per-gpu` parsed but unused in this checkout | `arguments.py:635, 1699-1700`; repo-wide grep (no other hits) |
| **§12 extension surfaces & config** | |
| Model arch = CLI fragment per model | `scripts/models/qwen3-4B.sh:1-17` |
| `--spec` callable/spec/provider escalation; qwen3-next plugin spec | `slime/backends/megatron_utils/model_provider.py:140-154`; `scripts/models/qwen3-next-80B-A3B.sh:17` |
| custom provider path; bridge mode + plugin bridge registration; freeze wrapper | `model_provider.py:62-82, 84-120, 242-283`; `slime/utils/arguments.py:140-147` |
| Megatron→HF substring dispatch, no registry; model_name from HF config class | `slime/backends/megatron_utils/megatron_to_hf/__init__.py:1-12, 36-60`; `actor.py:132`; `arguments.py:220-228` |
| Qwen2 converter (QKV split, gate/up chunk) | `megatron_to_hf/qwen2.py:5-71` |
| raw/bridge iterator selection; `get_hf_weight_chunks` contract | `update_weight/hf_weight_iterator_base.py:4-29` |
| Param-info exchange/dedup/assert; buffer-size bucketing | `update_weight/hf_weight_iterator_direct.py:108-211` |
| HF↔Megatron config cross-validation | `slime/backends/megatron_utils/arguments.py:33-77, 112-123` |
| Reward dispatch: custom path, per-sample rm_type, boxed_ prefix, remote retry | `slime/rollout/rm_hub/__init__.py:22-105`; `arguments.py:1178-1236` |
| Dynamic filter dataclass + legacy adapter + drop metrics | `slime/rollout/filter_hub/base_types.py:5-37`; `filter_hub/dynamic_sampling_filters.py:9-15` |
| Rollout seams: custom generate (per-sample), rollout fn contract, data source | `sglang_rollout.py:265-277`; `arguments.py:230-243, 376-384, 523-528` |
| External engines: validator-proxy with field check skip-list | `slime/backends/sglang_utils/sglang_engine.py:172-194, 565, 606-616`; `arguments.py:447-459` |
| Train backend closed: choices=["megatron"], hard import, abstract base | `arguments.py:1413`; `slime/ray/actor_group.py:79-81`; `slime/ray/train_actor.py:101-123` |
| 3-phase arg parsing; pre-parse flags; namespace merge; 3 validators | `arguments.py:1405-1464` |
| SGLang CLI prefix hijack + skipped args; ServerArgs reflection | `slime/backends/sglang_utils/arguments.py:41-113, 174-197`; `sglang_engine.py:567-575` |
| `reset_arg`; nested add_*_arguments | `arguments.py:19-32, 36-1390` |
| Validators mutate: ckpt fallback, `del args.offload`, colocate forcing, derived defaults | `arguments.py:1656-1668, 1719-1727, 1746-1763, 1782-1788, 1818-1832` |
| YAML points: custom-config, megatron per-role (coercion, refused keys), sglang-config, eval-config | `arguments.py:1467-1540, 1548-1588, 1810-1816`; `sglang_utils/arguments.py:122-134`; `slime/ray/rollout.py:1118-1139`; `sglang_engine.py:577-595` |
| Args copied into actors; wandb_run_id & router side channels | `slime/ray/placement_group.py:184-187`; `actor_group.py:101-109`; `wandb_utils.py:79, 151-154`; `rollout.py:1112-1113` |
| **§13 startup & failure** | |
| Startup steps 1–12 | `train.py:9-36, 67-71, 105-107`; `placement_group.py:41-108, 111-203`; `rollout.py:359-392, 952-960, 1020-1102`; `sglang_engine.py:53-100, 203-220`; `actor_group.py:46-99`; `train_actor.py:29-92`; `actor.py:44-154`; `initialize.py:33-113`; `model.py:146-186, 817-856` |
| HF config read serialized per local rank behind gloo barriers | `actor.py:67-72` |
| Health monitor: pause/resume slaved to offload; kill whole multi-node engine; grace period | `slime/utils/health_monitor.py:10-58, 105-177`; `rollout.py:481, 497, 511, 533`; `arguments.py:462-487` |
| Engine shutdown deregisters then kills process tree | `sglang_engine.py:315-347` |
| HTTP 60×1s retry vs no try around `task.result()` | `slime/utils/http_utils.py:165-198`; `sglang_rollout.py:429-431` |
| Recovery at weight-sync; reconnect on num_new_engines; weight_version check; CI fault injection | `actor.py:543-577`; `rollout.py:269-310, 411-430, 482-483, 527-554`; `sglang_engine.py:489-498` |
| Train fail-fast; 10-min NCCL timeout; ckpt+cursor restart state; FAILED semantics | `train.py:51-64, 85`; `arguments.py:102`; `train_actor.py:63-66`; `actor.py:84-88, 511-536`; `placement_group.py:168-178`; `slime/utils/types.py:31-39` |
| **§14 observability & engineering** | |
| Shared wandb run, secondaries, step-metric axes, TB tee | `wandb_utils.py:22-79, 151-204`; `logging_utils.py:49-55`; `initialize.py:108-113` |
| Open-metrics scraping of /engine_metrics; enable_metrics forced | `wandb_utils.py:82-135`; `sglang_engine.py:543-545`; `train.py:19-21`; `rollout.py:394-405` |
| Timer singleton; with_defer + inverse_timer wait accounting; per-step reset | `slime/utils/timer.py:15-103`; `actor.py:45, 345-510`; `train_metric_utils.py:13-48` |
| tflops/tok_per_s; seq_lens stashed on Timer | `train_metric_utils.py:25-42`; `slime/utils/data.py:307`; `megatron_utils/data.py:598-610` |
| Rollout perf/quality metrics (tokens_per_gpu_per_sec, longest-sample, zero_std, spec, prefix cache) | `rollout.py:478-486, 1175-1283` |
| Megatron-side rollout/pass@k/multi-turn logging | `megatron_utils/data.py:388-595` |
| Sample-attached tracing; PD meta keys; failure-isolated | `slime/utils/trace_utils.py:16-32, 112-116, 155-171, 429-457`; `sglang_rollout.py:200-202` |
| TrainProfiler + engine HTTP profiling | `slime/utils/profile_utils.py:13-42`; `actor.py:65, 152, 488`; `sglang_engine.py:455-487` |
