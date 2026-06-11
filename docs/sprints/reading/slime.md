# slime — Architecture Reading

Repo: `/home/mingfeiguo/Desktop/slime` (THUDM/slime-style Megatron + SGLang RL post-training framework).
All `file:line` references below are relative to that repo root and were spot-checked against the source on 2026-06-10.

## 1. Repo layout & module organization

Top level: two entrypoint scripts `train.py` (synchronous loop) and `train_async.py` (one-step-ahead pipelined loop) sit at the repo root, alongside the `slime/` package, a `slime_plugins/` package (per-architecture model-bridge plugins and an HTTP rollout-buffer service), plus `docs/`, `examples/`, `scripts/`, `tests/`, `tools/` (checkpoint converters, profiling). Confirmed by the repo root listing and `known_first_party = ["slime", "slime_plugins"]` in `pyproject.toml:18`.

Package map of `slime/` (directories verified on disk):

| Directory | Owns |
|---|---|
| `slime/ray/` | All Ray orchestration: `placement_group.py` (GPU allocation + factory functions), `actor_group.py` (`RayTrainGroup`), `train_actor.py` (`TrainRayActor` base), `rollout.py` (`RolloutManager`, `RolloutServer`, `ServerGroup`), `ray_actor.py` (tiny `RayActor` base with IP/port helpers, `slime/ray/ray_actor.py:4-10`), `utils.py` (`Lock` actor, `NOSET_VISIBLE_DEVICES` env list) |
| `slime/backends/megatron_utils/` | Training backend: `actor.py` (`MegatronTrainRayActor`), `model.py`/`loss.py`/`data.py` (fwd/bwd, loss, data iterators), `checkpoint.py`, `megatron_to_hf/` (name/layout conversion), `update_weight/` (`update_weight_from_tensor.py`, `update_weight_from_distributed.py`, HF weight iterators) |
| `slime/backends/sglang_utils/` | Inference backend: `sglang_engine.py` (`SGLangEngine` actor that supervises an SGLang HTTP server subprocess), `sglang_config.py` (multi-model / multi-group YAML config), `arguments.py` |
| `slime/rollout/` | Data-generation logic: `sglang_rollout.py` (async generate loop hitting the router over HTTP), `data_source.py` (`RolloutDataSource(WithBuffer)`), `sft_rollout.py`, `rm_hub/` (reward models), `filter_hub/` (dynamic sampling filters) |
| `slime/utils/` | args parsing, http utils, timers, seqlen balancing, `health_monitor.py`, `distributed_utils.py` (gloo group, `init_process_group`) |
| `slime_plugins/rollout_buffer/` | Standalone FastAPI "Rollout Buffer Server" for agentic/external generation (`buffer.py:10-15`: `app = FastAPI(title="Rollout Buffer Server", ...)`) |
| `slime_plugins/mbridge/`, `slime_plugins/megatron_bridge/` | Per-model Megatron↔HF bridge plugins (glm4moe, qwen3_next, deepseek_v32, gpt_oss, ...) |

The split is deliberate: `slime/ray/` knows nothing about Megatron math; the only backend coupling is `actor_group.py:79-81` importing `MegatronTrainRayActor` as the actor implementation class.

Size norms: the core package is ~20k LOC. Most files sit at 100-650 lines; only three exceed 1000: `utils/arguments.py` (1841), `ray/rollout.py` (1283), `megatron_utils/loss.py` (1212). The 1841-line `arguments.py` is one giant `get_slime_extra_args_provider` containing ~19 nested `def add_*_arguments(parser)` groups (`arguments.py:36-1348`) — the config surface is centralized, not scattered. The whole framework threads a single argparse `Namespace` named `args` through every constructor (e.g. `RolloutManager.__init__(self, args, pg)`, `rollout.py:353`); there is no config-object hierarchy except the YAML-backed `SglangConfig`/`ModelConfig`/`ServerGroupConfig` (`rollout.py:1118-1139`).

## 2. Architecture overview

Three roles, all coordinated by a single driver Python process running `train.py`:

1. **Training** — `RayTrainGroup` = list of `MegatronTrainRayActor` Ray actors (1 per GPU) that together form a `torch.distributed` world (each actor sets `MASTER_ADDR/RANK/WORLD_SIZE` env vars and calls `dist.init_process_group`, `slime/ray/train_actor.py:41-48,63-66`).
2. **Rollout** — one `RolloutManager` Ray actor (CPU-only, `num_gpus=0`, `slime/ray/placement_group.py:184-187`) which owns the data source and spawns `SGLangEngine` actors; each `SGLangEngine` launches the actual SGLang inference server as a **spawned subprocess** (`launch_server_process` → `multiprocessing.Process(target=launch_server)`, `slime/backends/sglang_utils/sglang_engine.py:63-79`) and registers it with an **sglang_router** gateway that itself runs as a daemon `multiprocessing.Process` inside the RolloutManager (`slime/ray/rollout.py:952-957`).
3. **Data buffer** — not a separate service in-core: `RolloutDataSourceWithBuffer` lives inside the RolloutManager process (`slime/ray/rollout.py:359-360`); partial/aborted samples are pushed back via `data_source.add_samples(aborted_samples)` (`slime/rollout/sglang_rollout.py:621-624`, buffer impl `slime/rollout/data_source.py:168-211`).

```
driver (train.py)                                     [Ray driver process]
  │  create_placement_groups → create_rollout_manager → create_training_models
  │
  ├─ RolloutManager (Ray actor, CPU)
  │    ├─ data_source: RolloutDataSourceWithBuffer (prompts + partial-sample buffer)
  │    ├─ sglang_router  (multiprocessing.Process, HTTP gateway)
  │    ├─ RolloutHealthMonitor threads (1/ServerGroup)
  │    └─ ServerGroup[*] ── SGLangEngine (Ray actor, 0.2 GPU)  ×N
  │                            └─ SGLang HTTP server (spawned subprocess, owns TP GPUs)
  │
  └─ RayTrainGroup("actor") [+ optional "critic"]
       └─ MegatronTrainRayActor (Ray actor, 1/GPU; torch.distributed NCCL world)

per rollout_id (train.py:67-99):
  driver ──ray──▶ RolloutManager.generate(rollout_id)
                    │ asyncio fan-out: HTTP POST /generate ──▶ router ──▶ SGLang servers
                    │ rewards via rm_hub; group-norm advantages; split by DP
                    ▼ returns list[Box(ray.put(rollout_data))]   (one ObjectRef per DP rank)
  driver ──ray──▶ actor_model.async_train(rollout_id, refs) → each Megatron actor
                    ray.get's its DP shard, computes log-probs/advantages, trains
  driver ──ray──▶ actor_model.update_weights() → Megatron actors push new weights
                    into SGLang engines (CUDA-IPC tensor handles or NCCL broadcast)
```

Control-flow evidence: the driver loop `rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))` then `ray.get(actor_model.async_train(rollout_id, rollout_data_ref))` then `actor_model.update_weights()` is `train.py:71-93`. `generate()` ends with `_convert_samples_to_train_data` + `_split_train_data_by_dp` which does `rollout_data_refs.append(Box(ray.put(rollout_data)))` per DP rank (`slime/ray/rollout.py:478-491,769-805`) — i.e. **rollout→train data moves through the Ray object store**, and each training actor fetches its DP partition in `_get_rollout_data` (`slime/backends/megatron_utils/actor.py:179-191`).

Async variant: `train_async.py:35-43` keeps one rollout in flight ahead of training (`rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)` before `async_train`), with weight sync gated to `update_weights_interval` boundaries and explicitly synced *before* updating so weights never change mid-generation (`train_async.py:69-73`); colocation is asserted off (`train_async.py:11`: `assert not args.colocate, "Colocation is not supported for async training."`).

## 3. Core scheduling & orchestration

slime's "scheduling" has three layers, each with its own loop:

1. **Driver layer (`train.py`)**: a single-process Python for-loop stepping by `rollout_id`, chaining `RolloutManager.generate` and `RayTrainGroup.async_train` Ray remote calls, and — in colocated mode — choreographing GPU memory offload/onload timing.
2. **Rollout layer (`RolloutManager` Ray actor + asyncio event loop)**: turns prompt groups into async HTTP requests against sglang_router, which load-balances across SGLang servers; implements over-sampling, dynamic filtering (DAPO-style), abort-based preemption, and partial-rollout recycling. Token-level continuous batching is entirely delegated to the SGLang servers (this repo never implements it; it only calls HTTP).
3. **Training layer (Megatron actor)**: slices one rollout into `num_steps_per_rollout` optimizer steps, each step further split into micro-batches by fixed `micro_batch_size` or a dynamic token budget (first-fit packing + seqlen balancing), then handed to the Megatron pipeline engine.

GPU resources sit on one unified Ray placement group: in colocate mode actor and rollout share the same bundles (`slime/ray/placement_group.py:89-99`), training actors take 0.4 GPU each (`placement_group.py:117`) and SGLang engine actors 0.2 GPU each (`slime/ray/rollout.py:99`) — fractional quotas exist only so Ray permits colocation; real memory isolation comes from torch_memory_saver offload/onload (§3.4). Bundles are re-sorted by node IP + physical GPU id so logical ranks align with physical GPUs (`placement_group.py:41-76`).

### 3.1 Driver main loop (sync version, `train.py:67-99`)

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

Each rollout step is a strict five-beat sequence (colocated): **generate → rollout-engine offload → train → train offload → rollout onload weights → update_weights → onload KV**. Note `onload_weights` and `onload_kv` are two separate beats (`train.py:91-96`): weights come back to GPU first so `update_weights` can overwrite them in place; KV cache and CUDA graphs resume only after the weight update, avoiding stacked memory peaks.

The async variant `train_async.py:35-43` launches the next rollout's generate one step ahead (one-step-ahead pipelining):

```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
```

Training step N runs in parallel with rollout N+1 (hence `train_async.py:11` asserts `not args.colocate` — both need GPUs simultaneously). Weight sync is throttled by `update_weights_interval`, and the in-flight generate must be `ray.get`-ed before updating — comment: "sync generate before update weights to prevent update weight in the middle of generation" (`train_async.py:69-73`). Off-policy lag is therefore explicitly bounded at 1 step.

### 3.2 Rollout generation scheduling loop (the most important loop)

Entry: `RolloutManager.generate` (`slime/ray/rollout.py:478-491`) → `_get_rollout_data` (`rollout.py:583`) → `call_rollout_fn` invoking the pluggable `generate_rollout` (default `slime/rollout/sglang_rollout.py:602-624`). `generate_rollout` pushes the coroutine onto a **singleton background event-loop thread** via `run(coro)` (`slime/utils/async_utils.py:8-36`, `AsyncLoopThread` + `run_coroutine_threadsafe`), so the RolloutManager actor itself stays synchronous; all async concurrency happens on that daemon thread.

Main loop `generate_rollout_async` (`slime/rollout/sglang_rollout.py:422-452`), annotated:

```python
while len(data) < target_data_size:                            # L422 target = rollout_batch_size groups
    while state.remaining_batch_size < target_data_size:       # L423 top up when in-flight groups run low
        samples = data_source(args.over_sampling_batch_size)   # L425 pull over_sampling_batch_size prompts at once
        state.submit_generate_tasks(samples)                   # L426 one asyncio task per group
    # wait for the generation to finish
    done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)  # L429
    for task in done:
        group: list[Sample] = task.result()
        ...
        dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)   # L442
        if not dynamic_filter_output.keep:
            state.remaining_batch_size -= 1                    # L445 filtered out → decrement in-flight count; outer while re-samples
            continue
        if len(data) < target_data_size:                       # L450
            data.append(group)
```

- **Batch granularity**: the scheduling unit is a *group* — `n_samples_per_prompt` samples of the same prompt (`data_source.py:107-117` stamps `group_index`/`index`). `submit_generate_tasks` creates one asyncio task per group (`sglang_rollout.py:136-149`).
- **Over-sampling / dynamic filtering**: each refill pulls `over_sampling_batch_size` prompts (defaults to `rollout_batch_size`, `slime/utils/arguments.py:1782-1787`); completed groups pass through `dynamic_sampling_filter` (e.g. DAPO's nonzero-variance check, help text at `arguments.py:345-354`); failing groups are dropped whole, triggering re-sampling. This is slime's dynamic-sampling scheduling policy: **continuously over-produce, harvest first-completed, stop when full**.
- **Preempt on full**: once `rollout_batch_size` groups are collected, `abort` fires (`sglang_rollout.py:461`). `abort` queries the router for all worker URLs and POSTs `/abort_request {"abort_all": True}` to every server (`sglang_rollout.py:352-361`), then drains pending tasks; with `--partial-rollout`, half-finished samples (those with partial responses) are tagged with `start_rollout_id` metadata and recycled (`sglang_rollout.py:374-384`), written back to the buffer by `generate_rollout` (`sglang_rollout.py:622-623` → `RolloutDataSourceWithBuffer.add_samples`, `slime/rollout/data_source.py:198-211`). This is the system's only preemption mechanism — request-level abort, not token-level preemption.
- **Concurrency & throttling**: each sample's generation in `generate_and_rm` is gated by a global semaphore (`sglang_rollout.py:260` `async with state.semaphore`), sized `sglang_server_concurrency (default 512) × rollout_num_gpus / rollout_num_gpus_per_engine` (`sglang_rollout.py:94-96`; default at `slime/backends/sglang_utils/arguments.py:39`). After acquiring, it checks `state.aborted` for fast bail-out (`sglang_rollout.py:261-263`).
- **Priority**: there is no priority queue. The buffer dequeue policy defaults to `pop_first` (FIFO, `data_source.py:225-229`), replaceable via `--buffer-filter-path` (`data_source.py:172-175`). `RolloutDataSourceWithBuffer.get_samples` drains the buffer first (partial-rollout leftovers get continued preferentially), then reads the dataset sequentially (`data_source.py:177-189`); the dataset is epoch-modulo indexed and shuffled per epoch (`data_source.py:90-103`).
- **Routing**: each request is a direct POST to sglang_router's `/generate` (`sglang_rollout.py:158, 201`); the router load-balances across servers. `dp_rank_context` maintains a least-used dp_rank counter (`sglang_rollout.py:119-129`) for custom generation functions; session_id consistent-hash routing is optional (`sglang_rollout.py:196-199`).
- **Inline rewards**: per-sample RM scoring runs asynchronously right after generation (`sglang_rollout.py:299-301`); `group_rm` mode batches scoring after the whole group completes (`sglang_rollout.py:336-340`) — RM calls share the event loop with generation and overlap naturally.

### 3.3 Training-side micro-batch scheduling

Before returning, `RolloutManager.generate` trims to a multiple of `global_batch_size` (`slime/ray/rollout.py:590-605`), then `_split_train_data_by_dp` partitions by DP rank and `ray.put`s each shard (`rollout.py:754-805`). With `--balance-data`, DP-level sequence-length balancing uses Karmarkar–Karp-style `get_seqlen_balanced_partitions` (`rollout.py:764-767`).

Each training actor pulls its shard in `train()` (`slime/backends/megatron_utils/actor.py:179-191`, via `process_rollout_data` indexing by `partition`, `slime/utils/data.py:299-310`), then `get_data_iterator` builds the micro-batch plan (`slime/backends/megatron_utils/data.py:290-385`):

```python
num_local_gbs = global_batch_size // dp_size
num_steps_per_rollout = num_local_samples // num_local_gbs          # data.py:323-324
...
num_microbatches.append(
    get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu * cp_size))  # data.py:349-351
num_microbatches = torch.tensor(...); dist.all_reduce(num_microbatches, op=MAX, group=dp_group)  # data.py:353-354
...
partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)  # data.py:372
```

- One rollout = `num_steps_per_rollout` optimizer steps (`global_batch_size = rollout_batch_size × n_samples_per_prompt / num_steps_per_rollout`, `slime/utils/arguments.py:1769`).
- **Dynamic batch size** (`--use-dynamic-batch-size`): first-fit bin-packing against the `max_tokens_per_gpu` token budget computes the minimal micro-batch count (`slime/utils/data.py:285-296`); a DP-group all-reduce MAX keeps pipeline step counts aligned; seqlen-balanced partitioning then redistributes samples into micro-batches. This is the training side's key mechanism for digesting the rollout long tail. `DataIterator` supports an explicit `micro_batch_indices` schedule (`data.py:226-287`).
- Each step calls Megatron's `forward_backward_func`: logprob recomputation runs `forward_only` (`slime/backends/megatron_utils/model.py:319-333`), training runs `train` → `train_one_step` per step (`model.py:647-662`). After `forward_only`, samples are restored to original order via `micro_batch_indices` (`model.py:349-356`).
- Within one train step, `train_actor` orchestrates the phases (`actor.py:401-509`): ref logprob → (optional teacher logprob) → old/current actor logprob → advantages → `train` → `weights_backuper.backup("actor")`. The ref/old-actor weight copies live as CPU backups via `TensorBackuper`; `_switch_model` swaps weights in place so the same GPU memory is reused (`actor.py:95-119, 254-258`).

### 3.4 Memory/cache lifecycle interleaved with scheduling

The repo does not implement paged KV/prefix cache itself (that lives inside the SGLang server), but it fully choreographs its lifecycle:

- **Three memory tags**: `GPU_MEMORY_TYPE_WEIGHTS / KV_CACHE / CUDA_GRAPH` (imported from sglang, `slime/ray/rollout.py:15`). `RolloutServer.onload_weights` restores only weights; `onload_kv` restores KV cache + CUDA graphs (`rollout.py:326-346`) — matching the driver's two-beat onload.
- **Offload decided by GPU-range overlap**: only server groups overlapping the Megatron GPU range need offloading (`needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus`, `rollout.py:1027`); engines on rollout-only GPUs run with `enable_memory_saver=False` (`rollout.py:1032-1033`).
- **Flush radix/prefix cache before release**: `release_memory_occupation` calls `flush_cache()` first (`slime/backends/sglang_utils/sglang_engine.py:357-359`); `flush_cache` won't return 200 while requests are pending and retries up to 60 times (`sglang_engine.py:291-308`).
- **Weight update ↔ cache consistency**: `update_weights` brackets with `pause_generation` + `flush_cache` (KV produced under old weights must be invalidated) and `continue_generation` afterwards (`update_weight_from_tensor.py:144-180`). Every weight sync therefore **clears the prefix cache** — an inherent RL cost (prefix hit-rate metrics tracked at `rollout.py:1265-1273`).
- **Training-side dual**: `MegatronTrainRayActor.sleep/wake_up` swap the whole training state out/in via `torch_memory_saver.pause()/resume()`, with `destroy_process_groups()` before offload and `reload_process_groups()` after onload (NCCL communicators also hold GPU memory, `actor.py:156-177`). During `update_weights`, `torch_memory_saver.disable()` temporarily escapes the saver region so IPC staging buffers are not counted as pausable (`actor.py:566`).

### 3.5 Weight-sync paths (the key synchronization point)

`MegatronTrainRayActor.update_weights` (`actor.py:539-591`): optionally recovers dead engines via the RolloutManager, then fetches `(engines, lock, num_new_engines, gpu_counts, gpu_offsets)` (`actor.py:548-550`), rebuilding communication groups when new engines appeared. Implementation is chosen by colocation in one line — `update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed` (`actor.py:127`):

- **Colocated → `UpdateWeightFromTensor` (CUDA IPC)**: each training rank converts its Megatron shard to HF-layout chunks, flattens into a `FlattenedTensorBucket`, serializes with SGLang's `MultiprocessingSerializer` (this is what carries CUDA IPC handles), `dist.gather_object`s the blobs over a per-engine **Gloo** group (groups built per engine GPU range, `update_weight_from_tensor.py:122-135`) to the engine's source rank, which calls `ipc_engine.update_weights_from_tensor.remote(...)` — "The HTTP server will only post meta data, and the real weights will be copied directly from GPUs" (`sglang_engine.py:266-289`). After each chunk the trainer frees the bucket and calls `torch.cuda.ipc_collect()` (`update_weight_from_tensor.py:158-170`). Hybrid case: engines whose GPUs fall outside the actor range fall back to the distributed path (`update_weight_from_tensor.py:85-115`).
- **Disaggregated → `UpdateWeightFromDistributed` (NCCL broadcast)**: per PP rank, the (DP=0, TP=0) source rank forms a fresh NCCL group `f"slime-pp_{pp_rank}"` of size `sum(engine_gpu_counts) + 1` with itself as rank 0 — engines join via the HTTP endpoint `init_weights_update_group` (`update_weight_from_distributed.py:62-79,252-298`). Updates are bucketed by `update_weight_buffer_size`: params are TP all-gathered, converted to HF names, then metadata goes via Ray/HTTP (`engine.update_weights_from_distributed.remote(names, dtypes, shapes, ...)`) while tensor data is `dist.broadcast(param.data, 0, group=group)` (`update_weight_from_distributed.py:310-337`). Expert (EP) params take a separate gather path (`update_weight_from_distributed.py:166-226`). Concurrent broadcasts from multiple PP sources would collide in NCCL, so each bucket spin-acquires a global Ray `Lock` actor "to prevent dead lock" (`update_weight_from_distributed.py:234-248`; lock impl `slime/ray/utils.py:38-57` — `acquire` returns False and callers busy-retry).

Both paths bracket the update with `pause_generation` / `flush_cache` ... `continue_generation` HTTP calls from rank 0 (`update_weight_from_distributed.py:88-90,139`; `update_weight_from_tensor.py:145-147,180`).

### 3.6 Fault tolerance × scheduling

`RolloutHealthMonitor` is a per-server-group daemon thread, pausable/resumable (health checks are impossible while offloaded); dead engines are restarted before `update_weights` by `recover_updatable_engines`, which redoes offload/onload and reconnects communication groups (`slime/utils/health_monitor.py:10-59`; `slime/ray/rollout.py:385-392, 527-548`; `rollout.py:269-310` `RolloutServer.recover`; `actor.py:543-564`). The `num_new_engines` count drives the "rebuild IPC/NCCL groups?" decision.

## 4. Distributed orchestration (Ray or alternative)

**Ray is the orchestration layer, end to end.** Everything else (torch.distributed NCCL/Gloo, HTTP, CUDA IPC, multiprocessing) lives *inside* the Ray actors.

### Placement & topology
- One single placement group covers all GPUs: colocated mode allocates `actor_num_nodes * actor_num_gpus_per_node` bundles and rollout reuses them at `rollout_offset = 0`; disaggregated mode allocates `actor + rollout_num_gpus` bundles with rollout starting at offset `actor_num_nodes * actor_num_gpus_per_node` (`slime/ray/placement_group.py:89-99`). Bundles are `{"GPU": 1, "CPU": 1}` with `strategy="PACK"` (`placement_group.py:43-44`).
- Because Ray gives no stable bundle ordering, a throwaway `InfoActor` (`@ray.remote(num_gpus=1)`, `placement_group.py:14-17`) is scheduled on every bundle to report `(node_ip, gpu_id)`, then bundles are re-sorted by (node IP, GPU id) for a deterministic rank→GPU mapping (`placement_group.py:49-76`).
- Training actors use **fractional resources** `num_gpus_per_actor=0.4` (`placement_group.py:117`) and SGLang engine actors `num_gpus = 0.2` (`slime/ray/rollout.py:99`), so in colocated mode a train actor and an inference engine share the same 1-GPU bundle without Ray complaining; the engine's real GPU binding comes from `base_gpu_id` passed into SGLang's server args (`rollout.py:103-104,137`, `sglang_engine.py:533`).
- Rank-0 train actor picks a free port and the rest join its `MASTER_ADDR:PORT` (`slime/ray/actor_group.py:87-99`); env hygiene includes `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` (so Ray doesn't mask devices, `actor_group.py:58`, list in `slime/ray/utils.py:17-25`) and `NCCL_CUMEM_ENABLE=0` "because sglang will always set NCCL_CUMEM_ENABLE to 0" (`actor_group.py:54-56`).
- Rollout engines support heterogeneous **ServerGroups** per model (regular / prefill / decode / encoder / placeholder) for PD- and encoder-disaggregation, each group with its own TP size and GPU offsets, all behind one router per model (`slime/ray/rollout.py:38-59,210-217,981-1115`). Multi-node engines have one Ray actor per node sharing a `dist_init_addr` (`rollout.py:891-898`).

### Mechanism stack summary
- **Ray**: actor lifecycle, placement groups, control RPCs, rollout→train data plane (object store `ray.put`/`ray.get` per DP shard).
- **torch.distributed NCCL**: Megatron training collectives; ad-hoc train→rollout broadcast groups. **Gloo**: CPU barriers and the colocated IPC gather.
- **HTTP**: data generation (asyncio client → sglang_router → SGLang servers) and all engine control endpoints (pause/flush/continue, update-weights metadata).
- **multiprocessing**: SGLang server and router run as plain subprocesses supervised by Ray actors.
- **CUDA IPC**: zero-copy colocated weight handoff.

Colocation memory time-slicing and fault tolerance are covered in §3.4/§3.6; the `torch_memory_saver` hook is injected via `LD_PRELOAD` in the train-actor runtime env (`actor_group.py:62-73`).

## 5. Code organization style: function granularity

slime's style is **"flat, linear orchestration kept monolithic; extraction only for reuse or for a real seam"**. Single-caller helpers are rare; long functions are organized with section comments and nested closures instead of being decomposed.

### Long functions deliberately kept monolithic

**1. The driver loop — `train()` in `train.py:9-102` (~94 lines).** The entire RL outer loop (placement groups → rollout manager → models → per-rollout generate/train/save/eval/offload) is one linear function reading like a runbook:

```python
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())
...
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```
(`train.py:71-85`). The only extractions are two nested closures `offload_train` and `save` (`train.py:42-64`) — extracted because each is called from inside the loop with branching on critic/actor roles, and nesting lets them close over `args`/models without parameter plumbing. The whole file is 107 lines; the train↔rollout↔weight-sync ordering is auditable on one screen.

**2. `start_rollout_servers()` — `slime/ray/rollout.py:981-1115` (~135 lines).** Builds routers + all server groups (including encoder/prefill/decode disaggregation) in one function. The repeated group-construction logic is a **nested closure** `_make_group` (`rollout.py:1020-1056`) mutating `engine_offset`/`gpu_offset` via `nonlocal` — extraction without promotion to module level, because it is meaningless outside this function. Phases are marked with banner comments instead of sub-functions: `# --- Phase 1: start encoder groups, wait, collect URLs ---` (`rollout.py:1059`, `1074-1076`).

**3. `MegatronTrainRayActor.train_actor()` — `slime/backends/megatron_utils/actor.py:401-509` (~109 lines).** Ref-model logprobs → teacher logprobs → actor logprobs → advantages → train → backup weights, all inline with `self._switch_model("ref"/"teacher"/"actor")` state transitions interleaved (`actor.py:413, 426, 435`). Splitting it would hide the model-tag switching order, which is the correctness-critical content. Similarly `generate_rollout_async()` (`slime/rollout/sglang_rollout.py:389-480`, ~92 lines) keeps the entire over-sample/dynamic-filter/abort loop in one body.

### Small helpers they DID extract — and why

- **Pure computations whose name documents a policy**: `_compute_rollout_offset` / `_compute_megatron_num_gpus` are 4-line functions (`slime/ray/rollout.py:965-978`) — "where do rollout GPUs start in the placement group" is a contract used by offload decisions; the docstring is the point. Same for `should_run_periodic_action` (`slime/utils/misc.py:73-94`), shared by save and eval cadence (`train.py:87,98`).
- **Reusable algorithmic utilities**: `_chunk_by_size` generator + `chunk_named_params_by_size` wrapper (`slime/utils/misc.py:122-146`) — weight bucketing for IPC transfer.
- **Default implementations of pluggable hooks**: `pop_first` (`slime/rollout/data_source.py:225-229`) is 5 lines, at module level because it is the *default value of a plugin slot* (`buffer_filter_path`, `data_source.py:172-175`) — it must be addressable by the same `load_function` mechanism as user overrides.
- **Genuine multi-caller seams**: `_prepare_prompt_ids` (`sglang_rollout.py:42-61`) isolates messy multimodal/token-reuse branching out of `generate()`.

The signal: extraction happens for **reuse, plugin slots, or documented policies**, never just to make a long function shorter.

### Dataclasses, ABCs, comments

- `grep -rn "Protocol" slime/` returns nothing — they use `abc.ABC` for the three real seams: `DataSource` (5 abstract methods, `slime/rollout/data_source.py:17-46`), `TrainRayActor` (`train_actor.py:101-123`), and `HfWeightIteratorBase`, which embeds its own factory with dict dispatch (`hf_weight_iterator_base.py:5-15`: `{"raw": HfWeightIteratorDirect, "bridge": HfWeightIteratorBridge}[args.megatron_to_hf_mode]`).
- Dataclasses are used heavily *with methods attached*: `ServerGroup` and `RolloutServer` are `@dataclasses.dataclass` holding engine handles plus `start_engines`/`offload`/`recover` logic (`rollout.py:38-346`). The central `Sample` dataclass nests its own `Status(Enum)` and two sub-dataclasses `SpecInfo`/`PrefixCacheInfo` (`utils/types.py:8-120`), and its `from_dict` derives the valid field set from the type itself rather than a hand-written list:

```python
field_names = set(Sample.__dataclass_fields__.keys())
init_data = {k: v for k, v in data.items() if k in field_names}
```
(`types.py:136-138`).
- The train batch is deliberately *not* typed: `RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]` (`types.py:190`) with optional keys added ad hoc (`rollout.py:730-747`). Output contracts crossing the plugin boundary DO get dataclasses plus a legacy shim: `RolloutFnTrainOutput`/`RolloutFnEvalOutput` and `call_rollout_fn` wrapping raw returns (`slime/rollout/base_types.py:7-26`).
- Comment style: WHY-comments about cross-system gotchas, never WHAT — the NCCL env note (`actor_group.py:54-56`); the `torch.cuda.ipc_collect()` rationale ("Free GPU tensors so the caching allocator can reuse the blocks, then release CUDA IPC cache entries whose consumers (sglang engines) have already closed their IPC handles", `update_weight_from_tensor.py:161-170`). Docstrings spell out *return contracts and caller obligations* for tricky concurrency: `ServerGroup.start_engines` documents the `(init_handles, port_cursors)` protocol and why cursors must be passed to the next group "so that different groups on the same node don't race for ports" (`rollout.py:71-78`). Visible debt ships in main: `# TODO remove`, `# TODO: this is ugly, move to somewhere else?` (`data_source.py:58-59,213`, `actor.py:188`).

## 6. Naming conventions

- **`*Manager` for the one Ray actor that owns a subsystem**: `RolloutManager` (`slime/ray/rollout.py:349-350`, `@ray.remote class RolloutManager: "The class to run rollout and convert rollout data to training data."`). There is exactly one `*Manager` in the core — it is not a filler suffix.
- **`*Group` for collections scheduled together**: `RayTrainGroup` (`slime/ray/actor_group.py:10`), `ServerGroup` / `RolloutServer.server_groups` (`slime/ray/rollout.py:39, 219`), file `placement_group.py`.
- **`*RayActor` for classes instantiated as Ray actors**: `RayActor` → `TrainRayActor` (`slime/ray/train_actor.py:28`) → `MegatronTrainRayActor` (`slime/backends/megatron_utils/actor.py:44`); the engine actor is created as `RolloutRayActor = ray.remote(SGLangEngine)` (`rollout.py:91`).
- **`async_` prefix = "returns Ray ObjectRefs, caller must ray.get"** — an explicit documented convention: `"Functions start with 'async' should return list of object refs"` (`actor_group.py:13`); see `async_init` / `async_train` (`actor_group.py:101,111`) vs blocking `save_model` / `update_weights` that `ray.get` internally (`actor_group.py:131-137`).
- **Paired lifecycle verbs, uniform across families**: `offload`/`onload`/`onload_weights`/`onload_kv` exist with identical names on `ServerGroup` (`rollout.py:177-207`), `RolloutServer` (`rollout.py:312-346`), and `RolloutManager` (`rollout.py:510-525`); the training side uses `sleep`/`wake_up` (`actor.py:157-177`), bridged by `RayTrainGroup.onload/offload` calling `wake_up/sleep` (`actor_group.py:139-143`). Each layer is grep-distinct but shape-consistent.
- **Deployment-mode strategy classes named by mechanism**: `UpdateWeightFromTensor` vs `UpdateWeightFromDistributed`, module names matching class names (`update_weight/update_weight_from_tensor.py:24`), selected in one line (`actor.py:127`).
- **`*_utils` package suffix for backend adapters**: `megatron_utils/`, `sglang_utils/`, plus generic `http_utils`, `memory_utils`, `ppo_utils` — "utils" here means "adapter layer over an external system", not junk drawer (`misc.py` is the actual junk drawer).
- **`*_hub` = registry of pluggable functions**: `rm_hub/` (reward models), `filter_hub/` (`slime/rollout/rm_hub/`, `slime/rollout/filter_hub/`).
- **String-dotted-path plugin args named `*_path`**: `rollout_function_path`, `data_source_path`, `custom_generate_function_path`, `buffer_filter_path`, resolved by `load_function` (`slime/utils/misc.py:9-17`; usage `slime/ray/rollout.py:359-371`).
- **snake_case verb chains naming the actual pipeline**: `generate_and_rm`, `generate_and_rm_group`, `eval_rollout_single_dataset` (`sglang_rollout.py:240,310,499`) — the name *is* the dataflow.
- Module-level private helpers carry `_` even in non-package-private files: `_log_rollout_data`, `_compute_spec_metrics`, `_start_router` (`rollout.py:911,1175,1255`).

## 7. End-to-end flow trace

One complete RL step (synchronous, colocated):

1. Driver launches generation: `train.py:71` `rollout_manager.generate.remote(rollout_id)`.
2. `RolloutManager.generate` → `_get_rollout_data` → `call_rollout_fn(generate_rollout)`: `slime/ray/rollout.py:478-484, 583`; `slime/rollout/base_types.py:19-26`.
3. `generate_rollout` pushes the coroutine onto the background event loop: `slime/rollout/sglang_rollout.py:621` → `slime/utils/async_utils.py:34-36`.
4. Over-sampling main loop pulls prompts and submits group tasks: `sglang_rollout.py:423-426` → `RolloutDataSourceWithBuffer.get_samples` (buffer first) `slime/rollout/data_source.py:177-189` → `submit_generate_tasks` `sglang_rollout.py:136-149`.
5. Group → per-sample tasks: `generate_and_rm_group` `sglang_rollout.py:310-333` → `generate_and_rm` (semaphore-throttled) `sglang_rollout.py:257-277`.
6. Model forward: `generate` builds the payload (input_ids, `return_logprob=True`) and POSTs to the router `/generate`: `sglang_rollout.py:174-201`; the router forwards to an SGLang server (started by `ServerGroup.start_engines` and registered with the router: `slime/ray/rollout.py:70-175`, `slime/backends/sglang_utils/sglang_engine.py:196-220`); returned tokens + logprobs are appended to the Sample without re-tokenization: `sglang_rollout.py:204-234`.
7. Reward: `async_rm`/`batched_async_rm`: `sglang_rollout.py:288-301, 336-340`.
8. Dynamic filtering + harvest: `sglang_rollout.py:429-452`; once full, abort all in-flight requests and recycle partials into the buffer: `sglang_rollout.py:461, 345-386, 622-623`.
9. Convert to train data: per-group reward normalization (GRPO mean/std) `slime/ray/rollout.py:655-680` → `_convert_samples_to_train_data` (tokens/loss_masks/rollout_log_probs, ...) `rollout.py:682-749` → DP split + `ray.put` per shard: `rollout.py:754-805`.
10. Driver receives refs, offloads rollout engines (flush cache + release): `train.py:71-74` → `rollout.py:510-513` → `sglang_engine.py:357-359`.
11. Train: `train.py:85` → `RayTrainGroup.async_train` (one `actor.train.remote` per worker) `slime/ray/actor_group.py:111-129` → `MegatronTrainRayActor.train` (wake_up → fetch shard to GPU) `actor.py:355-374, 179-191` → `train_actor`: micro-batch plan `megatron_utils/data.py:290-385`, ref/actor logprob recompute `actor.py:410-448` + `model.py:213-358`, advantages `actor.py:464`, `train` per step `model.py:553-662`, finally CPU-backup the new weights `actor.py:496`.
12. Weight sync: `train.py:92-93` `onload_weights` + `actor_model.update_weights()` → `actor.py:539-591` → `UpdateWeightFromTensor.update_weights` (pause → flush_cache → chunked CUDA-IPC send → continue) `update_weight_from_tensor.py:138-181` → engine HTTP `/update_weights_from_tensor` `sglang_engine.py:266-289`.
13. Restore KV + CUDA graphs, next iteration: `train.py:96` → `rollout.py:341-346`.

## 8. Ideas worth borrowing for wm-infra

1. **Keep the outer RL loop a 100-line flat script.** slime's `train.py` makes the generate → offload-rollout → train → save → update_weights → onload ordering auditable at a glance (`train.py:67-99`). wm-infra's GRPO trainer-side loop (rollout → reward → advantage → step → weight sync) should stay this flat; resist wrapping it in a `TrainerRunner` class. The engine-internal phase machine (EngineLoop → IterationRunner) is a different animal and rightly stays structured.

2. **Adopt the `async_` = "returns refs/futures, caller gathers" naming rule.** One documented prefix (`actor_group.py:13`) eliminates the most common Ray/asyncio API ambiguity. wm-infra mixes async generation (Wan job queue) with blocking calls; a single greppable convention for "this returns handles" is cheap and high-value.

3. **Two-step onload vocabulary, identical at every fan-out layer.** `offload`/`onload_weights`/`onload_kv` have the same names on group → server → manager, each layer firing non-blocking refs and only the top `ray.get`ing (`rollout.py:177-207` vs `312-346` vs `510-525`). For wm-infra's colocated diffusion-rollout + FSDP-train memory dance, copy both the *split* (weights vs KV/cache resume are separate steps so the weight update can interleave: `train.py:91-96`) and the name uniformity.

4. **Weight-sync as two mechanism-named strategy classes chosen by topology.** `UpdateWeightFromTensor` (colocated: serialize → gloo `gather_object` → CUDA-IPC to engine, `update_weight_from_tensor.py:209-267`) vs `UpdateWeightFromDistributed` (NCCL broadcast), selected by `args.colocate` in one line (`actor.py:127`). wm-infra's `trainers/weight_sync` should converge on exactly this shape: name the class after the transport, not after "syncer/manager".

5. **Plugin slots = dotted-path CLI args + a 7-line `load_function`, with defaults living in-tree as ordinary functions.** Every research-customizable point (rollout fn, reward post-process, buffer filter, data source class) is `--*-path` (`rollout.py:359-371`), and the default (`pop_first`) is itself loaded the same way. Dramatically lighter than a registry framework and a good fit for wm-infra's reward/`rm_hub`-style needs — but it trades static checkability for flexibility; keep wm-infra's typed registry for model families where the set is closed.

6. **`from_dict` derived from `__dataclass_fields__`** (`types.py:136-142`) — exactly the "derive validation sets from the source of truth" rule wm-infra already enforces; slime additionally `setattr`s unknown keys back on, which is the right forgiving behavior for checkpoint-format evolution of `Sample`-like records.

7. **`inverse_timer` for idle-time attribution.** A singleton `Timer` plus a context manager that *ends* a timer on entry and *restarts* it on exit (`utils/timer.py:83-89`), used as `with inverse_timer("train_wait"), timer("train"):` (`actor.py:408`) so "time training waited for rollout" falls out for free. For wm-infra's engine loop, this is the cheapest way to measure per-phase bubble time (e.g. denoise-batch wait vs VAE decode) without a tracing framework.

8. **Don't type the cross-process batch prematurely.** `RolloutBatch` as a documented dict alias with optional keys (`types.py:187-190`) let slime bolt on `rollout_log_probs`, `routed_experts`, `teacher_log_probs`, multimodal tensors (`rollout.py:733-747`) without touching a schema. wm-infra's rollout→GRPO handoff has the same open-ended growth pattern (per-step latents, logprobs, KL refs); a dict-with-alias plus a converter function (`_convert_samples_to_train_data`, `rollout.py:682-748`) beats a frozen dataclass *at this one boundary* — keep typed state everywhere else.

9. **Nested closures over module-level single-caller helpers.** `_make_group` with `nonlocal` cursors (`rollout.py:1020-1056`) and `train.py`'s nested `save`/`offload_train` show the house style: when logic is loop-local, keep it lexically inside the orchestrator. This matches wm-infra's existing no-single-caller-helpers rule and is evidence the pattern scales to 1300-line production files.

10. **Over-sample + first-completed harvest + abort as the rollout scheduling policy.** The `asyncio.wait(FIRST_COMPLETED)` loop with over-sampling, dynamic filtering, and `/abort_request {"abort_all": True}` + partial-rollout recycling (`sglang_rollout.py:422-461, 345-386`) is the cleanest known treatment of the long-tail-generation problem. wm-infra's diffusion rollouts have a different tail shape (fixed denoise steps), but the same pattern applies to variable-length AR world-model rollouts and to reward-model scoring overlap.

11. **InfoActor probe for deterministic GPU mapping.** Ray placement-group bundles have no stable ordering; slime schedules a throwaway 1-GPU actor on each bundle to learn `(node_ip, gpu_id)` and re-sorts (`placement_group.py:14-17, 49-76`). Any Ray-based multi-node layout in wm-infra will hit the same problem.

## 9. Source-of-truth index

| Claim | Source |
|---|---|
| Sync train loop: generate → train → update_weights, offload/onload choreography (five beats) | `train.py:67-99` |
| Two-beat onload: `onload_weights` then `onload_kv` after update | `train.py:91-96` |
| Flat ~94-line driver with nested `save`/`offload_train` closures | `train.py:9-102`, `42-64` |
| Async loop pipelines next generate; no colocate; sync generate before weight update | `train_async.py:11, 35-43, 69-73` |
| Single PG, `{"GPU":1,"CPU":1}` bundles, PACK, colocate vs disaggregated offsets | `slime/ray/placement_group.py:41-44, 79-108` |
| InfoActor probe + bundle reordering by (IP, GPU id) | `slime/ray/placement_group.py:14-17, 49-76` |
| RayTrainGroup: 1 actor/GPU, fractional 0.4 GPU, master-addr bootstrap | `slime/ray/placement_group.py:111-119`; `slime/ray/actor_group.py:83-99` |
| MegatronTrainRayActor backend binding (only Ray↔Megatron coupling) | `slime/ray/actor_group.py:79-83` |
| TrainRayActor sets MASTER_*/RANK envs, `dist.init_process_group` | `slime/ray/train_actor.py:41-48, 63-66` |
| RolloutManager CPU-only actor; owns data source + rollout fn | `slime/ray/placement_group.py:183-187`; `slime/ray/rollout.py:349-392` |
| ServerGroup/RolloutServer dataclasses, PD/EPD groups, engine 0.2 GPU, `base_gpu_id` | `slime/ray/rollout.py:38-59, 91-143, 981-1115`; `sglang_engine.py:533` |
| Multi-node engines share `dist_init_addr` | `slime/ray/rollout.py:891-898` |
| Router as daemon `multiprocessing.Process` | `slime/ray/rollout.py:952-961` |
| SGLang server as spawned subprocess; registers with router | `slime/backends/sglang_utils/sglang_engine.py:63-79, 196-220` |
| `RolloutManager.generate` entry; trim to `global_batch_size` multiple | `slime/ray/rollout.py:478-491, 583, 590-605` |
| Rollout data → `ray.put` per DP rank (`Box`); train actors fetch shard | `slime/ray/rollout.py:754-805`; `slime/backends/megatron_utils/actor.py:179-191`; `slime/utils/data.py:299-310` |
| Background singleton event-loop thread (`run_coroutine_threadsafe`) | `slime/utils/async_utils.py:8-36` |
| Over-sampling + `FIRST_COMPLETED` + dynamic-filter main loop | `slime/rollout/sglang_rollout.py:422-452` |
| Scheduling unit = group (`n_samples_per_prompt`); task per group | `sglang_rollout.py:136-149`; `slime/rollout/data_source.py:107-117` |
| Concurrency semaphore = server_concurrency × gpus/engine_gpus; fast abort check | `sglang_rollout.py:94-96, 260-263`; `slime/backends/sglang_utils/arguments.py:39` |
| `abort_all` preemption + partial-rollout recycling into buffer | `sglang_rollout.py:345-386, 461, 622-623`; `data_source.py:198-211` |
| Buffer FIFO (`pop_first`), pluggable `buffer_filter`; buffer-first `get_samples`; epoch modulo + shuffle | `data_source.py:90-103, 172-189, 225-229` |
| HTTP `/generate` with input_ids + `return_logprob=True`; incremental token append | `sglang_rollout.py:158, 174-234` |
| `dp_rank_context` least-used counter; session-id routing | `sglang_rollout.py:119-129, 196-199` |
| Inline per-sample RM / batched `group_rm` | `sglang_rollout.py:288-301, 336-340` |
| GRPO group-norm rewards; `_convert_samples_to_train_data` | `slime/ray/rollout.py:655-680, 682-749` |
| DP seqlen balancing (`--balance-data`) | `slime/ray/rollout.py:764-767` |
| Dynamic micro-batch: first-fit packing + DP all-reduce MAX + balanced partitions | `slime/backends/megatron_utils/data.py:338-380`; `slime/utils/data.py:285-296` |
| `num_steps_per_rollout` / `global_batch_size` derivation | `megatron_utils/data.py:321-324`; `slime/utils/arguments.py:1769` |
| `forward_only` logprob recompute + order restore; per-step train | `slime/backends/megatron_utils/model.py:213-358, 553-662` |
| `train_actor` phase orchestration (ref→teacher→old→adv→train→backup); `_switch_model`/`TensorBackuper` | `slime/backends/megatron_utils/actor.py:95-119, 254-258, 401-509` |
| Updater class selection by colocate (one line) | `slime/backends/megatron_utils/actor.py:127-134` |
| `update_weights`: recover engines, reconnect on `num_new_engines > 0` | `slime/backends/megatron_utils/actor.py:539-591` |
| CUDA IPC path: per-engine Gloo gather groups, serialize buckets, `update_weights_from_tensor`, `ipc_collect` | `update_weight/update_weight_from_tensor.py:122-170, 209-267` |
| IPC hybrid fallback for engines outside the actor GPU range | `update_weight_from_tensor.py:85-115` |
| "HTTP only posts metadata, weights copied directly from GPUs" | `slime/backends/sglang_utils/sglang_engine.py:266-289` |
| NCCL path: per-PP group `slime-pp_{pp_rank}`, world = Σengine_gpus + 1, bucketed broadcast, EP path | `update_weight/update_weight_from_distributed.py:62-140, 166-226, 252-298, 310-337` |
| Ray `Lock` actor serializing broadcasts (spin acquire) | `slime/ray/utils.py:38-57`; `update_weight_from_distributed.py:234-248` |
| pause/flush/continue generation around update | `update_weight_from_distributed.py:88-90, 139`; `update_weight_from_tensor.py:145-147, 180` |
| Memory tags WEIGHTS/KV_CACHE/CUDA_GRAPH; staged restore | `slime/ray/rollout.py:15, 326-346` |
| `needs_offload` by GPU-range overlap; `enable_memory_saver=False` otherwise | `slime/ray/rollout.py:1026-1033` |
| `flush_cache` before release; retry while pending | `slime/backends/sglang_utils/sglang_engine.py:291-308, 357-359` |
| Prefix-cache hit-rate metrics | `slime/ray/rollout.py:1265-1273` |
| Train sleep/wake_up: torch_memory_saver + process-group destroy/reload; `disable()` escape during update | `megatron_utils/actor.py:156-177, 566` |
| `torch_memory_saver` injected via LD_PRELOAD runtime env | `slime/ray/actor_group.py:62-73` |
| Health monitor daemon threads + engine recovery | `slime/utils/health_monitor.py:10-59`; `slime/ray/rollout.py:269-310, 385-392, 527-548` |
| Rollout buffer plugin = FastAPI service | `slime_plugins/rollout_buffer/buffer.py:10-15` |
| `NOSET_VISIBLE_DEVICES` + `NCCL_CUMEM_ENABLE=0` env hygiene | `slime/ray/utils.py:17-25`; `slime/ray/actor_group.py:53-58` |
| `start_rollout_servers` monolith + `_make_group` nonlocal closure + phase banners | `slime/ray/rollout.py:981-1115, 1020-1056, 1059, 1074-1076` |
| Extracted tiny helpers: offset computations, periodic-action policy, chunk utils | `slime/ray/rollout.py:965-978`; `slime/utils/misc.py:73-94, 122-146` |
| `RolloutManager` `@ray.remote` class + docstring | `slime/ray/rollout.py:349-351` |
| `async_` prefix convention docstring; async vs blocking methods | `slime/ray/actor_group.py:10-13, 101-137` |
| `*RayActor` hierarchy; `RolloutRayActor = ray.remote(SGLangEngine)` | `slime/ray/train_actor.py:28`; `megatron_utils/actor.py:44`; `slime/ray/rollout.py:91` |
| offload/onload names uniform at 3 layers; sleep/wake_up bridge | `slime/ray/rollout.py:177-207, 312-346, 510-525`; `actor.py:156-177`; `actor_group.py:139-143` |
| `load_function` + `*_path` plugin args | `slime/utils/misc.py:9-17`; `slime/ray/rollout.py:359-371` |
| 1841-line `arguments.py` with nested `add_*_arguments` groups | `slime/utils/arguments.py:36-1348` |
| No `Protocol`; ABCs `DataSource`/`TrainRayActor`/`HfWeightIteratorBase` dict-dispatch factory | grep result; `data_source.py:17-46`; `train_actor.py:101-123`; `hf_weight_iterator_base.py:4-15` |
| `Sample` nested `Status` enum + sub-dataclasses; `__dataclass_fields__`-derived `from_dict` | `slime/utils/types.py:8-144` |
| `RolloutBatch` dict alias; ad-hoc optional keys | `slime/utils/types.py:187-190`; `slime/ray/rollout.py:730-747` |
| Output-contract dataclasses + legacy shim | `slime/rollout/base_types.py:7-26` |
| WHY comments: NCCL env, `ipc_collect`, port-cursor contract | `actor_group.py:54-56`; `update_weight_from_tensor.py:161-170`; `rollout.py:71-78` |
| Shipped TODO debt | `data_source.py:49, 58-59, 213`; `actor.py:188` |
| `Timer` singleton + `inverse_timer` usage | `slime/utils/timer.py:15-89`; `actor.py:408` |
| File size distribution | `wc -l` over `slime/**/*.py` (top: arguments.py 1841, rollout.py 1283, loss.py 1212) |
