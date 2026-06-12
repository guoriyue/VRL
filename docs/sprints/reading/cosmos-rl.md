# cosmos-rl — Architecture Reading

Repo: `/home/mingfeiguo/Desktop/cosmos-rl` (NVIDIA Cosmos-RL). All `path:line` references are relative to the repo root and were spot-checked against this checkout.

One-paragraph summary: Cosmos-RL is a **single-controller, HTTP + Redis** RL post-training system with **no Ray dependency**. One controller process runs a FastAPI app and forks its own `redis-server`; worker replicas — **policy** (trainers) and **rollout** (generators) — are plain `torchrun` jobs that self-register over HTTP. The controller pushes msgpack-packed **commands** and **rollout training data** through per-replica Redis streams; cross-replica weight movement uses dynamically (re)built raw NCCL communicators with the controller acting as UID rendezvous KV.

---

## 1. Repo layout & module organization

Top level: `cosmos_rl/` (the Python package), `configs/` (per-model TOML experiment configs: `qwen3/`, `deepseek-v3/`, `cosmos-predict2-5/`, `stable-diffusion-3-5`, …), `reward_service/` (separate `cosmos_rl_reward` package), `tests/`, `docs/`, `tools/`, `examples/`. Two console entry points are declared in `pyproject.toml`: `cosmos-rl = "cosmos_rl.launcher.launch_all:main"` and `cosmos-cli = "cosmos_rl.cli.cli:main"` (`pyproject.toml:[project.scripts]`).

Package map (`cosmos_rl/`):

| Directory | Owns |
|---|---|
| `launcher/` | Process launcher: `launch_all.py` (placement + spawning), `launch_controller.sh` / `launch_replica.sh`, `worker_entry.py` (role switch by `COSMOS_ROLE`), `utility.py` (`NodesManager` placement) |
| `dispatcher/` | The **controller**: `run_web_panel.py` (FastAPI app), `controller.py` (singleton `Controller`), `command.py` (command taxonomy), `replica.py` (`Atom`/`Replica` bookkeeping), `status.py` (Policy/Rollout status managers, command triggering), `protocol.py` (pydantic request schemas, `MESH_NAMES`), `api/client.py` (HTTP client used by workers), `data/` (data fetcher + per-model data packers), `algo/` (GRPO/reward computation helpers) |
| `policy/` | Training side: `train.py` → `policy_entry.py` → `worker/` (`rl_worker.py`, `sft_worker.py`, `wfm_worker.py`), `trainer/` (llm/diffusers/vla/wfm trainers), `model/` (per-family `parallelize.py` + `weight_mapper.py`), `kernel/`, `config/` |
| `rollout/` | Generation side: `rollout_entry.py` → `worker/` (`rollout_control.py` = main control worker, `weight_sync.py` = async WeightSyncThread), backends `vllm_rollout/`, `trtllm_rollout/`, `vla_rollout/`, `wfm_rollout/`, `diffuers_rollout/` (sic — typo shipped) |
| `reference/` | Teacher replica entry for distillation (`reference_entry.py`) |
| `colocated/` | Colocated mode: in-process `ColocatedController` + `CommandDispatcher` (in-memory queues replacing Redis) |
| `collective/` | `P2RCollectiveManager` — policy→rollout NCCL/ZMQ-IPC transport |
| `comm/` | `CommMixin` (register/heartbeat/redis init shared by all workers), `WorkerBase` ABC |
| `utils/` | `distributed.py` (`HighAvailabilitylNccl`, `DistKVStore`), `pynccl*.py` (raw NCCL bindings), `parallelism.py` (`ParallelDims`), `parallelism_map.py` (`ParallelizedShardMapper`), `redis_stream.py` (`RedisStreamHandler`) |

The README states the design intent: replica specialization (Policy=consumer, Rollout=producer), a "Single-Controller Architecture" with a messaging system, and "Dynamic NCCL Process Groups for on-the-fly GPU [un]registration" (`README.md:13-27`).

Package layout is **role-first, not layer-first**: `policy/` = trainer replica, `rollout/` = generation replica — the RL role is the namespace.

---

## 2. Architecture overview

**Single controller, many torchrun replicas.** The controller is one Python process running a FastAPI app plus a Redis server it forks itself:

```python
# cosmos_rl/dispatcher/controller.py:157-161
redis_server_cmd = f'redis-server {redis_cfg_path} --dbfilename {random_db_file_name} --save ""'
redis_server_proc = subprocess.Popen(redis_server_cmd, shell=True, ...)
```

served by uvicorn (`dispatcher/run_web_panel.py:680-685`). A background thread in the FastAPI lifespan monitors replica heartbeats (`run_web_panel.py:104-124`).

**Unit model.** Each GPU process is an `Atom` ("Atom is the smallest unit of a computation mesh. Usually it is a single GPU process", `dispatcher/replica.py:30-43`); a `Replica` is "a single `DP Replicate` unit" composed of TP/CP/PP/DP_SHARD atoms with a per-replica `command_queue` (`dispatcher/replica.py:163-197`). Atom ranks are indexed by `MESH_NAMES: List[str] = ["pp", "dp_shard", "cp", "tp"]` (`dispatcher/protocol.py:21`).

**Two control channels:**
1. **HTTP (request/response)** — every worker holds an `APIClient` for register/unregister/heartbeat/shard-info/NCCL-handshake/train-ack/prompt-fetch (`dispatcher/api/client.py:68-581`; endpoints in `run_web_panel.py:170-537`). On registration, rank 0 of each replica spawns a heartbeat `mp.Process` (`comm/base.py:287-298`).
2. **Redis streams (push)** — controller publishes `Command`s to a per-replica stream (`<replica>_command`, `utils/redis_stream.py:136-148`) and rollout samples to the policy replica's stream (`<replica>_rollout`, `redis_stream.py:199-216`): `redis_handler.publish_rollout(msgpack.packb(rollout.model_dump()), self.name)` (`dispatcher/replica.py:274-283`); workers poll via `RedisStreamHandler.subscribe_command/subscribe_rollout` (`utils/redis_stream.py:126-254`).

**Command system.** `CommandType` enumerates `WEIGHT_RESUME, BUILD_MESH, POLICY_TO_POLICY_BROADCAST/UNICAST, POLICY_TO_ROLLOUT_UNICAST, ROLLOUT_TO_ROLLOUT_BROADCAST, DATA_FETCH, ALL_REDUCE, STOP, VALIDATE, DUMMY` (`dispatcher/command.py:27-38`). Each command class has a classmethod `trigger(...)` that msgpack-packs itself and publishes to the target replicas' Redis streams (e.g. `PolicyToRolloutUnicastCommand.trigger` publishes to both src policy and dst rollout streams, `command.py:282-308`). Workers register handlers via decorators, e.g. `@CommMixin.register_policy_command_handler(DataFetchCommand)` (`policy/worker/rl_worker.py:510`), dispatched through `PolicyCommandRegistry`/`RolloutCommandRegistry` (`command.py:482-512`).

```
                       ┌──────────────── Controller process ────────────────┐
                       │ FastAPI (uvicorn) ── Controller singleton          │
                       │   PolicyStatusManager / RolloutStatusManager       │
                       │   ParallelizedShardMapper (P2R shard scheme)       │
                       │ redis-server (subprocess, forked by controller)    │
                       └──────┬──────────────────────────▲─────────────────┘
            HTTP: register / heartbeat /                 │ HTTP: put_rollout_group,
            shard infos / NCCL uid KV /                  │ train_ack, next_prompt
            send-recv insts                              │
   Redis streams: Commands ↓        Commands ↓           │
┌─────────────────────────────┐   ┌──────────────────────┴───────────────┐
│ Policy replica (torchrun ×N)│   │ Rollout replica (torchrun ×M, vLLM   │
│ main loop: execute_command  │   │ external_launcher)                   │
│ threads: fetch_command,     │   │ main loop: consume_command →         │
│  fetch_rollouts(rank0),     │   │  request_new_prompts → generate →    │
│  HA-NCCL build-mesh,        │   │  report_rollouts                     │
│  heartbeat (mp.Process)     │   │ threads: query_command(rank0),       │
│                             │   │  WeightSyncThread (own CUDA stream)  │
└─────────────┬───────────────┘   └───────────▲──────────────────────────┘
              │   P2R weight sync: NCCL unicast│(or ZMQ CUDA-IPC same-GPU)
              └────────────────────────────────┘
        P2P inter-policy mesh: HighAvailabilitylNccl (BUILD_MESH cmd)
        R2R broadcast: NCCL comm over rollout global mesh
```

**Policy main loop** (`policy/worker/rl_worker.py:792-853`): rank 0 starts `fetch_command_thread` and `fetch_rollouts_thread` (each runs an asyncio loop in a Python thread), then the main thread spins on `broadcast_command()` (rank 0 drains its buffer and `dist_util.broadcast_object_cpu`s the command list to all ranks, `rl_worker.py:625-635`) and `execute_command()`. Training is driven by `DataFetchCommand`: the handler pulls `items_count` rollouts from `data_queue`, scatters them across DP ranks (`dispatch_rollouts`, `rl_worker.py:677-767`), runs `trainer.step_training(...)`, and posts a train ack over HTTP (`rl_worker.py:538-565`). `BuildMeshCommand` is special-cased in the fetch thread and pushed straight to the HA-NCCL background thread so mesh rebuilds never block training (`rl_worker.py:586-597`).

**Rollout main loop** (`rollout/worker/rollout_control.py:1755-1823`): `consume_command()` (rank 0 dequeues, then `broadcast_object_cpu` to all ranks, `rollout_control.py:1560-1593`), then `request_new_prompts` → HTTP `get_batched_prompt` from the controller → `one_step_generation()` → `report_rollouts()` posts completions back via HTTP. Prompt staleness is enforced locally: a prompt is generated only if `payload.weight_version <= current_weight_version + allowed_outdated_steps` (`rollout_control.py:1807-1815`).

**Closing the loop on the controller.** `put_rollout_group` filters outdated rollouts and round-robins them into policy replicas' Redis streams (`run_web_panel.py:434-508`); when all policy replicas ack a step, the controller marks replicas READY and triggers weight sync + the next `DataFetchCommand` (`dispatcher/status.py:1248-1264`). Throttling against off-policy drift (soft/hard inflight caps, DAPO retry budget) lives in `Controller._get_batched_prompt_impl` (`dispatcher/controller.py:284-352`).

---

## 3. Core scheduling & orchestration

There are three event loops: the **controller** (purely reactive — it has **no scheduler thread**; all scheduling decisions fire inside HTTP handlers), the **rollout-worker main loop**, and the **policy-worker main loop**.

### 3.1 Controller-side queues and state

The scheduling state lives in `PolicyStatusManager` (`cosmos_rl/dispatcher/status.py:98-148`):

```python
self.rollout_buffer = Queue()            # status.py:124  — global FIFO of finished rollouts
self.samples_on_the_fly = 0              # status.py:126  — prompts handed out × n_generation
...
self.rollout_buffer_per_rank: List[Queue] = []   # status.py:148 — per-mesh-rank buffers
```

Each policy replica has a status machine `UNINITIALIZED → READY → RUNNING → REDUCED → END/VALIDATED` (`status.py:78-95`). Replicas are `Replica` objects made of `Atom`s (one per GPU process, `cosmos_rl/dispatcher/replica.py:30-52`); a replica only participates once `all_atoms_arrived` (`replica.py:267-272`, product of mesh `group_size` == number of registered atoms).

### 3.2 Prompt admission: batching policy + backpressure throttles

Rollout workers *pull* prompts via `GET /next_prompt` (`run_web_panel.py:412-422`) → `Controller._get_batched_prompt_impl` (`controller.py:245-469`). This function is the **admission scheduler**:

1. **Batch sizing & weight-version tagging** (`controller.py:264-276`):
```python
rollouts_per_global_batch = self.config.train.train_batch_per_replica * len(self.policy_status_manager)
global_batch_size = math.ceil(rollouts_per_global_batch / self.config.rollout.n_generation)
...
weight_version_for_current_batch = self.policy_status_manager.current_step + (
    self.policy_status_manager.total_pending_rollouts() // rollouts_per_global_batch)
```
Every prompt is stamped with the *predicted* policy step at which its rollouts will be trained.

2. **Soft throttle** — if pending rollouts exceed `(allowed_outdated_steps + 1) × rollouts_per_global_batch`, the fetch size `n` is clamped to `outdated_rollout_fetch_batch_size` (`controller.py:291-303`). DAPO gets its own estimate scaled by `max_retry_for_on_policy` (`controller.py:304-338`).

3. **Hard throttle** — `max_inflight_steps` caps total in-flight samples; beyond it the controller returns an empty batch (`controller.py:348-352`):
```python
hard_threshold = max_inflight * rollouts_per_global_batch
if current_pending_rollouts >= hard_threshold:
    return [], is_validation
```

4. **On-policy exactness** — in fully-synchronized (non-DAPO) mode, exactly `global_batch_size` prompts are accounted per weight version via `weight_version_to_prompt_num`, overflowing to the next version (`controller.py:369-404`); exceeding `max_retry_for_on_policy` raises (`controller.py:423-434`).

5. Finally `samples_on_the_fly += fetched × n_generation` (`controller.py:464-467`).

### 3.3 The central scheduling decision: `try_trigger_data_fetch_and_training`

Finished rollouts arrive via `POST /rollout` (`run_web_panel.py:434-507`), are filtered for staleness (`filter_outdated_rollouts`, `status.py:812-855`: a rollout is dropped if `estimated_step - rollout.weight_version > allowed_outdated_steps`, where `estimated_step` accounts for queue depth and position in batch — `status.py:829-838`), then `put_rollout` enqueues each into `rollout_buffer` (or `rollout_buffer_per_rank[prompt_idx % n_ranks]`) and **immediately** calls the trigger (`status.py:744-766`).

`try_trigger_data_fetch_and_training` (`status.py:1362-1456`) is the heart of step scheduling:

- **Gate** (`status.py:1383-1387`): fire only when every replica is READY/REDUCED **and** `rollouts_enough_for_one_step()` — i.e. `total_pending_rollouts() >= train_batch_per_replica × n_replicas` (`status.py:1306-1324`).
- **Step advance** (`status.py:1394-1397`): `self.remain_samples_num -= required_rollouts; self.current_step += 1` — the controller owns the global step counter / weight version.
- **Data dispatch** (`status.py:1434-1439`) — interleaved round-robin so each replica gets `items_count` rollouts:
```python
for _ in range(items_count):
    for replica in arrived_replicas:
        rollout = self.rollout_buffer.get()
        replica.put_rollout(rollout, self.redis_handler)
```
`Replica.put_rollout` publishes to the `<replica>_rollout` Redis stream (`replica.py:274-283`). In `data_dispatch_as_rank_in_mesh` mode, per-rank queues are first sorted by `prompt_idx` (`status.py:1410-1433`).
- **Command dispatch** (`status.py:1444-1456`): a `DataFetchCommand(items_count, global_step, total_steps, remain_samples_num, do_save)` is published to every replica's command stream and the replica is marked RUNNING.

### 3.4 Step completion → weight sync → next step

When a policy replica finishes a step it calls `POST /train_ack` (`run_web_panel.py:516-531`) → `PolicyStatusManager.train_ack` (`status.py:1035-1267`). Once **all** replicas are REDUCED (`status.py:1062`):

```python
need_sync_weight = step % self.config.train.sync_weight_interval == 0
need_sync_weight = need_sync_weight or step == total_steps        # status.py:1070-1072
...
if need_sync_weight:
    self.trigger_weight_sync(any_loaded_replica, rollout_status_manager, step, total_steps)
self.try_trigger_data_fetch_and_training()                        # status.py:1259-1264
```

`trigger_weight_sync` (`status.py:1269-1304`) issues two commands: `PolicyToRolloutUnicastCommand` (one policy replica → one rollout replica, sharded NCCL) and `RolloutToRolloutBroadcastCommand` (that rollout replica → all rollout replicas).

Registration-time orchestration is also command-driven: the first complete policy replica gets `WeightResumeCommand` (`status.py:429-437`), then `PolicyToPolicyBroadcastCommand` across policy replicas (`status.py:554-559`), `BuildMeshCommand` assigns mesh ranks (`command.py:132-145`), and the first rollout replica receives an initial P2R unicast (`status.py:1721-1742`).

### 3.5 Rollout worker main loop — line by line

`DisaggregatedRolloutControlWorker.work()` starts a daemon command-poller thread, optional prefetch thread, then enters the loop (`cosmos_rl/rollout/worker/rollout_control.py:2201-2224`). The poller blocking-reads the Redis command stream and enqueues to a local `Queue` (`rollout_control.py:1416-1449`; in async-R2R mode P2R/R2R commands bypass the queue straight to a WeightSyncThread, `1430-1447`).

`_main_loop_impl` (`rollout_control.py:1755-1823`):

```python
while not self.shutdown_signal.is_set():
    self.consume_command(cmd_pred=None)                  # 1759: drain & execute commands
    if async_mode != AsyncR2RSyncMode.DISABLED:
        process_wst_deferred_actions(self)               # 1763-1764
    if self.validation_flag.is_set():
        self.do_validation()                             # 1766-1767
    if not self.state.weight_synced():
        continue                                         # 1769-1770: HARD GATE — no generation before first weight sync
    _, is_validation, _, _ = self.report_rollouts()      # 1772: flush reward-scored payloads to controller
    if self._is_async_rollout:
        self.stream_generation_step()                    # 1777-1779: async (vllm_async) path
        continue
    if not self.state.prompt_fetch_end():
        no_more_prompts = self.request_new_prompts(...)  # 1781-1786: HTTP pull from controller
        ...                                              # 1787-1795: end-of-data state transitions
    if self.state.prompt_consume_end(): continue         # 1797-1801
    elif self._prompt_queue.empty(): continue            # 1802-1803
    else:
        first_payload: RLPayload = self._prompt_queue.queue[0][0]
        is_valid_prompt_for_current_weight_version = (
            first_payload.weight_version
            <= self.current_weight_version
            + self.config.train.train_policy.allowed_outdated_steps)   # 1807-1812
        if not is_valid_prompt_for_current_weight_version:
            continue                                     # 1814-1815: stall until weights catch up
        self.one_step_generation()                       # 1817
```

Key scheduling semantics:
- `State` is a bitmask state machine `UNINITIALIZED → WEIGHT_SYNCED → PROMPT_FETCH_END → PROMPT_CONSUME_END` (`cosmos_rl/rollout/__init__.py:58-88`).
- **Freshness gating** (lines 1807-1815) implements asynchrony control on the worker side: a prompt stamped for weight version `v` won't be generated until `current_weight_version >= v - allowed_outdated_steps`, complementing the controller-side throttles.
- `request_new_prompts` fetches only on global rank 0 (`batch_size × dp_size`), then broadcasts + round-robin scatters across DP ranks (`rollout_control.py:1459-1558`).
- `consume_command` drains commands until quiescent, with the special rule that after a `PolicyToRolloutUnicastCommand` it keeps waiting for the paired R2R (`rollout_control.py:1595-1629`).

**Preemption mid-generation (the most distinctive mechanism).** Synchronous `vllm.LLM.generate()` can run for minutes, so cosmos-rl monkey-patches the vLLM engine step (`cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:77-119`):

```python
def step(self, *args, **kwargs):
    ...
    if (COSMOS_ROLLOUT_STEP_INTERVAL > 0
            and self._cosmos_step_counter % COSMOS_ROLLOUT_STEP_INTERVAL == 0):
        consume_command(cmd_pred=partial(cmd_pred, enable_validation=enable_validation))
    return orig_step(*args, **kwargs)
llm_engine.step = types.MethodType(step, llm_engine)
```

i.e. every N engine iterations the worker pauses decoding, executes pending P2R/R2R weight-sync commands (updating weights **under in-flight sequences** — safe because prefix caching is off, see 3.7), reports rewards every `COSMOS_ROLLOUT_REPORT_INTERVAL` steps, and resumes. This is the only "preemption"; there is no request priority — all queues are FIFO.

**Async rollout path** (`rollout.mode == "async"`, vllm_async backend): a `RolloutTaskScheduler` runs its own asyncio loop (`cosmos_rl/rollout/worker/asynchronous/rollout_task_scheduler.py:320-368`):

```python
while self._running.is_set():
    completed_tasks = {task for task in self._active_tasks if task.done()}   # 333
    ...
    if not self._paused.is_set():
        while (len(self._active_tasks) < self.max_concurrent_requests
               and not self.task_queue.empty()):                             # 346-349
            rollout_task = self.task_queue.get_nowait()
            task = asyncio.create_task(self._generate_single(rollout_task))  # 355
    await asyncio.sleep(self.check_interval)
```

Concurrency control is a simple `max_concurrent_requests` admission window; `pause()/resume()` (with `paused(wait_for_active_tasks=True)` context manager) drains active tasks before weight sync (`rollout_task_scheduler.py:595-707`). The main loop feeds it with `stream_generation_step` → `_stream_generation_feed_prompts`, which asks the controller for only `min(batch_size, max_concurrent − pending − active)` prompts (`rollout_control.py:1976-1981`) and collects completions non-blockingly (`rollout_control.py:2024-2071`).

**Prompt prefetch**: optional background thread overlaps the controller HTTP round-trip with generation, guarded by `_prompt_fetch_lock` to prevent double fetch (`rollout_control.py:2118-2199`; lock semantics at `1468-1473`).

### 3.6 Policy worker main loop

`RLPolicyWorker.main_loop` (`cosmos_rl/policy/worker/rl_worker.py:792-853`) is a pure command executor — the policy worker never decides anything itself:

```python
abort = False
while True:
    abort_at_this_round = abort
    if abort_at_this_round:
        time.sleep(30)                       # 840-842: grace for a final P->R
    self.broadcast_command()                 # 844: rank0 drains buffer, dist-broadcasts to all ranks
    while len(self.command_buffer.queue) > 0:
        cmd = self.command_buffer.get_nowait()
        abort = self.execute_command(cmd) or abort   # 845-847
    if abort_at_this_round: break
```

Two daemon threads feed it: `fetch_command` (Redis `subscribe_command`; `BuildMeshCommand` is short-circuited into the HA-NCCL object without touching the main loop, `rl_worker.py:570-611`) and `fetch_rollouts` (rank 0 only, Redis `subscribe_rollout` → `self.data_queue`, `rl_worker.py:256-272`).

The training step itself is the `DataFetchCommand` handler `execute_data_fetch` (`rl_worker.py:510-568`): it sets `replica_batch_for_this_step = command.items_count`, calls `dispatch_rollouts()` — rank 0 **blocks** on `data_queue.get(block=True, timeout=None)` for `items_count` rollouts and round-robins them across DP ranks via `dist.scatter_object_list` (`rl_worker.py:711-767`) — then `trainer.step_training(...)` (`rl_worker.py:539-547`) and finally `api_client.post_policy_train_ack(...)` from the master rank (`rl_worker.py:558-565`). The loop exits when `command.replica_should_stop()` i.e. `global_step >= total_steps` (`command.py:472-475`).

### 3.7 Memory / cache management as it interacts with scheduling

Cosmos-RL **delegates KV-cache management entirely to the inference backend** and deliberately disables the parts that conflict with RL weight updates. vLLM engine construction (`cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:298-327`):

```python
self.rollout_engine = LLM(
    model=model_path,
    enable_sleep_mode=False,  # enable sleep could corrupt the cuda allocator.
    ...
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=self.rollout_config.gpu_memory_utilization,
    max_num_batched_tokens=2048 if 2048 >= policy_config.model_max_length
        else policy_config.model_max_length,
    enable_chunked_prefill=self.rollout_config.enable_chunked_prefill,
    # Always disable prefix caching, since RL will change the underlying model.
    # The prefix cache will be invalid after training.
    enable_prefix_caching=False,
    ...)
```

- **Prefix caching is hard-disabled** (`vllm_rollout.py:317-319`, same in async backend `vllm_rollout_async.py:181`) precisely because the locked-step patch mutates weights between engine steps — a prefix cache would serve logits from stale weights. Paged KV/eviction inside a generation remains vLLM-internal; cosmos-rl never touches block tables.
- **Weight-sync staging memory** is actively managed: during P2R receive, non-contiguous/dtype-mismatched shards go through temp tensors tracked in `temp_recv_tensor_queue` with CUDA events; when the queue exceeds `COSMOS_RECV_TENSOR_QUEUE_SIZE` the worker synchronizes and frees (`rollout_control.py:499-524`). FSDP modules are explicitly resharded before receiving to avoid unsharded-param memory spikes (`rollout_control.py:490-493`). Receives are pipelined in NCCL groups of `COSMOS_P2R_NCCL_GROUP_SIZE` with completions flushed on a separate `copy_stream` (`rollout_control.py:1092-1175`).
- **Controller-side buffer hygiene**: Redis runs with `maxmemory 500G / allkeys-lfu` (`controller.py:147-150`) and streams are capped via `maxlen=STREAM_MAXLEN` (`redis_stream.py:144,207`). When outdated rollouts whose completions are `nccl:`-prefixed handles (GPU buffers held by the rollout worker for NCCL payload transfer) are discarded, the controller publishes explicit cleanup messages so the worker frees those GPU buffers immediately rather than by age (`status.py:857-948`).
- The scheduling-memory coupling is the throttle chain in 3.2: `samples_on_the_fly` + `max_inflight_steps` bound the rollout buffer (and hence Redis/GPU payload memory) as a function of training progress.

---

## 4. Distributed orchestration (Ray or alternative)

### 4.1 No Ray

There are zero Ray imports in the repo (`grep -rn "^import ray\|^from ray"` over all `*.py` returns nothing — verified in this checkout; `ray` does not appear in `pyproject.toml`). Ray's roles are replaced by:

| Ray would provide | Cosmos-RL replacement |
|---|---|
| Cluster launcher / placement groups | `launch_all.py` + `NodesManager.replica_placement` greedy GPU packing per node (`launcher/utility.py:451-554`); Lepton cloud jobs as alternative (`launch_all.py:551-612`) |
| Actors / remote tasks | Plain subprocesses: `launch_controller.sh` (python controller) and `launch_replica.sh` → **torchrun** per replica with `--rdzv_backend c10d` (`launcher/launch_replica.sh:119,152-157`); `mpirun` instead for the TRT-LLM backend |
| Control RPC | FastAPI HTTP + Redis streams (Section 2) |
| Object store / data plane | Redis (rollouts, msgpack), NCCL (weights), ZMQ `PAIR` sockets + CUDA-IPC tensor handles for same-device transfer in colocated_separated mode (`collective/collective.py:262-290, 352-367`) |
| GCS/KV for rendezvous | Controller's in-memory KV via HTTP: `post_nccl_comm_initiator` / `post_nccl_comm_acceptor` (`dispatcher/api/client.py:276-311`, served at `run_web_panel.py:352-371`) |

### 4.2 Process topology and placement

`launch_all.py` computes per-replica GPU needs as the product of `tp_size × dp_replicate_size × pp_size × cp_size × dp_shard_size` from the TOML (`launcher/launch_all.py:421-456`), decides replica counts (optionally from `--p2r-ratio`, `launch_all.py:460-495`), then `NodesManager` assigns each replica a contiguous GPU range on a worker node and emits one `launch_replica.sh` command per replica; the same script runs on every node and each node executes only its own slice `global_launch_settings[cur_work_idx]` (`launch_all.py:760-884`). The replica process learns its role from `COSMOS_ROLE` (`Policy`/`Rollout`/`Reference`/`Controller`) exported by the launch scripts (`launcher/launch_replica.sh` role block; `launcher/worker_entry.py:31-37`). Each worker pulls the full config from the controller over HTTP at startup (`policy/policy_entry.py:32-39`), so only the controller needs the TOML.

Three modes set by `mode` in config: `disaggregated` (default), `colocated`, `colocated_separated` (`launch_all.py:510-548`). In colocated mode rollout replicas get **no** separate processes (`n_rollouts` forced to 0 in placement, `utility.py:508-513`); a `ColocatedController(Controller)` runs *inside* the worker process and replaces Redis with an in-memory `CommandDispatcher` ("uses Python Queue to simulate the command publish-subscribe mechanism", `colocated/utils.py:22-49`; `colocated/controller.py:74`).

### 4.3 Dynamic NCCL groups (the elasticity story)

Cosmos-RL does not use `torch.distributed` for cross-replica comm — only inside one replica (init in `policy_entry.py:47-48`). Cross-replica it builds raw NCCL communicators through its own pybind layer (`utils/pynccl.py`), with the controller as UID rendezvous:

```python
# utils/distributed.py:533-547 (HighAvailabilitylNccl.__execute_build_mesh)
if rank == 0:
    nccl_group_id = create_nccl_uid()
    self.api_client.post_nccl_comm_initiator(unique_pair_name, nccl_group_id)
else:
    nccl_group_id = self.api_client.post_nccl_comm_acceptor(unique_pair_name)
```

`HighAvailabilitylNccl` runs a dedicated daemon thread consuming `BuildMeshCommand`s: on every command it first aborts the existing communicator and rebuilds (`distributed.py:479-500`); NCCL ops are wrapped in `__do_nccl_op_with_retry` with a timeout watchdog, and failures are reported to the controller via `post_nccl_comm_error` (`distributed.py:577-609`). On the controller, NCCL error reports start a delayed sweep that unregisters hung replicas and re-triggers `BuildMeshCommand` for the survivors (`dispatcher/controller.py:609-668`; `status.py:482-490`). This is the "fault-tolerant and elastic" mechanism the README advertises.

### 4.4 Weight sync (policy → rollout → rollout)

1. **Shard scheme on the controller.** Both sides post their parameter shard layouts after model build (`rl_worker.py:204-213` policy; `rollout_control.py:274-292` rollout). The singleton `ParallelizedShardMapper` "gather[s] the shard information for policy and rollout then generate[s] the send and receive instructions" using multiprocessing (`utils/parallelism_map.py:915-946`); workers fetch their per-rank instruction lists via `post_policy_shard_send_insts` / `post_rollout_shard_recv_insts` (`run_web_panel.py:266-323`).
2. **Trigger.** After each step's train acks, the controller emits a `PolicyToRolloutUnicastCommand` (one policy replica → one rollout replica) followed by a `RolloutToRolloutBroadcastCommand` to fan out to all rollout replicas (`dispatcher/status.py:1269-1304`).
3. **P2R transfer.** The policy handler slices each mapped tensor per instruction (`local_view.cosmos_slice(tensor_split_strategys)`) and sends through `P2RCollectiveManager`, batching sends inside `nccl_group_start/end` windows (`rl_worker.py:398-477`). The P2R communicator is a fused group of size `src_replica_size + dst_replica_size` with rollout ranks offset by the policy world size (`collective/collective.py:125-138`); in `colocated_separated` mode same-GPU pairs skip NCCL entirely and pass CUDA-IPC handles over a ZMQ PAIR socket (`collective.py:352-377`).
4. **R2R fan-out.** Rollout replicas build a separate "global mesh" NCCL comm per `BuildMeshCommand` — one comm per local rank index across replicas (`rollout_control.py:372-421`) — and broadcast received weights over it. An opt-in `WeightSyncThread` executes P2R/R2R on its own CUDA stream into a parameter-only *buffer model*, copied to the live vLLM weights at generation boundaries, so weight sync overlaps generation (`rollout/worker/weight_sync.py:16-50`; routing at `rollout_control.py:1407-1449`).

### 4.5 Parallelism config

`ParallelDims.from_config` carries `dp_replicate/dp_shard/cp/tp/pp/ep` (`utils/parallelism.py:85-118`) and builds the torch `DeviceMesh` in order `["pp", "dp_replicate", "dp_shard", "cp", "tp"]` plus derived submeshes (`parallelism.py:231-261`), with a second `["pp", "dp_shard_with_ep", "ep"]` mesh for MoE (`parallelism.py:360-371`). On the rollout side vLLM is embedded **inside** the torchrun ranks rather than spawning its own workers: `LLM(..., distributed_executor_backend="external_launcher", ...)` (`rollout/vllm_rollout/vllm_rollout.py:298-304`) — which is exactly why rollout replicas can participate directly in the cosmos-managed NCCL groups.

---

## 5. Code organization style: function granularity

cosmos-rl's split rule is consistent: **orchestration stays monolithic; algorithmic math and reusable mechanisms get extracted**. Four concrete cases:

**1. `GRPOTrainer.step_training` — a ~978-line deliberately monolithic training step.**
`cosmos_rl/policy/trainer/llm_trainer/grpo_trainer.py:940-1917` (next `def reference_reset` starts at 1918). One method covers: one-time PP forward swizzling, CUDA timing events, multi-turn vs single-turn sample selection, positive-NLL flag setup, mini-batch math, old/ref logprob placeholders, PP microbatch validation — and onward through the optimizer step. Sample of the inline phase logic:

```python
# Do it once
if (
    pp_last_stage
    and self.parallel_dims.pp_enabled
    and not hasattr(self, "swizzled_forward")
):
    # Swizzle the forward function to return the current per-token logprobs.
    orig_forward = self.model.forward
```
(`grpo_trainer.py:954-961`). There is no helper soup — between line 940 and 1918 the only nested defs are tiny local closures.

**2. Extracted small helpers: pure-math, paper-citable units.**
`_apply_off_policy_mask` is a 44-line module-level function (`grpo_trainer.py:67-110`) with a literature citation as its reason to exist:

```python
def _apply_off_policy_mask(
    per_token_loss: torch.Tensor,
    ...
) -> torch.Tensor:
    """
    Off-Policy Sequence Masking.
    Reference:
    - DeepSeek-V3.2 Sec.3.1 Off-Policy Sequence Masking
```
Likewise `compute_loss` (`grpo_trainer.py:113-357`, ~245 lines) is module-level, not a method — it takes `config: CosmosConfig` and tensors explicitly, so it is callable from both the normal path and the PP-swizzled forward (`_swizzle_pp_grpo_forward`, `grpo_trainer.py:358-547`). Extraction criterion: **shared by ≥2 call paths or stateless math**, not "the function got long".

**3. The wrapper+`_impl` split — extraction only for lifecycle correctness.**
`DisaggregatedRolloutControlWorker.main_loop` (`cosmos_rl/rollout/worker/rollout_control.py:1735-1753`) is a 19-line wrapper whose only job is the try/finally around the ~69-line `_main_loop_impl` (`rollout_control.py:1755-1823`):

```python
try:
    self._main_loop_impl()
finally:
    wst = getattr(self, "_weight_sync_thread", None)
    if wst is not None:
        wst.stop()
```
with the docstring `"""Core main loop extracted for clean WST lifecycle management."""` (`rollout_control.py:1756`). They split a function for a *stated mechanical reason*, not for line count.

**4. Long methods absorb complexity via nested closures instead of private methods.**
`recv_weight_shard` (`rollout_control.py:462-816`, ~355 lines) keeps its tensor-staging logic as a nested `def recv_tensor_creator(underlying_tensor_view)` at `rollout_control.py:495-548` and a `completion_lambda` at `:615`, because both close over method-local state (the temp recv-tensor queue, target dtype). The class itself is 2,226 lines with ~40 methods — granularity inside the class is coarse.

**Size norms are very permissive**: 1000–2500-line files are normal for core modules — `rollout_control.py` 2226, `grpo_trainer.py` 2145, `policy/config/__init__.py` 1972, `dispatcher/status.py` 1786, `utils/parallelism_map.py` 1535 (`wc -l` over package). `PolicyStatusManager` alone spans `status.py:98-1507` (~1,400 lines, one class). One file = one subsystem, not one class per file.

**Type-machinery norms:**
- **Dataclasses for passive records only**: `Atom`/`ReplicaStatus`/`Replica` (`replica.py:30,145,163`) and `ParallelDims` — a flat dataclass of parallelism degrees with a `from_config` static constructor (`cosmos_rl/utils/parallelism.py:84-105`). Inconsistent though: `WeightSyncInstruction` is pure data yet hand-written with an `__init__` and `:param:` docstrings (`parallelism_map.py:35-60`).
- **A pseudo-dataclass idiom on commands**: every `Command` subclass declares bare class-level annotations *after* `__init__` assignments, e.g. `replica_name: str` at `command.py:106`, `src_replica_name: str / dst_replica_names: List[str]` at `command.py:171-173` — schema-as-documentation, since serialization is just `msgpack.packb(self.__dict__)` (`command.py:63-64`).
- **Enums: StrEnum for anything on the wire, plain Enum internally, bare class for int flags**: `class CommandType(StrEnum)` (`command.py:27-38`), `class PolicyStatus(StrEnum)` (`status.py:78`); internal `class TrainerPhase(enum.Enum)` (`grpo_trainer.py:61-64`), `class AsyncR2RSyncMode(Enum)` (`weight_sync.py:75`); enum-lite `class CommandScope: GLOBAL = 0; LOCAL = 1` (`command.py:41-43`) and `class Role` (`protocol.py:24`).
- **No `typing.Protocol` anywhere** — grep over the package finds zero; all interfaces are `ABC` (`comm/base.py:365`, `rollout_base.py:30`, `policy/trainer/base.py:105`). Dispatch by structure is done via registries + decorators instead: `@CommMixin.register_policy_command_handler(PolicyToRolloutUnicastCommand)` on `execute_policy_to_rollout_unicast` (`policy/worker/rl_worker.py:349-350`), implemented as a classmethod decorator writing into a class-level registry (`comm/base.py:52-72`).
- **Pydantic for all config, with self-documenting `Field(description=...)`** generating the docs site via a `CustomJsonSchemaGenerator` that strips `hide_in_doc` fields (`policy/config/__init__.py:44-70,73-105`); validation via `@model_validator(mode="after")` (`config/__init__.py:104-108`).
- **Comment style**: author-tagged TODO/FIXME with initials — `# FIXME: (lms) refactor this hard-coded check...` (`policy/train.py:39`), `# TODO(zjx): there need failure tolerance for nccl send and recv` (`rl_worker.py:301`), `# Note: (jiaxinc) We still need to upload the output text, even if it is empty...` (`rollout_control.py:1874-1877`). Heavy use of **asserts-with-prose as design documentation**, e.g. the main loop assert explaining *why* async r2r sync is incompatible with async rollout mode (`rollout_control.py:1739-1746`). They also leave commented-out code in place with the replacement rationale right below it (`rollout_control.py:1866-1877`).

---

## 6. Naming conventions

- **Role-first package layout, not layer-first**: top-level packages are `cosmos_rl/dispatcher/`, `cosmos_rl/policy/` (training side), `cosmos_rl/rollout/` (inference side), `cosmos_rl/reference/`, `cosmos_rl/comm/`, `cosmos_rl/launcher/`. "Policy" = trainer replica, "rollout" = generation replica — the RL role is the namespace.
- **Backend-suffixed subpackages**: `rollout/vllm_rollout/`, `rollout/trtllm_rollout/`, `rollout/wfm_rollout/`, `rollout/vla_rollout/`, `rollout/diffuers_rollout/` (sic — typo shipped).
- **Class suffix taxonomy** (real names):
  - `*Command` — one class per control-plane verb: `WeightResumeCommand`, `BuildMeshCommand`, `PolicyToRolloutUnicastCommand`, `RolloutToRolloutBroadcastCommand`, `DataFetchCommand` (`cosmos_rl/dispatcher/command.py:99,123,239,315,382`). Names encode **direction and cast type** (`PolicyToPolicy…Broadcast` vs `…Unicast`).
  - `*Manager` for stateful bookkeeping: `PolicyStatusManager` (`cosmos_rl/dispatcher/status.py:98`), `RolloutStatusManager` (`status.py:1508`).
  - `*Base` for ABCs: `WorkerBase` (`cosmos_rl/comm/base.py:365`), `PolicyWorkerBase` (`cosmos_rl/policy/worker/base.py:33`), `RolloutBase` (`cosmos_rl/rollout/rollout_base.py:30`).
  - `*Mixin` for cross-cutting capability: `CommMixin` carries registration/heartbeat/redis (`comm/base.py:52`), mixed in as `class PolicyWorkerBase(WorkerBase, CommMixin)` (`policy/worker/base.py:33`).
  - `*Registry`: `CommandRegistry`, `PolicyCommandRegistry`, `RolloutCommandRegistry` (`command.py:482-512`), `RolloutRegistry` (`rollout_base.py:233`).
  - `*Request` pydantic wire models: `HandshakeInitiatorRequest`, `RolloutRequest`, `NcclErrRequest` etc., all in one protocol module (`cosmos_rl/dispatcher/protocol.py:37-172`).
- **Domain vocabulary invented and documented**: `Atom` = "the smallest unit of a computation mesh. Usually it is a single GPU process" (docstring, `cosmos_rl/dispatcher/replica.py:31-36`); `Replica` groups atoms.
- **Aggressive abbreviations once a term is established**: `p2r`/`r2r` for policy-to-rollout / rollout-to-rollout (`_execute_p2r_recv` at `rollout_control.py:1034`; `enqueue_p2r`, `_execute_r2r`, `r2r_barrier` in `cosmos_rl/rollout/worker/weight_sync.py:296,387,521`), `wst` for WeightSyncThread (`process_wst_deferred_actions`, `ensure_wst`, `weight_sync.py:237,650`), `wfm` for world foundation model (`wfm_trainer.py`, `wfm_rollout/`).
- **snake_case verb functions, module-level, with `worker` as explicit first arg** — `weight_sync.py` is essentially procedural: `get_async_r2r_sync_mode(worker)`, `create_buffer_model(worker)`, `sync_buffer_to_live(worker)`, `install_inference_sync(worker)` (`weight_sync.py:99,114,178,672`) rather than more methods on the already-huge worker class.
- **Hook-suffix discipline on the backend ABC**: `RolloutBase` exposes `post_init_hook`, `post_init_engine_hook`, `pre_get_params_for_sync_hook` / `post_get_params_for_sync_hook` (`rollout_base.py:53,115,138,162`) — explicit `*_hook` names mark the extension surface.
- **Warts they tolerate**: `HighAvailabilitylNccl` (typo'd "l", `cosmos_rl/utils/distributed.py:427`), `diffuers_rollout` dir; descriptive-but-long `DisaggregatedRolloutControlWorker` (`rollout_control.py:93`).

---

## 7. End-to-end flow trace

One GRPO step in disaggregated mode. Each hop is `file:line` in this checkout.

**Phase A — prompt acquisition (rollout replica)**
1. Rollout main loop iteration: `cosmos_rl/rollout/worker/rollout_control.py:1758-1782` — passes `state.weight_synced()` gate, calls `request_new_prompts`.
2. Rank 0 HTTP fetch: `rollout_control.py:1476-1478` `api_client.get_next_prompt(batch_size × dp_size)`.
3. Controller route: `cosmos_rl/dispatcher/run_web_panel.py:412-418` → `controller.get_batched_prompt`.
4. Admission/throttling/weight-version tagging: `cosmos_rl/dispatcher/controller.py:245-469`; actual dataset read in `self.data_fetcher.get_batched_prompt` (`controller.py:360-367`); `samples_on_the_fly` bumped at `controller.py:464-467`.
5. Prompts broadcast + DP scatter, pushed to `_prompt_queue`: `rollout_control.py:1518-1557`.

**Phase B — generation + reward**
6. Freshness check then `one_step_generation()`: `rollout_control.py:1807-1817`.
7. `_call_rollout_generation` injects `current_weight_version` and calls backend: `rollout_control.py:1715-1732`.
8. vLLM generate: `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:639-643` (`rollout_engine.generate(new_prompts, sampling_params=...)`); every `COSMOS_ROLLOUT_STEP_INTERVAL` engine steps the patched `llm_engine.step` consumes weight-sync commands mid-generation (`vllm_rollout.py:93-119`).
9. Filter empty/invalid completions; TP-rank-0/last-PP-stage only (`should_report`, `rollout_control.py:250-252`) stamps results with `current_weight_version` and enqueues reward computation: `rollout_control.py:1889-1918` → `reward_dispatcher.enqueue_rewards_cal`. Rewards **and GRPO advantages are computed on the rollout worker**, not the controller (`cosmos_rl/dispatcher/algo/grpo.py:38-47` `compute_advantage = (r − mean)/(std+eps)`).
10. Next loop iteration drains scored payloads: `report_rollouts` (`rollout_control.py:1670-1713`) → `api_client.post_rollout_completion(RolloutRequest(...))`.

**Phase C — controller buffering & step trigger**
11. Route `POST /rollout`: `run_web_panel.py:434-507` — `extract_rollouts` flattens per-prompt payloads into `Rollout` objects carrying reward/advantage (`cosmos_rl/utils/payload.py:40-99`).
12. Staleness filter: `cosmos_rl/dispatcher/status.py:812-855` (`filter_outdated_rollouts`).
13. Enqueue + trigger: `controller.py:510-517` → `status.py:768-797` (`put_rollouts`) → `status.py:744-766` (`put_rollout` → `try_trigger_data_fetch_and_training`).
14. When gated conditions hold: `status.py:1391-1397` `current_step += 1`; round-robin dispatch of rollouts to each policy replica's Redis stream `status.py:1434-1439` → `replica.py:274-283` (`publish_rollout`); `DataFetchCommand` published per replica `status.py:1444-1456`.

**Phase D — policy training step**
15. Policy rank 0 rollout-fetch thread pulls the stream into `data_queue`: `cosmos_rl/policy/worker/rl_worker.py:256-272`.
16. Command thread receives `DataFetchCommand`: `rl_worker.py:570-597`; main loop executes it: `rl_worker.py:844-847` → `execute_data_fetch` `rl_worker.py:510-568`.
17. `dispatch_rollouts` blocks for `items_count` rollouts and DP-scatters: `rl_worker.py:711-767`.
18. `trainer.step_training(rollouts, current_step, ...)`: `rl_worker.py:539-547` → GRPO loss/minibatching in `cosmos_rl/policy/trainer/llm_trainer/grpo_trainer.py:940-1050` (advantages read straight off rollouts at `grpo_trainer.py:1026-1027`; backward at `grpo_trainer.py:1777`).
19. Master rank acks: `rl_worker.py:558-565` `post_policy_train_ack`.

**Phase E — weight sync back to rollout**
20. Route `POST /train_ack`: `run_web_panel.py:516-531` → `status.py:1035-1062`; when all replicas REDUCED, `need_sync_weight` computed `status.py:1070-1077`.
21. `trigger_weight_sync`: `status.py:1269-1304` — publishes `PolicyToRolloutUnicastCommand` (to one policy + one rollout replica) and `RolloutToRolloutBroadcastCommand` (to all rollout replicas) (`command.py:282-308, 345-370`).
22. Policy side P2R send: `rl_worker.py:349-497` — per-shard instructions fetched once from controller (`rl_worker.py:365-369`), sliced views sent via grouped NCCL (`rl_worker.py:398-477`).
23. Rollout side P2R recv: `rollout_control.py:1002-1032` → `_execute_p2r_recv` `rollout_control.py:1034-1201`, ending in `self.state.set_weight_synced()` (`rollout_control.py:1199`).
24. R2R broadcast to all rollout replicas over the mesh built by `BuildMeshCommand`: `rollout_control.py:1206-1331`; afterwards `current_weight_version = command.weight_step` (`rollout_control.py:1355-1360`), validation flag possibly set (`1364-1381`), shutdown if `weight_step >= total_steps` (`1383-1394`, `command.py:372-375`).
25. New `current_weight_version` unblocks Phase A's freshness gate (`rollout_control.py:1807-1815`) and step 20's `try_trigger_data_fetch_and_training` (`status.py:1264`) launches the next training step — closing the loop.

Termination: dataset exhaustion → `get_batched_prompt` returns `is_end` → rollout sends end signal (`rollout_control.py:1631-1646`) → controller recomputes `total_steps` from leftover buffer and may issue a fake last `DataFetchCommand` (`run_web_panel.py:437-485`, `status.py:1375-1381`; the `is_fake_last_cmd` branch in `try_trigger_data_fetch_and_training`, `status.py:1375-1390`).

Worker entry chain for reference: `cosmos-rl` console script → torchrun → `cosmos_rl/policy/train.py:26-58` / `cosmos_rl/rollout/rollout_entry.py:25-50` → `worker.work()/main_loop()` (`cosmos_rl/rollout/worker/llm_worker.py:65-92`; `cosmos_rl/policy/worker/base.py:115-121`).

---

## 8. Ideas worth borrowing for wm-infra

1. **Self-triggering command classes beat a central dispatcher switch.** Each `Command` subclass carries a `@classmethod trigger(...)` that builds itself, publishes via redis, and mutates controller-side bookkeeping (`PolicyToRolloutUnicastCommand.trigger` flips `dst_replica.weights_loaded_in_view_of_command` and one-shots `_do_weight_sync_check_flag`, `command.py:282-308`). Instead of EngineLoop knowing how to emit every message, give each control message a `trigger`/`from_dict` pair so creation, side effects, and serialization live in one greppable place. Their direction-encoding names (`PolicyToRolloutUnicast`) would map cleanly to e.g. `TrainerToRolloutWeightSync`.
2. **Decorator-registered handler table mapping command type → worker method** (`@CommMixin.register_policy_command_handler(...)` at `rl_worker.py:292-511`, registry in `comm/base.py:52-86`, with a backend dimension for rollout: vllm/trtllm in `command.py:497-512`). wm-infra's IterationRunner phase handling is implicit in `_advance`; a registry keyed by phase (and model family, mirroring their backend key) would make adding a model family additive instead of edit-the-switch.
3. **Keep step_training-style monoliths monolithic — extract only pure math and dual-path code.** Their GRPO step is ~978 lines and readable top-to-bottom; only `compute_loss`/`_apply_off_policy_mask` left, because PP swizzling needs to call the same loss (`grpo_trainer.py:113,358,940`). This matches wm-infra's existing no-single-caller-helpers stance — validation that a serious NVIDIA RL codebase made the same call. Borrow the `main_loop`/`_main_loop_impl` try-finally split (`rollout_control.py:1735-1756`) for the EngineLoop: a thin wrapper that owns background-thread shutdown is the *one* extraction worth making.
4. **Precomputed weight-sync instruction plan as plain data.** `WeightSyncInstruction` / `WeightSyncInstructionsPerParam` / `WeightSyncInstructionsGroup` (`parallelism_map.py:35-82`) describe `(policy_rank, rollout_rank, per-dim slice strategy)` and are computed once by `ParallelTopoMapperGroup` (`parallelism_map.py:792`), then merely executed in `recv_weight_shard`. For wm-infra's trainer→rollout sync across FSDP shards, separating "plan the resharded transfer" (data) from "execute NCCL ops" (loop) is the structure to copy — but as actual `@dataclass`es, fixing their inconsistency.
5. **`weight_version` staleness gating at both ends.** Every prompt is stamped at admission with the predicted training step (`controller.py:264-276`); the rollout main loop refuses to generate it until `first_payload.weight_version <= self.current_weight_version + allowed_outdated_steps` (`rollout_control.py:1807-1815`); the controller drops late rollouts on arrival (`status.py:812-855`). Stamping every wm-infra prompt/request with the weight step it was scheduled under, plus an explicit `allowed_outdated_steps` config, is a cheap, borrowable correctness guard for GRPO on-policyness with continuous batching.
6. **Skip Ray; their stack proves FastAPI + Redis streams + Popen is enough** for multi-replica RL with dynamic scaling (`controller.py:84-230`, `command.py:114`, `launcher/utility.py:797`). wm-infra already has a FastAPI gateway; a Redis-stream command channel per replica name is the minimal increment for disaggregated rollout/training, with `BuildMeshCommand` (`command.py:123-149`) as the template for rank assignment. The colocated mode shows the same command vocabulary can be re-hosted on in-memory queues (`colocated/utils.py:22-49`) — design the command bus as an interface, not as "Redis".
7. **Mid-generation command checkpoints.** Patching the inference engine's step function to drain a command queue every N iterations (`vllm_rollout.py:77-119`) is the cheapest way to make a long synchronous `generate()` interruptible for weight sync — directly applicable to wm-infra's long video-diffusion denoise loops: check a command/feedback mailbox every K denoise steps instead of only at request boundaries. Corollary: any cache keyed on weights (prefix cache analog) must be disabled or versioned (`vllm_rollout.py:317-319`).
8. **Async weight sync into a buffer model on a side CUDA stream** (`weight_sync.py:16-50`), with `paused(wait_for_active_tasks=True)`-style drain semantics in the async scheduler (`rollout_task_scheduler.py:595-707`), hides P2R/R2R latency behind generation. The wm-infra analog: receive new trainer weights into a shadow parameter set during decode, swap at a denoise-loop boundary.
9. **Hook-suffix discipline for backend ABCs.** `RolloutBase`'s `post_init_hook` / `pre_get_params_for_sync_hook` etc. (`rollout_base.py:53,115,138,162`) make the extension surface obvious vs ordinary methods. wm-infra's model-family interface (Wan/cosmos) would benefit from the same naming split between "engine calls this every step" and "family may override around weight sync".
10. **One anti-pattern to avoid**: `Command.deserialize` is a hand-maintained if/elif chain over `CommandType` (`command.py:67-87`) that duplicates the subclass list and will silently rot — exactly the duplicated-constant failure mode wm-infra's AGENTS.md bans; derive that map from a registry instead.

---

## 9. Source-of-truth index

Merged from all three readers; deduplicated. Items marked ✔ were re-verified directly against this checkout during merge.

| Claim | Path:lines |
|---|---|
| Entry points `cosmos-rl`/`cosmos-cli` | `pyproject.toml` `[project.scripts]` (≈:57-59) |
| README: single-controller, dynamic NCCL groups | `README.md:13-27` |
| No Ray anywhere ✔ (grep `^import ray\|^from ray` = 0 hits) | repo-wide grep over `*.py`; absent from `pyproject.toml` |
| torchrun (c10d rdzv) per replica; mpirun for trtllm; `COSMOS_ROLE` | `cosmos_rl/launcher/launch_replica.sh:119,152-157` + role block |
| Role dispatch | `cosmos_rl/launcher/worker_entry.py:31-105` |
| Launcher computes GPUs from parallelism product; p2r-ratio; modes | `cosmos_rl/launcher/launch_all.py:421-456,460-495,510-548` |
| Per-node command slice + process monitor loop; Lepton path | `cosmos_rl/launcher/launch_all.py:551-612,760-884,898-933` |
| Node placement (`NodesManager`); colocated n_rollouts=0; Popen | `cosmos_rl/launcher/utility.py:451-554,508-513,797-801` |
| Controller forks redis-server ✔; LFU maxmemory; status managers setup | `cosmos_rl/dispatcher/controller.py:147-199` (fork at :157-161) |
| Prompt admission: batch sizing, weight-version tagging | `cosmos_rl/dispatcher/controller.py:245-276` |
| Soft throttle (allowed_outdated_steps) / DAPO variant | `cosmos_rl/dispatcher/controller.py:291-338` |
| Hard throttle (max_inflight_steps) | `cosmos_rl/dispatcher/controller.py:348-352` |
| On-policy exact prompt accounting + retry cap; samples_on_the_fly | `cosmos_rl/dispatcher/controller.py:369-434,464-467` |
| NCCL error sweep / hang replica cleanup | `cosmos_rl/dispatcher/controller.py:609-668`; `status.py:482-490` |
| FastAPI app, lifespan heartbeat monitor, uvicorn | `cosmos_rl/dispatcher/run_web_panel.py:100-127,680-685` |
| HTTP routes: register/NCCL KV/shard insts; /next_prompt, /rollout, /train_ack | `cosmos_rl/dispatcher/run_web_panel.py:170-537` (:412-422, :434-507, :516-531) |
| `Atom`/`Replica` model; all_atoms_arrived; rollout publish via Redis | `cosmos_rl/dispatcher/replica.py:30-52,163-197,267-272,274-283` |
| `MESH_NAMES = ["pp","dp_shard","cp","tp"]` ✔; `Role`; `*Request` models | `cosmos_rl/dispatcher/protocol.py:21,24,37-172` |
| Command enum ✔ (incl. ALL_REDUCE/STOP/VALIDATE/DUMMY); CommandScope | `cosmos_rl/dispatcher/command.py:27-43` |
| msgpack serialize; if/elif deserialize chain (anti-pattern) | `cosmos_rl/dispatcher/command.py:63-87` |
| Command classes + `trigger` classmethods; BuildMesh rank assignment | `cosmos_rl/dispatcher/command.py:99-479` (:132-145, :282-308, :345-375, :444-475) |
| Registries (Policy/Rollout, backend-keyed) | `cosmos_rl/dispatcher/command.py:482-512` |
| Rollout buffer queues, per-rank queues; PolicyStatus enum | `cosmos_rl/dispatcher/status.py:78-148` |
| put_rollout(s) → immediate trigger | `cosmos_rl/dispatcher/status.py:744-797` |
| Outdated-rollout filtering (estimated_step) | `cosmos_rl/dispatcher/status.py:812-855` |
| NCCL payload cleanup for discarded rollouts | `cosmos_rl/dispatcher/status.py:857-948` |
| train_ack → all_reduced → weight sync + next step | `cosmos_rl/dispatcher/status.py:1035-1077,1259-1264` |
| trigger_weight_sync (P2R + R2R commands) | `cosmos_rl/dispatcher/status.py:1269-1304` |
| rollouts_enough_for_one_step | `cosmos_rl/dispatcher/status.py:1306-1324` |
| Step trigger gate + step advance + round-robin dispatch + DataFetchCommand ✔ | `cosmos_rl/dispatcher/status.py:1362-1456` |
| Registration orchestration (WeightResume, P2P broadcast, first P2R) | `cosmos_rl/dispatcher/status.py:404-630,1702-1786` |
| `RolloutStatusManager` | `cosmos_rl/dispatcher/status.py:1508` |
| Worker register/heartbeat (mp.Process), redis init; CommMixin/WorkerBase; handler decorator | `cosmos_rl/comm/base.py:52-106,249-303,331-365` |
| Policy worker pulls config via HTTP; torch.distributed only intra-replica | `cosmos_rl/policy/policy_entry.py:32-48` |
| Policy fetch threads, BuildMesh short-circuit, command broadcast | `cosmos_rl/policy/worker/rl_worker.py:256-272,570-635` |
| DataFetch handler → dispatch_rollouts → step_training → ack | `cosmos_rl/policy/worker/rl_worker.py:510-568,677-767` |
| Policy main loop (pure command executor) | `cosmos_rl/policy/worker/rl_worker.py:792-853` |
| P2R send (policy): shard insts fetch, cosmos_slice, nccl_group batching | `cosmos_rl/policy/worker/rl_worker.py:349-497` |
| Policy shard-info post | `cosmos_rl/policy/worker/rl_worker.py:204-213` |
| Rollout main loop ✔ (gates, freshness check :1807-1815, one_step_generation) | `cosmos_rl/rollout/worker/rollout_control.py:1755-1823` |
| `main_loop`/`_main_loop_impl` lifecycle split; assert-as-design-doc | `cosmos_rl/rollout/worker/rollout_control.py:1735-1756,1739-1746` |
| Rollout State bitmask machine | `cosmos_rl/rollout/__init__.py:58-88` |
| Command poller thread + WST routing | `cosmos_rl/rollout/worker/rollout_control.py:1407-1449` |
| consume_command quiescence / P2R-pairing logic | `cosmos_rl/rollout/worker/rollout_control.py:1560-1629` |
| Prompt fetch + DP scatter on worker; fetch lock; prefetch thread | `cosmos_rl/rollout/worker/rollout_control.py:1459-1558,2118-2199` |
| End-of-data signal; report_rollouts; generation call chain | `cosmos_rl/rollout/worker/rollout_control.py:1631-1646,1670-1732` |
| Reward enqueue, should_report (TP0/last-PP) | `cosmos_rl/rollout/worker/rollout_control.py:250-252,1889-1918` |
| `recv_weight_shard` ~355 lines, nested closures, temp-queue backpressure, FSDP reshard | `cosmos_rl/rollout/worker/rollout_control.py:462-816` (:490-524, :495-548, :615) |
| P2R recv handler + pipelined NCCL groups + copy_stream | `cosmos_rl/rollout/worker/rollout_control.py:1002-1201` |
| R2R broadcast, weight-version/validation/shutdown bookkeeping; R2R global mesh build | `cosmos_rl/rollout/worker/rollout_control.py:372-421,1206-1405` |
| Rollout shard-info post; work() entry | `cosmos_rl/rollout/worker/rollout_control.py:274-292,2201-2226` |
| Async stream feed sizing + non-blocking collection | `cosmos_rl/rollout/worker/rollout_control.py:1976-1981,2024-2071` |
| Mid-generation preemption: patched vLLM `llm_engine.step` ✔ | `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:77-120` |
| vLLM init ✔: external_launcher, prefix cache off, sleep off, chunked prefill | `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:298-327`; async `vllm_rollout_async.py:168-181` |
| vLLM generate call | `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:639-643` |
| Async scheduler loop, max_concurrent admission, pause/resume drain | `cosmos_rl/rollout/worker/asynchronous/rollout_task_scheduler.py:320-368,595-707` |
| Async buffer-model weight sync design; procedural worker-first functions; AsyncR2RSyncMode | `cosmos_rl/rollout/worker/weight_sync.py:16-50,75,99-692` |
| `HighAvailabilitylNccl` (typo): bg thread, abort+rebuild, UID handshake, retry+error report | `cosmos_rl/utils/distributed.py:427-609` (:479-500, :533-547, :577-609) |
| NCCL UID KV over HTTP | `cosmos_rl/dispatcher/api/client.py:276-311`; `run_web_panel.py:352-371` |
| APIClient surface | `cosmos_rl/dispatcher/api/client.py:68-581` |
| P2R fused comm ranks; ZMQ CUDA-IPC same-device path | `cosmos_rl/collective/collective.py:125-138,234-290,342-423` |
| `ParallelizedShardMapper` (controller-side shard scheme, mp) | `cosmos_rl/utils/parallelism_map.py:915-946` |
| `WeightSyncInstruction*` plan-as-data; `ParallelTopoMapperGroup` | `cosmos_rl/utils/parallelism_map.py:35-82,792` |
| `ParallelDims` dataclass + DeviceMesh order + EP mesh | `cosmos_rl/utils/parallelism.py:84-118,231-261,360-371` |
| Redis streams (XADD/XREAD, maxlen cap) | `cosmos_rl/utils/redis_stream.py:126-254` (:144, :207) |
| Rollout payload → Rollout objects (extract_rollouts) | `cosmos_rl/utils/payload.py:40-99` |
| GRPO advantage formula (computed on rollout worker) | `cosmos_rl/dispatcher/algo/grpo.py:38-47` |
| `step_training` ~978-line monolith; PP swizzle; backward | `cosmos_rl/policy/trainer/llm_trainer/grpo_trainer.py:940-1917` (:954-972, :1026-1027, :1777) |
| `_apply_off_policy_mask` / `compute_loss` / `_swizzle_pp_grpo_forward`; TrainerPhase | `grpo_trainer.py:61-64,67-110,113-357,358-547` |
| Colocated in-memory CommandDispatcher / ColocatedController | `cosmos_rl/colocated/utils.py:22-49`; `cosmos_rl/colocated/controller.py:74` |
| `RolloutBase` `*_hook` extension surface; RolloutRegistry | `cosmos_rl/rollout/rollout_base.py:30-233` (:53, :115, :138, :162) |
| Pydantic config + doc-schema generator + model_validator | `cosmos_rl/policy/config/__init__.py:44-108` (file: 1972 lines) |
| Author-tagged TODO/FIXME; commented-out code with rationale | `policy/train.py:39`; `rl_worker.py:301`; `rollout_control.py:1866-1877` |
| End-of-data handling (fake last DataFetch) | `run_web_panel.py:437-485`; `status.py:1375-1390` |
| Worker entry chain | `cosmos_rl/policy/train.py:26-58`; `cosmos_rl/rollout/rollout_entry.py:25-50`; `cosmos_rl/rollout/worker/llm_worker.py:65-92`; `cosmos_rl/policy/worker/base.py:115-121` |
| File-size distribution (1000–2500-line core modules) | `wc -l` over `cosmos_rl/**/*.py` |

---

# Part II — Deep Dive

Part I read the system at the orchestration level. Part II goes below it: the training engine's tensor-level mechanics (§10), the data-structure-level life of a sample (§11), the plugin/extension surfaces (§12), startup order and failure handling (§13), and a second round of borrow/avoid judgments (§14). All `path:line` references re-verified against this checkout during the merge.

## 10. Training engine internals: from rollout samples to gradients

Part I §3.6 covered `step_training` at the loop level (DataFetchCommand → dispatch → ack) and §5 noted it is a ~978-line deliberate monolith. This section is what happens *inside*.

### 10.1 Batch hierarchy and the phase machine

One `step_training` call receives one replica's batch of `Rollout` objects (already carrying `reward` and `advantage` computed on the rollout worker, Part I §7 step 9). Inside the step there are four nested batch levels (`cosmos_rl/policy/trainer/llm_trainer/grpo_trainer.py:1034-1047`):

```
train_batch_per_replica  (whole step batch)
  └─ batch_size_per_optimize   # one optimizer.step() per slice (grpo_trainer.py:1106)
       └─ mini_batch           # one forward+backward per mini-batch (grad accumulation unit)
            └─ pp_micro_batch_size  # only when PP is enabled
```

Config semantics: `mini_batch` "is used to split the batch per optimization into smaller batches to fit into GPU memory" and `batch_size_per_optimize` "is split into smaller batches which each performs one step optimization" (`cosmos_rl/policy/config/__init__.py:546-554`). On top of this, `mu_iterations` (μ in the GRPO paper) replays the whole batch μ times against frozen old-logprobs (`config/__init__.py:541-544`; loop at `grpo_trainer.py:1098-1102`).

The step body is a phase machine (`grpo_trainer.py:61-65, 1074-1079`):

```python
trainer_phases = []
if need_compute_ref:
    trainer_phases.append(TrainerPhase.REF_COMPUTE)
if need_compute_old_ahead:
    trainer_phases.append(TrainerPhase.OLD_LOGP_COMPUTE)
trainer_phases.append(TrainerPhase.TRAIN)
```

- `REF_COMPUTE` runs only when `kl_beta != 0`. There is **no second model instance for the reference policy** — the trainer keeps a CPU-resident `reference_state_dict` and *swaps it into the live model* to compute reference logprobs, then swaps back (`grpo_trainer.py:1955-1970`):

```python
def _swap_model_state_dict(self):
    kl_beta = self.config.train.train_policy.kl_beta
    if kl_beta != 0.0:
        with torch.cuda.stream(self.train_stream):
            model_state_dict = self.model.state_dict()
            ...
            ref_clone = reference_state_dict[key].clone()
            reference_state_dict[key].copy_(value)   # stash current policy
            value.copy_(ref_clone)                   # load reference weights
        return True, kl_beta
```

  The swap-back happens at the start of the next phase (`grpo_trainer.py:1089-1091`). The reference snapshot is captured at `weight_resume` time as detached CPU copies (`grpo_trainer.py:2029-2036`) and optionally refreshed every `reference_reset_interval` steps, optionally rebuilding the optimizer too (`grpo_trainer.py:1918-1949`).
- `OLD_LOGP_COMPUTE` only runs when `batch_size > per_optimize_batch_size` and rollout logprobs are not reused (`grpo_trainer.py:1064-1067`) — because once the first optimizer step fires mid-batch, later mini-batches can no longer compute π_old on the fly; old logprobs must be precomputed for the whole batch under the unmodified weights.
- Ref/old phases run under `torch.set_grad_enabled(phase == TrainerPhase.TRAIN)` and `set_model_eval()` (`grpo_trainer.py:1086-1097`).

### 10.2 Logprob recomputation

Per mini-batch, the data packer turns prompt+completion into `input_ids` + `logprob_masks` (`grpo_trainer.py:1011-1018, 1237-1242`); the forward yields raw logits which are temperature-scaled to match sampling: `raw_logits = raw_logits / config.train.train_policy.temperature` (`grpo_trainer.py:1552-1561`). Logits are cast to `train.logprob_dtype` (default `"float32"`, `config/__init__.py:852-856`) before the gather (`grpo_trainer.py:1996-2003`).

The core recompute is `compute_logprobs` in `cosmos_rl/utils/util.py:919-996`: shift labels by one, select only loss tokens, and return a **flat ragged layout**:

```python
shifted_input_ids[:, :-1] = input_ids_batch[:, 1:]
...
effective_input_ids = shifted_input_ids[logprob_masks]   # [n_logprob_tokens,]
...
masked_seqlens = logprob_masks.sum(dim=-1)               # [bsz,]
cu_seqlens[1:] = torch.cumsum(masked_seqlens, dim=0)
logps = selective_log_softmax(effective_logits, effective_input_ids)
```

(`utils/util.py:944-996`; `selective_log_softmax` at `utils/util.py:314`). All downstream loss math operates on this `[n_logprob_tokens]` flat vector with `cu_seqlens` sequence boundaries — there is no padded `[B, T]` loss tensor. Token entropy is computed on the side, chunked to bound memory (`entropy_from_logits_with_chunking`, `utils/util.py:906`; collected at `util.py:976-986`).

**Where π_old comes from — three mutually exclusive sources** (`grpo_trainer.py:1606-1634`):
1. *On-the-fly*: first μ-iteration without precompute → `old = current_per_token_logprobs.detach()` (`grpo_trainer.py:1628-1630`) — i.e. ratio ≡ 1 on the first pass, the standard GRPO trick.
2. *Precomputed-ahead* via the `OLD_LOGP_COMPUTE` phase (`grpo_trainer.py:1597-1604`).
3. *Rollout-engine logprobs* when `use_rollout_logprobs_for_loss` is set: the vLLM-sampled logprobs are concatenated and used directly as π_old (`grpo_trainer.py:1616-1626`), with `prompt_logprobs + completion_logprobs` masked by `logprob_masks` (`grpo_trainer.py:1373-1397`). This cannot be combined with `use_decoupled_loss` (validator at `config/__init__.py:691-692`).

### 10.3 The loss function, line by line

Part I §5 noted `compute_loss` is a module-level pure function shared by the normal and PP-swizzled paths (`grpo_trainer.py:113-353`); here is the math. Advantages arrive per-sequence, expanded to per-token and masked (`grpo_trainer.py:1198-1203, 1636-1639`; flattened inside `compute_loss` at 130-132).

**Importance ratio** (`grpo_trainer.py:162-203`): `negative_approx_kl = current_token_logps - old_per_token_logps`; for the `gspo` variant it is averaged per sequence (clamped at 10) and broadcast back to tokens via `.expand` to keep gradient flow (`:170-196`); otherwise clamped to ±20 per token. Then `importance_ratio = exp(...)`.

**Policy gradient term** — two regimes (`grpo_trainer.py:205-244`):

```python
if config.train.train_policy.aipo_rho is not None:
    # one-sided clipping (AIPO) to correct off-policyness of async rollouts
    per_token_loss = -torch.clamp(importance_ratio, max=rho) * current_advantages
else:
    # the standard grpo loss with dual-clip PPO: https://arxiv.org/pdf/1912.09729
    ...
    loss1 = importance_ratio * current_advantages
    loss2 = importance_ratio_clipped * current_advantages       # clip to [1-eps_low, 1+eps_high]
    if config.train.train_policy.variant == "gspo":
        per_token_loss = -torch.min(loss1, loss2)
    else:
        loss3 = -config.train.train_policy.lower_bound_ratio * current_advantages
        clip_losses1 = -torch.min(loss1, loss2)
        clip_losses2 = torch.min(loss3, clip_losses1)
        per_token_loss = torch.where(current_advantages < 0, clip_losses2, clip_losses1)
```

i.e. dual-clip PPO: the extra `lower_bound_ratio` clamp applies only to negative-advantage tokens. `epsilon_low/high` may be set negative to disable a side (`grpo_trainer.py:215-224`).

**Decoupled loss (AREAL)** — when the rollout engine's own logprobs are shipped (`use_decoupled_loss`, `config/__init__.py:629-639`), a *behavior* importance weight corrects π_old vs π_behavior drift, hard-zeroing tokens above the cap (`grpo_trainer.py:246-256`):

```python
behav_kl = old_per_token_logps - rollout_per_token_logps
behav_imp_weight = torch.exp(behav_kl)
behav_mask = (behav_imp_weight <= config...behav_imp_weight_cap) ...
behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
per_token_loss = per_token_loss * behav_imp_weight
```

**Off-policy sequence masking (DeepSeek-V3.2)** — `_apply_off_policy_mask` (`grpo_trainer.py:67-110`) zeroes whole sequences whose advantage is negative *and* whose mean `log π_old − log π_θ` exceeds `off_policy_masking_delta`: `seq_mask = ((advantage_by_sequence >= 0) | (seq_mean_logprob_diff <= delta))` (`grpo_trainer.py:105-109`).

**KL to reference** — the k3 estimator with optional importance-sampling debiasing (`grpo_trainer.py:271-293`):

```python
kl_ratio = ref_per_token_logps - current_token_logps
kl_ratio = torch.clamp(kl_ratio, min=-20, max=20)
if config.train.train_policy.unbiased_kl_estimate:
    importance_sampling_ratio = torch.exp(current_token_logps - old_per_token_logps)
    kl_loss = (importance_sampling_ratio * (torch.exp(kl_ratio) - kl_ratio - 1)).clamp(min=-10, max=10)
else:
    kl_loss = (torch.exp(kl_ratio) - kl_ratio - 1).clamp(min=-10, max=10)
```

**Aggregation** — per-sequence token sums are then reduced under four selectable normalizations (`grpo_trainer.py:295-348`): `seq-mean-token-mean` (with optional Dr.GRPO fixed `unbiased_loss_max_tokens` denominator, `:312-321`), `seq-mean-token-sum`, `token-mean` (with optional `balance_dp_token`, which all-reduces the token count across the intra-replica DP group **and** across replicas via the HA-NCCL comm so every DP worker divides by the global token count, `:326-343`), and `token-sum`. Return value: `(per_token_loss + kl_loss * kl_beta, per_token_loss, kl_loss)` (`grpo_trainer.py:349-353`).

**Add-ons applied at the call site** (`grpo_trainer.py:1739-1767`): entropy bonus `loss += -entropy_coeff * effective_entropy`, and an optional "positive-NLL" auxiliary — plain NLL on tokens of positive-reward samples (`positive_flags` built from `rollout.reward > 0` at `grpo_trainer.py:995-1004`): `l_nll = -current_per_token_logprobs[flat_mask].mean(); loss = loss + pos_coef * l_nll`. Finally each mini-batch loss is scaled by `loss_scaling_factor = len(mini_batch)/len(per_optimize_batch)` (`grpo_trainer.py:1181-1183, 1771`) before `loss.backward()` (`grpo_trainer.py:1777`).

The same trainer doubles as an **on-policy distillation engine**: teacher logprobs fetched per-UUID from a reference replica are converted into *advantage corrections* rather than a separate loss — reverse-KL or generalized top-k JSD per token, optionally discounted-future-summed along the sequence, then `advantages = current_advantages + (-kl_penalty_coef * divergence)` (`grpo_trainer.py:692-835`, key line 807/828; wiring at 1205-1235, 1640-1714).

### 10.4 The pipeline-parallel path: loss inside the swizzled forward

With PP, loss must be computed inside the last stage's forward so `Schedule1F1B` can backprop it. `step_training` monkey-patches the last stage's `forward` once (`grpo_trainer.py:954-972`) with `_swizzle_pp_grpo_forward` (`grpo_trainer.py:358-544`), which receives smuggled per-microbatch metadata as input tensors — `mini_batch_ids`, `micro_batch_ids`, `loss_scaling`, `is_computing_ref`, `is_computing_old_ahead` are packed into `user_mini_batch` on the trainer side (`grpo_trainer.py:1420-1491`) because the PP schedule only passes tensors. The swizzled forward recomputes logprobs from raw logits, stores ref/old logprobs into trainer-held lists keyed by `[mini_batch_id][micro_batch_id]` and returns `None` in non-train phases (`:435-458`); in train phase it calls the same `compute_loss` and returns `loss.unsqueeze(0) * loss_scaling` (`:510-544`), which a trivial `pp_loss_fn = loss.mean()` consumes (`grpo_trainer.py:2134-2145`). The PP-warmup quirk — "the first micro-batch could get processed multiple times" — is handled explicitly (`grpo_trainer.py:489-496`). Middle stages call `pp_scheduler.step(position_ids=...)` only (`grpo_trainer.py:1513-1517`).

### 10.5 Parallelism mechanics: per-family `parallelize()` (TP/CP/FSDP2/PP/EP)

There is no Megatron-LM engine anywhere in the training path — parallelism is **torchtitan-style**: DTensor TP plans + FSDP2 `fully_shard` + `PipelineStage`/`Schedule1F1B` (patched local copies, `cosmos_rl/patch`). The only Megatron content is borrowed MoE kernels (`cosmos_rl/policy/kernel/megatron_moe/`, e.g. a runtime import of `megatron.core.transformer.moe.moe_utils.pad_routing_map` at `token_dispatcher.py:284`).

Each model family ships its own `parallelize.py`; the canonical one is `cosmos_rl/policy/model/gpt/parallelize.py:46-213`, applied in fixed order — PP split → TP → CP → compile → FSDP/HSDP/DDP:

- **TP** (`apply_tp`, `gpt/parallelize.py:233-338`): classic colwise q/k/v + rowwise o_proj/down_proj with `SequenceParallel` norms, `RowwiseParallel` embedding, `ColwiseParallel` lm_head with `Shard(1)` input; per-block `parallelize_module` with optional Float8 TP classes and async-TP via `torch._inductor.config._micro_pipeline_tp` + symmetric memory (`:329-333`).
- **CP** is **Ulysses** (all-to-all head/sequence exchange), not ring attention: every block's `attn_func` is wrapped by `ulysses_attn_func(original_attn_func, cp_mesh)` (`gpt/parallelize.py:216-230`), with input slicing done in the trainer via `slice_inputs_for_ulysses` before forward and restored after (`grpo_trainer.py:1348-1368, 1539-1550`). Because TP/CP shard the sequence dim, sequence lengths are rounded up to `seq_len_multiple = cp * tp` (`llm_trainer.py:260`, applied at `grpo_trainer.py:1193-1197`), further rounded to 16 under FP8/FP4 due to `torch._scaled_mm`'s K%16 constraint (`llm_trainer.py:263-270`).
- **FSDP2** (`apply_fsdp`, `gpt/parallelize.py:366-423`): per-transformer-block `fully_shard` over mesh `("dp_replicate", "dp_shard_cp")` for HSDP or `("dp_shard_cp",)` for plain FSDP (`:82-95`) — note CP ranks participate in parameter sharding. `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` by config (`config/__init__.py:843-862`). The `reshard_after_forward` policy is explicit:

```python
if pp_enabled:
    # For PP, do not reshard after forward to avoid per-microbatch all-gathers
    reshard_after_forward = False
else:
    # do not reshard the last block since FSDP would prefetch it immediately
    reshard_after_forward = int(layer_id) < len(model.layers) - 1
```
(`gpt/parallelize.py:402-410`); embeddings always reshard "to keep embedding sharded during other calculations to save memory" (`:420-422`).
- **PP** builds one `PipelineStage` per rank and a `Schedule1F1B` with `n_microbatches = mini_batch // pp_micro_batch_size` for GRPO (`gpt/parallelize.py:162-203`), enforcing `n_microbatches % pp == 0` again per step (`grpo_trainer.py:1055-1061`). The model splits itself (`model.apply_pipeline_split(pp_rank, pp_size)`, `:445-454`).
- **MoE/EP** (`qwen3_moe/parallelize.py`): experts are DTensor-sharded on dim 0 (the expert dim) over the EP mesh by a custom `_ExpertParallel` `ParallelStyle` (`qwen3_moe/parallelize.py:191-213`), and FSDP uses **two meshes** — `fsdp_mesh_moe` wraps `transformer_block.mlp.experts` separately from the dense `fsdp_mesh_no_moe` wrap of the block (`qwen3_moe/parallelize.py:278-346`), matching `ParallelDims.dp_shard_with_ep` ("we can have dp_shard=64 for attention and ep=4, dp_shard_with_ep=16 for MoE", `utils/parallelism.py:97-102`).

**Materialization protocol** (`llm_trainer.py:89-156`): the model is built on `meta` device, parallelized, then `_apply`-materialized to GPU — or to **CPU when `fsdp_offload`** (`:104-112, 123-136`); under PP "model_parts are deep copies with FSDP applied. Only materialize them — the original model still holds all layers unsharded and must NOT be materialized" (comment at `:125-127`). Weights are then populated by `load_hf_weights` or checkpoint (§10.8). Default master dtype is fp32 — params live in `master_dtype` outside forward/backward, bf16 inside, with FP8/FP4 conversion applied while still on meta (`llm_trainer.py:92-100`; `policy/model/base.py:621-624`).

### 10.6 Train↔rollout layout resharding: a slice-algebra plan, not a model reshape

Part I §4.4 covered the orchestration (shard-info posts, `ParallelizedShardMapper`, instruction fetch); here is the math. Critically, **the training model is never resharded** for weight sync. The controller computes, once, a per-rank instruction plan describing which *sub-slices* of each parameter each (policy_rank → rollout_rank) pair must move, and workers just execute sliced NCCL sends/recvs.

The unit is `DimSliceInfo {offset, total_size, length}` — a rational interval of a tensor dimension (`cosmos_rl/utils/dim_slice_info.py:29-74`), with `simplify()` dividing out the gcd so `[2/4, len 1/4]` ≡ `[2/4..3/4]` in lowest terms.

**Policy side** — shard info is read straight off DTensor metadata: for each named parameter, `extract_infomation_from_dtensor` walks `param.placements` against `mesh.mesh_dim_names` and uses `param.__create_chunk_list__()` offsets/sizes to produce per-dim `DimSliceInfo` (`dim_slice_info.py:242-300`; caller `parallelism_info_for_dtensor_params`, `utils/parallelism_map.py:461-539`). Fused parameters that the rollout engine wants split (e.g. packed qkv) are decomposed via `weight_mapper.policy_decompose_param_1_to_n_for_sync` and intersected with the local shard (`parallelism_map.py:487-531`).

**Rollout side** — vLLM doesn't use DTensor, so sharding is inferred by **module-type inspection**: `determine_tp_dim` switches on `QKVParallelLinear` / `MergedColumnParallelLinear` / `RowParallelLinear` / `ColumnParallelLinear` / `VocabParallelEmbedding` (plus `FusedMoE` and model-specific special cases for gpt-oss/qwen3.5, and TRT-LLM's `TensorParallelMode.COLUMN/ROW`) to recover which dim is TP-sharded (`parallelism_map.py:590-713`), including the `num_kv_head_replicas > 1` case where KV heads are replicated across TP ranks (`:620-623, 540+`).

**Plan generation** — for every parameter and every (p_rank, r_rank) pair, intersect the two slice sets dim-by-dim:

```python
all_dims = shard_info.keys() | r_shard_info.keys()
for d in all_dims:
    p_tensor_split_strategy, r_tensor_split_strategy = (
        tensor_overlap_info_at_dim(shard_info, r_shard_info, d)
    )
```
(`parallelism_map.py:1152-1166`); `tensor_overlap_info_at_dim` unifies denominators, computes the interval overlap, and re-expresses it relative to each side (`dim_slice_info.py:211-240`). Empty overlap ⇒ no instruction. Replicated copies on either side (FSDP-replicate dims, KV replicas) are deduplicated by `policy_to_rollout_assign` over "dup rank" lists so each byte is sent exactly once (`parallelism_map.py:1168-1180, 267+`). The output is msgpack-packed `WeightSyncInstructionsGroup` lists per rank (`parallelism_map.py:35-100, 1252-1258`), executed by `rl_worker.execute_policy_to_rollout_unicast` slicing local views (`dim_slice_info.py:77-100`; sender at `rl_worker.py:349-497`, Part I §4.4). The sender's tensors come from `model.weight_sync_transforms` cached as `map_w_from_policy_to_rollout` (`grpo_trainer.py:2006-2020`).

**Intra-policy elasticity resharding** — when a new policy replica joins, it receives *everything* (model, optimizer, LR scheduler, RNG state, and the reference state dict) via `sync_all_states`, which iterates sorted state-dict keys, wraps each (possibly CPU/offloaded, possibly DTensor-local) tensor into a CUDA view, and streams it through the HA-NCCL broadcast/send hooks (`llm_trainer.py:315-443`; handlers `execute_policy_to_policy_broadcast/unicast`, `rl_worker.py:292-348`). This works only because all replicas share an identical parallel layout — same-key, same-local-shape transfers, no slicing needed.

### 10.7 Optimizer, LR schedule, grad clipping, gradient accumulation

**OptimizersContainer: one `torch.optim` instance per parameter.** `build_optimizers` (`cosmos_rl/policy/trainer/optm/__init__.py:300-419`) supports `Adam/AdamW` plus torchao `Adam8bit/AdamW8bit` (`:37-41, 381-386`), with per-model-part LRs — `optm_lr` may be a float, list, or `{module_path: lr}` dict; the dict form resolves dotted module paths against `named_modules()` with a `global` fallback bucket (`llm_trainer.py:176-237`). The container's structure is unusual:

```python
if optimizer_kwargs_copy.get("fused", False):
    # Group the parameters by device mesh to do optimizer fusion.
    ...
    for params in parameters_by_mesh.values():
        optimizer = optimizer_cls(params, **optimizer_kwargs_copy)
else:
    for name, p in model.named_parameters():
        if p.requires_grad:
            optimizer = optimizer_cls([p], **optimizer_kwargs_copy)
```
(`optm/__init__.py:139-186`) — non-fused mode creates **one optimizer object per parameter**; fused mode groups by `p.device_mesh` (required because `fused=True` can't mix DTensors from different meshes). `step()/zero_grad()` chain over all of them (`:200-207`); `state_dict()` uses `torch.distributed.checkpoint.get_optimizer_state_dict(..., flatten_optimizer_state_dict=True)` namespaced as `idx-{model_part}-{key}` (`:209-233`), which is what makes per-rank checkpointing and `sync_all_states` work uniformly. Under FP8/FP4 a `register_step_post_hook` re-quantizes after each step (`llm_trainer.py:308-313`).

**LR schedule rebuilt at the first step.** All schedulers are `LambdaLR` over a single `warmup_stable_decay` lambda: linear warmup from `warmup_start_factor`, a stable plateau, then `linear|sqrt|cosine|none` decay floored at `min_lr_factor`, with fractional `optm_warmup_steps <= 1.0` interpreted as a ratio (`optm/__init__.py:500-620`, lambda body `:550-606`). Because GRPO's true `total_steps` is only known once the controller observes the dataset, the trainer builds an initial scheduler with `total_steps=1e6` (`grpo_trainer.py:2073-2074`) and **rebuilds it on the first `DataFetchCommand`**, carrying over `state_dict` so resume offsets survive (`update_lr_schedulers`, `grpo_trainer.py:2076-2095` — "only until the first step, we can know the exact total steps from the controller"). `lr_schedulers.step()` fires once per training step, after all mini-batches (`grpo_trainer.py:1879-1880`).

**Gradient accumulation and the optimizer boundary.** Accumulation is plain summed `.backward()` per mini-batch with `loss_scaling_factor` pre-scaling (§10.3). The optimizer boundary is `all_reduce_states` (`grpo_trainer.py:2097-2132`), normally invoked once per `per_optimize_batch` slice (`grpo_trainer.py:1803-1811`), or every `COSMOS_GRPO_STEP_INTERVAL` mini-steps if that env var is set (`:1788-1801`). It does three things in order on `train_stream`:

1. **Cross-replica gradient averaging** over the elastic HA-NCCL comm (intra-replica FSDP reduction already happened inside backward): `gradient_reduce_across_dp_replicas_` extracts `.grad` (DTensor → `to_local()`), buckets by dtype into ≤200 MB chunks, concatenates, casts to fp32, and `allreduce(AVG)`s each chunk through the raw comm, copying results back (`cosmos_rl/utils/distributed.py:93-181`; first invocation gets a 30-minute timeout to absorb replica-join skew, `:159-164`).
2. **Global grad-norm clipping**, hand-rolled because `clip_grad_norm_` "only computes gradient norm along DP/FSDP/TP dimensions. We need to manually reduce the gradient norm across PP stages" (`distributed.py:185-216`): params are grouped by their `device_mesh`, per-group `get_total_norm` is computed (DTensor norms resolved with `.full_tensor()`), groups are combined under the p-norm, then all-reduced over the PP mesh, and finally `torch.nn.utils.clip_grads_with_norm_` applies the shared total norm per group (`distributed.py:222-286`). `optm_grad_norm_clip` defaults to 1.0; `<= 0` switches to norm-reporting-only (`config/__init__.py:820-822`; `grpo_trainer.py:2128`). The PP-stage subtlety is handled: ranks with `model_part is None` still pass an empty param list — "GradNorm across pp stages will fail if some rank does not join the barrier" (`grpo_trainer.py:2114-2120`).
3. `self.optimizers.step(); self.optimizers.zero_grad()` (`grpo_trainer.py:2130-2131`).

Two further batch-shaping mechanisms feed this: optional **sequence packing** (cu_seqlens-style varlen packing of the mini-batch, disabled automatically under PP or incompatible models, `grpo_trainer.py:1269-1286, 1314-1336`) and **dynamic mini-batch re-arrangement by token budget** — when `max_token_len_per_mini_batch` is set, `rearrange_mini_batches` bin-packs samples by effective sequence length (scaled by cp_size) instead of fixed `mini_batch` counts, coordinated across replicas via the HA-NCCL comm (`grpo_trainer.py:1116-1145`).

### 10.8 Checkpointing and resume semantics

**What is saved.** Two parallel artifact streams, both triggered from inside `step_training` on the **master replica only** when the controller's `DataFetchCommand` carried `do_checkpoint` (`grpo_trainer.py:1883-1911`):

1. **HF safetensors export** (consumption format) — every checkpoint step if `export_safetensors`, always at the last step: DTensors are `full_tensor()`-gathered, mapped through `weight_mapper.policy_map_local_key_to_hf_key`, chunked into ≤4 GB files written only by the `(dp_shard=0, cp=0, tp=0)` corner ranks (PP stages write disjoint `model-{pp_rank}-of-{pp_size}-*.safetensors`, manifests all-gathered over the PP group), then handed to a daemon thread that saves CPU chunks, writes `model.safetensors.index.json`, copies `*.py/*.json` from the source model for `trust_remote_code`, and optionally uploads to HF hub / S3 (`llm_trainer.py:445-790`). LoRA runs force `trainable_only` and emit `adapter_model.safetensors` + `adapter_config.json` (`:461-468, 750-757`).
2. **Cosmos checkpoint** (resume format) — `CheckpointMananger.save_checkpoint` (sic — typo shipped, like Part I §6's `HighAvailabilitylNccl`) writes, under `output_dir/checkpoints/step_<N>/policy/`, four plain `torch.save` files **per rank**: `model_rank_{r}.pth` (full `state_dict()`, DTensors saved as-is), `optimizer_rank_{r}.pth`, `scheduler_rank_{r}.pth`, and `extra_info_rank_{r}.pth` containing `{rng_state, step, total_steps, remain_samples_num, is_final}` plus a `cosmos_config` dump (`utils/checkpoint.py:434-526`). Because only `dp_replicate_coord==0` ranks save, the expected file count is `world_size // dp_replicate` (`checkpoint.py:115-138`).

**Async mode** (`save_mode == "async"`): state dicts are first offloaded to CPU synchronously (`offload_state_dict_cpu`, `checkpoint.py:405-417`), then four saves run on a 4-worker thread pool, and a fifth future writes a `.rank_{r}_complete` marker only after all four finish (`checkpoint.py:524-600`). The previous step's futures are joined before a new save starts, and `finalize()` drains them at exit (`:418-432`).

**Retention**: `save_check` keeps `max_keep` step dirs FIFO, except the dir symlinked as `best/` — if best is oldest, the second-oldest is deleted instead; best is chosen by `val_score` (direction inferred from whether the metric name contains "loss") and persisted across restarts in a score file + `best/checkpoints`, `best/safetensors` symlinks (`checkpoint.py:759-831`).

**Resume** is controller-initiated: the first `WeightResumeCommand` (Part I §4.4) lands on one policy replica, whose handler calls `trainer.weight_resume()` and then **posts the loaded `ckpt_extra_info` back to the controller** for cross-validation against the data fetcher (`rl_worker.py:499-509`; endpoint `run_web_panel.py:340-345` → `data_fetcher.validate_after_resume`). The load order inside `weight_resume` (`grpo_trainer.py:2026-2071`):

1. If `kl_beta != 0`, *always* load HF weights first and snapshot them to the CPU `reference_state_dict` — the reference policy is the pre-training model, not the resumed one (`:2029-2036`).
2. If `train.resume` is set, load the cosmos checkpoint over it; `FileNotFoundError`/corruption falls back to HF with a log, not a crash (`:2038-2057`).
3. Otherwise plain HF load.

`model_resume_from_checkpoint` → `CheckpointMananger.load_checkpoint` resolves candidates — `resume: str` is taken verbatim; `resume: true` globs every `<root>/<timestamp>/checkpoints/step_*` across **all previous run timestamps**, sorted by step descending (`checkpoint.py:176-219`) — validates completeness via `cosmos_config` + all `.rank_*_complete` markers (`checkpoint.py:139-174`; corrupted dirs are pruned at startup, `:100-113`), then per-rank loads scheduler (rebuilt first with the checkpointed `total_steps` via the `Callable` path), model (`strict=False` per PP stage with module-path prefix stripping), optimizer, and RNG state, finishing with `gc.collect()` + `torch.cuda.empty_cache()` "to avoid fragmentation-induced OOM during the first training step after resume" (`checkpoint.py:618-723`; PP kwargs from `llm_trainer.py:816-830`). Because files are rank-indexed raw `torch.save` blobs, **resume requires the identical parallel layout** — there is no DCP-style resharding-on-load.

The controller, independently, seeds its `current_step` from the same extra info via its own data-fetcher read: `current_step=self.data_fetcher.ckpt_extra_info.get("step", 0)` and `remain_samples_num` (`dispatcher/controller.py:170-194`), so step numbering, LR schedule, and dataset position all restart consistently; non-master replicas are then populated by `PolicyToPolicyBroadcastCommand` → `sync_all_states` (§10.6) rather than by re-reading files.

### 10.9 Training-side GPU memory management

Part I §3.7 covered the rollout-side levers (prefix cache off, recv staging queues). The training-side levers, smallest to largest:

1. **Logits-row elimination (`interested_tokens`)** — the biggest RL-specific saver. When the replica is pure-DP (`dp_shard_coord[1] == world_size`, i.e. no TP/CP sharding of the sequence), the trainer passes `logprob_masks` as `interested_tokens` (`grpo_trainer.py:1288-1297`, with the comment "The interested_tokens will be unevenly distributed across ranks. So do not enable interested_tokens in TP."). The model then slices hidden states **before** the lm_head: `h = h[interested_tokens]; h = self.norm(h); output = self.lm_head(h)` (`gpt/__init__.py:503-509`) — the `[B, T, vocab]` logits tensor is never materialized for prompt/padding tokens; `compute_logprobs` handles both layouts via `is_full_logits` (`utils/util.py:955-962`).
2. **Activation checkpointing** — per-layer flag set after parallelization (`model.set_gradient_checkpointing_enabled(config.policy.model_gradient_checkpointing)`, `llm_trainer.py:151-154`); the forward routes flagged blocks through `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` (`gpt/__init__.py:486-499`). A selective-op `_save_list` (mm, SDPA, max) exists for SAC (`gpt/parallelize.py:341-350`).
3. **Activation offloading to CPU** — torchtune's `OffloadActivations` saved-tensors-hooks manager (attributed in-source, `utils/activation_offloading.py:28-30`): forward activations > 1 KB are moved to (pinned) CPU memory and brought back in backward, guarded by a 60% host-RAM watermark (`:32-120`). The lm_head is wrapped in a `NoOpManager` exemption because round-tripping its activation "is expensive ... we actually use more memory if we offload it" (`:404-418`). It wraps the non-PP forward only (`with self.act_offloading_ctx_manager:` at `grpo_trainer.py:1533-1536`) and is constructed with `use_streams=False` — "FIXME: (lms) use_streams=True could cause NaN in backward" (`llm_trainer.py:253-256`).
4. **FSDP CPU offload** (`fsdp_offload`) — `CPUOffloadPolicy()` in the `fully_shard` config (`gpt/parallelize.py:394-395`) plus CPU materialization of the meta model (`llm_trainer.py:104-112, 123`); explicitly asserted incompatible with PP (`config/__init__.py:1882-1885`). `sync_all_states` copes with offloaded tensors by staging through CUDA views and copying back (`llm_trainer.py:362-377`).
5. **Reshard-after-forward tuning** — per-block policy described in §10.5, exposed as `fsdp_reshard_after_forward = always|never|default` (`config/__init__.py:868-872`).
6. **Precision splits as memory policy** — params bf16 in compute / fp32 master, fp32 FSDP reductions, fp32 logprob math only on the masked tokens (`config/__init__.py:837-862`); optional FP8/FP4 weight quantization converters applied on meta (`llm_trainer.py:92-100`).
7. **Step-shaping** — token-budgeted dynamic mini-batches and sequence packing (§10.7) bound activation peaks independently of sample count; the reference model lives on CPU and is *swapped*, never duplicated, on GPU (§10.1); `torch.cuda.empty_cache()` after parallelize/materialize (`llm_trainer.py:156`) and after checkpoint load (`checkpoint.py:698-703`); async checkpointing stages everything to CPU before touching disk (§10.8).

---

## 11. Sample & data lifecycle

Part I §7 traced one GRPO step as *control flow* (which function calls which). This section traces the same step as *data*: which object the sample lives in at every hop, which process owns it, and where each field is written and read.

### 11.1 The life of one sample — numbered hop chain

**Hop 1 — dataset row → `RLPayload` (controller process).** A dataset item is wrapped lazily at `__getitem__` time; the dataset index becomes the sample's permanent identity `prompt_idx`:

```python
# cosmos_rl/dispatcher/data/__init__.py:38-43 (RLDataset.__getitem__)
def __getitem__(self, idx: int) -> IdxAndRLPayload:
    prompt = self.dataset[idx]
    if isinstance(prompt, RLPayload):
        prompt.prompt_idx = idx
        return idx, prompt
    return idx, RLPayload(prompt=prompt, prompt_idx=idx)
```

The controller builds a standard `torch.utils.data.DataLoader` over this with `collate_fn=RLPayload.collate_fn` and a `DistributedSampler(num_replicas=1, rank=0)` (`dispatcher/data/data_fetcher.py:213-220, 334-351`); `RLPayload.collate_fn` just unzips `(idx, payload)` pairs (`dispatcher/data/schema.py:153-164`).

**Hop 2 — prompt admission (HTTP pull).** Part I §3.2 covered the throttles; the data-level detail is the **`weight_version` stamp**. In fully-synchronized mode each payload gets exact per-version accounting:

```python
# cosmos_rl/dispatcher/controller.py:373-399
while (weight_version_for_each_payload in self.weight_version_to_prompt_num
       and self.weight_version_to_prompt_num[weight_version_for_each_payload]
       >= global_batch_size):
    ...
    weight_version_for_each_payload += 1
...
payload.weight_version = weight_version_for_each_payload
```

(DAPO instead stamps every payload of the batch with `weight_version_for_current_batch`, `controller.py:405-420`; non-RL/scaling fallback stamps `0`, `controller.py:463`.) `reference_answer` is attached from the dataset only when needed (`data_fetcher.py:493-512`); in `train.local_dataset` mode, `prompt`/`conversation`/`reference_answer` are **stripped to `None`** before going over the wire so only `prompt_idx` travels (`data_fetcher.py:582-589`).

**Hop 3 — distribution inside the rollout replica.** Rank 0 re-hydrates local-dataset prompts by index (`rollout_control.py:1495-1509`), `broadcast_object_cpu`s `(prompts, is_end)` to all ranks, then **round-robin scatters across DP ranks**: `rank_prompts = prompts[rank::ranks_to_scatter]` (`rollout_control.py:1518-1554`), finally `prompt_queue.put(prompts)` (`rollout_control.py:1556-1557`).

**Hop 4 — generation and token/logprob capture.** `one_step_generation` pops a whole batch — `payloads_list: List[RLPayload] = self._prompt_queue.get()` (`rollout_control.py:1928`) — and calls `rollout_generation` (`rollout/vllm_rollout/vllm_rollout.py:948-970`, single-turn at `:593-730`). Sampling params are built once: `n=self.config.rollout.n_generation`, `logprobs=0` (or `distillation.top_k`), `prompt_logprobs=0` when `collect_rollout_logprobs` is on (`vllm_rollout.py:224-247`). Each prompt → one `RolloutResult` holding `n_generation` completions:

```python
# cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:702-722
response.append(RolloutResult(
    prompt=payloads[i].prompt,
    completions=[output.outputs[j].text for output in outputs for j in ...],
    completion_logprobs=logprobs,
    completion_token_ids=token_ids,
    cumulative_logprob=[output.outputs[j].cumulative_logprob ...],
    prompt_logprobs=prompt_logprobs,
    prompt_token_ids=prompt_token_ids,
))
```

`parse_logprobs` (`vllm_rollout.py:430-474`) converts vLLM's `{token_id: Logprob}` dicts into per-position lists where **index 0 is always the actually-sampled token's logprob** (top-k alternatives follow only in distillation mode). Logprobs are dropped (`logprob = []`) unless `collect_rollout_logprobs` is set (`vllm_rollout.py:585-586`). Any exception inside generation returns `[]` for the whole batch (`vllm_rollout.py:724-729`).

**Hop 5 — validity filtering & result-to-payload merge.** `_filter_valid_rollout_results_and_report` (`rollout_control.py:1825-1919`): empty completions are **not dropped but replaced with the EOS token** so fully-synchronized accounting stays exact —

```python
# cosmos_rl/rollout/worker/rollout_control.py:1874-1880
# We still need to upload the output text, even if it is empty. (replace empty with eos_token)
# Because if fully synchronized mode is enabled, we need to make sure the expected
# number of global_batch_size is reached at exact time.
output_texts.append(output_text if output_text != "" else self.eos_token)
```

(only an all-empty group is skipped, `skip_output = (total_generation_count - empty_generation_count) <= 0`, `:1883`; multi-turn drops conversations with no non-empty assistant message, `:1840-1855`). Then, **only on `should_report` ranks** (tp_coord 0 + last PP stage, `rollout_control.py:250-252`), the `RolloutResult` fields are copied back into the original `RLPayload` plus `old_payload.weight_version = self.current_weight_version` (re-stamped with the *actual* generating version, overwriting the controller's prediction, `:1893-1902`), local-dataset `reference_answer` is re-queried (`:1905-1910`), and the payload enters the reward pipeline: `self.reward_dispatcher.enqueue_rewards_cal(valid_payloads, False, self.current_weight_version, bypass_reward=...)` (`:1913-1918`).

**Hop 6 — reward + advantage (§11.3).** Output: the same `RLPayload`s now with `rewards`, `advantages`, `filter_rewards`, `n_ignore_prefix_tokens`, `valid`, `report_metrics` populated.

**Hop 7 — report to controller.** `report_rollouts` (`rollout_control.py:1670-1713`), called at the top of every main-loop iteration, drains finished reward tasks; for DAPO it first runs `dynamic_sampling` which **drops `valid=False` payloads on the worker** and counts them in `metadata` (`rollout_control.py:1648-1668`); `data_packer.get_rollout_output` serializes non-text completions; with `rollout_as_token_ids` the completion **strings are blanked** (`payloads[i].completions = [""] * ...`, `:1701-1702`) since token ids suffice. Then `POST /rollout` with `RolloutRequest(src_replica_name, payloads, metrics, is_end=False)` (`:1704-1710`).

**Hop 8 — controller unpacks group → per-completion `Rollout`.** `put_rollout_group` (Part I §7 step 11) calls `extract_rollouts` (`utils/payload.py:40-125`), which zips the per-completion parallel lists of one `RLPayload` into N `Rollout` objects (one per completion) and **distributes `extra_info` per index when a value's length matches the group size** (`payload.py:104-119`). The flattened list goes through the staleness filter (Part I §3.3) into `rollout_buffer`; note `put_rollout` also appends a missing EOS to `completion` when `include_stop_str_in_output` is set (`status.py:749-759`).

**Hop 9 — dispatch to policy replicas.** Round-robin Redis publish as in Part I §3.3; the stream is capped at `maxlen=RedisStreamConstant.STREAM_MAXLEN` = 10000 entries (`utils/redis_stream.py:199-216`, `utils/constant.py:161`).

**Hop 10 — policy worker ingest.** Rank-0's `fetch_rollouts` thread: `Rollout.model_validate(msgpack.unpackb(x)) ... self.data_queue.put_nowait(rollout)` (`policy/worker/rl_worker.py:256-272`). On `DataFetchCommand`, `dispatch_rollouts` blocks for `items_count` rollouts, assigns them to DP ranks round-robin, and `dist.scatter_object_list`s them (`rl_worker.py:677-767`); in local-dataset mode it **re-populates `prompt`/`conversation` from the local dataset by `prompt_idx`** (`rl_worker.py:701-708`).

**Hop 11 — training consumption.** `GRPOTrainer.step_training` (`grpo_trainer.py:940-1027`) reads: `rollout.prompt` (or `completed_conversation`), `rollout.completion` (or column 0 of `completion_token_ids` when `rollout_as_token_ids`), `rollout.n_ignore_prefix_tokens`, `rollout.advantage` (→ `advantages_t` tensor, `:1026-1027`), `rollout.reward` (only for positive-NLL flags, `:995-1004`). Tokenization happens here via `data_packer.get_policy_input`:

```python
# cosmos_rl/dispatcher/data/packer/decoder_only_llm_data_packer.py:92-96
return DecoderOnlyLLMDataPacker.RLPolicyInput(
    input_ids=input_ids + completion_ids,
    logprob_masks=[0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
    + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
    + [0],
)
```

i.e. the loss mask covers exactly the completion tokens minus the shared-prefix ignore count, shifted by one for next-token prediction. With `use_decoupled_loss`, `rollout.completion_logprobs[...][0]` (the sampled-token logprobs from vLLM) are injected as `user_mini_batch["rollout_logprobs"]` (`grpo_trainer.py:1243-1268`); with `use_rollout_logprobs_for_loss` they become `old_per_token_logprobs` directly in `compute_loss` (`grpo_trainer.py:475-486`). Config invariants enforce that both modes force `collect_rollout_logprobs=True` (`policy/config/__init__.py:680-692`).

**Hop 12 — discard.** The `rollouts` list is a local of `execute_data_fetch`/`step_training`; after `post_policy_train_ack` (`rl_worker.py:558-565`) nothing retains them — there is no replay buffer. Persistence is only progress metadata: the trainer checkpoints `{"remain_samples_num": remain_samples_num}` (`grpo_trainer.py:1900-1910`). Redis stream entries age out via `maxlen=10000`; the controller's `rollout_buffer` entries were already removed by `.get()` at dispatch.

### 11.2 The sample data structures, field by field

Three schemas carry the sample (all pydantic):

**`RLPayload`** (`dispatcher/data/schema.py:49-168`) — the *group-level* record (1 prompt × n_generation completions). Field map:

| Field | Producer | Consumer |
|---|---|---|
| `prompt`, `conversation` | `RLDataset.__getitem__` (`data/__init__.py:38-43`); nulled for local_dataset (`data_fetcher.py:582-589`), re-hydrated at `rollout_control.py:1495-1509` | `data_packer.get_rollout_input` (`vllm_rollout.py:627`); copied into `Rollout` |
| `prompt_idx` | dataset index (`data/__init__.py:41-43`) | local-dataset re-query (both workers), `data_dispatch_as_rank_in_mesh` modulo routing (`data_fetcher.py:594-604`, `status.py:760-763`) |
| `reference_answer` | controller `_next_payload` (`data_fetcher.py:502-511`) or rollout-side re-query (`rollout_control.py:1905-1910`) | `RolloutGroup.compute_rollouts` → reward fns (`reward/base.py:59-63`) |
| `weight_version` | controller stamping (`controller.py:392-420`), **overwritten** with actual generating version at `rollout_control.py:1900` | rollout-side freshness gate (`rollout_control.py:1807-1815`), controller staleness filter (`status.py:823-838`) |
| `completions` | `RolloutResult.completions` copy (`rollout_control.py:1895`) | reward fns; tokenized by `get_policy_input` |
| `completion_token_ids` / `completion_logprobs` (`List[List[List[…]]]`: completion × position × top-k, index 0 = sampled token) | `parse_logprobs` (`vllm_rollout.py:430-474`) via `:1896-1897` | trainer decoupled-loss path (`grpo_trainer.py:1243-1268`), `rollout_as_token_ids` training input (`grpo_trainer.py:987-992`) |
| `prompt_logprobs` / `prompt_token_ids` | `get_prompt_logprobs_and_token_ids` (`vllm_rollout.py:476+`) | distillation/teacher paths |
| `cumulative_logprob` | vLLM per-output (`vllm_rollout.py:715-719`) | "most likely mode reward" metric (`reward/base.py:76-97`) |
| `rewards`, `advantages`, `filter_rewards`, `valid`, `n_ignore_prefix_tokens`, `report_metrics` | `LocalRewardCalculator.compute_rewards` (`reward/local_calculator.py:181-359`) | `dynamic_sampling` (valid), `extract_rollouts`, trainer |
| `extra_info` | rollout engine (`rollout_control.py:1902`) | per-index split in `extract_rollouts` (`payload.py:104-119`); remote reward modality (`remote_calculator.py:198`) |
| `teacher_result_uuids` | `enqueue_teacher_calculation` (`rollout_control.py:1912, 2081+`) | policy teacher prefetch (`rl_worker.py:271-272, 742-748`) |

**`Rollout`** (`dispatcher/data/schema.py:171-247`) — the *per-completion* training record, created **only on the controller** by `extract_rollouts` (`payload.py:67-99`): scalar `completion`, `reward`, `advantage`, `filter_reward`, `n_ignore_prefix_tokens`, `weight_version`, plus 2-D `completion_token_ids`/`completion_logprobs` (position × top-k). This is the unit that flows controller → Redis → policy `data_queue` → DP scatter → trainer.

**`RolloutResult`** (`rollout/schema.py:21-55`) — engine-internal output of `rollout_generation`, never serialized; merged back into `RLPayload` at `rollout_control.py:1893-1902`.

**`RLPolicyInput`** (`decoder_only_llm_data_packer.py:18-24`) — `{input_ids, logprob_masks}` per sample; collated to padded `input_ids` (pad_token) + boolean `logprob_masks` tensors (`:122-149`).

### 11.3 Reward computation plumbing

**Where rewards run: on the rollout worker, not the controller** (Part I §7 step 9 noted this; here is the machinery). Each rollout worker owns a `RewardDispatcher(payload_per_task=COSMOS_REWARD_DISPATCHER_PAYLOAD_PER_TASK)` (default 64 payloads/task; `rollout_control.py:191-193`, `utils/constant.py:78-80`), set up with `num_workers=COSMOS_REWARD_DISPATCHER_CONCURRENCY` (default 2) **only on the reporting rank**, `0` elsewhere (`rollout_control.py:254-264`).

**Local path** — async pool. `enqueue_rewards_cal` chunks payloads by 64 and submits `compute_rewards` futures to a `ProcessPoolExecutor` (or `ThreadPoolExecutor` in non-text mode, because threads can share tensors/videos without pickling: `reward/dispatcher.py:93-111`); each pool worker holds a singleton `LocalRewardCalculator` initialized via the executor `initializer` (`dispatcher.py:64-111`). `dequeue_rewards_cal` is non-blocking and **strictly FIFO**: it only pops the head if `task_queue.queue[0].done()` (`dispatcher.py:244-256`), so a slow reward task head-of-line-blocks reporting but generation continues. The actual math: per group, `Reward.compute_reward` sums `weight × fn(completion, reference)` over the configured fns into `total_rewards` and `filter × val` into `filter_rewards` (`dispatcher/algo/reward.py:451-496`; if no filter fns, `filter_rewards = total_rewards`, `:494-495`), then GRPO advantage:

```python
# cosmos_rl/dispatcher/algo/grpo.py:41-45
mean = np.mean(rewards)
if self.unbiased:
    result = rewards - mean
else:
    result = (rewards - mean) / (np.std(rewards) + self.eps)
```

`LocalRewardCalculator.compute_rewards` then runs the DAPO group test — `if len(set([rollout.filter_reward for rollout in rollouts_group])) > 1` keep `valid=True`, else mark the whole group `valid=False` (`reward/local_calculator.py:238, 287, 333`) — and the shared-prefix de-bias: groups of completions sharing a token prefix ≥ `min_filter_prefix_tokens` *with differing rewards* get `n_ignore_prefix_tokens = len(shared_prefix)` so the ambiguous prefix is masked from the loss (`local_calculator.py:244-267`, consumed in the logprob mask of §11.1 hop 11).

**Failure handling (local):** built-in reward fns swallow their own exceptions and return `0.0` (e.g. `boxed_math_reward_fn` catches `TimeoutException` and bare `Exception`, `algo/reward.py:213-222`; same in `single_choice`/`gsm8k`/`format`). There is no payload-level retry — a crashing custom fn would poison the future and surface at `.result()`.

**Remote path** (`use_remote_reward`) — synchronous enqueue, asynchronous fetch. `enqueue_rewards_cal` batches payloads by **total completion count ≤ `remote_batch_size`** (`dispatcher.py:183-207`), and `compute_rewards` POSTs one `application/octet-stream` request — JSON header + `\n` + `np.save` tensor (video frames are first VAE-encoded to fp16 latents on-GPU, `remote_calculator.py:224-241`) — storing `uuid → {payloads, replica, stage, step, completions_per_payload}` maps (`:262-268`). `get_results` polls uuids strictly in order and **breaks on the first not-ready uuid, retrying it next cycle** (`remote_calculator.py:368-400`); per-fn scores are clamped/weighted/summed (`:288-344`), advantages computed inline with NaN→zeros fallback (`:421-424`). Note the remote path returns a *slimmed* `RLPayload` (no token ids/logprobs/conversations, `:426-434`).

**Bypass** — `bypass_reward=True` zero-fills `rewards/advantages/filter_rewards` and skips the pool entirely (`dispatcher.py:151-181`), used when the policy computes its own objective.

### 11.4 Buffering, filtering, staleness — what gets dropped, retried, carried over

Drop/keep decision points, in pipeline order (the controller-side staleness formula was covered in Part I §3.3; items 3-7 are new):

1. **Pre-generation stall (not a drop).** A fetched prompt waits in `_prompt_queue` until `payload.weight_version <= current_weight_version + allowed_outdated_steps` (`rollout_control.py:1805-1815`; default `allowed_outdated_steps=4`, `policy/config/__init__.py:566-571`).
2. **Empty completions → padded with EOS, not dropped** (`rollout_control.py:1874-1887`) — keeps global-batch accounting exact; only an all-empty group is skipped.
3. **DAPO dynamic sampling → dropped at the worker.** `valid=False` groups (uniform filter-reward) never reach the controller; counts travel as `metrics` and the controller decrements bookkeeping: `self.remain_samples_num -= filter_records.get("filtered_positive"/"filtered_negative", 0)` (`status.py:799-810`). The replacement mechanism is implicit: `rollouts_enough_for_one_step()` stays false, so the rollout workers keep pulling *new* prompts (fresh dataset items, not re-rolls of the same prompt) until the buffer fills; `max_retry_for_on_policy` bounds how many `global_batch_size` multiples may be fetched per weight version before `RuntimeError` (`controller.py:422-434`).
4. **Staleness filter → dropped at the controller**, position-aware (formula in Part I §3.3). Discards decrement `remain_samples_num` and are tallied under `filter_records["outdated"]` (`status.py:844-849`); NCCL-payload rollouts additionally get explicit GPU-buffer cleanup messages (`status.py:851-853, 857-869`).
5. **On-policy overflow → dropped silently**: once `on_policy_rollout_completed`, late arrivals only decrement `remain_samples_num` (`status.py:774-781`).
6. **Carried over between steps:** anything left in `rollout_buffer` after a dispatch simply waits — dispatch takes exactly `items_count` per replica (`status.py:1430-1439`) and `weight_version` stamping already accounted for queue depth, so leftovers train at their predicted step. `samples_on_the_fly` is incremented at prompt fetch (`controller.py:464-467`) and decremented only at train-ack (`status.py:1063-1067`), making it the in-flight gauge both throttles read.
7. **No retry anywhere for a failed generation** — `rollout_generation` exceptions yield `[]` and `one_step_generation` returns `False` (`rollout_control.py:1938-1939`); the prompts were already popped from `_prompt_queue` and are lost to the epoch (accounting catches up via the end-of-data `recompute_total_steps` path, `run_web_panel.py:442-483`).

### 11.5 Dataset/prompt-source handling: epochs, shuffling, resume

**Single logical dataloader, owned by the controller.** Sampler is `DistributedSampler(num_replicas=1, rank=0, shuffle=dataloader_shuffle, seed=dataloader_seed)` (`data_fetcher.py:213-220`; defaults `shuffle=True`, `seed=0`, `policy/config/__init__.py:387-393`) — i.e. the controller is the only "rank", and shard-style distribution happens downstream by HTTP pull (or by `prompt_idx % mesh_size` in `data_dispatch_as_rank_in_mesh` mode, `data_fetcher.py:514-547,590-617`, with a `fetched_data_buffer` parking lot for indices that belong to other ranks).

**Epoch rollover lives inside `get_batched_prompt`:** on `StopIteration`, `epoch += 1`, `set_epoch(self.epoch)` reshuffles deterministically, and a fresh iterator continues filling the same request batch; past `config.train.epoch` it flips `is_end=True` (`data_fetcher.py:549-578`). Total budget is precomputed as `len(train_set) × n_generation × epoch` into `remain_samples_num` (`data_fetcher.py:190-198`).

**Resume is sample-count arithmetic, not sampler state.** The trainer checkpoints only `remain_samples_num` (`grpo_trainer.py:1900-1910`); on restart the controller derives epoch and offset from it and skips forward:

```python
# cosmos_rl/dispatcher/data/data_fetcher.py:247-262, 267-281, 291-301
remain_samples_num = self.ckpt_extra_info.get("remain_samples_num", remain_samples_num)
self.epoch = (self.config.train.epoch - math.ceil(remain_samples_num /
              (len(self.dataset.train_set) * self.config.rollout.n_generation))) + 1
...
train_dataloader_bias = (max(0, len(...) - (math.ceil(remain_samples_num / n_generation))
                          % len(...))) % len(self.dataset.train_set)
...
self.train_sampler = SkippingSampler(base_sampler=self.train_sampler, skip_samples=...)
```

`SkippingSampler` consumes the first N indices of the (epoch-seeded, hence reproducible) base sampler exactly once (`policy/trainer/sampler.py:20-53`). This is why resume reproduces the same prompt order: shuffle is a pure function of `(seed, epoch)`. Resume also disables the P2R weight-sync check (`PolicyToRolloutUnicastCommand._do_weight_sync_check_flag = False`, `data_fetcher.py:239-242`).

**`local_dataset` mode** turns the controller's dataset into a `TensorDataset(torch.arange(N))` of pure indices (`data_fetcher.py:182-188`), and both rollout and policy workers instantiate a `WorkerDataFetcher` that re-resolves `prompt_idx → prompt/conversation/reference_answer` from their local copy (`data_fetcher.py:663-740`; rollout at `rollout_control.py:1495-1509`, policy at `rl_worker.py:701-708`) — the control plane then moves only indices, rewards, and token ids, never raw media.

**Validation prompts** flow through the same `get_batched_prompt` with `validation_step` selecting a dedicated iterator (`data_fetcher.py:484-487, 636-654`); validation reports take a separate endpoint (`run_web_panel.py:425-431`) and never enter `rollout_buffer`.

---

## 12. Extension surfaces: adding a model / reward / rollout backend

Cosmos-RL has four parallel plugin registries, all the same shape (class-level dict + `register()` decorator + `get_*_cls()` lookup): `ModelRegistry` (`cosmos_rl/policy/model/base.py:546`), `WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY` (`base.py:796`), `TrainerRegistry` (`cosmos_rl/policy/trainer/base.py:192-233`), `RolloutRegistry` (`cosmos_rl/rollout/rollout_base.py:233-274`), plus a data-packer registry (`cosmos_rl/dispatcher/data/packer/base.py:55-88`) and a parallelism-strategy registry (`cosmos_rl/utils/parallelism_registry.py:32-80`). One decorator call populates up to three of them at once.

### 12.1 Adding a new policy model — traced via Qwen3MoE

**Step 1 — implement `BaseModel` and register.** A model family is one directory under `cosmos_rl/policy/model/<family>/` containing `__init__.py` (the nn.Module), `parallelize.py`, and `weight_mapper.py` (qwen3_moe has exactly these three, plus `weight_converter.py`). Registration is a single decorator that binds the model class to its weight mapper:

```python
# cosmos_rl/policy/model/qwen3_moe/__init__.py:383-384
@ModelRegistry.register(Qwen3MoeWeightMapper)
class Qwen3MoE(BaseModel):
```

The decorator writes **three** registries keyed by HF `model_type` strings returned from the class's own `supported_model_types()` (`["qwen3_moe"]` at `qwen3_moe/__init__.py:402-403`):

```python
# cosmos_rl/policy/model/base.py:553-560
model_types = model_cls.supported_model_types()
...
ModelRegistry._MODEL_REGISTRY[model_type] = model_cls
WeightMapper.register_class(model_type, weight_mapper_cls)
if data_packer_cls is not None:
    BaseDataPacker.register(model_type, data_packer_cls)
```

(`pi05` is the example that also passes a packer: `@ModelRegistry.register(Pi05WeightMapper, default_data_packer_cls=PI05DataPacker)`, `model/pi05/__init__.py:336`.)

**Step 2 — discovery is filesystem-based, not import-list-based.** `cosmos_rl/policy/model/__init__.py:42-136` (`_auto_import_models`, executed at module import, line 136) iterates every subdirectory of `policy/model/`, imports it, and pulls in any `BaseModel` subclass via `inspect.getmembers`; `diffusers/*.py` files are scanned for `DiffuserModel` subclasses and `wfm/models/cosmos_policy.py` separately. Dropping a new directory in is sufficient — no central list to edit. Import failures are warnings, not errors (`__init__.py:69-70`), so an optional-dependency model degrades gracefully.

**Step 3 — config detection.** There is no `model_type` field in the TOML. The controller/worker resolves the class from the HF checkpoint itself:

```python
# cosmos_rl/policy/model/base.py:611-619
model_type = hf_config.model_type
is_supported_model_type = model_type in ModelRegistry._MODEL_REGISTRY
if not is_supported_model_type or config.train.force_use_hf:
    ...
    model_type = COSMOS_HF_MODEL_TYPES
model_cls = ModelRegistry._MODEL_REGISTRY[model_type]
```

`COSMOS_HF_MODEL_TYPES = "hfmodel"` (`utils/constant.py:64`) is the registered generic-HF fallback (`model/hf_models/__init__.py:49`); if the native class throws during load, `build_hf_model` retries once with the fallback before giving up (`base.py:715-733`). Diffusers models are routed by `config.policy.is_diffusers` and keyed on the diffusers `_class_name` instead of `model_type` (`base.py:738-748, 786-791`). The rollout worker performs the same lookup independently and falls back the same way (`rollout/worker/rollout_control.py:161-175`).

**Step 4 — the interface contract.** The abstract surface a new model must implement (`base.py:437-529`): `supported_model_types()` (static), `parallelize_fn` (property returning `(fn, self)` — Qwen3MoE lazily imports its own `parallelize.py`, `qwen3_moe/__init__.py:557-561`), `apply_pipeline_split`, `get_position_ids` (declared "due to that Context Parallelism requires the shuffle of both input_ids and position_ids", `base.py:466-482`), `load_hf_weights`, `separate_model_parts`, `from_pretrained`, `get_nparams_and_flops`; plus non-abstract hooks `post_to_empty_hook` (re-init buffers like `inv_freq` after meta→device materialization, `base.py:451-456`) and `step_hook` (`base.py:458-464`). `build_hf_model` constructs under `init_on_device("meta")` by default (`base.py:702, 707-713`) — real weights arrive later (§13.1 step 12).

**Step 5 — the weight mapper is half the work.** The `WeightMapper` (`base.py:794+`) owns name translation and 1→N splitting between training-side and rollout-engine-side parameter layouts. The contract is name consistency: "The final mapped name from this function should be consistent with the name from `policy_map_local_key_to_hf_key` for the same parameter" (`base.py:813-826`, `rollout_prepare_recv`). Qwen3MoE shows what's actually involved — remapping vLLM's fused-expert names and even TRT-LLM's `next_layer_layernorm` aliasing, and splitting fused QKV / gate-up tensors with a backend-dependent ordering swap:

```python
# cosmos_rl/policy/model/qwen3_moe/weight_mapper.py:73-80
gate_proj_weight = weight[:, : dim_1 // 2]
up_proj_weight = weight[:, dim_1 // 2 :]
if self.backend == "trtllm":
    gate_proj_weight, up_proj_weight = up_proj_weight, gate_proj_weight
```

(`self.backend` is set per rollout engine via `setup_rollout_backend`, validated against `_WEIGHT_MAPPER_BACKEND_SUPPORTED = ["vllm", "trtllm"]`, `base.py:795, 1007-1013`; TRT-LLM calls it at `rollout/trtllm_rollout/trtllm_worker.py:145`.) If the rollout engine shards a weight in a non-default way, the model additionally registers a per-model resharding function: `@register_parallelism_strategy("deepseek_v3", role=ParallelismStrategyRole.ROLLOUT)` (`model/deepseek_v3/weight_mapper.py:408`), surfaced through `WeightMapper.get_policy/rollout_parallelism_strategy` (default `[]`, `base.py:896-900`) and consumed by `ParallelTopoMapperGroup` when computing P2R send/recv instructions (§10.6).

### 12.2 Adding a new reward

Two tiers exist:

**Tier 1 — built-in string-keyed rewards.** `REWARD_FUNC_MAPPING` maps `RewardFn` enum values to module-level functions of signature `(to_be_evaluated: str, reference: str|None, **kwargs) -> float` (`cosmos_rl/dispatcher/algo/reward.py:286-293`; the enum at `utils/constant.py:102-115`). The TOML selects them as `reward_function = {"boxed_math" = 1.0, "format" = 0.5}`; the config validator normalizes a bare string or list into the weighted-dict form (`policy/config/__init__.py:423, 666-669`). This table is hard-coded — there is **no registry** for string-named rewards; new built-ins require editing `REWARD_FUNC_MAPPING`.

**Tier 2 — the intended extension path: pass callables through the custom entry script.** Every launch entry (`cosmos_rl/launcher/worker_entry.py:10-27`) accepts `reward_fns: Optional[List[Callable]]`, `filter_reward_fns`, `val_reward_fns` and forwards them to whichever role the process is (`worker_entry.py:38-94`). The shipped GSM8K example does exactly this:

```python
# cosmos_rl/tools/dataset/gsm8k_grpo.py:375-383
launch_worker(
    dataset=get_dataset,
    val_dataset=get_val_dataset,
    # Override the reward functions defined in toml
    reward_fns=[custom_reward_fn],
    data_packer=GSM8kDataPacker(tool_agent=tool_agent),
    ...
)
```

Inside the `Reward` aggregator, explicit callables **override** the TOML: "Using provided reward functions ... `config.train.train_policy.reward_function` will be ignored" (`dispatcher/algo/reward.py:347-355`). Each entry may be a bare callable or `(callable, weight)` tuple; filter rewards (for DAPO-style dynamic sampling) can reference a training reward by index (`reward.py:313-345`). A reward may also return `(value, components_dict)` for per-component metric logging (`reward.py:410-421, 439-447`), and `group_reward_calculation` switches to batched invocation where the fn receives the whole completion list (`reward.py:396-427`).

Because rewards execute in `ProcessPoolExecutor` workers on the rollout replica (§11.3), a custom local reward fn must be **picklable** (it is shipped to pool worker processes via the executor `initializer`, `reward/dispatcher.py:100-111`).

### 12.3 Adding a new rollout backend (and trainer)

**Contract.** Subclass `RolloutBase` (`rollout/rollout_base.py:30`); `__init__` stores config/dims/device then immediately invokes your `post_init_hook(**kwargs)` (`rollout_base.py:45-50`). Abstract: `post_init_hook`, `rollout_generation(payloads, stream, data_packer, data_fetcher, is_validation) -> List[RolloutResult]`, `init_engine(quantization, seed, load_format)`, `get_underlying_model()` (`rollout_base.py:52-103`). Optional hooks mark the rest of the surface (Part I §6 noted the `*_hook` naming): `post_init_engine_hook(consume_command_hook, report_rollouts_hook, validation_flag)` — this is how vLLM's engine step gets the mid-generation command checkpoint injected — and the quantization-aware `pre/post_get_params_for_sync_hook` pair around weight-sync view construction (`rollout_base.py:115-184`).

**Register + select.** One decorator and one config key:

```python
# cosmos_rl/rollout/example_custom_rollout/example_custom_rollout.py:44-45
@RolloutRegistry.register("example_hf")
class ExampleHFRollout(RolloutBase):
```
```python
# cosmos_rl/rollout/worker/rollout_control.py:130-132
self.rollout: RolloutBase = RolloutRegistry.get_rollout_cls(
    self.config.rollout.backend
)(self.config, self.parallel_dims, self.device)
```

`rollout.backend` is a plain string field — "Currently support `vllm`, `vllm_async` and `trtllm`, and other custom backends" (`policy/config/__init__.py:1489-1491`). Registered in-tree: `vllm` (`vllm_rollout/vllm_rollout.py:144`), `vllm_async` (`vllm_rollout_async.py:98`), `vla` (`vla_rollout/vla_rollout.py:125`), `diffusion_nft_rollout` (`diffuers_rollout/nft_rollout.py:31`), `example_hf`. Two caveats in the wiring: (1) **trtllm bypasses the registry entirely** — `LLMRolloutWorker.build_runner` special-cases `backend == "trtllm"` into `TRTLLMRolloutWrapper` before the registry-driven `DisaggregatedRolloutControlWorker` path (`rollout/worker/llm_worker.py:76-104`; the trailing `else: raise ValueError` there is unreachable since the if/elif covers both cases — custom backends go through the registry lookup inside the control worker), because trtllm needs `mpirun` instead of torchrun (`launcher/launch_replica.sh`, `LAUNCH_BINARY="mpirun"` in the rollout branch); (2) async mode is allow-listed: `SUPPORT_ASYNC_BACKEND = ["vllm_async"]` with an assert (`rollout_control.py:99, 207-211`). Command handlers are also backend-keyed — `RolloutCommandRegistry` is a two-level dict `{backend: {CommandType: handler}}` (`dispatcher/command.py:497-512`) registered via `@CommMixin.register_rollout_command_handler(cmd, backend="vllm")` (`comm/base.py:64-72`), so a new backend can override just the weight-sync handlers while inheriting the worker loop.

**Custom trainer mirrors this exactly**: `@TrainerRegistry.register(trainer_type="grpo")` on `GRPOTrainer` (`policy/trainer/llm_trainer/grpo_trainer.py:547`), looked up from config at `rl_worker.py:881-891` via `config.train.train_policy.trainer_type` (field defaults `"sft"`/`"grpo"` per policy type, `config/__init__.py:115-118, 371-374`). In-tree trainers are eagerly imported by `policy/trainer/__init__.py:16-35` (this is what fires the decorators); a custom trainer registered in a user script works because the script body runs in every worker process before `launch_worker()` — the documented recipe is `tools/custom_example/custom_grpo_trainer.py:44-61` ("Implement the custom GRPO trainer ... register it with a unique trainer_type ... Specify the custom trainer_type and rollout_backend_type in the cosmos config file").

### 12.4 The config/args system

**Declaration.** One pydantic tree in `cosmos_rl/policy/config/__init__.py` (1972 lines, Part I §5 covered the doc-schema generator): `Config{custom, train, rollout, policy, logging, profiler, validation, distillation, vla, redis, eth_ips, mode}` (`config/__init__.py:1793-1819`). The module tail materializes `COSMOS_CONFIG_SCHEMA = Config.model_json_schema(schema_generator=CustomJsonSchemaGenerator)` (`config/__init__.py:1970-1972`); internal plumbing fields like `redis`/`eth_ips` are schema-hidden via `hide_in_doc` (`config/__init__.py:1805-1814`). `custom: Dict[str, Any]` (`:1794-1796`) is the official escape hatch for user-script settings.

**Validation has three layers:**
1. A `mode="before"` validator that *infers the algorithm from field shape*: if `train.train_policy` contains any of `temperature/epsilon_low/epsilon_high/kl_beta/use_remote_reward` it stamps `type="grpo"`, else `type="sft"` (`config/__init__.py:1847-1863`) — the discriminator for the `Union` of `SFTDataConfig`/`GrpoConfig` (both declare `type: Literal[...]`, `:113, 369`).
2. Per-subconfig `@model_validator(mode="after")` blocks (≈18 of them); they don't only assert, they **mutate**: e.g. `allowed_outdated_steps` is raised to `sync_weight_interval - 1` and `max_inflight_steps` re-clamped "so the hard throttle never fires before the soft throttle" (`config/__init__.py:1910-1932`).
3. Cross-field invariants in `Config.check_params_value` — PP batch divisibility, GRPO+LoRA forbids compile and TP>1 (`:1866-1909`), distillation force-enables `rollout_as_token_ids`/`bypass_reward` (`:1957-1963`).

`Config.from_dict` additionally injects a run timestamp and rewrites `output_dir = output_dir/<timestamp>` (`config/__init__.py:1822-1836`).

**Plumbing — controller is the single source of truth.** Only the controller process reads the TOML: `run_web_panel.main` does `toml.load` → `CosmosConfig.from_dict` → `controller.setup(...)` (`dispatcher/run_web_panel.py:636-672`). `Controller.setup` then *writes runtime discoveries back into the config object*: the chosen Redis port and node IPs (`self.config.redis = str(redis_free_port)`; `self.config.eth_ips = ";".join(ips)`, `controller.py:135-140`). The launcher also patches `n_init_replicas` into the dict before serializing it to a tempfile (`launcher/launch_all.py:733-750`). Workers never see the TOML path: they fetch the post-setup config over HTTP from the `/meta` endpoint (`run_web_panel.py:162-167`) and re-run the full pydantic pipeline locally — `metadata = api_client.get_controller_metadata(); cosmos_config = CosmosConfig.from_dict(metadata["config"])` (`policy/policy_entry.py:32-39`; identically `rollout/worker/llm_worker.py:39-49`). Workers still mutate their local copy for runtime facts (`dp_shard_size == -1` resolved from the actual mesh, `rollout_control.py:109-111`).

**CLI args are minimal by design** — `worker_entry_parser()` defines just `--port`, `--redis-port`, `--config` plus passthrough (`dispatcher/data/packer/base.py:30-44`); every behavioral knob lives in the TOML. The launcher's own argparse (`launch_all.py:103-344`) only covers placement (GPUs, nodes, `--p2r-ratio`, Lepton cloud flags) and forwards `--config`/`--script` to the shell launchers.

---

## 13. Startup sequence & failure handling

### 13.1 Startup order-of-operations (disaggregated GRPO)

Part I §4.2 covered the launcher's placement math; this is the temporal ordering, including what is deliberately lazy.

1. **`cosmos-rl --config x.toml [script.py]`** → `launch_all.main` loads the TOML as a plain dict and computes per-replica GPU products from `[policy|rollout].parallelism` (`launcher/launch_all.py:396-456`).
2. Replica counts decided (optionally `--p2r-ratio`), `n_init_replicas` patched into the dict, dict dumped to a tempfile that becomes the *only* config file anyone reads (`launch_all.py:733-756`).
3. `NodesManager.replica_placement(...)` + `finalize()` produce per-node command lists; controller command built as `launch_controller.sh --config <tmpfile> --port <port> [--script user.py]` (`launch_all.py:752-783`).
4. Controller subprocess spawned; launcher **blocks** on `wait_for_url_ready(control_url, ...)` before spawning any replica (`launch_all.py:845-866`).
5. Inside the controller: `launch_controller.sh` runs `python -m cosmos_rl.dispatcher.run_web_panel` (default `SCRIPT`, `launcher/launch_controller.sh:6`) → config load + `controller.setup()` (`run_web_panel.py:636-672`) → wandb init if configured (`controller.py:109-110`), `ControllerDataFetcher` built (`controller.py:122-133`), **redis-server forked** on a discovered free port written into `config.redis` (`controller.py:135-173`) → `uvicorn` serves the FastAPI app (`run_web_panel.py:680-685`); the lifespan hook starts the heartbeat-monitor thread (`run_web_panel.py:104-124`).
6. Per replica: `launch_replica.sh` exports `COSMOS_ROLE` and execs `torchrun --rdzv_backend c10d -m cosmos_rl.policy.train` (policy) / `cosmos_rl.rollout.rollout_entry` (rollout), or the user script if `--script` was given (`launcher/launch_replica.sh`, role block + `LAUNCH_CMD` tail). A user script ends with `launch_worker(...)` which switches on `COSMOS_ROLE` (`launcher/worker_entry.py:31-94`).
7. **Policy rank-path:** `train.main` → `policy_entry` (`policy/train.py:26-58`): `APIClient("POLICY")` fetches `/meta`, config re-validated (`policy_entry.py:32-41`); `ParallelDims.from_config` + `init_distributed()` + `build_mesh` (`policy_entry.py:44-48`); `RLPolicyWorker(...)` constructed (`policy_entry.py:62-73`).
8. `PolicyWorkerBase.__init__` (`policy/worker/base.py:34-90`): HF config fetched, device pinned, `check_config()` batch-divisibility asserts (`base.py:92-113`), then `init_comm()` (`base.py:70`) → replica name = broadcast UUID, Redis stream handler init, **HTTP register of every rank as an Atom**, `dist.barrier()`, rank 0 spawns the heartbeat `mp.Process` (`comm/base.py:88-106, 249-303`). Registration happens **before** any model is built.
9. `RLPolicyWorker.__init__` continues: `build_runner` → `TrainerRegistry.get_trainer_cls(trainer_type)(...)` (`rl_worker.py:88-93, 881-891`). `LLMTrainer.__init__` builds the model **on meta device** (`ModelRegistry.build_model`, `llm_trainer.py:89`), applies FP8/FP4 conversion (`:92-100`), calls the family `parallelize_fn` (`:114-121`), materializes meta→GPU/CPU and runs `post_to_empty_hook` (`:123-146`). No weights from disk yet.
10. Back in the worker: `HighAvailabilitylNccl` background thread + `DistKVStore` (`rl_worker.py:98-107`), then `prepare_shard_infos_for_weight_sync_insts()` all-gathers per-rank shard layouts and rank 0 POSTs them to the controller (`rl_worker.py:129, 169-213`).
11. **Controller reacts to registration** (`PolicyStatusManager.register`, `dispatcher/status.py:384-485`): when the *first* policy replica has all atoms, it triggers `WeightResumeCommand` (`status.py:429-437`); if rollout replicas already arrived, it chains `PolicyToRolloutUnicastCommand` + `RolloutToRolloutBroadcastCommand` to seed rollout weights (`status.py:451-471`); `post_register_hook` → `trigger_rebuild_mesh` → `BuildMeshCommand` (`status.py:487-495`).
12. Policy `execute()` → `main_loop()` starts the fetch-command thread (all ranks) and fetch-rollouts thread (rank 0), then spins on `broadcast_command()`/`execute_command()` (`rl_worker.py:792-853`, `policy/worker/base.py:115-121`). The `WeightResumeCommand` handler now performs the actual checkpoint/HF weight load: `self.trainer.weight_resume()` (`rl_worker.py:499-509`; HF load itself at `llm_trainer.py:792-815`).
13. **Rollout rank-path:** `rollout_entry.run_rollout` → `LLMRolloutWorker` (config fetch, task-type gate that exits non-RL runs, `rollout/worker/llm_worker.py:39-56`) → `init_distributed` + mesh → `DisaggregatedRolloutControlWorker` (`llm_worker.py:80-92`). Its `__init__` registers with the controller via `RolloutWorkerBase.runner_init → init_comm()` (`rollout/__init__.py:91-115`), instantiates the backend from the registry (`rollout_control.py:130-132`) — but the vLLM `post_init_hook` deliberately leaves `self.rollout_engine = None` (`vllm_rollout/vllm_rollout.py:164-196`).
14. `work()` starts rank 0's command-query thread and enters `main_loop` (`rollout_control.py:2201-2226`). **Engine creation is lazy**: the first `PolicyToRolloutUnicastCommand` calls `lazy_initialize_rollout_engine(load_format)` — `load_format="auto"` (load real HF weights) if this replica is the unicast destination, `"dummy"` otherwise since NCCL will overwrite them (`rollout_control.py:1014-1017, 973-1000`). `init_engine` builds `LLM(..., distributed_executor_backend="external_launcher")` (`vllm_rollout.py:249, 298-304`), then `prepare_shard_infos_for_weight_sync_insts()` posts rollout shard infos (`rollout_control.py:274-334`).
15. **First step ready:** rollout main loop's `state.weight_synced()` gate opens after the seeding P2R/R2R completes; prompts flow (Part I §7 Phase A), rollouts accumulate on the controller, and `try_trigger_data_fetch_and_training` issues the first `DataFetchCommand` per policy replica (`status.py:1444-1456`).

### 13.2 Failure handling

**The detection primitive is the heartbeat, and the unit of recovery is the replica.** Every replica's rank 0 runs a *separate OS process* posting heartbeats every `COSMOS_HEARTBEAT_SEND_INTERVAL=60s` (`comm/base.py:287-298, 343-362`; constants `utils/constant.py:30-33`). Being a daemon *process* (not a thread), it keeps beating even if the main loop hangs — so heartbeat timeout catches *crashes*, while *hangs* are caught by the NCCL watchdog path below. The controller lifespan thread sweeps both managers every `COSMOS_ROLLOUT_SCAN_INTERVAL=10s` (`run_web_panel.py:104-124`, `constant.py:35`) and declares death after `COSMOS_HEARTBEAT_TIMEOUT=200s`:

```python
# cosmos_rl/dispatcher/status.py:234-239
if now - replica.status.heartbeat_timestamp > COSMOS_HEARTBEAT_TIMEOUT:
    logger.warning(f"[Controller] Policy {replica.name} is dead")
    dead_replicas.add(replica.name)
for replica_name in dead_replicas:
    self.unregister(replica_name)
```

**Rollout worker dies.** `RolloutStatusManager.unregister` pops the replica and — unless training already finished — rebuilds the rollout mesh over the survivors with a fresh `BuildMeshCommand` and updates the data fetcher's mesh size (`status.py:1608-1624, 1692-1700`). Nothing requeues in-flight *prompts* explicitly; the staleness/throttle machinery absorbs the loss (prompts were only counted in `samples_on_the_fly`, and rollouts the dead worker never reported simply never arrive — the buffer-threshold trigger recomputes against the surviving replica count). On the worker side, a clean exit also unregisters via `atexit` (rank 0 only, `comm/base.py:303-315`).

**Training step throws.** There is **no** try/except around `trainer.step_training` — `execute()` prints the traceback and re-raises (`policy/worker/base.py:115-135`), torchrun tears down all ranks of that replica, heartbeats stop, and the controller path above runs `PolicyStatusManager.unregister` → `trigger_rebuild_mesh` over surviving policy replicas + `recompute_total_steps()` + rollout-buffer re-bucketing (`status.py:362-382, 487-495`). Crucially the **launcher tolerates replica death**: in the process-monitor loop a non-zero exit only kills the whole job if it is the controller (`if controller_id == -1 or i == controller_id: ... sys.exit(1)`, `launcher/launch_all.py:902-920`) — otherwise the dead process is just removed from the list. So elasticity-down is real, but there is no supervisor that *respawns* a replica; scale-up means launching a new `launch_replica.sh` by hand/orchestrator, which re-enters the registration path of §13.1 step 11.

**Hangs (NCCL) rather than crashes.** Part I §4.3 covered the retry wrapper; the recovery protocol on top: on failure the worker does not die — it reports and waits for a new mesh:

```python
# cosmos_rl/utils/distributed.py:600-606
except Exception as e:
    # mark the communicator is not ready
    self.is_comm_ready.clear()
    # report the error to the controller
    # the communicator will destroy before buildmesh
    self.api_client.post_nccl_comm_error(self.replica_name, e)
```

(`__do_nccl_op_with_retry`, `max_retry = 3` at `distributed.py:435`, watchdog at `utils/pynccl.py:360`). Controller side, `set_replica_ncclerror` opens a `COSMOS_NCCL_ERROR_CLEAN_REPLICA_DELAY=10s` debounce window (invoke-id check so only the *last* report acts, `controller.py:609-623`, `constant.py:39-40`); `post_ncclerror` then inverts the report set — **replicas that did NOT report an error are presumed hung** — unregisters them, and the unregister path re-triggers `BuildMeshCommand` for the reporters (`controller.py:633-668`). The same function shows the asymmetry of the whole story: `Role.ROLLOUT` raises `NotImplementedError` (`controller.py:624-627, 663-666`) — NCCL-error elasticity exists only for the policy mesh; rollout replicas are covered only by heartbeat death.

**Transport-level resilience elsewhere:** every worker→controller HTTP call goes through `make_request_with_retry` with exponential backoff over alternative URLs (`utils/network_util.py:55-95`), defaults `max_retries=60, retries_per_delay=5, initial_delay=1.0, max_delay=60.0, backoff_factor=2.0` (`constant.py:67-75`). Mesh-shrink mid-step is also handled at the data level: a `DataFetchCommand` with `items_count==0` is an explicit "fake step" that skips training but still ACKs, keeping the step state machine consistent (`rl_worker.py:526-553`).

### 13.3 Observability

**Logger.** A single `logging.getLogger("cosmos")` configured at import, level from `COSMOS_LOG_LEVEL` (`utils/logging.py:19-33`); the launcher's `--debug` flag just exports that env var (`launch_all.py:398-399`).

**Metric pipeline is "workers measure, controller aggregates and publishes".** Sinks are chosen by `logging.logger = ["console", "wandb"]` (`LoggingConfig`, `config/__init__.py:1563-1597`). Only the **controller** initializes wandb (`controller.py:109-110`; `init_wandb` uses `config.train.timestamp` as resumable run id, `utils/report/wandb_logger.py:49-100`); workers never log to wandb directly — they ship dicts:

- **Trainer side:** `GRPOTrainer.step_training` brackets the whole step with CUDA events (`grpo_trainer.py:974-975`, `end_event.record()` at `:1815`) and, on the master rank only, builds `report_data` — `train/iteration_time` (event elapsed), `train/loss_avg|loss_max`, `train/learning_rate`, KL stats, `train/grad_norm`, accumulated `self.metrics`, and optional MFU via `compute_mfu(model, n_tokens=acc_n_tokens, iter_time, num_gpus, dtype)` gated by `logging.report_mfu` with an honest FIXME that TP/PP makes rank-0 MFU inaccurate (`grpo_trainer.py:1838-1877`; `compute_mfu` at `utils/util.py:563`). This dict rides the train-ACK HTTP call (`rl_worker.py:558-565`).
- **Rollout/reward side:** reward-quality and length metrics are computed by the controller **at dispatch time** from the buffered rollouts of the step — `train/reward_{mean,std,max,min}`, `rollout/completion_length_*`, `rollout/advantage_*`, `rollout/filter_reward_*` (`dispatcher/status.py:1460-1505`); per-rollout `report_metrics` dicts (from custom rewards) are folded in via `aggregate_report_data(..., prefix="train/")` (`status.py:1498-1504`).
- **Publication:** when all replicas ACK a step, `train_ack` averages/maxes across replica dicts, merges with the dispatch-time metrics into `train_report_data[step]` (a `RollingDict(maxlen=20)`, `status.py:130`), adds dynamic-sampling filter ratios (`status.py:1149-1162`), converts `rollout_images`/`rollout_videos` entries into `wandb.Image`/`wandb.Video` with prompt+reward captions (`status.py:1180-1217`), then fans out to wandb (`log_wandb(data, step)`, `wandb_logger.py:102-107`), a one-line console summary (Reward Mean/Std/Max/Min, completion lengths, loss, entropy, grad norm, KL, iteration time — `status.py:1221-1228`), and user `custom_logger_fns(report_data, step)` each wrapped in its own try/except "to avoid the error of custom logger function affecting the fundamental logging system" (`status.py:1236-1248`). Validation results follow a parallel path with `val/` prefixes (`status.py:638-713`).

**Profiling is remotely triggerable.** `CosmosProfiler` wraps `torch.profiler` with wait/warmup/active scheduling (`utils/profiler.py:31-114`); besides static TOML enablement, the controller can switch it on per-step — `DataFetchCommand.do_profile` carries `active_steps/rank_filter/record_shape/profile_memory/with_stack/with_modules` and the handler calls `profiler.start_dynamic(...)` (`rl_worker.py:512-520`), steps it after each training step (`rl_worker.py:555`), and the ACK reports `profiler.check_finished()` so the controller can clear the replica's profile flag (`rl_worker.py:559-563`, `status.py:1079-1082`). Traces are exported as chrome trace `.json.gz` under `<output_dir>/profile_trace/<replica>_<rank>/`, optionally uploaded to S3 on a thread pool, and the path is POSTed back to the controller (`profiler.py:216-255`). The controller additionally serves a live HTML status panel of all replicas (`run_web_panel.py:129-158`).

---

## 14. Engineering details worth copying (and anti-patterns)

Additions to Part I §8's ten items, from the deep dive. Worth copying:

1. **Reference policy as a CPU state-dict swap, not a second model.** `_swap_model_state_dict` (`grpo_trainer.py:1955-1970`) keeps the KL reference on CPU and swaps it into the live (sharded) model for the REF_COMPUTE phase — zero extra GPU memory, works under any parallelism because the swap is key-by-key on the *local* state dict. For wm-infra GRPO with a frozen reference flow model, this beats holding a second FSDP instance.
2. **Logits-row elimination via `interested_tokens`.** Slicing hidden states *before* the lm_head so `[B,T,vocab]` is never materialized for non-loss tokens (`gpt/__init__.py:503-509`; gated to pure-DP at `grpo_trainer.py:1288-1297`), paired with the flat-ragged `[n_logprob_tokens] + cu_seqlens` loss layout (`utils/util.py:944-996`). The analog for diffusion RL: only materialize per-step logprob tensors for the timesteps that enter the loss.
3. **Rational-interval slice algebra for layout-mismatched weight sync.** `DimSliceInfo{offset,total_size,length}` with gcd `simplify()` and `tensor_overlap_info_at_dim` interval intersection (`dim_slice_info.py:29-74, 211-240`) turns "FSDP/TP layout on one side, vLLM module-type layout on the other" into pure dimension math computed once. Part I §8 item 4 recommended plan-as-data; this is the actual algebra to lift.
4. **Resume as sample-count arithmetic.** Checkpoint only `remain_samples_num`; rederive epoch + offset and skip forward with a one-shot `SkippingSampler` over a `(seed, epoch)`-deterministic shuffle (`data_fetcher.py:247-301`; `policy/trainer/sampler.py:20-53`). One integer replaces sampler state serialization — only valid because shuffle is a pure function of seed and epoch, which is the property to preserve in wm-infra's data loaders.
5. **Pad, don't drop, to keep distributed accounting exact.** Empty completions are replaced with the EOS token because "we need to make sure the expected number of global_batch_size is reached at exact time" (`rollout_control.py:1874-1880`). When per-step counts are load-bearing (fully-synchronized on-policy mode), a degenerate sample is safer than a missing one.
6. **Re-stamp `weight_version` with the *actual* generating version.** The controller's stamp is a prediction; the worker overwrites it after generation (`rollout_control.py:1893-1902`) so the staleness filter judges reality, not the forecast. Any wm-infra weight-version tag should follow the same two-phase stamp.
7. **Rebuild the LR scheduler when `total_steps` becomes known**, carrying `state_dict` over (`grpo_trainer.py:2073-2095`) — the clean answer to "the trainer starts before the dataset size is known" in any disaggregated design.
8. **Lazy engine init keyed on load format.** The rollout engine is built only at the first weight-sync command, with `load_format="dummy"` for replicas whose weights will arrive via NCCL anyway (`rollout_control.py:973-1017`) — skips a redundant multi-GB HF load per replica.
9. **Config validators that enforce *relationships*, not just ranges**: clamping `max_inflight_steps` so "the hard throttle never fires before the soft throttle" (`config/__init__.py:1910-1932`). Encode inter-knob invariants in the validator rather than in docs.
10. **Heartbeat as a separate OS process + inverted hang detection.** A daemon `mp.Process` heartbeat (`comm/base.py:287-298`) cleanly splits crash detection (heartbeat stops) from hang detection (NCCL watchdog: the replicas that *didn't* report an error are the hung ones, `controller.py:633-668`). The inversion is the clever part — a hung rank can't report itself.
11. **Remotely triggerable profiler riding the existing command channel** (`DataFetchCommand.do_profile` → `profiler.start_dynamic`, `rl_worker.py:512-520`; chrome traces + S3, `profiler.py:216-255`) — no restart to profile step N of a long run. Directly portable to wm-infra's engine commands.
12. **Filesystem-scan model discovery with warn-on-import-failure** (`policy/model/__init__.py:42-136`): a new model family is a dropped-in directory, and an optional-dependency family degrades to a warning instead of breaking everyone's import.

Anti-patterns to avoid (beyond Part I §8 item 10's `Command.deserialize` chain):

13. **`REWARD_FUNC_MAPPING` is a hard-coded table, not a registry** (`dispatcher/algo/reward.py:286-293`) — inconsistent with the four registries everywhere else, so built-in rewards require editing core code while models/trainers/backends don't. (The callable-injection path, §12.2 tier 2, is the redeeming design.)
14. **trtllm bypasses `RolloutRegistry`** with a special-case branch plus an unreachable `else: raise ValueError` (`rollout/worker/llm_worker.py:76-104`) — when one backend's launch constraints (mpirun) leak into the worker factory, the registry abstraction silently stops being the single dispatch point.
15. **One `torch.optim` object per parameter** in the non-fused path (`optm/__init__.py:139-186`) — workable only because `step()` loops, but it multiplies per-optimizer overhead and makes optimizer-state introspection awkward; the fused per-mesh grouping is the sane shape.
16. **Rank-indexed raw `torch.save` checkpoints** (`utils/checkpoint.py:434-526`) lock resume to the identical parallel layout — no DCP resharding-on-load. Fine for fixed clusters; a trap for elastic ones (ironic, given the elasticity story elsewhere). wm-infra should keep DCP-style resharding-capable checkpoints.
17. **Silent fallback on checkpoint corruption** — `weight_resume` catches `FileNotFoundError`/corruption and falls back to HF weights with only a log line (`grpo_trainer.py:2038-2057`); a typo'd resume path trains from scratch instead of failing fast.
18. **Asymmetric elasticity** — NCCL-error recovery is `NotImplementedError` for rollout replicas (`controller.py:624-627, 663-666`), and there is no respawn supervisor (`launch_all.py:902-920`): "fault-tolerant and elastic" means *shrink* for policy, *heartbeat-death only* for rollout. State the actual coverage of any HA claim.

---

## 15. Part II source-of-truth index

One merged table from the three deep-dive readers, deduplicated against each other and against Part I §9. Spot-checked items during merge: `_swap_model_state_dict`, `compute_logprobs`, `reshard_after_forward` policy, empty-completion EOS padding, `SkippingSampler` + resume arithmetic, `ModelRegistry.register_model`, `post_ncclerror` inversion, heartbeat sweep, one-optimizer-per-param, trtllm bypass, `interested_tokens` lm_head slicing — all confirmed.

| Claim | Path:lines |
|---|---|
| Batch hierarchy: per_optimize / mini_batch sizing | `cosmos_rl/policy/trainer/llm_trainer/grpo_trainer.py:1034-1047`; `cosmos_rl/policy/config/__init__.py:541-554` |
| Phase machine REF/OLD/TRAIN; grad-enable + μ loop | `grpo_trainer.py:1063-1102` |
| Reference model = CPU state-dict swap ✔ | `grpo_trainer.py:1955-1970`; snapshot `grpo_trainer.py:2029-2036`; periodic reset `grpo_trainer.py:1918-1949` |
| Temperature-scaled logits; logprob dtype cast | `grpo_trainer.py:1552-1561, 1996-2003`; `config/__init__.py:852-856` |
| Flat-ragged logprob recompute (masked select + cu_seqlens) ✔ | `cosmos_rl/utils/util.py:919-996` (selective_log_softmax `:314`; entropy chunking `:906`) |
| π_old three sources (on-the-fly / precomputed / rollout) | `grpo_trainer.py:1597-1634`; rollout-as-old `grpo_trainer.py:1373-1397, 1616-1626` |
| Importance ratio, GSPO seq ratio, clamps | `grpo_trainer.py:162-203` |
| AIPO one-sided clip vs dual-clip PPO | `grpo_trainer.py:205-244` |
| Decoupled-loss behavior weight (AREAL) + cap | `grpo_trainer.py:246-256`; `config/__init__.py:629-639, 691-692` |
| Off-policy sequence masking math | `grpo_trainer.py:67-110, 258-269` |
| k3 KL term, unbiased variant; combined return | `grpo_trainer.py:271-293, 349-353` |
| 4 loss normalizations; balance_dp_token global allreduce | `grpo_trainer.py:295-348` |
| Entropy bonus, positive-NLL, loss scaling, backward | `grpo_trainer.py:1739-1784` |
| PP swizzle smuggled microbatch metadata; pp_loss_fn | `grpo_trainer.py:358-544, 954-972, 1420-1517, 2134-2145` |
| Distillation: teacher KL/JSD as advantage corrections | `grpo_trainer.py:692-835, 1205-1235, 1640-1714` |
| parallelize order PP→TP→CP→compile→FSDP; HSDP mesh names | `cosmos_rl/policy/model/gpt/parallelize.py:46-113` |
| TP plan; async TP; Float8 TP | `gpt/parallelize.py:233-338` |
| Ulysses CP wrap; trainer-side slicing; seq_len_multiple | `gpt/parallelize.py:216-230`; `grpo_trainer.py:1348-1368, 1193-1197`; `llm_trainer.py:260-270` |
| FSDP2 per-block, reshard_after_forward policy ✔, CPUOffloadPolicy | `gpt/parallelize.py:366-423` |
| PP PipelineStage/Schedule1F1B, microbatch constraints | `gpt/parallelize.py:114-211`; `grpo_trainer.py:1055-1061` |
| EP: Shard(0) expert ParallelStyle; dual-mesh FSDP for MoE | `cosmos_rl/policy/model/qwen3_moe/parallelize.py:191-213, 278-346`; `utils/parallelism.py:97-102` |
| No Megatron engine; only borrowed MoE kernels | `cosmos_rl/policy/kernel/megatron_moe/token_dispatcher.py:284` (sole `megatron` import in package) |
| Meta-device build + materialization, fsdp_offload to CPU | `cosmos_rl/policy/trainer/llm_trainer/llm_trainer.py:89-156` |
| DimSliceInfo slice algebra; overlap intersection | `cosmos_rl/utils/dim_slice_info.py:29-100, 211-240` |
| Policy shard info from DTensor placements/chunk meta | `dim_slice_info.py:242-300`; `cosmos_rl/utils/parallelism_map.py:461-539` |
| Rollout shard info from vLLM/TRT-LLM module types | `parallelism_map.py:590-713` |
| Send-instruction generation (per-pair dim intersection, dup dedup) | `parallelism_map.py:1115-1258` |
| Elastic replica join: full state sync (model/optim/lr/rng) | `llm_trainer.py:315-443`; `cosmos_rl/policy/worker/rl_worker.py:292-348` |
| One-optimizer-per-param (non-fused) / per-mesh (fused) ✔ | `cosmos_rl/policy/trainer/optm/__init__.py:139-186` |
| Optimizer state via DCP flatten, idx-namespaced | `optm/__init__.py:209-233` |
| Per-module-path LR dict resolution | `llm_trainer.py:176-237`; `optm/__init__.py:300-419` |
| warmup_stable_decay LambdaLR; rebuild at first step | `optm/__init__.py:500-620`; `grpo_trainer.py:2073-2095`; step cadence `grpo_trainer.py:1879-1880` |
| all_reduce_states: bucketed cross-replica AVG allreduce | `grpo_trainer.py:2097-2132`; `cosmos_rl/utils/distributed.py:93-181` |
| Hand-rolled global grad-norm clip incl. PP reduce | `distributed.py:185-286`; `config/__init__.py:820-822` |
| Optimizer-step cadence (per optimize-slice or env interval) | `grpo_trainer.py:1788-1811` |
| Dynamic token-budget minibatches; sequence packing gates | `grpo_trainer.py:1116-1145, 1269-1336` |
| Checkpoint trigger from step_training (master replica) | `grpo_trainer.py:1883-1911` |
| Safetensors export (4GB chunks, corner-rank writes, PP manifest, async upload) | `llm_trainer.py:445-790` |
| Cosmos ckpt: per-rank pth files + extra info + markers; async pipeline | `cosmos_rl/utils/checkpoint.py:434-616` |
| Saving-rank count = world/dp_replicate; completeness check | `checkpoint.py:115-174` |
| Retention/best-symlink rotation | `checkpoint.py:759-831` |
| weight_resume order; HF fallback; kl_beta forces HF-first | `grpo_trainer.py:2026-2071` |
| load_checkpoint (multi-timestamp glob, PP prefix-strip, empty_cache) | `checkpoint.py:176-219, 618-723`; `llm_trainer.py:816-830` |
| Resume validation round-trip; controller seeds step from extra info | `rl_worker.py:499-509`; `cosmos_rl/dispatcher/run_web_panel.py:340-345`; `cosmos_rl/dispatcher/controller.py:170-194` |
| interested_tokens lm_head row elimination (DP-only gate) ✔ | `grpo_trainer.py:1288-1297`; `cosmos_rl/policy/model/gpt/__init__.py:445-509` |
| Per-layer grad checkpointing wiring | `llm_trainer.py:151-154`; `gpt/__init__.py:486-499` |
| Activation offloading (torchtune), lm_head exemption, use_streams=False FIXME | `cosmos_rl/utils/activation_offloading.py:28-120, 380-420`; `llm_trainer.py:253-256`; usage `grpo_trainer.py:1533-1536` |
| Dtype policy (master/param/reduce/logprob); FP8/FP4 on meta | `config/__init__.py:837-872`; `llm_trainer.py:92-100`; `cosmos_rl/policy/model/base.py:621-624` |
| `RLPayload` schema (all fields incl. logprobs/valid/filter_rewards) | `cosmos_rl/dispatcher/data/schema.py:49-168` |
| `Rollout` per-completion schema; `RolloutResult` engine schema | `cosmos_rl/dispatcher/data/schema.py:171-247`; `cosmos_rl/rollout/schema.py:21-55` |
| `RLDataset.__getitem__` sets `prompt_idx` | `cosmos_rl/dispatcher/data/__init__.py:30-49` |
| Controller dataloader, DistributedSampler(num_replicas=1), collate | `cosmos_rl/dispatcher/data/data_fetcher.py:213-220, 334-351` |
| Epoch rollover + set_epoch reshuffle; is_end | `cosmos_rl/dispatcher/data/data_fetcher.py:549-578` |
| Resume: remain_samples_num → epoch/bias → SkippingSampler ✔ | `cosmos_rl/dispatcher/data/data_fetcher.py:239-301`; `cosmos_rl/policy/trainer/sampler.py:20-53` |
| local_dataset: prompt stripping / index dataset / worker re-query | `cosmos_rl/dispatcher/data/data_fetcher.py:182-188, 582-589, 663-740` |
| weight_version stamping (exact per-version accounting; DAPO; SFT) | `cosmos_rl/dispatcher/controller.py:263-276, 369-434, 446-463` |
| Sampling params: n_generation, logprobs=0, prompt_logprobs | `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:199-247` |
| Single-turn generation → RolloutResult assembly | `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:593-730` |
| parse_logprobs (index 0 = sampled token); collect gating | `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:430-474, 533-590` |
| Generation exception → return [] | `cosmos_rl/rollout/vllm_rollout/vllm_rollout.py:724-729` |
| Empty-completion EOS padding ✔; validity filter; payload merge | `cosmos_rl/rollout/worker/rollout_control.py:1825-1919` |
| weight_version re-stamp with actual generating version | `cosmos_rl/rollout/worker/rollout_control.py:1893-1902` |
| RewardDispatcher: pool, 64/task, FIFO dequeue, bypass, remote batching | `cosmos_rl/reward/dispatcher.py:27-264`; `cosmos_rl/utils/constant.py:78-84`; setup `rollout_control.py:191-193, 254-264` |
| Reward fns swallow exceptions → 0.0 | `cosmos_rl/dispatcher/algo/reward.py:213-222, 244-247, 280-283` |
| Weighted reward sum + filter rewards | `cosmos_rl/dispatcher/algo/reward.py:451-496` |
| DAPO valid flag + shared-prefix n_ignore_prefix_tokens | `cosmos_rl/reward/local_calculator.py:226-359` |
| RolloutGroup.compute_rollouts; most-likely-mode reward | `cosmos_rl/reward/base.py:40-140` |
| Remote reward: enqueue/uuid maps/poll-retry/clamp/NaN guard | `cosmos_rl/reward/remote_calculator.py:104-178, 180-268, 346-436` |
| dynamic_sampling drop at worker; token-id blanking on report | `cosmos_rl/rollout/worker/rollout_control.py:1648-1713` |
| extra_info per-index split in extract_rollouts | `cosmos_rl/utils/payload.py:104-119` |
| DAPO filter bookkeeping; on-policy overflow drop; samples_on_the_fly decrement | `cosmos_rl/dispatcher/status.py:774-810, 1063-1067` |
| Policy ingest; dispatch_rollouts local re-hydrate by prompt_idx | `cosmos_rl/policy/worker/rl_worker.py:256-272, 677-767` (re-hydrate `:701-708`) |
| step_training field consumption; advantages tensor | `grpo_trainer.py:940-1027` |
| logprob mask construction (prefix ignore, -1 shift) | `cosmos_rl/dispatcher/data/packer/decoder_only_llm_data_packer.py:59-149` |
| rollout logprobs as old/decoupled loss inputs; config coupling | `grpo_trainer.py:475-486, 1243-1268`; `cosmos_rl/policy/config/__init__.py:566-611, 630-692` |
| remain_samples_num checkpointed by trainer | `grpo_trainer.py:1900-1910` |
| Model auto-discovery scans `policy/model/` dirs at import | `cosmos_rl/policy/model/__init__.py:42-136` |
| `@ModelRegistry.register` fills model+weight-mapper+packer registries ✔ | `cosmos_rl/policy/model/base.py:546-593`; example `policy/model/qwen3_moe/__init__.py:383`; packer example `model/pi05/__init__.py:336` |
| Model class chosen by HF `model_type`, `hfmodel` fallback + retry | `policy/model/base.py:596-619, 715-733`; `utils/constant.py:64`; rollout side `rollout/worker/rollout_control.py:161-175` |
| Diffusers models keyed on `_class_name`, routed by `policy.is_diffusers` | `policy/model/base.py:738-748, 786-791` |
| BaseModel abstract contract (parallelize_fn, load_hf_weights, …) | `policy/model/base.py:437-529`; Qwen3MoE `parallelize_fn` `qwen3_moe/__init__.py:557-561` |
| WeightMapper name/split contract; trtllm gate/up swap | `policy/model/base.py:794-925, 1007-1013`; `qwen3_moe/weight_mapper.py:25-80`; `rollout/trtllm_rollout/trtllm_worker.py:145` |
| Per-model rollout shard strategies registry | `utils/parallelism_registry.py:32-80`; `model/deepseek_v3/weight_mapper.py:408` |
| Built-in rewards hard-coded table; TOML weighted dict | `dispatcher/algo/reward.py:286-293, 296-378`; `policy/config/__init__.py:423, 666-669` |
| Custom rewards via `launch_worker(reward_fns=...)`, override TOML | `launcher/worker_entry.py:10-94`; `tools/dataset/gsm8k_grpo.py:375-383`; `dispatcher/algo/reward.py:347-355` |
| Reward component dicts; group/batched reward invocation | `dispatcher/algo/reward.py:313-345, 396-447` |
| RolloutBase contract + hooks; registry; backend from `rollout.backend` | `rollout/rollout_base.py:30-274`; `rollout_control.py:130-132`; `config/__init__.py:1489-1491`; `example_custom_rollout.py:44` |
| trtllm bypasses registry (mpirun path) ✔; async allow-list | `rollout/worker/llm_worker.py:76-104`; `rollout_control.py:99, 207-211` |
| TrainerRegistry + `trainer_type` config selection; custom-trainer recipe | `policy/trainer/base.py:192-233`; `grpo_trainer.py:547`; `rl_worker.py:881-891`; `config/__init__.py:115-118, 371-374`; `tools/custom_example/custom_grpo_trainer.py:44-61` |
| Config tree, schema-hidden fields, `custom` escape hatch | `policy/config/__init__.py:1793-1819, 1970-1972` |
| GRPO/SFT type inferred from field shape (before-validator) | `policy/config/__init__.py:1840-1864` |
| Validators mutate (throttle clamps, distillation flags); from_dict timestamp | `policy/config/__init__.py:1822-1836, 1866-1967` |
| Controller-only TOML; runtime values written into config; workers fetch `/meta` | `dispatcher/run_web_panel.py:636-672, 162-167`; `dispatcher/controller.py:135-173`; `policy/policy_entry.py:32-39`; `rollout/worker/llm_worker.py:39-49` |
| Minimal CLI args; launcher argparse scope | `dispatcher/data/packer/base.py:30-44`; `launcher/launch_all.py:103-344` |
| Launch order: controller first, `wait_for_url_ready`, replica spawn | `launcher/launch_all.py:733-783, 845-885` |
| Register-before-model-build; check_config asserts | `policy/worker/base.py:34-113`; `comm/base.py:88-106, 249-303` |
| WeightResume/first-P2R triggered by registration | `dispatcher/status.py:384-485` |
| Rollout engine lazy init with auto/dummy load_format | `rollout_control.py:973-1017`; `vllm_rollout/vllm_rollout.py:164-196, 249, 298-304` |
| Heartbeat sweep → unregister ✔; rebuild mesh; rollout unregister path | `run_web_panel.py:104-124`; `status.py:228-239, 362-382, 487-495, 1573-1624, 1692-1700`; `utils/constant.py:30-35` |
| atexit unregister (rank 0) | `comm/base.py:303-315` |
| Launcher tolerates replica death, dies only with controller | `launcher/launch_all.py:899-933` |
| Training exception: re-raise, no retry; fake-step ACK | `policy/worker/base.py:115-135`; `rl_worker.py:526-553` |
| NCCL retry/watchdog/report; debounce sweep; inverted hang detection ✔; rollout NotImplemented | `utils/distributed.py:427-609`; `utils/pynccl.py:360`; `dispatcher/controller.py:609-668`; `constant.py:39-40` |
| HTTP exponential-backoff retry everywhere | `utils/network_util.py:55-95`; `constant.py:67-75` |
| Logger + COSMOS_LOG_LEVEL; --debug | `utils/logging.py:19-33`; `launch_all.py:398-399` |
| Step timing via CUDA events; MFU; report_data on ACK | `grpo_trainer.py:974-975, 1815, 1838-1877`; `utils/util.py:563`; `rl_worker.py:555-565` |
| Controller aggregates metrics; wandb/console/custom sinks; media logging | `dispatcher/status.py:1035-1264, 1458-1505`; `utils/report/wandb_logger.py:49-107`; `controller.py:109-110`; `config/__init__.py:1563-1597` |
| Remote-triggerable torch profiler, chrome traces, S3 upload | `utils/profiler.py:31-255`; `rl_worker.py:512-520, 555-563`; `status.py:1079-1082` |
| Live HTML status panel | `run_web_panel.py:129-158` |
