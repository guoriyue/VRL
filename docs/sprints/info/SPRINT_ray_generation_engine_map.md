# INFO: Ray generation engine — class roster + relationships

> Map of the Ray rollout/generation engine: who owns whom, the launch + generate
> call chains, and why there are so many things named `*Executor`. Verified by
> reading the source 2026-06-28 (file:line cited). Purpose: kill the "four layers
> all called execute" confusion and pin each class's real responsibility.

## TL;DR — 5 layers, each a real boundary (not duplication)

```
GenerationRuntime (protocol)  ── collector-facing: generate / update_weights / release / lifecycle
  └ RayGenerationRuntime          owns executor + weight_sync + worker actors; resident OR release-per-collect
      └ RayGenerationExecutor     DRIVER scheduler: plan chunks → dispatch → OOM-degrade → version-check → gather
          └ RayGenerationWorker   thin Ray ACTOR adapter (+ GPU/node metadata); delegates to core
              └ GenerationWorkerCore   Ray-INDEPENDENT exec: load_policy + version-slot safety + forward + _to_cpu
                  └ GenerationChunkExecutor (protocol)  MODEL-family contract: forward_chunk_plan + gather_chunks
                        └ DiffusionChunkExecutorBase / AR executor   the actual encode→denoise→decode
```
Different processes: Runtime/Executor live on the **driver**; Worker/Core/model-executor live in each **Ray actor**. That process split is why they can't be merged.

## The two call chains (verified)

**① Launch (build the engine) — `RayGenerationLauncher.launch` (launcher.py:51):**
```
config(RayGenerationConfig) + GenerationRuntimeLaunchContract(family, model_build, executor_cls,
       runtime_builder) + ChunkGatherer + RolePlacement
 → RayActorGroup.launch(worker_cls=RayGenerationWorker, startup_method="load_policy")  (launcher.py:87)
       each actor: load_policy → GenerationWorkerCore._build_executor (worker.py:360)
                   → build_runtime_bundle(build) → model + executor_cls(model)  = the GenerationChunkExecutor impl
 → wrap actors in DistributedWorkerHandle
 → RayGenerationExecutor(planner=DistributedExecutionPlanner, workers, gatherer, pipelined)  (launcher.py:127)
 → RayGenerationWeightSync(workers)  (launcher.py:139)
 → RayGenerationRuntime(executor, weight_sync, owned_workers, colocated)  (launcher.py:144)
 → return runtime
```

**② Generate (one rollout request):**
```
collector → runtime.generate(request)                                   (runtime.py:97)
  → stamps request.policy_version if missing → executor.execute(request)
RayGenerationExecutor.execute(request)                                   (executor.py:48)
  → DistributedExecutionPlanner.plan_with_engine → chunks + per-worker assignments
  → IF pipelined and len(workers)==1:  _execute_request_pipelined → worker.execute_request_pipelined (single-worker path)
     ELSE:                              per-chunk RayActorJob → run_actor_jobs → worker.execute_chunk.remote (multi-worker data-parallel)
  → StaleSlotDiscard check → _degrade_oom_chunks → policy_version assert
  → gatherer.gather_chunks(...)  → GenerationOutput
RayGenerationWorker.execute_chunk(envelope)  (actor adapter)             (worker.py / ray:57)
  → GenerationWorkerCore.execute_chunk(envelope)                         (execution/worker.py:147)
       load_policy → version-slot safety (has_trainable_state / activate_trainable_state /
                     StaleSlotDiscard / version mismatch) → _profile_forward_chunk → _to_cpu(output)
       _profile_forward_chunk → executor.forward_chunk_plan(request, chunk, stage)  ← the MODEL executor
DiffusionChunkExecutorBase.forward_chunk_plan  (the GenerationChunkExecutor impl)
  → build_prompt_stage_input → run_prompt_encode_stage → run_prepare_stage → run_denoise_stage → run_decode_stage
```

**③ Weight sync:** `runtime.update_weights` (runtime.py:116) → `RayGenerationWeightSync.push_to_rollout_workers` → each `worker.update_weights` → `core.update_weights` (load state into model/slot).

## Per-class responsibility (and why each stays)

| class | file | role | merge it? |
|---|---|---|---|
| `GenerationRuntime` (Protocol) | protocols.py:45 | collector-facing contract (generate/update_weights/release/shutdown) | — contract |
| `RayGenerationRuntime` | ray/runtime.py:33 | owns executor+weight_sync+actors; **lifecycle** (resident vs release-per-collect via `_RuntimeLease`); stamps policy_version; non-draining-sync flag | keep — lifecycle owner |
| `RayGenerationExecutor` | ray/executor.py:24 | **driver scheduler**: chunk plan, dispatch (per-chunk OR per-request-pipelined), OOM-split, version assert, gather. Holds NO model | keep — driver side |
| `RayGenerationWorker` | ray/worker.py:17 | **Ray actor adapter** (thin): execute_chunk / execute_request_pipelined / load_policy / update_weights → delegates to core; carries GPU/node metadata | keep thin — Ray needs an actor class; putting logic here forces Ray-in-tests |
| `GenerationWorkerCore` | execution/worker.py:28 | **Ray-independent** worker exec: load/release policy, weight sync, **version-slot safety**, profiler, `_to_cpu`. Builds + owns the model executor | keep — testable without Ray; owns version safety |
| `GenerationChunkExecutor` (Protocol) | protocols.py:76 | **MODEL-family contract**: `forward_chunk_plan` (run one chunk) + `gather_chunks`. `@runtime_checkable` | keep — cross-family contract (diffusion+ar), runtime isinstance-checked by `_require_chunked_executor` (worker.py:532) |
| `DiffusionChunkExecutorBase` / AR executor | diffusion/executor.py:336, ar/executor.py | the **actual** per-chunk forward (encode→denoise→decode) + gather | keep — the model work |

**Support cast:** `DistributedExecutionPlanner` + `ChunkPlacementPolicy` (chunk→worker placement), `ChunkGatherer` (Protocol, driver-side assembly — separate from the executor's own gather_chunks), `RayGenerationWeightSync` (weight push), `run_actor_jobs` (bounded multi-worker dispatch, the data-parallel distributor), `RayGenerationLauncher` + `GenerationRuntimeLaunchContract` + `RayGenerationLaunchInputs` (assembly), `_RuntimeLease` (release-per-collect state).

## Why "too many Executor" feels confusing (the real naming wart)

```
RayGenerationExecutor   = DRIVER scheduler (orchestrates chunks across workers)   ← a RUNTIME executor
GenerationChunkExecutor = MODEL-family contract (forward_chunk_plan + gather)      ← NOT a runtime executor; it's the model contract
```
They share the word "Executor" but live on opposite sides (driver scheduler vs model forward contract). That's the cognitive load. `GenerationChunkExecutor` is **load-bearing** (2 implementers diffusion+ar, the worker's `executor` type, the `_require_chunked_executor` runtime check, a test) — **cannot be deleted** without losing the cross-family contract + runtime validation. The only clean option is a **rename** (e.g. `GenerationModelExecutor` / `ChunkForwarder`) to stop it colliding with the driver `RayGenerationExecutor` — pure taste, ~8 files. Decision pending; not a correctness issue.

## Minor known leak

`forward_plan_pipelined` / `execute_request_pipelined` (the single-worker chunk-pipeline) are **not** in the `GenerationChunkExecutor` protocol — the executor reads them via `getattr`. That is defensible: the pipelined path is **diffusion-only + single-worker-gated** (`len(workers)==1`, executor.py:63), so it is an optional capability, not a cross-family contract method. Forcing it into the shared protocol would require AR to implement it.

## One-line mental model

```
Runtime = lifecycle owner (driver) → Executor = chunk scheduler (driver) → Worker = Ray actor shell
→ Core = version-safe execution (actor) → ChunkExecutor = model forward+gather contract (actor)
跨 family 靠 ChunkExecutor protocol;跨进程靠 Runtime/Executor(driver) vs Worker/Core(actor) 的分界。
```
