# Sprint: Overengineering Audit Cleanup

## Goal

Delete thin wrappers, hollow ABCs, delegation classes, and hidden side-effects identified in the 2026-05-23 audit. No behavior change. Fewer lines, fewer concepts.

---

## Tier 1 — Pure Delegation Classes (Highest ROI)

### Task 1 — Delete `ARPipelineExecutorBase`

**File:** `vrl/generation/ar/executor.py`

**Problem:** 11 methods, every one is `return self.layout.method(...)`. 100% passthrough to `ARRequestLayout`. The only method with real logic is `forward_batch_plan()`.

**Cut:**
- Delete `ARPipelineExecutorBase`
- Extract `forward_batch_plan()` as a standalone function
- Subclasses own their `ARRequestLayout` directly and call `self.layout.foo()` themselves

**Tests:** `tests/generation/ar/`, `tests/models/test_janus_kv_decode.py`, `tests/engine/generation/`

---

### Task 2 — Delete `RayGenerationWorker` class

**File:** `vrl/generation/ray/worker.py`

**Problem:** 7 methods, every one is `return self.core.method()`. The only unique behavior is `_ray_metadata()` which queries Ray for node IP + GPU IDs.

**Cut:**
- Delete `RayGenerationWorker` class
- Pass `_ray_metadata` logic as a lambda/closure to `GenerationWorkerCore`:
  ```python
  GenerationWorkerCore(worker_id, contract, metadata_provider=lambda: {
      "worker_id": worker_id,
      "node_ip": current_node_ip(),
      "gpu_ids": current_gpu_ids(),
  })
  ```
- Update `RayGenerationExecutor` / launcher to construct `GenerationWorkerCore` directly as a Ray actor

**Tests:** `tests/generation/ray/`

---

## Tier 2 — Hollow ABCs / Single-Implementor Protocols

### Task 3 — Delete `AttentionLayerBase`

**File:** `vrl/nn/layers/attention/base.py`

**Problem:** ABC with one abstract method: `debug_info()`. Zero structural value. `debug_info()` is a debugging helper, not an architectural boundary.

**Cut:**
- Delete `AttentionLayerBase`
- Call `.debug_info()` duck-typed at call sites (or drop if unused)
- Keep `AttentionCacheView` dataclass (it's fine, move to relevant file if needed)

**Tests:** `tests/nn/layers/`

---

### Task 4 — Convert `Algorithm` ABC to Protocol

**File:** `vrl/algorithms/base.py`

**Problem:** Two abstract methods, no shared code. Pure interface doc masquerading as ABC.

**Cut:**
- Replace `class Algorithm(ABC)` with `class Algorithm(Protocol)`
- Remove `@abstractmethod` decorators (Protocol uses structural typing)
- No behavior change, but honest about what it actually is

**Tests:** `tests/algorithms/`

---

### Task 5 — Convert `Trainer` ABC to Protocol

**File:** `vrl/trainers/core/base.py`

**Problem:** Same as `Algorithm`. Three abstract methods, nothing shared.

**Cut:**
- Replace `class Trainer(ABC)` with `class Trainer(Protocol)`

**Tests:** `tests/e2e/test_real_checkpoint_rl.py`, trainer tests

---

### Task 6 — Delete `ChunkGatherer` Protocol

**File:** `vrl/generation/protocols.py`

**Problem:** `PipelineExecutor` already requires `gather_chunks()` with the same signature. `ChunkGatherer` is a subset protocol with no callsite that uses the narrower type — every object implementing `ChunkGatherer` also implements `PipelineExecutor`.

**Cut:**
- Delete `ChunkGatherer` from `protocols.py`
- Replace any `ChunkGatherer` type hints with `PipelineExecutor`

**Tests:** `tests/architecture/test_generation_rollout_boundaries.py`

---

## Tier 3 — Files That Shouldn't Be Standalone Files

### Task 7 — Merge `generation/execution/ids.py` into `types.py`

**File:** `vrl/generation/execution/ids.py`

**Problem:** Single function `build_sample_rows()` in its own file with its own module docstring.

**Cut:**
- Move `build_sample_rows()` into `vrl/generation/execution/types.py`
- Delete `ids.py`
- Update all imports

**Tests:** `tests/engine/generation/`

---

## Tier 4 — Factory Abuse / Hidden Side-Effects

### Task 8 — Flatten reward registry

**File:** `vrl/rewards/functions/registry.py`

**Problem:** `register_reward()` and `get_reward()` are `dict.__setitem__` / `dict.__getitem__` wrappers. `_register_builtins()` is called inside `from_dict()` on every invocation for a fixed set of builtins.

**Cut:**
- Delete `register_reward()`, `get_reward()`, `_register_builtins()`
- Replace with a module-level dict literal populated at import time:
  ```python
  from vrl.rewards.functions.aesthetic import AestheticReward
  ...
  _REWARD_REGISTRY: dict[str, type[RewardFunction]] = {
      "aesthetic": AestheticReward,
      ...
  }
  ```
- `MultiReward.from_dict()` reads directly from `_REWARD_REGISTRY[name]`

**Tests:** `tests/rewards/test_multi.py`

---

### Task 9 — Remove hidden `batch.context` mutation from `build_rollout_iteration()`

**File:** `vrl/rollouts/orchestration/types.py:52-84`

**Problem:** `build_rollout_iteration()` mutates `batch.context` on each batch as a side effect while constructing a `RolloutIteration`. Callers don't expect their batch objects to be modified inside a factory function.

**Cut:**
- Remove the `batch.context` mutation from inside `build_rollout_iteration()`
- Move it to each call site explicitly (or add a separate `annotate_batches()` step)
- The function should only construct and return `RolloutIteration`

**Tests:** `tests/rollouts/test_orchestration.py`

---

### Task 10 — Fix silent exception swallow in `_launch_contract_policy_version()`

**File:** `vrl/generation/ray/runtime.py:148-157`

**Problem:**
```python
try:
    contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
except Exception:
    return None
```
Swallows all exceptions silently. If the contract is the wrong type, that should be caught explicitly.

**Cut:**
- Replace with explicit `isinstance` check:
  ```python
  if not isinstance(launch_contract, GenerationRuntimeLaunchContract):
      try:
          launch_contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
      except (TypeError, ValueError):
          return None
  ```

**Tests:** `tests/generation/ray/test_rollout_launcher.py`

---

## Tier 5 — Duck-Type Archaeology (Lower Priority)

### Task 11 — Fix `RolloutLifecycle._collector_runtime()` duck-typing

**File:** `vrl/rollouts/orchestration/lifecycle.py:115-119`

**Problem:**
```python
def _collector_runtime(self):
    try:
        return self.collector.runtime
    except Exception:
        return getattr(self.collector, "_runtime", None)
```
try/except + `_runtime` fallback means objects passed as `collector` don't agree on attribute names. `self.collector` is typed `Any`.

**Cut:**
- Define a `RolloutCollector` Protocol with a `.runtime` property
- Type `self.collector` as `RolloutCollector`
- Delete the try/except fallback

**Tests:** rollout orchestration tests

---

## Already in `SPRINT_over_engineering_cleanup.md` (Cross-reference)

These were already identified and should be tracked in that doc:
- Flatten reward registry (overlaps Task 8 above — consolidate)
- Remove `collect_policy_version()` from `RolloutLifecycle`
- Remove `RolloutScheduleMode.from_value()`
- Delete `CompositeReward`
- `GenerationIdFactory` → plain function
- Merge executor protocol hierarchy (`FamilyPipelineExecutor` / `ChunkedFamilyPipelineExecutor`)
- Delete empty marker protocol `PipelineChunkResult`

---

## Do Not Touch

| Abstraction | Why it stays |
|---|---|
| `RolloutLifecycle` | Real shared logic: offload, restore, weight sync, cache flush |
| `RayGenerationRuntime` + `ReleasableRayGenerationRuntime` | Genuinely divergent behavior (actor lifecycle) |
| `MultiReward` | Real state (`last_components`), real aggregation |
| `FamilyCapability` / capability system | Models genuinely differ in capabilities |
| `GenerationRuntime` Protocol | Real rollout/generation boundary |
| `RayGenerationExecutor` | Isolates Ray actor / distributed execution details |
| `ChunkGatherer` (if callers exist) | Re-verify before deleting |

---

## Acceptance Criteria

- Deleted abstractions don't appear in any public imports
- No compat aliases left behind
- All tests in `tests/` pass
- `grep -r "ARPipelineExecutorBase\|RayGenerationWorker\|AttentionLayerBase\|ChunkGatherer\|_register_builtins\|register_reward\|get_reward" vrl/` returns empty (or only definition sites being deleted)

**Full verification:**
```bash
python -m pytest -q \
  tests/algorithms/ \
  tests/generation/ \
  tests/engine/generation/ \
  tests/rewards/ \
  tests/rollouts/ \
  tests/nn/ \
  tests/architecture/test_generation_rollout_boundaries.py
```
