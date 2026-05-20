# Sprint: Codex Over-Engineering Cleanup

**Goal**: Remove abstraction layers, dead code, and single-path factories that Codex introduced
but provide no real value. No behavior changes — pure structural cleanup.

**Rule**: Only delete/simplify things where there is **one concrete path** and no realistic
second path in the near term. Legitimate DRY (`ARPipelineExecutorBase`, `FamilyCapability`)
is not touched.

---

## Phase 1 — Dead Code (zero risk)

### 1.1 Delete `CompositeReward`

**File**: `vrl/rewards/composite.py`, `vrl/rewards/__init__.py`

`CompositeReward` duplicates `MultiReward`'s weighted-sum logic and is **never instantiated**
anywhere in the codebase. Only exists in `__init__.py` export.

- Delete `vrl/rewards/composite.py`
- Remove import + `__all__` entry from `vrl/rewards/__init__.py`

---

## Phase 2 — Class-to-Function (`GenerationIdFactory`)

**File**: `vrl/generation/execution/ids.py`, `vrl/generation/ray/executor.py`

`GenerationIdFactory` is a class with one public method and no state. It's always constructed
as `GenerationIdFactory()` (no args) and immediately called. There's nothing to inject or
swap — it's a function dressed as a class.

**Change**:
- Convert `GenerationIdFactory.build_sample_rows` to a module-level function
  `build_sample_rows(request: GenerationRequest) -> list[GenerationSampleRow]`
- In `executor.py`: replace `self.id_factory.build_sample_rows(request)` →
  `build_sample_rows(request)`, remove `id_factory` field
- Update `__init__.py` exports in `execution/` and `generation/`

**Keep `GenerationIdFactory` as a deprecated alias** if there are external callers:
```python
class GenerationIdFactory:  # backward compat shim, remove after cleanup
    def build_sample_rows(self, request):
        return build_sample_rows(request)
```
Actually no — check grep, there are no external callers outside this repo. Delete directly.

---

## Phase 3 — Flatten Protocol Hierarchy (`FamilyPipelineExecutor`)

**File**: `vrl/generation/protocols.py`

`FamilyPipelineExecutor` is the base protocol with `family`, `task`, `workload_signature`.
`ChunkedFamilyPipelineExecutor` extends it adding `forward_chunk_plan` and `gather_chunks`.

**Problem**: `FamilyPipelineExecutor` alone is **never used as a type annotation** anywhere
in the codebase. Every usage is `ChunkedFamilyPipelineExecutor`. The two-level hierarchy
is hypothetical future-proofing for a non-chunked executor that doesn't exist.

**Change**:
- Merge `FamilyPipelineExecutor` fields/methods directly into `ChunkedFamilyPipelineExecutor`
- Remove `FamilyPipelineExecutor` class
- Update `__all__` and `generation/__init__.py` exports

```python
# Before: two classes
class FamilyPipelineExecutor(Protocol):
    family: str
    task: str
    def workload_signature(...) -> WorkloadSignature: ...

class ChunkedFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    def forward_chunk_plan(...) -> PipelineChunkResult: ...
    def gather_chunks(...) -> GenerationOutput: ...

# After: one class
class PipelineExecutor(Protocol):
    family: str
    task: str
    def workload_signature(...) -> WorkloadSignature: ...
    def forward_chunk_plan(...) -> PipelineChunkResult: ...
    def gather_chunks(...) -> GenerationOutput: ...
```

Rename to `PipelineExecutor` to drop the redundant "Chunked/Family" prefix noise. Update
all callsites (worker.py, executor.py, diffusion/executor.py, ar/executor.py).

---

## Phase 4 — Replace Empty Protocol Marker (`PipelineChunkResult`)

**File**: `vrl/generation/protocols.py`, all callsites

`PipelineChunkResult` is an empty `Protocol` with no methods and no properties. It's
a marker type used as `PipelineChunkResult | None` or `Sequence[PipelineChunkResult]`.
Since it has no structural constraints, it provides no type safety — mypy/pyright will
accept any object.

**Change**:
```python
# Replace
class PipelineChunkResult(Protocol):
    """Family-specific chunk payload returned before final GenerationOutput gather."""

# With
type ChunkResult = Any  # family-specific opaque payload
```

Or simply use `Any` inline at callsites. Either way, remove the empty Protocol class.

**Callsites to update**:
- `vrl/generation/execution/types.py:56` — `output: ChunkResult | None`
- `vrl/generation/diffusion/gather.py:33` — `chunks: Sequence[ChunkResult]`
- `vrl/generation/ar/layout.py` — same
- `vrl/generation/ray/executor.py:111` — `chunk_outputs: list[ChunkResult]`
- `vrl/models/ar/janus_pro/runtime.py`, `nextstep_1/runtime.py` — return type annotations

---

## Phase 5 — Inline Single-Path Reward Factory

**File**: `vrl/rewards/inference.py`, `vrl/rewards/ray/launcher.py`, `vrl/rewards/video_reward.py`

`build_reward_inference_runtime` in `inference.py` validates `inference_runtime == "ray"`
then immediately delegates to `build_reward_ray_runtime`. Zero branching. Only caller is
`video_reward.py:64`.

`rewards/ray/launcher.py` (`build_reward_ray_runtime`) in turn just constructs
`RewardInferenceActorRuntime(RayActorMethodRuntime(...))` — the real logic.

**Change**:
- Inline `build_reward_ray_runtime` body into `build_reward_inference_runtime` in `inference.py`
- Delete `vrl/rewards/ray/launcher.py`
- Remove the intermediate `build_reward_ray_runtime` import hop
- `vrl/rewards/ray/` keeps only `runtime.py` (`RewardInferenceActorRuntime`)

```python
# inference.py — after merge
def build_reward_inference_runtime(cfg, *, init_ray=True, ray_init_kwargs=None):
    runtime = str(cfg.get("inference_runtime", ""))
    if runtime != "ray":
        raise ValueError("reward inference_runtime must be 'ray'")
    worker_config = cfg.get("worker_config")
    if worker_config is None:
        raise ValueError("...")
    if not isinstance(worker_config, Mapping):
        raise TypeError("...")
    from vrl.rewards.ray.runtime import RewardInferenceActorRuntime
    from vrl.ray.runtime import RayActorMethodRuntime
    return RewardInferenceActorRuntime(RayActorMethodRuntime(
        worker_cls=RewardScoringWorker,
        ...
    ))
```

---

## Explicitly NOT Cleaned Up (Legitimate Abstractions)

| Symbol | Reason to keep |
|--------|---------------|
| `ARPipelineExecutorBase` | Real DRY over JanusPro + NextStep — both delegate to `self.layout`, saves ~15 repeated methods |
| `FamilyCapability` system | Drives runtime planning (chunk schedule, axes, stages) — not static config |
| `ChunkGatherer` Protocol | Used separately from `PipelineExecutor` (diffusion gather is decoupled from executor) |
| `RolloutConfig` wrapper | Typed boundary passed across collector/script layers — dict-wrapping is thin but callers benefit from named type |
| `RayGenerationExecutor` | Genuinely complex chunk dispatch — not thin |
| `GenerationRuntime` Protocol | Two impls possible (local + Ray); runtime boundary worth naming |

---

## Acceptance Criteria

- [ ] `CompositeReward` deleted; no import errors
- [ ] `GenerationIdFactory` replaced with `build_sample_rows` function; `id_factory` field removed from `RayGenerationExecutor`
- [ ] `FamilyPipelineExecutor` removed; `ChunkedFamilyPipelineExecutor` renamed to `PipelineExecutor`
- [ ] `PipelineChunkResult` empty protocol removed; callsites use `Any` or type alias
- [ ] `rewards/ray/launcher.py` deleted; `build_reward_inference_runtime` inlined
- [ ] All tests pass: `pytest tests/ -x -q`
- [ ] No new `TYPE_CHECKING` guards introduced (don't paper over import errors)
