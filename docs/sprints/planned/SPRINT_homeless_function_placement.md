# SPRINT: Homeless-function placement — giving shared helpers a subject

> **来源**：2026-07-25 的全仓 homeless-function 审计（6 域猎手 × 逐条对抗验证，
> 38 条候选 → 31 条存活 / 7 条驳回）。它是 `SPRINT_config_resolution_consolidation.md`
> 那轮去重 sweep 的**续集**：上一轮把重复的函数体合并成共享自由函数是对的，但没有
> 把它们放回各自的主语类型上，于是 `self` 以三种形态回来了——`owner=` 字符串、
> `Any` 参数、N 份兄弟子类里的同形包装。
>
> **四条放置规则已固化到 AGENTS.md**（`### Placement — where a shared helper belongs`），
> 那是本 sprint 最重要的产出；下面的 10 组重构是它的具体应用。
>
> **非目标清单（第 332 行起）同样是产出的一部分**：那 7 条经对抗验证判定必须保持自由
> 函数，理由都写在表里。后续 sweep 不要再"清理"它们。

## 执行状态

| 组 | 内容 | 状态 |
|---|---|---|
| 前置 | `models/utils.py` 拆分（adapter → `peft_adapter.py`；权重 → 新模块）、`utils/validation.py` 归 trajectory | 执行中（tranche A） |
| 1 | denoise 四个 opt-in mixin | 执行中（tranche A） |
| 2 | `DiffusionModelBase`/`DiffusersPipelineModelBase` 吸收 `from_build` 等 | 执行中（tranche A） |
| 3 | `ARModelBase` + 新 `ARReplayCore` | 执行中（tranche A） |
| 4-10 | executor 基类 / `ray/utils.py` 解散 / `ModelBuild`+`RuntimeBundle` / `ray/resources.py` / `rollouts/` / `trainers/` / `nn/` | 待执行（tranche B） |

---

# Placement plan — giving the homeless functions a subject

**The one question this plan answers:** every function below was hoisted out of a type during a dedup sweep, and the `self` it lost came back as a string, an `Any`, or N copies. This says which type takes it back.

**The one sentence to internalize:** *if a parameter exists only to tell the function who called it, what type its own argument really is, or which constant this copy uses, that parameter is a `self` that was dropped on the floor — put the body on the thing that was passing it.*

---

## Merge notes and adjudications (read before executing)

Two defects were reported twice under different ids. They are one change each:

| Merged | Adjudication |
|---|---|
| `ar-replay-owner-threading` + `resolve-image-token-replay-owner` | Method is **private**: `ARModelBase._resolve_image_token_replay`. The diffusion twin is already private — `DiffusionModelBase._replay_inputs_for_step` (`vrl/models/steps/denoise/base.py:246`) — and this helper is not on the `ReplayModel` protocol face (`replay_forward` / `disable_adapter` only). A public name would hand every AR family a new outward method for nothing. |
| `ar-replay-core-loader` + `load-replay-core-owner-and-core-cls` | `ARReplayCore` lands in **`vrl/models/steps/token/base.py`**, not `loader.py`. `base.py` already imports `torch.nn` (`:23-24`) and is the AR type-hierarchy module (`ARModelBase:56`, `ARReplayRolloutStubs:113`); `loader.py`'s module scope imports only `typing` (`:15-17`) and should keep that property. `base.py → loader.py` is acyclic (loader imports nothing from `vrl`). The owner string is derived as `type(self).__name__` (→ `"Emu3ReplayCore"`), **not** a `checkpoint_name = "Emu3"` ClassVar — a hand-written display literal on the class is the same defect one level up, and `ARModelBase.load_trainable_state` (`base.py:70`) and `ARReplayRolloutStubs` (`base.py:123`) already use `type(self).__name__` for exactly this. `checkpoint_subfolder` stays a ClassVar (a real per-family checkpoint layout fact, `glm_image/model.py:87`). |

---

## 1. Four opt-in mixins in `vrl/models/steps/denoise/common/`

**Value: highest (largest verified deletion, one sub-item already implemented and green). Risk: lowest (opt-in; a wrong bases list fails at class-creation time, not at runtime).**

**Destination:** `vrl/models/steps/denoise/common/` — already the named owner of `ChunkedLatentDecoder`/`LatentDecodePlan`, `DiffusionBackboneRunnerBase`, `combine_cfg`; every participating family already imports the package `__init__`.

| Moves | From | To |
|---|---|---|
| `decode_latents` ×6 | sd3_5:381, pixart_sigma:397, lumina2:347, hunyuan_image:413, sana:391, hunyuan_video:296 | new `VaeDecodeMixin` in `common/latent_decode.py`, class attr `_decode_output_layout: LatentOutputLayout = "image_bchw"` (hunyuan_video sets `"video_btchw"`) |
| `SamplingState` fields + `export_replay_tensors` + `export_batch_context` + `restore_eval_state` ×4 | sana:58/342/349/364, lumina2:61/300/308/319, mochi:80/303/311/322, pixart_sigma:101/348/355/370 | new `common/masked_prompt.py`: `MaskedPromptSamplingState`, `TrainTimestepMaskedPromptSamplingState` (lumina2+mochi), `MaskedPromptCollectorMixin` with `sampling_state_cls` |
| `build_branch` ×4 | sana:80, lumina2:78, mochi:97, pixart_sigma:123 | `EncoderAttentionMaskRunnerBase(DiffusionBackboneRunnerBase)` in `common/backbone.py`, class attr `branch_extra_kwargs: dict = {}` (pixart sets `{"added_cond_kwargs": _ADDED_COND_KWARGS}`) |
| `finalize_noise_pred` ×2 | lumina2:115, qwen_image:111 | folded into `combine_cfg(..., normalize: bool = False)` in `common/cfg.py`; selected by `cfg_normalization: bool` on `DiffusionBackboneRunner`/`Base`, read at `common/backbone.py:153` via `getattr(self.runner, "cfg_normalization", False)` next to the existing `base=getattr(...)` |

**Stays free / stays overridden:** `combine_cfg`, `pack_batched_cfg`, `require_tensor`, `pack_eval_timestep` — pure tensor math, no subject. `decode_latents` stays overridden in flux/qwen_image (latent unpacking), mochi/cogvideox (`latents_mean/std` + permute), cosmos/wan/echo/causvid. `DiffusionBackboneRunnerBase` gets **no** `build_branch` default — its docstring's fail-loud requirement is correct; the new class is a sibling opt-in.

**Call-site shape:**
```python
# before
class SanaModel(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    def build_branch(self, request, branch): ...        # 21 lines
    def decode_latents(self, latents): ...              # 18 lines
    def export_replay_tensors(self, state): ...         # 14 lines
# after
class SanaModel(VaeDecodeMixin, MaskedPromptCollectorMixin, LoraModelMixin,
                DiffusersPipelineModelBase, EncoderAttentionMaskRunnerBase):
    sampling_state_cls = SanaSamplingState
    cfg_normalization = False        # explicit; the comment explaining WHY lives here
```

**Load-bearing detail:** `decode_latents` is `@abstractmethod` on `DiffusionModelBase` (`denoise/base.py:135`), so `VaeDecodeMixin` **must precede** `DiffusersPipelineModelBase` in the bases list, exactly as `LoraModelMixin` already precedes it to beat `apply_lora` at `base.py:385`. Verified: mixin listed last → `__abstractmethods__ == {'decode_latents'}` → `TypeError` at instantiation. Verified safe for the replay subclasses: `SD3_5ReplayModel(DiffusersReplayModelBase, SD3_5Model)` still resolves `decode_latents` to `ReplayRolloutStubs`'s raiser.

Export all four new names from `vrl/models/steps/denoise/common/__init__.py` (families import the package, never the submodule).

**Verify:** `.venv/bin/python -m pytest tests/models/steps/denoise tests/models/families -q` (denoise baseline 144 passed) then `.venv/bin/ruff check <touched> && .venv/bin/ruff format --check <touched>`

**Estimate:** ≈ **-320 net lines**, ~13 source files + ~6 test files. (The masked-prompt sub-item alone was implemented in a worktree and measured **-234 / +60**, full `tests/models`+`tests/generation`+`tests/rollouts`+`tests/trainers` green.)

---

## 2. `DiffusionModelBase` / `DiffusersPipelineModelBase` absorb what every diffusion model has

**Value: very high (9 families × a 21-36 line classmethod). Risk: medium — one real trap, already located.**

**Destination:** `vrl/models/steps/denoise/base.py` — `DiffusionModelBase:72` and `DiffusersPipelineModelBase:533`, which already declare these names as `NotImplementedError` stubs.

**Moves down to `DiffusionModelBase` (from `DiffusersPipelineModelBase:586-602`, replacing the stubs at `:388`/`:435`):**
- `trainable_modules` → `{"transformer": self._require_transformer()}` (`_require_transformer` exists at `:269`, preserves fail-loud)
- `apply_full_finetune`
- new default `_set_transformer(self, transformer): self.transformer = transformer` — closing a real hole: `DiffusionModelBase.torch_compile_transformer` (`:397`) and `LoraModelMixin.apply_lora` (`common/lora.py:96,109`) already call a method the class never defined.

Then delete `trainable_modules` from `echo/model.py:151`, `causvid/model.py:367`, `cosmos/anima/model.py:191`; `apply_full_finetune` from `echo:174`, `causvid:388`; `_set_transformer` from `cosmos/anima:206`.

**Concrete `from_build` on `DiffusersPipelineModelBase`,** driven by three declarations:
```python
_pipeline_classname: str = ""            # resolved via getattr(diffusers, name) — the mechanism
                                         # vrl/models/loader.py:35-38 already uses for transformer_classname
_frozen_encoder_names: tuple[str, ...] = ("text_encoder",)
_prompt_encoder_on_cpu: bool             # every collapsing family declares it EXPLICITLY, even when
                                         # it equals the default — the memory-decision comment needs an anchor
```
Collapses **9** families, not the 5 originally proposed: mochi:132, cogvideox:156, lumina2:133, pixart_sigma:169, qwen_image:147, plus flux, hunyuan_image, hunyuan_video, sd3_5 once the encoder list is a tuple. flux keeps a 3-line override (`nft_previous_adapter` guard, `model.py:141-146`, then `return super().from_build(build)`); sana keeps a full override (scheduler swap + fp16 clamp before `cls(...)`).

**⚠ THE TRAP — do not delete these `_set_transformer` overrides:**
`WanT2VReplayModel` (`wan_2_1/model.py:881`) and `WanI2VReplayModel` (`:1215`) are `(ReplayRolloutStubs, Wan*DiffusersModel)`, so they sit **below** `DiffusersPipelineModelBase` in the MRO. Removing the override resolves to the pipeline-syncing version, whose `self.pipeline` raises. Executed and confirmed: `RuntimeError: WanT2VReplayModel does not own a diffusers pipeline`. This is a live FSDP/DDP path (`vrl/trainers/strategy.py:507` → `:929`/`:590`). Same reason keeps `DiffusersReplayModelBase._set_transformer` (`base.py:660`). Also keep: wan `trainable_modules`(:333)/`apply_full_finetune`(:227), predict2_5 `apply_full_finetune`(:297), anima `apply_full_finetune`(:182, real `dtype=self._dtype`), and the family `_set_transformer` in echo/causvid/sd3_5.

**One behavior note to state in the commit:** sd3_5 currently uses direct attribute access `pipeline.text_encoder`; the shared loop uses `getattr(..., None)`, so a missing encoder becomes a silent skip instead of `AttributeError`. Acceptable (the SD3 pipeline always has all three), but name it.

**Also delete while here:** the stale comment at `sd3_5/model.py:148-149` referencing a `getattr`/`memory` fallback that no longer exists in that body.

**Verify:** `.venv/bin/python -m pytest tests/models -q` plus an MRO assertion per family and per `*ReplayModel`.

**Estimate:** ≈ **-190 net lines**, ~11 files.

---

## 3. `ARModelBase` and a new `ARReplayCore` in `vrl/models/steps/token/base.py`

**Value: high (5 AR families × replay prologue, 3 × checkpoint loader). Risk: low (purely additive to a non-ABC base).**

**Destination:** `vrl/models/steps/token/base.py` — `ARModelBase:56` and a new `ARReplayCore(nn.Module)` sibling of `ARReplayRolloutStubs:113`.

**3a. `resolve_image_token_replay` → `ARModelBase._resolve_image_token_replay`.** Delete the free function (`vrl/models/interfaces/replay.py:161-186`), its `__all__` entry (`:277`) and the re-exports at `vrl/models/interfaces/__init__.py:14,41`. The five families become one line:
```python
replay, image_token_ids = self._resolve_image_token_replay(batch, timestep_idx, request)
```
at `janus_pro/model.py:382`, `emu3:488`, `glm_image:539`, `llamagen:337`, `nextstep_1:243` — each of which today passes the identical `owner=type(self).__name__`.

**Stays free in `interfaces/replay.py`:** `require_zero_replay_timestep` and `require_replay_segments` — real non-AR callers exist: `causvid/model.py:447-448` (a `DiffusionModelBase` subclass) calls both, `janus_pro/model.py:361,366` calls both in its R1 branch, `denoise/base.py:175` calls one, and `tests/models/interfaces/test_replay_model_contract.py:116-140` unit-tests them with `owner="test"`. `single_segment_result` stays (pure wrapper, no subject).

**Required test edit:** `tests/models/interfaces/test_replay_model_contract.py:143-156` passes a bare `object()` as `self`. Replace with `model = replay_cls.__new__(replay_cls)` — this preserves what the test actually asserts (the guards fire before any model state is touched) and keeps the substring matches valid.

**3b. `load_replay_core` + `load_replay_core_checkpoint` → `ARReplayCore`:**
```python
class ARReplayCore(nn.Module):
    checkpoint_subfolder: ClassVar[str | None] = None

    def __init__(self, config: Any) -> None:      # all three cores do exactly this today
        super().__init__()                        # emu3:584, janus_pro:973, glm_image:707
        self.config = config

    @classmethod
    def from_pretrained(cls, model_path, *, device, dtype,
                        revision=None, trust_remote_code=False) -> "ARReplayCore": ...
    def _load_checkpoint(self, checkpoint_dir: str) -> None: ...   # owner = type(self).__name__
```
`Emu3ReplayCore` (`emu3/model.py:572`), `GlmImageReplayCore` (`glm_image:695`, sets `checkpoint_subfolder`), `JanusProReplayCore` (`janus_pro:965`) subclass it and change to `super().__init__(config)`. The three single-caller `_load_*_replay_core_from_pretrained` wrappers collapse to `Emu3ReplayCore.from_pretrained(config.model_path, device=..., dtype=...)` at `emu3:614`, `glm_image:735`, `janus_pro:1012`. Janus keeps its `janus` package pre-flight in its wrapper (`janus_pro/model.py:1095-1101`) — **not** in `__init__`, because `tests/models/steps/token/test_training_capability.py:235` injects a stub core in an environment with no `janus` installed.

**No `trust_remote_code` class attr:** it is a user-facing schema key (`JanusProModelSection.trust_remote_code`, `janus_pro/config.py:20`, parametrized True/False at `tests/config/test_schema.py:559`), dead for emu3/glm and always overridden for janus. Keyword only.

**Stays free in `loader.py`:** `resolve_hf_checkpoint_dir` — genuinely subject-less, and `nextstep_1/model.py:115,121,406` uses it with no replay core in sight (monkeypatched at `tests/models/families/nextstep_1/test_model_loading.py:25,65` through the `loader` module object, so do **not** relocate it). **Rewrite `loader.py`'s module docstring** (`:1-13`) — it describes the loading logic that just left.

**Ride along in the same commit (already decided, same destination):** `ARModelBase` absorbing `_lm_trunk` (base default + class attr) and `require_module_attrs` (`token/base.py:33`, whose `owner=` is the same defect).

**Verify:** `.venv/bin/python -m pytest tests/models/steps/token tests/models/interfaces tests/models/families/{emu3,glm_image,janus_pro,llamagen,nextstep_1} -q`

**Estimate:** ≈ **-110 net lines**, ~12 files.

---

## 4. The chunk-executor bases own construction, gathering, and runner selection

**Value: high (10 constructors + 5 gather wrappers + 3 forward_plan copies). Risk: low-medium (test fakes break loudly).**

**Destinations:**
- `vrl/generation/bindings/full_sequence_denoise/executor.py :: DiffusionChunkExecutorBase` (`:173`)
- `vrl/generation/bindings/chunk_autoregressive_denoise/executor.py :: ChunkAutoregressiveDenoiseExecutorBase` (`:94`)
- `vrl/generation/bindings/token_autoregressive/executor.py :: ARChunkExecutorBase` (`:21`)
- new `vrl/generation/execution/executor_base.py :: ChunkExecutorBase(GenerationChunkExecutor)` for the shared `gather_chunks`/`forward_plan`
- `vrl/utils/cuda_memory.py` + `vrl/generation/execution/chunk_placement.py` for the CUDA peak readers

**4a. Constructors.** Both diffusion bases get
```python
def __init__(self, model: Any, *, samples_per_chunk: int | None = None) -> None:
    self.model = model
    if samples_per_chunk is not None:
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))
```
and `ARChunkExecutorBase` gets `def __init__(self, model: Any) -> None: self.model = model`. Deletes 10 constructors: wan_2_1/runtime.py:29, predict2:31, predict2_5:30, echo:106 (diffusion), causvid:22 (→ the **chunk-autoregressive** base, not the diffusion one), janus_pro:99, emu3:64, glm_image:69, llamagen:101, nextstep_1:99 (AR). predict2 expresses its 8 as a class attr. `cosmos3/runtime.py:62` shrinks to `del samples_per_chunk; super().__init__(model)`; magi_1 keeps its validating ctor with `super().__init__(model)`. The keyword must stay keyword-only and optional — `worker.py:695` constructs by dotted string as `executor_cls(model, **executor_kwargs)` and `samples_per_chunk` is injected only when the registry declares `accepts_samples_per_chunk` (`ray/launcher.py:312-319`).

**Do NOT** add `__init__` to `ARDiscreteTokenRunner` (`models/steps/token/base.py:140`) — it neither declares nor reads `model`; adding a field it never uses to save two lines is the inverse defect. `glm_image/runner.py:73`, `llamagen/runner.py:54`, `PagedCFGTokenRunner` stay as-is.

**Before deleting the AR constructors,** move the `model` duck-type contract lists out of the janus_pro/emu3/glm_image `__init__` docstrings into their class docstrings — that is the only record of each family's model interface.

**4b. `gather_chunks` / `forward_plan`.** `ChunkExecutorBase` holds one concrete `gather_chunks` and one concrete `forward_plan` (the `run_sample_chunks_with_oom_retry` + gather pair). The three binding bases change their base from `GenerationChunkExecutor` to `ChunkExecutorBase` (one line each); nextstep_1/janus_pro executors follow automatically. `vrl/generation/protocols.py` stays a pure contract — `GenerationChunkExecutor` keeps exactly the two members `_require_chunked_executor` (`worker.py:758`) checks.

The gatherer binding must end up with **one** construction site. The registry already owns it (`vrl/families/registry.py:156 gatherer_cls`, `:408-413 new_gatherer`, consumed at `ray/launcher.py:276`), so the base default should read the registry: `FAMILY_REGISTRY[self.family].new_gatherer().gather_chunks(...)`. Verified acyclic (registry imports nothing from `vrl.generation`) and all five gatherers are zero-arg constructible. *If* you prefer a `gatherer_cls` class attribute instead, you must delete `ModelFamilyEntry.gatherer_cls` in the same pass and reorder the classes (the gatherers are defined **after** the executors — `token_autoregressive/executor.py` executor:209/gatherer:313 — so a class-body reference NameErrors at import).

Do not merge the three `plan()` facades — their bodies genuinely differ, and `docs/sprints/done/SPRINT_runtime_payload_smallest_truth.md:41` records the decision to keep them. Do not delete executor-level `gather_chunks` access: `forward_plan_pipelined` (`full_sequence_denoise/executor.py:295`, live via `worker.py:553`) depends on it.

**4c. Native-runner override.** Add `_native_runner_reason: str | None = None` to `ARChunkExecutorBase` beside `_runner_cls`/`_runner_attention_family` (`:44-45`); build the message as `f"{self.family} does not support request.sampling.attention_backend={backend!r}: {self._native_runner_reason}"` so the text stays byte-identical. Delete the overrides at `glm_image/runtime.py:82` and `llamagen/runtime.py:146`. Split the existing guard in two (`_runner_cls` required always; `_runner_attention_family` required only after the native branch returns). Repoint the two now-stale docstrings at `glm_image/runner.py:19` and `llamagen/runner.py:9`.

**4d. CUDA peak readers** (same files, same pass). Add `cuda_peak_allocated_bytes()`, `cuda_peak_allocated_mb()`, `reset_cuda_peak()` to `vrl/utils/cuda_memory.py` (function-local `import torch`, matching that module's torch-free module scope). Delete `ARRequestLayout.peak_memory_mb` (`token_autoregressive/layout.py:187-196`) and `reset_cuda_peak`/`cuda_phase_peak_bytes` (`steps/denoise/loop.py:51-67`) plus their `__all__` entries; the hand-written `/ (1024 * 1024)` at `full_sequence_denoise/executor.py:557` and `loop.py:275` collapses to the shared MB reader. Move `_cuda_occupancy_snapshot` (`loop.py:70-82`) to `vrl/generation/execution/chunk_placement.py` beside `ChunkMemoryReading`, deriving its key names from `fields(cls)` rather than re-listing them — those four keys are the byte-admission telemetry contract (`chunk_placement.py:115-120`), not general CUDA semantics.

**Test fakes that must be fixed in the same commit** (they break loudly): `tests/generation/steps/denoise/test_preallocation.py` `_Executor`/`_StageTrackingExecutor`/`_UnitVideoExecutor` → `def __init__(self): super().__init__(_Model())`; `tests/generation/bindings/token_autoregressive/test_engine_selection.py` `_Executor()` → `_Executor(object())`.

**Verify:** `.venv/bin/python -m pytest tests/generation tests/models/families -q` (a full-suite run of this change measured 3616 passed / 53 skipped)

**Estimate:** ≈ **-110 net lines**, ~16 source + 3 test files.

---

## 5. Dissolve `vrl/generation/ray/utils.py`; `RayGenerationConfig` takes its own validators back

**Value: high (deletes a whole incohesive module). Risk: very low (all moves are same-package, no signature semantics change).**

**Destinations:** `DistributedWorkerHandle` (`vrl/generation/execution/types.py:26`), `vrl/generation/ray/executor.py`, `vrl/generation/ray/launcher.py`, `RayGenerationConfig` (`vrl/generation/ray/config.py:72`).

| Function (`ray/utils.py`) | Goes to | Why |
|---|---|---|
| `require_installed_policy_version` (`:89`) | **method** `DistributedWorkerHandle.require_installed_policy_version(installed, expected)` | `worker_id=worker.worker_id` at both call sites (`weight_sync.py:45,75`) is the lost self |
| `require_correlated_result` (`:110`), `is_oom_error` (`:132`) | module-private `_require_correlated_result` / `_is_oom_error` in `ray/executor.py` (their only caller, and their home before commit `08ac6710`) | subject-less, but they have no business in a shared bag |
| `all_workers_support_versioned_slots` (`:30`), `validate_worker_gpu_ids` (`:58`), `require_chunk_gatherer` (`:19`) | module-private `_`-prefixed functions in `ray/launcher.py` (sole caller) | **not** methods: `RayGenerationLauncher` is `@dataclass(slots=True)` whose only fields are `init_ray`/`ray_init_kwargs` (`launcher.py:41-50`) — `config` is a `launch()` parameter, so a method would be a fake `self` |

**Keep the `ray: Any` parameter** on `_all_workers_support_versioned_slots`. Passing the ray module explicitly is the uniform repo shape (`health_monitor._terminalize`, `vrl/ray/dependencies.inspect_cluster`, `vrl/ray/resource_cleanup.kill_actors`, `vrl/ray/placement.cross_node_preflight`) and five tests at `tests/generation/ray/test_runtime_config.py:298-343` pass the real `local_ray` fixture. The two monkeypatches (`:387`, `:532`) only need the new `_`-prefixed name; the patch mechanism is unchanged.

**Same commit, `ray/config.py`:** fold `validate_colocated_replay_memory` (`:127`) and `_validate_driver_cuda_ownership` (`:165`) into private methods of `RayGenerationConfig` — both take the config as a parameter and their only production caller is `validate_driver_state` (`:111-124`) passing `self`. Remove `validate_colocated_replay_memory` from `__all__` (`:287`). `validate_driver_state` remains the single public entry (`launcher.py:215`). `_driver_cuda_devices` / `_get_device` / `_iter_parameter_devices` / `_cuda_device_index` **stay free** — they walk an arbitrary module tree.

Rewrite `tests/trainers/test_memory_guards.py:80,94` to drive `config.validate_driver_state(driver_bundle=...)`, which is already how `tests/generation/ray/test_runtime_config.py:886-937` does it. Repoint `tests/generation/ray/test_oom_split.py:24` to `executor._is_oom_error`.

**Do NOT** touch `validate_worker_gpu_ids(config, ...)`'s config parameter — despite the identical-looking signature, its caller (`launcher.py:119`) is an external holder, not `self`.

**Verify:** `.venv/bin/python -m pytest tests/generation/ray tests/trainers/test_memory_guards.py -q` and `git status` shows `vrl/generation/ray/utils.py` deleted.

**Estimate:** ≈ **-40 net lines** and one module removed, ~8 files.

---

## 6. The two model-layer dataclasses take back their own derived reads

**Value: high as a rule (kills four "no-op parameter" patterns at once). Risk: zero at runtime, highest test churn of any item — 24+ fake-object test cases.**

**Destination:** `vrl/models/interfaces/runtime.py` — `ModelBuild:161` and `RuntimeBundle:321`.

**6a. `ModelBuild` curated views.** Move `model_revision_kwargs` / `model_pretrained_kwargs` / `model_config_revision_kwargs` (`vrl/models/loader.py:9-32`) onto `ModelBuild` as `@property revision_kwargs`, `@property pretrained_kwargs`, `def config_revision_kwargs(self, field)` — beside the existing `memory`/`lora`/`num_steps` views whose docstring (`runtime.py:171-177`) already claims exactly this role. Their `build: Any` + defensive `getattr` guard declared fields (`revision` at `:182`, `model_config` at `:194`). Deletes ten lazy `from vrl.models.loader import model_revision_kwargs` imports (cosmos3:90, predict2:197, predict2_5:199, anima/runtime:58, anima/model:87, wan config:104, wan model:185,957, echo/runtime:49, echo/model:249, denoise/base:517). Repoint the comment at `vrl/config/model_schema.py:131`.

Retype only `load_diffusers_transformer` / `load_diffusers_scheduler` / `load_flow_match_scheduler` to `build: ModelBuild`. Leave the three quantization functions alone — they still contain strings of `getattr(build, ...)`, and annotating without removing them is cosmetic.

**6b. `RuntimeBundle.loads_full_generation_modules: bool`** (keyword, no default) replaces `metadata: dict` + `LOADS_FULL_GENERATION_MODULES_KEY` (`:28`) + `full_generation_bundle_metadata` (`:102`) + `minimal_replay_bundle_metadata` (`:108`) + `bundle_loads_full_generation_modules` (`:114`, `bundle: Any` + getattr-probing a class 200 lines below in the same file). Four construction sites: `denoise/build.py:120,147`, `token/build.py:120-122` (→ `not replay`), `magi_1/runtime.py:64`. One consumer: `ray/config.py:145` → `if not bundle.loads_full_generation_modules: return`. Delete the two prose blocks (`runtime.py:22-27`, `:346-354`) that explain the key — **this is why item 6 lands before item 5**, which would otherwise have to repoint them.

**6c. Two no-op identity strings, deleted while in the neighbourhood:**
- `memory_owner` (`vrl/models/steps/denoise/build.py:38-43`) — its sole caller passes `f"{entry.family} model"` (`:186-190`) three lines after proving `build.family == entry.family` (`:180-183`). Compute it inside from `build.family`. `apply_generation_memory_policy`'s own `owner` stays (a second, un-derivable caller labels a bare VAE at `vrl/scripts/eval/wan_phys_ab_sample.py:69`).
- `label` on `load_weights_into` / `validate_weights_for` (`vrl/models/utils.py:58,84`) — `type(<first positional arg>).__name__` in 5 of 5 production sites. Derive once inside `validate_weights_for` as `label = type(unwrap_compile_and_ddp(module)).__name__`, placed before the first raise. Correction to the original risk note: `unwrap_compile_and_ddp` deliberately does **not** peel PEFT (`utils.py:23-24`), so only compile/DDP wrappers change name. Rewrite the `token/base.py:60-65` docstring that documents the parameter, and drop `label="adapter policy"` at `tests/nn/quantization/test_fp8.py:280`. **Land this as part of the already-decided `vrl/models/utils.py` split** (PEFT lifecycle → `peft_adapter.py`, weight loading → a weights module) so the parameter never survives the move; also promote the function-local import at `denoise/base.py:324` to module scope, where its four siblings already are (`:30-35`).

**Test churn — the real cost, and the point:** 8 files / 24 cases build `SimpleNamespace` fakes that bypass `ModelBuild.__post_init__` (`runtime.py:206-243` requires non-empty `family` and a `RolePrecision`-shaped `precision`): `tests/models/test_loader.py` ×1, `tests/scripts/eval/test_sana_checkpoint_compare.py` ×1, `tests/scripts/eval/test_sana_aesthetic_checkpoint_eval.py` ×2, `tests/models/families/cosmos/anima/test_artifact_resolution.py` ×5, `.../predict2/test_model_loading.py` ×1, `tests/models/families/echo/test_model_loading.py` ×4, `.../sd3_5/test_model_loading.py` ×1, `.../wan_2_1/test_model_loading.py` ×9. Plus `RuntimeBundle` fakes at `tests/models/interfaces/test_minimal_replay_runtime_wiring.py` (delete the two tests that only mirror the literal), `tests/trainers/test_memory_guards.py:74,91`, `tests/generation/ray/test_ray_resident_session.py:72`, `test_rollout_launcher.py:95`, `test_runtime_model_contract.py:164,179`, `test_runtime_config.py:37`. **Do not** keep the free functions as forwarding shims — that leaves two read paths and cures nothing.

**Verify:** `.venv/bin/python -m pytest tests/ -q` (must return to zero failures; production-path baseline measured clean, all 24 failures were fake objects)

**Estimate:** ≈ **-60 net lines**, ~12 source + ~14 test files.

---

## 7. `vrl/ray/resources.py` — the resolved plan and the role configs answer their own questions

**Value: medium-high. Risk: low (all same-file), moderate test rewrite.**

**Destination:** `ResolvedDistributedResources` (`vrl/ray/resources.py:117`), `RoleResourceConfig` (`:16`) and a new `WorkerRoleResourceConfig`.

**7a. Two device derivations become members** beside `rollout_num_gpus`/`colocated`/`requires_trainer_reservation` (`:143-158`): `trainer_torch_device` (`:453`) → `@property`; `reward_torch_device` (`:462`) → method keeping its one real `trainer_device` argument, with its two internal calls (`:509,515`) becoming `self.trainer_torch_device`. Call sites: `vrl/run.py:118,147`, `scripts/common/factory.py:93,94,162`, `scripts/common/online.py:734`, `scripts/families/wan_2_1/train_dpo.py:145`.

**`format_distributed_resource_plan` (`:518`) STAYS a free function** and does **not** become `__str__`. Repo-wide, renderers are free functions (`format_host_memory`, two `format_report`s in `scripts/perf/`); a `def format`/`__str__` on a dataclass appears **zero** times. The class docstring at `:120-122` names it as the reader of a display-only field — keeping the name keeps that audit greppable. `build_bundle_layout` (`:1077`) stays a free factory (it constructs a different type).

**7b. Role name stops travelling beside the role config.** Add `role: ClassVar[str] = "trainer"` to `RoleResourceConfig` (+ `key_prefix` property), a new `WorkerRoleResourceConfig(RoleResourceConfig)` carrying `gpus_per_worker`/`num_workers` and the methods `requested_gpu_count(available_count)` / `resolve_num_workers(resolved_gpu_count, allow_zero_workers=False)`, and have `RolloutResourceConfig`/`RewardResourceConfig` subclass it with their own `role` and `gpu_pool`. The five hand-destructured blocks at `:725,747,850,869,894` collapse to `rollout_config.requested_gpu_count(available_count=...)`. Drop the `role: str` kwarg from `_resolve_role_devices` (`:641`), `_resolve_role_num_workers`, `_requested_role_gpu_count`, **and `_explicit_role_gpu_count` (`:621`) — missed by the original finding and the strongest evidence the name is derivable.**

Two hard constraints, both verified: `gpus_per_worker`/`num_workers`/`gpu_pool` must **not** be lifted to the trainer base — `find_unknown_keys` today rejects `distributed.resources.trainer.gpus_per_worker`, and lifting turns them into accepted no-op knobs. `ClassVar` is excluded from `dataclasses.fields()` even under `slots=True`, so the schema key derivations at `vrl/config/schema.py:788-800` are unaffected. Rename the `resolve_num_workers` parameter to `resolved_gpu_count` — it receives `len(rollout_devices)`, not the `num_gpus` request field of the same name.

Free bonus, fix it: `_resolve_role_devices` passes an unprefixed `f"{role}.devices"` (`:652-653`) while the rollout path passes the full path, so the same failure prints two different key spellings. `self.key_prefix` unifies them; no test asserts the unprefixed form.

**7c. Stop eroding the plan in `vrl/scripts/common/`.** Annotate `resources: ResolvedDistributedResources[| None]` at `factory.py:61,138` and `online.py:201` (both files already import from the module: `factory.py:13`, `online.py:23`). Move the plan-consistency block `factory.py:93-113` to `vrl/ray/resources.py` as **`require_reward_device(resolved, device)`**, a free function next to `reward_torch_device` — it is a validator over the plan plus a caller-supplied device, and it matches the module's existing shape; carry the `CUDA_VISIBLE_DEVICES`/placement-ordinal WHY comment (`:88-92`) with it verbatim. Narrow `_initialize_ray_cluster` (`online.py:94`) to take `cross_node: bool` — it reads nothing else, and that deletes the fake-resources factory at `tests/scripts/test_online_ray_cluster.py:34-35` entirely. Annotate `vrl/ray/placement.py:89 cross_node_preflight` while there.

**Test edits:** `tests/ray/test_resources.py` (~15 assertions), `tests/config/test_load_all_experiments.py:742`, `tests/scripts/test_online_lifecycle.py:415-424` (drop two monkeypatches, extend the SimpleNamespace at `:320`), `tests/scripts/test_wan_dpo_config.py:175-176,297-298`, `tests/scripts/test_wan_dpo_checkpoint_identity.py:29-30`. The `format_distributed_resource_plan` monkeypatches stay untouched — that is the payoff of keeping it free.

**Verify:** `.venv/bin/python -m pytest tests/ray/test_resources.py tests/scripts/test_online_lifecycle.py tests/scripts/test_online_ray_cluster.py tests/scripts/test_wan_dpo_config.py tests/scripts/test_wan_dpo_checkpoint_identity.py tests/scripts/test_common_factory.py tests/config/test_load_all_experiments.py -q`

**Estimate:** ≈ **-70 net lines**, ~10 files.

---

## 8. `vrl/rollouts/` — the accumulators and the evaluator base

**Value: medium. Risk: low, but the coordinator signature change has runtime-visible test fallout.**

**Destinations:** `RolloutStats` (`vrl/rollouts/stats.py:38`), `RolloutIteration` (`vrl/rollouts/orchestration/types.py:45`), new `ReplayEvaluatorBase` (`vrl/rollouts/evaluators/base.py`).

**8a. `record_phase` → `RolloutStats.phase(name)`.** `record_phase(phase_times: dict, name)` (`orchestration/rollout_runtime.py:33-39`) is a context manager over the exact reduction `RolloutStats.add_phase` already performs (`stats.py:58-63`), in a class whose module docstring says it exists to replace "the hand-threaded `phase_times: dict[str, float]` passed through ~12 files". Both schedules currently hold **both** accumulators and merge at the end. Add `@contextlib.contextmanager def phase(self, name)` routing through `add_phase`, delete `record_phase` and its `__all__` entry (`:248`), and retype the seven `phase_times: dict[str, float]` annotation sites across six `RolloutRuntimeCoordinator` methods (`:71,76,110,168,176,182,189`) to `stats: RolloutStats`.
- `strict_on_policy.py`: delete `schedule_phases` (`:72`) and the merge (`:130`) — `stats` already exists at `:73` before the first coordinator call. `:161` becomes `park_training_state_for_rollout(RolloutStats())`; keep the parameter required so the five recording sites stay branch-free.
- `continuous/owner.py` has **two shapes**: `commit_weights` (216-247) collapses as proposed; `next_iteration` (119-176) must **keep** the local (retyped to `startup_stats = RolloutStats()`) because `_start_pipeline` records at `:122-128` before the iteration's stats object exists at `:153` — merge with `iteration.stats.merge(startup_stats)` at `:176` and leave a comment saying why the local is load-bearing.
- Runtime-breaking test fakes (bare dicts into the real coordinator): `tests/rollouts/orchestration/test_driver_frozen_offload.py:79,81,85,95,96` and `test_orchestration.py:395,434`. Stale annotations on duck-typed fakes (still work, positional): `test_strict_failure_path.py:46,52,56,59,64,69`, `continuous/test_owner.py:131,380`, `continuous/test_contracts.py:91,94`.

**8b. `annotate_batch_context` → `RolloutIteration.annotate_batch_context()`** (same file, `types.py:95-108`): one parameter, of the type declared 50 lines above, reads six of its fields and mutates its batches. Keep the `return self` so `strict_on_policy.py:131-140` stays a single expression; drop it from `__all__` (`:116`). **`build_rollout_iteration` STAYS a free factory** — a kwargs-only constructor helper is a legitimate free function and `docs/sprints/done/SPRINT_function_organization_audit.md:97` already exempts it; narrow that doc line to name the factory specifically. Call sites: `strict_on_policy.py:131`, `continuous/consumer.py:210`; tests at `tests/rollouts/orchestration/test_iteration_types.py:11,81,107`; doc pointer at `docs/sprints/reading/SPRINT_batch_context_dict_adjudication.md:15`.

**8c. Evaluator preamble → `ReplayEvaluatorBase` mixin** in `vrl/rollouts/evaluators/base.py`, **not** on the `Evaluator` Protocol. All five evaluators open `evaluate()` with `require_replay_model(model, owner="XEvaluator.model")` — literally `f"{type(self).__name__}.model"` — at `token/token_logprob.py:51,53`, `token/continuous_token_logprob.py:43`, `token/multi_segment_token_logprob.py:42`, `denoise/sde_logprob.py:61`, `denoise/chunk_autoregressive_logprob.py:39`. The mixin holds `_require_models(self, model, ref_model=None)`; the five classes become `class XxxEvaluator(ReplayEvaluatorBase, Evaluator)` — the shape `SingleProcessStrategy(_TrainingStateParking, Strategy)` already uses — and each `evaluate()` opens with one line. Verified by execution: MRO is `A -> ReplayEvaluatorBase -> Evaluator -> Protocol`, `Evaluator.__protocol_attrs__` stays `['evaluate']`, so the ~15 duck-typed test fakes keep satisfying the structural check. `require_replay_model` keeps its `owner` kwarg — the mixin is now the thing passing it.

**Verify:** `.venv/bin/python -m pytest tests/rollouts tests/trainers -q` (baselines 328 / 451) and `grep -rn record_phase vrl/ tests/` returns nothing.

**Estimate:** ≈ **-45 net lines**, ~9 source + ~8 test files.

---

## 9. `vrl/trainers/` — one strategy mixin, one adapter-export owner, one family adapter evicted

**Value: medium. Risk: medium (MRO ordering is silently wrong-answer here, not fail-loud).**

**9a. `_UnshardedStateStrategy` mixin** beside `_TrainingStateParking` (`vrl/trainers/strategy.py:167`), holding the **five** SingleProcess≡DDP bodies whose shared precondition is "every rank already holds the full unsharded tensor": `export_checkpoint_state` (`:336`/`:965`), `load_checkpoint_state` (`:346`/`:986`), `load_full_checkpoint_state` (`:357`/`:997`), `export_optimizer_state` (`:368`/`:1008`), `load_optimizer_state` (`:376`/`:1018`). Name the mixin after the **precondition**, and have its docstring name `FSDPStrategy` as the counterexample that overrides all five.
```python
class SingleProcessStrategy(_TrainingStateParking, _UnshardedStateStrategy, Strategy)
class DDPStrategy(_UnshardedStateStrategy, Strategy)
class FSDPStrategy(_TrainingStateParking, Strategy)      # unchanged
```
**Keep duplicated, deliberately:** `backward` (all three) — each copy's comment *is* the per-backend correctness argument ("FSDP2 reduce-scatters gradients inside the backward hooks" / "DDP all-reduces … this IS the synchronized step"), and it fits neither mixin's name. `clip_grad_norm` (SingleProcess/DDP) — same text, different theorem, and `FSDPStrategy._clip_cpu_offloaded_grad_norm` disproves the DDP version's premise. The proposed `_ProcessGroupStrategy` (barrier/all_ranks_succeeded/shutdown) — net LOC ≈ 0, and `all_ranks_succeeded`'s real body is already deduped into module-level `_distributed_all_ranks_succeeded` (`:1139`).

**MRO is load-bearing and fails *silently*:** with `class C(Strategy, Mixin)` the Protocol's `...` stub wins and returns `None`, while `callable(getattr(obj, name, None))` is still `True` — so `vrl/trainers/checkpointing.py:868,905` would skip the load instead of raising. **Every mixin precedes `Strategy`.**

Separately, the `DDPStrategy` docstring (`:876`) is stale on two counts ("the same full-state path FSDP uses", "only prepare_model diverges") — fix it in the same commit.

**9b. Adapter exports get one owner, in two halves** (dependency direction is one-way: `checkpointing.py:20` imports `models.interfaces.runtime`, so the models side must not learn about `AdapterExport`).
- Models side: `adapter_roots` property on `DiffusionModelBase` (returns the `save_pretrained`-capable entries of `trainable_modules`) and on `ARModelBase` (returns `{"model": self.language_model}` — token families set `trainable_modules={"model": model}` at `steps/token/build.py:116`, which is why the AR branch must reach through). Mirror it onto `RuntimeBundle` as a plain field filled at the four construction sites, exactly as `trainable_modules`/`scheduler`/`raw_handle` already are (`denoise/build.py:114-120`); magi keeps the empty default. Snapshot timing verified safe — LoRA is attached before bundle construction.
- Trainers side: `build_adapter_exports(bundle, *, use_lora)` in `vrl/trainers/checkpointing.py`, which already owns `LORA_WEIGHTS_NAME` (`:33`), `AdapterExport` (`:157`) and the `lora_weights/<root>` namespacing (`:694`). This deletes the `step_kind` branch at `online.py:942-946`, the two `_export_*_lora` free functions (`online.py:377,404`, both `bundle: Any` with a `getattr(..., {})` guarding a declared field), and the third construction site at `wan_2_1/train_dpo.py:278-282` (which also silently drops `transformer_2`).
- **Behavior guard:** the `save_pretrained` filter must stay on the *denoise* `adapter_roots`, not in `build_adapter_exports`. Today the AR path raises loudly via `AdapterExport.__post_init__` (`checkpointing.py:164`) while the denoise path silently drops; centralizing the filter would downgrade AR to silent no-export.
- Move `tests/scripts/test_online_metrics.py:38-51` to `tests/trainers/test_checkpointing.py` against a real bundle, and add a token-side case pinning the raise.

**9c. `wan_forward` leaves the family-neutral trainer.** Move `wan_forward` (`vrl/trainers/offline/dpo.py:113-130`) to `vrl/scripts/families/wan_2_1/train_dpo.py` beside `_build_encoders` (`:22`), its sole caller (`:219`), and drop it from `vrl/trainers/offline/__init__.py:7,14`. **`ForwardFn` and the "caller provides forward_fn" contract stay in `dpo.py`** — that is the genuine protocol boundary; only the one concrete implementation moves. Fix the two docstrings the move falsifies: `:155-156` ("`wan_forward` provided here") and the module docstring `:3-12`, whose "two batteries-included paths" are actually the `_inject_noise` prediction_type branches (`:227-254`), not forward adapters. The recipe module imports no torch at module level today — use `if TYPE_CHECKING: import torch` so config dispatch does not start dragging torch in. Land the two tests in `tests/scripts/` (precedent: `tests/scripts/test_wan_dpo_encoders.py` already imports the private `_build_encoders` from that module).

**Verify:** `.venv/bin/python -m pytest tests/trainers tests/scripts -q`

**Estimate:** ≈ **-60 net lines**, ~11 files.

---

## 10. `vrl/nn/` — the quantized-linear swap and the torch-native AR backend

**Value: medium (kills a 5-way duplicated policy table and an injection seam with no production producer). Risk: low.**

**10a. `QuantizedLinear.swap_linears` classmethod** (`vrl/nn/quantization/base.py:16`). `swap_linears_to_fp8` (`fp8.py:198-213`) and `swap_linears_to_nvfp4` (`fp4.py:231-248`) differ by exactly one hunk out of sixteen lines; the knowledge that differs is already class-shaped (`quantization_scheme`, `cache_buffer_names` are class attributes). Two hooks: `default_target_profile` class attr (ATTENTION_MLP / MLP_ONLY) and `can_replace(cls, linear) -> bool` (default `True`).
- The sentinel is mandatory — `target_profile=cls.default_target_profile` in the signature is dead code (`cls` does not exist at `def`-evaluation). Use `None` + resolve inside, **before** the loop, so `tests/nn/quantization/test_fp4.py:229` / `test_fp8.py:141` ("invalid profile raises, model untouched") still hold.
- `Fp4Linear.can_replace` must share a predicate with `Fp4Linear.__init__` (`fp4.py:153-162`), which **already** performs the same `% NVFP4_K_ALIGNMENT` / `% NVFP4_N_ALIGNMENT` check — extract `_alignment_error()` so construction raises and traversal skips off one source. Net effect: one fewer copy, not one more.
- Collapse the fourth and fifth copies of the same table: `vrl/scripts/perf/quantized_sd3_forward_profile.py:232-235` and `:248-251` become a `{"fp8": Fp8Linear, "nvfp4": Fp4Linear}` lookup plus `cls.default_target_profile`. This is what gives the new class attribute a second, independent behavior consumer. `vrl/models/loader.py:170,173` keeps its hardcoded log strings (it dispatches by format string and holds no class) — leave a comment pointing at `QuantizedLinear.default_target_profile`.
- Delete the dead `"DEFAULT_EXCLUDE"` re-forward in `fp8.py:217`'s `__all__`.
- **`drop_quantized_masters` (`base.py:105`) stays a free function** — it is scheme-agnostic, dispatches on nothing, and constructs nothing. Leave a comment saying the asymmetry with `swap_linears` is deliberate.

**10b. `TorchNativeDecoderAttentionBackend` takes the shape of its paged sibling** (`vrl/nn/modules/torch_attention.py:40`). Give it `__init__(self, *, trunk, config)` setting `self.trunk` and `self.backend_label` — copying the fallback form at `ar_decoder.py:59-61` verbatim — and a private `_forward(embeds, mask, kv=None)` that raises inline. Deletes `require_past_key_values` (`:81`, whose `label` is threaded down two levels from `config.extra["backend_label"]`), `_native_decode_fns`, and the `PrefillFn`/`StepFn` aliases; `build_torch_native_backend` collapses to a single constructor call parallel to `build_vllm_attention_backend`.
- `_last_token_hidden` moves in as a `@staticmethod`, unchanged — it is genuinely subject-less, and it is **not** a duplicate of `paged_attention_helpers.py:71` (that one rejects `[B,T,H]`; this one selects the last token and has a `last_hidden_state` → `hidden_states[-1]` fallback that flattening would bury).
- **Do not** promote `backend_label` to a typed `ARAttentionConfig` field — `docs/sprints/done/SPRINT_design_smell_audit.md:174-181` already recorded that as "withdrawn"; the change already removes the one unguarded `config.extra[...]` read.
- One real signature-surface change to declare in the PR: merging prefill and step means prefill now passes `past_key_values=None` explicitly. Equivalent for the HF decoders in play, but say so or keep the `kv is None` branch.
- Rewrite the module docstring (`:11-14`) — its claim that NextStep-1 supplies its own `_init_kv`/`_step_llm` handles through this seam is false (`nextstep_1/model.py:344-347` returns a plain `peel_peft(self.language_model)`), and it is the only evidence the injection seam looks alive.
- **Non-goal:** do not merge glm_image's two hand-written `past_key_values` guards (`runner.py:155-159,238-242`). glm_image cannot use this backend at all (it injects `position_ids`, `runner.py:17`, `runtime.py:89`), so sharing the guard would spread the label-threading to a third site. Three copies → two is the correct stopping point.

**Verify:** `.venv/bin/python -m pytest tests/nn -q`

**Estimate:** ≈ **-45 net lines**, ~8 files.

---

## NON-GOALS — what looks identical and must stay a free function

These were investigated and refuted. Do not "clean them up" in a follow-up sweep.

| Symbol | File | Why it stays free |
|---|---|---|
| `require_chunk_gatherer`, `_require_chunked_executor` | `generation/ray/utils.py:19`, `generation/execution/worker.py` | Not type erosion. Both already import the Protocol they guard; the `Any` is because the **input** is an unvalidated dotted-string import (`registry.py:408-413`, `worker.py:689`) — the same shape `isinstance` itself has. (They do relocate under item 5, but as free functions, and the separate hand-rolled check at `worker.py:601-606` is **dead**, not a third copy: `self.executor` is assigned only via a `-> GenerationChunkExecutor` builder and is dereferenced two lines later. Delete it; do not "share" it.) |
| `estimate_batch_bytes` | `rollouts/orchestration/continuous/types.py:48` | Already home. One typed param, output fully determined, sole producer (`producer.py:450`) and sole readers (`queue.py:74,89,106`) are all inside `continuous/`. Promoting it to `rollouts/batch/ops.py` or `RolloutBatch.nbytes` would advertise a deliberately-underestimating backpressure heuristic as a general size API — explicitly ruled out at `docs/sprints/done/SPRINT_memory_budgeted_microbatch.md:198` — and would drag torch + `vrl.trajectory` into the dependency-free `batch/core.py`. |
| `resolve_prompt_example_artifacts`, `_artifact_values` | `trainers/data/artifacts.py` | `field_name` names a **manifest key**, not a caller (the `require_exact_int(..., path=)` case). Attaching a by-name, metadata-fallback accessor to `PromptExample` would violate the contract stated in that class's own comment (`prompts.py:37-39`). Separately, 4 of the 5 branches of the resolver have no production consumer — the correct treatment is deletion, not promotion. And the proposed `dataclasses.replace(self, ...)` was measured to **share** the `metadata`/`request_overrides` dicts with the source object, replacing today's explicit copies. |
| `format_online_metric_row`, `build_online_metric_row`, `online_metric_columns` | `trainers/metrics_io.py` | No `Any`, no caller identity, no import cycle. `build_online_metric_row` is a cross-package adapter over `vrl.algorithms` metric trees and scheduler telemetry keys — its change driver is the producer, not the row. `online_metric_columns` has no instance at its main call site (`scripts/supervise.py:151` calls it with zero args). The module is a deliberately designed stable CSV protocol. |
| `_model_section_class_for_family`, `sampling_section_class_for_family`, `*_from_path` | `config/schema.py` | `family: Any` is the **raw YAML value** boundary — verified to produce config-domain errors for `None`, `""`, `123`. Five of six call sites hold no entry. Inlining would copy the None-guard to 2-3 sites, and `schema.py:372-380` **catches** that ValueError to select a permissive fallback. The `*_from_path` pair is already fully typed, pure, and memoized. |
| `_validate_yaml_home` | `config/builders.py` | `owner` names the dataclass being checked (the error domain), not the caller. The proposed destination produces a reproducible import cycle, and the suggested lazy-import escape hatch was measured to make `import vrl.config.schema` raise. |
| `vrl/utils/nsys_report.py` (whole module), `format_report`, `report_to_dict` | `utils/nsys_report.py` | Not symptom (E): 756 lines, 16 exported names, 6 dataclasses, 14 contract tests. Moving it into `scripts/perf/common/` inverts the library/entrypoint layering and splits it from its pair, `utils/profiling.py`. And `format_report`-as-method would be the **only** dataclass render method in the repo — the established shape is `HostMemorySnapshot` (derived scalar = property) + `format_host_memory(snapshot)` (rendering = free function), two files away in the same package. |

Also staying free inside the restructures above, for the same reason: `require_zero_replay_timestep` / `require_replay_segments` / `single_segment_result` (`models/interfaces/replay.py`), `resolve_hf_checkpoint_dir` (`models/steps/token/loader.py`), `drop_quantized_masters` (`nn/quantization/base.py:105`), `_last_token_hidden` (as a staticmethod, unflattened), `format_distributed_resource_plan` + `build_bundle_layout` (`ray/resources.py`), `is_oom_error` / `require_correlated_result` (relocated, not promoted), `build_rollout_iteration` (`rollouts/orchestration/types.py:69`), `ForwardFn` + the forward-adapter seam (`trainers/offline/dpo.py`), and the four bundle-tree walkers `_driver_cuda_devices` / `_get_device` / `_iter_parameter_devices` / `_cuda_device_index` (`generation/ray/config.py`).

---

## The four placement rules

Apply these mechanically before adding a parameter or a subclass method.

**Rule 1 — A name argument is a lost `self`.**
If you are about to write `owner=`, `label=`, `what=`, `context=`, or `prefix=` and the value at every call site is `type(self).__name__` or your own family's literal, the function belongs on your base class and should derive the name from `type(self).__name__`.
*Counter-test that keeps it free:* the string names a **config key** (`path="model.family"`), a **manifest field** (`field_name`), or a **data label supplied by a second, un-derivable caller**. `require_exact_int(value, *, path)` is correct as written.

**Rule 2 — `Any` needs a receipt.**
A parameter typed `Any` must be justified by a real import cycle or a genuinely unvalidated input (raw YAML, a dotted-string import, an arbitrary user module tree). Otherwise annotate it. If annotating it makes the owning type the function's only required parameter, it is a property or a method on that type — and the defensive `getattr(x, "field", default)` guarding a declared field goes away with it.
*Applied:* `build: Any` → `ModelBuild` (item 6), `bundle: Any` → `RuntimeBundle` (items 6, 9), `resources: Any` → `ResolvedDistributedResources` (item 7).
*Kept:* `gatherer: Any` in a registry-import guard, `ray: Any` as a lazy-framework handle.

**Rule 3 — Two copies is a coincidence; three is a class attribute.**
When the same body appears in ≥2 sibling subclasses and differs only in a constant, the constant becomes a class attribute and the body becomes a shared default. Choose the shape by asking *what happens if a future family forgets*:
- forgetting must **fail loud** → keep the abstract method and add an **opt-in mixin** (`VaeDecodeMixin`, `EncoderAttentionMaskRunnerBase`) listed **first** in the bases;
- forgetting is safe → a **base default** (`DiffusionModelBase.trainable_modules`, `ARChunkExecutorBase.__init__`).
And **always put mixins before the Protocol/ABC in the bases list** — get it wrong on an ABC and you get a `TypeError` (fine); get it wrong on a `Protocol` and the `...` stub silently wins and returns `None` (`strategy.py`, verified).
*Counter-test that keeps the copies:* the two bodies are identical text but assert **different theorems**, and each copy's comment is that proof (`backward`, `clip_grad_norm`). Merging deletes the argument, not just the duplication.

**Rule 4 — An untyped bag with a closed key set is a field.**
If grep finds a finite key set, a small number of writers, and a reader that has to `getattr`-probe or `.get(KEY, default)`, replace the bag with a typed dataclass field and delete the key constant plus its factory/reader trio. Every field of a resolved struct must have a non-logging consumer.
*Applied:* `RuntimeBundle.metadata` → `loads_full_generation_modules: bool` (item 6), `phase_times: dict[str, float]` → `RolloutStats` (item 8), `config.extra["backend_label"]` → `self.backend_label` (item 10).

---

## Sequencing and totals

Hard ordering constraints (everything else is independent):

- **1 before 2** — both edit the same nine family `model.py` files; the mixin bases lists should settle before `from_build` collapses.
- **6 before 5** — item 6 deletes the `RuntimeBundle.metadata` prose at `runtime.py:22-27,346-354`; item 5 would otherwise have to repoint it to a symbol going private.
- **3 lands as one commit** with the already-decided `ARModelBase` absorption of `_lm_trunk` / `require_module_attrs` (same file).
- **6c's `label` drop lands as part of the already-decided `vrl/models/utils.py` split**, not before it.
- Within **4**, the `ARChunkExecutorBase.__init__` and `_native_runner_reason` edits touch the same file — one commit.

| # | Restructure | Destination | Net lines | Files |
|---|---|---|---|---|
| 1 | Four opt-in mixins | `models/steps/denoise/common/` | ≈ **-320** | ~19 |
| 2 | Diffusion base absorbs `from_build` + transformer members | `models/steps/denoise/base.py` | ≈ **-190** | ~11 |
| 3 | `ARModelBase` + new `ARReplayCore` | `models/steps/token/base.py` | ≈ **-110** | ~12 |
| 4 | Executor bases: ctor, gather, forward_plan, native runner, CUDA peak | `generation/bindings/*`, `generation/execution/` | ≈ **-110** | ~19 |
| 5 | Dissolve `ray/utils.py`; `RayGenerationConfig` privates | `generation/ray/`, `execution/types.py` | ≈ **-40** + 1 module | ~8 |
| 6 | `ModelBuild` views, `RuntimeBundle` flag, `memory_owner`, `label` | `models/interfaces/runtime.py` | ≈ **-60** | ~26 |
| 7 | Resolved plan + role configs | `ray/resources.py` | ≈ **-70** | ~10 |
| 8 | `RolloutStats.phase`, `RolloutIteration`, evaluator mixin | `rollouts/` | ≈ **-45** | ~17 |
| 9 | Strategy mixin, adapter exports, `wan_forward` eviction | `trainers/` | ≈ **-60** | ~11 |
| 10 | `QuantizedLinear.swap_linears`, torch-native backend | `nn/` | ≈ **-45** | ~8 |
| | **Total** | | **≈ -1050** | **~140 (≈45 test)** |

Per-restructure gate, in addition to the listed pytest command:
`.venv/bin/ruff check --fix <touched .py> && .venv/bin/ruff format <touched .py> && .venv/bin/ruff check <touched .py> && .venv/bin/ruff format --check <touched .py>` — touched files only, never repo-wide.