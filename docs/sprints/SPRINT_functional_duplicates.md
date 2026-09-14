# Functional duplicate sweep

Status: batch 1 landed (2026-09-13); one-liner audit added the same day. Scope is functional duplication — two
places computing the same thing under different names or framings — not
textual repetition. Every candidate was read at the source, its call sites
and tests, before deciding; each verdict below cites the paths.

## Method

1. Listed all 401 `vrl/**/*.py` modules with their docstrings and grouped
   them by purpose (weight sync, parking, deadlines, tree hashing, dtype
   parsing, judges, checkpoint evals, perf probes, family runtimes).
2. AST-normalized every function body (names/constants stripped) and
   compared digests to find identical logic under different names.
3. For each hit, read the surrounding contract to decide whether it is a
   real duplicate, a layered design, or a look-alike with a different
   numeric or protocol obligation.
4. Fixed the cases whose consolidation removes a second owner of the same
   behavior without changing pinned digests or pipeline-faithful arithmetic.
   Ran ruff on touched files and the affected test lanes after each change.

Not counted as cleanup: renames, or moving identical branches into a
classmethod. Each landed change deletes one implementation.

## Landed

### Qwen-VL video judges — `f5780440f`

`vrl/rewards/models/videoscore2.py` and `vrl/rewards/models/cosmos3_reasoner.py`
each carried an identical `_messages` (system rubric + video + user
template), an identical regex-to-`1..5` integer parser and an identical
axes-plus-mean normalizer; the AST scan flagged `_messages` byte-identical
and the other two identical modulo axis count. Only prompts, regex and
axis names differed. `QwenVLVideoJudge` (`vrl/rewards/models/qwen_vl_judge.py`)
now declares `system_prompt`, `user_template`, `score_regex`, `score_axes`
and implements `_messages`, `_parse`, `parse_integer_scores`,
`normalize_scores` once. VideoScore2 keeps only its soft-score merge
override. The parsing tests call the classmethods directly; no module-level
compatibility aliases were kept.

Kling's `_normalize_scores` (`vrl/rewards/models/kling_video_reward.py:548`)
is a different contract (renames a raw `VQ/MQ/TA/Overall` dict, drops
missing) and was left alone.

### Fixed Anima eval sampler — `b6dd020e5`

`vrl/scripts/eval/anima_fixed_eval._generate` re-implemented
encode → `DenoiseRequest` → `prepare_sampling` → native scheduler loop →
decode, which `vrl/scripts/eval/denoise_generation.generate_images` already
owns (used by `image_checkpoint_eval` and the Anima generation archive).
`anima_geneval_eval` and `anima_tag_adherence_eval` delegate to `_generate`,
so all Anima evaluators now share one sampler and one `ImageSampling`
projection. The test stub in `tests/scripts/eval/test_anima_fixed_eval.py`
was already broken by the registry-driven `parse_config` (it replaced the
whole family entry); it now patches only `resolve_model_build`.

### Legacy Qwen2-VL key relocation — `306cc32d1`, then `hub.py`

`vrl/rewards/models/hpsv3.py` and `vrl/rewards/models/kling_video_reward.py`
both hand-coded the transformers 4.52 rename (`model.*` →
`model.language_model.*`, `visual.*` → `model.visual.*`) because they overlay
a fine-tuned state dict with `load_state_dict(strict=True)` after
`from_pretrained` (HPSv3 resizes embeddings first, Kling wraps in PEFT
first), which bypasses the conversions `from_pretrained` applies. The two
copies were first merged into one hand-written rule; on review the rule
itself was redundant: transformers registers exactly it under
`Qwen2VLForConditionalGeneration` in `transformers.conversion_mapping`.
`vrl/rewards/models/hub.py::relocate_checkpoint_keys(model, state)` now
runs the checkpoint through the model's own registered transforms (walking
the MRO, since the reward heads subclass the ForConditionalGeneration class
and transformers looks the mapping up by class name), peeling a PEFT wrapper
and keeping its prefix. No Qwen2-VL layout knowledge remains in VRL.
Verified by strict-loading the real `MizzenAI/HPSv3` and
`KlingTeam/VideoReward` checkpoints from the local HF cache on CPU.

Observed while verifying, not changed: `HPSv3Qwen2VLRewardModel.from_pretrained`
of the `Qwen/Qwen2-VL-7B-Instruct` base reports every text-tower key as
MISSING/UNEXPECTED for the same class-name-lookup reason, then the HPSv3
overlay strict-loads everything, so the result is correct but the base load
is wasted work and noisy.

### Storage dtype table — `306cc32d1`

`vrl/scripts/denoise/encode_targets.py` mapped `--storage-dtype` through a
local `{bf16,fp16,fp32}` table; `vrl/models/dtypes.resolve_torch_dtype`
owns that alias table and `vrl/trajectory/storage.py` already uses the
same `preserve`/resolve shape. The table is gone.

Other local dtype helpers were checked and kept:
`kling_video_reward._torch_dtype` adds `auto`/fallback semantics on top of
`resolve_torch_dtype`; `vllm_omni_diffusion_profile._DTYPE` is a policy map
(fp8 → bf16 storage), not an alias table.

### H3 replay component loading — `2fefaf05a`

`vrl/models/families/vdn_h3/runtime.py` and
`vrl/models/families/minimax_h3/runtime.py` loaded the same transformer,
flow video scheduler and audio scheduler with the same kwargs; VDN-H3 only
adds `install_hybrid_attention` afterwards. `load_h3_replay_components`
returns the shared `MiniMaxH3ReplayModel` kwargs. `VDNH3BatchExecutor`'s
pass-through `__init__` was removed.

### Perf probe timing — `923b23847`

`vrl/scripts/perf/gpu_preflight._cuda_time_ms` and
`vrl/scripts/perf/quantized_sd3_forward_profile._latency_and_memory`
re-rolled the CUDA-event warmup/record/synchronize loop owned by
`vrl/scripts/perf/common/timing.py` (already used by the linear, compile
and backward-MFU probes). Both now build on it; `cuda_step_times_ms` was
added so the SD3 profile's per-call percentile list and `cuda_median_ms`
share one loop.

### Generator runtime source digest — `06e2641a4`

`vrl/scripts/eval/denoise_generation.GeneratorRuntimeIdentity.capture`
hashed `vrl/**/*.py` with its own framing; it now calls
`vrl.models.source_integrity.runtime_source_tree_sha256` (the CausVid and
MAGI-1 source pin). The digest is compared only between archives written
by the same code, never against a pinned constant, so the framing change
is safe.

## Reviewed and deliberately left

### Per-channel VAE latent statistics (wan, anima, predict2, mochi, minimax_h3)

`vrl/models/families/wan_2_1/model.py:798`, `cosmos/anima/model.py:422`,
`cosmos/predict2/model.py:590`, `mochi/model.py:264`,
`minimax_h3/model.py:717` all build `latents_mean`/`latents_std` as
`[1, C, 1, 1, 1]` tensors, and wan/anima's encode side mirrors the decode.
The AST scan flagged wan≡anima. They were not merged because the
arithmetic order differs per family and mirrors the diffusers pipeline it
reproduces: wan/anima compute `1.0 / std` in float32 then cast and divide
(diffusers `WanPipeline`), predict2 folds in `sigma_data`, mochi divides by
`scaling_factor`. A shared helper that returns `(mean, std)` and multiplies
changes bf16 rounding versus the pipeline. A helper that merely builds the
two tensors would save four lines per site and keep two arithmetic
variants; `VaeDecodeMixin` in `vrl/models/steps/denoise/common/latent_decode.py`
already documents this family as intentionally separate.

### `prepare_replay` in pixart_sigma and mochi

`vrl/models/families/pixart_sigma/model.py:308` and `mochi/model.py:288`
are the same five lines around a family-specific scheduler factory
(`pixart_ddim_scheduler`, `standard_mochi_scheduler`). Extracting it means
an attribute pointing at the factory — indirection without deleting logic.

### Sampling-state `__post_init__` CFG twins

`sd3_5/model.py:70` and `cosmos/predict2/model.py:171` prove the same
invariant (`do_cfg` implies the family's uncond twin is present) on
different fields. Same shape, different fields; nothing to share.

### Three remaining tree hashers

- `vrl/models/source_integrity.runtime_source_tree_sha256` — vendored
  source pins (CausVid `fe5e028a…`, MAGI-1 `e3ec1712…`), now also the
  generator runtime identity.
- `vrl/rewards/models/countgd._runtime_tree_digest` — a manifest protocol
  (`schema`, `algorithm`, `file_count`, 8-byte length framing) whose value
  `COUNTGD_RUNTIME_TREE_SHA256 = e41c4fd6…` is the pin behind the
  CountGD equivalence run (133 files, 445 detections bitwise identical) and
  is written by the Bazel installer in `tools/countgd/`.
- `vrl/models/checkpoint_identity._hash_local_node/_hash_regular_file` —
  stat-guarded, directory-aware, symlink-cycle-checked hashing of a live
  checkpoint directory that must detect concurrent modification.

The first two could share one walker if CountGD adopted the
`source_integrity` framing, which means re-pinning the digest, bumping the
manifest schema and regenerating the manifest through the Bazel rule, then
re-running the equivalence check on a GPU host. Deferred: the pin is
current evidence and the dedupe is ~30 lines. The checkpoint hasher has a
different obligation (fail on files changing under it) and stays separate.

### Checkpoint-eval script family

`image_checkpoint_eval`, `anima_exact_count_checkpoint_eval`,
`wan_hpsv3_checkpoint_eval`, `wan_robotics_checkpoint_eval`,
`sana_aesthetic_checkpoint_eval` each own an argparse surface, a grid/seed
layout, and a report. Shared pieces already live in `denoise_generation`
(`seed_for`, `generate_one_video`, `generate_images`, `ImageSampling`).
What remains different is protocol, not code:

- `wan_robotics_checkpoint_eval._seed_for` (`base + sample*stride + row`)
  vs `denoise_generation.seed_for` (`base + prompt*samples + sample`):
  changing the formula changes which seeds existing shards were generated
  with, and `_load_and_validate_shards` checks them.
- The two `_write_contact_sheets` (`image_checkpoint_eval.py:677`,
  `anima_exact_count_checkpoint_eval.py:711`) render different layouts
  (per-sample rows with seed captions vs blind A/B panels with a sealed
  key). Same intent, different reviewer protocol.
- `sana_aesthetic_checkpoint_eval._generate_images` runs the frozen native
  SANA pipeline on purpose (`denoise_generation` says so in its docstring).

### Layered, not duplicated

- Weight sync: `vrl/trainers/weight_sync.py` (trainer-side flatten and
  send), `vrl/models/weight_utils.py` (receiver and version slots, documented
  as the inverse), `vrl/generation/ray/weight_sync.py` (awaited version
  coordination across Ray ranks). One owner per direction.
- Parking: `vrl/models/parking.py` (the CuMem pool and model parking, after
  `402f52ccb`/`6b723075e` merged the backends) and
  `vrl/generation/execution/memory_parking.py` (rollout-side scheduling of
  it). The remaining split is mechanism vs policy.
- Deadlines: `vrl/utils/deadline.py` is the transport-neutral core;
  `vrl/ray/operation_deadline.py` subclasses it with Ray cancellation, as
  both docstrings state.
- Reward function bindings: declarative registrations, one per reward.
- `emu3._apply_lora` ≡ `glm_image._apply_lora`: both are two-line wrappers
  over `install_token_lora_adapter`; the shared function is the
  consolidation, the wrappers are the family binding.

## One-line helpers: are they necessary?

An AST pass over `vrl/**/*.py` found 296 functions whose body is a single
`return` (dunder, `@property` and `@abstractmethod` excluded), 25 of them
pure pass-throughs (the call forwards exactly the parameters it received).
Each was read with its call sites. The test: does the name carry knowledge a
reader would otherwise have to reconstruct, or is it only a second name for
one expression at one call site?

### Inlined — `ecdb9cff9`, `nextstep_1`

Single-use helpers that only renamed a call; the docstring, where it
carried a reason, became a comment at the call site:

- `vrl/models/sequence_parallel.py::_group_info` → the two `dist` calls.
- `vrl/models/families/wan_2_1/model.py::_defer_wan_trainable_device_move`
  → a named bool where it was tested.
- `vrl/models/families/emu3/model.py::_replay_grid_dims` →
  `replay_context_image_size(...)` with its constant arguments.
- `vrl/models/families/janus_pro/runtime.py::_format_t2i_prompt` → the
  template f-string in the one comprehension that used it.
- `vrl/models/steps/denoise/common/lora.py::_lora_transformer` →
  `self.transformer` (no family overrode it; the docstring's reason is now
  the comment).
- `vrl/config/schema.py::_model_section_class_from_path` /
  `_sampling_section_class_from_path` → `import_from_path`; the
  `functools.cache` wrapped a function that is already cached by
  `sys.modules`.
- `vrl/models/families/nextstep_1/model.py::_last_hidden` →
  `kv["last_hidden"]`.

### Kept, with the reason

- **Framework surface.** Ray actor methods in `vrl/generation/ray/worker.py`
  (`sleep`, `execute_batch`, `probe_batch_size`, …) forward to `self.core`;
  the actor must expose them by name. `nn.Module.forward` one-liners
  (`denoise/base.py`, `aesthetic.py`), pydantic `field_validator`s
  (`precision.py`, `schema.py`), `extra_repr`.
- **Protocol implementations.** `Strategy.export_checkpoint_optimizer_state`
  delegating to `export_optimizer_state` in the unsharded backend (the split
  exists for the sharded backends; its docstring says why),
  `Algorithm.compute_advantages_from_tensors`, `Executor.parse_sampling_params`,
  `finalize_token`, `can_replace`, `_loss_weight`.
- **Overridable hooks with a non-trivial default elsewhere.**
  `_replay_forward_step_index` (Cosmos returns the real index, the base
  returns 0), `_lora_dtype` (five family overrides), `_loads_vq`.
- **Named predicates and records reused 3+ times.** `_is_auto`,
  `_is_configured`, `_uses_cfg`, `_optional_path`, `_clamp_score`,
  `_wire_envelope` and the `*_to_wire` family (the wire contract),
  `to_report_record`, `_checkpoint_record`, `seed_for`.
- **Adapters that change the callable's kind.** `actor_pool.wait_for_result`
  turns an injected `Awaitable` factory into a coroutine so
  `asyncio.create_task` accepts it.
- **Vendored code** (`llamagen/vendor/*`: `GPT_7B` … `VQ_16`) stays as
  upstream wrote it.
- **Multi-line constructors** (`server._overloaded_error`) are not
  one-liners; inlining a 12-line error into the admission branch would hurt
  the branch more than the helper costs.

## One-statement classes

The same AST pass listed 96 classes whose body is at most one statement
after the docstring (vendored code excluded). Grouped by what the class is
for:

### Removed

Empty subclasses that neither added a member nor were dispatched on:
`Lumina2SamplingState`, `MochiSamplingState`, `PixArtSigmaSamplingState`,
`SanaSamplingState` (over the shared masked-prompt states) and
`JanusProARState` (over `PagedCFGARState`). `sampling_state_cls` / `state_cls`
now name the shared class. `Emu3ARState` stays because it adds a field.

### Kept, by role

- **Typed exceptions** (`TerminalRuntimeError`, `StaleSlotDiscard`,
  `TrajectoryReaderError`, `_GroupProductionError`, …): the empty body is
  the point — callers catch the type.
- **Composition points.** The empty `<Family>ReplayModel(DiffusersReplayModelBase,
  <Family>Model)` classes (sd3_5, qwen_image, sana, lumina2, hunyuan_*,
  cogvideox, predict2, `WanI2VReplayModel`, `VDNH3ReplayModel`) are where
  the shared replay base meets one family's forward; the registry names
  them by path, tests instantiate them, and `test_family_mro.py` pins their
  MRO. A factory that composed them at runtime would trade nine visible
  declarations for one dynamic `type()` call.
- **Capability markers checked with `issubclass`:** `CumemRewardFunction`
  (the reward registry routes CuMem-pool allocation on it).
- **One-method Protocols** (`StatsSink`, `QueryableCompletion`,
  `GenerationWeightSync`, `RemoteReadyScorer`, …) and **single-field config
  sections** (`DDPConfig`, `KlingVideoRewardProductionConfig`, the sampling
  section ladder): schema, not logic.
- **Single-override subclasses** (`MiniMaxH3FlowScheduler.step`,
  `_SaturatedLinearAttnProcessor.__call__`, the `*ReplayModel.prepare_replay`
  overrides): the override is the family difference.

## Verification

Per change: ruff on touched files; the family's tests. Final sweep
(`CUDA_VISIBLE_DEVICES=""`): `tests/architecture tests/rewards
tests/scripts/eval tests/scripts/perf tests/models/families/{vdn_h3,minimax_h3}
tests/scripts/test_encode_sft_targets.py tests/scripts/test_anima_generate.py`
→ 854 passed, 7 skipped, 1 failed. The failure is
`tests/architecture/test_generation_rollout_boundaries.py::test_generation_model_imports_stay_on_public_floor`,
pre-existing from the parking consolidation (`memory_parking.py` importing
`vrl.models.parking`), not touched here.

## Next batch candidates

1. CountGD runtime-tree digest onto `source_integrity` (requires GPU
   re-verification and a manifest schema bump; see above).
2. A `(mean, std)` builder for Wan-style VAE statistics only if a numeric
   parity test against the diffusers pipeline is added first.
3. `vrl/rewards/models/unified_reward_video.py` does not subclass
   `QwenVLVideoJudge` although it loads a Qwen-VL judge the same way: it
   samples a fixed `num_frames` instead of `fps`, and `_parse_axis_scores`
   reads declared score ranges (`x/10` style) and rescales them, so only the
   loader/generate prelude would be shared. Worth doing once a real video
   fixture covers its frame sampling.
