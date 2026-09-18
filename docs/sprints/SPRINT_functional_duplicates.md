# Functional duplicate sweep

Status: batch 1 landed (2026-09-13); one-liner and one-statement-class audits the same day; batch 2 ("A1 and A2 not merged") 2026-09-13; boundary rule added and the masked-prompt base reversed 2026-09-17. Scope is functional duplication — two
places computing the same thing under different names or framings — not
textual repetition. Every candidate was read at the source, its call sites
and tests, before deciding; each verdict below cites the paths.

## Boundary: infrastructure is deduplicated, model glue is not

Decided 2026-09-17 after the masked-prompt base (below) was reversed. The
rule for every later batch:

- **Infrastructure** — code a family calls but does not describe the model:
  the CFG caller and branch packer (`DenoiseBackboneCaller`), pipeline
  load/freeze/placement (`DiffusersPipelineModelBase`), LoRA attach, VAE
  decode plans, timestep utilities, schedulers, replay-tensor helpers, the
  SDE/DDIM log-prob math. One implementation; a second owner is a defect.
- **Model glue** — the per-family `encode_prompt` / `prepare_sampling` /
  `forward_step` / `build_branch` / trajectory export and restore, and the
  family's sampling-state fields. Each family owns its copy, even when two
  families' copies are line-for-line equal today. This is the SGLang /
  vLLM / diffusers / transformers "one model, one file" convention: a
  reader learns a family from one file, a checkpoint quirk in one family
  cannot move another, and the next family is added by copying a file, not
  by fitting a base. The AST-digest scan will keep flagging these; the
  verdict is "model glue, kept".

A shared base for model glue is justified only when the families share a
**checkpoint contract**, not a call shape: the same transformer signature
enforced by one upstream class (wan's t2v/i2v pair on one `WanTransformer`),
or one replay stub set (`DiffusersReplayModelBase`). "These four pipelines
happen to return the same tuple" is a call shape.

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

### Removed, then restored

`Lumina2SamplingState`, `MochiSamplingState`, `PixArtSigmaSamplingState`,
`SanaSamplingState` and `JanusProARState` are empty subclasses of the shared
states. They were removed (`b00473cdf`) on the "no member, no dispatch"
test and reverted (`6c045386a`) on review: the family-named state is what a
reader sees in `forward_step(self, state: SanaSamplingState)` and in the
parity tests, and it ties the family to its shared state without a detour
through `sampling_state_cls`. Rule for next time: an empty subclass that
names the family at the signature level is documentation, not duplication.

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

## Batch 2: A1 and A2 that never used each other

The first batch caught identical bodies. This pass looked for the softer
case: two implementations of one operation that share a helper but each
still carry the rest of the logic. Method: cluster every function of eight
or more lines by the set of calls it makes (Jaccard >= 0.6 across different
modules), then read each cluster. Landed, one commit each:

- **Masked-prompt families** — `f3ebb5ff3`, then `dbeb242114`; **reversed
  2026-09-17** (see the boundary rule above). SANA, PixArt-Sigma, Lumina2
  and Mochi shared `MaskedPromptCollectorMixin` for the trajectory boundary
  but each still owned `encode_prompt`, `prepare_sampling` and
  `forward_step` (4 x ~150 lines). Landed first as
  `MaskedPromptModelMixin`, then folded with the collector mixin and
  `EncoderAttentionMaskRunnerBase` into one base, `MaskedPromptDenoiseModel`.
  Net: 632 lines removed, 301 added, and the base ended up with **eight
  extension points for four users** (`sampling_state_cls`,
  `branch_extra_kwargs` — pixart only, `_default_max_sequence_length`,
  `_default_guidance_scale`, `_pipeline_encode_kwargs`,
  `_backbone_output_dtype` — sana only, `_sampling_scheduler`,
  `_latent_shape_args` — mochi only, `_backbone_timestep`). A base that is
  mostly hooks is model glue in disguise: reading pixart required reading
  `masked_prompt.py`, and the fifth candidate (qwen_image: same
  "sequence + mask, no pooled" shape, but `encoder_hidden_states_mask` +
  `img_shapes`) already did not fit. The four families now list
  `DiffusersPipelineModelBase, DenoiseBackboneRunnerBase` (plus
  `VaeDecodeMixin` where decode is scale + shift) and carry their own
  glue; the constants that were class knobs are literals at the call that
  uses them, and the `num_train_timesteps` field lives only on the two
  states (lumina2, mochi) whose transformer clock needs it. The family
  sampling states keep their real fields instead of an empty subclass.
- **Fail-closed record parsers** — `512f2c143`. `GroundedOcrConfig`,
  `OcrScoringPolicy`, `ImageSampling`, `GeneratorRuntimeIdentity` each
  re-implemented "mapping whose keys are exactly my fields".
  `vrl.utils.validation.require_exact_dataclass_fields`.
- **Previous-adapter objectives** — `6080982b9`. DiffusionNFT and V-GRPO
  shared the flow-time normalization (`/1000` + EDM guard), the seeded
  double evaluation behind their lr=0 invariants, and the adapter refresh.
  `vrl/algorithms/previous_adapter.py`.
- **AR prompt tokenization** — Janus-Pro, LlamaGen, NextStep-1 each ran
  the same tokenizer call → `ARRequestLayout.right_pad` → device move.
  `ARRequestLayout.tokenize_right_padded`.
- **Ray actor cleanup** — `32f1ac074`. `kill_actors` returned failures and
  four owners re-wrote the same raise/`add_note`.
  `raise_if_kill_failures` / `note_kill_failures`.
- **Danbooru train/eval split** — anatomy and safety ran the same
  group → proportional counts → interleave pipeline over different keys.
  `manifest_rows.split_rows_proportionally`.
- **Per-rank batch-result fold** — `76a69a70c`. Engine and executor
  applied the same terminal-over-OOM-over-stale precedence.
  `generation.execution.types.combine_rank_batch_results`.
- **Canonical JSON digests** — `355d34a36`. Six sites spelled out
  `json.dumps(sort_keys, compact)` + `sha256`. `json_files.canonical_json_sha256`
  with `ensure_ascii`/`allow_nan` exposed so every existing digest is
  minted exactly as before. The packaged aesthetic asset now goes through
  `sha256_file` (`1ba3c45e6`, digest re-verified against the pin).

Clusters read and left: the pooled-embedding families' `forward_step`
(flux, hunyuan_*, sd3_5, qwen_image, cogvideox) already share
`DiffusionBackboneCaller`; what remains per family is the transformer's
kwargs, which is the family. `encode_video_to_latents` /`decode_latents`
across wan/anima/predict2/mochi stays as in batch 1 (pipeline-faithful
arithmetic order). `RewardInferenceConfig.from_mapping` is a different
contract from the exact-fields parsers (None/instance pass-through, unknown
keys only).

## Verification

Per change: ruff on touched files; the family's tests. Final sweep
(`CUDA_VISIBLE_DEVICES=""`): `tests/architecture tests/rewards
tests/scripts/eval tests/scripts/perf tests/models/families/{vdn_h3,minimax_h3}
tests/scripts/test_encode_sft_targets.py tests/scripts/test_anima_generate.py`
→ 854 passed, 7 skipped, 1 failed. The failure is
`tests/architecture/test_generation_rollout_boundaries.py::test_generation_model_imports_stay_on_public_floor`,
pre-existing from the parking consolidation (`memory_parking.py` importing
`vrl.models.parking`), not touched here.

Reversal sweep (2026-09-17, `CUDA_VISIBLE_DEVICES=""`):
`tests/models/families/{sana,lumina2,mochi,pixart_sigma} tests/models/steps/denoise
tests/models/interfaces tests/models/test_checkpoint_identity.py
tests/models/test_loader.py tests/architecture` (same deselect) → 416 passed,
1 skipped. The four backbone-parity tests are GPU-lane and were not re-run
here; their bodies are unchanged and compare the family forward against the
diffusers pipeline, so they are the check to run before the next GPU window.

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
