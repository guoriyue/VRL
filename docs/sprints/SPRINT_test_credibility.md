# Test credibility audit

Status: in progress (2026-09-12). The scope is the current VRL test suite,
not just the uncommitted OCR tests. No coverage-count or deletion quota.

## Acceptance criteria

- Retained tests protect an identifiable runtime behavior, numerical invariant,
  input boundary, or real failure. Their claims must not exceed their execution.
- Remove redundant checks and speculative cases without weakening assertions,
  tolerances, or production behavior to obtain a passing suite.
- Keep model computation, gradient direction, rollout/replay parity, effective
  configuration, and checkpoint recovery protection where it is meaningful.
- Do not introduce production seams for tests or launch long GPU training.
- Verify each batch and distinguish reviewed scope from unaudited scope.

## Starting state

There were 365 `test_*.py` files. Existing uncommitted changes were confined to
the Anima reward-search handoff, OCR reward implementation/preset, and OCR tests.
Those changes belong to earlier work and are preserved. The working tree uses
`ocr_match`, not the `ocr_exact` name in older experiment notes.

The supplied AGENTS instructions apply; no on-disk AGENTS.md was present.
Prior cleanup reports under `docs/sprints/done/` are historical evidence, not
proof that today's tests are correct.

## Batch 1: completed

### Removed

- `tests/architecture/test_docstring_truth.py`: 159 lines of stop words,
  stemming, similarity heuristics, and tests of that heuristic. The only
  consumers were this file's own prose-style gate. It protected wording,
  not runtime correctness. No replacement style framework was added.
  The originating sprint now explicitly records retirement.
- `test_model_family_registry_stays_import_light` in
  `tests/architecture/test_generation_rollout_boundaries.py`: a static check
  that ignored every non-`vrl` import, including `torch`. The existing
  `tests/rollouts/runtime/test_family_registry.py::test_model_section_imports_do_not_load_model_runtimes`
  imports the registry and all model/sampling sections in a real subprocess
  and checks that heavy dependencies and runtime modules were not loaded.
  Retain that stronger check instead of maintaining two definitions.
- The same scanner's unused `allow_path_prefixes` argument, branch, and
  `_is_relative_to` helper: no caller supplied an exception prefix.
- A stale conftest comment claiming the optional lane had no members.
  Kling, RAFT, and VideoScore2 tests actually use it; the lane is unchanged.

### Corrected

- `tests/algorithms/test_multisegment_token_grpo.py`: the supposed zero-weight
  test previously disabled selfcheck before reaching the weight gate. It now
  enables that segment, uses valid negative categorical log probabilities,
  and checks that making its weight positive changes the loss. The weighted
  aggregation example also uses negative log probabilities with the same
  old/new differences. No production math was changed.
- `tests/rollouts/replay/test_multisegment_token_logprob.py`: the fake discarded
  `batch` and rebuilt a hardcoded token table. It now reads the actual action
  tensor and supplies only model logits, exercising the evaluator's trajectory
  token lookup. Deleted the duplicate payload table and forwarding method.
  Assertions cover both segments' temperature, actual old probabilities and
  masks, and the trajectory primary even when enabled order differs.
  The original log-prob comparison tolerances are preserved explicitly.

### Retained, with limits

- OCR tests execute real reward arithmetic over supplied recognized text.
  They do not prove recognition quality or RL improvement. Both Paddle result
  layouts are genuine dependency-adapter boundaries. Duplicate-line alignment
  cases exercise different edit operations, not arbitrary invalid inputs.
  The existing engine-injection seam is documented by earlier design work;
  no replacement worker-config interface or deep private access was introduced.
- Token log-prob math compares the chunked implementation against full
  log-softmax. Storage-policy tests keeping integer token IDs intact protect
  a real dtype boundary; floating log probabilities and masks are legitimate.
- Import-layer checks and their shared scanner protect dependency boundaries.
  Remaining module-level protocol/import-boundary tables are not business
  vocabularies. Do not flatten the scanner to save lines.
- `real_cover` has actual labels and a CLI report consumer. It checks whether
  references and explanations exist, not whether real model tests were run.
  Its existence is not evidence that every labeled double is trustworthy.

### Verification

- Before removal: prose gate, static import gate, and real subprocess import
  check: 4 passed. OCR plus grounded OCR baseline: 37 passed, 1 skipped.
- Token/replay worker baseline: 46 passed; changed two files: 6 passed.
  In-memory mutations replacing trajectory token IDs or selecting the last
  enabled segment as primary were caught by the corrected tests.
- Combined CPU verification: 123 passed, 1 skipped, 2 deselected:
  architecture, family registry, multisegment GRPO/replay, token log-prob math,
  token GRPO, trajectory storage, OCR, and grounded OCR. Real OCR remained
  opt-in. The two Ray packaging traversal tests were deselected.
- Ruff check/fix, format, check, and format-check passed for the four touched
  Python files. No production source changed in this batch.

## Batch 2: configuration, checkpoint behavior, and test isolation

### Removed or merged

- Removed reward filename/layout checks and their three sole-use filename
  helpers from `tests/architecture/test_generation_rollout_boundaries.py`.
  `vrl/rewards/functions/registry.py::_register_builtins` uses explicit imports;
  `vrl/rewards/runtime.py::InProcessRewardScorer._ensure_model` resolves a
  configured factory path. Neither scans for `<registry-key>.py`. Extra shared
  modules are not a runtime error. The old directory-cleanup sprint now records
  this retirement; import-layer boundaries remain.
- Removed the synthetic conditional-pytestmark test in
  `tests/architecture/test_real_cover_labels.py`. An AST scan found no actual
  conditional pytestmark assignments. The parser's generic AST traversal is
  unchanged; no syntax-specific fallback or replacement framework was added.
- Removed the Cosmos target-video config test that built a fake video, report,
  and manifests without any consumer. Its recipe does not enable production
  validation, so `gate_production` returns before reading provenance. General
  experiment loading already checks its actual behavior.
- Removed the schema I2V test that only parsed a task string while claiming
  production-contract validation. Existing contract and Wan I2V provenance
  tests actually execute that validation and remain.
- Removed three signature/export/source-text checks in
  `test_validation_tiers.py`; retained direct-root validation, parse-versus-
  launch separation, and real subprocess torch-free config parsing. Deleted
  the unused `_kling_video_reward_kwargs` helper.
- Merged the weak Adam moment-only restore test into the existing real disk
  save/restore/second-update test in `test_state_restore.py`, retaining step
  and global-step assertions and deleting the sole-use moment extractor.
- Merged profiling manifest assertions into the existing CPU trace test. The
  smoke function already reads the saved JSON; the second identical profiling
  run added no different execution path. All field assertions remain.

### Corrected

- `test_schema.py` falsy projection uses valid LoRA rank/alpha and positive
  frame/text lengths. False, None, empty lists, and zero dropout still protect
  presence preservation; zero-rank models are not needed for that invariant.
- `test_state_restore.py` now restores a different model weight separately
  from trainer counters and checks the pre-collect sync payload.
  `test_trainable_state.py` checks initial and post-update payload values,
  rather than only counting pushes.
- `test_checkpointing.py` saves weights 3 then 7 into the same directory and
  verifies 7 after loading. Previously identical saves could pass if the
  second save did nothing. CPU tensor-tree comparisons remain CPU tests;
  cross-device comparisons now have an explicit GPU marker.
- `test_lazy_exports.py` imports its own target before reading its source.
  Running only its TYPE_CHECKING checks reproduced three KeyErrors before the
  change and three passes after it. No assertion was relaxed.

### Scope and retained boundaries

- Read all 13 `tests/config/test_*.py` files and shared helpers against the
  configuration implementation. Changes above are completed; additional
  candidates below are not silently counted as resolved.
- Read checkpointing, online state restore/trainable state, precision-drift,
  and trust-region tests against their production paths. Real EMA/raw-state,
  owned-buffer, atomic publication, and parity protections remain. External
  checkpoint progress has an explicit integer protocol, so representative
  float/string/bool rejection is not an imagined internal token case.
- Reviewed the five utils test files. JSON atomic writes and media conversion
  use actual files/tensors; NVML doubles replace hardware observations while
  retaining actual PID selection and fail-closed behavior. The real CuMem GPU
  test remains opt-in to this CPU validation. Profiling push/pop doubles are
  limited to emission bookkeeping, not claims about observed GPU traces.
- Keep package-export and subprocess tests: these check actual lazy facades,
  not arbitrary function parameter names. Keep shared Ray lifecycle fixtures,
  independent expected capability matrices, and protocol/schema constants.

### Verification

- Directory-gate baseline: 2 passed. After its removal, architecture and
  selected reward construction/runtime tests: 54 passed, 2 deselected.
- Config directory: 337 passed before, 332 passed after removing five tests.
- Checkpoint/state batch: 127 passed, 3 deselected before and after. Removing
  the redundant Adam test and separating CPU/GPU comparisons offset in count.
  In-memory fault injection confirmed zeroed sync payloads and a no-op second
  save are rejected. No persistent mutation scripts or production edits.
- Architecture/utils: initially 7 errors from missing `pynvml`, despite
  `nvidia-ml-py` being a core declared dependency. Installed lockfile version
  13.610.43 into `.venv`; rerun: 71 passed, 3 deselected. No skip was added to
  hide the dependency failure. One upstream CPU profiler warning remains.
- Changed Python files passed scoped Ruff check/fix, format, check, and
  format-check. Combined CPU verification across config, utils, architecture,
  checkpoint/state/precision/trust-region, and selected reward runtime tests:
  **597 passed, 6 deselected**, 15.78 seconds. GPU/distributed/slow lanes and
  the two Ray packaging traversal tests were excluded. The 15 warnings were
  upstream torch profiler and TorchScript deprecation warnings. This is not
  a whole-repository result.

## Batch 3: real config projection and script boundaries

- Collection-mode overrides now pass through `load_config` and `build_configs`
  before asserting the trainer field. Previously the test only constructed the
  target dataclass, so dropping the YAML projection would remain green.
- Dataset presets come from `list_bundled_configs("dataset")`, not a five-name
  table. Removed the copied loader enum and private registry initialization.
  The 24 actual assets include a defaults-based DrawBench variant: an initial
  raw-YAML check correctly exposed missing inheritance in the test itself.
  Use the public `load_config` composition path, not a special-case whitelist.
- Three algorithm recipe tests now build the full typed bundle once; removed
  their duplicate build-only cases. V-GRPO keeps its distinct build check.
  The family-registry presence fixture uses one sampling step, preserving its
  meaningful zero guidance, False, None, and empty-list checks.
- Removed the trust-region class-flag mirror; existing trainer construction
  acceptance/rejection tests still exercise all three algorithms' behavior.
- Anima's generated-image substitute now records actual prompt/seed/count
  inputs and compares them with saved metadata. This checks the CLI handoff,
  not real diffusion output. An in-memory mutation passing seed+1 while leaving
  metadata unchanged failed the corrected test.
- Online metrics now use `TrainStepMetrics`, not four hand-maintained namespace
  objects. The checkpoint progress assertion was stale: baseline failed on
  missing `next_step`. Retained exact equality and added this required field,
  after checking save, `TrainingCheckpoint.next_step`, and restored policy
  version consumers. This is a contract correction, not tolerance relaxation.
- Supervisor cleanup removes a same-import constant comparison and a child
  process test solely checking absence of the retired `attempt_id` field.
  Merged the normal-gradient/no-verdict assertion into the existing normal-to-
  consecutive-spike test. Actual process groups, retry, rank outcomes, and
  checkpoint selection protections remain. Runtime filename constants and
  the shared subprocess fixtures retain their genuine protocol roles.

Verification: config/registry selected baseline 86 passed; trust-region
baseline 7 passed; four targeted supervisor checks passed before cleanup and
the resulting complete supervisor file passed all 56 checks. Script pair
baseline was 24 passed/1 failed (stale next_step); after correction 25 passed,
including real two-rank Gloo. Scoped Ruff stages and diff whitespace checks
passed. No production changes or persistent fault-injection scripts.
Combined batch-3 CPU run: **464 passed**, 23.14 seconds, covering all config
tests, family registry, trust-region guards, Anima generation CLI, online
metrics, and the complete supervisor file. CUDA visibility was disabled;
GPU/distributed/slow markers were excluded. The unmarked Gloo and supervisor
subprocess tests ran for real.

The script audit read `test_anima_generate`, `test_common_factory`,
`test_online_metrics`, `test_online_run_config`, and `test_supervise` against
their implementations. Remaining findings: the factory's empty-LoRA test
claims unchanged model output but only inspects an option dictionary; Anima's
image-float conversion test belongs with shared media tests; some supervisor
private-counter assertions and config layout-only policies still need review.

## Batch 4: model behavior instead of configuration claims

- Removed the factory test claiming a fresh Wan adapter preserves base output
  based only on `lora_config.get("init_lora_weights", True)`. Its claim now
  lives in the existing Wan PEFT test: a real CPU Linear projection is compared
  before and after the real `WanT2VDiffusersModel.apply_lora` call, with exact
  equality and effective dropout checks. No full Wan model or image-quality
  result is claimed. A process-local mutation disabling zero initialization
  was rejected by this comparison; production source was never changed.
- Moved CHW float image conversion from the Anima script tests to shared media
  tests. It already called only `image_to_uint8_hwc`; retain shape/dtype and
  compare every output pixel, not just the maximum red value.
- `test_loader.py` now uses the actual CPU FlowMatch scheduler for Sana's
  `flow_shift` translation. Only external loading is substituted; real
  timesteps/sigmas must match shift=3 and differ from the original shift=1.
  Removed two handcrafted dependency classes and a duplicate revision test.
- Removed a precision getter identity-only test; real autocast cases still
  consume that getter. Removed an unreachable Wan branch in a checkpoint
  identity parameterized test whose inputs are three other families.
- Retained weight version eviction/aliasing, exact installed bytes, missing
  expert/frozen parameter checks, real precision recomputation/hook cleanup,
  and local-file/source identity tests. Signed zero, NaN, float64, and bf16 are
  explicit byte-preservation cases, not invented token inputs. Versioned torch
  API doubles remain valid for the supported dependency range.
- Keep `_build` versus `_token_build` fixture boundaries and checkpoint commit
  fixture constants. They encode different construction contracts; combining
  them into a switch-heavy builder would make the tests harder to read.

Verification: Wan/factory/Anima/media baseline 50 passed, after 49 passed;
loader/precision/weight-utils/identity baseline 63 passed, after 61 passed.
Combined CPU run: **110 passed**, 2.72 seconds. Scoped Ruff four stages and
diff checks passed. No dangling references to the removed/renamed test names
were found in tests, source, docs, or CI. No GPU work or production edits.

## Batch 5: checkpoint failure coordination

Removed two rank-agreement doubles from `tests/trainers/test_checkpointing.py`:
publication failure on the sixth agreement and peer EMA swap failure on the
ninth agreement. These depended on the number of internal collectives rather
than injecting the actual fault. Existing tests in
`tests/trainers/test_fsdp_gather_distributed.py` inject an artifact-write failure
on rank 0 or an EMA swap failure on one rank, using real two-process CPU gloo,
FSDP shards, PEFT parameters, and checkpoint coordination. They verify both
ranks fail, raw weights are restored, and no checkpoint is published. The
distributed restoration assertion now also checks the temporary EMA snapshot
is cleared, preserving the removed local test's snapshot protection.

Keep the local rollback-failure test for now: it uniquely checks the original
exception cause and weights left at the EMA value. The distributed rollback
test currently checks propagation but not those two facts. Also keep the
primary-write cleanup test, which checks staging-directory cleanup. No new
fixture abstraction, production interface, or constant table was introduced;
the existing worker is a justified process-entry and shared setup boundary.

Verification: targeted baseline 5 passed; checkpoint and distributed files
after deletion 117 passed, 1 GPU test deselected, with 14 upstream JIT warnings.
Final rerun including the snapshot assertion: **117 passed, 1 deselected**,
23.23 seconds. Scoped Ruff checks and formatting passed; removed names have no remaining
references in tests, source, docs, or CI. No production code or GPU work.

## Remaining work

The goal is not complete. The remaining 364 test files have not all had their
bodies, producers, consumers, and claimed protections audited.

- Review memory-policy source-string checks against actual behavioral
  coverage before removing protection; do not replace them with a bigger
  source scanner.
- Config follow-ups: review layout-only recipe policies against actual loading
  consumers. Projection, preset inventory, duplicate recipe builds, and the
  zero-step presence fixture were addressed in batch 3.
- Checkpoint follow-ups: magic collective-call-count fault injection overlaps
  real gloo tests; verify equivalent failure paths before removing those
  doubles. The duplicate trust-region class-flag assertion was removed.
- Review configuration and script tests for mirrored defaults, duplicated
  dispatch logic, and mocked-away application of options.
- Review trainer/checkpoint tests for real state restoration, update paths,
  gradients, and parity rather than method-call assertions alone.
- Review model/generation/reward tests and their fixtures, including positive
  categorical log-prob examples elsewhere in token tests. Keep valid numerical
  invariants while making fixtures semantically realistic.
- Verify later batches and the broader CPU suite, documenting optional,
  distributed, GPU, and real-weight gaps rather than claiming full validation.

No commit, GPU training, checkpoint deletion, or unrelated cleanup was performed.

## Batch 4 (2026-09-12, second session): structural tests, same-shape merges, one stale edit finished

Worked alongside the batches above from a separate session; this batch only
touched files the earlier batches had not modified, except where noted.

### Removed / simplified

- `tests/trainers/test_ddp.py`: dropped `test_ddp_and_single_process_share_one_rollout_export`
  (asserted which class defines the method via `vars()`; the end-to-end
  key-space test next to it is the real protection).
- `tests/models/families/wan_2_1/test_model_loading.py`: dropped
  `test_i2v_replay_takes_plumbing_from_t2v_replay_and_forward_math_from_i2v`
  (MRO-owner assertions on eleven members, including a private one). The two
  offload-mode tests are now one two-row table; same assertions.
- `tests/trainers/test_strategy.py`: the checkpoint-optimizer-export contract
  test keeps the protocol and behavior checks and loses three `vars()` lines.
- `tests/utils/test_lazy_exports.py`: three installer unit tests folded into
  one; the `__dir__` sort-order assertion is gone. Package-level checks unchanged.
- `tests/rewards/test_model_hub.py` (3 -> 1 table),
  `tests/trainers/online/test_precision_drift_guard.py` (auto mode, 3 -> 1
  table), `tests/config/test_schema.py` (six `test_unknown_*_raises` -> one
  dotted-path table), `tests/config/test_new_algorithm_recipes.py` (2 -> 1).
- `vrl/models/families/vdn_h3/model.py`: removed the diagnostics-only
  `hybrid_blocks` property; its single reader was the hybrid-attention test,
  which now counts `vendor.iter_hybrids(...)` directly. Executed against the
  declared dependencies (see below): vdn_h3 + minimax_h3 26 passed.
- `tests/scripts/eval/test_sana_checkpoint_compare.py`: dropped
  `test_model_precision_snapshot_records_effective_backend` -- it patched
  `float32_precision_state` to a literal dict and asserted the snapshot
  contained that dict (a mock echo). The sibling
  `..._records_materialized_dtypes` reads real module dtypes and stays.

### Finished an in-flight edit

`tests/scripts/test_online_lifecycle.py` had been switched to a pure recorder
`_FakeSchedule` (batch 2/3) but five tests still asserted the old cascaded
counts and were red. They now pin `state["shutdown_order"]`, which is the
witness the recorder actually records, and the two error-injection tests use
`schedule_shutdown_raises`. Production `online.py` is unchanged; HEAD's
version of the file passes 27/27, and so does this one.

### Environment finding: the venv is behind the declared dependencies

`pyproject.toml` declares `diffusers>=0.40.0,<0.41`; `.venv` has 0.39.0 and
no `pynvml`. With the declared versions on `PYTHONPATH` (installed into a
scratch target dir, venv untouched) every group previously written off as
"environmental" passes: `test_scheduler_logprob_parity` (vdn/minimax),
`cosmos3/test_backbone_parity` (`return_dict`), `tests/utils/test_cuda_memory.py`
(NVML), and `test_online_metrics::..._threads_required_model_identity`
(52 passed). The vendored `third_party/vdn-minimax-h3` submodule was not
checked out either; it now is.

The two GPU tests also pass on the shared 5090 (a user render job was
resident; the CuMem test measures per-process memory via NVML, so it does not
need the card to itself):
`test_real_cumem_parking_with_another_process_allocation` 1 passed;
`test_vllm_paged_attention_writes_real_cuda_kv_cache` 1 passed. The latter's
failure was never an ABI mismatch: `vllm._C` loads and vllm declares
`torch==2.11.0`, matching the venv. The venv's vllm install is simply missing
~100 of vllm's own declared dependencies (`cbor2`, `gguf`, `openai`, ...).
`vrl/nn/layers/attention/paged.py` reports every import failure as "wheel does
not match the PyTorch/CUDA ABI", which sent this audit down the wrong path
twice -- worth rewording to include the underlying exception. Completing the
install inside the venv is NOT safe: uv's resolution would downgrade
`transformers` 5.13 -> 4.57 and `huggingface-hub` 1.23 -> 0.36 and add
`tokenspeed-triton` 3.8 (known to break VRL); the scratch `--target` dir with
those excluded is how the test was run.

### Per-test evidence for the gradient / parity / checkpoint files

64 files, 497 tests, profiled per test (numeric assertion / raises / state /
backward / monkeypatch targets / call-record assertions): 148 numeric, 175
raise-based, 165 state, 9 with no assertion of their own (fixtures raising).
The 42 without numeric or raise evidence that also patch or assert on call
records were each read: 41 pin real behavior (which tensors and dtypes reach
the real transformer, unscale-before-clip witnessed by gradient magnitude,
strategy-seam routing, non-primary ranks joining the collective, eval scripts
generating base before restore), one was the echo above.

### Scans run over all 365 files (evidence for what was left alone)

Normalized-AST duplicate bodies (34 groups; three merged above, the rest are
same shape / different theorem), monkeypatch-constant echo (5 candidates, all
false positives), private-only assertions (7, all pin real behavior),
production defs referenced only from tests (4: `hybrid_blocks` removed;
cosmos `NoOpCosmosSafetyChecker.check_*` is a diffusers-called interface;
`minimax_h3.final_audio_rows` / `decode_audio` have no production caller —
a production dead-code candidate, out of this audit's scope). Gradient /
parity / checkpoint files were profiled for real tensor ops, backward calls,
numeric asserts, and patched targets; every family backbone-parity file runs
a real tiny transformer with `assert_close`, and the two gradient files that
patch collaborators (`test_grad_scaler`, `test_offline_dpo_timesteps`) assert
decisions and combination arithmetic, not the patched values.

## Batch 5 (2026-09-12, second session): environment drift and the structural repeats

### Environment: `.venv` re-synced to the lock

`uv sync --frozen --group test --group lint --extra cosmos --extra reward
--extra reward-service --extra data --extra detection --extra ocr`, then the
editable `third_party` install CI performs. Net effect on the venv: diffusers
0.39 -> 0.40, `rtmlib` (undeclared, unused by `vrl/`) and the broken `vllm`
(declared conflict with `cosmos`; needs its own `ar-vllm` environment) removed,
the `data`/`reward` extras installed. `transformers` 5.13 and release `triton`
3.6 untouched. README's setup command referenced a non-existent `dev` extra;
it now shows the CI command and names the two conflicting extras. CLAUDE.md
records the rule.

### Structural repeats

- **Duplicated doubles.** Of the 24/23/20 same-named `_request`/`_model`/
  `_state` helpers, only five pairs were verbatim duplicates (same normalized
  AST); the rest are same-name-different-shape. Under the repo's 3+-files
  rule only one qualified: the Janus request/rows shared by three replay
  evaluator tests, now `tests/rollouts/replay/_helpers.py`. Two-file pairs
  (`_executor`, sd3_5 `_model`, `_state`, `_TinyTransformer`) stay
  self-contained on purpose.
- **Constructor bypasses.** Nine `object.__new__(OnlineTrainer)` sites across
  three files collapsed onto `bare_trainer(**attrs)` in
  `tests/trainers/online/_helpers.py`. The remaining bypasses are single-file
  or family-fixture cases (cosmos replay export, nextstep fixture, wan
  pipeline error, family MRO) and stay where they are.
- **Theorem-per-module.** `tests/trainers/test_checkpointing.py` (2387 lines,
  85 tests mixing four subjects) is now four modules --
  `test_checkpointing.py` (save/restore/identity/publication),
  `test_checkpoint_adapter_export.py`, `test_checkpoint_schema_restore.py`,
  `test_checkpoint_cli.py` -- over one `_checkpoint_helpers.py`. Collected ids
  are identical to the pre-split file (112).
- **Implicit CUDA.** `tests/conftest.py` gained an autouse fixture pinning
  `torch.cuda.is_available()`/`device_count()` to no-GPU for every test not in
  the `gpu` lane. Tests that model a GPU host still patch on top; the eight
  files that already patched explicitly are unaffected.
