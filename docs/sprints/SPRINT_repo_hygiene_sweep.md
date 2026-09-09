# SPRINT: repo hygiene sweep

Executing the outstanding items of `planned/SPRINT_full_repo_bloat_audit.md` §8.2,
one reviewable commit each, under `GOAL_repo_hygiene_sweep.md`.

`kept` is a finished outcome: the item was examined and deliberately left alone,
and the reason is the deliverable. `blocked` means it needs something this run
must not do (a GPU numerical gate, a public config-key migration).

## Ledger

| # | item | source | verdict | evidence | commit |
|---|---|---|---|---|---|
| 1 | three lazy-export facades -> shared installer | §8.2 | changed | All 46 table entries had the value duplicate its own key, and the `__getattr__`/`__dir__` bodies were identical apart from a docstring. Public surface verified unchanged before/after: `__all__`, the `AttributeError` text, and torch-free import all identical; `dir()` differs only by module-private names that were never in `__all__`. | (this commit) |
| 2 | `vrl/algorithms/grpo/__init__.py` as a fourth facade | §8.2 neighbour | kept | Not the same machinery: it deliberately exports nothing (`__all__ = []`, 10 lines) because `vrl.config.algorithm` imports each objective by submodule. Nothing to share. | — |
| 3 | sana `from_build` hand-copies the shared loader | §8.2 | changed | Its first six steps were the shared `DiffusersPipelineModelBase.from_build` verbatim; only the DPM->FlowMatchEuler swap and the fp16 saturation clamp are SANA's. Ten sibling families already opt in by declaring the three attributes, which the base docstring names as the intended form. New tests in `test_model_base.py` pin the swap (shift preserved), the encoder placement and the clamp, and **pass identically against the pre-refactor code**. | (this commit) |
| 4 | reward inference default-config construction sites | §8.2 | kept | The two `registry.py` fallbacks are not duplication to fold: each docstring states they are the boundary for direct `MultiReward` construction, where every component is in-process. `builders.py` resolves per name with `.get(name, default)` and `schema.py` parses the typed section — three different shapes, one shared meaning already stated in prose. Folding them would need a named constructor on `RewardInferenceConfig` with two callers, which the thin-function rule does not justify. | — |
<<<<<<< HEAD
=======
| 5 | cosmos3 replay `set_num_steps` override | §8.2 | changed | The pinned Cosmos3-Nano scheduler config (`use_dynamic_shifting: false`, revision the preset pins) makes the shared body exact, and `Cosmos3ReplayModel` already overrides `scheduler` to return its own. Identical situation to `CosmosPredict25ReplayModel`, whose override the audit removed with the same comment. A new cross-family guard covers 15 replay classes and goes red when the scheduler property is pointed at the pipeline. | (this commit) |
| 6 | `tests/generation/execution/test_worker_sleep.py::test_real_cumem_one_shot_scope_sleep_wake_in_subprocess` fails under a concurrent GPU job | observed | blocked | Passes in isolation; fails in a full run while another process allocates on the card, because it bounds residual with a device-wide `mem_get_info`. Same root cause as the parking gate fixed in `f24d076f`, which now reads the driver's per-process table. Not in the audit queue and not caused by this sweep; recorded so the next run does not chase it as a regression. | — |
| 7 | wan `WanI2VReplayModel` MRO | §8.2 | changed | It restated five members of `WanT2VReplayModel` verbatim and hand-delegated two more, with a comment saying inheritance was avoided to keep the I2V forward math. Listing the t2v replay first and the i2v wrapper second gets both: an attribute-ownership diff over all 146 members shows exactly those seven moving and no forward method changing hands. Guarded in the wan loading tests, which go red when the inheritance is removed. | (this commit) |
| 8 | `retain_artifacts` | §8.2 | kept | Not scaffolding: it is a live config knob reached as `reward.kwargs.<name>.retain_artifacts`, and `tests/e2e/test_real_checkpoint_rl.py` sets it to keep reward artifacts for inspection. Removing a public config key is outside this sweep. |
| 9 | reward service `_execute` / `_revalidate_cached` | §8.2 | kept | Both are live handlers in `vrl/rewards/service/server.py` (the scoring path and the idempotent-replay path), called from the same dispatch. Only their `CancelledError` block is byte-identical; the other two handlers differ by error code and log line -- SCORING_FAILED versus INTERNAL_ERROR is the semantic difference that justifies two bodies, so folding them would delete the argument rather than the duplication. The audit rated this low value and it is. |
| 10 | scripts long-tail dead CLI flags | §8.2 | kept | The gate the audit installed answers it: `vrl.scripts.lint.dead_flags` reports **435 declared flags, all with a consumer** (350 at the audit's close). There is no long tail left to sweep. |
| 11 | `--vbench-*` decision | §8.2 | kept | `vbench==0.1.5` is a declared extra in `pyproject.toml`, deliberately isolated because it hard-pins `transformers==4.33.2`; the flags are consumed at `video_reward_suite.py:150-160`, and absence degrades to empty `vbench_*` columns plus a warning rather than an error. Live and already designed for the missing-extra case. |
| 12 | `init-dirs` | §8.2 | kept | Consumed by `tests/data/test_artifact_manifest_validation.py` and documented as a `vrl.scripts.data.setup` subcommand; its directory table was examined and kept by the ALL_CAPS audit for the same reason. |
<<<<<<< HEAD
>>>>>>> 613b9bba (docs(sprints): record five audit items examined and deliberately kept)
=======
| 13 | `_OFFLINE_DPO_*_FIELDS` cross-validation | §8.2 | changed (actor half) | The audit asked for derivation or a cross-check; derivation is out of scope because the list *is* the recipe's public config surface. A runtime cross-check instead: instrument attribute access on the actor section across the recipe's two pure-config resolvers and assert every allow-listed name is read. Adding an unread name to the list turns it red. Writing it found that a builder-only version reports a false positive on `gradient_checkpointing`, which `train_dpo.py` reads through `enable_transformer_gradient_checkpointing` -- the field is live and stays. The trainer half is left as-is: all seven of its names are read as plain `trainer.<name>` attributes and a symbol grep finds them, so they carry none of the `required("name")` rot risk. | (this commit) |
| 14 | precision `_select` -> `cfg_path` | §8.2 | kept | Resolved upstream: `fab177b7` ("delete the duck-typed accessors; every reader takes the parsed root") removed both symbols. Neither `_select` nor `cfg_path` exists in `vrl/` any more. |
| 15 | `model.lora.init` / `init_lora_weights` dual alias | §8.2 | blocked | Removing an alias is a public config-key migration, which this sweep must not do. |
>>>>>>> d0c87069 (test(config): the offline-DPO actor allow-list must have a reader per field)

## Notes for the reviewer

- The whole-repo audit's conclusion stands: there is no structural rot. These
  items are the ones its execution batches deliberately deferred, each with a
  recorded reason.
- One test failure is pre-existing on this machine and unrelated:
  `tests/nn/kernels/test_vllm_paged_attention_real_ops.py` fails by upstream
  design here.
