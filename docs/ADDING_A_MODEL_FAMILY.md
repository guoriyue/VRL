# Adding a Model Family

This guide describes the current model-family seam in `visual-rl`. A standard
full-sequence denoise family is a family-owned model module, one declarative registry
entry, bundled YAML presets, and contract tests. It does not need a
family-specific runner, executor, runtime builder, or config extractor.

The physical layout is family-first: family code lives under
`vrl/models/families/`, while reusable model machinery lives under
`vrl/models/steps/denoise/`. Generation code is separated into policy steps,
batch execution, and concrete bindings under
`vrl/generation/{steps,execution,bindings}/`. Do not add new `models/ar`,
`models/diffusion`, `generation/ar`, or `generation/diffusion` paths.

The `vrl/models/families/` package root owns the import-light registry, names,
and policy semantics; its `<family>/` subdirectories own concrete implementations.

Every registry entry classifies the trainable policy by its
`generation_regime` (`full_sequence` or `chunk_autoregressive`). See
[`MODEL_TAXONOMY.md`](MODEL_TAXONOMY.md).

The existing `sd3_5` family is the smallest complete example.

## 1. Choose the correct family shape

Use the descriptor-driven denoise path when one trainable transformer and one
scheduler are enough to build rollout and replay runtimes. The shared
`GenericDenoiseBatchExecutor` in the `full_sequence` binding owns prompt, prepare,
denoise, and decode orchestration; `DenoiseFamilyBuild` tells the shared builders
which model classes and upstream transformer to load.

```text
vrl/models/families/<family>/
  __init__.py
  model.py
```

Keep a family-specific `runtime.py` only when it provides a real execution
boundary, such as reference-conditioning payload preparation or a custom staged
executor. Wan I2V and the Cosmos families are examples. Do not add thin wrapper
builders that only forward constants to `vrl.models.steps.denoise.build`.

## 2. Implement the denoise model contract

A family has two classes. The model class is what the trainer replays. It
subclasses `DiffusersModelBase` (or `DenoiseModelBase` when the family has no
diffusers transformer) and holds the transformer, the scheduler and the
per-step math:

- `forward_step`: run one trainable transformer step without stepping the
  scheduler.
- `restore_eval_state`: rebuild the private state from trajectory tensors for
  replay.

The sampling class subclasses the pipeline base and the model class, as in
`class MyPipelineModel(DiffusersPipelineModelBase, MyModel)` (or
`DenoisePipelineModelBase` when the family has no diffusers pipeline), and adds
what only sampling needs:

- `from_build`: load the generation pipeline and freeze generation-only modules.
- `encode_prompt`: return the conditioning tensors used by the family.
- `prepare_sampling`: return a private state containing at least `latents`,
  `timesteps`, and `scheduler`.
- `decode_latents`: decode final latents to an image or video tensor.
- `export_batch_context` and `export_replay_tensors`: project private rollout
  state into the shared trajectory contract that `restore_eval_state` reads
  back.

The replay builder constructs the model class from the transformer and
scheduler alone; `SD3_5Model` and `SD3_5PipelineModel` show the usual pair.
Replay therefore never loads prompt encoders, a VAE or the pipeline, and
never inherits a sampling member it would have to switch off.

`DenoiseModelBase.replay_forward` rebuilds the recorded denoise state and calls
the current transformer. This replay path is the RL correctness boundary: its
distribution must match rollout-time sampling so `old_log_prob` remains valid.
Sampling and replay run the same `forward_step`, inherited from the model class.
Do not add algorithm code for a model family.

## 3. Register a descriptor-driven full-sequence denoise family

Add one `_full_sequence_entry` in `vrl/models/families/registry.py`:

```python
_register_model_family(
    _full_sequence_entry(
        family="my_model",
        task="t2i",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.my_model.model:MyPipelineModel",
            replay_cls="vrl.models.families.my_model.model:MyModel",
            transformer_classname="MyTransformer2DModel",
        ),
    ),
)
```

Use the canonical family name in `cfg.model.family`. Functional conditioning
such as a reference image or video belongs on each `GenerationInput`; it is not
registry metadata or an executor-constructor setting. Set `scheduler_classname`
when replay must load a scheduler other than the shared flow-match scheduler.

When replay cannot be expressed as one diffusers transformer plus one scheduler
(a non-diffusers loader, a second scheduler, a weight graft), omit
`transformer_classname` and give `MyModel` its own `from_build(build)`; the
registry then calls it and applies LoRA, compile and precision as usual. Honor
`build.materialize_weights` there only if the loader can build a meta skeleton.

External aliases live only in `vrl/models/families/names.py`. Add one there when
an existing public spelling must remain accepted; do not copy aliases onto the
runtime entry.

If one checkpoint supports multiple runtime protocols, each experiment must
still name the exact registry entry rather than relying on an algorithm-based
inference rule. The algorithm validates compatibility; it never rewrites the
configured family.

`executor_cls` is intentionally absent above, so `_full_sequence_entry` selects
`vrl.generation.bindings.full_sequence.executor:GenericDenoiseBatchExecutor`. Add a
family executor only when its body performs family-specific work; a renamed
pass-through executor is not an extension point.

For a temporal-chunk policy, use `_chunk_autoregressive_entry` with an
explicit family executor under `vrl.models.families.<family>.runtime`. Do not
infer the generation regime from the checkpoint or selected algorithm.

## 4. Add bundled config layers

Add model defaults and at least one experiment under the packaged preset tree:

```text
vrl/config/presets/model/my_model/default.yaml
vrl/config/presets/experiment/my_model/online_grpo_<reward>.yaml
```

Mirror
`vrl/config/presets/experiment/sd3_5/online_grpo_ocr.yaml` and replace
only the model and sampling layers. Algorithm, rollout, reward, and distributed
layers remain shared. Keep checkpoint paths portable: use a Hub ID, a repository-
relative canonical path, or an explicit runtime override, never a contributor's
home directory. The CLI uses the logical name without the package prefix or extension:

```bash
vrl-train --config experiment/my_model/online_grpo_<reward>
```

## 5. Extend the contract matrix and tests

The shared protocol matrices in `tests/models/interfaces/test_runtime_model_contract.py`
and `tests/models/interfaces/test_replay_model_contract.py` derive rollout and
replay model classes from the registry descriptor. A standard
descriptor family therefore needs no second hand-maintained class table.

Add family tests that cover:

- lightweight model construction and upstream loading arguments;
- `RuntimeModel` and `ReplayModel` protocol conformance through the shared matrix;
- rollout-to-replay tensor/context projection;
- backbone parity when the family wraps an upstream transformer;
- any custom executor branch, if the generic executor is insufficient.

Keep family tests under `tests/models/families/<family>/`. Put reusable step
contract tests under `tests/models/steps/denoise/`, and generation tests
under the matching `tests/generation/{steps,execution,bindings}/` boundary.

Before the first costly image training run, use the test-owned rollout preview
with a real experiment config. It derives family support from `FAMILY_REGISTRY`,
reads the configured prompt manifest, runs the registered production executor,
and writes individual images plus the resolved settings. There is no per-family
preview adapter, profile YAML, scorer, or automatic visual-quality verdict.
Objective native/backbone/replay parity belongs in family-owned model and e2e
tests; a human reviews the preview images for prompt adherence, color blocks,
posterization, collapse, and other visible defects.

Run the CPU-safe structural checks before any real-model experiment:

```bash
CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m vrl.config.lint
CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest \
  tests/rollouts/runtime/test_family_registry.py \
  tests/quality \
  tests/models/interfaces -q
```

Then drive the registered production model and denoise math without Ray or a
trainer. The probe fails on non-finite/collapsed output and can verify the first
rollout-to-replay step with `--check-replay`:

```bash
python -m vrl.scripts.generation.full_sequence_probe \
  --family my_model --path <checkpoint> --dtype bf16 \
  --float32-precision tf32 --outer-autocast --check-replay
```

## 6. Promotion bar

A family is **Runnable** after its config, runtime, rollout, and replay contracts
pass and it produces an artifact. Mark it **Validated** only after a real training
run proves optimizer updates, a non-flat reward, generated artifacts, and changed
weights. Keep the run's config and metrics in its output directory and link it from
the family's README before changing the status; run records are not committed.

## Checklist

- [ ] `model.py` implements generation and replay state projection.
- [ ] the replay class owns no generation-only modules.
- [ ] one registry entry uses `DenoiseFamilyBuild` and the matching shared
      builders.
- [ ] registry `generation_regime` describes the trainable policy rather than the
      whole checkpoint (especially for hybrid or staged models).
- [ ] the real experiment YAML generates individual images through the shared
      rollout preview before a costly training run.
- [ ] a human reviews those images and records whether the configured rollout is
      fit for the planned training run.
- [ ] every independent model dependency (text encoder, tokenizer, VAE) has
      its own revision field; never reuse the primary repository's commit.
- [ ] no family `runtime.py` or executor exists without real custom semantics.
- [ ] model and experiment presets live under `vrl/config/presets/`.
- [ ] the registry-derived interface matrix and family-specific tests cover both
      rollout and replay model classes.
- [ ] config, registry, interface, and parity tests pass.
- [ ] the full-sequence denoise probe passes with the real checkpoint.
- [ ] a real run clears the promotion bar before the README says **Validated**.

The model-family seam is a core
project asset.
