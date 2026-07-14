# Adding a Model Family

This guide describes the current model-family seam in `visual-rl`. A standard
diffusion family is a model module, one declarative registry entry, bundled YAML
presets, and contract tests. It does not need a family-specific runner, executor,
runtime builder, or config extractor.

The existing `sd3_5` family is the smallest complete example.

## 1. Choose the correct family shape

Use the descriptor-driven diffusion path when one trainable transformer and one
scheduler are enough to build rollout and replay runtimes. The shared
`DiffusionChunkExecutor` owns prompt, prepare, denoise, and decode orchestration;
`DiffusionFamilyBuild` tells the shared builders which model classes and upstream
transformer to load.

```text
vrl/models/diffusion/<family>/
  __init__.py
  model.py
```

Keep a family-specific `runtime.py` only when it provides a real execution
boundary, such as reference-conditioning payload preparation or a custom staged
executor. Wan I2V and the Cosmos families are examples. Do not add thin wrapper
builders that only forward constants to `vrl.models.diffusion.build`.

AR families have a different execution shape. They keep `model.py`, `runner.py`,
and `runtime.py` because their token-loop state, executor, and model-config
projection are family-specific. Bundle assembly and model-build resolution are
shared through `ARFamilyBuild`; mirror the nearest AR descriptor instead of
adding forwarding builders.

## 2. Implement the diffusion model contract

The rollout model normally subclasses `DiffusionModelBase` or
`DiffusersPipelineModelBase` and implements these generation methods:

- `from_build`: load the generation pipeline and freeze generation-only modules.
- `encode_prompt`: return the conditioning tensors used by the family.
- `prepare_sampling`: return a private state containing at least `latents`,
  `timesteps`, and `scheduler`.
- `forward_step`: run one trainable transformer step without stepping the
  scheduler.
- `decode_latents`: decode final latents to an image or video tensor.
- `export_batch_context`, `export_replay_tensors`, and `restore_eval_state`:
  project private rollout state into the shared trajectory contract and rebuild
  it for replay.

The trainer replay model should load only the trainable transformer and scheduler,
not prompt encoders, a VAE, or the full pipeline. `SD3_5ReplayModel` shows the
usual pattern: reuse the rollout model's math while replacing generation-only
ownership with a minimal constructor.

`DiffusionModelBase.replay_forward` rebuilds the recorded denoise state and calls
the current transformer. This replay path is the RL correctness boundary: its
distribution must match rollout-time sampling so `old_log_prob` remains valid.
Do not add algorithm code for a model family.

## 3. Register a descriptor-driven diffusion family

Add one `_diffusion_entry` in `vrl/families/registry.py`:

```python
_register_model_family(
    _diffusion_entry(
        family="my_model",
        task="t2i",
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.my_model.model:MyModel",
            replay_cls="vrl.models.diffusion.my_model.model:MyReplayModel",
            transformer_classname="MyTransformer2DModel",
        ),
    ),
)
```

Use the canonical family name in `cfg.model.family`. Functional conditioning
such as a reference image or video belongs on each `GenerationInput`; it is not
registry metadata or an executor-constructor setting. Set `scheduler_classname`
when replay must load a scheduler other than the shared flow-match scheduler.
Set `requires_lora=True` only when the model implementation genuinely rejects
full-parameter training.

External aliases live only in `vrl/families/names.py`. Add one there when
an existing public spelling must remain accepted; do not copy aliases onto the
runtime entry.

If one checkpoint supports multiple runtime protocols, each experiment must
still name the exact registry entry (for example, `janus_pro_r1`, not
`janus_pro` plus an algorithm-based inference rule). The algorithm validates
compatibility; it never rewrites the configured family.

`executor_cls` is intentionally absent above, so `_diffusion_entry` selects the
shared `DiffusionChunkExecutor`. Add a family executor only when its body performs
family-specific work; a renamed pass-through executor is not an extension point.

## 4. Add bundled config layers

Add model defaults and at least one experiment under the packaged preset tree:

```text
vrl/config/presets/model/diffusion/my_model.yaml
vrl/config/presets/experiment/diffusion/my_model/online_grpo_<reward>.yaml
tests/quality/protocols/families/my_model.yaml
```

Mirror
`vrl/config/presets/experiment/diffusion/sd3_5/online_grpo_ocr.yaml` and replace
only the model and sampling layers. Algorithm, rollout, reward, and distributed
layers remain shared. Keep checkpoint paths portable: use a Hub ID, a repository-
relative canonical path, or an explicit runtime override, never a contributor's
home directory. The CLI uses the logical name without the package prefix or extension:

```bash
vrl-train --config experiment/diffusion/my_model/online_grpo_<reward>
```

## 5. Extend the contract matrix and tests

The shared protocol matrix in `tests/models/interfaces/__init__.py` derives
rollout and replay model classes from the registry descriptor. A standard
descriptor family therefore needs no second hand-maintained class table.

Add family tests that cover:

- lightweight model construction and upstream loading arguments;
- `RuntimeModel` and `ReplayModel` protocol conformance through the shared matrix;
- rollout-to-replay tensor/context projection;
- backbone parity when the family wraps an upstream transformer;
- any custom executor branch, if the generic executor is insufficient.
- a test-owned inference profile for every supported checkpoint identity,
  native vs production comparison mode, and true/false corruption cases.

Run the CPU-safe structural gates before any real-model experiment:

```bash
CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m vrl.config.lint
CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest \
  tests/rollouts/runtime/test_family_registry.py \
  tests/quality \
  tests/models/interfaces -q
```

## 6. Promotion bar

A family is **Runnable** after its config, runtime, rollout, and replay contracts
pass and it produces an artifact. Mark it **Validated** only after a real training
run proves optimizer updates, a non-flat reward, generated artifacts, and changed
weights. Record the reproducible run under `docs/runs/` before changing the README
status.

## Checklist

- [ ] `model.py` implements generation and replay state projection.
- [ ] the replay class owns no generation-only modules.
- [ ] one registry entry uses `DiffusionFamilyBuild` and the shared builders.
- [ ] the test-owned quality profile names every supported immutable checkpoint
      identity without adding quality state to the production registry.
- [ ] every independent model dependency (text encoder, tokenizer, VAE) has
      its own revision field; never reuse the primary repository's commit.
- [ ] no family `runtime.py` or executor exists without real custom semantics.
- [ ] model and experiment presets live under `vrl/config/presets/`.
- [ ] `FAMILY_MODEL_CLASSES` and family-specific tests cover both model classes.
- [ ] config, registry, interface, and parity tests pass.
- [ ] a real run clears the promotion bar before the README says **Validated**.

See [`docs/NORTH_STAR.md`](NORTH_STAR.md) for why the model-family seam is a core
project asset.
