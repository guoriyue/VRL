# Adding a Model Family

This guide shows how to make a diffusion or autoregressive image/video model
RL-trainable in `visual-rl`. Start from the thinnest supported path. A standard
diffusers-backed diffusion family is one model module, one registry descriptor,
config layers, and tests. Do not create a family `runtime.py`, executor, runner,
or training entrypoint unless the model has behavior the shared path cannot express.

## 1. Choose the integration path

### Descriptor-driven diffusion family

Use this path when the family has one trainable transformer, a diffusers pipeline,
and the shared diffusion rollout/replay lifecycle. `sd3_5` and `sana` are the
reference implementations.

```text
vrl/models/diffusion/<family>/
  __init__.py
  model.py
```

The registry supplies the generic runtime spec extractor, runtime/replay builders,
executor, gatherer, and training entrypoint. Per-family executor values such as
`num_frames` or `max_sequence_length` belong in `configs/model/.../executor`, not
in Python constants.

### Custom diffusion adapter

Keep a thin `runtime.py` or custom executor only for a real protocol difference:
reference-image request construction, a non-diffusers artifact loader, multiple
transformers, or family-specific chunk behavior. Examples are Wan I2V, Cosmos,
and Echo. The adapter should contain only that difference and reuse the generic
builders for everything else.

### Autoregressive family

AR families have genuinely different token/cache protocols and therefore keep a
model, runner, and runtime adapter. Mirror the closest family (`janus_pro`,
`nextstep_1`, `emu3`, `glm_image`, or `llamagen`) and register its rollout and replay
builders. Training still uses the shared `vrl.scripts.ar.train:train_ar_grpo`
entrypoint.

## 2. Implement the model contract

The correctness boundary is `replay_forward(request)` on the runtime model. The
rollout records a denoising or token trajectory; replay recomputes the current
policy signals for the same state and step. With unchanged weights, fresh and
collection-time log-probs must agree. Never overwrite or reconstruct
`old_log_prob` from the current policy.

For a descriptor-driven diffusion family, mirror `SanaModel` and its replay model:

- `from_spec` loads the upstream pipeline and freezes generation-only modules;
- `encode_prompt`, `prepare_sampling`, `forward_step`, and `decode_latents` own
  family/upstream behavior;
- `build_branch` maps the family transformer signature into the shared CFG caller;
- the replay class restores the recorded state and implements family scheduler
  details without loading text encoders or the VAE.

Do not add algorithm code. Algorithms consume the shared replay signals.

## 3. Register a descriptor-driven diffusion family

Add one `_diffusion_entry` in `vrl/rollouts/families/registry.py`:

```python
register_rollout_family(
    _diffusion_entry(
        family="my_model",
        task="t2i",
        aliases=(),
        runtime_builder="vrl.models.diffusion.build:build_family_runtime_bundle",
        runtime_spec_extractor=(
            "vrl.models.diffusion.build:extract_family_runtime_spec"
        ),
        request_prefix="my_model",
        default_task_type="text_to_image",
        supports_reference_conditioning=False,
        build=DiffusionFamilyBuild(
            model_cls="vrl.models.diffusion.my_model.model:MyModel",
            replay_cls="vrl.models.diffusion.my_model.model:MyModelReplayModel",
            transformer_classname="MyModelTransformer2DModel",
            scheduler_classname=None,
            task_variant="t2i",
            memory_owner="MyModel VAE",
        ),
    ),
)
```

Leave `executor_cls` unset to use `GENERIC_DIFFUSION_EXECUTOR`. Set
`scheduler_classname` only when replay must load a non-flow scheduler such as DDIM
or UniPC. Set `supports_reference_conditioning=True` only when the production
request actually consumes a reference image.

If the descriptor cannot express the family, keep the same registry shape but
point only the necessary fields at a custom adapter. Do not duplicate generic
bundle assembly in that adapter.

## 4. Add config layers

Model, sampling, and experiment configs are separate axes:

```text
configs/model/diffusion/<family>/<variant>.yaml
configs/sampling/<modality>/<shape>.yaml
configs/experiment/diffusion/<family>/online_grpo_<reward>.yaml
```

The experiment should compose the shared recipe and use the generic diffusion
training entrypoint:

```yaml
trainer:
  entrypoint: vrl.scripts.diffusion.train:train_diffusion_grpo
```

Keep checkpoint paths portable. Repository configs must use a Hub ID, a relative
canonical path, or an explicit runtime override; never commit a user home path.

## 5. Validate the family

Add tests for behavior, not directory shape:

- model/upstream backbone parity at the same prompt, seed, schedule, and precision;
- rollout-to-replay log-prob parity;
- descriptor runtime and minimal replay-bundle wiring;
- config load and validation;
- one opt-in real-checkpoint update through `tests/e2e/test_real_checkpoint_rl.py`.

Run the repository gates, then the real recipe:

```bash
ruff check .
python -m vrl.config.lint
pytest -m "not e2e and not slow_test" -q
vrl-train --config experiment/diffusion/my_model/online_grpo_<reward>
```

A family is **Runnable** when real weights load, rollout/replay parity passes, and
an optimizer step changes trainable weights. Mark it **Validated** only after a
reproducible held-out evaluation shows a non-flat learning signal and the run
record is committed under `docs/runs/`.

## 6. Checklist

- [ ] Choose descriptor, custom diffusion adapter, or AR adapter from actual model behavior.
- [ ] Implement rollout and replay against the upstream scheduler/transformer contract.
- [ ] Add one registry entry; use generic builders/executor/entrypoint where possible.
- [ ] Add model, sampling, and experiment config layers without user absolute paths.
- [ ] Add backbone, replay-parity, config, and real-checkpoint update coverage.
- [ ] Run the validation gates before claiming Runnable.
- [ ] Publish a reproducible learning curve before claiming Validated.

See [`docs/NORTH_STAR.md`](NORTH_STAR.md) for the product rationale behind this
contract.
