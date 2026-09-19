# Runtime configuration composition

Keep reusable choices independent. An execution recipe owns the model/runtime
constraints; reward presets own scoring; dataset presets own manifests and
sampling. Training budget, learning rate, and output location are run arguments,
not reasons to create another experiment YAML.

## Compose at launch

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5/online_grpo_pickscore \
  +reward=ocr +dataset=ocr \
  actor.optim.lr=1e-5 trainer.total_epochs=2 \
  trainer.output_dir=outputs/sd3_5_ocr_composed
```

This illustrates configuration composition, not a recommended recipe. The
standard OCR dataset must exist at the paths declared by `dataset/ocr`;
composition does not generate it. Model-by-reward combinations, coefficients,
learning rates, and run lengths belong in the user's launch command or
external run config, not another checked-in experiment YAML. LoRA and
full-transformer entrypoints stay separate where a family ships both (e.g.
`experiment/sd3_5/online_grpo_ocr_fsdp_2x1_fullparam`), because the two have
different reference-model and memory constraints; do not approximate one
with the other by changing only `model.use_lora`.

The central loader processes:

1. The `--config` source and its own `defaults`.
2. Each `+group=option` preset, in command-line order, including that preset's
   own defaults. For example, `+model/cosmos=predict2_2b` selects a nested
   model group, while `+sampling/image=512` selects image geometry.
3. All ordinary `section.field=value` overrides, last, regardless of their
   position among preset arguments.
4. Interpolation resolution and mandatory-value validation.

Missing presets and malformed selections fail. This is additive composition:
selecting two reward components retains both; selecting two overlapping policies
for the same component makes later values win. It does not remove existing keys.
Start new combinations from a neutral recipe, not a historical experiment that
already embeds another reward or data policy. Legacy `/group=option` replaces
matching defaults entries and is a different mechanism; it is not required for
the workflow above.

Both the training CLI and supervisor forward this same argument list. There is
one loader, not a second configuration registry or per-model launcher.

## LoRA configuration

Adapter settings live under `model.lora`; `model.use_lora` selects adapter
training versus full-parameter training. Model presets keep their existing
rank, alpha, and target-module lists. Override those at launch, rather than
adding a new model/reward preset:

```bash
model.use_lora=true model.lora.rank=32 model.lora.alpha=64 \
model.lora.dropout=0.0 model.lora.init_lora_weights=gaussian
```

`ModelBuild.lora` resolves these settings into the shared `LoraSection`, then
the shared model calls PEFT's `LoraConfig` and `get_peft_model` directly.
There is no second adapter manager. This follows the separation in
[Miles Diffusion's PEFT setup](https://github.com/radixark/miles_diffusion/blob/ebd55fc1e597520322e0225997552c0807e9293b/miles/backends/fsdp_utils/actor.py#L603):
shared run inputs, with real family defaults kept separate.

The lightweight family schemas own defaults: Gaussian initialization and
PEFT adapter upcasting normally; Wan and CausVid retain `init_lora_weights=true`,
and Wan retains `autocast_adapter_dtype=false` for weight-sync dtype parity.
Both initialization choices start with zero B weights; the A distribution
and RNG recipe differ. `path` loads a trainable PEFT adapter after topology
validation. `parameter_dtype: float32` explicitly stores trainable adapters
in FP32; with FSDP it requires `distributed.training.fsdp.precision_policy=none`.
`autocast_adapter_dtype` is PEFT's storage upcast option, not forward autocast.

Migration from the former layout:

| Old setting | Current setting |
| --- | --- |
| `model.lora_parameter_dtype` | `model.lora.parameter_dtype` |
| `model.nft_previous_adapter` / `model.lora.previous_adapter` | Remove; the algorithm contract requests the previous policy |
| `model.lora.init` (never consumed) | Remove it; use `model.lora.init_lora_weights` for initialization |

The removed spellings are rejected, not silently aliased. The persisted v1
checkpoint identity retains its original canonical keys, so this spelling
change alone does not invalidate existing checkpoints. Historical run
snapshots retain their original configuration and need these key migrations
if relaunched. An explicit non-default adapter-upcast policy changes identity.

The algorithm contract, not a model recipe, owns the previous-policy and
reference-policy requirements (`requires_previous_policy`,
`requires_reference_policy`; GRPO's reference follows `kl_coef > 0`). The
model serves both from `DenoiseModelBase.previous_policy()` /
`reference_policy()` as snapshots of whatever is trainable, so a LoRA adapter
and a full fine-tune run the same objective code: NFT and V-GRPO admit
`model.use_lora=false`. The objectives evaluate the re-noised clean latent
through the shared `replay_forward_with_latents` (the same family forward the
SDE replay uses, with classifier-free guidance forced off), so no family
declares an objective-specific forward hook or capability flag: any
full-sequence family with a trainer replay recipe is admitted, and a timestep
grid that does not normalize into `[0, 1]` (Cosmos Predict2's EDM grid) fails
at the first loss. Only SD3.5, Flux, and Predict2.5 have been exercised on
this path with LoRA; the full-parameter path is covered by CPU tests only.

The previous policy is a snapshot taken at the first sync and refreshed after
every optimizer step; it is never checkpointed (a resumed run starts it from
the restored weights). The reference is the pre-training policy: under an
adapter the base weights (nothing copied), otherwise a snapshot the trainer
takes after sharding and before any checkpoint restore. Neither snapshot is
part of checkpoint identity.

## Validation tiers

`model.memory` fields and types are validated by the shared schema. The family
registry does not maintain a second list of allowed memory fields. Support is
checked when building the rollout model: CPU residency must be honored by the
loader, and VAE decode options require an actual VAE memory target. Unsupported
options therefore fail during model construction, potentially after loading
weights. MAGI-1 rejects these options at its subprocess builder boundary.

After composition every entrypoint parses the merged config through the same
typed boundary, and validation is split into three tiers, one module and one
registry each. Where a new check goes is decided by what it needs:

| Tier | Module | Runs from | Needs | Examples |
| --- | --- | --- | --- | --- |
| 1. Section shape | `vrl/config/schema.py` (pydantic) | `parse_config` | the section itself | closed keys, types, `rollout.sde.type` membership, `data.manifest` required by loader |
| 2. Cross-section rules | `vrl/config/rules.py`, `check_cross_section_rules` | `RootConfig`'s validator (so also `parse_config`) | two or more parsed sections, nothing else | `algorithm.kind` needs `rollout.sde`; `algorithm.sft_weight` needs `data.sft_latents`; offline DPO's consumed surface |
| 3. Launch gates | `vrl/config/validation.py`, `TRAINING_GATES` | `require_training_config` (training launches only) | the precision policy or a runtime module | torch.compile compatibility matrix, unguarded rollout drift. Filesystem checks (dataset provenance via `DatasetProvenance.from_config`, `vrl/trainers/data/provenance.py`; reward backends) run in `python -m vrl.scripts.rewards.preflight` |

Tier 2 must stay import-light because eval and perf tools pay for it on every
parse; a check that needs `vrl.trainers` or `vrl.models.interfaces` is a tier 3
gate. Resolution-time validation that needs a resolved object (the GPU
topology in `vrl/ray/resources.py`, the rollout schedule, reward parking) stays
with the resolver that produces that object and runs after `build_configs`.

Whether the configured rewards can actually score the configured rows (a
reward that reads a target clip or a caption target off each prompt) is not a
config question, so no gate answers it. Run the reward once, before training,
on the same rows and metadata projection the collector will use:

```bash
python -m vrl.scripts.rewards.preflight --config experiment/wan_2_1/online_grpo_kling_video_reward \
  --prompts 4 --device auto
```

It builds every configured component, scores synthetic media of the configured
geometry for the first rows of `data.manifest` (`--eval` for the eval
manifest), prints per-component scores, and exits non-zero on the first
component that raises. The scores themselves mean nothing; the pipeline does.

## Composing rewards

For several independent rewards, select each component and set its coefficient
explicitly, for example `+reward=aesthetic +reward=pickscore`, then
`reward.components.aesthetic=0.3 reward.components.pickscore=0.7`. Existing
resource, parking, and reward validation still apply. Shared placement fields
use the final overlay's value; composition does not create per-component GPU
isolation. See the [reward configuration guide](../vrl/config/presets/reward/README.md).

OCR scoring options belong under `reward.kwargs.ocr`, independently of the
generator. The shared `reward/ocr` preset declares the defaults; select an
engine and text-matching policy explicitly when an experiment needs different
semantics. Optional debug artifacts use
`'reward.kwargs.ocr.debug_dir=${trainer.output_dir}/reward_debug'` (quote shell
interpolation). Retired OCR datasets and qualification rules are historical experiment
evidence, not additional requirements for the shared OCR reward.

## Compose an independent evaluation policy

The shared image checkpoint evaluator loads generator identity and sampling
from the run's recorded config. Compose its rewards and held-out data at launch,
without creating a model/reward-specific evaluator or training-experiment YAML:

```bash
python -m vrl.scripts.eval.image_checkpoint_eval \
  --run-dir outputs/sd3_5_color_light_composed \
  --eval-policy-config reward/wd_tagger \
  --eval-policy-override +reward=pickscore \
  --eval-policy-override +dataset=anime_craft \
  --strata bucket prompt_style --per-stratum 6 \
  --samples-per-prompt 2 --seed 91000 \
  --checkpoint candidate=outputs/sd3_5_color_light_composed/checkpoint-final \
  --dry-run
```

This requires the recorded run/checkpoint and evaluation data to exist. Policy
overrides affect only held-out data and judging; they cannot silently replace
the evaluated generator or sampling settings. If no policy config is supplied,
overrides apply to a copy of the saved run config. Resolved judging/data content
is recorded in the evaluation protocol, together with input hashes. Prompt
metadata is passed through to the selected rewards, including OCR, tagging,
and object-count targets; reward choice is not hardcoded in the evaluator.
Scores retain every component and the weighted `r_total`. A batched judge must
use `images_per_call=1` for independent judgments or enough cells for all arms
(base plus checkpoints). If a reward declares `expected_group_size`, it must
match that arm count. Set these through `--eval-policy-override` as needed;
the evaluator rejects incompatible training group settings before generation
instead of silently changing the judging protocol. Nested reward recording
destinations are removed so scoring cannot append to a training debug archive.

With no checkpoint selector, all complete checkpoints are discovered by their
recorded epoch, deduplicating `checkpoint-final`. Use `--epochs 4,8,16` for a
subset or repeat `--checkpoint LABEL=PATH` for explicit candidates. Base is
always generated first. This entrypoint supports registered full-sequence,
native-step text-to-image denoisers, not reference-conditioned or autoregressive
models. SANA's frozen official-solver benchmark and the video-specific benchmark
protocols remain separate: sharing statistics must not change their sampling.

Each evaluation writes `images/` and a content-bound `generation_manifest.json`.
If scoring fails after generation finishes, rerunning the same command reuses
the verified images without loading the generator. A successful run atomically
publishes `report/` containing scores, `summary.json`, `curve.csv`, `curve.png`,
blinded contact sheets and a separate `blind_key.json`. Completed reports cannot
be overwritten. Changed settings or an incomplete generation require a new
`--output-dir`; older family-specific archives are not silently migrated.

For a new plot of existing scores, no generator or reward model is needed:

```bash
python -m vrl.scripts.eval.score_report \
  --scores outputs/sd3_5_color_light_composed/checkpoint_evaluation/report/scores.jsonl \
  --score-key wd_tagger \
  --output-dir outputs/color_light_curve
```

The model-independent input is JSONL with `checkpoint_label`, integer `epoch`,
`prompt_index`, `sample_index`, `seed`, `prompt`, and `r_<component>` scores.
Checkpoint arms must share the exact prompt/sample/seed grid. The report first
averages samples within each prompt, then bootstraps prompt means and paired
deltas against base. Multiple seeds of one prompt are not independent prompts.
Image statistics and seed-diversity curves are diagnostics, not quality rewards;
higher saturation, brightness, or pixel distance is not automatically better.
Human review is still needed, particularly when evaluation reuses a training
reward. Historical video reports retain their original per-cell statistics.

## Reproducibility and historical presets

Every actual run saves its composed `resolved_config.yaml`. Resume attempts save
their own `resume_config_*.yaml` without overwriting the original. Checkpoints,
evaluation manifests, scores, and their hashes remain the experiment evidence.
Reproducing a historical run uses those recorded settings, not today's neutral
defaults with a similar name.

The `cosmos-predict2-anima` family, its two experiment entrypoints, its
standalone generator, the `anima_*` dataset presets and the `anima_*`
evaluators were removed on 2026-09-18. Sprint reports under `docs/sprints`
keep the historical commands; the anime datasets and the image evaluators
remain generator-independent. The `codex_image_qa` LLM-judge reward and its
anime rubric presets were removed on 2026-09-19: its test-retest agreement
(0.1-0.5, ties on ~60% of prompts) made it unusable as a GRPO reward.

An earlier reward audit retired the family-specific person-critic canary and
the unavailable production critic entrypoint. The offline person-critic research
chain has also been archived: dedicated source, tests, dataset presets, protocol
assets, person-count/integrity datasets, and the rejected Luna person-count rubric
are no longer part of the active repository. Historical evidence is retained in
the sprint report. Retired code/config/text-data paths identify members in
`/home/mingfeiguo/Desktop/vrl-person-research-archive-jB20H6/before.tar.gz`.
Moved datasets, media, and probes retain their repository-relative layout under
that archive directory's `files/`; see its `README.md` for the exact inventory.

CountGD person counting, grounded OCR, and tag adherence remain available. They
accept image artifacts and task metadata independently of the generator; no
family-specific person research dataset is required by the framework. Generator independence
does not establish reward accuracy or resistance to reward hacking on every image
distribution.

A later evaluation cleanup archived the OCR qualification reporter,
its dedicated dataset generators, five OCR dataset presets and their data, the
rejected requested-token grounding rubric, the anatomy probe/report chain, and
the old objective-C tag-adherence evaluation entrypoint. The shared tag reward
and its separate NSFW datasets and run evidence remain; the later composition
cleanup retired the model/reward combination YAML, not those assets.
Shared OCR and grounded-OCR rewards, text-matching logic, and the standard
`dataset/ocr` preset remain available. Reusable checkpoint, fixed-panel, and
exact-count evaluation also remain; an experiment-specific qualification script
is not a framework dependency. See the
[evaluation cleanup inventory](/home/mingfeiguo/Desktop/vrl-eval-cleanup-archive-3AI2zQ/README.md)
for archived paths and recovery instructions. Historical sprint commands refer
to the archived versions, not current launch entrypoints.


### Explicit data and run-result names

- `load_prompt_dataset_index` reads the existing prompt dataset index format.
  Existing dataset paths and YAML keys retain their spelling.
- `vrl.scripts.data.video_world.dataset_index` builds Video2World dataset rows.
- SANA evaluation shares `SANA_EVAL_SAMPLING_CONFIG` and
  `SANA_EVAL_SCHEDULER_CONFIG`. Checkpoint comparison writes
  `evaluation_record.json` and returns its path as `evaluation_record`.
- `TrainingRunResultWriter` writes `training_run_result.json`, or
  `training_run_result.rank-N.json` under torchrun. The outcome field is
  `status` (`success`, `failed`, or `terminated`); aggregated worker records
  use `rank_results`. The supervisor reads the new names only. Historical
  `run_verdict.json` files are not rewritten automatically.


### Continuous rollout components

`ContinuousRolloutThread` manages the dedicated thread and event loop.
Its `_ContinuousRolloutController` coordinates producer, queue, consumer,
and weight synchronization on that loop.

`ContinuousRolloutProducer` tracks each batch in `_PromptBatchProgress`
and asynchronous tasks in `_running_tasks`. `PendingRewardCapacity`
reserves group/byte capacity from generation admission through reward completion;
it stores accounting, not rollout payloads.

Completed `ScoredRollout` records enter `ScoredRolloutQueue`.
`ContinuousRolloutConsumer` retrieves the requested complete batch and returns
the shared `RolloutIteration` type. Configuration keys and metric names are
unchanged.
