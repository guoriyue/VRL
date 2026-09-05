# Runtime configuration composition

Keep reusable choices independent. An execution recipe owns the model/runtime
constraints; reward presets own scoring; dataset presets own manifests and
sampling. Training budget, learning rate, and output location are run arguments,
not reasons to create another experiment YAML.

## Compose at launch

```bash
python -m vrl.scripts.train \
  --config experiment/anima_preview3/online_grpo \
  +reward=ocr +dataset=ocr \
  actor.optim.lr=1e-5 trainer.total_epochs=2 \
  trainer.output_dir=outputs/anima_ocr_composed
```

This illustrates configuration composition, not a recommended Anima training
recipe. The standard OCR dataset must exist at the paths declared by
`dataset/ocr`; composition does not generate it. The neutral Anima entrypoint
deliberately requires reward, data,
learning rate, and output directory, and defaults to one training iteration.
Choose a longer budget explicitly at launch. It retains the measured
eager execution, generation/replay batch-1, and parity gates. Use
`experiment/anima_preview3/online_grpo_fullparam` for the distinct single-GPU
full-transformer memory contract; do not approximate that contract by changing
only `model.use_lora`.

These are the only two bundled Anima training entrypoints. Neither selects a
reward or dataset. Model-by-reward combinations, coefficients, learning rates,
and run lengths belong in the user's launch command or external run config,
not another checked-in experiment YAML. The two entrypoints remain separate
because LoRA and full-transformer updates have different reference-model and
single-GPU memory constraints.

The central loader processes:

1. The `--config` source and its own `defaults`.
2. Each `+group=option` preset, in command-line order, including that preset's
   own defaults. For example, `+model/cosmos=anima_preview3` selects a nested
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

## Judge, rubric, and data are separate choices

The `codex_image_qa_anime_*` names identify anime-oriented scoring rubrics;
they do not implement another reward model or require Anima weights.
Reuse the same rubric for another anime generator rather than copying it into
that generator's experiment directory. Judge identity and durable rollout
recording remain independent overlays:

```bash
python -m vrl.scripts.train \
  --config experiment/anima_preview3/online_grpo \
  +reward=codex_image_qa \
  +reward=codex_image_qa_anime_color_light \
  +reward=codex_image_qa_luna_scored \
  +dataset=anima_color_light_ddrl \
  algorithm.sft_weight=0.001 sampling.num_steps=40 \
  actor.optim.lr=2e-5 trainer.total_epochs=1 trainer.save_freq=1 \
  trainer.output_dir=outputs/anima_color_light_composed
```

The DDRL dataset refers to generated anchor images and encoded clean latents;
those assets must exist before training. Selecting a dataset does not generate
them. The SFT coefficient and step count above are explicit experiment choices,
not properties of the judge. Saved old results do not establish that this new
combination improves quality.

The single color-and-light corpus lives in `datasets/anima/color_light/`: 256
training prompts, 96 independent evaluation prompts, and 32 development prompts
retained from earlier experiments. Development prompts are not part of the
formal evaluation split. DDRL expects newly prepared anchors and latents under
`data/external/anima/color_light/`; the historical 64-prompt anchor set does not
cover this merged training corpus.

The remaining `dataset/anima_*` names identify real Anima-generated reference
images/latents and their prompt manifests, not model/reward combinations. Those
reference assets are generator-specific; changing the model does not make its
training latents interchangeable. The rubric and judge remain independent.

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
interpolation). Retired Anima OCR datasets and qualification rules are historical
experiment evidence, not additional requirements for the shared OCR reward.

## Anima generation outputs

The standalone Anima generator writes one reusable archive, not a debug dump:

- `images/*.png`: generated images.
- `run_config.json`: model/adapter identity, sampling settings, and runtime provenance.
- `metadata.jsonl`: image-to-prompt mapping, batch seeds, and reward metadata for evaluation.
- `anchor_manifest.jsonl`: the same images as clean targets for the SFT encoder and DDRL datasets.

Paired evaluation reads and checks this archive, and the target encoder consumes
the anchor manifest. The duplicate `metadata.csv` export has been removed;
`metadata.jsonl` is the canonical image index. Existing archives remain readable.
Output-directory ownership checks prevent one generation run from overwriting
or relabeling another. None of these files saves another model checkpoint.

## Compose an independent evaluation policy

The quality evaluator continues to load generator identity and sampling from
the run's recorded config. Its separate evaluation policy can now be assembled
without a training-experiment YAML:

```bash
python -m vrl.scripts.eval.anima_codex_quality_checkpoint_eval \
  --run-dir outputs/anima_color_light_composed \
  --eval-policy-config reward/codex_image_qa \
  --eval-policy-override +reward=codex_image_qa_anime_color_light \
  --eval-policy-override +reward=codex_image_qa_luna \
  --eval-policy-override +dataset=anima_color_light_ddrl \
  --prompts-per-bucket-style 6 \
  --checkpoint candidate=outputs/anima_color_light_composed/checkpoint-final \
  --dry-run
```

This requires the recorded run/checkpoint and evaluation data to exist. Policy
overrides affect only held-out data and judging; they cannot silently replace
the evaluated generator or sampling settings. If no policy config is supplied,
overrides apply to a copy of the saved run config. Resolved judging/data content
is still hashed, and overlay arguments are recorded in policy provenance.

## Reproducibility and historical presets

Every actual run saves its composed `resolved_config.yaml`. Resume attempts save
their own `resume_config_*.yaml` without overwriting the original. Checkpoints,
evaluation manifests, scores, and their hashes remain the experiment evidence.
Reproducing a historical run uses those recorded settings, not today's neutral
defaults with a similar name.

The Anima reward/data/run-length combinations retired on 2026-09-04 are not
replaced with compatibility YAMLs or a JSON table of the same combinations.
Their historical commands in sprint reports describe past runs, not current
launch entrypoints. In particular, the rejected full-parameter quality and
exact-count experiments are not promoted by this migration.

All remaining Anima reward combinations, including aesthetic, general-quality,
and safety variants, have now been retired. Their tests compose independent
presets instead; the generation CLI defaults to the model preset and does not
load a training reward. The three anime rubric presets were renamed from
`codex_image_qa_anima_*` to `codex_image_qa_anime_*`, without changing scores or
adding compatibility aliases. See the
[final Anima composition inventory](/home/mingfeiguo/Desktop/vrl-anima-composition-archive-VIWBEO/README.md).
Other model families' established recipes are outside this migration.

The subsequent reward audit retired the Anima-specific person-critic canary and
the unavailable production critic entrypoint. The offline person-critic research
chain has also been archived: dedicated source, tests, dataset presets, protocol
assets, person-count/integrity datasets, and the rejected Luna person-count rubric
are no longer part of the active repository. Historical evidence is retained in
the sprint report. Retired code/config/text-data paths identify members in
`/home/mingfeiguo/Desktop/vrl-person-research-archive-jB20H6/before.tar.gz`.
Moved datasets, media, and probes retain their repository-relative layout under
that archive directory's `files/`; see its `README.md` for the exact inventory.

CountGD person counting, grounded OCR, tag adherence, the shared Codex exact-count
scoring mechanism, and reusable exact-count evaluation remain available. They
accept image artifacts and task metadata independently of the generator; no
Anima person research dataset is required by the framework. Generator independence
does not establish reward accuracy or resistance to reward hacking on every image
distribution.

The subsequent evaluation cleanup archived the Anima OCR qualification reporter,
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
