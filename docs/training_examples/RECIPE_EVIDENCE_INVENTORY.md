# Recipe evidence inventory — 2026-09-09

Source revision: `c78b3d34211f00bb9928d40110f90c3fba28611f`.
The [machine-readable snapshot](recipe_evidence_inventory_20260909.json) includes
every experiment path, resolved-config digest, configured artifact observations,
manifest presence, real-checkpoint case overrides, and the existing real-cover
labels. It is a dated observation, not a support matrix or validation registry.

## Scope and findings

The inventory enumerated `vrl/config/presets/experiment/**/*.yaml`, composed each
preset with `vrl.config.loading.load_config` without extra overrides, and inspected
its configured output directory relative to the repository. It imported the
existing e2e `CASES` and used `tests.real_cover.collect_labels` to inspect test
metadata. It did not execute a model, search arbitrary external run archives,
or infer historical environment identities. Presence alone is not integrity proof.

- 75 experiment YAMLs were found. 73 compose with their current defaults.
- Two Anima templates require user composition: `online_grpo` needs learning rate,
  data, reward, and output directory; `online_grpo_fullparam` needs data, reward,
  and output directory. They are unresolved templates, not failed training runs.
- None of the successfully composed presets' configured output directories had
  `metrics.csv`, launch records, checkpoint metadata, or completed evaluation
  markers. This does not say that no historical runs exist elsewhere.
- Five presets reference at least one absent manifest/report, listed by path in
  the snapshot. Three need `droid_overfit_validation.jsonl`; two need the DROID
  full-targets text-to-video train/eval/source-report export.

| Family | Experiment files |
| --- | ---: |
| cogvideox | 1 |
| cosmos-predict2 | 11 |
| cosmos-predict2.5 | 7 |
| echo | 1 |
| emu3 | 1 |
| flux | 5 |
| glm_image | 1 |
| hunyuan_image | 1 |
| hunyuan_video | 1 |
| janus_pro | 2 |
| janus_pro_r1 | 2 |
| llamagen | 1 |
| lumina2 | 1 |
| minimax_h3 | 1 |
| mochi | 1 |
| nextstep_1 | 1 |
| pixart_sigma | 1 |
| qwen_image | 1 |
| sana | 7 |
| sd3_5 | 6 |
| unresolved template | 2 |
| vdn_h3 | 1 |
| wan | 15 |
| wan_2_1_i2v | 4 |

## Existing tests are not recipe curves

`tests/e2e/test_real_checkpoint_rl.py` contains 12 cases. The real test performs
one online update, checks trainable parameter changes, finite loss/gradient,
reward variation and applicable rollout/replay logprob parity. The executor runs
through a direct test runtime, not the production Ray topology. Case-specific
resolution, step count, guidance, prompt count, and precision overrides are
preserved in the snapshot rather than described as equivalent to full recipes.

Eleven cases substitute the reward: most use an index reward; configured transport
cases replace its model with a tensor-mean factory. Only
`cosmos_predict2_kling_real_reward` retains its configured reward model.
`cosmos_anima` and `cosmos_anima_safe` also use synthetic replay rollouts.
None of this inventory establishes that any opt-in GPU case ran on this host.

The existing `real_cover` labels link doubles to real counterparts or tracked gaps;
the snapshot preserves their reasons and lanes. Labels and a GPU marker are not
execution evidence. The checked-in CI workflow runs `pytest -m "not e2e"` in the
unit job and explicitly documents the opt-in GPU/distributed coverage gap. There
is no repeated full-recipe GPU curve gate established by that workflow.

## SD3.5 OCR reference candidate

Use `experiment/sd3_5/online_grpo_ocr` as the initial real image reference, with
explicitly recorded overrides. The local SD3.5 Medium snapshot revision is
`b940f670f0eda2d07fbb75229e779da1ad11eb80`: 45 readable files, approximately
48.9 GB logical size, and no observed broken symlinks. This is a cache availability
check, not full checkpoint content validation or a successful model load.
The train/eval OCR manifests exist; their exact hashes and line counts are in the
snapshot. Line counts are not validated prompt counts.

Current launch prerequisites are incomplete: the active Python environment has
neither `paddle` nor `paddleocr`, and the one RTX 5090 is shared with other live
processes. The existing SD3.5 smoke case checks at least 24 GiB total device memory;
it does not check currently free memory. That guard is neither a reservation on
this shared GPU nor a measured full OCR recipe memory requirement. No other
process was stopped and no GPU training was launched during this inventory.

The checked-in May 7 qualitative example is useful historical context, but its
200-epoch output directory is not present here. Its note says 8 samples per prompt;
the current preset uses 16. It also describes a historical runtime, while today's
preset enables transformer compile. Re-running the present config under that
old note's command cannot reproduce the old configuration by assumption. Keep the
historical note unchanged; it is not a current baseline or a locked metric report.

## Required next evidence

Provision an isolated OCR-capable environment and an available GPU, then run the
same explicitly pinned SD3.5 recipe twice with independent output directories.
Record all proxy changes if memory requires reduced resolution, steps, or batch
size. A short proxy verifies only that declared scope. Preserve attempt/launch
records, metrics, final checkpoints and independently configured evaluation
archives; the new integrity checks can associate these artifacts but cannot create
learning evidence. Full curves and deterministic/statistical regression remain
unproven until actual runs and comparisons exist.
