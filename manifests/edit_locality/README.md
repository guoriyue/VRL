# Localized editing reward study

This small diagnostic set separates useful commercial editing requests: recolor
one garment part, recolor a selected furniture instance, change upholstery while
preserving shape, and replace a whole furniture item. It contains no no-op
requests or artificially degraded candidates. It is not a representative benchmark.

`tasks.json` defines ten original requests and one feedback retry, each with seeds
0–3. Development uses the armchair and sweater photographs; heldout uses two
previously unseen room photographs. The retry belongs to development and is
judged against its original instruction, not its longer generation instruction.

`region_checks.json` contains source-defined target/protected witness patches for
six color tasks, plus local views for a crop ablation. The patches were manually
chosen by the assistant. They are not segmentation masks or automatic localization.
A failing protected patch establishes a local color violation; passing all patches
does not prove that the entire image is correct. The v2 royal-blue hue was selected
during development and frozen before heldout inspection. Do not tune it to heldout
outputs. V3 subsequently corrects a source annotation mistake: the dining wooden-frame
patch overlapped upholstery. The frozen V2 file is retained; V3 results are exploratory
annotation-corrected results, not an untouched heldout evaluation. Material,
print-fidelity, and replacement tasks are not covered by these
color checks.

Copy these JSON files to `outputs/qwen_image_21_reward_study/` and place the four
original JPEGs under its `sources/` directory using the names in `sources.json`.
The source page URLs provide attribution; exact local input hashes are recorded
with the run. Full-resolution outputs, checkpoints, source photos, contact sheets,
and the standalone HTML report stay in the output directory.

Commands from the repository root:

```bash
# Generation environment: the lock-synced Qwen-Image-2.1 integration environment.
/tmp/vrl-qwen21-integration-venv/bin/python -m vrl.scripts.eval.qwen_edit_locality

# Separate upstream reward environment (Transformers 4.57.0, not the generation env).
HF_HUB_OFFLINE=1 /tmp/vrl-edit-reward-env/bin/python -m vrl.scripts.eval.edit_reward_baseline --reward editscore
HF_HUB_OFFLINE=1 /tmp/vrl-edit-reward-env/bin/python -m vrl.scripts.eval.edit_reward_baseline --reward editreward
HF_HUB_OFFLINE=1 /tmp/vrl-edit-reward-env/bin/python -m vrl.scripts.eval.edit_reward_baseline --reward editscore --view task_crop

/tmp/vrl-qwen21-integration-venv/bin/python -m vrl.scripts.eval.edit_color_constraints
/tmp/vrl-qwen21-integration-venv/bin/python -m vrl.scripts.eval.edit_reward_report
```

Every command accepts `--out`. Scorers accept `--upstream-root`, which defaults to
`/tmp/vrl-edit-research`; it must contain the official EditScore and EditReward
checkouts. All three model checkpoints and the shared Qwen2.5-VL base must already
be cached. Generation and scoring must run sequentially on the single 32 GB GPU.
Scorers resume by candidate name, so use a fresh output directory for changed
weights, rules, prompts, or settings. Do not reuse a scored name for a different image.

Before scoring new candidates, record score-blind visual review in
`visual_labels_<split>.json`, with a `labels` mapping from candidate name to
`joint_success` (`true`, `false`, or `null`) and `evidence`. Null labels are excluded
from binary success calculations. These run labels are assistant judgments, not
independent human preference labels. Tied rewards use the lowest seed; the report
also lists all tied winners and gives pairwise ties half credit.

The color correction is `min(EditScore / 10, target_color, protected_color)`.
EditReward retains its original unbounded score and is compared by within-request
ranking, never by treating its magnitude as an EditScore-equivalent percentage.
No policy weights are trained by these commands.
