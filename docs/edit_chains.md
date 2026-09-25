# Edit chains (`agentic`)

An edit chain is one source image and an ordered list of editing instructions:
the multi-step form of an image-editing task. The `agentic` package runs chains
for evaluation and trains the editor on them. It depends on `vrl`; `vrl` never
imports it.

## Manifest

One chain per JSONL line:

```json
{"chain_id": "page-12", "source": "pages/12.png",
 "steps": ["Change the coat to blue. Keep everything else unchanged.",
           {"prompt": "Replace the second bubble's text with HELLO.", "target_text": "HELLO"}],
 "requirement": "Keep the character's face and the background unchanged.",
 "reward_assets": {"target_image": "pages/12_target.png"},
 "metadata": {"series": "demo"}}
```

- A step is an instruction string or a prompt-manifest row without conditioning
  media; unknown step keys become that step's reward metadata.
- `requirement` and `reward_assets` reach the reward as metadata (the Edit-R1
  `requirement` shape); `metadata` is shared by every step and a step's own keys win.
- Paths resolve relative to the manifest. `chain_id` defaults to `<stem>:<line>`.

## Run

`run_chain(chain, editor, judge, output_dir=...)` edits step by step (each step on
the previous output), then the judge scores every state (source plus each
output) in one call. It writes `run.json`: the chain, each step's output
artifact, `state_scores`, and the final score.

## Train the editor

```bash
python -m agentic.scripts.train_edit_chains --config experiment/qwen_image_21/<recipe> \
  --chains chains.jsonl trainer.output_dir=outputs/chains
```

The config is an ordinary `vrl` training config; the chains replace its prompt
rows. Each chain runs as one `run_chain` with `GroupEditor` as both editor and
judge: every step generates `n_samples_per_prompt` candidates through the
rollout collector, one is drawn uniformly (driver RNG, checkpointed with the
run) as the next state and written under `<output_dir>/edit_chains/<chain_id>/`,
and after the last step all groups are scored in one reward call. The drawn
sample's score becomes that state's score; the source is left unscored.

Each sample's reward metadata names its parent as `reference_images`, plus
`prompt_id` (`<chain_id>:<step>`), `chain_id`, `chain_step`, `chain_length`,
`chain_source` and `chain_parent_sample_id`, so rewards judge each step against
the image it actually edited: doing the requested edit and keeping earlier edits.
The trainer sees one GRPO group per step; no credit flows from a later step to an
earlier one, and a chain can continue from a failed edit. `vrl` knows none of
this: a chain is an `OwnedCollection`, a prompt item with a `collect` method.

## Evaluate the editor

```bash
python -m agentic.scripts.evaluate_chains --path /abs/qwen-image-2.1 --revision PINNED \
  --chains chains.jsonl --requirements requirements.json \
  --reward-config reward.yaml --reward-revision v1 --output outputs/chain-eval
```

`requirements.json` is a list of `{"name", "axis", "threshold", "direction"?, "active_from"?}`
over the judge's score axes. Each chain runs with the frozen `LocalEditor` (one
image per step) and `RewardJudge`, its states are exported to `media.jsonl` for
independent rescoring, and `report.json` says for each requirement which step
first met it and which step lost it (`vrl.rewards.sequences`).

`--reward-config` takes the framework's `RewardConfig` (CPU components in-process,
HTTP components with `expected_model_version`); `--reward-endpoint` with
`--reward-model` uses EditReward directly. A learned score is not a locality proof.

## Reference: Edit-R1

[Edit-R1](https://github.com/PKU-YuanGroup/Edit-R1) (commit 4217f25) is the
closest complete open-source recipe for editor-only RL on Qwen-Image-Edit: rows
of `{"prompt", "image", "requirement"}`, a vLLM reward server returning
`{scores}`, flow-GRPO style sampling with DiffusionNFT. A chain step is that
row with its image supplied by the previous step.
