# Sequential editing with edit chains

An edit chain trains the editor on a declared multi-step task with the existing
one-shot GRPO trainer. There is no learned controller, no stop action and no
terminal return: every step is an ordinary prompt group whose conditioning image
is one sample of the previous step. Learned control over the same editor lives
in the separate `agentic` package.

## Manifest

`data.loader: edit_chain_manifest` reads one chain per JSONL line:

```json
{"chain_id": "page-12", "source": "pages/12.png",
 "steps": ["Change the coat to blue. Keep everything else unchanged.",
           {"prompt": "Replace the second bubble's text with HELLO.", "target_text": "HELLO"}],
 "metadata": {"series": "demo"}}
```

- `source` resolves relative to the manifest and must exist.
- A step is an instruction string or a prompt-manifest row without
  `reference_image(s)` or `reference_video`; unknown keys become that step's
  reward metadata, as in prompt manifests. Reward-only targets such as
  `target_text` or `target_image` stay per step.
- `metadata` is shared by every step; a step's own keys win.
- `chain_id` defaults to `<manifest stem>:<line>`; ids must be unique.
- Unknown row keys are rejected.

```yaml
data:
  loader: edit_chain_manifest
  manifest: manifests/edits/chains.jsonl
  sampler:
    type: random_without_replacement
```

The sampler draws chains, so `rollout.prompts_per_batch` counts chains and each
chain contributes one prompt group per step.

## Collection

For each sampled chain, step 0 is generated from `source`; then one sample of the
group is drawn uniformly from the driver RNG, written to
`<trainer.output_dir>/edit_chains/<chain_id>/`, and becomes the reference image
of step 1, and so on. The parent is never chosen by the reward, so later steps
train on the policy's own state distribution. All groups of the call are scored
in one reward call after generation; overlap modes do not apply.

Each sample's reward metadata carries `reference_images` (its parent),
`prompt_id` (`<chain_id>:<step>`), `chain_id`, `chain_step`, `chain_length`,
`chain_source` and `chain_parent_sample_id`. Rewards that compare an output to
`reference_images`, such as EditReward or the CPU verifiers, therefore judge each
step against the image it actually edited: doing the requested edit and keeping
the parent's content, including earlier steps' edits. Rewards that need the
original source or a cumulative target read `chain_source` or per-step targets.

Advantages, zero-advantage admission and the ledger see one group per step.
Group ids are distinct within a collection call and are renumbered downstream.
A chain's steps stay within one collection call, so `actor.prompts_per_collection`
splits between chains, not inside one.

## What this does not do

- No credit flows from a later step to an earlier one; an early step is paid
  only through its own group's reward.
- Branching from a random sample means a chain can continue from a failed edit.
  That is on-policy data, not a curated trajectory.
- The written parent images are training artifacts, not a lineage proof for
  offline audit; `agentic` keeps the content-bound episode traces.
