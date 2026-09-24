# Agentic visual control (`agentic`)

`agentic` trains a Qwen3-VL controller that decides which edit to apply next, or
to stop, while Qwen Image 2.1 stays a frozen editing tool. It sits beside the
`vrl` framework and depends on it; `vrl` never imports it. Editor training,
including declared multi-step tasks, stays in `vrl` (see `sequential_editing.md`).

## Pieces

| Module | Contents |
| --- | --- |
| `agentic/episode.py` | `Task`, `Action`, `Artifact`, `Observation`, `Decision`, `Score`, `PolicyStamp`; the `Controller` / `Editor` / `Judge` protocols; `Episode`, which runs one episode and rebuilds training steps from a trace |
| `agentic/roles.py` | `LocalEditor` (frozen family model through its batch executor) and `RewardJudge` (a `RewardRuntime`) |
| `agentic/controller.py` | `CategoricalController`: single-token action labels, softmax with temperature, replay tensors on disk |
| `agentic/trainer.py` | `ControllerTrainer`: return-to-go credit, leave-one-out baseline, GRPO clipped surrogate, checkpoints |
| `agentic/export.py` | an episode's images as a media manifest for independent rescoring |
| `agentic/scripts/` | collect one episode, train, compare to baselines, probe replay, run a scripted sequence, export media |

## Task and episode

A task JSON names a source image and an explicit action vocabulary with exactly
one stop:

```json
{"task_id": "chair-blue",
 "instruction": "Change only the chair upholstery to blue.",
 "requirement": "Keep the magazine holder and the floor unchanged.",
 "source": "source.png",
 "context_images": {"style reference": "swatch.png"},
 "reward_assets": {"target_image": "target.png"},
 "actions": [{"name": "blue", "kind": "edit", "instruction": "Change only the chair upholstery to blue."},
             {"name": "stop", "kind": "stop"}]}
```

The row decides what the controller sees, as in Edit-R1's
`{"prompt", "image", "requirement"}` rows: `instruction` and `requirement` go
into its prompt text, `context_images` are shown after the original and current
images under their names, and `requirement` also reaches the reward as metadata.
Paths resolve relative to the task file. `reward_assets` go only to the judge.
Unknown fields fail, so an unsupported constraint cannot look enforced.

The prompt itself is one template in `agentic/prompt.py`, rendered from those
fields plus the previous action and the remaining budget; the action list is
lettered and the controller answers with one letter.

One episode: the judge scores the source; the controller sees the original and
current images and picks an action; an edit runs the editor on the current image
and the judge scores the result at cost `tool_cost`; stop or the call budget
ends it. The final score is paid once on the last decision. Controller, editor
and judge activate and park in turn on a shared GPU; a failed park retires the
role and the episode ends with an error trace, which never trains.

`episode.json` records the task, seed, the three policy identities, every
observation and decision with its old log-probability, each edit output, the
per-step rewards and returns.

## Train

```bash
python -m agentic.scripts.train_visual_controller \
  --path /abs/qwen-image-2.1-snapshot --revision PINNED --tasks tasks.jsonl \
  --output outputs/controller-rl --reward-config reward.yaml --reward-revision v1 \
  --resolution 256 --steps 8 --max-tool-calls 2 --tool-cost 0.1 --temperature 4 \
  --episodes-per-task 2 --updates 2 --learning-rate 0.0001 --deterministic
```

Each update collects at least two fresh episodes per task, computes each
decision's return-to-go (later costs plus the final score), subtracts the mean
return of the task's other episodes, and takes one clipped policy-gradient step.
Recomputed log-probabilities must match the sampled ones within
`replay_tolerance`. Loss sums decisions within an episode and averages episodes.

Checkpoints hold the LoRA parameters, Adam state, process RNG, the controller's
temperature / observation mode, its base identity (a hash of the backbone files,
prompt text and LoRA shape) and the run contract. Resume with
`--resume checkpoint-1.pt --updates 3` into a new output directory; the
checkpoint must come from the same base model, and controller settings follow it.

## Compare

```bash
python -m agentic.scripts.visual_control \
  --path ... --revision ... --tasks heldout.jsonl --output outputs/eval \
  --controller-checkpoint outputs/controller-rl/checkpoint-2.pt \
  --reward-config reward.yaml --reward-revision v1 --temperature 4 --repeats 2
```

Each task and seed runs immediate stop, one fixed edit, uniform random actions,
the controller, and best-of-N under the same maximum editor budget. Keep
evaluation sources disjoint from training tasks yourself; the script does not
check it. The judge that selects best-of-N is also the optimized reward, so
rescore `selected_media.jsonl` independently before claiming improvement.

`agentic.scripts.visual_sequence` runs a declared edit order through the same
roles without a learned controller and reports per-requirement preservation; it
is a baseline for a planner, not evidence of one.

## Rewards

`--reward-config reward.yaml` takes the framework's `RewardConfig`: CPU components
run in-process, HTTP components need `expected_model_version`, and a service that
shares the GPU must advertise a parking lease. `--reward-endpoint` with
`--reward-model` uses EditReward directly. The judge always scores against the
original task and source, so a learned score is not a locality proof.

## Evidence

The controller path is implemented end to end with real models: multi-step
orchestration, categorical replay with zero parity error, nonzero gradients,
checkpoint restore. It has not beaten the fixed one-edit baseline. The last
512-episode run moved held-out net return from 0.659 to 0.693 while fixed
one-edit scored 0.805, and already-finished tasks were edited more, not less.
Details and artifact paths: `docs/reports/visual_rl_engine_20260922/ACCEPTANCE.md`.

## Reference: Edit-R1

[Edit-R1](https://github.com/PKU-YuanGroup/Edit-R1) (commit 4217f25) is the
closest complete open-source recipe for editor-only RL on Qwen-Image-Edit, and
the yardstick for how lean this package can be:

- Reward interface: one HTTP POST of `{images, ref_images, prompts, metadatas}`
  returning `{scores}`; the server runs Qwen2.5-VL-32B in vLLM and scores as the
  expected value over the digit logits 0-5 of a fixed rubric prompt. Our
  `RewardJudge` over an HTTP `RewardRuntime` is the same shape.
- Training: a flow-GRPO style sampler (12 images per prompt, per-prompt reward
  statistics, zero-std prompt banning) with the DiffusionNFT loss. No hashes,
  identities or replay audits; on-policy consistency is trusted within the run.
- Data: `{"prompt", "image", "requirement"}` JSONL. A task row here carries the
  same fields (`instruction`, `source`, `requirement`) plus the action list and
  optional context images; in `vrl`, a prompt-manifest row or an edit-chain step
  carries `requirement` as reward metadata the same way.

What Edit-R1 does not have and this package keeps: an explicit stop action, a
per-call cost, and a separate controller policy whose decisions are replayed
against stored processor tensors.
