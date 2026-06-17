# SPRINT: Streaming Rollout Accumulation

状态：implemented（landed VRL main `4c85f3b`，2026-06-16；本文为完成记录）。这个 sprint 修复 Cosmos Predict2.5 + Kling reward
paper-shaped RL batch 在单机上 OOM 的根因：当前 trainer 把 `rollout_batch_size` 个 prompt
的全部 rollout/replay 数据一次性收齐后才训练，导致 `32 * 8 = 256` 个视频样本和 replay
tensors 堆在 driver host RAM 里。

触发证据：

```text
run: outputs/_scratch_cosmos25_kling_paper_batch_20260614_164932
config: rollout_batch_size=32, n_samples_per_prompt=8, gradient_accumulation_steps=32
failure: ray.exceptions.OutOfMemoryError
memory: 87.33GB / 91.88GB > Ray threshold 0.95
top user: trainer process 76.41GB
stage: collector.collect_unscored -> runtime.generate -> RayGenerationWorker.execute_chunk
metrics: only header; no reward_artifacts; no checkpoint
```

结论：这不是 optimizer step OOM，也不是 replay 数学错误。Replay 机制本身是对的；错误在
batch 生命周期：`gradient_accumulation_steps` 只在训练 loop 里切 backward，但当前
rollout collection 仍然先收完整个 optimizer batch。

## 0. Core Decision

**不新增 `actor.optimizer_batch_conditions`。**

现有字段直接承担目标语义：

```yaml
rollout:
  rollout_batch_size: 32        # optimizer target batch: input conditions per update
  n_samples_per_prompt: 8       # per-condition GRPO/NFT group
  sample_batch_size: 1          # denoise engine chunk size

actor:
  gradient_accumulation_steps: 32  # split target batch into rollout/train microsteps
```

推导：

```text
rollout_microbatch_conditions = rollout_batch_size / gradient_accumulation_steps
```

paper-shaped Cosmos run：

```text
32 / 32 = 1 condition per rollout microbatch
1 condition * 8 samples = complete advantage group
32 microbatches -> 1 optimizer.step()
```

这个设计保留用户的 mental model：

```text
rollout_batch_size = final optimizer batch size
gradient_accumulation_steps = PyTorch-style accumulation split count
sample_batch_size = generation engine chunk size
```

## 1. Current Failure Mode

当前入口按 epoch 一次抽出 `rollout_batch_size` 个 examples：

```python
idx = sample_prompt_indices(
    rng,
    num_examples=len(examples),
    rollout_batch_size=trainer_config.rollout_batch_size,
    ...
)
example_batch = [examples[i] for i in idx]
metrics = await trainer.step(example_batch)
```

`trainer.step()` 当前先完整 collect：

```python
batch = await self.collect_training_batch(prompts)
return await self.train_on_rollout_batch(batch)
```

而 `collect_training_batch()` 会一次性持有整个 `iteration.batches`：

```python
iteration = await self.rollout_schedule.next_iteration(
    list(self.prompts),
    group_size=cfg.n_samples_per_prompt,
)
all_batches = iteration.batches
```

只有进入 `train_on_rollout_batch()` 之后，`gradient_accumulation_steps` 才生效：

```python
grad_accum_batches = int(cfg.gradient_accumulation_steps)
...
for batch_start in range(0, len(filtered_batches), grad_accum_batches):
    ...
    loss.backward()
optimizer.step()
```

所以 `gradient_accumulation_steps=32` 没有保护 rollout memory。它保护的是已经收齐后的
backward，而 OOM 发生在更早的 rollout generation / driver accumulation 阶段。

## 2. Target Behavior

目标行为：

```text
for each optimizer update:
  sample rollout_batch_size conditions once
  split them into gradient_accumulation_steps prompt microbatches

  for each prompt microbatch:
    collect rollout for microbatch
    score reward for microbatch
    compute advantages within each complete n_samples_per_prompt group
    run backward without optimizer.step()
    release rollout/replay tensors

  optimizer.step()
  EMA step
  DiffusionNFT previous-policy sync
  rollout weight sync
  write one metric row for the full target batch
```

关键正确性：

- Advantage normalization 仍然正确，因为每个 microbatch 必须包含完整 prompt group：
  `n_samples_per_prompt=8`。
- 32 个 prompt 不需要同时在内存里才可以算 advantage；GRPO/NFT 的 group normalization 是
  per-prompt group，不是跨 prompt batch。
- 所有 microbatches 必须使用同一个 policy version；optimizer step 和 rollout weight sync 只能在
  full target batch 结束后发生。
- `after_optimizer_step()`、EMA、checkpoint global step 都必须按 optimizer update 计数，不按
  microbatch 计数。

## 3. Config Semantics

### 3.1 新语义

```text
rollout.rollout_batch_size:
  target input conditions per optimizer update

actor.gradient_accumulation_steps:
  number of rollout/train microsteps inside that optimizer update

rollout.sample_batch_size:
  generation backend chunk size only; does not define RL batch size
```

### 3.2 Validation

Add fail-fast validation:

```text
gradient_accumulation_steps >= 0
if gradient_accumulation_steps <= 0:
  rollout_microbatch_conditions = rollout_batch_size
else:
  rollout_batch_size % gradient_accumulation_steps == 0
  rollout_microbatch_conditions = rollout_batch_size // gradient_accumulation_steps
  rollout_microbatch_conditions >= 1
```

`gradient_accumulation_steps=0` 保留“整批一次 collect/train/step”的兼容路径。

### 3.3 Existing Config Migration

这个 sprint 会改变 positive `gradient_accumulation_steps` 的实际边界。旧语义是“在已经 collect
完的 batches 内，每多少个 rollout batches 做一次 optimizer step”；新语义是“一个 optimizer
target batch 被切成多少个 rollout/train microsteps”。

需要逐个检查现有 online configs：

- `rollout_batch_size=32, gradient_accumulation_steps=32`：
  paper target 32，microbatch 1，one optimizer step。
- `rollout_batch_size=8, gradient_accumulation_steps=4`：
  target 8，microbatch 2，one optimizer step。
- `rollout_batch_size=4, gradient_accumulation_steps=4`：
  target 4，microbatch 1，one optimizer step。
- `gradient_accumulation_steps=1`：
  target batch 不切分，one optimizer step。

如果某个旧 config 真的依赖“一个 epoch 内多个 optimizer steps”，它应该通过更小的
`rollout_batch_size` 表达，而不是把 `gradient_accumulation_steps=1` 当作“每组 step 一次”。

## 4. Implementation Plan

### T1. Extract Optimizer-Step Boundary

Refactor `OnlineTrainer.train_on_rollout_batch()` so backward and optimizer step are separable.

Target shape:

```text
collect_training_batch(prompts) -> TrainingBatch
backward_on_training_batch(batch, loss_scale=...) -> partial metrics
finish_optimizer_update(...) -> grad norm, optimizer.step, EMA, after_optimizer_step
```

Rules:

- No optimizer step inside microbatch backward.
- `optimizer.zero_grad()` only after full target update step.
- GradScaler skipped-step handling remains centralized in `finish_optimizer_update()`.
- `DiffusionNFT.after_optimizer_step()` runs only after real optimizer step.
- EMA runs only after real optimizer step.

Do not create a new thin file for this. The boundary belongs in `vrl/trainers/online/trainer.py`
because it is trainer step lifecycle, not a new protocol.

### T2. Add Streaming Accumulation Loop In Online Recipe

Change the online recipe epoch loop to:

```text
idx = sample_prompt_indices(... rollout_batch_size=target_conditions ...)
example_batch = [examples[i] for i in idx]
microbatches = split example_batch by rollout_microbatch_conditions

trainer.begin_optimizer_update()
for microbatch in microbatches:
  components.reward_fn.reset_components? no, see T3
  batch = await trainer.collect_training_batch(microbatch)
  await trainer.backward_on_training_batch(batch, ...)
  release batch references
metrics = await trainer.finish_optimizer_update(...)
write one metric row
```

Implementation detail: keep `reward_fn.reset_components()` once per optimizer update, before the
first microbatch. Reward component rows should aggregate all microbatches in that update.

### T3. Metrics Aggregation

One output row must still represent the full optimizer target batch.

Aggregate:

- reward mean/std over all samples across microbatches.
- component reward means over all samples.
- loss / policy_loss / kl / grad diagnostics as sample/timestep-weighted means where possible.
- `group_size` should remain `n_samples_per_prompt`.
- `trained_prompt_num` should sum unique prompt groups across microbatches.
- phase timings should sum collection/reward/replay/backward times across microbatches.

Avoid reporting one row per microbatch; that would make the run look like 32 optimizer updates when it is
one optimizer update.

### T4. Memory Lifecycle

After each microbatch backward:

- Drop local references to `TrainingBatch`, `RolloutBatch`, videos, trajectory, replay extras.
- If tensors were moved to GPU for replay, release them before the next microbatch.
- Keep only scalar metrics and accumulated gradients.

Acceptance memory target for Cosmos Predict2.5 paper shape:

```text
rollout_batch_size=32
n_samples_per_prompt=8
gradient_accumulation_steps=32
sample_batch_size=1

host RAM should not grow linearly toward 32 prompt groups.
peak host RAM should stay close to one prompt group plus trainer/reward overhead.
```

### T5. Validation And Tests

Add focused tests:

1. `gradient_accumulation_steps` validation:
   - 32 / 32 ok.
   - 8 / 4 ok.
   - 6 / 4 fails with clear message.
   - 0 means unsplit full batch.

2. Trainer fake collector:
   - `rollout_batch_size=4`, `gradient_accumulation_steps=4`, `n_samples_per_prompt=2`.
   - collector sees four calls of one prompt each.
   - optimizer steps once.
   - global_step increments once.
   - `after_optimizer_step()` runs once.
   - EMA step runs once.

3. Accumulated gradient equivalence:
   - Compare old full-batch path vs streaming microbatch path on deterministic fake tensors.
   - Final parameter delta matches within tolerance.

4. Metrics:
   - one metric row per optimizer update.
   - `trained_prompt_num == rollout_batch_size`.
   - reward mean/std match all collected samples, not last microbatch only.

5. Cosmos config pin:
   - `online_nft_kling_video_reward` keeps `n_samples_per_prompt=8`,
     `rollout_batch_size=32`, `gradient_accumulation_steps=32`,
     `sample_batch_size=1`.

### T6. Real Smoke

Run a bounded Cosmos smoke:

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward \
  trainer.output_dir=outputs/_smoke_cosmos25_streaming_accum \
  trainer.total_epochs=1
```

Acceptance:

- No Ray host OOM.
- `metrics.csv` has exactly one data row.
- `trained_prompt_num=32`.
- `group_size=8`.
- reward artifacts/debug outputs exist.
- checkpoint-final exists.

Then run a shorter debug continuation if needed:

```text
trainer.total_epochs=3
```

Acceptance is stability and sane reward logging, not reward improvement yet.

## 5. Non-Goals

- Do not add `actor.optimizer_batch_conditions`.
- Do not reduce paper batch to hide the OOM.
- Do not disable Ray OOM monitor as a workaround.
- Do not make `sample_batch_size` mean RL batch size.
- Do not change `n_samples_per_prompt`; paper group size remains 8.
- Do not implement SFT/diffusion regularization in this sprint.
- Do not change ReplayModel ownership; this sprint changes rollout/trainer lifecycle, not model loading.
- Do not move replay tensors to disk unless streaming still cannot bound host RAM.

## 6. What Changes / What Stays

Change:

- `rollout_batch_size` becomes the optimizer target condition count in online trainer semantics.
- Positive `gradient_accumulation_steps` becomes the number of rollout/train microsteps for that target.
- Online trainer gains an explicit backward-only microstep and finish-update boundary.
- Metrics aggregation spans microbatches.

Stay unchanged:

- `n_samples_per_prompt` remains the advantage group size.
- `sample_batch_size` remains generation backend chunk size.
- ReplayModel remains the minimal train-time model surface.
- Reward function stays a scorer over generated artifacts; only call granularity changes.
- Config schema should not gain a duplicated target-batch field.

Architecture hygiene:

- No module-level ALL_CAPS validation list should duplicate trainer dataclass fields. Derive config field
  validation from existing schema/builders.
- Keep the existing `collect_training_batch()` / `train_on_rollout_batch()` split; it is a real lifecycle
  boundary and should not be flattened.
- Add only trainer-local helper methods where they express lifecycle phases (`backward_on_training_batch`,
  `finish_optimizer_update`). Do not create thin standalone utility files for a few lines of trainer state
  choreography.

## 7. References

- `vrl/scripts/common/online.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/orchestration/strict_on_policy.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/rollouts/batch/core.py`
- `vrl/rollouts/batch/ops.py`
- `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
- `outputs/_scratch_cosmos25_kling_paper_batch_20260614_164932/train.log`
