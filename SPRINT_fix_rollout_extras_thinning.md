# SPRINT：RolloutBatch extras thinning

状态：主路径已完成。`log_probs` / `token_mask` / `r1_segments` / model replay payload 不再从 `RolloutBatch.extras` 读取；replay 通过 `TrajectoryBatch` replay refs 和 resolver 解析。`RolloutBatch.extras` 字段仍保留为通用 metadata/diagnostic 承载层，不再是训练事实源。

## 目标

把 `RolloutBatch.extras` 从训练事实源降级为兼容 payload。

历史上 `extras` 混了三类东西：

1. 兼容投影：`log_probs`、`token_mask`、`r1_segments`、`primary_segment`。
2. 真实 replay 输入：`timesteps`、prompt embeds、attention masks、`saved_noise`、`latents_clean`。
3. 调试和指标：`kl`、`reward_before_kl`、decoded intermediates。

这个 sprint 分两次执行；当前主路径收口已经完成。

第一轮只建立 trajectory resolver / views / validation，让后续 evaluator 和 algorithm strict cleanup 有稳定读取层。第一轮不要急着删除 extras，也不要把 evaluator/algorithm 迁移都吞进这个 sprint。

第二轮在 evaluator 和 algorithm 已经 strict 之后回来完成 extras 降级：trainer/model replay/packer 停止把 `log_probs`、`token_mask`、`r1_segments` 当训练事实源。当前代码已经完成这一步。

## 不做的事

- 不立刻删除 `RolloutBatch.extras` 字段。
- 不删除 still-needed replay payload，直到模型 replay 已经能从 trajectory refs 解析。
- 不把 replay 输入塞进新的 loose dict。
- 不改变 reward 数值或 algorithm loss 公式。
- 第一轮不迁移 evaluator 输出契约；evaluator strict cleanup 已完成并合并到总 unification sprint。
- 第一轮不迁移 algorithm loss 入口；那属于 `SPRINT_fix_algorithm_strict_input.md`。

## 执行顺序

推荐顺序：

```text
1. 本 sprint Phase 1：resolver / view / validation only
2. Evaluator strict cleanup（已完成，原 sprint 文档已合并并删除）
3. SPRINT_fix_algorithm_strict_input.md
4. 回到本 sprint Phase 2-5：extras 降级和 replay view
```

这样可以避免在 trainer/evaluator/algorithm 三条链路里重复改同一组 logprob/mask 读取逻辑。

## 当前应降级的 extras 字段

这些字段不应再作为事实源：

```text
log_probs
token_mask
r1_segments
primary_segment
initial_image
final_image
selfcheck
reward_before_kl
kl
```

目标事实源：

```text
training_view.loss_units[*].old_log_prob_ref
training_view.loss_units[*].mask_ref
trajectory.segments[name]
trajectory.reward_views[name]
TrajectoryBatch.metrics / OutputBatch.metrics
```

## 当前仍要迁移的 replay 字段

这些字段现在仍有消费者，不能直接删：

```text
timesteps
prompt_embeds
pooled_prompt_embeds
negative_prompt_embeds
negative_pooled_prompt_embeds
init_latents
latents_clean
diffusion_nft_noise
prompt_attention_mask
uncond_input_ids
uncond_attention_mask
saved_noise
```

目标事实源：

```text
TrainingView.loss_units[*].replay_input_refs
TrajectorySegment.tensors[name]
ReplayInput.tensor_refs
```

## 实施阶段

### Phase 1：新增 trajectory resolver

新增或编辑：

```text
vrl/engine/trajectory/resolver.py
vrl/engine/trajectory/views.py
vrl/engine/trajectory/validation.py
```

要求：

- 按 `TrainingView.loss_units` 解析 `action_ref`、`old_log_prob_ref`、`mask_ref`。
- 按 `replay_input_refs` 解析 replay tensors。
- 支持 `axis_index` / `timestep_idx` 切片。
- resolver 返回明确 typed object，不返回 loose dict。
- shape mismatch 和 missing ref 必须 fail fast。

### Phase 2：trainer 停止读 `extras["log_probs"]`

执行时机：在 evaluator strict 和 algorithm strict 之后。

编辑：

```text
vrl/trainers/online.py
```

要求：

- debug first-step parity 和主训练循环都从 resolver 读取 old logprob。
- 没有 trajectory/training_view 时才允许 legacy extras fallback，并给明确 deprecation path。
- 删除或改写 `RolloutBatch.extras["log_probs"] is required` 这类错误。

### Phase 3：evaluator mask / old logprob 迁到 trajectory refs

执行时机：跟随已完成的 evaluator strict cleanup 收口。这里不重新定义 evaluator 输出契约，只删除对 legacy extras 的残留读取。

编辑：

```text
vrl/rollouts/evaluators/trajectory.py
vrl/rollouts/evaluators/ar/token_logprob.py
vrl/rollouts/evaluators/ar/continuous_token_logprob.py
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
```

要求：

- token mask 从 resolver 或 `SegmentSignal.mask` 来，不再依赖 `extras["token_mask"]`。
- R1 evaluator 不再依赖 `extras["r1_segments"]`。
- `primary_segment` 从 `TrainingView.primary_segment` 或 `TrajectorySignalBatch.primary_segment` 来。

### Phase 4：model replay compatibility view

新增或编辑：

```text
vrl/engine/trajectory/replay.py
vrl/models/families/sd3_5/policy.py
vrl/models/families/wan_2_1/policy.py
vrl/models/families/cosmos/predict2/policy.py
vrl/models/families/cosmos/predict2_5/policy.py
vrl/models/families/janus_pro/policy.py
vrl/models/families/nextstep_1/policy.py
```

要求：

- 从 trajectory resolver 生成 batch-like replay view，先喂给现有 `replay_forward(...)`。
- diffusion replay 读取 `timesteps`、prompt embeds、Cosmos `init_latents`。
- AR replay 读取 prompt ids/mask、uncond ids/mask、actions。
- NextStep replay 读取 `saved_noise`。
- DiffusionNFT replay 读取 `latents_clean`、prompt embeds、timesteps、可选 noise。

### Phase 5：packer 停止默认膨胀 extras

编辑：

```text
vrl/rollouts/packers/trajectory.py
vrl/rollouts/batch.py
```

要求：

- strict trajectory-backed batch 默认不复制大体积 replay dict 到 extras。
- legacy projection 必须显式开启或集中在 compatibility helper。
- `stack_batches()`、`_select_batch()`、`_move_training_batch_to_device()` 继续支持 extras，但训练语义不依赖 extras。

## 测试计划

编辑：

```text
tests/trainers/test_online.py
tests/rollouts/evaluators/test_trajectory_signals.py
tests/rollouts/test_multisegment_token_logprob.py
tests/rollouts/test_trajectory_packer.py
tests/models/test_janus_replay.py
tests/models/test_diffusion_policy_module.py
tests/algorithms/test_diffusion_nft.py
tests/rollouts/test_batch.py
```

新增断言：

- trajectory-backed training 在没有 `extras["log_probs"]`、`extras["token_mask"]`、`extras["r1_segments"]` 时仍通过。
- action、old logprob、mask 都从 `TrainingView` refs 解析。
- replay compatibility view 能替代 raw extras。
- packer strict mode 不再复制整份 R1 segments 到 extras。

## 完成标准

Phase 1 完成标准：

- resolver 能按 `TrainingView.loss_units` 解析 action、old logprob、mask。
- resolver 能按 replay refs 解析 replay tensors。
- resolver 有 missing ref、axis mismatch、shape mismatch 的 fail-fast 测试。
- trainer/evaluator/algorithm 行为不变，SD3.5 OCR gate 不变。

完整 sprint 完成标准：

- trainer/evaluator/algorithm 主路径不再把 `extras["log_probs"]`、`extras["token_mask"]`、`extras["r1_segments"]` 当事实源。
- replay 输入由 trajectory refs 解析。
- `RolloutBatch.extras` 只保留 compatibility、diagnostic、temporary legacy payload。
- 通过：

```bash
pytest tests/trainers/test_online.py \
  tests/rollouts/evaluators/test_trajectory_signals.py \
  tests/rollouts/test_multisegment_token_logprob.py \
  tests/rollouts/test_trajectory_packer.py \
  tests/models/test_janus_replay.py \
  tests/models/test_diffusion_policy_module.py \
  tests/algorithms/test_diffusion_nft.py \
  tests/rollouts/test_batch.py
```

## 参考路径

- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/batch.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/packers/trajectory.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/trajectory/views.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/trajectory/builders.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/trajectory.py`
