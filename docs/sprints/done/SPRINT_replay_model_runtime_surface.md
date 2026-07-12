# SPRINT: ReplayModel runtime surface（设计，未实施）

状态：已落地（本设计提出的 ReplayModel runtime surface 在代码中已基本实现，文档落后于实现）。P0 build_replay_bundle 全 family 接入 + online.py:400 优先选择 + test_minimal_replay_runtime_wiring.py 断言；P1 frozen_offload/frozen_module.py/trainer_frozen_targets/apply_trainer_memory_policy 已从源码移除（仅剩 2 处过时测试名字符串）；P2 generation_only_modules/runtime_role/loads_full_generation_modules metadata 早于本文一个月落地（1472306, 2026-05-14）；P3 trainer.py first_step_logprob_parity→training_debug.jsonl + test_scheduler_logprob_parity.py；P4 diffusion_nft_prepare_transformer_input 落地（54623cb）并由 vrl/algorithms/diffusion_nft.py 消费。仅 P5（多卡 compile/FSDP）属显式未来工作。
结论是：**要维护两个 runtime surface，但不要维护两份完整模型。**

```text
Generation model
  拥有完整 pipeline
  负责 prompt / reference image encode
  负责 latent prepare / denoise rollout / decode
  负责导出 trainer replay 所需 tensors

Replay model
  只拥有 trainer 需要的模块
  从 rollout 导出的 replay tensors 恢复采样状态
  只跑 trainable forward
  只暴露 trainable state 给 checkpoint / EMA / weight sync
```

## 0. 当前证据

online trainer 已经优先走 replay bundle：

```python
bundle_builder = definition.build_replay_bundle or definition.build_bundle
```

每个 diffusion online entrypoint 都应该提供 `build_replay_bundle`。以 Wan 2.1 为例，
trainer replay bundle 只加载 transformer + scheduler：

```python
model = replay_cls(
    transformer=load_diffusers_transformer(...),
    scheduler=load_diffusers_scheduler(...),
    device=build.device,
)
```

Wan I2V replay model 明确不拥有 generation-only modules：

```python
class WanI2VReplayModel(...):
    """Replay-only Wan I2V model that owns no text, image, VAE, or pipeline modules."""
```

这说明 replay 不是可选优化，而是 trainer runtime 的主边界。`frozen_offload`
这种“先加载完整 pipeline，再把冻结模块挪 CPU”的策略只适合临时兜底；主路径应该是
generation-only modules 根本不进入 trainer 进程。

## 1. 目标

把 ReplayModel 从“省显存的最小 wrapper”提升成“RL replay 正确性的边界”。

目标能力：

```text
1. replay parity：证明 trainer replay 与 rollout-time 信号一致
2. conditioning integrity：证明 prompt/reference conditioning 没漂
3. step-sliced replay：只搬当前 timestep 需要的数据
4. algorithm input builders：让 GRPO / NFT / DPO 从 replay tensors 构造训练输入
5. trainable-only state：checkpoint / EMA / weight sync 永远只碰 trainable modules
6. replay-only compile / FSDP / precision：只优化 trainer 需要的 transformer 路径
7. absent module metadata：把“不加载”变成可测试契约，而不是口头假设
```

## 2. 该加什么

### 2.1 Replay parity guard

每个 diffusion ReplayModel 应提供或接入统一 parity probe：

```text
输入：RolloutBatch + timestep index
检查：
  rollout old_log_prob vs trainer fresh log_prob
  timestep domain / scheduler domain
  CFG 分支行为
  conditioning tensors shape / dtype / device
输出：可写入 training_debug.jsonl 的结构化记录
```

这不是调试糖，而是 RL 信号的 correctness gate。没有 parity，reward 再好也可能训练错目标。

### 2.2 Conditioning integrity checks

ReplayModel 应验证 rollout 存下来的 conditioning tensors，而不是重新 encode：

```text
SD3.5:
  prompt_embeds
  negative_prompt_embeds
  pooled_prompt_embeds
  negative_pooled_prompt_embeds

Cosmos Predict2:
  prompt_embeds
  reference/video conditioning tensors
  frame / layout metadata

Wan I2V:
  prompt_embeds
  negative_prompt_embeds
  image_embeds
  condition
```

Wan I2V 的重点是 `image_embeds` 与 `condition`：reference image 的 encode 与 VAE
conditioning 必须发生在 rollout 侧，trainer 只消费保存后的 tensor。ReplayModel
应该 fail loud，而不是静默用缺失 conditioning 跑一个“看似能 forward”的错误训练。

### 2.3 Step-sliced replay tensors

当前 replay 已经有按 timestep slice 的基础。下一步应把这变成显式契约：

```text
TrajectoryResolver 只取当前 timestep
ReplayModel 只把当前 step tensor 搬到 trainer device
大块 replay tensors 可留在 CPU / pinned memory / future artifact store
```

这比 freeze 更强：freeze 只处理模块显存，step-sliced replay 能减少训练数据显存。

### 2.4 Algorithm-specific input builders

ReplayModel 可以暴露算法专属但 family-owned 的 input builder，例如：

```python
diffusion_nft_prepare_transformer_input(...)
dpo_prepare_transformer_input(...)
grpo_replay_debug_inputs(...)
```

这些方法不应该放在 generic trainer 里，因为 transformer 的输入结构是 family-specific。
算法只声明它需要什么；family ReplayModel 负责把 replay tensors 转成可 forward 的结构。

Wan I2V 的高价值方向：

```text
diffusion_nft_prepare_transformer_input(
  latents_clean,
  prompt_embeds,
  negative_prompt_embeds,
  image_embeds,
  condition,
  timesteps,
)
```

这样 Wan I2V 能在 NFT / DPO 类算法里继续不加载 text encoder、image encoder、VAE。

### 2.5 Trainable-only state contract

ReplayModel 应成为 trainable state 的唯一入口：

```text
load_trainable_state(...)
trainable_modules
checkpoint export modules
EMA parameters
Ray / future NCCL weight sync
```

这保证 trainer 永远不会把 text encoder、image encoder、VAE、pipeline 混进 checkpoint
或 weight sync。freeze model 很难保证这一点，因为完整对象已经在进程里。

### 2.6 Replay-only compile / FSDP / precision

未来多卡和 compile 应落在 ReplayModel surface：

```text
只 compile transformer replay forward
只 FSDP shard trainable transformer / adapter
只对 replay math 应用 trainer precision policy
generation pipeline 不进入 trainer graph
```

这能避免 full pipeline object 污染 trainer graph，也让 FSDP / checkpoint state dict 更干净。

### 2.7 Absent module metadata

继续保留并强化 replay bundle metadata：

```text
runtime_role = "minimal_replay_model"
loads_full_generation_modules = false
replay_modules = ("transformer", "scheduler")
generation_only_modules = ("text_encoder", "image_encoder", "vae", "pipeline", ...)
absent_modules = ...
```

验收标准不是“没有 OOM”，而是测试能证明 trainer bundle 不拥有 generation-only modules。

## 3. 每个 family 的维护规则

每个新 RL family 必须实现：

```text
Generation model:
  from_build / runtime bundle
  encode_prompt / encode_reference
  prepare_sampling
  forward_step
  decode_latents
  export_batch_context
  export_replay_tensors

Replay model:
  minimal __init__(transformer, scheduler, device)
  pipeline property fail loud
  restore_eval_state
  replay_forward through shared DiffusionModelBase
  trainable_modules / load_trainable_state
  family-specific algorithm input builders when needed
```

这不是两份完整模型。共享逻辑应保留在 base class、backbone runner、scheduler/math helpers、
trajectory resolver 中；两个 surface 只分离“谁拥有 generation-only modules”。

## 4. 该删什么

`frozen_offload` 不应作为 trainer 主路径保留。

应该删除：

```text
model.memory.frozen_offload
vrl/trainers/frozen_module.py
train entrypoint 中 apply_trainer_memory_policy(...) 调用
DiffusionModelBase.trainer_frozen_targets()
围绕 frozen_offload 的测试
```

删除前提：所有 online RL entrypoint 都有 `build_replay_bundle`，并且测试覆盖 trainer
实际选用 replay bundle。

## 5. 该保留什么

```text
build_replay_bundle thin functions
ReplayModel classes
minimal_replay_bundle_metadata(...)
ReplayModel / RuntimeModel protocols
TrajectoryResolver step slicing
model.memory.vae_decode
```

`build_replay_bundle` 这些 thin functions 要保留。它们是 family/runtime boundary，
不是无意义包装。统一形状能提升 grepability、debuggability、checkpoint resume 可读性。

`MEMORY_POLICY_METADATA_KEY` 也应保留，它是 bundle metadata schema key。

`MODEL_MEMORY_SECTIONS` 如果只剩 `vae_decode`，可以继续作为 schema boundary；但不应继续把
`frozen_offload` 当有效 section。

## 6. 非目标

```text
不把 generation model 和 replay model 合并成一个“聪明模型”
不把 family-specific restore/input builder 下沉到 generic trainer
不为了少几行代码扁平化 build_replay_bundle
不在 trainer 里重新 encode prompt / reference image
不把 freeze list 当长期 memory system
```

## 7. 实施顺序

```text
P0 证明现状：
  grep 所有 online entrypoint 是否都有 build_replay_bundle
  加测试断言 trainer bundle 使用 minimal replay model

P1 删除 trainer frozen_offload：
  删除 config section、parser、entrypoint hook、旧测试
  保留 vae_decode generation memory policy

P2 强化 replay metadata：
  每个 replay bundle 写 generation_only_modules / absent_modules
  测试 generation-only modules 不存在于 trainer model

P3 加 parity / conditioning guard：
  先接 Wan I2V 与 Cosmos Predict2
  写入 training_debug.jsonl

P4 加 algorithm input builders：
  先为 Wan I2V 预留 DiffusionNFT input builder
  再按实际算法扩展 DPO / GRPO debug builder

P5 接 replay-only compile / FSDP：
  只作用于 transformer / adapter
  checkpoint / EMA / weight sync 只走 trainable state contract
```

## 8. 验收

```text
pytest -q tests/models/interfaces/test_minimal_replay_runtime_wiring.py
pytest -q tests/rollouts/replay/
pytest -q tests/trainers/online/
pytest -q tests/config/
```

新增验收点：

```text
WanI2VReplayModel 不拥有 pipeline / text_encoder / image_encoder / vae
Wan I2V trainer replay 可验证 image_embeds + condition 完整性
所有 diffusion online entrypoint 使用 build_replay_bundle
删除 frozen_offload 后 config unknown-key lint 仍通过
checkpoint / weight sync 只包含 trainable modules
```

## 9. 参考路径

```text
vrl/scripts/common/online.py
vrl/scripts/diffusion/wan_2_1/train.py
vrl/models/diffusion/wan_2_1/model.py
vrl/models/diffusion/wan_2_1/runtime.py
vrl/models/diffusion/base.py
vrl/models/interfaces/replay.py
vrl/models/interfaces/runtime.py
vrl/models/replay_loading.py
tests/models/interfaces/test_minimal_replay_runtime_wiring.py
```
