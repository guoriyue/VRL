# SPRINT：Cosmos-Predict2.5 DiffusionNFT 剩余验收

## 结论

这个 sprint 还不能删除。

当前代码链路已经完成大部分实现，并且 stub video reward 的真实 Cosmos-Predict2.5 DiffusionNFT optimizer update 已经跑通：

```text
outputs/cosmos_predict2_5_diffusionnft_real_opt_check_20260511_233700/optimization_check.json
```

关键结果：

```text
global_step = 1
grad_norm = 0.01920050010085106
reward_std = 0.011377330869436264
adv_zero_rate = 0.0
trainable_sha256_changed = true
```

但是这次 run 仍然是 stub reward，不是 real `dance_grpo` 或 `cosmos_reason1`：

```text
outputs/cosmos_predict2_5_diffusionnft_real_opt_check_20260511_233700/resolved_config.yaml
reward.kwargs.video_reward.backend = stub
```

所以还没有满足原 sprint 的最终验收项：

```text
real dance_grpo or cosmos_reason1 video reward run finishes at least 1 optimizer step
```

## 已完成，不再重复做

- Cosmos-Predict2.5 独立 family 已落地：

```text
vrl/models/families/cosmos/predict2_5/
```

- Cosmos Predict2 2B debug path 已从旧根目录迁到版本化子目录：

```text
vrl/models/families/cosmos/predict2/
```

- family runtime 已统一到 `runtime.py`，不再维护单独 `builder.py` / `executor.py`：

```text
vrl/models/families/cosmos/predict2/runtime.py
vrl/models/families/cosmos/predict2_5/runtime.py
```

- DiffusionNFT algorithm contract 已落地：

```text
vrl/algorithms/diffusion_nft.py
configs/base/algorithm/diffusion_nft.yaml
tests/algorithms/test_diffusion_nft.py
```

- Video reward interface 已落地：

```text
vrl/rewards/video_reward.py
vrl/rewards/remote_video.py
configs/base/reward/video_reward.yaml
tests/rewards/test_video_reward.py
```

- Cosmos-Predict2.5 DiffusionNFT 配置已落地：

```text
configs/model/cosmos/predict2_5_2b.yaml
configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml
configs/sampling/cosmos_predict2_5_512p_93f.yaml
```

- stub video reward optimizer update 已验证，LoRA trainable weights 确实变化。

## 剩余必须完成

### 1. 跑 real video reward

必须使用真实 video reward backend，不接受 stub。

优先级：

```text
1. cosmos_reason1
2. dance_grpo
```

可接受方式：

- remote reward service：对齐 cosmos-rl official-style deployment。
- explicit local wrapper：只能在明确标注为 local backend 时使用，不能伪装成 remote parity。

不可接受：

- `backend: stub`
- frame-level aesthetic reward
- Cosmos-Predict2 2B aesthetic-only GRPO
- supervised V2W / SFT / reconstruction loss

### 2. 验证 real reward optimizer update

真实 reward run 至少要完成 1 个 optimizer step，并记录：

```text
global_step > 0
reward_std > 0 or nonzero advantage batch
grad_norm > 0
trainable_sha256_changed = true
```

需要保留类似下面的机器可读检查文件：

```text
outputs/<real_reward_run>/optimization_check.json
```

### 3. 保存真实 reward debug artifact

remote reward 必须保存 raw enqueue / fetch response：

```text
outputs/<real_reward_run>/reward_debug/remote_video_reward.jsonl
```

local reward 必须保存等价 raw response / score breakdown，至少包含：

```text
reward_name
score_key
scores
prompt
video_infos
```

### 4. 保存 generated video artifact

真实 reward run 需要保存生成视频或帧 artifact，方便人工复查 reward 是否真的看到了模型输出：

```text
outputs/<real_reward_run>/artifacts/
```

至少要能对应：

```text
prompt
seed
sample_id
reward_score
```

### 5. 再决定是否删除 sprint

只有上面全部满足后，才能删除这个 sprint 文件。

删除前还要确认：

- README 没有把 Cosmos-Predict2 2B aesthetic-only GRPO 写成推荐路线。
- README 没有把 Cosmos-Predict2.5 DiffusionNFT 写成 validated route，除非真实 reward run 已通过。
- 不新增 Cosmos-Predict2.5 gap doc，直到真实训练验证完成。

## 下一步命令目标

目标不是再跑 stub，而是跑真实 reward：

```text
experiment = cosmos_predict2_5_2b_diffusionnft
reward.backend = remote or explicit local
reward.reward_name = cosmos_reason1 or dance_grpo
reward.score_key = overall_reward
rollout.n >= 2
trainer.total_epochs = 1
```

完成后用最新 run 的 `optimization_check.json` 和 reward debug artifact 决定是否删除本文件。
