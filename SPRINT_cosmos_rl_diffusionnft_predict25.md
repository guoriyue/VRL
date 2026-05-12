# SPRINT：Cosmos-Predict2.5 + DiffusionNFT + Remote Reward

## 0. 结论

当前有效方向收敛为：

```text
model = nvidia/Cosmos-Predict2.5-2B
algorithm = DiffusionNFT-style GRPO
reward = video-level reward interface
primary reward = dance_grpo 或 cosmos_reason1
checkpoint format = diffusers / LoRA-compatible
```

本 sprint 不再继续强化旧路线：

```text
model = nvidia/Cosmos-Predict2-2B-Video2World
algorithm = 本 repo 当前 GRPO + FlowMatchingEvaluator
reward = aesthetic-only
```

旧路线只能保留为 mechanism/debug reference，不能再作为 “有效 Cosmos RL recipe”。后续实现和 README 必须把它降级为 deprecated / debug-only。

## 1. 保留与移除

### 保留：Cosmos-RL DiffusionNFT 路线

这是本 sprint 的主目标：

```text
/home/mingfeiguo/Desktop/cosmos-rl/configs/cosmos-predict2-5/cosmos-predict2-5-2b-720-nft.toml
```

关键配置：

```toml
[policy]
model_name_or_path = "nvidia/Cosmos-Predict2.5-2B"
model_revision = "diffusers/base/post-trained"
is_diffusers = true

[train.train_policy]
type = "grpo"
trainer_type = "diffusion_nft"
use_remote_reward = true

[[train.train_policy.remote_reward.reward_fns]]
name = "dance_grpo"
weight = 1.0
score_key = "overall_reward"
```

### 移除：deprecated FlowGRPO / DDRL 主线

不把下面这条路线作为当前实现目标：

```text
/home/mingfeiguo/Desktop/cosmos-rl/configs/cosmos-predict2-5/cosmos-predict2-5-2b-480-grpo.toml
/home/mingfeiguo/Desktop/cosmos-rl/configs/cosmos-predict2-5/cosmos-predict2-5-2b-720-reason-embedding-ddrl.toml
```

原因：

- cosmos-rl 文档把这段列在 `RL (deprecated)`。
- DDRL/FlowGRPO 可以作为算法参考，但不应该成为本 repo 当前 Cosmos RL 主线。
- 当前 sprint 只做 DiffusionNFT-style RL + video-level reward，不做 supervised training。

### 移除：supervised V2W SFT / LoRA post-training

不在本 sprint 做 supervised training，不实现 V2W SFT，不实现 supervised LoRA post-training。

移除原因：

- 用户目标只做 RL。
- supervised V2W 数据格式和训练目标会分散当前 DiffusionNFT RL 实现。
- 本 sprint 的训练成功标准必须来自 reward-driven optimizer update，而不是 reconstruction / diffusion supervised loss。

本 sprint 不再引用下面这种 supervised 数据形态作为实现目标：

```text
datasets/example/
  metas/*.txt
  videos/*.mp4
```

### 移除：Cosmos-Predict2 2B + aesthetic-only GRPO 有效训练路线

需要从活跃 sprint/README 里移除 “有效 RL” 叙述。保留代码可以，但必须标明：

```text
status = debug-only / mechanism validation
known issue = aesthetic-only reward produces flat rewards and zero advantage
not accepted = final Cosmos RL recipe
```

验收时不能再用 `cosmos_predict2_2b_grpo` 的 aesthetic-only run 证明 Cosmos RL 成功。

## 2. DiffusionNFT Reward 是什么

DiffusionNFT 本身不是一个 reward。它是训练器/算法路径，reward 可以来自本地 reward、stub reward，或 cosmos-rl 官方风格的 remote reward service。

这里需要的是 video-level reward，不是必须 remote。remote reward 是 cosmos-rl 的工程部署方式：`dance_grpo` 和 `cosmos_reason1` 这类 reward model 很重、依赖环境复杂、吞吐慢，单独服务化更容易扩展和隔离依赖。本 repo 第一版应做成可插拔 reward interface，支持：

```text
local/stub backend = 本机小规模验证训练链路
remote backend = 对齐 cosmos-rl official-style deployment
```

Cosmos-RL reward service 支持：

```text
video:
  cosmos_reason1
  dance_grpo

image:
  hpsv2
  pickscore
  hpsv3
  image_reward
  ocr
  gen_eval
  unified_reward
```

### `dance_grpo`

实现：

```text
/home/mingfeiguo/Desktop/cosmos-rl/reward_service/cosmos_rl_reward/model/dance_grpo.py
```

它基于 DanceGRPO / VideoAlign，输出 video-level scores：

```text
VQ = visual quality
MQ = motion quality
TA = text alignment
overall_reward = VQ + MQ + TA
```

Cosmos-Predict2.5 NFT config 当前取：

```text
score_key = overall_reward
```

这比 `aesthetic` 更适合 video RL，因为它同时看视频质量、运动质量、文本对齐。

### `cosmos_reason1`

实现入口：

```text
/home/mingfeiguo/Desktop/cosmos-rl/reward_service/cosmos_rl_reward/configs/rewards.toml
```

默认模型：

```text
nvidia/Cosmos-Reason1-7B-Reward
```

这是 Cosmos 系列更接近官方的 video reward model。优先级应高于手写 frame-based reward；如果本机资源不足，再用 `dance_grpo`、local stub reward、或 mock service 做 pipeline validation。

## 3. 参考来源

本地参考：

```text
/home/mingfeiguo/Desktop/cosmos-rl
/home/mingfeiguo/Desktop/cosmos-rl/docs/wfm/overview.rst
/home/mingfeiguo/Desktop/cosmos-rl/reward_service/README.md
/home/mingfeiguo/Desktop/cosmos-rl/reward_service/cosmos_rl_reward/configs/rewards.toml
/home/mingfeiguo/Desktop/cosmos-rl/configs/cosmos-predict2-5/cosmos-predict2-5-2b-720-nft.toml
```

本 repo paper：

```text
docs/papers/cosmos_predict2_5_world_simulation_2511.00062.pdf
```

外部参考：

```text
https://nvidia-cosmos.github.io/cosmos-rl/wfm/overview.html
https://github.com/nvidia-cosmos/cosmos-rl
https://github.com/nvidia-cosmos/cosmos-predict2.5
https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
https://huggingface.co/nvidia/Cosmos-Reason1-7B-Reward
```

## 4. 模型架构视角：Add / Edit / Delete

这一节只从 model architecture / training architecture 角度定义变更边界，避免把旧的 Predict2 2B GRPO recipe 混进新路线。

### Add

新增 Cosmos-Predict2.5 独立配置：

```text
configs/model/cosmos/predict2_5_2b.yaml
configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml
```

新增 DiffusionNFT algorithm contract：

```text
vrl/algorithms/diffusion_nft.py
configs/base/algorithm/diffusion_nft.yaml
tests/algorithms/test_diffusion_nft.py
```

新增 video reward interface：

```text
vrl/rewards/video_reward.py
vrl/rewards/remote_video.py
configs/base/reward/video_reward.yaml
tests/rewards/test_video_reward.py
```

新增 `cosmos` family 的版本化子目录，避免 Predict2 2B 和 Predict2.5 adapter 混在同一层：

```text
vrl/models/families/cosmos/predict2/
vrl/models/families/cosmos/predict2_5/
```

目标结构：

```text
vrl/models/families/cosmos/
  __init__.py
  common.py
  predict2/
    __init__.py
    policy.py
    builder.py
    executor.py
  predict2_5/
    __init__.py
    policy.py
    builder.py
    executor.py
    diffusion_nft_adapter.py
```

`predict2/` 承接当前 Cosmos-Predict2 2B debug path；`predict2_5/` 是新的 DiffusionNFT RL path。两个版本的 model contract 必须独立可测试，不能靠一个 adapter 里堆版本分支。

### Edit

模型 registry / builder：

```text
vrl/models/registry.py
vrl/models/families/cosmos/__init__.py
vrl/models/families/cosmos/common.py
vrl/rollouts/families/specs.py
```

需要支持 `cosmos_predict2_5` 或等价显式 family key，不能让 `cosmos` 同时隐式代表 Predict2 2B 和 Predict2.5。

Cosmos Predict2 2B 迁移：

```text
vrl/models/families/cosmos/policy.py
vrl/models/families/cosmos/builder.py
vrl/models/families/cosmos/executor.py
```

迁移到：

```text
vrl/models/families/cosmos/predict2/policy.py
vrl/models/families/cosmos/predict2/builder.py
vrl/models/families/cosmos/predict2/executor.py
```

迁移必须保持现有 2B debug tests 通过；迁移后旧路径可以保留薄 shim 一段时间，但不能作为新实现入口。

Cosmos Predict2.5 新增：

```text
vrl/models/families/cosmos/predict2_5/policy.py
vrl/models/families/cosmos/predict2_5/builder.py
vrl/models/families/cosmos/predict2_5/executor.py
vrl/models/families/cosmos/predict2_5/diffusion_nft_adapter.py
```

`diffusion_nft_adapter.py` 只放 DiffusionNFT 训练所需的 bridge：prepared transformer input、logprob/trajectory projection、video reward payload metadata。不要把 reward client 或 trainer loop 写进 model adapter。

rollout / packer：

```text
vrl/rollouts/packers/diffusion.py
vrl/rollouts/collector/requests.py
vrl/rollouts/runtime/*
```

必须保留 video reward 需要的 generated video、prompt、seed、video_infos、reward metadata。不能让 reward 从全局变量或训练脚本状态读取 payload。

trainer：

```text
vrl/trainers/online.py
vrl/scripts/cosmos/train.py
```

需要分离 GRPO mechanism trainer 和 DiffusionNFT trainer path。DiffusionNFT 缺少 required tensors / reward score 时 fail-fast，不 fallback 到 existing GRPO。

config loader：

```text
vrl/config/loader.py
tests/config/test_load_all_experiments.py
```

需要识别 `algorithm.kind=diffusion_nft`、video reward config、Predict2.5 model config。

### Delete / Deprecate

删除旧 gap 文档：

```text
docs/cosmos_rl_gap.md
```

训练验证成功前不新增 Cosmos-Predict2.5 gap 文档。后续只有在真实 DiffusionNFT run 证明 optimizer update、非平 reward、生成 artifact、LoRA 权重变化之后，才新增 replacement gap doc。

不新增、不维护以下有效训练路线：

```text
Cosmos-Predict2 2B + aesthetic-only GRPO
Cosmos-Predict2.5 supervised V2W SFT / LoRA
deprecated FlowGRPO / DDRL WFM path
```

现有文件可以保留用于 debug，但 README / sprint / config 注释必须标清：

```text
cosmos_predict2_2b_grpo = debug-only mechanism run
```

## 5. 阶段一：先清理当前 repo 叙述

修改：

```text
README.md
```

要求：

- `cosmos_predict2_2b_grpo` 标为 debug-only / mechanism run。
- 明确记录真实 run 结果：`reward_std=0`、`adv_zero_rate=1`、`global_step=0`，不能算有效训练。
- 删除 `docs/cosmos_rl_gap.md`；训练验证成功前不新增 Cosmos replacement gap doc。
- 删除 `SPRINT_cosmos_rl_video2world.md`，避免旧 2B/aesthetic GRPO sprint 继续作为实现入口。
- README 当前只保留 `experiment/sd3_5_ocr_grpo` 作为 canonical/promoted recipe。
- Cosmos-Predict2.5 + DiffusionNFT 暂时只保留在本 sprint 内，不写进 README 推荐入口。
- 删除或下移 “aesthetic-only Cosmos RL recipe” 作为用户推荐入口。

## 6. 阶段二：Video Reward Interface

新增/修改：

```text
vrl/rewards/video_reward.py
vrl/rewards/remote_video.py
vrl/rewards/multi.py
configs/base/reward/video_reward.yaml
tests/rewards/test_video_reward.py
```

要求：

- `video_reward` 是统一入口，backend 可以是 `stub`、`local`、`remote`。
- 支持 image/video payload。
- 支持 reward function name，例如 `dance_grpo`、`cosmos_reason1`。
- 支持 response 多 score key，例如 `overall_reward`、`vq_reward`、`mq_reward`、`ta_reward`。
- score 必须 finite。
- timeout / HTTP error / missing score key 全部 fail-fast。
- 保存 raw response 到 reward debug artifact。
- 支持 stub video reward，用于无服务环境下验证训练链路，但 stub 必须显式命名，不能伪装成 official reward。

第一版配置建议：

```yaml
reward:
  components:
    video_reward: 1.0
  kwargs:
    video_reward:
      backend: remote
      enqueue_url: ${oc.env:REMOTE_REWARD_ENQUEUE_URL}
      fetch_url: ${oc.env:REMOTE_REWARD_FETCH_URL}
      token: ${oc.env:REMOTE_REWARD_TOKEN}
      reward_name: dance_grpo
      score_key: overall_reward
```

## 7. 阶段三：DiffusionNFT algorithm 壳

新增/修改：

```text
vrl/algorithms/diffusion_nft.py
vrl/config/loader.py
configs/base/algorithm/diffusion_nft.yaml
tests/algorithms/test_diffusion_nft.py
tests/config/test_load_all_experiments.py
```

要求：

- 新增 `algorithm.kind = diffusion_nft`。
- 不改现有 `grpo` 行为。
- 第一版只实现配置、loss contract、fail-fast。
- 不能悄悄复用 GRPO 假装 DiffusionNFT。
- 如果训练 step 缺少 DiffusionNFT 需要的 logprob、prepared transformer input、reward score，直接报错。
- 记录和 cosmos-rl `nft_beta`、`kl_beta`、`mini_batch`、`uncentralized_training` 的差距。

## 8. 阶段四：Cosmos-Predict2.5 model config

新增：

```text
configs/model/cosmos/predict2_5_2b.yaml
configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml
```

可能修改：

```text
vrl/models/families/cosmos/predict2_5/policy.py
vrl/models/families/cosmos/predict2_5/builder.py
vrl/models/families/cosmos/predict2_5/executor.py
vrl/models/families/cosmos/predict2_5/diffusion_nft_adapter.py
vrl/rollouts/families/specs.py
```

要求：

- 使用 `nvidia/Cosmos-Predict2.5-2B`。
- 明确 `model_revision = diffusers/base/post-trained` 或本 repo 对应字段。
- LoRA target modules 参考 cosmos-rl NFT config：

```text
to_k
to_out.0
to_q
to_v
ff.net.0.proj
ff.net.2
```

- 不沿用 Predict2 2B LoRA target modules。
- 支持 video reward payload 的必要 metadata。
- 先做 load + 1 prompt inference validation，再做 RL validation。

## 9. 阶段五：DiffusionNFT rollout / trainer 对齐

需要研究并映射：

```text
/home/mingfeiguo/Desktop/cosmos-rl/cosmos_rl/policy/trainer/diffusers_trainer/nft_trainer.py
/home/mingfeiguo/Desktop/cosmos-rl/cosmos_rl/policy/model/diffusers/cosmos_predict2_5_model.py
/home/mingfeiguo/Desktop/cosmos-rl/cosmos_rl/rollout/wfm_rollout/wfm_rollout.py
```

本 repo 可能要改：

```text
vrl/scripts/cosmos/train.py
vrl/trainers/online.py
vrl/rollouts/packers/diffusion.py
vrl/rollouts/runtime/*
```

目标：

- rollout 返回 DiffusionNFT 训练需要的 logprob / latent trajectory / prepared transformer input。
- trainer 可以接 video reward score，并在非零 reward variance 下执行 optimizer step。
- packer 保留 video frames、prompt、seed、reward metadata、video_infos。
- checkpoint 导出 LoRA，并能用 diffusers 加载。

## 10. 阶段六：真实验证

### 10.1 Stub video reward validation

目的：验证代码链路，不声称官方训练成功。

最小条件：

```text
model = Cosmos-Predict2.5-2B
algorithm.kind = diffusion_nft
reward = video_reward_stub
batch = 1 prompt
rollout = 2+ generations
train = at least 1 optimizer step
```

通过标准：

- 模型能加载。
- 能生成视频或帧 artifact。
- stub video reward 返回不同 finite score。
- DiffusionNFT loss 非 NaN。
- backward 成功。
- `global_step > 0`。
- checkpoint LoRA 权重发生变化。

### 10.2 Real video reward validation

目的：验证真实 video reward。优先尝试 remote reward service；如果服务环境不可用，可以先用本地 `cosmos_reason1` / `dance_grpo` wrapper，但不能把 local wrapper 伪装成 official service parity。

优先级：

```text
1. cosmos_reason1
2. dance_grpo
```

通过标准：

- remote backend 正常 enqueue / fetch；local backend 正常直接返回 score。
- response 包含配置的 `score_key`。
- `reward_std > 0` 或至少出现非零 advantage batch。
- `grad_norm > 0`。
- `global_step > 0`。
- 保存 reward raw response 和 generated video artifact。

## 11. 完成标准

本 sprint 完成需要同时满足：

- `cosmos_predict2_2b_grpo` 在 README 中不再作为推荐 Cosmos RL recipe。
- `video_reward` interface 测试通过。
- `algorithm.kind=diffusion_nft` 有独立配置和测试。
- Cosmos-Predict2.5 2B config 能加载。
- stub video reward 1-step DiffusionNFT run 通过，且 `global_step > 0`。
- real `dance_grpo` 或 `cosmos_reason1` video reward run 至少完成 1 个 optimizer step。
- metrics/checkpoint 证明 LoRA 权重实际改变。
- README 当前只保留 SD3.5 OCR canonical recipe；Cosmos 训练成功前不新增 README 推荐路线。
- Cosmos DiffusionNFT 真实训练成功后，再新增 gap doc，并把 README 从 planning-only 升级到 validated route。
