# SPRINT：Cosmos-RL DiffusionNFT 与 Cosmos-Predict2.5 实现

## 0. 目标

把本 repo 的 Cosmos RL 路线从 Cosmos-Predict2 2B mechanism run，升级到更接近官方 cosmos-rl 的 WFM RL 实现。

本 sprint 独立于 `SPRINT_cosmos_rl_video2world.md`。只有当前 Cosmos-Predict2 2B Video2World sprint 稳定后再执行。

## 1. 参考来源

本地参考：

```text
/home/mingfeiguo/Desktop/cosmos-rl
/home/mingfeiguo/Desktop/cosmos-rl/configs/cosmos-predict2-5/cosmos-predict2-5-2b-720-nft.toml
/home/mingfeiguo/Desktop/cosmos-rl/configs/cosmos-predict2-5/cosmos-predict2-5-2b-480-grpo.toml
/home/mingfeiguo/Desktop/cosmos-rl/reward_service
```

本 repo paper：

```text
docs/papers/cosmos_predict2_5_world_simulation_2511.00062.pdf
```

外部参考：

```text
https://arxiv.org/abs/2511.00062
https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/
https://nvidia-cosmos.github.io/cosmos-rl/wfm/overview.html
https://github.com/nvidia-cosmos/cosmos-rl
https://github.com/nvidia-cosmos/cosmos-predict2.5
```

## 2. 阶段一：写差距文档

新增：

```text
docs/cosmos_diffusionnft_predict25_gap.md
```

需要对比：

- 本 repo `GRPO + FlowMatchingEvaluator` vs cosmos-rl `diffusion_nft`。
- 本 repo in-process reward vs cosmos-rl remote reward service。
- 本 repo Cosmos-Predict2 2B vs Cosmos-Predict2.5 2B / 14B。
- 本 repo single-node runtime vs cosmos-rl policy/rollout parallelism。
- 本 repo manifest format vs cosmos-rl WFM dataset format。

## 3. 阶段二：新增 DiffusionNFT algorithm 壳

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
- 第一版只实现配置和 fail-fast，不能悄悄复用 GRPO 假装 DiffusionNFT。
- 如果训练步缺少 DiffusionNFT 需要的 logprob / prepared transformer input，直接报错。

## 4. 阶段三：对齐 cosmos-rl model 接口

需要研究并映射：

```text
/home/mingfeiguo/Desktop/cosmos-rl/cosmos_rl/policy/model/diffusers/
```

本 repo 可能要改：

```text
vrl/models/families/cosmos/policy.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
vrl/rollouts/collector/factory.py
vrl/rollouts/runtime/*
```

目标接口：

- rollout 能返回训练所需 logprob / latent trajectory。
- replay/evaluator 能重建 DiffusionNFT 所需输入。
- reference / condition video 信息不能在 pack/gather 后丢失。
- 支持 video reward payload 的必要 metadata。

## 5. 阶段四：remote reward service

新增/修改：

```text
vrl/rewards/remote_video.py
vrl/rewards/multi.py
configs/base/reward/remote_video.yaml
tests/rewards/test_remote_video.py
```

要求：

- 支持 image/video payload。
- 支持 response 多 score key。
- score 必须 finite。
- timeout / HTTP error / missing key 全部 fail-fast。
- 保存 reward debug raw response。

## 6. 阶段五：Cosmos-Predict2.5 model config

新增：

```text
configs/model/cosmos/predict2_5_2b.yaml
configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml
```

可能修改：

```text
vrl/models/families/cosmos/policy.py
vrl/models/families/cosmos/builder.py
```

要求：

- 使用 `nvidia/Cosmos-Predict2.5-2B`。
- LoRA target modules 参考 cosmos-rl NFT config，不沿用 Predict2 2B。
- 明确 checkpoint revision / backend。
- 先做 load + 1 prompt inference smoke，再做 RL smoke。

## 7. 阶段六：训练 smoke

最小 smoke：

```text
model = Cosmos-Predict2.5-2B
algorithm.kind = diffusion_nft
reward = remote_video 或 stub remote reward
batch = 1 prompt
rollout = 1 generation
train = 1 optimizer step
```

通过标准：

- 能加载模型。
- 能生成视频或帧 artifact。
- reward service 能返回 finite score。
- DiffusionNFT loss 非 NaN。
- backward 成功。
- 保存 checkpoint。

## 8. 完成标准

本 sprint 完成需要同时满足：

- `docs/cosmos_diffusionnft_predict25_gap.md` 完成。
- `algorithm.kind=diffusion_nft` 有独立配置和测试。
- Cosmos-Predict2.5 2B config 能加载。
- remote video reward 能通过测试。
- 1-step Cosmos-Predict2.5 DiffusionNFT smoke run 通过。
- README 明确区分：
  - Cosmos-Predict2 2B mechanism run
  - Cosmos-Predict2.5 DiffusionNFT run
  - official cosmos-rl reproduction gap
