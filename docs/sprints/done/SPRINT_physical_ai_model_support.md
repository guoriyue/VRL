# SPRINT: Physical-AI 模型支持路线图 — Cosmos 3 / robotics / video

状态：**DONE / robotics 方向已收官（2026-07-09 终局对账；原 planned 2026-06-25）**。
性质：**模型支持优先级 + 架构边界决策**，不是直接落地代码。
这份 sprint 承接 `SPRINT_model_family_coverage.md`，但把范围从“纯图像/视频生成模型覆盖”
升级到 **Physical AI**：世界生成、机器人动作、闭环环境 rollout、物理/时序 verifier。

> **终局对账（2026-07-09，对 git 实况）**：路线图被执行到 Tier-2/P3，随后 robotics 方向主动收官：
> - **P0+P1 落地（2026-06-24）**：model-support matrix + VLA/Env 类型化契约 + 两个 Cosmos3 probe，
>   实录 [[SPRINT_physical_ai_p0p1_landing]]（info/）。Cosmos3 generator probe **负结论**（不在
>   Diffusers 路径）；DROID policy eval-only 契约成立。
> - **Tier-2 真机跑通（2026-06-25）**：OpenVLA-OFT 7B × LIBERO-10 真实 MuJoCo 闭环 eval 走通 P1 契约，
>   实录 [[SPRINT_physical_ai_tier2_openvla_libero]]。关键翻案：官方 OFT checkpoint 是 continuous
>   L1 回归 head，**无可复算 logprob** → 天然 eval/SFT-only（推翻本文 §Tier-2 的 token-policy 假设）。
>   P3 的 PI0.5 flow-logprob probe 证明 RL-eligibility（44b7947c）。
> - **方向收官（2026-06-27，d22d7d5d）**：整个 VLA/Env 层（contract / policy / libero adapter /
>   support-matrix / probes / eval 脚本，-2080 行）从 main 删除，换成 video_world 的 future-reward
>   action path——仓库定位收敛为 **RLHF-finetune 世界模型**，不做 WM-as-env robotics。
>   两份 info/ 实录与本文保留为决策档案；若未来重启 robotics 线，从 d22d7d5d^ 找回契约实现。
> - **Cosmos3 主线移交**：generator 进 diffusion seam 由 [[SPRINT_cosmos3_full_support]] 接管
>   （blocked/run-verify-gated；`cosmos3` family skeleton 已在 registry）。本文 §3.1「不先塞进
>   diffusion family」的保守边界已被 cosmos-rl 原生 logprob 契约调查取代（见该 sprint）。
> - **Tier-0（video diffusion 主线）不受影响**，由各家族/reward sprint 继续。本文归档 done/。

## 0. 一句话

下一阶段不应该继续只补 image diffusion 家族。我们应该把模型支持分成三条线：

```text
video/world-model generator  -> 继续做，优先 Cosmos / Wan / Echo / Cosmos 3 Nano
robotics action policy       -> 新建 VLA/Env rollout seam，先 eval/probe，再谈 RL
Cosmos 3 omnimodal           -> 先接 Nano probe，不先碰 Super full training
```

真正要押的是 **video + robotics + world-action loop**。普通 T2I 模型覆盖只保留必要基线，
不要把资源耗在模型动物园。

## 1. 现状：本仓库已经有的资产

本仓库视频侧不弱，已经有：

| family | 当前状态 | 路径 |
|---|---|---|
| Cosmos Predict2 | video/world diffusion | `configs/model/diffusion/cosmos/predict2_2b.yaml` |
| Cosmos Predict2.5 | video/world diffusion + NFT recipes | `configs/model/diffusion/cosmos/predict2_5_2b.yaml` |
| Cosmos Anima Preview3 | image/video-adjacent Cosmos family | `configs/model/diffusion/cosmos/anima_preview3.yaml` |
| Wan2.1 / Wan2.2 | T2V/I2V diffusion | `configs/model/diffusion/wan_2_1/`, `configs/model/diffusion/wan_2_2/` |
| Echo | LTX-derived video transformer | `configs/model/diffusion/echo/release.yaml` |

这说明核心缺口不是“再接一个 diffusion runner”，而是：

```text
1. Cosmos 3 这种 omnimodal MoT 模型是否能进入现有 diffusion/AR seam？
2. robotics action policy 是否需要独立 Env/VLA rollout seam？
3. video reward / verifier / simulator loop 怎么成为主资产？
```

## 2. 外部模型面：cosmos-rl 与 Cosmos 3

### 2.1 cosmos-rl 已支持的有用方向

cosmos-rl 对我们最有价值的是 **robotics/VLA 支持形态**，不是它的普通 image recipes。

| 模型/环境 | 类型 | 对我们的价值 |
|---|---|---|
| PI0.5 / PI0 | diffusion/flow action policy | 高。机器人动作生成和 diffusion logprob 有相似性，但 payload 是 action，不是 image/video latent |
| OpenVLA / OpenVLA-OFT | token-action VLA | 高。适合复用 AR/token policy 思路，但 reward 来自 simulator |
| Cosmos-Policy-LIBERO-Predict2-2B | Cosmos policy eval | 中。先当 eval/policy-server 参考，不直接承诺训练 |
| LIBERO | MuJoCo long-horizon manipulation | 高。最适合作为本仓库第一个 closed-loop robot eval |
| BEHAVIOR-1K | OmniGibson/IsaacSim household tasks | 高但重。适合第二阶段 |
| RoboTwin | robot manipulation sim | 中。配置和 test 已有，但生态成本要实测 |
| ManiSkill | manipulation benchmark | 中。代码有 wrapper，先作为 adapter reference |

cosmos-rl 的 WFM 侧支持 SD3、Cosmos-Predict2.5、SANA Image/Video。这里对我们不是新方向；
我们的 Wan/Cosmos/Echo 栈已经更贴近 video RL。

### 2.2 Cosmos 3 当前模型面

Cosmos 3 是更重要的变化。官方仓库把它定义为 omnimodal world model，统一处理/生成：

```text
language, image, video, audio, action
```

当前公开模型面：

| 模型 | 规模 | 应先支持到什么程度 |
|---|---:|---|
| Cosmos3-Nano | 16B | **P0/P1 probe**：Diffusers/vLLM-Omni inference + tensor contract 盘点 |
| Cosmos3-Super | 64B | track/eval only，不先 full training |
| Cosmos3-Super-Text2Image | 64B | track；不是主线 |
| Cosmos3-Super-Image2Video | 64B | track；只在资源足够时做 evaluator/probe |
| Cosmos3-Nano-Policy-DROID | 16B | **robotics priority**：先 eval/policy-server adapter，再谈 RL |

Cosmos 3 generator 支持 text/vision/sound/action conditioning，输出 image/video/sound/action/text。
它还明确有 forward dynamics 与 action policy use case。这正好对应我们的未来方向：

```text
world generation  +  action-conditioned rollout  +  robot policy learning
```

## 3. 架构决策

### 3.1 Cosmos 3 不先塞进现有 diffusion family

现有 diffusion family 假设比较清楚：

```text
prompt/text embeds
latent x_t
timestep/scheduler
denoiser transformer
SDE/flow logprob
decoded image/video artifact
```

Cosmos 3 是 AR reasoner + diffusion generator 的 unified MoT。它可能能通过 Diffusers path
暴露 generator denoise loop，但不能先假设它等价于 SD3/Wan/Cosmos2.5。P0 必须先回答：

```text
generator path 是否暴露 trainable denoiser？
每个 denoise/action step 是否有可复算 logprob？
action output 是 diffusion token、JSON action、还是 policy-server API？
Diffusers path 与 vLLM-Omni path 的 tensor/layout 是否一致？
```

在这些问题没定前，只做 **probe adapter**，不做训练接入。

### 3.2 Robotics/VLA 需要新 seam

机器人 rollout 的最小闭环不是 `prompt -> video -> reward`，而是：

```text
env.reset(task)
obs_t -> policy -> action_chunk
env.step(action_chunk)
trajectory -> success/reward/video/logprob
```

所以不要把 VLA 硬塞进 `vrl/models/diffusion/*`。应新增一条显式 seam：

```text
vrl/models/vla/...
vrl/rollouts/envs/...
vrl/rollouts/families/vla_registry 或 family registry 的 vla 分支
ActionTrajectoryBatch / EnvRolloutBatch
```

这不是为了多文件而多文件。它是 protocol boundary：机器人动作、环境状态、success signal、
episode video、action logprob 的 ownership 都和 diffusion artifact 不同。

### 3.3 保持现有 video diffusion seam 不动

Wan / Cosmos2.5 / Echo 仍走现有 diffusion seam。不要为了 Cosmos 3 / robotics 改坏已有路径。
薄文件如 family `runtime.py` / `runner.py` 继续保留，因为它们是跨家族一致的 framework adapter，
不是无意义拆分。

## 4. 优先级

### Tier 0 — 守住现有 video diffusion 主线

继续维护：

```text
Cosmos Predict2.5 + Kling/physics/video reward
Wan2.1/2.2 I2V physics reward
Echo video experiments
```

这是现有可训练资产，不因 Cosmos 3 出现而停掉。

### Tier 1 — Cosmos3-Nano generator probe

目标不是训练，目标是判定它能不能进本仓库：

```text
1. 用官方 Diffusers path 跑 Nano T2V/I2V 最小样本
2. 记录 component graph：reasoner、generator、tokenizer、scheduler、media tokenizer
3. 记录 denoise/action tensor：shape、dtype、timestep、conditioning、output media/action
4. 判断是否能复用 DiffusionPolicy / RolloutBatch / replay logprob
5. 写一份 negative/positive decision note
```

验收：

```text
可以生成一个短视频或明确记录依赖/权限/显存 blocker
能说明 Cosmos3-Nano 是否属于现有 diffusion seam
不能训练也算成功，只要 contract 被说清楚
```

### Tier 1 — Cosmos3-Nano-Policy-DROID eval adapter

这是 robotics 方向最贴 Cosmos 3 的入口。先做 eval，不做 RL：

```text
policy context -> action output
optional rollout video
success metric / action trace
```

验收：

```text
能通过 policy-server 或本地 inference 得到 action trajectory
能把 action trajectory 包成 ActionTrajectoryBatch 草稿
能说明 DROID policy 与 LIBERO/BEHAVIOR/RobotWin 的 embodiment gap
```

### Tier 2 — PI0.5 + LIBERO

这是第一个真正适合本仓库做 RL 的 robotics model：

```text
PI0.5 = diffusion/flow action policy
LIBERO = 轻量闭环环境
GRPO/DAPO = cosmos-rl 已有参考
```

先支持 eval + logprob probe，再决定训练。

验收：

```text
LIBERO reset/step/video/success signal 跑通
PI0.5 action_chunk 输出与 env action_dim 对齐
action logprob contract 有单元测试或明确不可得结论
```

### Tier 2 — OpenVLA-OFT + LIBERO / RoboTwin

OpenVLA-OFT 是 token-action policy，更像 AR rollout：

```text
obs image + instruction -> action tokens -> denormalized action
```

它适合验证 `token policy + simulator reward` 这条线，但不应和 diffusion action policy 混成同一个实现。

验收：

```text
LIBERO eval 跑通
RoboTwin 只做 smoke，除非环境安装成本可控
token logprob 与 action success 能同时记录
```

### Tier 3 — BEHAVIOR-1K / IsaacSim

BEHAVIOR-1K 价值高，但工程重。等 LIBERO seam 稳定后再上。

```text
先不把 IsaacSim 当 P0，因为环境安装/渲染/物理栈会吞掉 sprint
```

### Tier 4 — Cosmos3-Super / Super-T2I / Super-I2V

64B 系列只 track，不作为当前训练目标。

```text
可以做 benchmark/eval/reward reference
不做 full-param training
不因为它 SOTA 就让系统设计绑定 64B
```

## 5. 执行计划

### P0. 盘点与 probe harness

- 新增 model-support matrix，记录每个候选模型的 input/output/action/logprob/env requirement。
- 为 Cosmos3-Nano 跑官方 inference probe，生成最小 T2V/I2V artifact 或记录 blocker。
- 为 Cosmos3-Nano-Policy-DROID 读 policy server path，确认 action payload。

### P1. VLA/Env rollout contract 草稿

定义最小 contract：

```text
EnvResetSpec
EnvObservation
ActionChunk
ActionTrajectoryBatch
EpisodeArtifact
EnvRewardSignal
```

核心原则：

```text
environment owns success/reward
policy owns action/logprob
collector owns episode grouping
trainer only consumes typed trajectory + reward
```

### P2. LIBERO first

- 接 LIBERO env wrapper 或复用 cosmos-rl 形状重写 adapter。
- 先跑 OpenVLA-OFT eval 或 PI0.5 eval，二选一即可。
- 记录 video artifact、success、episode length、action trace。

### P3. PI0.5 action diffusion logprob probe

- 确认 `flow_sde` / `flow_cps` / `flow_noise` 哪个能提供训练所需 logprob。
- 如果 logprob contract 不稳，PI0.5 先停在 eval/SFT，不进 GRPO。

### P4. 训练 gate

只有同时满足：

```text
env smoke pass
policy eval pass
action/logprob contract pass
reward variance exists
episode artifact storage bounded
```

才新增 RL recipe。

## 6. 非目标

- 不把 Cosmos 3 Super 当第一目标。
- 不把 VLA 强行塞进 diffusion family。
- 不在没有 simulator smoke 的情况下写 robotics trainer。
- 不先做 BEHAVIOR-1K/IsaacSim；LIBERO 过了再说。
- 不因为 cosmos-rl 有某个 config 就照搬；只借 contract 和 run shape。
- 不重写现有 Wan/Cosmos2.5/Echo diffusion seam。

## 7. 风险

| 风险 | 处理 |
|---|---|
| Cosmos 3 generator API 不暴露训练需要的 denoise/logprob | 只做 inference/eval adapter，训练不接 |
| action policy 输出是 server-level JSON，不是可训分布 | eval only，不伪造 logprob |
| robotics 环境成本吞掉 sprint | LIBERO first，BEHAVIOR-1K later |
| VLA 与 diffusion payload 混乱 | 新建 `ActionTrajectoryBatch`，不复用 image/video artifact 字段 |
| 64B 模型资源不可控 | Super 系列只 track |

## 8. 与已有 sprint 的关系

- `SPRINT_model_family_coverage.md`：仍负责 FLUX/Qwen-Image 等纯生成模型覆盖。
- `SPRINT_efficient_rollout_program.md`：负责 rollout 省算/调度，不决定模型优先级。
- `SPRINT_speculative_diffusion_rollout.md`：负结果仍有效；Cosmos 3 不自动改变 exact speculative 的高维方差墙。
- `SPRINT_video_character_motion_reward.md`：可作为 video/world reward 的下游验证。

## 参考

- NVIDIA Cosmos 3 official repo: https://github.com/nvidia/cosmos
- NVIDIA Cosmos Framework: https://github.com/NVIDIA/cosmos-framework
- Cosmos 3 technical report: https://arxiv.org/abs/2606.02800
- Cosmos 3 project page: https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- Cosmos 3 Hugging Face collection: https://huggingface.co/collections/nvidia/cosmos3
- Cosmos-RL repo: https://github.com/nvidia-cosmos/cosmos-rl
- cosmos-rl VLA docs: `/home/mingfeiguo/Desktop/cosmos-rl/docs/vla/overview.rst`
- cosmos-rl WFM docs: `/home/mingfeiguo/Desktop/cosmos-rl/docs/wfm/overview.rst`
- cosmos-rl PI0.5 B1K config: `/home/mingfeiguo/Desktop/cosmos-rl/configs/pi05/pi05-b1k-grpo-colocate.toml`
- cosmos-rl OpenVLA-OFT RoboTwin config: `/home/mingfeiguo/Desktop/cosmos-rl/configs/openvla-oft/openvla-oft-7b-fsdp2-8p8r-colocate-robotwin.toml`
- cosmos-rl Cosmos-Policy eval config: `/home/mingfeiguo/Desktop/cosmos-rl/configs/cosmos-policy/cosmos-policy-libero10-eval.toml`
- 本仓库 Cosmos/Wan/Echo configs: `/home/mingfeiguo/Desktop/wm-infra/configs/model/diffusion/`
