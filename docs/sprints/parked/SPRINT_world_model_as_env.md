# SPRINT: 世界模型即环境（world-model-as-env）—— 用自己的 Cosmos/Wan 背后驱动 env.step()

Status: **PARKED (2026-07-18)**. Resume when the typed environment contract lands
on main and a real action-conditioned checkpoint is available for the live
action-fidelity gate. The static probe alone is not executable support.

## 0. 结论先行（Phase-0 静态已跑出 = `go_prototype`）

已退役的 `world_model_steppability_probe.py` 曾完成无 GPU 静态盘点，结论
**`go_prototype`**。静态结果保留如下；当前唯一活门是
`vrl/scripts/eval/cosmos_predict25_frame_prefix_gate.py` 的真权重、真视频前缀张量检查；
它不证明当前 wrapper/forward 已消费这些张量：

- **接缝是现成的**:`feat/physical-ai-vla-contract` 分支已交付 typed `Env`/`Policy`/`ActionChunk`/`ActionTrajectoryBatch` 契约 + `LiberoEnv` 模板 + Ray/weight-sync/编排——全和扩散无关,可原样复用。
- **可步进的两个零件,在你手上的 backbone 上都存在**(probe verified):
  - frame-prefix 条件槽 **存在**:`Cosmos2_5_PredictBasePipeline.prepare_latents` 已带 `video` + `num_frames_in` 参数(我们 wrapper 现在只是传 `num_frames_in=0` 没用它)。
  - action 注入接缝 **可扩展**:`DiffusionBackboneInput.extra` 是 free-form `dict`,Wan-I2V 已经用它穿 `condition`/`image_embeds`,加一个 `action` key 非破坏性。
- **omni generator 已不再是本门的前提**：历史 Cosmos3 探针已被正式
  `vrl/models/families/cosmos/cosmos3/` 实现取代；本 Phase 仍刻意用
  Cosmos-Predict2.5 验证 frame-prefix seam。

→ 所以这不是 bandit 的换皮,是一次**刻意的 scope 扩张**;但 Phase-0 静态门已过,**真正的悬念前移到 Phase-1 的 action-fidelity**(加了 action,生成的下一帧到底听不听 action),那个要 weights。

## 1. 复用图（grounded：as-is / 改造 / 新写）

**整体 ~70-75% 可复用,但不对称:Target II ≈90% 原样,Target I 只复用下半截基础设施。** 分界线:内层机器(去噪循环、weight-sync、reward 契约、trajectory store、optimizer/strategy/checkpoint)能复用;外层"一次性 prompt→生成→打分整段"的 bandit 编排必须翻写。

### AS-IS —— 四根范式无关支柱(两目标通吃,最高杠杆)

| 复用 | 路径 |
|---|---|
| 权重同步(换 `sync_state_getter` 指向的 bundle 即可) | `vrl/trainers/weight_sync.py`、`vrl/generation/ray/weight_sync.py` |
| 整个 Ray 层(WorldModelEnv = 又一个 rollout 角色) | `vrl/ray/{resources,placement,actor_pool}.py` |
| 训练下半截(loss-agnostic,白送多卡 + 8-bit Adam) | `vrl/trainers/{strategy,fsdp,checkpointing,ema}.py` + optimizer 工厂 |
| 去噪内循环 + 注入接缝(**已存在**) | `generation/diffusion/executor.py:675-873`(`run_denoise_steps`)、`models/diffusion/common/backbone.py`(`DiffusionBackboneInput.extra`)、`cosmos/predict2/model.py`(V2W 已有 prefix-conditioning) |

外加 **reward + trajectory store**(设计上范式中立):`vrl/rewards/inference.py` 是 `(prompt,media)→score`、transport model-agnostic;`vrl/trajectory/types.py` 轴里**已有 `frame`/`observation`/`action`**(本就为存 episode 设计)。VLM critic = 抄 `rewards/models/videoscore2.py` 的 ~40 行壳 + 一行注册。

### ADAPT —— 骨架复用、改一处

| 组件 | 改什么 |
|---|---|
| `generation/diffusion/executor.py` | `run_denoise_steps` 内层**原样**;把外层 `forward_chunk_plan:468-496` 的"一次整段"翻成逐 chunk 自回归(reset→prepare(前缀,action)→去噪→解码→喂回) |
| `cosmos/predict2_5/model.py:320-375` `prepare_sampling` | 现写死 `prepare_latents(video=None, num_frames_in=0)`(冷启动);改成喂上一 chunk 帧 + `num_frames_in>0` + `extra['action']`,跨步带 latent |
| Wan-I2V | 只单图条件,**无 `num_frames_in` 多帧前缀** → 历史弱于 Cosmos;**Cosmos-Predict2.5 当主力 backbone,Wan 当 fallback** |
| producer/consumer/strict_on_policy | cadence 范式无关;`collect_prompt_batches` 换成 env-rollout collect,batch 类型泛化(`group_ids` 在 `ActionTrajectoryBatch` 已有) |
| VLM critic | 抄 `videoscore2` 壳,换 system prompt + `action_following` score keys + ckpt |

### NEW —— 现有 infra 不覆盖

1. **env-collector**(闭环 reset→step→`ActionTrajectoryBatch`)— I(II 复用 `collector/core.py` 只换 scorer)
2. **有状态 `WorldModelEnv` worker**(跨 `step()` 持 episode latent 状态)— both
3. **逐 chunk 自回归外层循环**(imagined rollout)— I
4. **action-conditioning encoder**(`ActionChunk.actions` → `extra`/embeds 张量)— both
5. **VLM-critic 内容**(`ActionFollowingVLMModel`:action-following system prompt + score keys + ckpt)— both
6. **`build_world_model_trajectory`**(frame 轴 + value/per-step-reward 张量)— I
7. **`ActorCriticTrainer` + value head + GAE + per-step reward 路径**— **I only**

## 2. 按训练目标拆

- **Target II(RL-finetune 世界模型)≈90% 原样复用**——它**就是**你在跑的 diffusion bandit:`OnlineTrainer.step` + `GRPO continuous.py` + `group_relative_advantages` + `build_diffusion_trajectory` + `collector/core.py` + `rewards.py score_many`(一标量/样本)全精确匹配。**唯一真新写 = reward 内容**(`ActionFollowingVLMModel`,走现有 registry)。无 per-step reward 问题,不重写 trainer。模型侧的两处 adapt(`prepare_sampling` 前缀条件、`extra['action']`)是机械抄 Predict2 V2W 已有路径。
- **Target I(actor-critic 训 policy)只复用下半截(~50%)**——硬件/strategy/optimizer/checkpoint/EMA/weight-sync/trajectory store + Ray 全复用;但上半截 RL objective 全新(env-collector、stateful worker、`build_world_model_trajectory`、`ActorCriticTrainer`+value+GAE)。

**第一步果断 Target II**:复用 ~90%,唯一新写是一个 VLM critic 壳。Target I 是独立第二里程碑。

## 别硬塞(踩坑警告)

- ❌ **别把 critic 塞进 `OnlineTrainer`/`Algorithm` 协议**:它只有 `compute_advantages_from_tensors`+`compute_loss`,内循环迭代**去噪步**,无 value/GAE/per-step seam。塞进去污染 II 依赖的 bandit 路径。写**平行的 `ActorCriticTrainer`**,只抬 optimizer/EMA/strategy/grad-accum helper。
- ❌ **别把去噪轨迹当 MDP 轨迹**:`run_denoise_steps` 出的是 `x_t→x_{t-1}` 去噪轨迹,不是 env-step 的 `ActionTrajectoryBatch`。
- ❌ **别把 `collector/core.py` 当 env loop**:名字 generic,实际写死 `request→generate→reward`(bandit);闭环 env 要新 collector,只 II 能骑现成。

## 4. 分阶段计划（每阶段一个 KILL-RISK 门）

复用本仓 probe/sprint 文化:**干净记录 blocker 本身就是 PASS**。沿用 `SPRINT_physical_ai_model_support.md` §P4 的五道训练门(env smoke / policy eval / action-logprob 契约 / **reward variance exists** / episode artifact 有界)。

- **Phase 0 —— 可步进/action-条件封锁(KILL-RISK)。已部分完成。**
  历史静态盘点**已跑 = `go_prototype`**；旧多用途脚本已退役。剩余 live 门由
  `cosmos_predict25_frame_prefix_gate.py --prefix-video ...` 独立承担：通过正式 family
  builder 加载真 checkpoint，再把真视频尾帧送入 `prepare_latents(num_frames_in>0)`，
  验证 upstream pipeline 能构造非空、shape-compatible 的前缀张量。wrapper/forward
  接线仍属于 Phase 1 实现范围。
- **Phase 1 —— 证明 1 步 action-条件的下一帧(真 KILL-RISK)。**
  给初始帧 + 一个 action → 一个**可见响应 action** 的下一 chunk(换 action → 换帧)。DIAMOND 式 n≈3 步去噪。**门:action-fidelity 非平凡**(帧不是 action-invariant)。这是整条路最大的开放问题——backbone 是为 text/reference 训的,能不能在合理预算内被教会听 action(Pandora 失败模式 = 控制信号太粗)。
- **Phase 2 —— 短闭环 episode + reward 方差(= §P4 "reward variance exists" 门)。**
  接 `WorldModelEnv`(reset→step×N,带 latent 前缀 + cond_mask 跨 chunk),插 `vrl/rewards/` 外部 VLM critic。门:episode 不崩(目标 horizon 内不漂移坍塌)+ critic 给非退化 reward + artifact 存储有界。
- **Phase 3 —— 接 RL recipe**(五门全过后):Target II = action-following reward → 现有 GRPO on world model。Target I 独立 sprint。

## 5. 诚实的赌注判断

**Phase-0 静态门已过,把这个从"可能 not_yet"升级成"原型在现有 backbone 上可行"——但真正的悬念是 Phase-1 的 action-fidelity。** 它仍一次性动三个承重假设:sequential env(跨步带状态、翻转驱动)、新的外部 reward(VLM critic,开放世界 reward 未解)、(Target I)协议没接缝的 value function。一致性是活跃前沿(Genie 3 最优"几分钟一致、真记忆约 1 分钟",是你任务 horizon 的硬上限)。

> **判决:Phase 0 已过 → 值得花 Phase 1 那个便宜的 action-fidelity probe。** 过了 → 这是值得专门推进的第二支柱(用自己的世界模型当 env,训 agent 而非只 finetune generator,即 DreamerV3/DIAMOND 形状)。Phase 1 堵了(backbone 教不会听 action)→ 大声说 "not yet",hold,守住 bandit 北极星,等上游交互式 checkpoint。**与仓库定位（原 NORTH_STAR，已删除）一致:这是 scope 扩张、不是把 bandit 训练换个说法。**

## 6. 验收

- [ ] 在 GPU 机运行 `cosmos_predict25_frame_prefix_gate.py --prefix-video ...`，把
  production-family pipeline 能构造 frame-prefix 张量的结论落本文件；不得外推为
  wrapper/forward 已消费这些张量。
- [ ] Phase-1 action-fidelity probe:换 action 换帧的定量证据(或 negative)。
- [ ] 过门后:`WorldModelEnv` 实现 `Env` 协议,episode 跑通,reward 方差存在。
- [ ] Target II 先行:action-following reward fn + 现有 GRPO,**不动 `OnlineTrainer` 的 bandit 路径**。
- [ ] 任何一门 KILL → 记录 blocker + "not yet",不强推。

**参考**
- 接缝/类型:`vrl/rollouts/envs/contract.py`、`vrl/models/vla/policy.py`、`vrl/rollouts/envs/libero.py`(分支 `feat/physical-ai-vla-contract`)
- live gate:`vrl/scripts/eval/cosmos_predict25_frame_prefix_gate.py`
- frame-prefix 槽:`Cosmos2_5_PredictBasePipeline.prepare_latents`(`video`/`num_frames_in`);wrapper `vrl/models/families/cosmos/predict2_5/model.py`(现传 `num_frames_in=0`)
- action seam:`vrl/models/steps/denoise/common/backbone.py`(`DiffusionBackboneInput.extra`)、`vrl/models/families/wan_2_1/model.py`(`extra={condition,image_embeds}`)
- Cosmos3 正式 owner:`vrl/models/families/cosmos/cosmos3/`（历史探针已退役）
- bandit trainer / 无 critic 协议:`vrl/trainers/online/trainer.py`、`vrl/algorithms/base.py`、`vrl/algorithms/grpo/multisegment.py`
- canonical:DIAMOND(2405.12399)、DreamerV3(2301.04104)、Genie 3+SIMA 2、Vid2World(2505.14357)
