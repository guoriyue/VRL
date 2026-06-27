# SPRINT: 世界模型即环境（world-model-as-env）—— 用自己的 Cosmos/Wan 背后驱动 env.step()

状态：**planned（2026-06-27）**。目标：让本仓支持"环境 = 世界模型"——实现一个 `WorldModelEnv`,满足已有的 `Env` 协议(`vrl/rollouts/envs/contract.py`,在 `feat/physical-ai-vla-contract` 分支),其 `step(ActionChunk) → (EnvObservation, EnvRewardSignal)` 跑一次**本仓自己的** Cosmos/Wan 前向去预测下一帧观测,而不是调 MuJoCo。先用 Phase-0 probe 把"模型能不能被 step"这关跑掉,过门才往下。

## 0. 结论先行（Phase-0 静态已跑出 = `go_prototype`）

`vrl/scripts/eval/world_model_steppability_probe.py` 已实跑(无 GPU 静态部分),结论 **`go_prototype`**:

- **接缝是现成的**:`feat/physical-ai-vla-contract` 分支已交付 typed `Env`/`Policy`/`ActionChunk`/`ActionTrajectoryBatch` 契约 + `LiberoEnv` 模板 + Ray/weight-sync/编排——全和扩散无关,可原样复用。
- **可步进的两个零件,在你手上的 backbone 上都存在**(probe verified):
  - frame-prefix 条件槽 **存在**:`Cosmos2_5_PredictBasePipeline.prepare_latents` 已带 `video` + `num_frames_in` 参数(我们 wrapper 现在只是传 `num_frames_in=0` 没用它)。
  - action 注入接缝 **可扩展**:`DiffusionBackboneInput.extra` 是 free-form `dict`,Wan-I2V 已经用它穿 `condition`/`image_embeds`,加一个 `action` key 非破坏性。
- **唯一封锁是 omni generator,且不需要它**:diffusers 0.37.1 没有 `Cosmos3*`/omni pipeline(与 `cosmos3_nano_generator_probe` 记录的封锁一致),只有 Cosmos 2.x。但 **Cosmos-Predict2.5 V2W / Wan-I2V 这两个 2.x backbone 就够做原型**,不必等上游。

→ 所以这不是 bandit 的换皮,是一次**刻意的 scope 扩张**;但 Phase-0 静态门已过,**真正的悬念前移到 Phase-1 的 action-fidelity**(加了 action,生成的下一帧到底听不听 action),那个要 weights。

## 1. 已有的接缝（reuse，全部可原样用）

- **`Env` 协议**(`vrl/rollouts/envs/contract.py`):`reset(spec)→EnvObservation`、`step(action)→(EnvObservation, EnvRewardSignal)`、`render_episode()`;`runtime_checkable`,签名对上即 drop-in。已强制"**环境独占 reward,policy 不准造 reward**"。
- **`LiberoEnv` 模板**(`vrl/rollouts/envs/libero.py`):持 `_t` + 帧 buffer 的模式,唯一要改的就是把 `self._env.step(...)` 换成世界模型前向。
- **动作/轨迹类型**:`ActionChunk`、`EnvObservation`、`ActionTrajectoryBatch`——刻意和扩散 `RolloutBatch` 平行,不复用。
- **`ActionPolicy` 协议** + RL-eligibility 规则(`can_replay_logprob=False` → trainer 拒绝 RL 更新)。
- **编排 / Ray / weight-sync**:`vrl/rollouts/orchestration`、`vrl/trainers/weight_sync.py`、`vrl/generation/ray/`——policy-agnostic,actor-in-world-model 需要的"把 trainable state 同步到 rollout worker"它们已经做。

## 2. 缺的（net-new，诚实清单）

1. **`WorldModelEnv` 适配器**:实现 `Env` 协议,`step()` 里跑 Cosmos/Wan 前向 + 解码成相机图 `EnvObservation` + 出 `EnvRewardSignal`。
2. **action 条件**(最硬):扩散路径里现在**零** action 输入。可用接缝 = Wan-I2V 的 `extra=` dict(`wan_2_1/model.py:740-745`)或 DIAMOND 式 AdaGN 注入。**注意**:加 `extra['action']` 是非破坏的,但 `WanI2V*BackboneRunner` 必须被改成**真的消费**它,否则不条件化。
3. **自回归逐 chunk + 跨步带状态**:今天整段原子生成(`executor.py:725-833` 一次跑完整段)。要把循环翻成"外层 chunk、每 chunk 一个去噪子循环、出 `obs_{t+1}` 后停"。跨 `step()` 要带:条件 latent 前缀(上 K 帧 VAE 编码,从生成帧来)+ `cond_mask`/`cond_indicator`。**扩散 DiT 没有 KV-cache 要带**(双向 attention 无状态;KV-cache 只在 AR 家族)。标准做法 = **Vid2World**(去噪当前 chunk 时历史帧保持 clean,防未来偷看)。
4. **reward 来源**:生成世界不能自评(会奖励自己的幻觉)。标准答案 = **外部 VLM critic**(Genie3+SIMA 用独立 Gemini)。`vrl/rewards/`(ABC/composite/remote/registry + video-reward suite)已能托管。
5. **policy/actor + actor-critic trainer**(仅 Target I 需要,见 §3)。

## 3. 两个训练目标 —— 第一步选 Target II

- **Target I —— 在(冻结)世界模型里训 policy**(MBRL/actor-critic on imagined rollouts)。真 sequential RL,要 value head + GAE。`vrl/algorithms/base.py` 的 `Algorithm` 协议**没有 critic 接缝**——硬塞会污染 bandit 路径。需要**独立的 `ActorCriticTrainer`**。更大、更险。
- **Target II —— RL-finetune 世界模型本身**(让生成听 action / 更一致)。**就是**现有 diffusion-GRPO bandit + 一个 action-following reward。**复用 ~95%**:整个 `OnlineTrainer` + GRPO/FlowDPPO/NFT + `multisegment.py`(已能按段加权 loss);per-frame reward 折现成一个标量 advantage **零 trainer 改动**,只加一个 reward fn。

**第一步果断 Target II**:codebase 重心(`OnlineTrainer`、GRPO/NFT、连续编排、weight-sync、replay-logprob parity 守卫)都是为"被训的是扩散生成器"建的,Target II 正是它;Target I 要一整套平行 trainer + critic,无现成。**先 II,I 作为独立第二里程碑,绝不把 critic 塞进 bandit `Algorithm` 协议。**

## 4. 分阶段计划（每阶段一个 KILL-RISK 门）

复用本仓 probe/sprint 文化:**干净记录 blocker 本身就是 PASS**。沿用 `SPRINT_physical_ai_model_support.md` §P4 的五道训练门(env smoke / policy eval / action-logprob 契约 / **reward variance exists** / episode artifact 有界)。

- **Phase 0 —— 可步进/action-条件封锁(KILL-RISK)。已部分完成。**
  `world_model_steppability_probe.py` 静态部分**已跑 = `go_prototype`**:frame-prefix 槽 + action seam 都在,omni generator 封锁但不需要。**剩 `--load-weights` 的 live 子检查**(用真 checkpoint 调 `prepare_latents(num_frames_in=K>0)` 喂帧前缀)待在 GPU 机上补跑。
- **Phase 1 —— 证明 1 步 action-条件的下一帧(真 KILL-RISK)。**
  给初始帧 + 一个 action → 一个**可见响应 action** 的下一 chunk(换 action → 换帧)。DIAMOND 式 n≈3 步去噪。**门:action-fidelity 非平凡**(帧不是 action-invariant)。这是整条路最大的开放问题——backbone 是为 text/reference 训的,能不能在合理预算内被教会听 action(Pandora 失败模式 = 控制信号太粗)。
- **Phase 2 —— 短闭环 episode + reward 方差(= §P4 "reward variance exists" 门)。**
  接 `WorldModelEnv`(reset→step×N,带 latent 前缀 + cond_mask 跨 chunk),插 `vrl/rewards/` 外部 VLM critic。门:episode 不崩(目标 horizon 内不漂移坍塌)+ critic 给非退化 reward + artifact 存储有界。
- **Phase 3 —— 接 RL recipe**(五门全过后):Target II = action-following reward → 现有 GRPO on world model。Target I 独立 sprint。

## 5. 诚实的赌注判断

**Phase-0 静态门已过,把这个从"可能 not_yet"升级成"原型在现有 backbone 上可行"——但真正的悬念是 Phase-1 的 action-fidelity。** 它仍一次性动三个承重假设:sequential env(跨步带状态、翻转驱动)、新的外部 reward(VLM critic,开放世界 reward 未解)、(Target I)协议没接缝的 value function。一致性是活跃前沿(Genie 3 最优"几分钟一致、真记忆约 1 分钟",是你任务 horizon 的硬上限)。

> **判决:Phase 0 已过 → 值得花 Phase 1 那个便宜的 action-fidelity probe。** 过了 → 这是值得专门推进的第二支柱(用自己的世界模型当 env,训 agent 而非只 finetune generator,即 DreamerV3/DIAMOND 形状)。Phase 1 堵了(backbone 教不会听 action)→ 大声说 "not yet",hold,守住 bandit 北极星,等上游交互式 checkpoint。**与 `docs/NORTH_STAR.md` 一致:这是 scope 扩张、不是把 bandit 训练换个说法。**

## 6. 验收

- [ ] Phase-0 probe 的 `--load-weights` live 子检查在 GPU 机补跑,结论(frame-prefix 真接受生成帧前缀?)落本文件。
- [ ] Phase-1 action-fidelity probe:换 action 换帧的定量证据(或 negative)。
- [ ] 过门后:`WorldModelEnv` 实现 `Env` 协议,episode 跑通,reward 方差存在。
- [ ] Target II 先行:action-following reward fn + 现有 GRPO,**不动 `OnlineTrainer` 的 bandit 路径**。
- [ ] 任何一门 KILL → 记录 blocker + "not yet",不强推。

**参考**
- 接缝/类型:`vrl/rollouts/envs/contract.py`、`vrl/models/vla/policy.py`、`vrl/rollouts/envs/libero.py`(分支 `feat/physical-ai-vla-contract`)
- probe:`vrl/scripts/eval/world_model_steppability_probe.py`(本 sprint 新增)
- frame-prefix 槽:`Cosmos2_5_PredictBasePipeline.prepare_latents`(`video`/`num_frames_in`);wrapper `vrl/models/diffusion/cosmos/predict2_5/model.py:338-351`(现传 `num_frames_in=0`)
- action seam:`vrl/models/diffusion/common/backbone.py:23`(`DiffusionBackboneInput.extra`)、`vrl/models/diffusion/wan_2_1/model.py:740-745`(`extra={condition,image_embeds}`)
- omni 封锁:`vrl/scripts/eval/cosmos3_nano_generator_probe.py`(分支)
- bandit trainer / 无 critic 协议:`vrl/trainers/online/trainer.py`、`vrl/algorithms/base.py`、`vrl/algorithms/grpo/multisegment.py`
- canonical:DIAMOND(2405.12399)、DreamerV3(2301.04104)、Genie 3+SIMA 2、Vid2World(2505.14357)
