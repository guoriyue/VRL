# Physical-AI 模型支持 — P0/P1 落地与 probe 决策记录

状态：**done（2026-06-24）**。性质：**P0 盘点 + P1 契约落地 + probe 决策记录**。
承接 `docs/sprints/done/SPRINT_physical_ai_model_support.md`。本 MR 只落地
sprint 中 **不依赖外部模型权重/模拟器** 的部分（P0 + P1）；Tier 2+ 的训练/环境
适配按 sprint 的非目标显式 gate 在后续 MR，方便拆分。

## 一句话

新建了 robotics/VLA 的 **类型化 seam**（环境拥有 reward、策略拥有 action/logprob、
collector 拥有 episode 分组、trainer 只消费 typed trajectory），加上一张 **可被测试
消费的 model-support matrix** 和两个 **Cosmos 3 probe**。两个 probe 都跑通并给出
诚实结论：Cosmos 3 generator 目前 **不在 Diffusers 路径里**，DROID policy 走
**eval-only** 契约成立。没有改动任何现有 Wan/Cosmos2.5/Echo diffusion 路径。

## 落地清单

### P0 — model-support matrix（盘点）

- `vrl/models/support_matrix.py`：`ModelSupportEntry` + `MODEL_SUPPORT_MATRIX`。
  每行记录 input/output/action payload/logprob 契约/env 需求/tier/status。
  probe 脚本读取对应行，`tests/models/test_support_matrix.py` 钉住结构不变量
  （key 唯一、action 模型必须声明 env、64B Super 只能 track、未 probe 的模型不能
  标 supported），所以这张表是 **被消费的资产**，不是会腐烂的文档。

### P1 — VLA/Env rollout 契约（草稿）

- `vrl/rollouts/envs/contract.py`：`EnvResetSpec` / `EnvObservation` /
  `ActionChunk` / `EnvRewardSignal` / `EpisodeArtifact` / `ActionTrajectoryBatch`
  + `Env` protocol。**刻意不复用** `RolloutBatch` 的 image/video-latent 字段
  （`observations=x_t` / `actions=x_{t-1}`），因为机器人 action、env state、success
  信号、episode video 的 ownership 不同（sprint §3.2）。
- `vrl/models/vla/policy.py`：`ActionPolicy` protocol，对标 `ReplayModel` 的
  replay/runtime 切分。eval-only 策略 `can_replay_logprob == False` 且
  `ActionChunk.log_prob is None`；可训练策略暴露可复算 logprob。
- 关键不变量：eval-only trajectory（`log_probs=None` → `is_trainable=False`）与
  可训练 trajectory 可区分，trainer 因此能 **拒绝** 在缺 logprob 时按 0 处理。
- `tests/rollouts/envs/test_vla_contract.py`：15 个 CPU 测试（含 reset/step 闭环
  smoke + protocol runtime-check），全过。

### P0 — probe harness

- `cosmos3_nano_generator_probe.py`（2026-08-10 已退役）：它完成了当时的
  Diffusers component/denoise 契约盘点。此后正式实现已落在
  `vrl/models/families/cosmos/cosmos3/`，旧探针不再是活入口。
- `vrl/scripts/eval/cosmos3_nano_policy_droid_probe.py`：盘点 DROID policy 的
  action payload，并把（真实或清晰标注的合成）轨迹打包进 `ActionTrajectoryBatch`
  草稿，记录 DROID vs LIBERO/RoboTwin/BEHAVIOR 的 embodiment gap。

## Probe 决策（本机实跑：RTX 5090 32GB, diffusers 0.37.1, torch 2.11）

| probe | 决策 | 依据 |
|---|---|---|
| Cosmos3-Nano generator | **negative** | 该 diffusers build 里 **没有 Cosmos3/Omni 的 Pipeline 类**（只有 Cosmos 2.x：`Cosmos2TextToImagePipeline` / `Cosmos2_5_PredictBasePipeline` 等）。omnimodal Cosmos 3 generator 尚未通过 Diffusers 暴露 → 现在不能进 diffusion seam。 |
| Cosmos3-Nano-Policy-DROID | **contract_validated_synthetic** | 无权重/无 server，用合成轨迹验证 seam 打通：`ActionTrajectoryBatch` 形状 `[1, 8, 16, 7]`，`is_trainable=False`（server-level JSON action 无可训分布，按 sprint 不伪造 logprob）。embodiment gap 已量化（DROID 7-dim 与 LIBERO 兼容；RoboTwin 14-dim、BEHAVIOR 26-dim 不兼容）。 |

两个结论都符合 Tier 1 验收：「不能训练也算成功，只要 contract 被说清楚」。

## 显式 gate（后续 MR，不在本 MR）

按 sprint §P4 训练 gate 与非目标，以下 **未** 落地，留给独立 MR：

- 不向 rollout family registry 注册任何 VLA family（没有可运行的 runtime builder
  之前注册 = 半截 family）。
- 不写 LIBERO/RoboTwin env 适配、不写 PI0.5/OpenVLA-OFT 策略实现、不写 robotics
  trainer/RL recipe（需先 env smoke + logprob 契约 pass）。
- 不碰 Cosmos3-Super 64B、不碰 BEHAVIOR-1K/IsaacSim。

## 复跑

```bash
python -m pytest tests/rollouts/envs tests/models/test_support_matrix.py -q
python -m pytest tests/rollouts/runtime/test_family_registry.py tests/generation/execution/test_chunk_gatherer.py -q
```
