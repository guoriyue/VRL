# SPRINT: Wan 2.1 I2V 14B GRPO proof run（图生视频 RL 落地验证）

状态：**parked（2026-07-09 复核）**。§6 核心契约已在真权重上验证 PASS；
单卡 GRPO 被 16.4B transformer 的驻留需求结构性阻塞。触发条件：至少两卡且 replay
分片路径可用，完成剩余 multi-GPU train step。

> **2026-07-11 reward lease 更新：**active 母配方不再声称两个 7B reward 能在同一
> 进程共享一张 GPU。Kling 是唯一共享 GPU/CuMem owner，VideoCon-Physics 显式在 CPU
> 执行；因此旧文中的“第三张 reward GPU”是历史拓扑，不再是当前 blocker。真正未解的
> 仍是 16.4B trainer replay 分片与多卡真机 step。单 Kling smoke 配方继续保留，因为它
> 避免 CPU VideoCon 的吞吐成本，适合契约验证。

> **2026-07-02 真机实测结论（单卡 RTX 5090 32GB）：**
> 1. **smoke run 实跑 OOM，且是结构性的**：`online_grpo_i2v_smoke_single_gpu` 在 driver 侧
>    `build_wan_2_1_replay_runtime_bundle` 的 LoRA attach 阶段 OOM（allocator 已占 30.65GiB /
>    31.33GiB）。根因：本 sprint §2 的"14B bf16 ~28GB"用的是 **T2V** 的参数量；I2V 带 image
>    cross-attention，**实际 16.4B 参数**（HF safetensors index total_size=65.6GB fp32 →
>    bf16 ≈ 32.8GB），**裸 transformer 就比整张卡大**。`offload_mode: sequential` 只救 rollout
>    （streaming 前向），救不了 replay 反传（权重必须全量驻留）。→ **单卡 32GB 上 I2V 14B 的
>    train step 不可能，无论怎么调 microbatch/ckpt/LoRA**；需 ≥2 卡（replay 分片/更大卡）。
> 2. **§6 的 I2V 专属硬契约已用真权重单独验证 PASS**：新探针
>    `vrl/scripts/eval/wan_i2v_logprob_parity_probe.py`（rollout 记录 old_log_prob →
>    `export_replay_tensors`（condition+image_embeds）过 CPU wire → `restore_eval_state` →
>    replay 前向重算 log_prob，逐步对比；与 `run_denoise_steps` / `DiffusionSDELogProbEvaluator`
>    同一代码路径）。结果（480p×33f×10 步，batch=2，sequential offload）：
>    **mean/max |fresh−old| = 0.0 逐位一致，ratio=1.0**（阈值 0.01，同 trainer
>    `debug.first_step`）；rollout 192s / replay 176s，**峰值仅 6.16GiB**；生成 mp4 已目检——
>    跟随首帧条件（汤锅→木勺入画搅动），真实运动非静止非噪声。报告
>    `outputs/wan_i2v_parity_probe/parity_report.json`。
> 3. **数据（任务 3）已就位**：`data/external/videophy_i2v_smoke/`（7 train + 2 eval，
>    frame-0 PNG 832×480 + manifest + report.json，本地 examples 绕过官方 403；未写 canonical 路径）。
>    运行时用 `data.manifest/eval_manifest/artifact_data_root/source_report` 四个 override 指过去。
> 4. **train step 仍属家族无关共享层**（GRPO loss / trainer 已在其他家族真跑过），I2V 家族喂给它的
>    唯一新东西正是本探针验证的 replay log_prob 与条件重建。**剩余唯一缺口 = 多卡资源上跑一次完整
>    smoke**（任务 2/4 合并为多卡项）。**链路实现已完整接好**（registry → I2V executor → reference-image
条件 → GRPO loss → replay 重建条件，见下 §1 的端到端证据）；本 sprint 是这条路径唯一的剩余项：
**把 Wan 2.1 I2V 14B GRPO 在真机上跑通一次**，验证「首帧条件穿过梯度回放」的契约与 reward 信号能动。

> 拆分理由同 [[SPRINT_wan_2_2_proof_run]]：代码 + CPU 单测属一类（基建已就绪）；proof run 是
> 「真机 + 显存 + 端到端契约」，性质不同、卡在不同资源（GPU 显存），独立成 sprint。
>
> 与 [[SPRINT_wan_2_2_proof_run]] 的关系：2.2 是 **dual-stage 2×14B**（~56GB bf16，需 3 GPU 角色，
> 2 卡硬阻塞）；本 sprint 是 **单塔 I2V-14B**，有 `offload_mode: sequential` 单卡 32GB 逃生口，
> **是最有希望在现有硬件上先跑通的 I2V RL 路径**。两者共享 reward 栈与 videophy 数据，复用 2.2 run
> 已踩平的坑（videophy 403 本地绕过、3 GPU 角色拓扑）。

---

## 0. 一句话

**仓库支持图生视频且 I2V 的 GRPO 链路全通——但从没实跑过一次。** `configs/model/diffusion/wan_2_1/i2v_14b.yaml`
的 `torch_compile` 注释自己写着 *"OFF until the I2V base GRPO run lands (multi-GPU-blocked)"*；recipe 把
`offload_mode` 设成 `none`；`outputs/` 下没有任何 I2V 训练 checkpoint。唯一实跑证据是纯推理探针
`vrl/scripts/eval/wan_i2v_base_sample.py`。**唯一缺的就是一次真正执行的 run。**

## 1. 现状：链路已全通（proof run 要验证的就是这条链）

| 环节 | 文件 | 关键证据 |
|---|---|---|
| experiment recipe | `configs/experiment/diffusion/wan_2_1/online_grpo_physics_i2v.yaml` | 组合 `/model/.../i2v_14b` + `/reward/kling_video_reward` + `/reward/videocon_physics` + `/dataset/videophy_i2v`；`entrypoint=...wan_2_1.train:train_wan_2_1_i2v_grpo` |
| 入口分发 | `vrl/scripts/train.py` | 解析 `trainer.entrypoint`（`module:function`）动态 import 调用 |
| 族训练 | `vrl/scripts/diffusion/wan_2_1/train.py` | `train_wan_2_1_i2v_grpo` → `run_online_recipe(family='wan_2_1_i2v', collector_kwargs_getter=_i2v_collector_kwargs)`；`_i2v_collector_kwargs` 校验每行 `reference_image` |
| 注册表选 executor | `vrl/rollouts/families/registry.py` | `wan_2_1_i2v`（task=i2v, `supports_reference_conditioning=True`）→ `Wan_2_1I2VChunkExecutor`，打开 `include_reference_image` |
| 采样（首帧条件） | `vrl/generation/diffusion/executor.py` | `ReferenceConditionedChunks` load 首帧；共享 `run_denoise_steps`：`forward_step → sde_step_with_logprob` 逐步记 GRPO 轨迹 |
| I2V 模型 | `vrl/models/diffusion/wan_2_1/{model,runner}.py` | 首帧 → CLIP `image_embeds` + 潜空间 `condition`；`runner` 通道拼接 `cat([latents, condition], dim=1)`；无 `reference_image` 直接 raise |
| 奖励 | `vrl/rewards/functions/{kling_video_reward,videocon_physics}.py` | 本地 HF 视频奖励，Ray pool，mp4；权重 `motion_quality 0.3 / physical_commonsense 0.7` |
| 损失 + 回放 | `vrl/algorithms/grpo/continuous.py`、`vrl/models/diffusion/wan_2_1/model.py` | GRPO clipped-PPO（模态无关）；`export_replay_tensors` 存 `condition+image_embeds`，`restore_eval_state` 重建 → **梯度也看到同一个首帧** |
| 权重同步 | `vrl/trainers/weight_sync.py` | LoRA 状态 CPU 化推给 Ray rollout worker（族无关） |

**关键设计点（proof run 的硬验收对象）**：首帧条件不是只在采样时用一下——回放时 `export_replay_tensors`
把 `condition`+`image_embeds` 存下、`restore_eval_state` 重建，所以 RL 目标真正看到首帧条件，**没有
「推理有条件、训练丢条件」的偷工**。这条「rollout 与 replay 条件一致」正是 proof run 必须逐步验证的契约。

## 2. 当前缺口（为什么还没跑）

| 缺口 | 现状 | 谁能补 |
|---|---|---|
| **显存：14B 单卡** | ~~14B bf16 ~28GB + LoRA + 激活，单张 32GB 边缘/超~~ **2026-07-02 实测证伪：I2V transformer 是 16.4B 参数，bf16 ≈ 32.8GB > 整张卡**（28GB 是 T2V 的数）。sequential offload 只救 rollout 前向；trainer replay 反传需全量驻留 → **单卡 train step 结构性不可能** | **多卡**（原任务 1 的"单卡挤进去"路线已死） |
| **拓扑：3 GPU 角色** | wan 物理 recipe 解析出 `trainer=[0] rollout=[1] reward=[2]`（videocon_physics 视频奖励独占 1 卡）。2 卡硬件放不下（见 [[SPRINT_wan_2_2_proof_run]] §8#1） | 任务 2（≥3 卡，或 reward 错时共卡） |
| **数据集：videophy 403** | `videophy_i2v` 要从官方 URL 解码 frame 0 作参考图，URL 返回 HTTP 403 | 已有本地绕过（任务 3，复用 2.2 run 的 manifest） |
| **真机 proof run** | 无 run 证据；契约（log_prob 一致）未触及 | 任务 4（阻塞在任务 1/2 解一） |

## 3. 任务 1 — 单卡 smoke 配置（code-only，可立即做）

加一个单卡可跑的 smoke override（**不改** `online_grpo_physics_i2v.yaml` 母配方，新建一个瘦 experiment
或用 CLI override），目标是在 1 张 32GB 卡上把链路跑到 generate→replay：

- `model.offload_mode: sequential`（启用 32GB 逃生口；接受其慢，proof run 不追吞吐）；
- 砍 reward 到**仅 `kling_video_reward`**（去掉 videocon_physics 这第二个奖励模型，省一张卡的显存压力）；
  smoke 阶段只需证明「reward 可动 + 契约成立」，不需要双奖励信号。
- 小 `total_epochs` + 小 `rollout.n_samples_per_prompt`（group 仍需 ≥ GRPO 下限以算 advantage）；
- `num_steps`/`sde.window` 沿用母配方（注意：`sampling.num_steps` 必须 ≥ rollout sde window 上界，
  见记忆 `project_cosmos_streaming_smoke` 的同类坑）。

> 这一步把「3 GPU 角色」缩成「rollout+reward 同卡（kling 单奖励）+ trainer 同进程」，让任务 4 在
> **单卡或 2 卡**上就能触发，绕过 3 卡硬阻塞。

2026-07-01 更新：任务 1 的 code-only 部分已落地为
`configs/experiment/diffusion/wan_2_1/online_grpo_i2v_smoke_single_gpu.yaml`。
它继承 `i2v_14b`、`480p_33f`、`10_step_cfg_4_5`、`kling_video_reward`、
`videophy_i2v`，并显式设置：

```yaml
model.offload_mode: sequential
model.torch_compile.enable: false
rollout.n_samples_per_prompt: 2
rollout.prompts_per_batch: 1
rollout.samples_per_chunk: 1
reward.kwargs.kling_video_reward.score_key: motion_quality
```

保持母配方不变；完整 `kling + videocon_physics` 物理 reward 栈仍属于任务 2/正式 run。
现有 `tests/models/diffusion/wan_2_1/test_backbone_parity.py` 已覆盖
`export_replay_tensors -> restore_eval_state` 保留 `condition` / `image_embeds`；
`tests/rollouts/runtime/test_runtime_inputs.py` 已把新 smoke 配方纳入 I2V runtime launch
contract 覆盖。真机验收仍需实际跑 `vrl-train --config
experiment/diffusion/wan_2_1/online_grpo_i2v_smoke_single_gpu`。

## 4. 任务 2 — 多卡正式 run 的资源（阻塞在硬件）

若要跑**带 videocon_physics 的完整物理 reward 栈**（母配方），需 ≥3 GPU（trainer/rollout/reward 各一），
或在 ≤2 卡上验证 reward 与 rollout 错时共卡（`reward_devices ∩ rollout_devices`，
`vrl/ray/resources.py`）。这是与 [[SPRINT_wan_2_2_proof_run]] 共享的硬件缺口，不阻塞任务 1/3。

## 5. 任务 3 — 数据集（复用 2.2 run 已踩平的绕过）

官方 `videophysics/videophy_test_public` 视频 URL 返回 403。复用 [[SPRINT_wan_2_2_proof_run]] §8#2 的本地
绕过时，`third_party/videophy/examples/*.mp4` 解码 frame 0 → PNG 后只能建到
`data/external/videophy_i2v_smoke/` 或 `_scratch_*` 路径。7 样本够 smoke 跑通、不够信号；不得写入
canonical `data/external/videophy_i2v/manifests/{train,eval}.jsonl` 冒充正式数据。

## 6. 验收标准（finishing criteria）

- ✅ **契约硬指标（2026-07-02 真权重 PASS，逐位一致）**：rollout 记录的 `old_log_prob` 与训练 replay 重算**逐步一致**——且 I2V 路径特有的
  `condition`+`image_embeds` 经 `restore_eval_state` 重建后，replay 前向与 rollout 前向数值对齐
  （这是 I2V 区别于 T2I 的唯一新增契约面，最容易出 bug 的地方）。见 header 更新 #2 的探针与数字；
- ⬜ I2V GRPO smoke 端到端不崩、`kling_video_reward` 曲线可动（按 `project_first_trustworthy_curve` 的判据：
  固定 prompt 集 + BLOCK test，>2σ 才算信号，不把噪声当 learning）——**多卡 gated**（header 更新 #1）；
- ✅(rollout 侧) / ❌(trainer 侧) 单卡 `offload_mode: sequential` 路径峰值显存在 32GB 内：rollout+replay 前向实测峰值 **6.16GiB**；
  trainer 反传侧被 16.4B bf16 权重（~32.8GB）结构性排除；
- ✅ T2I / T2V 既有路径回归不变（2026-07-02 全量 CPU 单测绿）。

## 7. 下游解锁（proof 通过后）

- 翻 `configs/model/diffusion/wan_2_1/i2v_14b.yaml` 的 `torch_compile.enable: true`
  （注释明说唯一 defer 理由就是「base run 未验」，compile 路径已验 parity-safe ~1.1x）；
- 解锁 [[SPRINT_wan_2_2_proof_run]]：2.1 单塔 I2V proof 通过后，dual-stage 2×14B 的 boundary 路由
  契约验证可在其上增量进行。

## 8. 阻塞 / 非目标

- **阻塞**：任务 4 的完整物理 reward 栈需 ≥3 卡（任务 2）；任务 1（单卡 smoke 配置）/任务 3（数据）
  **不阻塞**，可先行——单卡 smoke 是本 sprint 优先项。
- **非目标**：
  - 不在本 sprint 调 reward 信号/超参到「真学到东西」——proof run 只证**链路跑通 + 契约成立**
    （reward 真实上升属后续 tuning sprint，见 `project_first_trustworthy_curve` 的负面教训）；
  - 不碰 GRPO/flow-matching 共享算法层（已模态无关，I2V 走同一条 loss）；
  - 不做 Wan 2.2 dual-expert（归 [[SPRINT_wan_2_2_proof_run]]）；
  - 不做 cosmos predict2 v2w 的 I2V proof（同类但独立家族，另起）。

## 9. 引用

- recipe / model / dataset：`configs/experiment/diffusion/wan_2_1/online_grpo_physics_i2v.yaml`、
  `configs/model/diffusion/wan_2_1/i2v_14b.yaml`、`configs/dataset/videophy_i2v.yaml`
- 入口 / 族训练：`vrl/scripts/train.py`、`vrl/scripts/diffusion/wan_2_1/train.py`（`train_wan_2_1_i2v_grpo`、
  `_i2v_collector_kwargs`）
- I2V 条件链：`vrl/rollouts/families/registry.py`（`wan_2_1_i2v`）、
  `vrl/generation/diffusion/executor.py`（`ReferenceConditionedChunks` / `run_denoise_steps`）、
  `vrl/models/diffusion/wan_2_1/{model,runner}.py`（`reference_image` raise / `export_replay_tensors` /
  `restore_eval_state` / `cat([latents, condition])` / `enable_sequential_cpu_offload`）
- 奖励：`vrl/rewards/functions/{kling_video_reward,videocon_physics}.py`、`configs/reward/kling_video_reward.yaml`
- 损失 / 回放：`vrl/algorithms/grpo/continuous.py`、`vrl/rollouts/evaluators/diffusion/sde_logprob.py`
- 资源 / 拓扑：`vrl/ray/resources.py`、`vrl/scripts/common/online.py`（strategy 校验）
- 推理基线（非 RL）：`vrl/scripts/eval/wan_i2v_base_sample.py`
- 相关 sprint：[[SPRINT_wan_2_2_proof_run]]（2×14B dual-stage，共享坑）、记忆
  `project_wan_i2v_14b_inference`（I2V 推理已 load-tested、GRPO 未跑）、
  `project_first_trustworthy_curve`（diffusion GRPO 判据）
