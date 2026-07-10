# SPRINT: Cosmos Predict2.5 RL — paper-faithful run（rollout 预算对齐论文）

状态：proposed / planned（2026-06-17）。**给另一台机器的要求**：复现 Cosmos-Predict2.5
RL 必须用 **paper-shaped 配置**（够显存才跑得起），**不要**沿用 field-notes 那次单卡
L40S 的缩水 override——那次只采了论文 **千分之二** 的 rollout 量，学不出东西。

> 来源：本篇核对了论文 PDF（`docs/papers/world-models/cosmos-predict2-5-world-foundation-models.pdf` 第 13 页
> §4.2.2 Reinforcement Learning）、仓库 config（`configs/experiment/diffusion/cosmos_predict2_5/`）、
> 以及 field-notes（`info/SPRINT_cosmos_training_field_notes.md`）。

## 0. 一句话

field-notes 那次单卡实跑**严重偏离论文**(每次更新只有 3 个 rollout vs 论文 256 个),
**但仓库的 `online_nft_kling_video_reward.yaml` 本身已经对齐论文**。"miss many rollout"
是**运行期被单卡 46GB 显存逼小**的妥协,不是配置 bug。要复现论文,**就跑那个已存在的
config,给够显存**(多卡或更大卡),别用 field-notes 的单卡 override。

## 1. 论文 §4.2.2 真实 RL 配方(已从 PDF 第 13 页核实)

原文要点:

> 每个 input condition 生成 **8 个 output**、每个 output **20 步去噪**,GRPO 组内归一化
> advantage;因显存约束,**每 2 个 conditional probability 算一次梯度、累积成 10 步**;
> **训练 256 步、batch size 32**;用 **fine-tuning 数据的 diffusion loss 做正则**(防 reward
> hacking);释放 **EMA** 权重。算法 **GRPO**,reward 用 **VideoAlign**。

→ 每次参数更新 = **8 × 32 = 256 个 rollout**;全程 256 次更新 = **65,536 个 rollout**。

## 2. 三方对比

| 项 | 论文 §4.2.2 | 仓库 config（`online_nft_kling_video_reward.yaml`） | field-notes 实跑（L40S 46GB） |
|---|---|---|---|
| 每 condition 输出 n | **8** | 8 ✅ | **3** ❌ |
| 每次更新 conditions（batch） | **32** | 32 ✅ | **1** ❌（§6: rbs=1） |
| **每次更新 rollout 数** | **256** | **256** ✅ | **3** ❌ |
| 去噪步数 | **20** | 20 ✅ | **10** ❌ |
| 梯度 timestep 切片 | **10**（每 2 步） | `tf=0.5`→10 ✅ | 未记 |
| 训练更新次数 | **256** | 256 ✅ | **50** ❌ |
| CFG | 未明示（GRPO 用 no-CFG 取干净 log-prob） | **no-CFG** ✅ | **CFG 7.0** ❌ |
| 分辨率/帧 | ≤720p/93f | 512p/93f | 480p/49f |
| 算法 | **GRPO** | DiffusionNFT ❌ | NFT |
| reward | **VideoAlign** | `kling_video_reward`（疑同源,待证） | kling |
| diffusion-loss 正则 | **有** | 没找到 ❓ | ❓ |
| lr（2B） | **3e-5 全参** | 1e-4 LoRA（注释:LoRA 要更高 lr、全参需多卡） | 未记 |

## 3. "miss many rollout" — 量化

- **每次更新**:实跑 3 个 rollout vs 论文 256 个 = **1.2%**。
- **全程**:实跑 50×1×3 = **150 个** vs 论文 256×32×8 = **65,536 个** = **~0.23%**。

实跑只采了论文 **千分之二** 的 rollout 量。GRPO 的学习信号是组内 advantage(`reward_std`),
rbs=1、n=3 时组内只有 3 个样本、每步只有 1 个 prompt——advantage 几乎是噪声。field-notes §6
观察到的 "reward bounced -1.2…-3.6 with no trend" 正是这个欠采样的后果。

## 4. 需要做什么(给另一台机器的 action)

**复现论文 = 跑仓库已对齐的 config,给够显存,不要单卡 override。**

```bash
# paper-shaped 配置已存在,直接用(NFT 变体):
#   configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml
#   → n_samples_per_prompt=8, rollout_batch_size=32, 512p_93f, 20_step_no_cfg,
#     total_epochs=256, timestep_fraction=0.5(→10 切片)
# 关键:这套 n=8/rbs=32/512p/93f 在单 L40S(46GB)塞不下,需要多卡或更大卡。
# 不要把它缩成 field-notes 的 n=3/rbs=1/480p/49f/10-step/CFG。
```

显存够之前,**不要**把 n/rbs/分辨率往下调来"先跑通"——那等于回到 field-notes 的欠采样,
得到的还是噪声曲线。

## 5. 仍未解决的偏差 / 待决策(即便用仓库 config 也离论文有差)

1. **算法**:论文是 **GRPO**,`online_nft_kling_video_reward.yaml` 用的是 **DiffusionNFT**。
   **决策(2026-06-20,定档):保留 NFT。** 2026-06-20 的 480p_33f DDP 2x1 run 跑了 6 个 epoch,
   reward 在 −4.37…−4.56 平带震荡、无学习信号——但根因是**欠采样/epoch 数太少**(论文 256 次
   更新,我们才 6 次;组内 n=8 的 advantage 信号短期就是噪声),**不是 NFT vs GRPO 的算法选择问题**。
   先把 NFT 跑够 epoch / 验 reward 模型能否区分(见 §7 待办),不要因为 flat 就翻算法。
   **但 paper-faithful GRPO 不是翻个 flag 能切的**:仓库原本**没有 predict2.5 的 GRPO recipe**;唯一的
   cosmos GRPO 是 `cosmos_predict2/online_grpo_{kling_video_reward,v2w_reference}.yaml`——**predict2_2b
   家族**(不是 predict2.5),且 v2w 那个是 **Video2World + per-sample 参考图条件**(`/dataset/video_world_v2w`、
   `reference_mode: per_sample`),走另一个数据/入口。
   **→ 已落地（2026-07-08）**:入口 `train_cosmos_predict25_grpo`(`vrl/scripts/diffusion/cosmos/train.py`,
   t2w、无 reference 接线)+ family recipe `configs/recipe/online/cosmos_predict25_grpo.yaml` + 实验
   `configs/experiment/diffusion/cosmos_predict2_5/online_grpo_kling_video_reward.yaml`(paper-shaped:
   n=8×rbs=32、512p_93f、20步no-CFG、timestep_fraction 0.5→10 切片、256 updates;有意偏差已在配置头注明:
   LoRA 1e-4 替 full-param 3e-5、ppo_epochs=4 替单 pass、无 diffusion-loss 正则)。config 校验套件过
   (`tests/config/test_load_all_experiments.py` 31 passed)。**跑起来仍需多卡/大卡显存**(见 §4),
   predict2_5 从未在 GRPO/SDE-logprob 评估器下实跑过——首跑先短 smoke 验 first_step parity。
2. **diffusion-loss 正则**:论文明确用它防 reward hacking;在 NFT recipe 里**没找到**对应实现。
   **行动:确认是否缺,缺则补**(否则容易 reward hack)。
   **→ 已确认缺;设计已定(2026-07-09,先设计后实施)**:
   - **确认**:在线路径无任何 diffusion-loss 项。`diffusion_sft_loss` 只被离线 DPO trainer 消费
     (`vrl/trainers/offline/dpo.py:334-340`,gated on `sft_weight>0`);schema 的
     `algorithm.sft_weight` 键在线无读者。在线的防 hack 锚只有 `kl_coef`(锚向旧策略,
     不锚向数据流形——语义弱于论文的 data-diffusion-loss)。
   - **这不是一个纯 loss 项,是一条数据通道**。论文在 **fine-tuning 数据**上算 loss →
     训练侧需要 (clean latents, prompt embeds) 对。两个结构性障碍:
     ① trainer 用 ReplayModel(只装 transformer+scheduler,**无 VAE/text-encoder**,
     minimal-replay 契约有测试锁定)——训练时现场 encode 不可行;
     ② 当前 paper-parity run 的数据集 `videophy` 是纯 prompt manifest(train.txt),
     **没有 target 视频**——即使机制齐了,这个 run 也没有数据可锚。有 target 视频的是
     droid 系 manifest(`data/artifacts.py` 的 `target_video` 字段已进 rollout metadata)。
   - **架构裁决:预编码 latents(Option A),不走 rollout-worker 现场编码(Option B)**。
     B 让生成 worker 替训练侧编码并每个 batch 重复运送同样的 target latents——把训练关切
     耦合进 rollout wire,且同一 prompt 的 latents 每 epoch 重复付费。A = 一次性离线
     encode(全 bundle,可借 `vrl/scripts/diffusion/generate.py` 的家族无关加载),产出
     (latents, embeds) shard;trainer 按 `algorithm.sft_weight>0` + `data.sft_latents`
     加载,MSE 项复用离线 DPO 已验证的构造(`_inject_noise`:flow_matching 走
     `scheduler.scale_noise`、target=noise-latents;epsilon/v-pred 走 add_noise/get_velocity
     ——与 `sampling.sde_type` 的家族映射一致)+ family `forward_step`(全家族 parity 0.0e+00)。
   - **纪律**:旋钮与数据通道**必须同一批落地**(无消费者的 sft_weight = no-op knob,
     违反 dead-field 规则);numerics 门 = 真权重下 sft_weight=0 逐位不变 + 小 lr 短跑
     diffusion loss 下降;predict2_5 的 UniPC sigma 域按 replay 侧同款 schedule 取 t
     (EDM 域翻车先例 c66bf11,不要另起换算)。
   **→ 通道已落地(2026-07-09,CPU 全链验证;GPU numerics 门未跑)**。实施相对设计有一处
   收紧:不预存 embeds——SFT 项复用**当前训练 batch 的 prompt 条件**(新模型方法
   `replay_forward_with_latents`,与 log-prob replay 走同一 state 重建/同一 timestep,
   sigma 域零第二换算路径),shard 只存 {prompt → 干净视频 latents}。五件套:
   ① 加噪构造 `diffusion_pretraining_pair`(vrl/math/diffusion/flow_matching.py,
   scheduler 拥有 forward process:flow=scale_noise+velocity 目标、ddim 系=add_noise+
   epsilon/v 目标);② `GRPOConfig.sft_weight` + schema 交叉校验(weight>0 无
   data.sft_latents 在 config load 即拒);③ shard 契约 save/load_sft_latents
   (trainers/data/artifacts.py,family 不匹配拒载);④ trainer 项
   `OnlineTrainer._sft_regularizer_loss`(每 chunk 一次额外 forward,严格/streaming
   两路都接,metrics 列 sft_loss);⑤ 编码脚本 vrl/scripts/diffusion/encode_targets.py
   (吃实验 config 保证同模型同几何,`CosmosPredict2Model.encode_video_to_latents` =
   decode 的精确逆 + diffusers pipeline conditioning 同款构造;其他 family 用前先补该方法)。
   **GPU 门(未跑)**:① encode 一个 droid manifest 并 eyeball decode 回放;② 真权重
   sft_weight=0 与关闭逐位一致;③ 小 lr 短跑 sft_loss 下降、reward 不崩。
   注意:videophy(本 parity run 的数据集)无 target 视频——paper run 想用此正则需先给
   数据集配 target(droid 系 manifest 天然满足)。
3. **reward 模型**:论文 **VideoAlign** vs config `kling_video_reward`——很可能同源(Kling 的
   VideoReward),但**没 100% 确认**。**行动:确认是不是同一个模型。**
4. **lr / 微调方式**:论文 3e-5 全参,config 是 1e-4 LoRA(注释解释:全参需多卡)。这是**有意的
   资源妥协**,不是错——有多卡时可切回 3e-5 全参。

## 6. 参考

- 论文:`docs/papers/world-models/cosmos-predict2-5-world-foundation-models.pdf` §4.2.2(p13)
- paper-shaped config:`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
- GRPO 变体:`configs/experiment/diffusion/cosmos_predict2/online_grpo_kling_video_reward.yaml`
- 单卡实跑记录:`docs/sprints/info/SPRINT_cosmos_training_field_notes.md`
- 单卡 runbook:`docs/sprints/info/SPRINT_cosmos25_kling_paper_recipe_runbook.md`
- 固定 eval 信号(判断有没有学到):`docs/sprints/done/SPRINT_cosmos_kling_fixed_eval_signal.md`
