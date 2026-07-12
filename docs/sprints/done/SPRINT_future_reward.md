# SPRINT: Future Reward —— 给 V2W 世界模型 RLHF 一个站得住的奖励

状态：**已落地（2026-06-28）= 探针 + DINOv2 + RAFT motion(零训练,都过探针);退役 pixel-L1**。目标：退役已被实测证伪的 `target_video_similarity`(64×64 像素 L1),换成一个**扛得住"糊/静止/均值"刷分**的 Future Reward,给 Cosmos Predict2 Video2World 世界模型的 GRPO 用。核心纪律:**任何 reward 进训练前,先过一个判别力探针**(就是杀掉 pixel-L1 那个)。

> **IDM action-following 已拆为独立 sprint [[SPRINT_idm_action_following_reward]]（2026-06-29）。** 它是 §2 排序里唯一评 STRONG 的主信号候选，但需自训一个小 IDM（设计 + EVA 来源调查全部搬过去了），用户先走零训练路线（dino+motion）。本 doc 只保留已落地的零训练 blend 与判别探针；要做动作空间打分看那个 sprint。
>
> **唯一剩下的真门 = Phase 4 真机 GRPO run（never run）。** 当前所有 PASS 都是离线判别探针（8 条 DROID target 上算分），证明了"分不开糊/静止的 reward 不合格"这一关；但**还没在真实 GRPO 上证明 dino+motion blend 能让 reward 单调升、三个 component mean 不发散**。这和 [[SPRINT_cosmos_predict2_2b_trustworthy_curve]] 是同一道 GPU/数据门——需要一次真机 V2W GRPO run。

## 0. 结论先行

- **现 reward 已死,退役不调参**:`target_video_similarity` = `1 − 64×64 像素 L1`(`vrl/rewards/models/target_video_similarity.py:157-165`),最优解是条件均值=糊。实测刷分(同一真 DROID target):静止=0.984、模糊均值=0.988、乱序=0.988、倒放=0.981,vs 完美=1.000,**全在 1~2% 内**,动态范围才 0.31。
- **已落地的零训练 blend（默认 recipe）**:
  ```
  target_dino_similarity: 1.0  # 感知锚 —— DINOv2 特征(含 temporal 项),抗糊,过判别探针
  motion_dynamics      : 0.2   # 质量 guard —— RAFT 光流幅度,静止视频天然=0
  ```
  → 两个组件都已落地、都过 §4 探针、零训练可直接跑(见「实施状态」)。
- **更强的主方案(IDM 当主信号) = 独立 sprint** [[SPRINT_idm_action_following_reward]]:`idm_action_following(1.0) + target_dino_similarity(0.3) + motion_dynamics(0.2)`,在低维动作空间打分,像素刷分全失效——但需自训一个小 IDM,故单列。
- **通用 KILL-RISK 门**:7 个对照样本判别探针,好预测必须比所有刷分候选高出**指标动态范围的 ≥25%**。pixel-L1 只高 ~4%,直接挂。
- **元判断(别跑偏)**:如果你要的是**纯预测准** → 该用 **SFT(diffusion loss on target clips)**,不是 RL,无 reward-hack 面。这个 sprint 只为 RL 真正该管的**不可监督属性**:未见指令下的可执行性、发散合理未来的物理性。

## 1. 动机:为什么现 reward 必须退役(实测)

`_features` 把帧 bilinear resize 到 64×64 再 flatten = 原始降采样像素;`_unit_similarity` = `1 − mean|gen−tgt|`。整个 reward = **像素 L1**。同一条真 target 喂不同"生成":

| 候选 | overall | 含义 |
|---|---|---|
| 完美匹配 | 1.000 | 上限 |
| 静止首帧(零运动) | **0.984** | reward 看不出静止 |
| 模糊时间均值 | **0.988** | ← 像素 L1 的数学最优解,RL 会塌到这 |
| 帧乱序 | **0.988** | 不衡量时序 |
| 倒放(动力学全错) | **0.981** | 不衡量因果 |
| 随机噪声 | 0.690 | 地板 |

→ 像素 L1 的最优是均值=糊,RL 最大化它 = 把世界模型训成模糊均值预测器。**退役。** 其周边管线(artifact decode、`metadata['target_video']`/`['reference_image']` plumbing `vrl/trainers/data/artifacts.py:194-201`、`score_key`、MultiReward)是好的,下面全复用。

## 2. 候选排序(给 V2W 世界模型 RL)

| 评级 | 家族 | 一句话 |
|---|---|---|
| **STRONG** | **IDM action-following**(RLIR 2509.23958 / EVA) | 在**低维动作空间**打分,不在像素;action-recoverability 与 PSNR/FVD 正交,扛糊/静止/乱序;且是唯一绑住真实机器人任务("这段未来听不听**这条指令**")的信号 |
| **PROMISING** | **学习特征感知**(DINOv2/CLIP 帧 cosine + V-JEPA2 clip) | 语义特征独立于糊 → 像素均值不再是最优;便宜、可 CPU;**但 per-frame cosine 仍 order-blind**(需补时序项),且**仍在模仿一条录像** |
| **PROMISING** | **物理合理性/时序一致**(VideoPhy-2、VBench Dynamic-Degree、RAFT 光流) | 光流 Dynamic Degree **天然把静止/糊打到地板**(flow≈0);但物理 VLM 单用会偷懒("干净静止图也合理"),且不绑指令 |
| **WEAK(当主信号)** | **VLM-as-judge**(VideoScore2 Qwen2.5-VL) | 语义上能杀糊/静止、soft EV 解码给平滑梯度;但**自带可 hack 轴**(讨好 judge 的纹理/长度/饱和度偏置)、最慢(7B CoT ~秒级)。**当 guard/诊断可以,当唯一主信号不行** |

## 3. 落地方案

- **零训练默认(本 sprint 落地) = `target_dino_similarity(1.0) + motion_dynamics(0.2)`**:DINOv2 感知锚(含 temporal 项,抗糊)当主 + RAFT 光流当静止 guard。两个都过 §4 探针、CPU 可跑、无新训练。这是两个 droid recipe 的默认。
- **IDM-主变体(更强,独立 sprint)** = [[SPRINT_idm_action_following_reward]]:把主信号换成在动作空间打分的 IDM(DINOv2 降到 0.3 当锚、motion 0.2 当 guard)。需自训小 IDM,故拆出去。
- **component 监控**:三个 component mean **每 epoch 都 log**(MultiReward 的 `last_components` 已自动 track),早期抓 hack。
- **VLM judge 不当主信号**:要么不进训练 reward,要么低权重当 held-out 诊断。

## 4. 通用 KILL-RISK 判别探针(每个 reward 必过 —— 这是核心门)

**任何候选 reward 进训练前,先过这个;就是杀掉 pixel-L1 的同一套 harness。** 取一条真 DROID target(及其 32 步动作 `a*` 给 IDM),造 7 个候选:① exact ② perceptual-blur(模糊均值)③ static-frozen ④ temporal-mean ⑤ frame-shuffle ⑥ wrong-action(别的 episode 的片段,配本 episode 的 prompt/`a*`)⑦ random;时序候选另加 reverse。

**PASS 要求**:`exact` 最高,**且 `exact − max(blur, static, temporal-mean) ≥ 指标动态范围的 0.25`**(pixel-L1 失败:gap≈0.012/0.31≈4%)。家族附加门:
- **IDM(主，独立 sprint)**:family 门（exact≥0.7、static/blur/shuffle/reverse/wrong-action ≤0.4、exact>wrong-action 2σ）见 [[SPRINT_idm_action_following_reward]] §3。
- **DINOv2/CLIP 帧 cosine**:blur/temporal-mean 要比 exact 低 **>0.15 cosine**。
- **时序/motion 项**:shuffle 和 reverse 都要远低于 exact;RAFT motion 的 static 必须塌到地板(这是它的全部意义)。

CPU 跑 CLIP/DINOv2/LPIPS/RAFT(7 clip × 33 帧,torch.hub,无 GPU);reward-pool GPU 跑 IDM/VLM(~46 clip × 32 对 = 秒级)。**一个分不开 exact 和 blur+static+shuffle 的 reward,不是合格的 Future Reward。**

## 5. Repo 接线(分阶段,真实路径)

已核实:torchvision 是 base dep(`pyproject.toml:46`,`raft_small` 在 `torchvision.models.optical_flow`),transformers(`:42`),kling 的 disk-artifact pool 模板,metadata plumbing。

**(A) 感知锚 `target_dino_similarity`(便宜、本地、先做)**
- 抄 `vrl/rewards/functions/target_video_similarity.py` + `models/target_video_similarity.py` 几乎原样(`default_execution="inline"`、`InProcessRewardRuntime`、解码 gen 帧 + 读 `metadata['target_video']`)。
- **只换坏核心**:把 `_features()`/`_unit_similarity()`(`models/target_video_similarity.py:157-165`)换成"每帧过冻结 DINOv2(`torch.hub('facebookresearch/dinov2','dinov2_vits14')`)→ 帧 cosine 均值",保留 sequence/final 加权结构。
- `score_key=target_dino_similarity`;无新依赖;CPU 可。注册 `registry.py`:`"target_dino_similarity": TargetDinoSimilarityReward`。

**(B) 主信号 `idm_action_following`（重、GPU-pool）→ 已拆到 [[SPRINT_idm_action_following_reward]]**
- reward 模板、IDM 网络设计、`target_actions` 数据接线（`vrl/scripts/data/video_world.py:176-181,427`，commit d22d7d5d）、自训步骤、KILL-RISK 探针门、EVA 来源调查全部搬到那个 sprint。本 doc 不再维护 IDM 细节。

**(C) 质量 guard `motion_dynamics`(RAFT、本地)**:抄 `TargetVideoSimilarityReward` 本地模板,只依赖 `torchvision.models.optical_flow.raft_small`,score = VBench Dynamic-Degree(top-5% 光流幅度均值)。

## 6. 诚实边界

- **感知不解决"单 target 模仿"**:DINOv2/CLIP/V-JEPA 仍是对**一条录像**打分;世界模型有很多合理未来,模仿一条会惩罚合理的发散 rollout。这正是 **IDM 当主、感知降到 0.3 当锚** 的原因。
- **IDM 本身是新的、也可 hack**:要训/找一个 DROID 帧对→动作 ckpt(sprint 主成本)。欠训的 IDM 读全局外观 = 和 pixel-L1 一样可 hack → **必须先过 §4 探针**。EVA 报告即使好 IDM,长 GRPO 后静止塌缩也会复发 → 保留 early-stop + held-out 探针 + RAFT motion 地板。
- **VLM judge 可被 reward-hack**(纹理/长度偏置,人评质量降而 RM 分不降)→ 只当冻结、低权重、多问题聚合 + policy KL-to-base 的诊断,别当训练 reward,且最慢。
- **元点**:要"尽量贴近录像未来" → **SFT(diffusion loss)碾压任何 RL reward**,直接可监督、无 hack 面、不用探针。RL 只为不可监督属性挣价值。**别让这个 sprint 漂成"花哨的模仿 loss"。**

## 7. 分阶段(每阶段一个 KILL-RISK 门)

- **Phase 0 —— 判别探针 harness ✅**:`vrl/scripts/eval/future_reward_discrimination_probe.py`(已用现 reward 复现 0.98 失败,PASS bar 已定)。任何新 reward 先过它。
- **Phase 1 —— DINOv2 感知锚 ✅**:换核心 → 过探针(gap_ratio 0.298 PASS,需 temporal 项)。
- **Phase 1.5 —— RAFT motion guard ✅**:static 塌到地板(gap_ratio 0.969 PASS)。
- **Phase 2/3 —— IDM action-following → 拆到 [[SPRINT_idm_action_following_reward]]**(动作标签进 manifest + 自训 IDM + reward + 过探针)。
- **Phase 4 —— 真机 GRPO 验证（唯一剩下的门，never run）**:把已落地的 `dino(1.0)+motion(0.2)` 零训练 blend 接 [[SPRINT_cosmos_predict2_2b_trustworthy_curve]] 的真机 V2W GRPO run,看 eval reward 升 >2σ + 两个 component mean 不发散(防 hack)。**离线探针 PASS ≠ 训练能学,这一步才闭环。**

## 验收

- [x] dino / motion **过 §4 判别探针**;数字落本文件(见「实施状态」)。
- [x] 可直接跑的零训练 recipe = dino(1.0)+motion(0.2),两组件都过探针(默认 recipe 已是此值)。
- [x] `target_video_similarity` **彻底删除**(2026-06-28):model/function/config/registry 项 + 旧 `target_video_similarity_probe.py` 自比对脚本全删;production validation 的 `require_target_video` 改 key 到 `target_dino_similarity`(同样读 `metadata['target_video']`,语义不丢)。下方表里的 pixel-L1 数字保留为"为什么删它"的存档。
- [→] action-following(IDM)**已拆为独立 sprint** [[SPRINT_idm_action_following_reward]](设计 + 自训步骤 + EVA 调查搬过去)。manifest 的 `target_actions` plumbing(commit d22d7d5d)留着但当前无 reward 读它。
- [ ] **Phase 4 真机 GRPO 验证（唯一剩下的门，never run）**:dino(1.0)+motion(0.2) 接 [[SPRINT_cosmos_predict2_2b_trustworthy_curve]] 的真机 V2W run,eval reward 升 >2σ + component mean 不发散。当前全部 PASS 都是离线探针,不是训练曲线。

## 实施状态（2026-06-28）

**探针实测数字（8 条真 DROID target,`droid_targets_eval.jsonl`,5090）** —— PASS bar = `exact` 最高且 `gap = exact − max(blur,static,temporal-mean) ≥ 0.25 × 动态范围`(动态范围 = exact − 全候选最低）:

| reward | exact | blur | static | temporal-mean | shuffle | reverse | wrong-clip | random | gap_ratio | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| `target_video_similarity`(pixel-L1,**已删除**,数字存档) | 0.994 | 0.944 | 0.983 | 0.985 | 0.982 | 0.979 | 0.776 | 0.756 | **0.039** | **FAIL**(复现 §1) |
| `target_dino_similarity`(0.4/0.2/0.4) | 0.826 | 0.210 | 0.577 | 0.573 | 0.586 | 0.565 | 0.209 | −0.008 | **0.298** | **PASS** |
| `motion_dynamics`(scale 50) | 0.355 | 0.036 | 0.027 | 0.026 | 0.527 | 0.367 | 0.355 | 1.000 | 0.969 | **PASS**(static 塌到地板) |
| `idm_action_following` | — | — | — | — | — | — | — | — | — | 拆到 [[SPRINT_idm_action_following_reward]] |

- **pixel-L1 复现死因**:exact 比刷分候选只高 ~1%,gap_ratio 4%,和 §1 的 0.012/0.31≈4% 一致 → 退役无疑。
- **DINOv2 的关键发现**:per-frame cosine 单用**过不了**(temporal-mean/static 同场景仍 ~0.76,gap_ratio 仅 0.15)。把 order-sensitive 的 temporal 项(相邻帧 embedding delta 的 cosine)权重提到和 appearance 同级(0.4/0.2/0.4)才过 —— 这正印证 §2 "per-frame cosine order-blind,需补时序项"。**默认权重已设成过探针的值;调低 temporal_weight 会重新挂。**
- **motion**:static/blur/temporal-mean 全塌到 ~0.03 地板,exact 0.36;random(噪声)和 shuffle 反而最高(光流幅度 order-agnostic)—— 这是 guard 的预期行为,不是 bug。

**落地的资产(留下来的)**:
- 探针(keystone):`vrl/scripts/eval/future_reward_discrimination_probe.py` —— reward-agnostic,8 候选,dino/motion 两条支路 + 每家族 PASS 规则。任何新 reward 先过它。(pixel-L1 支路随 reward 一起删了;它的失败数字已存档在上表。)
- DINOv2 锚:`vrl/rewards/{models,functions}/target_dino_similarity.py` + `configs/reward/target_dino_similarity.yaml`。
- RAFT motion guard:`vrl/rewards/{models,functions}/motion_dynamics.py` + `configs/reward/motion_dynamics.yaml`。
- 共享:帧解码 helper 进 `vrl/utils/media.py`(`read_video_frames`/`sample_frames`/`align_frame_counts`/`frames_thwc_to_float`)+ `decode_artifact_frames` 进 `vrl/rewards/base.py`(没动被退役的 pixel model)。
- registry 注册 dino + motion(可复用积木);两个单 reward 组 config。**probe + 这两个组 = 真资产,任何实验按需 compose。**
- **两个 droid recipe 默认 = dino(1.0)+ motion(0.2),零训练、可直接跑**(原本指向死掉的 pixel-L1,必须重指一个能用的;这是默认值不是承诺,每个实验可 override)。kling 等是 opt-in `/reward/*` 积木。
- 测试:`tests/rewards/functions/test_future_reward.py`(探针候选构造 + 各家族判定逻辑的 CPU 单测)。

**删掉的(给没在走的 IDM 路建的脚手架,从没端到端跑过)**:`vrl/rewards/{models,functions}/idm_action_following.py`、bridge scorer `idm_action_score.py`、训练脚本 `train_droid_idm.py`、`configs/reward/idm_action_following.yaml`。用户选了零训练 dino+motion;IDM 设计与重建步骤已搬到 [[SPRINT_idm_action_following_reward]],以后真要再建照那个 sprint。

**怎么跑探针**:`python -m vrl.scripts.eval.future_reward_discrimination_probe --reward <name> --manifest data/external/video_world/manifests/droid_targets_eval.jsonl --out outputs/probe_<name>.jsonl --device cuda`(`<name>` = target_dino_similarity / motion_dynamics)。

**IDM action-following（重建步骤 + 5 源来源调查 + EVA 架构结论）→ 全部搬到 [[SPRINT_idm_action_following_reward]]**。一句话:查证后没有现成 IDM 能直接插(契约对不上 DROID 单臂/只吃帧/7 维),只能自训一个 vision-only 小 IDM,设计与数据接线见那个 sprint。

**已知环境/边界**:
- `.venv`(uv 管理)原本缺 torchvision,本次已 `uv pip install torchvision==0.26.0+cu130`(RAFT 需要;DINOv2 走 torch.hub,无需 transformers)。
- 不相关的既有红测试(非本 sprint 引入):OCR 两条(`.venv` 缺 `Levenshtein`)。（注:`test_reward_models_live_under_models` 和 `test_shared_ray_substrate_stays_domain_neutral` 两条架构红已于 2026-06-29 修复,见 `done/SPRINT_weak_test_cleanup.md`。）

**参考**
- RLIR(inverse rewards,world-model post-train):arXiv:2509.23958 · EVA(IDM reward → executable actions):arXiv:2603.17808
- DINOv2:2304.07193 · V-JEPA2:2506.09985 · VideoPhy-2:2503.06800 · VBench:2311.17982 · VideoScore2:2509.22799
- Repo:`vrl/rewards/models/target_video_similarity.py:157-165`(死核心)、`vrl/rewards/functions/{registry,kling_video_reward,videoscore2,phymotion}.py`、`vrl/rewards/models/{videoscore2,phymotion}.py`、`vrl/trainers/data/artifacts.py:194-201`、`vrl/scripts/data/video_world.py:176-181,427`、`pyproject.toml:42,46`
