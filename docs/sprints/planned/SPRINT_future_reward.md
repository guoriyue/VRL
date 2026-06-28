# SPRINT: Future Reward —— 给 V2W 世界模型 RLHF 一个站得住的奖励

状态：**planned（2026-06-27）**。目标：退役已被实测证伪的 `target_video_similarity`(64×64 像素 L1),换成一个**扛得住"糊/静止/均值"刷分**的 Future Reward,给 Cosmos Predict2 Video2World 世界模型的 GRPO 用。核心纪律:**任何 reward 进训练前,先过一个判别力探针**(就是杀掉 pixel-L1 那个)。

## 0. 结论先行

- **现 reward 已死,退役不调参**:`target_video_similarity` = `1 − 64×64 像素 L1`(`vrl/rewards/models/target_video_similarity.py:157-165`),最优解是条件均值=糊。实测刷分(同一真 DROID target):静止=0.984、模糊均值=0.988、乱序=0.988、倒放=0.981,vs 完美=1.000,**全在 1~2% 内**,动态范围才 0.31。
- **推荐主方案(MultiReward blend)**:
  ```
  idm_action_following : 1.0   # 主信号 —— 在低维动作空间打分,像素刷分全失效
  target_dino_similarity: 0.3  # 感知锚 —— DINOv2 特征,保持在真实图像流形上,抗糊
  motion_dynamics      : 0.2   # 质量 guard —— RAFT 光流幅度,静止视频天然=0
  ```
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

## 3. 推荐主方案

**主信号 = IDM action-following**(EVA/RLIR 配方):用一个 **inverse-dynamics model** 从生成的相邻帧对回归出动作,reward = 和 DROID **指令动作**的匹配度。每一种像素刷分都在动作空间失效:静止→IDM 预测≈零运动 ≠ 非零指令;糊→抹掉的末端执行器没有连贯 Δpose;乱序/倒放→方向错;别的片段→匹配的是**错的指令**。RLIR 明确报告 pixel-L1+LPIPS 会 reward-hack(视频变暗),而 IDM reward 亮度/糊不变,action-following 提升 5-10%。

- **DINOv2 项是感知锚不是驱动**(0.3):把生成帧拉回真实图像流形(IDM 在真 DROID 上训,容易被没见过的 off-manifold 输入骗),特征抗糊,不重新引入 L1 漏洞。
- **RAFT motion 项是纯质量 guard**(0.2):静止视频光流≈0,**对塌缩静止的硬地板**(EVA 警告长 GRPO 后静止塌缩会复发)。
- 三个 component mean **每 epoch 都 log**(MultiReward 的 `last_components` 已自动track),早期抓 hack。
- **VLM judge 不当主信号**:要么不进训练 reward,要么低权重当 held-out 诊断。

## 4. 通用 KILL-RISK 判别探针(每个 reward 必过 —— 这是核心门)

**任何候选 reward 进训练前,先过这个;就是杀掉 pixel-L1 的同一套 harness。** 取一条真 DROID target(及其 32 步动作 `a*` 给 IDM),造 7 个候选:① exact ② perceptual-blur(模糊均值)③ static-frozen ④ temporal-mean ⑤ frame-shuffle ⑥ wrong-action(别的 episode 的片段,配本 episode 的 prompt/`a*`)⑦ random;时序候选另加 reverse。

**PASS 要求**:`exact` 最高,**且 `exact − max(blur, static, temporal-mean) ≥ 指标动态范围的 0.25`**(pixel-L1 失败:gap≈0.012/0.31≈4%)。家族附加门:
- **IDM(主)**:score = `(1/31)·Σ match(IDM(o_i,o_{i+1}), a*_i)`。要 **exact ≥ 0.7 且 static/blur/shuffle/reverse/wrong-action 全 ≤ 0.4**(差 >0.3,>5× 噪声地板),**且 exact 比 wrong-action 高 >2σ**(wrong-action 是决定性测试:运动一样真,只是指令不同)。**若 static 或 blur 落在 exact 0.1 内 → IDM 在读全局外观,和 pixel-L1 一样可 hack,FAIL。**
- **DINOv2/CLIP 帧 cosine**:blur/temporal-mean 要比 exact 低 **>0.15 cosine**。
- **时序/motion 项**:shuffle 和 reverse 都要远低于 exact;RAFT motion 的 static 必须塌到地板(这是它的全部意义)。

CPU 跑 CLIP/DINOv2/LPIPS/RAFT(7 clip × 33 帧,torch.hub,无 GPU);reward-pool GPU 跑 IDM/VLM(~46 clip × 32 对 = 秒级)。**一个分不开 exact 和 blur+static+shuffle 的 reward,不是合格的 Future Reward。**

## 5. Repo 接线(分阶段,真实路径)

已核实:torchvision 是 base dep(`pyproject.toml:46`,`raft_small` 在 `torchvision.models.optical_flow`),transformers(`:42`),kling 的 disk-artifact pool 模板,metadata plumbing。

**(A) 感知锚 `target_dino_similarity`(便宜、本地、先做)**
- 抄 `vrl/rewards/functions/target_video_similarity.py` + `models/target_video_similarity.py` 几乎原样(`default_execution="inline"`、`LocalRewardRuntime`、解码 gen 帧 + 读 `metadata['target_video']`)。
- **只换坏核心**:把 `_features()`/`_unit_similarity()`(`models/target_video_similarity.py:157-165`)换成"每帧过冻结 DINOv2(`torch.hub('facebookresearch/dinov2','dinov2_vits14')`)→ 帧 cosine 均值",保留 sequence/final 加权结构。
- `score_key=target_dino_similarity`;无新依赖;CPU 可。注册 `registry.py`:`"target_dino_similarity": TargetDinoSimilarityReward`。

**(B) 主信号 `idm_action_following`(重、GPU-pool、sprint 主体)**
- 抄 disk-artifact pool 模板 `functions/kling_video_reward.py` → `functions/action_following.py`:`default_execution="pool"`,`_init_disk_artifact_reward(model_factory="vrl.rewards.models.action_following:ActionFollowingIDMModel", default_artifact_format="mp4", default_score_key="action_match")`。
- Model:`__call__(artifact, request)->dict`;独立 IDM ckpt 走 `models/phymotion.py` 的**外部进程 bridge** 范式(读 mp4+动作,出 JSON);**别复用 phymotion 的 scorer**(它恢复 SMPL 人体网格,对机械臂没意义)。
- score_keys:`{action_match, ee_pose_match, gripper_match, overall}`。
- **NEW 资产**:(1) **IDM checkpoint**——DROID 帧对→7-DOF 动作(1 GPU 几小时,或从现成 DROID policy 蒸馏);(2) **动作标签进 manifest**。
- **动作标签数据接线(关键)**:动作就在生成器已读的 `lerobot/droid_100` parquet 同一行。`vrl/scripts/data/video_world.py:_iter_lerobot_v21_target_clips`(~:427)现在只取 `['episode_index','frame_index','task_index','index']`;加 `action.cartesian_position[6]` + `action.gripper_position[1]`(15Hz 同行),按 `frame_index<33` 切片,在 row builder(~`:176-181`)和 `target_video` 一起 emit `metadata['target_actions']=[[7dof]*32]`。reward 从 `artifact.metadata` 读指令动作,不读像素。
- 注册 `"idm_action_following": ActionFollowingReward`;blend 走 `MultiReward.from_dict`(§2 权重)。

**(C) 质量 guard `motion_dynamics`(RAFT、本地)**:抄 `TargetVideoSimilarityReward` 本地模板,只依赖 `torchvision.models.optical_flow.raft_small`,score = VBench Dynamic-Degree(top-5% 光流幅度均值)。

## 6. 诚实边界

- **感知不解决"单 target 模仿"**:DINOv2/CLIP/V-JEPA 仍是对**一条录像**打分;世界模型有很多合理未来,模仿一条会惩罚合理的发散 rollout。这正是 **IDM 当主、感知降到 0.3 当锚** 的原因。
- **IDM 本身是新的、也可 hack**:要训/找一个 DROID 帧对→动作 ckpt(sprint 主成本)。欠训的 IDM 读全局外观 = 和 pixel-L1 一样可 hack → **必须先过 §4 探针**。EVA 报告即使好 IDM,长 GRPO 后静止塌缩也会复发 → 保留 early-stop + held-out 探针 + RAFT motion 地板。
- **VLM judge 可被 reward-hack**(纹理/长度偏置,人评质量降而 RM 分不降)→ 只当冻结、低权重、多问题聚合 + policy KL-to-base 的诊断,别当训练 reward,且最慢。
- **元点**:要"尽量贴近录像未来" → **SFT(diffusion loss)碾压任何 RL reward**,直接可监督、无 hack 面、不用探针。RL 只为不可监督属性挣价值。**别让这个 sprint 漂成"花哨的模仿 loss"。**

## 7. 分阶段(每阶段一个 KILL-RISK 门)

- **Phase 0 —— 判别探针 harness**:把 §4 固化成 `vrl/scripts/eval/future_reward_discrimination_probe.py`(已用现 reward 复现 0.98 失败,PASS bar 已定)。任何新 reward 先过它。
- **Phase 1 —— DINOv2 感知锚**(便宜先做):换核心 → 过探针(blur/temporal-mean 要掉 >0.15 cosine)。**过不了就停。**
- **Phase 2 —— 动作标签进 manifest**:改 `video_world.py` emit `target_actions`;重建 droid manifest;验证一条记录带 32×7 动作。
- **Phase 3 —— IDM reward**:训/取 IDM ckpt → `action_following.py` reward → **过 §4 探针(尤其 static/blur ≤0.4、exact>wrong-action 2σ)**。过不了 = IDM 没学好,回炉。
- **Phase 4 —— blend + motion guard**:MultiReward 三项,真实 GRPO 上看 reward 升 + 三个 component mean 不发散(防 hack)。

## 验收

- [ ] 每个 reward(dino / idm / motion)**过 §4 判别探针**;数字落本文件。
- [ ] `idm_action_following` 在探针上 exact 比 wrong-action 高 >2σ,static/blur ≤0.4。
- [ ] manifest 带 `target_actions`;reward 从 metadata 读动作不读像素。
- [ ] blend 在 1-GPU GRPO smoke 上 reward 上升且 component means 稳定(无单项飙升=无 hack)。
- [ ] `target_video_similarity` 退役(从 recipe 移除,或降为 0 权重 + 注释"dead: pixel-L1, see SPRINT_future_reward §1")。

**参考**
- RLIR(inverse rewards,world-model post-train):arXiv:2509.23958 · EVA(IDM reward → executable actions):arXiv:2603.17808
- DINOv2:2304.07193 · V-JEPA2:2506.09985 · VideoPhy-2:2503.06800 · VBench:2311.17982 · VideoScore2:2509.22799
- Repo:`vrl/rewards/models/target_video_similarity.py:157-165`(死核心)、`vrl/rewards/functions/{registry,kling_video_reward,videoscore2,phymotion}.py`、`vrl/rewards/models/{videoscore2,phymotion}.py`、`vrl/trainers/data/artifacts.py:194-201`、`vrl/scripts/data/video_world.py:176-181,427`、`pyproject.toml:42,46`
