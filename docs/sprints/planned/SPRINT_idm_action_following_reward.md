# SPRINT: IDM action-following reward —— 在低维动作空间给 V2W 世界模型打分

状态：**planned / 自训-gated（2026-06-29，从 [[SPRINT_future_reward]] 拆出）**。范围：给 Cosmos Predict2 Video2World 的 GRPO 加一个**主信号** reward —— 用一个 inverse-dynamics model（IDM）从生成的相邻帧对回归出动作，reward = 与 DROID **指令动作**的匹配度。这是 [[SPRINT_future_reward]] §2 排序里唯一评 **STRONG** 的候选，但用户先走了零训练路线（dino+motion），故 IDM 的 reward/bridge/训练脚本曾建好后删除；本 sprint 把它作为独立目标重新立项，**设计与来源调查全部保留在此**，照此重建即可。硬前置 = **自训一个小 IDM checkpoint**（没有现成的能直接插，见 §5）。

## 0. 结论先行

- **为什么要它**：world model RL 真正该管的是**不可监督属性** —— "这段未来听不听**这条指令**"。像素/感知 reward 都在模仿一条录像，会惩罚合理的发散未来；IDM 在**低维动作空间**打分，每一种像素刷分（静止/糊/乱序/倒放/别的片段）在动作空间都失效。RLIR(2509.23958) 明确报告 pixel-L1+LPIPS 会 reward-hack（视频变暗），IDM reward 亮度/糊不变，action-following 提升 5-10%。
- **目标 blend（IDM 当主信号）**：
  ```
  idm_action_following : 1.0   # 主信号 —— 在动作空间打分，像素刷分全失效
  target_dino_similarity: 0.3  # 感知锚 —— DINOv2 特征，把生成帧拉回真实图像流形（抗糊，防 off-manifold 骗 IDM）
  motion_dynamics      : 0.2   # 质量 guard —— RAFT 光流幅度，静止视频天然=0（EVA 警告长 GRPO 后静止塌缩会复发）
  ```
  注：dino+motion 这两个积木已由 [[SPRINT_future_reward]] 落地并过探针，本 sprint 只新增 idm 一支 + 把权重切到 IDM-主。
- **硬纪律（KILL-RISK 门）**：IDM **进训练前必须过 §3 判别探针**。欠训的 IDM 读全局外观 = 和 pixel-L1 一样可 hack。过不了别进训练。
- **元判断**：如果你要的是**纯预测准** → 该用 SFT(diffusion loss on target clips)，不是 RL，无 reward-hack 面。IDM reward 只为 RL 真正该管的不可监督属性（未见指令下的可执行性）挣价值。**别让它漂成"花哨的模仿 loss"。**

## 1. 为什么是动作空间（对照像素/感知的失效）

同一条真 DROID target 喂不同"生成"，像素 L1（已退役的 `target_video_similarity`）刷分全在 1~2% 内（静止 0.984 / 糊均值 0.988 / 乱序 0.988 / 倒放 0.981 vs 完美 1.000）。动作空间为什么扛得住：

| 刷分手段 | 像素/感知表现 | IDM 动作空间表现 |
|---|---|---|
| 静止首帧 | reward 看不出 | IDM 预测≈零运动 ≠ 非零指令动作 → 低分 |
| 糊/时间均值 | 数学最优解（RL 会塌到这） | 抹掉的末端执行器没有连贯 Δpose → 低分 |
| 帧乱序/倒放 | 不衡量时序 | 方向错 → 低分 |
| 别的片段（wrong-action） | 同分布看不出 | 匹配的是**错的指令** → 低分（决定性测试） |

**action-recoverability 与 PSNR/FVD 正交**，这是它当主信号的全部理由。

## 2. 设计（采纳 EVA 验证过的架构）

- **网络**：vision-only **卷积 backbone + spatial-softmax + MLP（~0.56M 参数）**。EVA(2603.17808) 正是证明了"vision-only IDM 当 reward"这条路；spatial-softmax 专抽末端执行器的 2D 位置，比 global-avg-pool 更适合恢复动作。可选 Octo-small(27M, MIT) encoder 蒸馏，但 JAX→torch 移植成本不优先。
- **输入/输出**：相邻帧对 `(o_i, o_{i+1})` → 7-DOF 动作（cartesian 6 + gripper 1）。`action_dim` 动态读取，`gripper_index = action_dim-1` 对 7/8 维都成立。
- **score 定义**：`score = (1/31)·Σ_i match(IDM(o_i, o_{i+1}), a*_i)`，`a*` 为 DROID 该 episode 的 32 步指令动作（从 manifest 读，不读像素）。
- **score_keys**：`{action_match, ee_pose_match, gripper_match, overall}`。
- **DINOv2 锚（0.3）的作用**：IDM 在真 DROID 上训，容易被没见过的 off-manifold 输入骗；感知锚把生成帧拉回真实图像流形，特征抗糊，不重新引入 L1 漏洞。
- **RAFT motion（0.2）的作用**：静止视频光流≈0，对塌缩静止的硬地板。

**动作约定真相（不影响自训）**：lerobot 把语义抹成 `motor_0..6`，权威转换脚本（`port_droid.py`/openpi）实际产 8 维关节，droid_100 是 7 维，确切语义无权威源能 pin 死。但这对自训**不重要** —— IDM 在同一批 manifest 标签上监督，学"帧对→该标签向量"即可。

## 3. KILL-RISK 判别探针（进训练前必过 —— 核心门）

[[SPRINT_future_reward]] 的自动探针脚本已删除（决定：判别质量由**人工可视化 rollouts** 判断，不再走自动 PASS/FAIL 门）。候选 battery 生成逻辑保留在 `vrl/scripts/eval/unified_reward_robotics_discrimination_probe.py` 的 `build_discrimination_candidates`（8 候选：exact / blur / static / temporal-mean / shuffle / reverse / wrong-action / random）——用它生成候选、给 IDM 打分后由人工比对排序，参考标准：

- **`exact ≥ 0.7`**，且 **static / blur / shuffle / reverse / wrong-action 全 ≤ 0.4**（差 >0.3，>5× 噪声地板）；
- **`exact` 比 `wrong-action` 高 >2σ**（wrong-action 是决定性测试：运动一样真，只是指令不同）；
- **若 static 或 blur 落在 exact 0.1 内 → IDM 在读全局外观，和 pixel-L1 一样可 hack，FAIL，回炉**。

通用门（所有 reward 共用）：`exact` 最高且 `gap = exact − max(blur, static, temporal-mean) ≥ 0.25 × 动态范围`（pixel-L1 失败：gap_ratio≈4%）。IDM/VLM 走 reward-pool GPU（~46 clip × 32 对 = 秒级）。

## 4. Repo 接线（真实路径）

**(A) 数据：动作标签进 manifest（关键前置）**
- 动作就在生成器已读的 `lerobot/droid_100` parquet 同一行。`vrl/scripts/data/video_world.py:_iter_lerobot_v21_target_clips`(~:427) 现在只取 `['episode_index','frame_index','task_index','index']`；加 `action.cartesian_position[6]` + `action.gripper_position[1]`（15Hz 同行），按 `frame_index<33` 切片，在 row builder(~:176-181) 和 `target_video` 一起 emit `metadata['target_actions']=[[7dof]*32]`。
- 动作路径已在 **commit d22d7d5d** 接好（`target_actions` plumbing 留着），需 `pyarrow`。重建命令：`vrl.scripts.data.setup video-world-targets ...`。

**(B) reward：`idm_action_following`（重、GPU-pool、sprint 主体）**
- 抄 disk-artifact pool 模板 `vrl/rewards/functions/kling_video_reward.py` → `functions/action_following.py`：`default_execution="pool"`，`_init_disk_artifact_reward(model_factory="vrl.rewards.models.action_following:ActionFollowingIDMModel", default_artifact_format="mp4", default_score_key="action_match")`。
- Model：`__call__(artifact, request)->dict`；独立 IDM ckpt 走 `vrl/rewards/models/phymotion.py` 的**外部进程 bridge** 范式（读 mp4+动作，出 JSON）。**别复用 phymotion 的 scorer** —— 它恢复 SMPL 人体网格，对机械臂没意义。
- reward 从 `artifact.metadata['target_actions']` 读指令动作，不读像素。
- 注册 `registry.py`：`"idm_action_following": ActionFollowingReward`；blend 走 `MultiReward.from_dict`（§0 权重）。

**(C) 共享积木（已落地，直接复用）**：帧解码 `vrl/utils/media.py`（`read_video_frames`/`sample_frames`/`align_frame_counts`/`frames_thwc_to_float`）、`decode_artifact_frames`（`vrl/rewards/base.py`）、dino + motion 两个组件 config。

## 5. 来源调查：为什么不从零也下不到能直接插的（5 源对抗式核实，2026-06-28）

**Route (b) 直接接现成 IDM = 死** —— 没有一个对得上"DROID 单臂 / 只吃帧 / 我们的 7 维标签"契约的：

- **EVA**(arXiv:2603.17808，HF `RobbinWang123/EVA` `IDM_singleview.pt` 348MB Apache-2.0，vision-only)：输出是 **14 维双臂 RoboTwin 关节空间**，不是 DROID。
- **Seer/PIDM**(2412.15109，DROID ckpt 在 Google Drive)：**必须喂本体状态 + 双相机 + 语言**，我们只有生成帧给不了。
- **OpenVLA/π0**：前向语言策略、重、要 state。

**结论**："下一个直接用"查证后走不通（契约对不上）；要 action-following 只能**自训一个小 IDM**，这是 contract-correct 的最省路，且架构有文献背书（EVA 的 vision-only conv + spatial-softmax + MLP）。

## 6. 分阶段（每阶段一个 KILL-RISK 门）

- **Phase 1 —— 动作标签进 manifest**：改 `video_world.py` emit `target_actions`；重建 droid manifest；验证一条记录带 32×7 动作。**（纯数据，可立即做）**
- **Phase 2 —— 训/取 IDM ckpt**：在 manifest 标签上监督训练 vision-only IDM（DROID 帧对→7-DOF，1 GPU 几小时）。重建训练脚本 `train_droid_idm.py`。
- **Phase 3 —— reward + 过探针**：`action_following.py` reward → **过 §3 探针**（尤其 static/blur ≤0.4、exact>wrong-action 2σ）。过不了 = IDM 没学好，回炉。
- **Phase 4 —— blend 进 recipe**：把权重切到 idm(1.0)+dino(0.3)+motion(0.2)，接 [[SPRINT_cosmos_predict2_2b_trustworthy_curve]] 的真机 GRPO run；看 reward 升 + 三个 component mean 不发散（防 hack）+ EVA 警告的长 GRPO 静止塌缩用 RAFT 地板 + early-stop + held-out 探针兜住。

## 7. 诚实边界

- **IDM 本身是新的、也可 hack**：欠训的 IDM 读全局外观 = 和 pixel-L1 一样 → **必须先过 §3 探针**。EVA 报告即使好 IDM，长 GRPO 后静止塌缩也会复发 → 保留 early-stop + held-out 探针 + RAFT motion 地板。
- **感知锚不是主信号**：DINOv2 仍是对一条录像打分，所以降到 0.3 当锚，IDM 当主。
- **元点**：要"尽量贴近录像未来" → SFT(diffusion loss) 碾压任何 RL reward（可监督、无 hack 面）。RL 只为不可监督属性挣价值。

## 8. 关键文件

- 数据：`vrl/scripts/data/video_world.py:176-181,427`（emit `target_actions`）、commit d22d7d5d（plumbing）
- reward 模板：`vrl/rewards/functions/kling_video_reward.py`（disk-artifact pool）、`vrl/rewards/models/phymotion.py`（外部进程 bridge 范式）、`vrl/rewards/functions/registry.py`
- 判别候选 battery：`vrl/scripts/eval/unified_reward_robotics_discrimination_probe.py` 的 `build_discrimination_candidates`（判别结论由人工可视化判断）
- 共享积木：`vrl/utils/media.py`、`vrl/rewards/base.py:decode_artifact_frames`、dino/motion 的 `{models,functions}` + config
- 真机曲线接入点：[[SPRINT_cosmos_predict2_2b_trustworthy_curve]]

## 9. 参考

- RLIR（inverse rewards, world-model post-train）：arXiv:2509.23958
- EVA（IDM reward → executable actions, vision-only conv + spatial-softmax）：arXiv:2603.17808
- 退役 pixel-L1 的实测与判别探针：[[SPRINT_future_reward]]
