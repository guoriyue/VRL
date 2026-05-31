# Sprint: Anime Body + Hand Geometry Reward（RTMW + HaMeR）

状态: proposed（替换已删 DWPose anatomy 系统，用更强的几何 reward 重建 body + hand 信号）

## 0. TL;DR / 一句话

用 **RTMW-x**（whole-body 2D 关键点，Apache-2.0，25ms/人）做全身姿态覆盖率 + 骨长 + 关节角异常，用 **HaMeR**（手部 MANO mesh，MIT，单手 ~40ms）做手指拓扑 + 3D 重投影误差，两路输出合并成一个 `anime_body_hand` composite reward，接进 MultiReward 框架。

之所以不用 VLM judge 做解剖 reward：VLM judge（Claude/Codex）给不出可解释的中间信号——reward 上升时分不清是"真正解剖对了"还是"风格让 VLM 看顺眼了"。几何奖励的失败路径是可见的，VLM judge 的失败路径是隐蔽的。

> 和已删的 DWPose 系统的本质区别：DWPose 手部 AP 明显低于 RTMW；旧系统没有专用手部 3D mesh 校验，手指拓扑靠 keypoint 几何近似；新系统手部单独建 HaMeR reward 分支，3D mesh > 2D keypoint 对手指 topology 的判断。

## 1. Context / 为什么做

`6f1e14e` 删掉了 DWPose + PoseStructureRewardModel 整套 anatomy reward，原因是"没有任何帮助，全是坏处"。根本问题不是几何 reward 方向错，而是工具链不够强：

- DWPose 在动漫/插画域的手部 AP 不足，大量 false negative（没抓到变形手）
- 旧 reward 没有独立手部 3D 分支，finger topology 用 2D 关键点几何近似，误判多
- 没有人类偏好校准层，几何分数和"看起来对"之间没有对齐

**RTMW-x** 是当前 OpenMMLab 主线 whole-body 模型，官方 release note 明确提升了 hand detail 精度，whole-body AP 超过 70；**HaMeR** 是 MIT 授权、专门为 occlusion / interaction 场景训练的 3D hand mesh 模型，finger topology 和 reproj error 是直接可用的 reward 信号。

## 2. 已就绪 / What's already wired

| 资源 | 路径 | 备注 |
|---|---|---|
| MultiReward 组合框架 | `vrl/rewards/functions/registry.py` | 加 component 即可 |
| RewardFunction + LocalRewardRuntime | `vrl/rewards/base.py` `vrl/rewards/runtime.py` | 新 reward 直接复用 |
| 图像 reward boilerplate | `vrl/rewards/functions/{aesthetic,pickscore}.py` + `models/{aesthetic,pickscore}.py` | Stage A/B 直接 mirror |
| `build_inmemory_artifacts` | `vrl/rewards/base.py:RewardFunction` | 已支持 image artifact |
| Pose optional dependency 先例 | `pyproject.toml [pose]` / `[pose-gpu]` | 新增 `[body-hand]` optional group 沿用此模式 |

## 3. Gap / 为什么不是开箱即用

1. **RTMW-x 需要 mmpose + onnxruntime**（或 torch mmpose inference）。mmpose 不在 pyproject.toml 主依赖，需要加 `[body-hand]` optional group。预训练权重从 OpenMMLab model zoo 下载，不是 HuggingFace，需要 `mim download` 或手动指定 URL。

2. **HaMeR 需要 SMPL-X 文件 + hamer 包**。hamer 不在 PyPI 主流，需要 `pip install git+https://github.com/geopavlakos/hamer`，且需要从官网申请 SMPL-X 模型文件（`SMPLX_NEUTRAL.npz`）。这个文件不能放进 repo，需要说明挂载路径。

3. **手部可见性门控**：只有手部在画面中可见且面积足够时才调用 HaMeR；否则手部 reward 设 neutral（0.5），不惩罚不奖励。需要在 RTMW 输出的 hand keypoint coverage 上做判断。

4. **动漫风格校准**：RTMW 和 HaMeR 都在真实人体数据上训练。直接把几何分数当 reward 会惩罚合法的 chibi 比例、简化手指等风格元素。需要离线用私有 anime 偏好对（good/bad body, good/bad hand）拟合一个 isotonic calibration，在 reward pipeline 里加载。这一层在 Stage D 做，Stage A–C 先用原始几何分数 + 宽松阈值上线。

## 4. 分阶段方案

### Stage A — RTMW whole-body reward（全身 2D 主干）

**目标**：`AnimeBodyPoseRewardModel`，输入 image，输出 body/hand keypoint coverage + 骨长比例误差 + 关节角异常率，映射为 `anime_body_pose` ∈ [0, 1]。

- `vrl/rewards/models/anime_body_pose.py`:
  - `__init__`: 加载 RTMW-x onnx checkpoint（路径由 `worker_config["model_path"]` 或 env `VRL_RTMW_MODEL_PATH` 指定）
  - `__call__(*, artifact, request)`:
    1. 从 artifact 拿图像 → resize to RTMW 输入尺寸（384×288）
    2. RTMW 推理 → 133 whole-body keypoints + confidence
    3. 计算子特征：
       - `body_coverage`: 主体 keypoints（17 body）mean confidence
       - `hand_coverage`: 手部 keypoints（42 hand）mean confidence
       - `bone_ratio_score`: 归一化后躯干/四肢骨长比例 vs. 参考比例的偏差
       - `joint_angle_score`: 肘膝关节角合法范围内比例
    4. 加权组合 → `anime_body_pose` score
    5. 返回 `{"anime_body_pose": score}`

  ```python
  # 起始权重（后续 calibration 调）
  score = (
      0.30 * body_coverage +
      0.20 * hand_coverage +
      0.25 * bone_ratio_score +
      0.25 * joint_angle_score
  )
  ```

- offline smoke test：1 张正常 anime 人物图（手可见）+ 1 张手变形图 + 1 张无人图，验证三者分数有明显区分度，无人图不崩溃。

### Stage B — HaMeR hand mesh reward（手部 3D 专项）

**目标**：`AnimeHandMeshRewardModel`，在手部可见时输出 finger topology + 重投影误差，映射为 `anime_hand_mesh` ∈ [0, 1]。

- `vrl/rewards/models/anime_hand_mesh.py`:
  - `__init__`: 加载 HaMeR checkpoint（`worker_config["model_path"]` 或 env `VRL_HAMER_MODEL_PATH`），加载 SMPL-X neutral body model（路径 `worker_config["smplx_path"]`）
  - `__call__(*, artifact, request)`:
    1. 从 RTMW 输出（或重新检测）拿手部 crop + handedness
    2. 若手部 coverage < `min_hand_coverage`（默认 0.3）→ 返回 `{"anime_hand_mesh": 0.5}`（neutral）
    3. HaMeR forward → MANO mesh + 3D joints
    4. 计算子特征：
       - `reproj_score`: 2D 重投影误差（像素），归一化，越小越好
       - `finger_topology_score`: 相邻指段骨长比一致性
       - `handedness_score`: 左右手判定置信度
    5. 返回 `{"anime_hand_mesh": score}`

- offline smoke：同一批图，手变形图的 `reproj_score` 或 `finger_topology_score` 应该明显低于正常图。

### Stage C — RewardFunction 包装 + Composite reward + 注册

- `vrl/rewards/functions/anime_body_pose.py` → `AnimeBodyPoseReward(RewardFunction)`，mirror `AestheticReward` 结构，`LocalRewardRuntime`。
- `vrl/rewards/functions/anime_hand_mesh.py` → `AnimeHandMeshReward(RewardFunction)`，同上。
- 注册到 `vrl/rewards/functions/registry.py`:
  ```python
  "anime_body_pose": AnimeBodyPoseReward,
  "anime_hand_mesh": AnimeHandMeshReward,
  ```
- `configs/experiment/diffusion/anima_preview3/online_grpo_body_hand.yaml`:
  ```yaml
  reward:
    components:
      aesthetic: 0.3
      anime_body_pose: 0.4
      anime_hand_mesh: 0.3
    kwargs:
      anime_body_pose:
        model_path: ${oc.env:VRL_RTMW_MODEL_PATH}
      anime_hand_mesh:
        model_path: ${oc.env:VRL_HAMER_MODEL_PATH}
        smplx_path: ${oc.env:VRL_SMPLX_PATH}
  ```

### Stage D — 离线校准（deferred，独立 sprint）

- 收集 2k–5k 对 pairwise anime 偏好标注（A/B 哪张手更好 / 全身更自然）
- 对 `anime_body_pose` 和 `anime_hand_mesh` 原始分分别拟合 isotonic regression
- 校准后分数更接近"人看起来对不对"，chibi 比例误罚问题在这一层解决
- **本 sprint 不实做**，Stage A–C 用宽松阈值先上线观察信号方向

## 5. 关键文件

**复用（不重写）**：
- `vrl/rewards/base.py` — RewardFunction + build_inmemory_artifacts
- `vrl/rewards/runtime.py` — LocalRewardRuntime
- `vrl/rewards/functions/aesthetic.py` / `models/aesthetic.py` — Stage A/B 直接 mirror 结构
- `vrl/rewards/functions/registry.py` — 加 import + entry

**新增**：
- `vrl/rewards/models/anime_body_pose.py`
- `vrl/rewards/functions/anime_body_pose.py`
- `vrl/rewards/models/anime_hand_mesh.py`
- `vrl/rewards/functions/anime_hand_mesh.py`
- `configs/experiment/diffusion/anima_preview3/online_grpo_body_hand.yaml`
- `vrl/scripts/eval/validate_body_hand_reward.py`（离线验证脚本，Stage A/B smoke test 用）

**修改**：
- `vrl/rewards/functions/registry.py`（加两个 import + entry）
- `pyproject.toml`（加 `[body-hand]` optional group: mmpose, onnxruntime, hamer）

## 6. 验证矩阵

| 阶段 | 验证 |
|---|---|
| A | 正常人物图 `anime_body_pose` > 0.65；明显变形图 < 0.45；无人图不崩溃，graceful fallback |
| B | 手变形图 `anime_hand_mesh` < 0.45；正常手 > 0.65；无手/手面积小 → neutral 0.5 |
| C | registry 可解析两个 reward；recipe yaml dry-validate 通过；MultiReward `from_dict({"anime_body_pose": 0.4, "anime_hand_mesh": 0.3})` 初始化不报错 |
| C | 短跑（~5 epoch，小 batch）：`last_components` 里两个子 reward 都有非 NaN 值，方向与肉眼判断一致（好图高分，坏图低分）|
| 全仓 | `ruff check vrl tests` + `pytest tests/rewards/` pass |

## 7. Open design decisions

- **RTMW 推理后端**：onnxruntime（更轻，不依赖完整 mmpose）vs. mmpose PyTorch（更灵活，支持更多模型切换）。优先 onnxruntime，失败再回 PyTorch。
- **RTMW 和 HaMeR 是否共享 worker**：如果两个 reward 各自跑一次推理，会重复做人体检测。Stage C 可以把 RTMW keypoint 输出传给 HaMeR 做 hand crop，避免双份检测——但这需要 artifact 携带中间结果，和当前 RewardFunction 接口有一定耦合。先各自独立推理，性能问题出现时再合并。
- **无手场景的 neutral 分数**：0.5 是中性，不奖不惩。如果训练后发现模型学会了"藏手得 0.5 比画坏手得 0.3 合算"，把无手 neutral 降到 0.4 或加"上臂可见但无手"惩罚。
- **chibi/夸张比例误判**：Stage A–C 用宽松阈值（骨长误差容忍 ±40% 而不是 ±20%）先上线；真正解决在 Stage D 的 anime 偏好校准。
- **HaMeR SMPL-X 文件申请**：需要用户提前在 https://smpl-x.is.tue.mpg.de 申请下载。挂载路径通过 `VRL_SMPLX_PATH` env 指定，不 hardcode。

## 8. 非目标

- **不**在本 sprint 做 Stage D 校准（需要私有 anime 偏好数据集，单独 sprint）。
- **不**加 SDPose-WholeBody（SD backbone 推理 120–300ms，不适合 RL inner loop；作为离线验证模型可考虑，但不进 reward 主链路）。
- **不**加 SAM 3D Body / Hand4Whole++ / SMPLest-X（太重，适合离线 hard-case 发现，不进每步 reward 计算）。
- **不**替换 aesthetic reward（几何 reward 和美学 reward 是互补，不是替换关系）。
- **不**处理视频帧间一致性（本 sprint 是 anima 图像 reward 范围，不入视频）。

## 9. References

- **RTMW**（MMPose, Apache-2.0）: whole-body AP > 70, 官方强调 hand detail 改善，[release note](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- **HaMeR**（MIT）: "Reconstructing Hands in 3D with Transformers", 专门强化 occlusion/interaction，[paper](https://arxiv.org/abs/2312.05251) / [repo](https://github.com/geopavlakos/hamer)
- **Human-Art / COCO-OOD**（SDPose 跨域 benchmark）：用于评估 reward 对动漫/插画域的泛化性，后续验证引用
- 研究报告（用户提供 2026-05-27）：分层几何 reward 策略，SDPose > RTMW 在动漫 OOD 上，但 RTMW 更适合 RL inner loop 延迟要求
