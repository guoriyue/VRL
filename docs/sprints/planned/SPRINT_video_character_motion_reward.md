# SPRINT: 视频角色一致性与真实运动 Reward 接入（in-progress）

状态：in-progress（2026-06-24）。目标：为“角色稳定、衣服/纹理稳定、舞蹈转圈时裙摆/布料运动可信”
找可直接复用的公开视频 reward / evaluator，并把最有价值的候选接入 VRL 的 reward/eval 面。

实现落在 `~/Desktop/vrl2/VRL` 的独立分支 `feat/videoscore2-reward`（与 wm-infra 主线分开）。

## 已完成（P0 + P1 全部 reward 接入）

忠实复用上游真实方法的候选都已接成 reward 或 eval adapter，全部沿用现有 `RewardFunction` +
`RewardModel` + `score_key` 形状，未改 reward schema。registry 注册了 3 个新 component：
`videoscore2` / `unified_reward_video` / `phymotion`。

**不做 identity / clothes 的手搓近似**：曾试过用整帧 CLIP 余弦做 identity、用颜色直方图 +
Laplacian 闪烁做 clothes，但这俩并不真的测“同一个人 / 同一条裙子”——静态片甚至无人片都能拿高分，
是 §2.6 自己警告过的弱代理，已删除。真正忠实的做法是 VBench-2.0 的 RetinaFace+ArcFace（identity）
和人物/衣物分割 + DINO/VQA（clothes），需要真权重才有意义，留到能装能验时再做，不上手搓版本。

learned VLM judges（Ray pool，mp4 artifact）：

- **VideoScore2** `vrl/rewards/models/videoscore2.py`：`TIGER-Lab/VideoScore2`（Qwen2.5-VL-7B，
  `AutoModelForVision2Seq` + `trust_remote_code`），greedy 生成 + 上游正则解析三维 1-5；
  默认 `soft_scores=true` 走 digit-token 期望值（连续 [1,5]），定位失败按维度回退整数。
  public keys：`visual_quality` / `text_alignment` / `physical_common_sense` / `overall`。
- **UnifiedReward-2.0** `vrl/rewards/models/unified_reward_video.py`：
  `CodeGoat24/UnifiedReward-2.0-qwen-7b`，16 帧 pointwise，按上游脚本输出 Alignment /
  Physics / Style（已是 1-5 浮点，天然连续）。rubric 默认在模块内（即模型输出语法），可用
  `worker_config.rubric_path` 指向 `configs/reward_rubrics/dance_cloth.yaml` 把注意力引到
  “同人/同裙/裙摆物理”。public keys：`alignment` / `physics` / `style` / `overall`。

人体动力学（可选外部环境）：

- **phymotion** `vrl/rewards/models/phymotion.py`：SMPL+MuJoCo 太重且 license 复杂，故不进基础
  依赖；以外部进程方式接入（`worker_config.phymotion_cmd` 带 `{video}`/`{output}` 槽，回读
  三轴 JSON）。public keys：`kinematic_plausibility` / `contact_balance` /
  `dynamic_feasibility` / `overall`。

eval / benchmark：

- `vrl/scripts/eval/video_reward_suite.py`：Kling 四维 + VBench（懒加载 custom_input）+ 自研模型
  列（`--videoscore2` / `--identity` / `--clothes`）+ 外部 benchmark 合并（`--merge-json
  prefix=path`，把 VMBench / VBench-2.0 / DynamicEval 各自 CLI 产出的 per-video JSON 折进 CSV，
  不猜它们未验证的 Python API）。

## 验证状态

- **单测**：每个 reward 都有 fake-actor facade 测试（score_key 选择 / 缺 key fail-fast /
  execution=pool / config 校验）+ 纯逻辑单测（VideoScore2 解析与 soft-score 对齐 /
  UnifiedReward 浮点解析与 rubric 加载 / PhyMotion 子进程与 JSON 解析）。全套 reward + config
  测试通过，无回归。
- **eval suite**：端到端跑通，产出 `eval_video_metrics.csv`（Kling 列 + VideoScore2 列 + 外部
  benchmark 合并列）。
- **VideoScore2 真机 7B 推理（2026-07-02 跑通，P0 gate PASS + 抓到一个真 bug）**：此前"已发起"的
  probe 实际卡在权重下载（4 shard 只下了 3 个）；续传后在 5090 真跑（wan_i2v_parity_probe 的两条
  480p mp4，fps=2，bf16）：**加载 2.9s（权重已缓存）、单视频推理 4.2-5.1s、峰值显存 15.8GiB
  alloc / 16.4GiB reserved**；digit token 1-5 全部解析为单 token，hard 解析路径正常
  （3/4/4 与 3/4/3），CoT + 最终评分行格式与 wrapper 假设一致。报告
  `wm-infra/outputs/videoscore2_probe/report_prefix_anchor_bug.json`。
- **真机核对发现 soft-score 对齐 bug（比回退更糟：静默锚错，回退率 0 但分是错的）**：模型 CoT 先写
  "Visual Quality Analysis:"（无数字），最终评分行是编号列表 `(1) visual quality: 3`——soft 路径
  从第一次 marker 命中往后找 digit，撞上列表编号 "1"，给出 soft=1.0 而 hard=3（text_alignment/
  physical 碰巧对齐只是格式运气）。修复：`_next_digit_step` 改为**锚定最后一次 marker 出现 + digit
  必须在 marker 后近窗口内**（评分行恒为 `<marker>: <digit>`；CoT 提及后远处的数字出窗 → 走 hard
  回退）。已加单测复现该场景（`tests/rewards/videoscore2/test_parsing.py::
  test_soft_scores_anchor_last_marker_not_cot_mention`，修复前红/修复后绿），reward 套件通过
  （2 个 OCR 红为 venv 缺 Levenshtein 的既有环境问题）。修复 + ±1 硬闸已合并为一个 commit
  **推上 VRL origin/main（`14a35069`，2026-07-07）**；**修复后真机复验 PASS（同日）**：sample00 visual_quality soft 1.0→**2.997**
  （≈hard 3），全部轴 soft≈hard 对齐、回退率 0，physical 3.85 展示出 soft 路径想要的整数间连续信号。
  复验报告 `wm-infra/outputs/videoscore2_probe/report_fixed.json`。P0 finishing criterion
  「fake tests 通过 + 本地 mp4 真实 inference」就此闭合。
- **加固（2026-07-07，回应"文本解析太启发式"的质疑）**：soft 路径结构性弱点 = 靠搜文字定位、与
  fail-fast 的 hard 正则解析互不校验。新增 `_merge_soft_with_hard` 硬闸：**soft 只允许在 hard
  整数 ±1.0 内做连续细化，超出即判定锚错 → 丢弃 + warning + 回退 hard**。上面那类 1.0 vs 3 的
  静默错分从此被结构性拦截（misanchor 单测 + merge 守卫单测各一，13 个 videoscore2 单测全绿）。
  分层结论：评分行文本格式是模型训练出的输出契约（上游官方推理同款正则，非我们的启发式）；
  我们自加的只有 soft 期望值扩展，现在它被约束为"只能细化、不能推翻"上游忠实解析。

## 外部 benchmark 已 vendoring（third_party 子模块）

按仓库既有约定（`third_party/<name>` git submodule + 必要时 `*_packaging` editable wrapper）把
无 PyPI 打包的研究仓库 vendoring 进来，主仓库只存 pinned 指针、不膨胀：

- `third_party/PhyMotion`（submodule）+ `vrl/scripts/eval/phymotion_score.py` 桥接脚本：把
  vendored `astrolabe.rewards.smpl_physics_score` 接到 `{video}->JSON` 契约；phymotion reward 的
  public keys 已对齐上游真实输出 `kinematic` / `contact` / `dynamic` / `overall`。
- `third_party/VMBench`、`third_party/DynamicEval`（submodule）：用各自 CLI 跑，结果经
  `video_reward_suite --merge-json prefix=path` 折进 CSV（不导入它们未验证的 Python API）。
- **VBench** 有 PyPI 打包 → 走 extra `pip install -e '.[videoeval]'`（不进 third_party，符合
  “vendored = 无打包” 的约定）。
- **VBench-2.0** 不是独立 Vchitect 仓库；其 Human-Fidelity 方法（RetinaFace+ArcFace identity、
  人物/衣物分割 clothes）尚未实现——只有装上真权重能验证时才做忠实版，不上手搓近似。

注：vendoring 只 pin 代码，权重和重型依赖（MuJoCo/SMPL、各 benchmark 的模型）仍由操作者在各自环境
安装。submodule 拉取后 `git submodule update --init third_party/<name>` 即可。

## 仍未做

- 各外部 benchmark 在真实环境里**实际跑通并出分**（已 vendoring + 合并边界就位，差各自重型安装/权重）。
- 组合训练 config（Phase D：`configs/experiment/...online_*_dance_cloth_reward.yaml`）与 before/after
  contact sheet 判读。
- VBench custom_input 是否支持 `temporal_flickering` 待真机确认（文档只列
  subject/background/motion_smoothness/dynamic_degree/aesthetic/imaging）。

## 0. Core Decision（先看这一段）

**不要再找一个“万能 heavy reward”替换 Kling。正确路线是：保留 Kling 的 `motion_quality`
作为通用运动质量底座，再接两个更细的信号层：**

1. **VideoScore2 / UnifiedReward 这类 learned judge**：负责视觉质量、文本对齐、物理/常识一致性，
   也能用 rubric 问“同一个角色、同一条裙子、布料运动是否合理”。
2. **VBench/VMBench/PhyMotion 这类可解释 metric**：负责拆出角色一致性、衣服一致性、闪烁、
   运动平滑、人体动力学可行性。

对“跳舞转圈裙摆飘起”这个目标，最有希望的组合不是单个模型，而是：

```text
reward =
  0.25 * kling_motion_quality
+ 0.20 * videoscore2_physical_common_sense
+ 0.20 * human_identity_consistency
+ 0.15 * clothes_texture_consistency
+ 0.20 * human_motion_dynamics
```

其中 `human_identity_consistency` / `clothes_texture_consistency` 初期从 VBench/VBench-2.0 思路拆，
`human_motion_dynamics` 优先试 PhyMotion；裙摆布料动力学本身没有一个成熟开源 reward 完整覆盖，
需要在上面这些 ready-copy 组件稳定后另做 domain reward。

## 1. 当前仓库基线

仓库已经有：

- `configs/reward/kling_video_reward.yaml`：KlingTeam/VideoReward，默认 `score_key: overall_reward`。
- `configs/reward/videocon_physics.yaml`：VideoCon-Physics，默认 `score_key: physical_commonsense`。
- `configs/reward/README.md`：已明确复合视频 reward 不要叠 aggregate；物理/运动复合时用
  `kling_video_reward.score_key=motion_quality`，VideoCon 用 `physical_commonsense`。
- `vrl/rewards/functions/registry.py`：多 reward 是原始分数线性加权，组件分数会进
  `last_components`，可以做 component-level 曲线。

所以本 sprint 不重写 reward 架构。新增候选都应走现有形状：

```text
vrl/rewards/functions/<name>.py     # RewardFunction facade
vrl/rewards/models/<name>.py        # model-backed scorer / external code wrapper
configs/reward/<name>.yaml          # score_key、artifact_format、worker_config
tests/rewards/<name>/...            # schema + score selection + smoke/fake model
```

架构边界：不要把长 prompt/rubric 作为 workflow 模块里的巨大 ALL_CAPS 常量；如果 VideoScore2 /
UnifiedReward 需要细粒度 rubric，把 rubric 放到 `configs/reward/rubrics/*.yaml` 或
`worker_config.rubric_path`，代码只负责加载和校验。

## 2. 候选分层（按 ready-copy 价值排序）

| 优先级 | 候选 | 结论 | 适合训练 reward 吗 | 适合固定 eval 吗 |
|---|---|---|---|---|
| P0 | **VideoScore2** | 最值得先接的 heavy learned video reward；公开 GitHub + HF 模型 + inference 脚本 | 是，作为 Ray reward worker | 是 |
| P0 | **VBench** | 最快可复用的 subject/temporal/motion metric 包 | 部分维度可训练，但先 eval | 是 |
| P1 | **PhyMotion** | 最贴近“人体真实运动 dynamics”的 ready-copy reward | 是，但依赖重 | 是 |
| P1 | **VMBench** | motion correctness 维度比普通 VBench 更细 | 先不要在线训练，太慢 | 是 |
| P1 | **UnifiedReward-2.0** | 通用 image/video reward，可用自然语言 rubric 补“衣服/裙摆”判断 | 可作为第二 judge | 是 |
| P2 | **VBench-2.0** | Human Identity / Human Clothes 方法高度相关 | 先确认代码可用；否则复现子指标 | 是 |
| P2 | **DynamicEval** | 对动态镜头/遮挡下的前景一致性有价值 | 先不训练 | 是/诊断 |

### 2.1 VideoScore2：P0 heavy learned judge

作用：替代“只用 Kling 总分”的大模型 judge，输出更适合训练的三维评分：
`visual quality`、`text-to-video alignment`、`physical/common-sense consistency`。项目页说明它基于
VideoFeedback2，使用 Qwen2.5-VL-7B-Instruct SFT 后再用 GRPO 对齐，且输出可解释 rationale。

接入形状：

- 新增 `vrl/rewards/models/videoscore2.py`，加载 HF `TIGER-Lab/VideoScore2`。
- 新增 `configs/reward/videoscore2.yaml`：
  - `score_key: physical_common_sense` 默认用于 motion/physics 复合。
  - `artifact_format: mp4`。
  - `worker_config.model_name: TIGER-Lab/VideoScore2`。
- 新增 public score keys：`visual_quality`、`text_alignment`、`physical_common_sense`、`overall`
  （以 wrapper 的 normalize 层为准，不能把上游自由文本 key 泄漏进训练配置）。

风险：VideoScore2 不是专门的 character/clothes reward。它的物理/常识维度能给“裙摆是否违反常识”
提供信号，但不会稳定拆出“同一条裙子的纹理是否保持”。因此它不能单独承担本任务。

### 2.2 VBench：P0 可拆 metric baseline

作用：先用 ready-copy metrics 建固定 eval 面，最小成本覆盖：

- `subject_consistency`
- `temporal_flickering`
- `motion_smoothness`
- `dynamic_degree`

接入形状：

- 先做 `vrl/scripts/eval/video_reward_suite.py`，读一个目录的 mp4 + prompt manifest，输出
  `eval_video_metrics.csv`。
- 不先接成 online reward；VBench 多维指标依赖各自模型，延迟/显存不可控。
- 只有当 fixed eval 能稳定区分“同人/换人、闪烁/不闪、静态/转圈”后，再把个别维度包装成
  `RewardFunction`。

### 2.3 PhyMotion：P1 人体真实动力学 reward

作用：这是现有候选里最贴近“跳舞动作是否物理可行”的 ready-copy reward。它从视频恢复 SMPL
人体 mesh，把动作 retarget 到 MuJoCo humanoid，再按三轴打分：

- kinematic plausibility
- contact and balance consistency
- dynamic feasibility

接入形状：

- 新增可选 reward `phymotion`，但默认不进入 `pyproject` 基础依赖；走 extra 或外部环境。
- wrapper 输出 `kinematic_plausibility`、`contact_balance`、`dynamic_feasibility`、`overall`。
- 首先只对人体舞蹈/转圈 prompt eval；确认 SMPL 恢复失败率、每视频耗时、显存和许可证。

风险：PhyMotion 管的是人体骨架/接触/动力学，不管裙子布料。它能防止“人漂浮、重心不对、转身不合理”，
但不能判断“裙摆是否自然飘起”。

### 2.4 VMBench：P1 motion correctness eval

作用：VMBench 的 perception-driven motion metrics 比普通 `motion_smoothness` 更接近人类判断，维度包括：

- Commonsense Adherence Score
- Motion Smoothness Score
- Object Integrity Score
- Perceptible Amplitude Score
- Temporal Coherence Score

接入形状：

- 先做 eval adapter，读取 VMBench 分数到统一 CSV。
- 如果 `Temporal Coherence` / `Object Integrity` 对舞蹈视频有稳定区分，再考虑包装成 reward。

风险：VMBench 是 benchmark/eval 形态，不是低延迟 reward。不要在 P0 把整套塞进在线训练 loop。

### 2.5 UnifiedReward-2.0：P1 通用 rubric judge

作用：UnifiedReward-2.0 支持 image/video 的 pairwise 和 pointwise scoring，维度包括
`Alignment`、`Coherence/Physics`、`Style`。它适合作为可配置 rubric judge，例如问：

```text
Does the video keep the same female character, same dress color/pattern/texture,
and plausible skirt motion while the character spins?
```

接入形状：

- 新增 `unified_reward_video` wrapper。
- rubric 从 YAML 加载，不写死在 Python 模块里。
- 只作为第二 judge / reranker；不要单独作为主优化目标。

风险：通用 VLM reward 对短 prompt 细节可能不稳定；它能“看懂问题”，但不一定比可解释 metric 更抗 reward hacking。

### 2.6 VBench-2.0：P2 human identity / clothes 方法源

作用：VBench-2.0 明确把 Human Fidelity 拆到 human identity 和 clothing temporal consistency：
identity 用 RetinaFace 检脸、ArcFace 特征相似度；clothes 用 video VQA 判断人物和衣服是否跨帧一致。

接入形状：

- 先验证是否已有可直接运行的 2.0 代码路径；如果没有，复现最小子集：
  - `human_identity_consistency`: RetinaFace/ArcFace 对首帧或 reference identity。
  - `clothes_consistency`: 人体/衣服 crop + video VQA 或 DINO/CLIP crop embedding。
- 先固定 eval；若方差低，再接 reward。

风险：clothes consistency 用 VQA 会慢且不一定数值平滑；如果要做训练 reward，优先用 embedding/segmentation
的连续分数，VQA 只做诊断。

### 2.7 DynamicEval：P2 遮挡/动态镜头诊断

作用：DynamicEval 指出 VBench motion smoothness 在遮挡/反遮挡、相机/前景运动时会误判，提出背景一致性
和前景对象一致性。舞蹈转圈和裙摆遮挡正好会触发这些失败模式。

接入形状：

- 作为 VBench motion smoothness 的诊断补充。
- 若有官方代码可用，接成 eval；没有则只保留为设计参考，不进 P0。

## 3. 目标 reward stack

先落一个可运行的组合，不等所有研究项完成：

```yaml
reward:
  components:
    kling_video_reward: 0.25
    videoscore2: 0.20
    phymotion: 0.20
    human_identity_consistency: 0.20
    clothes_texture_consistency: 0.15
  kwargs:
    kling_video_reward:
      artifact_format: mp4
      score_key: motion_quality
    videoscore2:
      artifact_format: mp4
      score_key: physical_common_sense
    phymotion:
      artifact_format: mp4
      score_key: overall
    human_identity_consistency:
      artifact_format: mp4
      score_key: identity_consistency
    clothes_texture_consistency:
      artifact_format: mp4
      score_key: texture_consistency
```

注意：这是目标形状，不是第一阶段就全部上线。P0 只要求 VideoScore2 + VBench eval 打通；
PhyMotion 和衣服/纹理 consistency 是 P1。

## 4. Phase Plan

### Phase A：外部代码和权重可用性 gate

完成以下 one-shot probes，结果写回本 sprint 或单独 `docs/sprints/info/`：

1. `VideoScore2`：clone/install/inference on one local mp4；记录单视频延迟、显存、输出字段。
2. `VBench`：跑 `subject_consistency`、`temporal_flickering`、`motion_smoothness`、`dynamic_degree`
   四个维度；记录依赖和输出格式。
3. `PhyMotion`：跑官方 demo 或一段本地人体视频；记录 SMPL 恢复失败率、MuJoCo 环境需求。
4. `VMBench`：跑一段静态/转圈 pair；确认 TCS/OIS/MSS 是否能区分。
5. `UnifiedReward`：跑 pointwise video scoring；确认 rubric 能否输出稳定数值。

删除 gate 产生的一次性脚本和中间视频，只保留结论、命令、commit/版本、输出摘要。

### Phase B：P0 接入

1. 接 `videoscore2` reward wrapper，Ray pool 路径，mp4 artifact。
2. 加 `configs/reward/videoscore2.yaml`。
3. 加 tests：
   - fake model 输出三维分数，`score_key` 选择正确。
   - 缺 key fail-fast。
   - debug JSONL 记录 public score keys。
4. 加 `video_reward_suite` 固定 eval 脚本，先支持 VBench 四维和现有 Kling component logs。
5. 在一组固定 prompt 上跑：
   - same-character dance spin
   - identity drift
   - static skirt / no spin
   - texture flicker
   - physically impossible body motion

### Phase C：P1 动力学和衣服一致性

1. 接 `phymotion` optional wrapper，显式标注环境依赖，默认不进普通 tests。
2. 实现 `human_identity_consistency` eval/reward：
   - face path：RetinaFace + ArcFace。
   - fallback：person crop DINO/CLIP embedding。
3. 实现 `clothes_texture_consistency` eval/reward：
   - person/dress/skirt crop。
   - color histogram + DINO/CLIP crop embedding。
   - texture flicker penalty（frame-to-frame embedding / high-frequency energy drift）。
4. 如果 VBench-2.0 官方代码可用，优先包装官方实现；否则按论文方法复现最小连续分数。

### Phase D：组合训练和判读

1. 新增实验 config：`configs/experiment/diffusion/*/online_*_dance_cloth_reward.yaml`。
2. 不看 training `reward_mean` 单点结论；必须看 fixed eval：
   - identity consistency 上升或不降。
   - clothes consistency 上升或不降。
   - motion/dynamics 上升。
   - Kling visual/text component 不崩。
3. 输出 contact sheet / mp4 grid，人工抽查同一批 prompt 的 before/after。

## 5. Finishing Criteria

P0 完成标准：

- `videoscore2` reward wrapper 可在 fake tests 下通过，并能在本地一段 mp4 上真实 inference。
- `video_reward_suite` 能输出固定 CSV，至少包含 Kling MQ + VBench 四维。
- sprint 记录每个外部工具的版本、命令、单视频耗时、显存、是否可在线训练。
- 不把任何大 rubric/prompt 模板写成 workflow 模块里的巨大 ALL_CAPS 常量。

P1 完成标准：

- `phymotion` 在至少一段人体动作视频上输出 `kinematic/contact/dynamic` 三个连续分数。
- identity/clothes consistency 至少能区分三类样本：同角色稳定、换脸/换人、衣服纹理闪烁。
- composite reward 的 component logs 全部进入 metrics；不能只看总分。

## 6. Non-Goals

- 不用 VideoScore2 或 UnifiedReward 单独替代所有 video reward。
- 不把 VBench/VMBench 全套直接塞进在线训练 loop；先 eval，确认 latency 和方差后再挑维度训练。
- 不在 P0 解决“裙摆物理仿真”。现成 reward 没有完整覆盖这个目标；P0 只把人体运动、
  身份/衣服一致性、通用物理 judge 打通。
- 不新增 reward schema 大改；沿用 `RewardFunction` + `RewardModel` + `score_key`。

## 7. References

- VideoScore2 GitHub：https://github.com/TIGER-AI-Lab/VideoScore2/
- VideoScore2 HF model：https://huggingface.co/TIGER-Lab/VideoScore2
- VideoScore2 paper：https://arxiv.org/html/2509.22799v1
- VideoScore2 project page：https://tiger-ai-lab.github.io/VideoScore2/
- PhyMotion GitHub：https://github.com/h6kplus/PhyMotion
- PhyMotion project page：https://phy-motion.github.io/
- PhyMotion paper page：https://huggingface.co/papers/2605.14269
- VBench GitHub：https://github.com/Vchitect/VBench
- VBench project page：https://vchitect.github.io/VBench-project/
- VBench-2.0 paper：https://arxiv.org/html/2503.21755v2
- VMBench GitHub：https://github.com/AMAP-ML/VMBench
- VMBench project page：https://amap-ml.github.io/VMBench-Website/
- VMBench paper：https://arxiv.org/html/2503.10076v2
- DynamicEval project page：https://nithincbabu7.github.io/DynamicEval/
- DynamicEval paper：https://arxiv.org/abs/2510.07441
- UnifiedReward GitHub：https://github.com/codegoat24/UnifiedReward
- UnifiedReward project page：https://codegoat24.github.io/UnifiedReward/
- UnifiedReward HF model：https://huggingface.co/CodeGoat24/UnifiedReward-2.0-qwen-7b
- Kling VideoReward GitHub：https://github.com/KlingAIResearch/VideoAlign
- Kling VideoReward HF：https://huggingface.co/KlingTeam/VideoReward

## 8. Repo References

- `configs/reward/kling_video_reward.yaml`
- `configs/reward/videocon_physics.yaml`
- `configs/reward/README.md`
- `vrl/rewards/functions/registry.py`
- `vrl/rewards/models/kling_video_reward.py`
- `vrl/rewards/models/videocon_physics.py`
