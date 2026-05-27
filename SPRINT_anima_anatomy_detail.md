# Sprint: 让 Anime Anatomy Reward 真正评估"排布"(不只是"在场")

状态:proposed(rule-based 富化 `geometry.py`)

## 0. 一句话 / TL;DR

**抓手是 fix the reward,但 rule-based 只能做兜底。** 当前 `pose/geometry.py`
只查"关节在不在 + 两个极端 bound",抓不到"排布是否合理"。但同样要警惕:
"5 指必须可见 / 肩必须在臀上 / 头身比要标准"这种硬规则会把**合法的遮挡 /
手势 / 视角 / chibi 风格**也罚了,reward 把模型逼到坍缩成单一 canonical pose。

**所以本 sprint 的设计**:
- **Stage 1-4(本 sprint 实做)**:rule-based 只评 **physically impossible**——
  反折的肘、内陷的指节这种"无论遮挡 / 手势 / 风格都不该出现"的几何。所有
  check **confidence-gated**(低置信度跳过,不罚)。这是 floor / 兜底。
- **Stage 5(单独 sprint,但架构锁定)**:**Claude-as-Judge VLM reward** ——
  Claude 视觉 + 世界知识直接判 anatomy plausibility,天然处理遮挡 / 手势 / 风格。
  小型 keypoint 分类器(5a)**砍掉**——上限不如 Claude,还多一条标数据 + 训
  模型的管线。

**逐函数 gap 诊断**:
- `_coverage` / `_visible_hand_count` / `_collapsed_hand_fraction` → 纯**在场**。
- `_joint_geometry_penalty` → 只用 `min_angle_degrees` 单边阈值,过伸的肘、
  反折的膝抓不到——Stage 1 加 `[min, max]` 范围 + confidence gate 修。
- `_limb_asymmetry_penalty` → 看左右比例,但不看绝对比例是否人形——Stage 5
  解决(用学到的 prior,而不是硬比例规则,因 chibi 等风格存在)。
- 手部 21 keypoint → 几何完全空载——Stage 1 加"指节折回 / 指尖反相"这种
  physically impossible 检查;"5 指 / 指序 / 长度比"这种**完整性**判断让给
  Stage 5。

## 0.5 设计原则 / Design principle(red-line — 写进文档,不要再被重复发明)

**Rule-based anatomy reward 必须满足 invariance under legitimate diversity。**

任何在合法的**遮挡 / 手势 / 视角 / 姿势 / 风格**下会产生 **false positive**
的硬规则,**不能**进 rule-based reward 层——会逼模型坍缩到单一 canonical pose,
反而毁了多样性。具体清单(本 sprint 主动排除,以后也不要加回来):

- ❌ "5 指必须可见" / "每只手必须有 N 个 keypoint" —— 破握拳 / 比 V / 侧视 / 背手 / 持物。
- ❌ "肩 y < 臀 y" 之类的拓扑序 —— 破躺姿 / 倒立 / 翻身 / 跳跃。
- ❌ "torso:leg ∈ [0.7, 1.4]" 之类的人形比例带 —— 破 chibi / 写意 / Q 版 / 透视。
- ❌ "前臂 / 上臂 = 0.8 ~ 1.2" 之类的肢段比例 —— 破透视 / 风格化 / 视角。
- ❌ "fingertips 必须落在掌中心扇形角度内" 之类的跨指排序 —— 破手势变体。

**允许进 rule-based 的标准**: physically impossible(不论遮挡 / 手势 / 风格
都不该出现)+ confidence-gated(低置信度 keypoint 不参与判定)。例如:

- ✅ 肘 / 膝 / 指节 over-extension(angle > 物理上限,如肘 > 195°)。
- ✅ 关节反折(angle < 物理下限,如膝 < 30°,且 keypoint 高置信度)。
- ✅ 指尖与掌中心反相(指尖在掌根一侧)。

**"是否看起来合理"这类需要从分布学的判断,一律走 Stage 5(学习式 prior /
VLM judge),不在 rule-based 里硬写。**

## 1. 背景与重构 / Context (corrected framing)

Base anima(`cosmos-predict2-anima`)出图四肢/手指/解剖细节差。

**之前的误判**:以为只要跑现成 `online_grpo_anatomy` recipe 就能解决。错。
用户的真问题:**当前 reward 只能查关节在不在图里,不能评估排布是否合理**。
shallow reward 下,RL 训多久天花板不变。

**Phase 3 把 `vrl/rewards/models/pose/geometry.py` 翻了一遍,确认现状**:

| 函数 | 实际在做什么 | 缺什么 |
|---|---|---|
| `_coverage` / `_visible_hand_count` / `_collapsed_hand_fraction` | 纯**在场**判断 | 不评估排布 |
| `_joint_geometry_penalty` | 只查**极端折回**(`angle < min_angle_degrees`)+ 段长比超 `max_segment_ratio` | 没有合理角度**范围**;过伸的肘 / 反折的膝抓不到 |
| `_limb_asymmetry_penalty` | 左右肢长比 > `max_ratio` 才罚 | 不问绝对比例是否人形 |
| 手部 21 keypoint | **完全没用进几何**——只用了 spread/count | 没有指数、指序、指长比例、指尖排布检查 |

底层逻辑:reward 不能推理 layout → RL 学不出 layout。本 sprint **不是训练
sprint**,是 **reward design sprint**——把 `geometry.py` 升级到能评估
arrangement plausibility。

## 2. 分阶段方案 / Recommended approach

### Stage 1 — 在 `geometry.py` 加 *physically-impossible* 检查(收窄版,confidence-gated)

**关键设计原则(修订)**:rule-based 只评 **physically impossible**——不论
遮挡 / 手势 / 风格都不可能合理的几何(肘 200° 反折、膝反向、指尖与掌反相)。
**完整性和合理性**(指数 = 5、躯干拓扑、人形比例)**不用硬规则**,因为:

- **遮挡**:挥手 / 持物 / 手伸出画面 → 看不到 5 指 ≠ 错;
- **手势**:握拳 / 比心 / 比 V → 不应该看到 5 个伸开的指尖;
- **视角**:手侧视 / 背手 → 部分手指被自身遮挡;
- **姿势**:躺着 / 倒立 / 翻身 → "肩在臀上"假设破裂;
- **风格**:chibi / 写意 → 人形比例假设破裂。

这些用硬规则罚 → reward 逼模型坍缩到"正面张开五指标准站姿",反而毁多样性。

**改 `vrl/rewards/models/pose/geometry.py`**,加 **2 个**新 penalty(收紧后),
全部 **confidence-gated**(任一 keypoint 置信度低 → 跳过该 check,**不罚**):

1. **`_anatomical_angle_penalty(person, *, joint_ranges, min_conf)`** —— 每个
   关节 bend 角度必须在物理范围 `[min, max]` 内(肘 [30°, 180°]、膝 [30°,
   180°] 等)。**只在 start/joint/end 三个 keypoint 都 ≥ `min_conf` 时计算**;
   低置信度直接跳过(不视为 violation)。范围设宽,**只抓 over-extension /
   反折** 这种物理不可能,不抓"角度有点怪"。

2. **`_finger_local_geometry_penalty(hand, *, min_conf)`** —— 对每只手中**高
   置信度连续可见**的 finger chain(thumb=0:5, index=5:9, middle=9:13,
   ring=13:17, pinky=17:21)单独评估其**局部几何**:
   - **指节折回**:同一指内三连节点(mcp→pip→dip)bend 角度 < 物理下限 → 罚;
   - **指尖与掌反相**:指尖应在掌中心远端方向上 → 否则罚。

   **不评估**指数、跨指顺序、长度对比——因为遮挡 / 手势下这些都可能是对的。

不做 `_body_topology_penalty` / `_body_proportion_penalty` /
`_limb_proportion_penalty` —— 这些都对姿势/风格不变性破得太严重。如果之后
确认需要,放到 Stage 5 由学习式 prior 承担。

全部**确定性、快速、可解释**,无模型依赖,纯 NumPy + 已有 `_Keypoint` 类型。

### Stage 2 — 在 `structure.py` 接进 score_request,暴露权重 + 置信度阈值

改 `vrl/rewards/models/pose/structure.py`,把 2 个新项作为加权 component
加进 `PoseStructureRewardModel.score_request` 的 mix,权重 + 置信度门通过
`worker_config` 暴露:

```yaml
anatomical_angle_weight: 1.0
finger_local_geometry_weight: 1.0
# Confidence gates(很重要——这是抗遮挡的核心)
geometry_min_keypoint_confidence: 0.4   # 高于全局 min_keypoint_confidence
finger_min_keypoint_confidence: 0.4
```

`vrl/rewards/functions/anime_anatomy.py` 的 `worker_config` 表面透传。
**所有新 penalty 都要在 per-sample diagnostics 里暴露**——包括每个 check
是否被 confidence gate 跳过、违例数等,便于事后分析哪些样本"被跳过"是合理
(遮挡),哪些是 DWPose 漏检(需要换模型)。

### Stage 3 — 用合成 pose 写单测(必须覆盖遮挡 / 手势 / 风格不变性)

扩 `tests/rewards/test_anime_anatomy.py`(pose backend 已 mock 过)。
**两类测试同等重要**:

**A. 正向触发(physically impossible 必须罚)**:
- 肘 over-extension(angle > 190°) → `_anatomical_angle_penalty` > 0;
- 肘反折(angle < 30°,但 keypoint 高置信度) → `_anatomical_angle_penalty` > 0;
- 单指内三连节点折回 → `_finger_local_geometry_penalty` > 0;
- 指尖与掌反相 → `_finger_local_geometry_penalty` > 0。

**B. 不变性(下面这些情况绝对不能罚——这是抗 false-positive 的 gate)**:
- **遮挡**:只 2 个手指 keypoint 高置信度,其余低 → 两个新 penalty 都 ≈ 0(跳过);
- **手势**:握拳 pose,所有指节高置信度但角度紧贴掌(全部在物理范围内)→ ≈ 0;
- **侧视手**:只见拇指 + 食指 + 中指 → ≈ 0;
- **躺姿**:整体 pose y 轴反转(肩在臀下)→ 不应触发任何 penalty(本 sprint 不做拓扑检查);
- **chibi**:头大身短(头/躯干 = 0.8) → 不应触发任何 penalty(本 sprint 不做比例检查);
- **半遮挡躯干**:腰部 keypoint 低置信度 → 不应该被 angle penalty 误伤。

**这是 gate**:reward 单测下区分不出"物理不可能 vs 合法遮挡/手势/风格",
RL 也区分不出 → 反而把对的样本罚了,模型坍缩。

### Stage 4 — 用 physically-impossible reward 跑 `online_grpo_anatomy`

Stages 1-3 全绿后:

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/anima_preview3/online_grpo_anatomy \
  reward.kwargs.anime_anatomy_structure.anatomical_angle_weight=1.0 \
  reward.kwargs.anime_anatomy_structure.finger_local_geometry_weight=1.0 \
  reward.kwargs.anime_anatomy_structure.geometry_min_keypoint_confidence=0.4 \
  reward.kwargs.anime_anatomy_structure.finger_min_keypoint_confidence=0.4 \
  trainer.output_dir=outputs/anima_anatomy_richreward_v1
```

**预期**:这一阶段只会消除最离谱的几何错误(反折肘、内陷指)。"手指数对不对、
姿势看起来合不合理"这些**完整性/合理性判断**需要 Stage 5——别把对这一阶段的
期望放太高。

### Stage 5(独立 sprint,但路线图锁定)—— Claude-as-Judge VLM Reward

**为什么 Stage 5 才是真正答案**:用户指出的核心 — "5 指不一定可见"、遮挡、
手势、风格 — 这些**根本不该用硬规则解**。能处理这类多样性的需要带世界知识
的视觉推理,Claude(或同级 VLM)直接命中——而且不需要标数据 / 自训分类器。

**Canonical 路:Claude visual judge**(明确选这条,**5a 小分类器砍掉**——
要标数据要训模型还做不过 Claude 的世界知识)。

#### 架构(对齐现有 reward 框架,不发明轮子)

新建 `vrl/rewards/functions/claude_anatomy_judge.py`:

```python
class ClaudeAnatomyJudge(RewardFunction):
    # 注册为 component name "claude_anatomy_judge"
    # score_batch(rollouts):
    #   1. RewardFunction.build_inmemory_artifacts(rollouts, media_type="image")
    #   2. base64 编码,组装 anthropic messages payload
    #   3. asyncio.gather(N 并发,N ≤ max_concurrent ≤ API rate limit)
    #   4. 每条结构化输出: {"score": 0..10, "reasons": [...]}
    #   5. score / 10.0 -> float ∈ [0,1],返回 list[float]
```

注册到 `vrl/rewards/functions/registry.py` 为 `claude_anatomy_judge`。
**不需要 ray reward pool**——Claude 本身就是远端 API,local transport 发并发
请求即可。

#### Recipe 用法(推荐 rule-based 兜底 + Claude 主信号 复合)

```yaml
reward:
  components:
    anime_anatomy_structure: 0.3   # Stage 1-4 兜底:physically-impossible
    claude_anatomy_judge: 0.7      # Stage 5 主信号:arrangement plausibility
  kwargs:
    claude_anatomy_judge:
      model: claude-sonnet-4-6      # 起步用 Sonnet,够强 + 便宜 5x
      max_concurrent: 16
      temperature: 0                # 确定性(GRPO advantage 需要)
      prompt: |
        Evaluate the character's anatomy in this image. Account for
        occlusion, gestures, viewing angle, and stylization (chibi /
        deformed are OK if intentional).
        Output ONLY JSON: {"score": 0..10, "reasons": [str, ...]}
```

#### 三件必须 lockdown 的事

1. **延迟**:`anthropic` SDK 并发 `asyncio.gather`,起步 16 并发;
   `rollout.n=8 × 8 prompts = 64 image / step`,Sonnet ~2s 单图 → 并发 16
   降到 ~8s / step reward。再嫌慢就走 ray transport 异步化 + 配合 continuous
   rollout queue sprint。
2. **成本**:Sonnet ≈ $0.005–0.01 / image,Opus ≈ $0.025–0.05 / image。
   1k step × 64 image ≈ $300–3000 / run。先用 Sonnet,prompt 压最短,几百
   step 短跑先看信号再决定长跑。
3. **稳定性**:`temperature=0` + structured output(JSON schema 强制
   `{score: int, reasons: [str]}`)。同一张图同一 prompt 同一温度必须返回同一
   score,GRPO 才能正确算 advantage。

#### 依赖 + 配置

- 加 `anthropic>=0.40` 到 `pyproject.toml`。
- API key 从环境变量 `ANTHROPIC_API_KEY` 取,不进 YAML 配置(与现有 reward
  worker_config 隔离凭证)。
- 失败模式:API 超时 / rate limit → 该 sample 的 reward 用 fallback
  (比如返回 0.5 中性值 + 日志告警),不要让训练步崩。

#### 为什么砍 5a(小型 keypoint 分类器)

- 要标"真 pose vs 垃圾 pose"数据(扰动生成的负样本 ≠ 真实 failure 分布);
- 训出来的 prior 上限不如 Claude 的世界知识;
- 维护一个额外模型;
- 一个不需要的中间环节。

**本 sprint 不实做 Stage 5**(独立 sprint 立项),但**架构和 prompt 草案
锁定在这里**,接手的人不用重新做架构决策。

## 3. 关键文件 / Critical files

- `vrl/rewards/models/pose/geometry.py` —— Stage 1,主要工作量;复用已有
  `_Keypoint`/`_PersonPose`/`_distance`/`_angle_degrees`。
- `vrl/rewards/models/pose/structure.py` —— Stage 2 接线 + diagnostics 表面
  (`PoseStructureRewardModel.score_request`)。
- `vrl/rewards/functions/anime_anatomy.py` —— `worker_config` 把新 kwarg 透传。
- `tests/rewards/test_anime_anatomy.py` —— Stage 3 合成 pose 单测。
- `configs/reward/anime_anatomy_structure.yaml` —— 选项:权重调好后烤进默认。
- `configs/experiment/diffusion/anima_preview3/online_grpo_anatomy.yaml` ——
  默认配置可工作时无需改;否则通过 dotlist 覆盖。

## 4. 验证矩阵 / Verification

| 阶段 | 验证方式 |
|---|---|
| 1 | REPL 跑合成 `_PersonPose`:正向(肘 200°、指节反折)→ penalty > 0;不变性(遮挡 / 握拳 / 侧视 / 躺姿 / chibi)→ penalty ≈ 0。 |
| 2 | `PoseStructureRewardModel.score_request` 对合成 batch 返回的 per-sample diagnostics 含 `anatomical_angle_penalty` / `finger_local_geometry_penalty` 字段 + "被 confidence gate 跳过"计数。 |
| 3 | `pytest tests/rewards/test_anime_anatomy.py` —— 正向 + 不变性两组用例全绿。**关键:不变性组失败 = false positive,reward 不合格,不能上 Stage 4**。 |
| 4 | `metrics.csv`:新 penalty 逐 epoch 下降;`eval_epoch_*/contact_sheet.png` 对照旧 reward baseline,**反折肘 / 内陷指**样本减少;同时**手势 / 遮挡 / 风格多样性**没有退化(主观对比 contact sheet)。 |
| 5(下一 sprint) | 学习式 prior:验证集 ROC-AUC 区分"真 anime pose vs 扰动 pose";接入后 contact sheet"看着像不像人"主观提升。 |
| 全仓 lint | `ruff check vrl tests` |

## 5. 执行命令 / Commands

每阶段的 canonical 命令(执行权限项):

```bash
# Stage 1-3: 改代码 + 跑单测 + lint
pytest tests/rewards/test_anime_anatomy.py -q
ruff check vrl tests

# Stage 4: 用 physically-impossible reward 训练
python -m vrl.scripts.train \
  --config experiment/diffusion/anima_preview3/online_grpo_anatomy \
  reward.kwargs.anime_anatomy_structure.anatomical_angle_weight=1.0 \
  reward.kwargs.anime_anatomy_structure.finger_local_geometry_weight=1.0 \
  reward.kwargs.anime_anatomy_structure.geometry_min_keypoint_confidence=0.4 \
  reward.kwargs.anime_anatomy_structure.finger_min_keypoint_confidence=0.4 \
  trainer.output_dir=outputs/anima_anatomy_richreward_v1
```

## 6. 非目标 / Non-goals(明确不做)

- **不**先跑现成 recipe 当第一抓手——这是之前的误判,reward 是天花板,先动它。
- **不**在 rule-based 里写"5 指必须可见"/"肩必须在臀上"/"头身比要标准"
  这种刚性完整性 / 拓扑 / 比例约束——这些会把合法的遮挡 / 手势 / 风格也罚了,
  让模型坍缩。这类"合不合理"的判断**留给 Stage 5 学习式 prior 解决**。
- **不**换 DWPose 为动漫域 pose 模型(独立 sprint;只有 Stage 4 后 keypoint
  抽取仍是瓶颈才上)。
- **不**在本 sprint 实做 Stage 5(学习式 prior / VLM judge),但**写进路线图**,
  因为它是"arrangement plausibility"的真正答案,本 sprint 的 Stages 1-4 只是
  floor / 兜底。
- **不**动数据 manifest、LoRA 容量、训练分辨率——这些是正交项,在 reward
  能评分了之后再讨论。
