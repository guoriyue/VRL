# SPRINT: 可验证的视觉 RL 环境（任务规格 + 分层 verifier + 失败归因）

状态：**planned（2026-09-21）**。研究与设计文档，未改代码。出发点是 CodeMidas
（arXiv 2609.22068 §3.2–3.4）和 MiMo-V2.6 技术博客里"环境和 verifier 怎么生产、怎么
自证可信"的机制，对照本仓 RL 环境的现状，给出三类现代视觉任务（分层拆解、局部编辑、
组合生成）的环境定义和一个"先查 verifier 再怪模型"的归因框架。

## 0. 结论先行

1. 本仓的 RL 环境是一个 **contextual bandit**：`(prompt, reference_image?)` → 一次生成
   → 一个标量。任务没有规格（spec），reward 没有结构，失败无法归因。这套能训
   "PickScore 更高的图"，训不了"把这张图拆成 N 个透明层、每层一个物体"或"只改遮罩
   内、遮罩外一像素不动"——不是算法不够，是**环境没有描述这些任务的语言，verifier
   没有检查这些任务的手段**。
2. CodeMidas 真正可迁移的不是"自动造任务"，而是四条纪律：**verifier 由执行落地**（跑
   参考实现记录结果，而不是让模型描述期望）；**verifier 先过 oracle**（参考解法在 6 个
   新容器里必须过，起始代码必须不过）；**审核 agent 找 verifier 和规格的不一致**（假阳
   / 假阴都算）；**只保留同一模型下既有成功又有失败的任务**，并且明确区分"全失败 = 任务
   太难"和"全失败 = 测试坏了"——后者靠前两条纪律排除。
3. 三类任务里，**分层拆解最接近"跑测试"**：把 N 层 alpha 合成回去必须还原原图，这是纯
   可执行的检查，不需要判别模型，oracle 也天然存在（数据集自带的层）。局部编辑一半可
   执行（遮罩外像素恒等）、一半要判别器（遮罩内是否按指令改了）。组合生成（GenEval）
   本仓已经有结构化 spec + 检测器 verifier + `why` 字符串，是现成的样板。
4. 归因框架分四层，从下往上排除：**精度层**（rollout/replay 概率一致，已有 parity gate）
   → **verifier 层**（oracle 必须过、退化输入必须不过）→ **任务层**（组内混合结果）→
   **模型层**（只有前三层干净才把失败记到模型头上）。今天本仓只有第一层。

## 1. 论文里可迁移的机制（精确到出处）

CodeMidas 把一个开源仓库变成一个 RL 任务：agent 找到公共入口和可观察行为，删掉核心
实现留出起点；**测试不是模型写的期望，而是调用参考实现记录下来的输出**（"invoking
reference implementations and recording outcomes"；CLI 用命令执行、纯函数用输入输出对、
有状态 API 用调用序列）。之后三道过滤（§3.4）：

| 机制 | 做法 | 排除什么 |
|---|---|---|
| 一致性验证 | 训练运行时设置下开 **6 个新容器：2 个装起点代码、4 个装参考解法**；参考解必须全过，起点必须不过 | verifier 坏、环境不可复现、题目已经"预先做完" |
| 泄漏对抗 rollout | 一个 agent 专门尝试从编译产物、缓存副本里恢复答案而不做开发 | 不做任务也能拿分的漏洞 |
| verifier 一致性审核 | 独立审核 agent 看"实现是否满足陈述"，并找**陈述与 verifier 行为的不一致**，假阳假阴都标 | 规格和测试说的不是一件事 |
| rollout 结果过滤 | 在目标模型 + 预算下跑多次，**只保留既有成功又有失败的任务** | 全失败（太难或题坏——但前两道已经把"题坏"筛掉，剩下的全失败才归为太难）、全成功（无梯度） |

规模：5,545 任务 / 3,185 仓库；消融里过滤后 3k 子集全面胜过未过滤 8k。这条消融是本
文最重要的数字：**环境质量比环境数量值钱**。

MiMo-V2.6 把 RL scaling 拆成三条线——batch/吞吐（1,568 样本/更新）、环境（7,000+）、
grader 算力（组内相对比较打分）——并列出"奖励设计、对抗性评测、异常检测、验证器交叉校
验"四道 reward hacking 防线。博客没有写失败归因机制；把"验证器交叉校验"落到可操作的
步骤，是本文 §4 要补的。

## 2. 本仓的环境现状

一次 rollout 的全部输入是 `GenerationInput(prompt, task_type, reference_image,
reference_video)`（`vrl/generation/types.py:31`），reward 侧看到的是
`RewardSample(prompt, output, sample_id, metadata)`（`vrl/rewards/types.py:24`），返回
`RewardOutput(scores, components)`。任务只有 `t2i / t2v / i2v / t2w / v2w` 五个字符串，
区分的是模态，不是任务。

已有、可复用的零件：

| 零件 | 位置 | 在新框架里的角色 |
|---|---|---|
| 结构化任务规格 + 可执行 verifier + 理由 | `manifests/geneval/*.jsonl` 的 `metadata.geneval.include[{class,count}]`；`GenEvalOwlRewardModel` 输出 `strict/partial/dense` 和 `why`（`vrl/rewards/models/geneval_owl.py`） | **唯一现成的"spec → 检查 → 理由"样板**，§5 契约照它推广 |
| 多分量 reward | `RewardOutput.components`、`reward.components` 权重、0 权重观测项（SANA preset 的 aesthetic hack 探测器） | 每个检查一个分量；权重 0 的分量就是"只观测不训练"的审核信号 |
| 组内零 advantage 过滤 | `nonzero_advantage_mask`、`adv_zero_rate`、`trained_prompt_num`（`trainer.py:1043-1081`） | 这就是 CodeMidas 的"只保留混合结果"——但今天它**在训练时静默丢弃**，不记录、不归因、不回流到数据侧 |
| 精度对账 | `replay_parity`、`debug.first_step`、`logprob_abs_diff_*`、`mismatch_kl` | 归因框架第一层，已完成 |
| 条件输入 | `reference_image` / `reference_video`（i2v 已用） | 局部编辑的源图、分层拆解的原图走同一个口 |

缺的：任务规格没有通用字段（geneval 塞在 `metadata` 里，reward 靠 `metadata_key` 约定
取）；没有 oracle 的概念（数据集里的参考答案不进环境）；verifier 没有自检；
`adv_zero_rate` 是一个汇总数字而不是按 prompt 的记录；没有"哪一步检查失败"的输出——
`GenEvalVerdict.why` 是唯一例外，且它不进 metrics。

## 3. 三类任务的环境定义

统一写法：**输入 / 规格 / oracle / 可执行检查 / 需要判别器的检查 / 已知 hack**。
"可执行"指不依赖任何学习模型、结果确定、可以像单元测试一样对 oracle 跑一遍。

### 3.1 分层拆解（一张图 → N 个透明层，一层一个物体）

- 输入：原图 `I`；规格：层数 `N` 与每层的物体描述 `[("cat", 0), ("sofa", 1), ("background", 2)]`。
- oracle：数据集自带的层（合成数据、PSD 导出、或 SAM 分割 + 补全得到的伪层）。
- 可执行检查：
  1. **重建**：按 z 序 alpha 合成 `over(L_N, …, over(L_2, L_1))` 与 `I` 的 PSNR/LPIPS
     超过阈值——这是整个任务的"跑测试"。
  2. **层数与格式**：恰好 N 层、每层 RGBA、alpha 不全 0 不全 1。
  3. **层间不重叠**：alpha 两两交集面积 / 并集面积 低于阈值（防止把整图复制 N 份）。
  4. **层非空且不是全图**：每层 alpha 覆盖率在 `[lo, hi]` 内。
- 判别器检查：每层 alpha>0 区域裁出来，检测器/CLIP 判定是规格里那个物体且**只有**那
  一个物体（复用 `geneval_owl` 的检测 + 计数逻辑）。
- 已知 hack（对抗 rollout 要专门试）：一层放整图其余层空 alpha（2、4 拦）；N 层平分
  像素而不按物体（判别器拦）；alpha 半透明糊边骗重建（3 + 边缘硬度检查）。
- 前置：需要一个输出 RGBA 的 family。`~/Desktop/VRL` 的 Qwen-Image-2.1 集成（提交
  `80b012dd`，"reference editing and RGBA rollout outputs"）做过 RGBA 输出与参考编辑
  的 rollout，`docs/reference/qwen_image_21_editing.md` 记录了未训练的属性编辑/整物替换
  探针。移植时只搬 family 代码和媒体工具，不搬那边的配置分层。

### 3.2 局部编辑（遮罩内按指令改，遮罩外不动）

- 输入：源图 `I`、遮罩 `M`、编辑指令 `e`；规格：`(M, e, 编辑类型 ∈ {attribute, replace, remove, add})`。
- oracle：编辑数据集的目标图（有则用；没有则只做可执行部分 + 判别器）。
- 可执行检查：
  1. **遮罩外恒等**：`‖(1−M)⊙(O − I)‖` 低于阈值（按 L1 和 LPIPS 各一条）。
  2. **遮罩内确实变了**：`‖M⊙(O − I)‖` 高于阈值——防止"什么都不改"拿满分 1。
  3. **几何不变**：输出分辨率、宽高比与源图一致。
  4. 有 oracle 时：遮罩内与目标图的 LPIPS。
- 判别器检查：VLM 判"遮罩内是否满足 `e`"（yes 概率，即 backlog F）；`replace/remove`
  类另加检测器确认旧物体消失/新物体出现。
- 已知 hack：改遮罩外一圈骗 VLM（1 拦）；只改颜色不改结构应对 replace（检测器拦）；
  输出整体轻微模糊让 LPIPS 都很小（2 + 锐度检查，`image_sharpness` 已有）。
- 与 3.1 的关系：3.1 是 3.2 的推广——"每一层"就是一个带 alpha 遮罩的区域；两者共用
  "遮罩外恒等 / 遮罩内变化"两条检查。

### 3.3 组合生成（GenEval 型）

已有。要做的只是把 `GenEvalVerdict.why` 和三档分数按 §5 的契约输出成结构化检查，
而不是三个平铺的浮点。它是 3.1 判别器检查的直接来源。

## 4. 归因框架："verifier 真的验证问题出在哪"

一个 prompt 组（同 prompt N 个样本）训练前后只有四种可见结果：全 0、全 1、混合、和
"reward 高但人看是坏的"。今天本仓把前两种静默丢掉、第四种靠事后看图。改成按层排除：

```text
第 1 层  精度      rollout 与 replay 的 log-prob 一致？          已有：replay_parity / first_step
   │ 不一致 → 训练计算的问题，与任务无关，停
第 2 层  verifier  oracle 过所有检查？退化输入（原图原样返回、空图、
                   整图复制 N 份）不过？                          缺：oracle 不进环境
   │ oracle 不过 → verifier 或规格坏，任务下线送审
   │ 退化输入过 → verifier 有洞，先补检查再训
第 3 层  任务      同一模型、同一预算下该 prompt 组是否有混合结果？  半有：adv_zero_rate 汇总
   │ 全 0（且第 2 层干净）→ 对当前模型太难，进"难题池"，不训
   │ 全 1 → 太简单或被 hack，跑对抗检查（§3 每类的已知 hack 列表）
第 4 层  模型      只有前三层干净，混合结果的组才产生梯度；这时的失败才记到模型头上
```

每层的产物都要落盘，不只是过滤：

- verifier 自检报告：每个任务的 oracle 检查结果 + 退化输入检查结果，一个任务一行
  （对应 CodeMidas 的 6 容器一致性验证）。任务进 manifest 前必须有这一行。
- 按 prompt 的结果记录：`prompt_id, step, n_pass, n_fail, checks_failed 直方图`。
  `adv_zero_rate` 是它的一个汇总；难题池和"太简单池"从它派生，并回流到数据侧的采样权
  重（backlog D 的 best-of-N 筛选和 `SPRINT_ngu_adaptive_sampling` 都从这张表取数）。
- hack 观测项：0 权重分量（已有机制）+ 对抗输入的分数。SANA aesthetic 那次 texture
  collapse 是 reward 高图坏的实例，当时靠人看曲线发现；有第 2 层后它应该在"退化输入
  （高频噪声贴图）过 PickScore"这一步被提前抓到。
- 判别器检查的可信度：判别器本身也要过 oracle（目标图必须 yes、源图必须 no），并记
  录一致率；MiMo 的"验证器交叉校验"落到这里就是两个判别器对同一批样本的一致率。

## 5. 契约草案（不写代码，先定名字和归属）

```python
# vrl/rewards/verify/...（归属：reward 域；任务规格随 manifest 行走）
@dataclass(frozen=True)
class TaskSpec:                 # 一个 manifest 行的结构化任务，取代塞在 metadata 里的约定
    kind: str                   # "layer_decomposition" | "masked_edit" | "geneval" | ...
    inputs: GenerationInput     # 现有：prompt + reference_image/video
    spec: Mapping[str, Any]     # kind 自己定义的字段（N、层描述；M、e；include）
    oracle: Mapping[str, str] | None   # 参考答案的 artifact 路径（层文件、目标图）

@dataclass(frozen=True)
class Check:                    # 一条检查的结果，是 reward 的最小单位
    name: str                   # "reconstruction", "outside_mask_identity", ...
    passed: bool
    value: float                # 连续量，供 dense reward 与阈值扫描
    why: str                    # GenEvalVerdict.why 的推广

@dataclass(frozen=True)
class Verdict:
    checks: tuple[Check, ...]
    def reward(self, weights) -> float      # 现有 reward.components 权重直接作用在 check 上
    def components(self) -> dict[str, float]   # 落到 RewardOutput.components
```

- 每个 `kind` 一个 verifier 类，拥有：对 `TaskSpec` 的可执行检查、判别器检查、**自检**
  （`verify_oracle(spec)` 与 `verify_degenerate(spec)`），和该 kind 的已知 hack 输入
  生成器。三者放一起是有意的：CodeMidas 的三道过滤都是 verifier 的一部分，不是训练器
  的一部分。
- 现有 `RewardModel.score_media(media, prompt)` 不变；verifier 是它上面的一层，把
  `RewardSample.metadata` 里的 `TaskSpec` 和输出媒体变成 `Verdict`。GenEval 是第一
  个迁移对象（它已经是这个形状，只是没起名字）。
- 训练器只多做一件事：把每组的 `Verdict` 汇总写进按 prompt 的结果记录（§4 第 3 层），
  过滤逻辑不变。

## 6. 第一个可执行 sprint：分层拆解环境

选它而不是局部编辑，因为它的核心检查（重建）是纯可执行的，oracle 天然存在，最接近
"写一个测试然后跑"；局部编辑的核心检查（遮罩内是否按指令）绕不开 VLM 判别器，第 2 层
的自检会更弱。

1. **数据**：合成 200 张 2–4 层图（物体贴图 + 背景，层文件即 oracle），manifest 行带
   `TaskSpec(kind="layer_decomposition")`。
2. **verifier**：§3.1 的 4 条可执行检查 + 1 条检测器检查；自检：oracle 层全过；三种退化
   输入（整图一层、N 份复制、平分像素）全不过。这一步**不需要 GPU 训练**，先做完。
3. **family**：移植 RGBA 输出的 family（Qwen-Image-2.1，见 §3.1 前置），rollout 一次
   输出 N 层——或退一步，N 次条件生成拼层（每次给"前面已生成的层"作参考图），先通
   路后效率。
4. **未训练基线**：跑第 3 层，得到每个任务的 `n_pass/n_fail`。预期大部分是全 0——此时
   不训，先按 §4 判断是 verifier 阈值太严（oracle 过但边缘检查卡住模型）还是任务真的
   太难，必要时把 N 降到 2、物体换成大且分离的。
5. **门槛**：至少 30% 的任务在未训练模型上出现混合结果，才开始 GRPO；训练指标除
   reward 外必须报告每条 check 的通过率曲线——只有 `reconstruction` 升、检测器检查不
   升，就是 hack 的早期信号。

## 7. 不做 / 不算

- 不做通用 agent 框架、不做 episode 循环（那是 `parked/SPRINT_agentic_visual_rl_program.md`
  的事）。本文的任务仍是单步生成，只是规格、verifier、归因变了。
- 不把 denoise step 的 parity 检查（第 1 层）和 verifier 检查（第 2 层）混为一谈：前者
  保证训练计算一致，后者保证学习目标可信，各自独立通过。
- 不用"训练 reward 上升"当环境可信的证据；证据是第 2 层自检报告和第 3 层的混合率。
- 不用 LLM 生成任务规格来代替 oracle：CodeMidas 的规格也是 agent 写的，但测试是执行参
  考实现录出来的——视觉任务里对应的是"检查由像素运算和 oracle 层定义"，不是"VLM 说像
  不像"。
