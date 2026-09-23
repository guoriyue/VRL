# SPRINT: Qwen-Image-2.1 物体移动编辑 RL（object-move edit GRPO）

状态：**CLOSED — BROKE**（2026-09-23）。GATE A 在人工接受"召回有漏判"后放行（§9），GATE B Spearman 0.491（§10），
dry run 全部通过（§11），run1 训练 40 update 后按"40 update 仍平"规则停止；held-out 上 ckpt 比 base **明显更差**
（COCO reward 0.107 → 0.001，SpatialEdit-Bench move 分 0.661 → 0.166，物体丢失率 23% → 65%），512 训练分辨率下同样坏（§12）。
对比页：<https://claude.ai/artifact/S4dL7D8VuD6o2D2PPUfxnk>。

## 0. 一句话

不训"通用编辑"（base 已经会），训**它明确做不好、结果一眼能看出来、reward 能用现成模型
几何验证的**编辑：把指定物体移到指定位置（左/右/上/下、推近/拉远），物体只能有一个，其他
内容不变。

## 1. 证据：base 在哪里弱

**本地实测**（官方 diffusers `QwenImage21Pipeline`，40 步，seed 0，7 张真实照片 ×
8 种指令；人工目测，不是基准分；脚本和图在会话 scratchpad `edit_probe/`，对比页
<https://claude.ai/artifact/Lh8A3xx9JbTuQNe4jRrTE3>）：

| 编辑 | 通过 / 7 | 典型失败 |
|---|---|---|
| remove | 7 | — |
| zoom_out | 6.5 | 幅度偏小 |
| color | 6 | 指令歧义时把人整个染色 |
| top_view | 5.5 | 物体身份漂移（颜色变） |
| zoom_in | 5 | 不动 |
| move_closer | 5 | 做成整体镜头推近而不是物体靠近 |
| rotate | 4 | 镜像翻转（文字变乱码）、不动 |
| **move_left** | **1** | **复制出第二个物体**（扶手椅、摩托、人）、不动、物体消失 |

**外部证据**：SpatialEdit（arXiv 2604.04911）上 Qwen-Image-Edit 的 object moving 0.311，
是所有子项最低；rotation 0.531。Qwen-Image-2.1 官方只报了文生图基准，没有编辑分。

## 2. 为什么选"物体移动"

1. **失败是确定的、可见的**：复制 / 不动 / 丢物体，看图就知道对不对。
2. **可以用现成模型验证**（不是手写规则）：开放词表检测器给出物体框 → 数量、位移方向、
   大小比例都是检测输出上的几何量；DINOv2 给物体身份和背景保持。仓库里 `geneval_owl`
   已经是"检测框 → 稠密分"的先例。
3. **base 的主要失败正好是 reward 最容易抓的**：复制 → 检测数量从 1 变 2。
4. 旋转、镜头类留作第二阶段：需要 Orient Anything V2 / VGGT / DA3，许可和
   "2D 变形骗过位姿估计"的风险都还没验证。

## 3. Reward 设计（全部包装现成模型）

每条训练数据带结构化元数据：`object`（检测短语）、`direction`（left/right/up/down/closer/farther）、
`magnitude`（小/中/大）——方向来自数据，不从自然语言里解析。

- **几何分 G**（OWLv2 或 Grounding DINO，原图和编辑图各检测一次）：
  - 数量：编辑图里该物体数 ≠ 原图数 → G = 0（抓复制、丢物体）
  - 位移：物体框中心**相对背景/其他物体**沿指令方向的位移，连续打分（抓"整图平移/裁剪"）
  - 大小：左右移动要求面积比 ≈ 1；closer/farther 要求面积比按方向变化（+ 可选 Depth-Anything-V2-Small 物体区域深度）
- **一致性分 C**（DINOv2，已有）：物体 crop 身份相似度 + 两个框之外的背景相似度
- **合成**：`sqrt(G × C)`（SpatialEdit 的做法），不用加权和——否则"原图不动"靠 C 就能拿高分
- **整体质量**：EditReward（已在仓库）只作小权重或下限；它在尺寸/镜头一致性上分别只有
  57.3 / 60.9（VCReward-Bench），小 judge 在 Edit-R1 里被 hack 过。

**钻空子清单与对应的抓手**：原图不动（G=0）、重新生成一张别的图（C 塌）、整图平移或裁剪代替移动
（相对位移 + 背景 C）、复制（数量）、删掉或糊掉物体（检测置信度下限）、改尺寸改比例（管线固定输出尺寸）。

## 4. 数据

- **训练 prompt 池**：SpatialEdit-500K `object_moving/`（Apache-2.0，Blender 渲染，带精确变换）
  + AnyEdit `movement`（7,543，CC-BY-4.0，含 COCO 等真实照片，`Bin1117/anyedit-split`）。
  两者都要转成上面的结构化元数据；AnyEdit 需要从指令里抽物体和方向，抽取后人工抽查。
- **Held-out 评测**：SpatialEdit-Bench move（276 条，自带 `bbox_gt`，有 Qwen-Image-Edit 基线，
  不进训练）+ 一小批真实照片（本地 probe 那 7 张类型）目测。

## 5. 执行顺序与关卡（前一关不过不往下）

1. **Reward 离线验证**（不训练）：本地 probe 的 7 条 move_left 输出 + 人造对照（原图不动、
   复制、整图平移、换场景、正确移动）→ 分数必须按"正确 > 其他所有"排序，且同一输入打分确定。
2. **Base 基线**：SpatialEdit-Bench move 276 条用我们的 reward 和 benchmark 自带打分各打一遍，
   两者排序要大体一致（reward 可信度的第二个证据）。
3. **接训练**：family 已支持参考图编辑；新 recipe = 编辑数据 + 新 reward。单卡：LoRA、
   512–768 px 训练（OCR 实验已证明小图训练能迁移到全分辨率）。
4. **短跑**：30–60 update 看 reward 曲线。
5. **Held-out**：同噪声配对 base vs ckpt，benchmark 分 + 目测复制率是否下降。

## 6. 非目标 / 注意

- 不做通用编辑 RL：base 的 remove/color/zoom_out 已经够好，信号小、结果难看出来。
- 不写手放区域规则、不手调颜色阈值（见已删除的 locality reward）。
- 许可：Qwen-Image-2.1 是 `qwen-research` 许可；YOLO（AGPL）不用；VGGT 是 CC-BY-NC。

## 7. GATE A 结果（2026-09-23）：FAIL

测试：7 张 probe 照片，每张 7 个候选，用 `object_move` 打分，同一输入打两次（全部确定性一致）。
候选：正确移动（只改旧框和新位置：旧框用模型自己的 remove 输出填，新位置贴原物体）、原图不动、
复制一个（贴在原物体旁边、不重叠）、整图平移、重新生成（模型 zoom_out 输出）、换一张照片、
base 模型自己的 move_left 输出。通过条件：正确移动 > 所有作弊。

第一轮（按类别数实例）3 张直接因为计数错误失败：机场里其他飞机、扶手椅旁的吊椅都被算成实例。
改成"只数 DINOv2 外观和被移动物体相似（cos ≥ 0.7）的实例"，同时修了 harness 的三个不公平
（床贴着左边缘无处可移、正确移动的背景被整张重生成、复制品与原物体重叠被 NMS 合并）后重跑：

| 照片 | 正确移动 | 最高作弊 | 结果 |
|---|---|---|---|
| cat | 0.923 | 0.778 整图平移 | PASS |
| armchair | 0.960 | 0.774 整图平移 | PASS |
| bed | 0.974 | 0.684 整图平移 | PASS |
| airplane | 0.000（计数 1→2） | 0.763 整图平移 | FAIL |
| motorcycle | 0.480（只有 9.6% 空间可移） | 0.605 整图平移 | FAIL |
| zebra | 0.000（贴过去与左边斑马重叠被合并，3→2） | 0 | FAIL |
| woman | 0.723 | 0.878 复制 | FAIL |

**两个根因（都不是调阈值能可靠解决的）：**

1. **背景保持信号太弱，整图平移几乎每张都拿 0.6–0.78。** DINOv2 同位置 patch 余弦对平移后的
   背景仍有 0.23–0.49（小位移时甚至 0.93），而它在最终分数里只以 ¼ 次方出现
   （`sqrt(G × sqrt(id × bg))`）。真正需要的是"背景有没有整体位移"的**几何**估计
   （例如用现成的特征匹配模型 LoFTR / LightGlue 估计全局变换，要求接近恒等），而不是外观相似度。
2. **OWLv2-base 的实例计数在"复制"和"拥挤场景"上不可靠。** 并排的复制人被框成一个
   （复制作弊 0.878 反而最高），相邻斑马被合并，机场背景飞机时有时无。要可靠地判"多了一个"
   需要实例分割级别的模型（SAM 2.1 / SAM 3 按文本分实例）或更强的检测器，并单独验证。

**base 自己的 move_left 输出**：7 张里 5 张得 0（与目测"失败"一致），cat 0.756（目测"部分成功"），
airplane 0（目测"部分成功，移动后变小"——检测不到匹配实例）。说明在真实失败上方向是对的，
但上面两个作弊通道没堵住，直接拿去 RL 会被整图平移/复制钻空子。

**决定**：按目标停在 GATE A，不启动数据转换和训练。reward 原型留在工作区未提交。
继续的前提是先换掉两个组件（全局变换估计 + 实例分割计数）并用同一套 7×7 候选重跑 GATE A。
原始分数：会话 scratchpad `edit_probe/gate_a/scores.json`，候选图同目录。

## 8. GATE A 第三轮（按 §7 的两个根因换组件）

- 背景：DINOv2 外观相似度 → **EfficientLoFTR**（`zju-community/efficientloftr`，Apache-2.0）几何匹配：
  背景匹配点里"位移 < 1% 对角线"的比例 × 相对原图自匹配的点数覆盖率；一致性改成 `identity × background`（乘积）。
- 计数：OWLv2 → **Grounding DINO base**（`IDEA-Research/grounding-dino-base`，Apache-2.0），仍按 DINOv2 外观匹配计数。
- 分数 `sqrt(geometry × identity × background)`。同一输入重复打分完全一致。

| 照片 | 正确移动 | 最高作弊 | 结果 |
|---|---|---|---|
| armchair | 0.933 | 0（其他全部） | PASS |
| bed | 0.915 | 0 | PASS |
| motorcycle | 0.430 | 0.390 复制 | PASS（余量很小） |
| airplane | 0.179 | 0.103 复制 | PASS（余量很小） |
| cat | 0.107 | 0 | PASS |
| zebra | 0（正确候选本身会盖住相邻斑马，计数 3→2） | 0 | FAIL（候选不成立） |
| woman | 0.404 | **0.619 复制** | FAIL |

**整图平移已经完全堵住**：7 张上 background 全部为 0（第二轮是 0.6–0.78）。
**复制仍然没堵住**：两个开放词表检测器（OWLv2、Grounding DINO）都把"并排的两个一模一样的人"
框成一个，所以复制在 airplane/motorcycle 上只比正确移动低一点，在 woman 上反超。而复制恰好是
base 最主要的失败方式（probe 里扶手椅、摩托、人），直接拿这个 reward 训练，GRPO 会被复制钻空子。

试过但撤回的：用 DINOv2 比较"编辑图在旧位置的内容"来判断物体是否离开（`vacated`）——它把位移小于
物体宽度的正确移动也一起打低（airplane 正确移动 0.18 → 0.02），同时我给 harness 加的"不可行排除"
误伤了 cat/armchair；两者都是在同 7 张图上调 reward/harness，按目标规则撤回。

**下一步需要的组件（不是调参）**：实例分割级别的计数——SAM 3 按文本分实例（`facebook/sam3`，
HF 上是 **gated，需要账号手动同意许可**），或 SAM 2.1 自动分割 + DINOv2 外观聚类数"像被移动物体的
分割块"。先单独验证它能把并排复制分成两个，再用同一套 7×7 候选重跑 GATE A。

## 9. GATE A 在规则选样的 COCO 集上（16 例）

§7–§8 的 7 张 probe 图里，飞机占画面宽度 98%、摩托 90%：根本没有地方"并排复制"，我造的"复制"候选
实际上是半重叠的近似正确移动，正确移动也挪不出自身宽度——测试本身不成立。于是按**固定规则**（不是手挑）
从 COCO val2017 GT 标注里选 16 例：该类别恰好一个非 crowd 实例、宽 ≤ 40% 画面、面积 3–25%、
一侧留空 ≥ 30% 宽、落点处没有其他标注物体（person 类排除：常被裁切/成群）。同样的规则就是训练数据的可行性过滤。
正确移动 = 用 GT 分割 mask 抠出物体、原位置 cv2 inpaint 补、贴到落点；作弊 = 原图不动、复制（不重叠）、
整图平移、裁剪放大（重新取景）、换一张照片。GT 物体框作为 `source_box` 元数据（真实训练数据也带框）；
检测下限 0.3 → 0.15（是否算作被移动物体由 DINOv2 身份匹配决定）。脚本 `edit_probe/gate_a2.py`。

结果 **12/16**，重复打分完全一致：

- 12 例通过：正确移动 0.36–0.97，所有作弊 0（tv、skateboard、motorcycle、chair、cat、stop sign、laptop、
  sheep、car、vase、bus、suitcase）。
- 16 例里**作弊的最高分 0.102**（dining table 的复制；这一例 GT "餐桌"几乎占满画面，背景点不存在，连"原图不动"
  的背景分都是 0——用例退化）。
- 另外 3 例失败都是**正确移动得 0**：handbag（包离开肩膀悬空）、banana（离开电话悬空）、surfboard（帆板贴进海里）——
  物体脱离上下文后 Grounding DINO 找不到它，按"丢失物体"处理。

**判断**：reward 的抗作弊部分成立（shift / crop-zoom / 换图 / 不动 全是 0，复制在非退化例上 ≤ 0.034）；
剩下的是**召回**问题——物体被移到不自然的位置时会漏检，正确的移动拿 0 分。这不会被钻空子，但会让一部分正确
样本没有正优势。GATE A 的字面标准（正确移动 > 所有作弊，逐例）没过，按目标规则停在这里，由人决定：
(a) 接受"抗作弊通过、召回有漏判"进入数据/训练，训练数据用同一可行性规则过滤；或
(b) 先加强召回（例如 SAM 类实例分割、或用源物体 DINOv2 外观在编辑图里做模板搜索），再重跑两套 GATE A。

### 9.1 试过并撤回：用物体自身的匹配点追踪位置（不靠重新检测）

想法：物体脱离上下文后检测器漏检，但纹理还能被 EfficientLoFTR 匹配，于是用"源物体框内的匹配点"的中位位移 +
尺度估计物体去向，找回比例作为"是否丢失"。结果更差：COCO 集 12/16 → 7/16，小物体/弱纹理物体（车、花瓶、笔记本、
停车牌）框内匹配点太少，正确移动也只得 0–0.1；7 张 probe 集仍 5/7。已撤回到 §9 的第四轮实现（复现 12/16）。
结论不变：剩下的漏判需要能在"物体被放到不自然位置"时仍找到它的实例级模型，而不是换一种匹配。

### 9.2 试过并撤回：检测器漏检时用 SAM（`facebook/sam-vit-base`，Apache-2.0，未 gated）分割编辑图、按 DINOv2 外观找回物体

COCO 12/16 → 13/16，但多通过的只是退化的 dining table 一例；handbag / banana / surfboard 仍为 0
（悬空物体被 SAM 切碎或与背景合并，外观相似度过不了 0.7）；probe 集 airplane 由通过变失败；每个样本多 1.4 s。
收益不抵复杂度，撤回到 §9 第四轮实现（复现 12/16）。

## 10. 数据 + GATE B（人工选 (a) 后继续）

- 训练池 `manifests/object_move/train.jsonl` **1117 条** = 600 COCO 2017 train（§9 同一可行性规则 + GT 框 `source_box`）
  + 517 SpatialEdit-500K `object_moving`（红框目标 → `target_box`）。构建脚本 `vrl/scripts/data/object_move.py`，
  图片在 `data/external/object_move`，dataset preset `vrl/config/presets/dataset/object_move.yaml`。
- Held-out：`heldout_coco.jsonl` 48 条（COCO val2017，同规则）、`heldout_bench.jsonl` 64 条（SpatialEdit-Bench move 子集，不进训练）。
- **GATE B**（base，512 px，SpatialEdit-Bench 64 条）：我们的 reward 均值 0.315（20 条为 0）；官方协议复现 overall 0.679 /
  IoU 0.721 / Score_oc 0.706。reward 与 benchmark overall 的 Spearman 起初 0.165——正确但变了尺寸的移动被判"丢失"；
  改成"每个编辑图检测分给最像的源实例"（f33995d3）后 **0.491**（与 IoU 0.376）。base 结果分布：移动 75%、不动 6%、复制 8%、丢失 11%。
- 官方协议的两处替换：IoU 框用 Grounding DINO（SAM 3 gated），judge 用 Qwen2.5-VL-7B（官方 72B AWQ 放不下）。

## 11. Dry run 与 run1

配方 `experiment/qwen_image_21/online_grpo_object_move`（与 OCR 实验同一套单卡设置：lora_wide、fp32 LoRA master、int8 Adam、
全量梯度检查点、不 compile、512 px、10 步无 CFG、noise 0.7、lr 3e-4、每次更新 48 样本 = 6 prompt × 8）。

启动命令（等 GPU 空闲后执行）：

```bash
python -m vrl.scripts.supervise --config experiment/qwen_image_21/online_grpo_object_move \
  --max-attempts 6 --health-metrics --health-max-grad-norm 0.5 \
  model.lora.parameter_dtype=float32 precision.float32_precision=ieee \
  trainer.output_dir=outputs/qwen_image_21_object_move_run1 trainer.total_epochs=60 trainer.save_freq=20 \
  trainer.debug.first_step=true
```

| 关卡 | 结果 |
|---|---|
| replay parity | max abs diff 2.4e-7（上限 1e-2）PASS |
| 梯度有限且非零 | grad_norm 2.2e-3 / 4.8e-4 / 5.9e-4 PASS |
| 峰值显存 < 30 GB | 27.7 GB（run1 27.9 GB）PASS |
| 单次更新 < 10 min | dry run 约 4.3 min；run1 实测 **约 5.4 min**（01:28 启动，ckpt-20 03:20，ckpt-40 05:04）PASS |

run1 训练曲线（每 10 update 平均）：

| update | reward_mean | reward_std | clip | adv_zero |
|---|---|---|---|---|
| 1–10 | 0.064 | 0.128 | 0.086 | ~0.5 |
| 11–20 | 0.035 | 0.060 | 0.061 | ~0.5 |
| 21–30 | 0.028 | 0.051 | 0.071 | ~0.5 |
| 31–40 | 0.040 | 0.073 | 0.071 | ~0.5 |

clip 最大 0.14（< 0.2），grad_norm 8e-4–1.6e-3，logprob 最大差 3.3e-4，health 未触发，0 次重启。40 update 仍平 → 按规则停止
（保存了 ckpt-20、ckpt-40）；`training_run_result.json` 的 failed 是我手动 SIGTERM 后清理阶段的记录，不是训练错误。

## 12. Verdict：BROKE

全部 held-out 用原生调度器 ODE、40 步、seed 0、每个 prompt 1 个样本，各 arm 用相同初始噪声（latent 哈希一致）。
脚本：`vrl.scripts.eval.image_checkpoint_eval`（reward/object_move preset）+ 会话 scratchpad 的 `outcomes.py`（复制/不动/丢失分桶）
和 `bench_score.py`（SpatialEdit-Bench 协议）。输出在 `outputs/qwen_image_21_object_move_run1/verdict/`。

| 集合 · 指标 | base | ckpt-20 | ckpt-40 | Δ40（95% bootstrap CI） |
|---|---|---|---|---|
| COCO held-out 48 · reward（1024 px） | 0.107 | 0.028 | 0.001 | −0.106（−0.158, −0.061） |
| COCO · 移动 / 不动 / 复制 / 丢失 | 35 / 21 / 21 / 23 % | 31 / 29 / 8 / 31 % | 19 / 0 / 17 / **65** % | |
| COCO · reward（512 px，训练分辨率） | 0.104 | 0.029 | 0.002 | −0.102（−0.159, −0.056） |
| COCO 512 · 丢失 | 31 % | 50 % | **79** % | |
| SpatialEdit-Bench 64 · reward | 0.312 | — | 0.003 | −0.309（−0.371, −0.245） |
| SpatialEdit-Bench · benchmark overall | 0.661 | — | 0.166 | −0.496（−0.565, −0.423） |
| SpatialEdit-Bench · IoU / Score_oc | 0.724 / 0.702 | — | 0.134 / 0.603 | |
| SpatialEdit-Bench · 移动 / 不动 / 复制 / 丢失 | 72 / 6 / 16 / 6 % | — | 42 / 5 / 13 / **41** % | |
| 镜头类 14 条 · 清晰度 | | — | Δ +0.02（CI 跨 0） | |

目测：ckpt-40 不再"在原图上改"，而是**整张重画**——画面变糊、有重影、物体常被画没；镜头类编辑（zoom-out / 俯视）也退化
（斑马数量变了、摩托俯视图扭曲、卧室变形），清晰度指标没抓到。ckpt-20 大体还像 base，已经开始变差。
**不是分辨率迁移问题**：512 训练分辨率下同样坏。

**原因（推断，未验证）**：reward 是乘积形式、极稀疏——训练中约一半的 prompt 组 8 个样本全是 0（adv_zero 0.47–0.55），整体 reward 均值只有约 0.04。
GRPO 组内标准化后，少数非零样本拿大正优势，其余"保留了参考图但没移对"的样本（不动、复制、近似移动）全部拿负优势。
这相当于持续压低"忠于参考图"的输出的概率，模型逐渐不再看参考图 → 整图重画 → 物体丢失。同一套配方在 OCR（稠密 reward）上是学得动的，
区别主要在 reward 密度。

**下一轮若要继续，先改的是信号，不是超参**（都需要单独验证，不在本 sprint 做）：
1. reward 加一个不依赖"移对了"的稠密项（例如背景保持 / 身份保持本身），让"忠于参考图"不会被一律判负；
2. 或对全零组 / 负优势做裁剪，只从正样本学（先离线看优势分布）；
3. 训练 prompt 先按 base 的成功率过滤到"8 个里 2–6 个能成"的区间，避免一半的组没信号。

## 13. 交付物

- Commits（未 push）：9a1902e4 reward、57c0fbf7 target_box、9712503b 评测支持参考图编辑、984aa4c9 数据、f33995d3 实例分配、d19314ba 配方。
- 对比页（前 16 条 COCO held-out，按顺序，不挑）：<https://claude.ai/artifact/S4dL7D8VuD6o2D2PPUfxnk>
- 数据与输出：`outputs/qwen_image_21_object_move_run1/`（ckpt、metrics、verdict/）、`outputs/qwen_image_21_object_move_base/`（GATE B）。
